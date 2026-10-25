"""Train the design-stack policy and hold the SLA as a constraint.

This is the experiment rig for ``docs/DESIGN.md``: reward **v3**
(``LinearSLAReward``), observation **v2** (``HeadroomObs``, 11 dims) and,
optionally, a *price* on every violating step that a dual-ascent callback
raises until violations stop.

Why a separate script instead of ``agents/train_rl.py``
-------------------------------------------------------
``train_rl.py`` is the workbook: four ``NotImplementedError`` stubs to fill in
against a shipped self-check. This file is deliberately standalone so the
design experiment can run while that workbook is still empty. It shares the
environment, not the code.

The two routes it can run
-------------------------
``--lambda-lr 0``     reward-only. The agent is paid and charged by v3 alone.
``--lambda-lr > 0``   Lagrangian. v3 is charged an extra ``price`` per
                      violating step; the callback raises the price while
                      violations occur and freezes it once they stop.

Measured reference points on this trace (full 8,640-step day, reward v3):

===========================  ==========  =========  =========
policy                       violations  avg pods   return v3
===========================  ==========  =========  =========
constant 9 pods              0           9.00       -116,640
trailing max 480 s, u=0.77   0           4.58       -60,088
DP ceiling (foresight)       -           -          -38,658
===========================  ==========  =========  =========

Usage
-----
    python agents/train_sla.py --design-check
    python agents/train_sla.py --dry-run
    python agents/train_sla.py --timesteps 200000 --tag r1-price0 --lambda-lr 0
    python agents/train_sla.py --timesteps 200000 --tag r2-lagrangian
    python utils/score_policy.py --policy model --model models/sla/r2-lagrangian.zip

Note that ``utils/evaluate.py`` cannot score these models: it rebuilds the
*stock* 5-dim environment before loading the ``VecNormalize`` statistics, so an
11-dim policy would be fed the wrong observation. Use
``utils/score_policy.py``, which builds the matching stack first.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python agents/train_sla.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from env.cloud_env import (  # noqa: E402
    ACTION_DELTAS,
    DEFAULT_COST_WEIGHT,
    DEFAULT_MAX_PODS,
    DEFAULT_MIN_PODS,
    DEFAULT_TRACE_PATH,
    CloudAutoscalingEnv,
    load_trace,
)
from env.design_wrappers import OBS_DIM, HeadroomObs, LinearSLAReward  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models" / "sla"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_CONFIG = {
    "timesteps": 200_000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 256,
    "n_envs": 4,
    "episode_steps": 1440,  # 4 hours of trace per training episode
    "seed": 42,
    "cost_weight": DEFAULT_COST_WEIGHT,
    # v3 charges churn on the *requested* delta, at 4.0 per pod (not the stock 0.5).
    "churn_weight": 4.0,
    "lambda_init": 0.0,
    "lambda_lr": 50.0,
    "lambda_max": 500.0,
}


# --------------------------------------------------------------------- the stack
def make_env(args: argparse.Namespace, episode_steps, random_start: bool, seed: int,
             price: float = 0.0):
    """Factory for one sub-environment: base env -> reward v3 -> obs v2 -> Monitor.

    The order matters for readability only: ``HeadroomObs`` reads pod counts from
    ``env.unwrapped`` and ``LinearSLAReward`` reads ``info``, so both orders give
    the same numbers (``--design-check`` asserts exactly that).
    """

    def _init():
        env = CloudAutoscalingEnv(
            trace_path=args.trace,
            episode_steps=episode_steps,
            random_start=random_start,
            initial_pods="auto",
            seed=seed,
        )
        env = LinearSLAReward(
            env,
            cost_weight=args.cost_weight,
            churn_weight=args.churn_weight,
            price=price,
        )
        env = HeadroomObs(env)
        return Monitor(env)

    return _init


def build_design_env(
    args: argparse.Namespace,
    episode_steps=None,
    random_start: bool = True,
    seed=None,
    n_envs: int | None = None,
    norm_reward: bool = True,
    price: float | None = None,
) -> VecNormalize:
    """Vectorised, normalised design-stack environment.

    ``clip_reward=np.inf`` is load-bearing. SB3's default is 10, and v3's own
    cap (5x ``violation_cost`` = 1,000) is a pricing decision made in the reward.
    Re-clipping in *normalised* units silently undoes it, and the policy then
    cannot tell a mild breach from an outage - the exact ambiguity v3 exists to
    remove.
    """
    n = args.n_envs if n_envs is None else int(n_envs)
    base_seed = args.seed if seed is None else int(seed)
    start_price = args.lambda_init if price is None else float(price)
    venv = DummyVecEnv(
        [make_env(args, episode_steps, random_start, base_seed + rank, start_price)
         for rank in range(n)]
    )
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=np.inf,
        gamma=args.gamma,
    )


def build_model(args: argparse.Namespace, vec_env: VecNormalize) -> PPO:
    """PPO with the workbook's defaults, on the 11-dim design observation.

    PPO rather than DQN: the reward is dense and heavily scaled, and PPO's
    advantage normalisation is what makes that survivable (``docs/DESIGN.md``
    section 9). ``net_arch=[128, 128]`` matches ``train_rl.py`` so runs stay
    comparable.
    """
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(LOGS_DIR / "tb"),
        verbose=1,
        device="cpu",
    )


# ------------------------------------------------------------------- callbacks
class SyncEvalCallback(BaseCallback):
    """Copy the training statistics into the eval environment before it is used.

    Two things have to cross over, in this order:

    * ``obs_rms`` - the eval env gets raw observations; the policy was trained on
      normalised ones. Without the copy every eval number is noise.
    * the current ``price`` - otherwise the eval env still charges whatever the
      price was when it was built, and the eval curve measures a different
      reward than the one being optimised.

    This must run *before* ``EvalCallback`` in the ``CallbackList``.
    """

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = copy.deepcopy(self.train_env.obs_rms)
            self.eval_env.ret_rms = copy.deepcopy(self.train_env.ret_rms)
        try:
            prices = self.train_env.env_method("get_price")
            if prices:
                self.eval_env.env_method("set_price", float(prices[0]))
        except AttributeError:  # a plain env without LinearSLAReward
            pass
        return True


class LagrangianCallback(BaseCallback):
    """Dual ascent on the SLA: price each violating step until none occur.

    The constrained problem is ``max E[reward_v3]  s.t.  violations == 0``. Its
    Lagrangian charges ``price`` per violating step, and the subgradient of the
    constraint is the violation *rate*, so the multiplier moves as

        price <- clip(price + lr * (rate - target), 0, max_price)

    Two properties of *this* MDP matter for reading the logs:

    * The environment is deterministic, so the multiplier does not settle at a
      shadow price in the usual sense - it ratchets up until breaching stops and
      then stays. That is intended: it is a barrier, not a market price.
    * Once the policy is clean, ``price`` stops moving and the rest of training
      is ordinary reward maximisation, so freezing is automatic
      (``freeze_after`` clean windows) and is reported as ``sla/frozen``.

    ``price`` is in reward units per violating step, comparable to the 1.5 per
    pod per step the capacity term charges.
    """

    def __init__(
        self,
        price: float = 0.0,
        lr: float = 50.0,
        max_price: float = 500.0,
        target_rate: float = 0.0,
        update_freq: int = 10_000,
        freeze_after: int = 3,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.price = float(price)
        self.lr = float(lr)
        self.max_price = float(max_price)
        self.target_rate = float(target_rate)
        self.update_freq = max(1, int(update_freq))
        self.freeze_after = max(1, int(freeze_after))
        self.frozen = self.lr <= 0.0  # lr == 0 means "reward-only": never adapt
        self.last_rate = 0.0
        self._violations = 0.0
        self._steps = 0
        self._pod_sum = 0.0
        self._clean_streak = 0

    def _on_training_start(self) -> None:
        self.model.env.env_method("set_price", self.price)
        self.logger.record("sla/price", self.price)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            self._violations += float(info.get("violation", info.get("sla_violation", 0)))
            self._pod_sum += float(info.get("active_pods", 0.0))
        self._steps += len(infos)
        if self._steps >= self.update_freq:
            self._update()
        return True

    def _update(self) -> None:
        rate = self._violations / max(1, self._steps)
        mean_pods = self._pod_sum / max(1, self._steps)
        self.last_rate = rate
        self._violations = 0.0
        self._steps = 0
        self._pod_sum = 0.0

        if rate <= self.target_rate:
            self._clean_streak += 1
        else:
            self._clean_streak = 0

        if not self.frozen:
            if self._clean_streak >= self.freeze_after:
                self.frozen = True
                if self.verbose:
                    print(f"[lagrangian] constraint met for {self.freeze_after} windows; "
                          f"freezing price at {self.price:.1f}")
            else:
                self.price = float(
                    np.clip(self.price + self.lr * (rate - self.target_rate),
                            0.0, self.max_price)
                )
                self.model.env.env_method("set_price", self.price)

        self.logger.record("sla/violation_rate", rate)
        self.logger.record("sla/mean_pods", mean_pods)
        self.logger.record("sla/price", self.price)
        self.logger.record("sla/frozen", float(self.frozen))
        if self.verbose and (not self.frozen or rate > 0.0):
            print(f"[lagrangian] steps={self.num_timesteps} rate={rate:.4f} "
                  f"pods={mean_pods:.2f} price={self.price:.1f}")


def make_callbacks(args: argparse.Namespace, train_env: VecNormalize,
                   eval_env: VecNormalize) -> CallbackList:
    """Evaluation, checkpoints and the constraint price, in order of coupling.

    ``SyncEvalCallback`` runs first so ``EvalCallback`` scores the policy with
    current statistics; reversing them scores every evaluation with stale ones.
    """
    callbacks = [
        SyncEvalCallback(train_env, eval_env),
        LagrangianCallback(
            price=args.lambda_init,
            lr=args.lambda_lr,
            max_price=args.lambda_max,
            update_freq=args.lambda_update_freq,
            freeze_after=args.lambda_freeze_after,
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODELS_DIR / "best"),
            log_path=str(LOGS_DIR / "eval"),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(MODELS_DIR / "checkpoints"),
            name_prefix=args.tag,
        ),
    ]
    return CallbackList(callbacks)


# --------------------------------------------------------------- design check
def _scripted_actions(kind: str, steps: int) -> list[int]:
    """Three action scripts: a hold, an envelope cycle, and a deliberate collapse.

    ``starve`` matters for the box check: it drives the fleet to one pod, so
    latency hits the 3,000 ms overload value and the observation must still be
    inside its ``Box``. A script that only cycles in the safe region proves
    almost nothing about the declared bounds.
    """
    if kind == "hold":
        return [2] * steps  # ACTION_DELTAS[2] == 0
    if kind == "starve":
        return [1] * steps  # -1 pod per step, floor at min_pods
    pattern = (4, 4, 4, 0, 0, 3, 3, 2, 1, 4)  # +2,+2,+2,-2,-2,+1,+1,0,-1,+2
    return [pattern[i % len(pattern)] for i in range(steps)]


def _build_check_env(args: argparse.Namespace, order: str, steps: int,
                     price: float = 0.0, start_pods: int = 9):
    """Design stack for the parity check, in one of the two wrapper orders."""
    kwargs = dict(cost_weight=args.cost_weight, churn_weight=args.churn_weight,
                  price=price)
    env = CloudAutoscalingEnv(
        trace_path=args.trace,
        episode_steps=steps,
        random_start=False,
        initial_pods=start_pods,
    )
    if order == "reward_inner":
        return HeadroomObs(LinearSLAReward(env, **kwargs))
    return LinearSLAReward(HeadroomObs(env), **kwargs)


def _expected_rewards(trace: np.ndarray, actions, start_pods: int, cfg) -> np.ndarray:
    """Per-step v3 reward as ``utils/design_probe`` computes it (same physics)."""
    from utils.design_probe import latency

    pods = int(start_pods)
    out = []
    for rps, action in zip(trace, actions):
        delta = ACTION_DELTAS[int(action)]
        applied = int(np.clip(pods + delta, DEFAULT_MIN_PODS, DEFAULT_MAX_PODS))
        out.append(cfg(applied, latency(float(rps), applied), applied - pods, delta))
        pods = applied
    return np.asarray(out, dtype=np.float64)


def _replay(env, actions):
    """Step ``env`` through ``actions``; return rewards, violations, observations."""
    rewards, violations, observations = [], [], []
    obs, _ = env.reset(seed=0)
    observations.append(np.asarray(obs, dtype=np.float32))
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        violations.append(int(info["violation"]))
        observations.append(np.asarray(obs, dtype=np.float32))
        if terminated or truncated:
            break
    return (np.asarray(rewards), np.asarray(violations),
            np.asarray(observations, dtype=np.float32))


def design_check(args: argparse.Namespace) -> int:
    """Five assertions that must hold before any learning number is quotable.

    1. wrapper reward == probe reward, wrapper order ``reward_inner``
    2. wrapper reward == probe reward, wrapper order ``obs_inner``
    3. every observation inside the declared ``Box``
    4. ``price`` arithmetic: ``reward(p) == reward(0) - p * violations``
    5. the vectorised env: ``env_method`` price plumbing, ``clip_reward=inf``
    """
    from utils.design_probe import reward_family

    print("== design check: reward parity, observation box, price, vec plumbing ==")
    trace = load_trace(args.trace)
    steps = min(int(args.check_steps), int(trace.size))
    trace = trace[:steps]
    cfg = reward_family("v3")
    start_pods = 9
    failures: list[str] = []
    checks = 0

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} {detail}")
        if not ok:
            failures.append(name)

    # 1 + 2 - reward parity in both wrapper orders
    for order in ("reward_inner", "obs_inner"):
        worst = 0.0
        for script in ("hold", "cycle", "starve"):
            actions = _scripted_actions(script, steps)
            env = _build_check_env(args, order, steps, start_pods=start_pods)
            rewards, _, _ = _replay(env, actions)
            expected = _expected_rewards(trace, actions[: rewards.size], start_pods, cfg)
            if rewards.size != expected.size:
                worst = float("inf")
            else:
                worst = max(worst, float(np.max(np.abs(rewards - expected))))
        report(f"reward parity [{order}]", worst <= 1e-9,
               f"max |wrapper - probe v3| = {worst:.3g} over 3 scripts x {steps} steps")

    # 3 - observation box
    observations = None
    for order in ("reward_inner", "obs_inner"):
        outside = 0
        total = 0
        for script in ("cycle", "starve"):
            env = _build_check_env(args, order, steps, start_pods=start_pods)
            _, _, obs = _replay(env, _scripted_actions(script, steps))
            total += len(obs)
            outside += sum(0 if env.observation_space.contains(o) else 1 for o in obs)
            if order == "reward_inner" and script == "starve":
                observations = obs
        report(f"observation box [{order}]", outside == 0,
               f"shape {tuple(obs.shape[1:])}, {outside} of {total} samples outside Box")

    if observations is not None:
        lows = observations.min(axis=0)
        highs = observations.max(axis=0)
        labels = ("cpu", "ram", "demand/max", "lat/sla", "pods/15", "headroom",
                  "slope", "peak60", "peak480", "impulse", "frac_day")
        print("  feature ranges over the starve script (fleet driven to 1 pod):")
        for index, label in enumerate(labels):
            print(f"    {index:2d} {label:10s} [{lows[index]:+.3f}, {highs[index]:+.3f}]")

    # 4 - price arithmetic and price plumbing on a single env. The script must
    # actually breach (``starve`` does); with a clean script the identity holds
    # trivially and the check proves nothing.
    actions = _scripted_actions("starve", steps)
    base = _build_check_env(args, "reward_inner", steps, price=0.0, start_pods=start_pods)
    r0, v0, _ = _replay(base, actions)
    priced = _build_check_env(args, "reward_inner", steps, price=37.0, start_pods=start_pods)
    r37, v37, _ = _replay(priced, actions)
    worst = float(np.max(np.abs((r0 - 37.0 * v37) - r37))) if r37.size else float("inf")
    # gymnasium's get_wrapper_attr walks down to whichever wrapper defines it,
    # which is what a caller has to do when the price lives under HeadroomObs.
    read_price = float(priced.get_wrapper_attr("get_price")())
    report("price arithmetic", bool(np.array_equal(v0, v37)) and worst <= 1e-9
           and read_price == 37.0,
           f"{int(v37.sum())} violations, max |reward(0) - 37*v - reward(37)| = {worst:.3g}")

    # 5 - vectorised plumbing: env_method reaches every sub-env
    venv = build_design_env(args, episode_steps=64, random_start=True, seed=args.seed,
                            n_envs=2, norm_reward=True, price=0.0)
    before = venv.env_method("get_price")
    venv.env_method("set_price", 12.5)
    after = venv.env_method("get_price")
    obs = venv.reset()
    _, rewards, _, infos = venv.step(np.array([2, 2], dtype=np.int64))
    ok = (before == [0.0, 0.0] and after == [12.5, 12.5]
          and venv.clip_reward == np.inf and venv.norm_obs
          and tuple(obs.shape) == (2, OBS_DIM)
          and bool(np.all(np.isfinite(rewards)))
          and all(info["constraint_price"] == 12.5 for info in infos))
    report("vectorised plumbing", ok,
           f"price {before} -> {after}, clip_reward={venv.clip_reward}, "
           f"obs={tuple(obs.shape)}, reward={np.round(rewards, 2).tolist()}")
    venv.close()

    print()
    if failures:
        print(f"design-check FAILED: {len(failures)} of {checks} -> {failures}")
        return 1
    print(f"design-check {checks}/{checks} PASS")
    return 0


# ----------------------------------------------------------------------- driver
def _dry_run(args: argparse.Namespace, train_env: VecNormalize) -> int:
    """Build the whole stack, take 32 random steps, and print what came back."""
    obs = train_env.reset()
    rng = np.random.default_rng(args.seed)
    actions = rng.integers(0, 5, size=(32, args.n_envs))
    rewards = []
    for action in actions:
        obs, reward, _, infos = train_env.step(action)
        rewards.append(float(np.mean(reward)))

    model = build_model(args, train_env)
    probe, _ = model.predict(obs, deterministic=True)

    print("== dry run ==")
    print(f"  obs shape     {tuple(obs.shape)} (expected (*, {OBS_DIM}))")
    print(f"  obs finite    {bool(np.all(np.isfinite(obs)))}  "
          f"range [{float(obs.min()):+.2f}, {float(obs.max()):+.2f}]")
    print(f"  reward        min {min(rewards):+.2f}  max {max(rewards):+.2f}  "
          f"mean {float(np.mean(rewards)):+.2f} (normalised)")
    print(f"  prices        {train_env.env_method('get_price')}")
    print(f"  PPO action    {np.asarray(probe).reshape(-1).tolist()}")
    train_env.close()
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--design-check", action="store_true",
                        help="Run the five pre-flight assertions and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the stack, take 32 random steps, exit.")
    parser.add_argument("--trace", type=Path, default=None,
                        help="Trace CSV (default: "
                             f"{DEFAULT_TRACE_PATH.relative_to(PROJECT_ROOT)}).")

    group = parser.add_argument_group("training")
    group.add_argument("--timesteps", type=int, default=DEFAULT_CONFIG["timesteps"])
    group.add_argument("--episode-steps", type=int, default=DEFAULT_CONFIG["episode_steps"],
                       help="Steps per training episode (default 1440 = 4 h of trace).")
    group.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG["n_envs"])
    group.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    group.add_argument("--learning-rate", type=float,
                       default=DEFAULT_CONFIG["learning_rate"])
    group.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"])
    group.add_argument("--n-steps", type=int, default=DEFAULT_CONFIG["n_steps"])
    group.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    group.add_argument("--cost-weight", type=float, default=DEFAULT_CONFIG["cost_weight"])
    group.add_argument("--churn-weight", type=float, default=DEFAULT_CONFIG["churn_weight"])

    constraint = parser.add_argument_group("SLA constraint (Lagrangian)")
    constraint.add_argument("--lambda-init", type=float, default=DEFAULT_CONFIG["lambda_init"],
                            help="Starting price, in reward units per violating step.")
    constraint.add_argument("--lambda-lr", type=float, default=DEFAULT_CONFIG["lambda_lr"],
                            help="Dual step size. 0 = reward-only run (no constraint).")
    constraint.add_argument("--lambda-max", type=float, default=DEFAULT_CONFIG["lambda_max"],
                            help="Price ceiling; also the fixed price if --lambda-lr 0.")
    constraint.add_argument("--lambda-update-freq", type=int, default=10_000,
                            help="Env steps per multiplier update.")
    constraint.add_argument("--lambda-freeze-after", type=int, default=3,
                            help="Clean windows before the price is frozen.")

    book = parser.add_argument_group("bookkeeping")
    book.add_argument("--eval-freq", type=int, default=20_000,
                      help="Env steps between evaluations.")
    book.add_argument("--checkpoint-freq", type=int, default=50_000)
    book.add_argument("--n-eval-episodes", type=int, default=2)
    book.add_argument("--check-steps", type=int, default=8_640,
                      help="Trace steps used by --design-check (default: a full day).")
    book.add_argument("--tag", type=str, default="sla-run",
                      help="Run name for checkpoints, tensorboard and the saved model.")
    book.add_argument("--out", type=Path, default=None,
                      help="Output directory (default: models/sla).")

    args = parser.parse_args(argv)
    if args.n_steps * args.n_envs < args.batch_size:
        parser.error(f"n_steps * n_envs ({args.n_steps * args.n_envs}) must be >= "
                     f"batch_size ({args.batch_size})")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.design_check:
        return design_check(args)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "tb").mkdir(parents=True, exist_ok=True)

    train_env = build_design_env(args, args.episode_steps, True, args.seed, norm_reward=True)
    eval_env = build_design_env(args, args.episode_steps, False, args.seed + 1_000,
                                n_envs=1, norm_reward=False)
    if args.dry_run:
        return _dry_run(args, train_env)

    model = build_model(args, train_env)
    started = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=make_callbacks(args, train_env, eval_env),
        tb_log_name=args.tag,
        progress_bar=False,
    )
    elapsed = time.time() - started

    out_dir = Path(args.out).expanduser() if args.out else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.tag}.zip"
    stats_path = out_dir / f"{args.tag}_vecnormalize.pkl"
    model.save(str(model_path.with_suffix("")))
    train_env.save(str(stats_path))

    regime = "reward-only" if args.lambda_lr <= 0 else f"lagrangian (price <= {args.lambda_max:g})"
    print()
    print(f"== trained ==  tag={args.tag}  regime={regime}  "
          f"steps={args.timesteps:,}  wall={elapsed:,.1f}s")
    print(f"  model       {model_path}")
    print(f"  normaliser  {stats_path}")
    print(f"  score it    python utils/score_policy.py --policy model --model {model_path}")
    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


