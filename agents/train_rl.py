"""Workbook: train a PPO/DQN autoscaler for the cloud environment.

You are given the environment, the evaluator, the metrics and the CLI. Four
functions in this file are yours to write; the rest already works.

    python agents/train_rl.py --self-check    # what is still missing? (~10 s)
    python agents/train_rl.py --dry-run       # watch the MDP act, no RL needed
    python agents/train_rl.py --algo ppo --timesteps 200000

THE FOUR STEPS
--------------
1. ``build_vector_env``  wrap the env in Monitor + DummyVecEnv + VecNormalize
2. ``build_model``       instantiate PPO or DQN on that vectorised env
3. ``make_callbacks``    evaluation, checkpoints, autoscaling metrics
4. ``save_artifacts``    persist the model *and* its normalisation statistics

Each function below carries a CONTRACT block: inputs, return value, and the
properties ``--self-check`` verifies. Implement to the contract, run
``--self-check``, and only then spend 20 minutes on a real run.

Ground rules
------------
* Do not edit ``env/cloud_env.py`` or ``utils/evaluate.py`` to make a run look
  good, and do not tune the reward weights to flatter a result. The evaluator
  always scores with the *default* weights - that gap between what you train
  on and what you are measured on is the subject of README "Reference results".
* ``--self-check`` is a smoke test, not a grader. A policy that passes it and
  still loses to a constant 9-pod deployment has not solved the task; the bar
  is Session 5 in ``docs/WALKTHROUGH.md``.
* Your numbers are only comparable to the reference rows if you keep
  ``--timesteps``, ``--seed`` and the reward weights at their defaults.

Reasoning, checkpoints and knowledge checks for each step:
``docs/WALKTHROUGH.md``. A worked implementation is in
``agents/solutions/train_rl_reference.py`` - read it only after you have tried.

THE COACHING QUESTIONS (answer before your first long run)
----------------------------------------------------------
Q1. Algorithm: PPO or DQN?
    PPO  - on-policy, stable, forgiving defaults, works with a dense shaped
           reward and VecNormalize. Recommended starting point.
    DQN  - off-policy + replay buffer, more sample-efficient per env step but
           far more sensitive to reward scale (our overload penalty is huge)
           and to the exploration schedule.

Q2. Hyperparameters to choose deliberately:
    learning_rate  (default 3e-4) - start here, halve it if the policy collapses
    gamma          (default 0.99) - 0.995 if you want the agent to look ~33 min
                                    ahead (200 steps) instead of ~17 min (100 steps)
    total timesteps(default 200k) - 200k-500k is a sensible band for this MDP
    n_steps        (default 2048) - rollout length per env for PPO
    batch_size     (default 256)  - must satisfy n_steps * n_envs >= batch_size

Q3. Reward handling: an overloaded step costs 0.05 * (3000 - 200)^2 = 392,000,
    which dwarfs the 1.5 * pods cost term. Options:
    (a) VecNormalize(norm_reward=True, clip_reward=10)  <- default here
    (b) env reward_scale < 1 (e.g. 1e-3)
    (c) clip latency inside the reward only
    (d) rebalance the weights themselves: ``--cost-weight`` / ``--sla-weight``.
        With the defaults (1.5 / 0.05) over-provisioning IS reward-optimal -
        one avoided violation is worth 392,000, while a whole day of one extra
        pod costs only 1.5 * 8,640 = 12,960. A cost-aware agent therefore needs
        a larger cost weight, not just more training steps.
    Compare runs in TensorBoard before picking a winner.

Q4. Live monitoring:
    tensorboard --logdir logs/tb --port 6006
    Watch rollout/ep_rew_mean (should trend up), train/approx_kl (keep < 0.05),
    train/explained_variance (> 0.5 means the critic is learning),
    autoscaling/mean_pods and autoscaling/sla_violation_rate.

Usage
-----
    python agents/train_rl.py --dry-run                     # env sanity check only
    python agents/train_rl.py --algo ppo --timesteps 200000
    python agents/train_rl.py --algo dqn --timesteps 300000 --learning-rate 1e-4
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python agents/train_rl.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cloud_env import (  # noqa: E402
    DEFAULT_CHURN_WEIGHT,
    DEFAULT_COST_WEIGHT,
    DEFAULT_SLA_WEIGHT,
    DEFAULT_TRACE_PATH,
    CloudAutoscalingEnv,
)

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_CONFIG = {
    "algo": "ppo",
    "total_timesteps": 200_000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 256,
    "n_envs": 4,
    "episode_steps": 1440,  # 4 hours of trace per training episode
    "seed": 42,
    "cost_weight": DEFAULT_COST_WEIGHT,
    "sla_weight": DEFAULT_SLA_WEIGHT,
    "churn_weight": DEFAULT_CHURN_WEIGHT,
}


class MetricsCallback(BaseCallback):
    """Record autoscaling-specific metrics to TensorBoard.

    ``VecMonitor``/``Monitor`` forwards our ``info`` dict, so the callback can
    aggregate pod count, p95 latency and the SLA violation rate over the last
    ``log_freq`` environment steps.
    """

    def __init__(self, log_freq: int = 1_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = max(1, int(log_freq))
        self._pods: list[float] = []
        self._latencies: list[float] = []
        self._sla: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or []:
            if "active_pods" in info:
                self._pods.append(float(info["active_pods"]))
                self._latencies.append(float(info["latency_ms"]))
                self._sla.append(float(bool(info["sla_violation"])))

        if self.n_calls % self.log_freq == 0 and self._pods:
            self.logger.record("autoscaling/mean_pods", float(np.mean(self._pods)))
            self.logger.record("autoscaling/p95_latency_ms", float(np.percentile(self._latencies, 95)))
            self.logger.record("autoscaling/sla_violation_rate", float(np.mean(self._sla)))
            self._pods.clear()
            self._latencies.clear()
            self._sla.clear()
        return True


class SyncVecNormalizeCallback(BaseCallback):
    """Keep the eval env's observation statistics in sync with the training env.

    ``VecNormalize`` computes its own running statistics per wrapper, so an eval
    env created from scratch would normalise observations differently from the
    policy's training env and every evaluation number would be misleading. The
    eval env is frozen (``training=False``) and receives a copy of the training
    statistics before each evaluation.
    """

    def __init__(self, train_env, eval_env, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = copy.deepcopy(self.train_env.obs_rms)
        self.eval_env.ret_rms = copy.deepcopy(self.train_env.ret_rms)
        return True


# --------------------------------------------------------------- STEP 1: envs
def build_vector_env(
    args: argparse.Namespace,
    episode_steps: int | None,
    random_start: bool,
    seed: int,
    n_envs: int | None = None,
    norm_reward: bool = True,
):
    """STEP 1 - CONTRACT

    Build the vectorised, normalised environment.

    Inputs
        args          parsed CLI (args.trace, args.gamma, the three weights)
        episode_steps horizon per env; ``None`` means "the whole trace"
        random_start  True for training, False for the frozen eval env
        seed          base seed; the env for ``rank`` must get ``seed + rank``
        n_envs        number of parallel envs (default: args.n_envs)
        norm_reward   True for training, False for the eval env

    Return
        a ``VecNormalize`` wrapping a ``DummyVecEnv`` whose sub-envs are
        ``Monitor(CloudAutoscalingEnv(...))``.

    Checked by --self-check
        * norm_obs=True, clip_obs=10.0, gamma=args.gamma
        * norm_reward is whatever the caller passed - never hard-coded
        * cost/sla/churn weights forwarded to the env
        * episode_steps / random_start forwarded unchanged, so the *same*
          function can build either the training env or the eval env
        * reset()/step() give finite, correctly shaped observations

    Why norm_obs is not optional: RPS spans ~6..480 while CPU and RAM live in
    [0, 1] and pod count in [1, 15]; unnormalised, one channel dominates the
    network input. Why norm_reward *is* optional: one overloaded step costs
    ~392,000, which swamps the 1.5 * pods term - normalising rewards is what
    makes PPO trainable at all, while the eval env must report raw rewards so
    runs stay comparable.
    """
    raise NotImplementedError(
        "Step 1 (build_vector_env) is not implemented yet - see its CONTRACT block")


# -------------------------------------------------------- STEP 2: the algorithm
def build_model(args: argparse.Namespace, vec_env):
    """STEP 2 - CONTRACT

    Instantiate the algorithm on the training env.

    Return
        ``PPO`` when ``args.algo == "ppo"``, ``DQN`` when it is ``"dqn"``.

    Required configuration (all checked by --self-check)
        * policy="MlpPolicy" with ``net_arch=[128, 128]``
        * env=vec_env, seed=args.seed, gamma=args.gamma
        * tensorboard_log under ``logs/tb`` - an unlogged run cannot be read
        * PPO: n_steps=args.n_steps, batch_size=args.batch_size
        * DQN: a replay buffer, ``learning_starts`` and an exploration schedule
          (off-policy training needs both, on-policy PPO does not)

    Note the ordering constraint: SB3 derives the policy's input size from the
    env it is given, so this must run *after* ``build_vector_env`` and be
    handed the ``VecNormalize`` (not the raw env).
    """
    raise NotImplementedError(
        "Step 2 (build_model) is not implemented yet - see its CONTRACT block")


# --------------------------------------------------- STEP 3: callbacks + logging
def make_callbacks(args: argparse.Namespace, train_env, eval_env, n_envs: int):
    """STEP 3 - CONTRACT

    Return a ``CallbackList`` with exactly these four entries, in this order:

        [0] SyncVecNormalizeCallback(train_env, eval_env)
        [1] EvalCallback(eval_env, ...)
        [2] CheckpointCallback(...)
        [3] MetricsCallback(log_freq=1_000)      # already written for you

    Checked by --self-check
        * EvalCallback is given *the* ``eval_env`` object passed in (identity,
          not a fresh one) and ``deterministic=True``
        * ``eval_freq == args.eval_freq // n_envs``: SB3 counts *per-env*
          steps, so without the division a "10,000 step" eval happens every
          40,000 env steps with n_envs=4
        * checkpoint path is ``models/checkpoints`` with
          ``name_prefix=args.algo``
        * the sync callback runs *before* EvalCallback. Reverse them and every
          evaluation is scored with stale ``obs_rms`` from the previous sync -
          the eval curve then disagrees with the final evaluate.py numbers.

    ``EvalCallback`` also needs ``best_model_save_path`` and ``log_path``;
    point them at ``models/best`` and ``logs/eval``.
    """
    raise NotImplementedError(
        "Step 3 (make_callbacks) is not implemented yet - see its CONTRACT block")


# --------------------------------------------------- STEP 4: persist the result
def save_artifacts(model, train_env, args: argparse.Namespace,
                   models_dir: Path | None = None) -> tuple[Path, Path]:
    """STEP 4 - CONTRACT

    Persist the trained artefacts into ``models_dir`` (default: ``models/``),
    creating the directory if needed, and return
    ``(model_path, vecnormalize_path)`` where the paths have no file suffix.

    Must write exactly two files:
        <algo>_cloud_autoscaling.zip
        <algo>_cloud_autoscaling_vecnormalize.pkl

    Checked by --self-check (it saves into a temp dir, then reloads the pair)
        * both files exist and the returned paths point at them
        * ``PPO.load`` / ``DQN.load`` and ``VecNormalize.load`` reopen them
        * the reloaded statistics match the observation shape

    The second file is not optional. The policy was trained on observations
    normalised by *this* VecNormalize instance; ``utils/evaluate.py`` finds the
    ``*_vecnormalize.pkl`` next to the model automatically, and without it the
    evaluation feeds raw observations to a policy that never saw one.
    """
    raise NotImplementedError(
        "Step 4 (save_artifacts) is not implemented yet - see its CONTRACT block")


# ------------------------------------------------------------------- CLI + main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=["ppo", "dqn"], default=DEFAULT_CONFIG["algo"],
                        help="RL algorithm (default: ppo).")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_CONFIG["total_timesteps"],
                        help="Total environment steps to train for.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"],
                        help="Discount factor; 0.995 gives the agent a ~10 min horizon.")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_CONFIG["n_steps"],
                        help="PPO rollout length per environment.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG["n_envs"],
                        help="Parallel environments (DummyVecEnv).")
    parser.add_argument("--episode-steps", type=int, default=DEFAULT_CONFIG["episode_steps"],
                        help="Training episode horizon in 10 s steps (default 1440 = 4 h).")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH,
                        help="Workload CSV (default: data/workload_trace.csv).")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N environment steps.")
    parser.add_argument("--checkpoint-freq", type=int, default=20_000,
                        help="Save a checkpoint every N environment steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=1,
                        help="Episodes per evaluation (1 full-day episode is already 8,640 steps).")
    parser.add_argument("--no-random-start", action="store_true",
                        help="Always start training episodes at hour 0 (less coverage).")
    parser.add_argument("--cost-weight", type=float, default=DEFAULT_CONFIG["cost_weight"],
                        help="Reward weight on active pods per step (default: 1.5).")
    parser.add_argument("--sla-weight", type=float, default=DEFAULT_CONFIG["sla_weight"],
                        help="Reward weight on (latency - SLA)^2 in ms (default: 0.05).")
    parser.add_argument("--churn-weight", type=float, default=DEFAULT_CONFIG["churn_weight"],
                        help="Reward weight on |delta pods| per step (default: 0.5).")
    parser.add_argument("--self-check", action="store_true",
                        help="Run the four step contracts against this file (~10 s, no training).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the environment, print a sample rollout, then exit.")
    args = parser.parse_args(argv)

    if args.n_envs < 1:
        parser.error("--n-envs must be >= 1")
    if args.episode_steps < 1:
        parser.error("--episode-steps must be >= 1")
    if min(args.cost_weight, args.sla_weight, args.churn_weight) < 0:
        parser.error("reward weights must be >= 0")
    if args.algo == "ppo" and args.n_steps * args.n_envs < args.batch_size:
        parser.error(
            f"--batch-size ({args.batch_size}) must be <= n_steps * n_envs "
            f"({args.n_steps} * {args.n_envs} = {args.n_steps * args.n_envs})"
        )
    return args


def _print_banner(args: argparse.Namespace) -> None:
    print("=" * 70)
    print(" Cloud autoscaling RL training")
    print("=" * 70)
    print(f"  algorithm        : {args.algo.upper()}")
    print(f"  timesteps        : {args.timesteps:,}")
    print(f"  learning rate    : {args.learning_rate}")
    print(f"  gamma            : {args.gamma}")
    print(f"  n_envs           : {args.n_envs}")
    print(f"  episode horizon  : {args.episode_steps:,} steps "
          f"({args.episode_steps * 10 / 3600:.1f} h)")
    print(f"  random start     : {not args.no_random_start}")
    print(f"  reward weights   : cost={args.cost_weight} sla={args.sla_weight} "
          f"churn={args.churn_weight}")
    print(f"  trace            : {args.trace}")
    print("=" * 70)
    print(" Steps: (1) build env  (2) build model  (3) callbacks  (4) learn + save")
    print(" Not implemented yet?  python agents/train_rl.py --self-check")
    print(" Reasoning per step:   docs/WALKTHROUGH.md")
    print("=" * 70)


def _dry_run(args: argparse.Namespace) -> int:
    """Validate the MDP without instantiating an SB3 model."""
    from stable_baselines3.common.env_checker import check_env

    env = CloudAutoscalingEnv(
        trace_path=args.trace,
        episode_steps=args.episode_steps,
        random_start=False,
        seed=args.seed,
        cost_weight=args.cost_weight,
        sla_weight=args.sla_weight,
        churn_weight=args.churn_weight,
    )
    print(f"[dry-run] trace          : {env.trace_path} ({env.trace_length:,} bins)")
    check_env(env, warn=True)
    print("[dry-run] check_env      : PASSED")
    print(f"[dry-run] observation    : {env.observation_space}")
    print(f"[dry-run] action space   : {env.action_space}")

    observation, info = env.reset(seed=args.seed)
    print(f"[dry-run] initial obs    : {np.round(observation, 3).tolist()}")
    print(f"[dry-run] initial pods   : {info}")

    print("[dry-run] hand-checked steps (action, pods, cpu, latency, reward parts):")
    for action in (2, 3, 3, 4, 2, 1, 2, 0):
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"  a={action} pods={info['active_pods']:2d} cpu={info['cpu_utilization']:.2f} "
            f"lat={info['latency_ms']:7.1f}ms | cost={info['cost']:6.1f} "
            f"sla={info['sla_penalty']:10.1f} churn={info['churn_penalty']:4.1f} "
            f"-> r={reward:12.1f}"
        )
        if terminated or truncated:
            break

    observation, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)  # sample() is entropy-seeded otherwise
    total_reward = 0.0
    violations = 0
    steps = 0
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(int(env.action_space.sample()))
        assert env.observation_space.contains(observation), "observation left the space"
        total_reward += reward
        violations += int(info["sla_violation"])
        steps += 1
        done = terminated or truncated

    print(f"[dry-run] random rollout : {steps:,} steps | total reward {total_reward:,.0f} "
          f"| SLA violation rate {100.0 * violations / max(steps, 1):.1f}%")
    env.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        from agents.self_check import run_self_check

        return run_self_check(sys.modules[__name__], args)
    _print_banner(args)
    if args.dry_run:
        return _dry_run(args)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # STEP 1 - environments. Training: random starts + reward normalisation.
    # Evaluation: full-day deterministic episode, no reward normalisation.
    train_env = build_vector_env(
        args,
        episode_steps=args.episode_steps,
        random_start=not args.no_random_start,
        seed=args.seed,
    )
    eval_env = build_vector_env(
        args,
        episode_steps=None,
        random_start=False,
        seed=args.seed + 1_000,
        n_envs=1,
        norm_reward=False,
    )
    eval_env.training = False  # freeze statistics; they are synced from train_env
    print(f"[train] observation vector: {train_env.observation_space}")

    # STEP 2 - algorithm.
    model = build_model(args, train_env)
    print(f"[train] policy network:\n{model.policy}")

    # STEP 3 - callbacks (evaluation, checkpoints, autoscaling metrics).
    callbacks = make_callbacks(args, train_env, eval_env, n_envs=args.n_envs)

    # STEP 4 - learn, then persist model + normalisation statistics together.
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=f"{args.algo}_{args.seed}",
        progress_bar=False,
    )

    model_path, vecnormalize_path = save_artifacts(model, train_env, args)
    elapsed = time.time() - started

    print("=" * 70)
    print(f"[train] finished {args.timesteps:,} steps in {elapsed / 60.0:.1f} min")
    print(f"[train] model        -> {model_path}.zip")
    print(f"[train] vecnormalize -> {vecnormalize_path}")
    print(f"[train] best model   -> {MODELS_DIR / 'best' / 'best_model.zip'}")
    print("=" * 70)
    print("Next steps:")
    print(f"  tensorboard --logdir {LOGS_DIR / 'tb'} --port 6006")
    print(
        "  python utils/evaluate.py "
        f"--model {model_path}.zip --algo {args.algo} --vecnormalize {vecnormalize_path}"
    )

    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotImplementedError as exc:
        print()
        print("=" * 70)
        print(f"[train] stopped: {exc}")
        print("[train] the contract for each step is in that function's docstring.")
        print("[train] see all four at once:  python agents/train_rl.py --self-check")
        print("[train] reasoning per step:    docs/WALKTHROUGH.md")
        print("=" * 70)
        raise SystemExit(2)
