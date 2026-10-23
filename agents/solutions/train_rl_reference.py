"""Reference solution for the cloud autoscaling RL training workbook.

This is the finished version of ``agents/train_rl.py``. Use it when you are
stuck on a step, or to compare behaviour after your own run:

    python agents/solutions/train_rl_reference.py --self-check
    python agents/solutions/train_rl_reference.py --dry-run
    python agents/solutions/train_rl_reference.py --algo ppo --timesteps 200000

The numbers in README.md under "Reference results" were produced with this
file. Read it *after* attempting the step yourself - the point of the workbook
is the reasoning in docs/WALKTHROUGH.md, not this listing.

THE FOUR CORE STEPS (mirrored 1:1 in the workbook)
--------------------------------------------------
1. Build & validate the environment ....... ``_dry_run`` / ``build_vector_env``
2. Instantiate the algorithm .............. ``build_model``  (PPO or DQN)
3. Attach callbacks + TensorBoard ......... ``make_callbacks``
4. Learn, save, and hand off .............. ``save_artifacts`` / ``main``

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
    python agents/solutions/train_rl_reference.py --self-check     # 4 x PASS
    python agents/solutions/train_rl_reference.py --dry-run        # env sanity check
    python agents/solutions/train_rl_reference.py --algo ppo --timesteps 200000
    python agents/solutions/train_rl_reference.py --algo dqn --timesteps 300000 \
        --learning-rate 1e-4
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # allow `python agents/solutions/train_rl_reference.py`
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
    """Wrap the environment in Monitor + DummyVecEnv + VecNormalize.

    Observation normalisation is essential here because the raw RPS channel
    ranges from ~20 to 480 while RAM/CPU live in [0, 1]. Reward normalisation
    (``norm_reward``) tames the ~392k overload penalty; evaluation envs must
    disable it so rewards stay comparable to the baseline.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    n_envs = n_envs if n_envs is not None else args.n_envs

    def make_thunk(rank: int):
        def _init():
            env = CloudAutoscalingEnv(
                trace_path=args.trace,
                episode_steps=episode_steps,
                random_start=random_start,
                seed=seed + rank,
                cost_weight=args.cost_weight,
                sla_weight=args.sla_weight,
                churn_weight=args.churn_weight,
            )
            return Monitor(env, filename=None)

        return _init

    vec_env = DummyVecEnv([make_thunk(rank) for rank in range(n_envs)])
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma,
    )


# -------------------------------------------------------- STEP 2: the algorithm
def build_model(args: argparse.Namespace, vec_env):
    """Instantiate PPO or DQN with a small MLP policy."""
    from stable_baselines3 import DQN, PPO

    common = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        tensorboard_log=str(LOGS_DIR / "tb"),
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    if args.algo == "ppo":
        return PPO(
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common,
        )

    return DQN(
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=args.batch_size,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
        **common,
    )


# --------------------------------------------------- STEP 3: callbacks + logging
def make_callbacks(args: argparse.Namespace, train_env, eval_env, n_envs: int):
    """Evaluation, checkpointing and custom autoscaling metrics."""
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    # Must run before EvalCallback so each evaluation uses current statistics.
    sync_callback = SyncVecNormalizeCallback(train_env, eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(LOGS_DIR / "eval"),
        eval_freq=max(1, args.eval_freq // max(1, n_envs)),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(1, n_envs)),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=args.algo,
        verbose=0,
    )
    return CallbackList(
        [sync_callback, eval_callback, checkpoint_callback, MetricsCallback(log_freq=1_000)]
    )


# --------------------------------------------------- STEP 4: persist the result
def save_artifacts(model, train_env, args: argparse.Namespace,
                   models_dir: Path | None = None) -> tuple[Path, Path]:
    """Persist the model and its observation statistics as a matched pair.

    A PPO/DQN checkpoint alone is not reproducible: the policy was trained on
    observations normalised by this VecNormalize instance, so evaluation must
    reuse the same ``obs_rms``. ``utils/evaluate.py`` therefore auto-detects
    ``<model>_vecnormalize.pkl`` next to the ``.zip``.
    """
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{args.algo}_cloud_autoscaling"
    vecnormalize_path = models_dir / f"{model_path.name}_vecnormalize.pkl"
    model.save(model_path)
    train_env.save(str(vecnormalize_path))
    return model_path, vecnormalize_path


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
    print(" Self-check:           python agents/solutions/train_rl_reference.py --self-check")
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
    raise SystemExit(main())
