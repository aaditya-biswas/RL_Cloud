"""Contract checks behind ``--self-check`` for the four training steps.

Every check returns a list of human-readable failures; an empty list means the
step satisfies its contract. Nothing here writes into ``models/`` or ``logs/``:
the save/reload check runs in a temporary directory.

The point of this module is feedback: run it after each step, read the failures,
fix your implementation, run it again. It never tells you *how* to implement a
step - that is in the ``CONTRACT`` block of each function in ``train_rl.py``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np

STEP_TITLES = {
    1: "Step 1 - environments      (build_vector_env)",
    2: "Step 2 - algorithm         (build_model)",
    3: "Step 3 - callbacks         (make_callbacks)",
    4: "Step 4 - save artefacts    (save_artifacts)",
}


def _call_step(step: int, function, *args, **kwargs) -> tuple[object | None, list[str]]:
    """Call one step implementation, turning failures into check messages."""
    try:
        return function(*args, **kwargs), []
    except NotImplementedError:
        return None, [f"not implemented yet - fill in {function.__name__}()"]
    except Exception as exc:  # noqa: BLE001 - any exception is a contract failure
        return None, [f"{function.__name__}() raised {type(exc).__name__}: {exc}"]
def _check_step1(module: ModuleType, args: argparse.Namespace):
    """Return ``(train_env, eval_env, failures)``."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from env.cloud_env import CloudAutoscalingEnv

    train_env, failures = _call_step(
        1, module.build_vector_env, args,
        episode_steps=args.episode_steps, random_start=True, seed=args.seed,
    )
    if failures:
        return None, None, failures

    eval_env = None
    try:
        if not isinstance(train_env, VecNormalize):
            return None, None, [
                f"build_vector_env must return a VecNormalize, got {type(train_env).__name__}"]
        if not isinstance(train_env.venv, DummyVecEnv):
            failures.append(
                f"VecNormalize must wrap a DummyVecEnv, got {type(train_env.venv).__name__}")
        elif len(train_env.venv.envs) != args.n_envs:
            failures.append(
                f"expected {args.n_envs} parallel envs, got {len(train_env.venv.envs)}")

        if isinstance(train_env.venv, DummyVecEnv):
            wrapped = train_env.venv.envs
            if not all(isinstance(env, Monitor) for env in wrapped):
                failures.append("each sub-env must be wrapped in Monitor (history + infos)")
            inner = wrapped[0].unwrapped
            if not isinstance(inner, CloudAutoscalingEnv):
                failures.append(
                    f"Monitor must wrap CloudAutoscalingEnv, got {type(inner).__name__}")
            else:
                if inner.episode_steps != args.episode_steps:
                    failures.append(
                        f"episode_steps={inner.episode_steps}, expected {args.episode_steps}")
                if inner.random_start is not True:
                    failures.append("the training env must use random_start=True")
                for name in ("cost_weight", "sla_weight", "churn_weight"):
                    expected = getattr(args, name)
                    actual = getattr(inner, name)
                    if not np.isclose(actual, expected):
                        failures.append(
                            f"env {name}={actual} but the CLI asked for {expected}")

        if train_env.norm_obs is not True:
            failures.append(
                "norm_obs must be True - RPS spans 6-480 while CPU/RAM live in [0, 1]")
        if train_env.norm_reward is not True:
            failures.append("the training env must keep norm_reward=True")
        if not np.isclose(float(train_env.clip_obs), 10.0):
            failures.append(f"clip_obs={train_env.clip_obs}, expected 10.0")
        if not np.isclose(float(train_env.gamma), args.gamma):
            failures.append(f"VecNormalize gamma={train_env.gamma}, expected {args.gamma}")

        # VecNormalize reports the *raw* env space but returns normalised values,
        # so the correct invariants are shape + finite + clipped to clip_obs.
        observation = np.asarray(train_env.reset()[0], dtype=np.float32)
        if observation.shape != train_env.observation_space.shape:
            failures.append(
                f"reset() returned shape {observation.shape}, "
                f"expected {train_env.observation_space.shape}")
        if not np.all(np.isfinite(observation)):
            failures.append("reset() returned NaN/inf observations")
        if np.any(np.abs(observation) > float(train_env.clip_obs) + 1e-5):
            failures.append(
                f"normalised observations exceed clip_obs={train_env.clip_obs}: "
                "is norm_obs actually enabled?")

        observation, _, _, _ = train_env.step(np.zeros(args.n_envs, dtype=int))
        batched_shape = (args.n_envs,) + tuple(train_env.observation_space.shape)
        if np.asarray(observation).shape != batched_shape:
            failures.append(
                f"step() returned shape {np.asarray(observation).shape}, expected {batched_shape}")
        if not np.all(np.isfinite(np.asarray(observation))):
            failures.append("step() returned NaN/inf observations")
        if np.any(np.abs(np.asarray(observation)) > float(train_env.clip_obs) + 1e-5):
            failures.append("step() returned observations outside +/-clip_obs")

        eval_env, failures_eval = _call_step(
            1, module.build_vector_env, args,
            episode_steps=None, random_start=False, seed=args.seed + 1_000,
            n_envs=1, norm_reward=False,
        )
        if failures_eval:
            failures.extend(failures_eval)
            return train_env, None, failures
        if not isinstance(eval_env, VecNormalize):
            failures.append("the eval env must also be a VecNormalize")
            return train_env, None, failures
        if eval_env.norm_reward is not False:
            failures.append(
                "the evaluation env must set norm_reward=False so rewards stay comparable")
        if len(eval_env.venv.envs) != 1:
            failures.append(
                f"the evaluation env must be a single env, got {len(eval_env.venv.envs)}")
        inner_eval = eval_env.venv.envs[0].unwrapped
        if inner_eval.episode_steps != inner_eval.trace_length:
            failures.append("the evaluation env must run the full day (episode_steps=None)")
        if inner_eval.random_start is not False:
            failures.append("the evaluation env must use random_start=False")
    except Exception as exc:  # noqa: BLE001 - a crash inside a check is itself a failure
        failures.append(f"step 1 check crashed: {type(exc).__name__}: {exc}")
    return train_env, eval_env, failures


def _check_step2(module: ModuleType, args: argparse.Namespace, train_env):
    """Return ``(model, failures)``."""
    model, failures = _call_step(2, module.build_model, args, train_env)
    if failures:
        return None, failures
    try:
        expected = args.algo.upper()
        if type(model).__name__.upper() != expected:
            failures.append(
                f"build_model returned {type(model).__name__} for --algo {args.algo}")
        if tuple(model.observation_space.shape) != tuple(train_env.observation_space.shape):
            failures.append(
                f"model observation space {model.observation_space.shape} does not match "
                f"the env {train_env.observation_space.shape}")
        if model.action_space != train_env.action_space:
            failures.append(
                f"model action space {model.action_space} does not match "
                f"the env {train_env.action_space}")
        if not getattr(model, "tensorboard_log", None):
            failures.append("tensorboard_log is empty - the run would be invisible in TensorBoard")
        if args.algo == "ppo":
            if model.n_steps != args.n_steps:
                failures.append(f"PPO n_steps={model.n_steps}, expected {args.n_steps}")
            if model.batch_size != args.batch_size:
                failures.append(
                    f"PPO batch_size={model.batch_size}, expected {args.batch_size}")
            if model.gamma != args.gamma:
                failures.append(f"PPO gamma={model.gamma}, expected {args.gamma}")
        try:
            layers = [layer.out_features for layer in model.policy.mlp_extractor.policy_net]
            if layers[:2] != [128, 128]:
                failures.append(f"policy net_arch={layers}, expected [128, 128]")
        except AttributeError:
            pass  # a different but valid architecture - not worth failing on
    except Exception as exc:  # noqa: BLE001
        failures.append(f"step 2 check crashed: {type(exc).__name__}: {exc}")
    return model, failures


def _check_step3(module: ModuleType, args: argparse.Namespace, train_env, eval_env):
    """Return ``(callbacks, failures)``."""
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )

    callbacks, failures = _call_step(3, module.make_callbacks, args, train_env, eval_env,
                                     args.n_envs)
    if failures:
        return None, failures
    try:
        if not isinstance(callbacks, CallbackList):
            return None, [
                f"make_callbacks must return a CallbackList, got {type(callbacks).__name__}"]
        names = [type(cb).__name__ for cb in callbacks.callbacks]
        for required in ("SyncVecNormalizeCallback", "EvalCallback", "CheckpointCallback"):
            if names.count(required) != 1:
                failures.append(f"expected exactly one {required}, found {names.count(required)}")
        if not any(isinstance(cb, EvalCallback) for cb in callbacks.callbacks):
            return callbacks, failures

        eval_cb = next(cb for cb in callbacks.callbacks if isinstance(cb, EvalCallback))
        checkpoint_cb = next(
            (cb for cb in callbacks.callbacks if isinstance(cb, CheckpointCallback)), None)
        sync_index = next(
            (i for i, cb in enumerate(callbacks.callbacks)
             if type(cb).__name__ == "SyncVecNormalizeCallback"), None)
        eval_index = next(
            (i for i, cb in enumerate(callbacks.callbacks) if isinstance(cb, EvalCallback)), None)

        if sync_index is None:
            failures.append("no SyncVecNormalizeCallback: evaluations would use stale statistics")
        elif eval_index is not None and sync_index > eval_index:
            failures.append(
                "SyncVecNormalizeCallback must run *before* EvalCallback, "
                "otherwise every evaluation uses stale normalisation statistics")
        if eval_cb.eval_env is not eval_env:
            failures.append("EvalCallback must be given the frozen eval env you built, "
                            "not a fresh one")
        if not eval_cb.deterministic:
            failures.append("EvalCallback must be deterministic")
        expected_freq = max(1, args.eval_freq // args.n_envs)
        if eval_cb.eval_freq != expected_freq:
            failures.append(
                f"eval_freq={eval_cb.eval_freq}; use args.eval_freq // n_envs "
                f"(={expected_freq}) because callbacks count *per-env* steps")
        if checkpoint_cb is None:
            failures.append("no CheckpointCallback")
        else:
            if Path(checkpoint_cb.save_path).name != "checkpoints":
                failures.append(
                    f"checkpoints should go to models/checkpoints, got {checkpoint_cb.save_path}")
            if checkpoint_cb.name_prefix != args.algo:
                failures.append(
                    f"checkpoint name_prefix={checkpoint_cb.name_prefix!r}, expected {args.algo!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"step 3 check crashed: {type(exc).__name__}: {exc}")
    return callbacks, failures


def _check_step4(module: ModuleType, args: argparse.Namespace, model, train_env):
    """Save into a temp dir, then reload the pair the way ``evaluate.py`` does."""
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from env.cloud_env import CloudAutoscalingEnv

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="autoscale-selfcheck-") as tmp:
        result, failures = _call_step(4, module.save_artifacts, model, train_env, args,
                                      Path(tmp))
        if failures:
            return failures
        try:
            found = sorted(path.name for path in Path(tmp).iterdir())
            expected_model = f"{args.algo}_cloud_autoscaling.zip"
            expected_stats = f"{args.algo}_cloud_autoscaling_vecnormalize.pkl"
            if expected_model not in found:
                failures.append(f"missing {expected_model} (found: {found or 'nothing'})")
            if expected_stats not in found:
                failures.append(
                    f"missing {expected_stats} - a model without its VecNormalize "
                    "statistics cannot be evaluated")
            if failures:
                return failures
            if result is not None and len(result) == 2:
                returned_model = Path(result[0]).name
                returned_stats = Path(result[1]).name
                if returned_model not in (expected_model, expected_model[:-len(".zip")]):
                    failures.append(f"save_artifacts returned model path {result[0]}")
                if returned_stats != expected_stats:
                    failures.append(f"save_artifacts returned stats path {result[1]}")

            model_path, stats_path = Path(tmp) / expected_model, Path(tmp) / expected_stats
            algorithm = PPO if args.algo == "ppo" else DQN
            reloaded = algorithm.load(str(model_path))
            stats = VecNormalize.load(str(stats_path), DummyVecEnv([CloudAutoscalingEnv]))
            if tuple(stats.obs_rms.mean.shape) != tuple(train_env.observation_space.shape):
                failures.append("reloaded VecNormalize statistics do not match the observation space")
            if reloaded.observation_space.shape != train_env.observation_space.shape:
                failures.append("reloaded model does not accept the env's observations")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"step 4 check crashed: {type(exc).__name__}: {exc}")
    return failures

def run_self_check(module: ModuleType, args: argparse.Namespace) -> int:
    """Run all four contract checks; return a process exit code (0 = all pass)."""
    print("=" * 74)
    print(" Self-check: does each of the four steps satisfy its contract?")
    print("=" * 74)

    results: dict[int, list[str]] = {}
    train_env = eval_env = model = None
    try:
        train_env, eval_env, results[1] = _check_step1(module, args)
        if results[1]:
            for step in (2, 3, 4):
                results[step] = ["blocked - step 1 has to pass first"]
        else:
            model, results[2] = _check_step2(module, args, train_env)
            if results[2]:
                results[3] = ["blocked - step 2 has to pass first"]
            else:
                _, results[3] = _check_step3(module, args, train_env, eval_env)
            expected_model = f"{args.algo}_cloud_autoscaling.zip"
            results[4] = (["blocked - step 2 has to pass first"] if model is None
                          else _check_step4(module, args, model, train_env))
    finally:
        for env in (train_env, eval_env):
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001 - cleanup must never mask a result
                    pass

    print()
    for step in (1, 2, 3, 4):
        failures = results.get(step)
        if failures is None:
            failures = ["not reached"]
        status = "PASS" if not failures else "FAIL"
        print(f"[{status}] {STEP_TITLES[step]}")
        for failure in failures:
            print(f"       - {failure}")

    passed = sum(1 for step in results if not results[step])
    try:
        script = Path(module.__file__).resolve().relative_to(Path.cwd())
    except (AttributeError, ValueError):
        script = Path(getattr(module, "__file__", "agents/train_rl.py")).resolve()
    print()
    print("-" * 74)
    print(f"{passed}/4 steps pass.")
    if passed == 4:
        print("The plumbing is correct. Train with:")
        print(f"  python {script} --algo {args.algo} --timesteps 200000")
    else:
        print("Read the CONTRACT block in the docstring of the failing function in")
        print(f"{script}, implement it, and run --self-check again.")
        print("docs/WALKTHROUGH.md has the reasoning behind each step.")
    print("-" * 74)
    return 0 if passed == 4 else 1

