"""Kubernetes-HPA-style rule-based autoscaler used as the benchmarking baseline.

The baseline only looks at CPU utilisation, exactly like a simple
HorizontalPodAutoscaler::

    cpu > 0.75  -> action 3 (+1 pod)
    cpu < 0.25  -> action 1 (-1 pod)
    otherwise   -> action 2 (hold)

Because it reacts to the *current* utilisation it always lags behind a ramp,
which is precisely the gap the RL agent is supposed to close.

Usage
-----
    python agents/baseline_scaler.py            # full-day benchmark
    python agents/baseline_scaler.py --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python agents/baseline_scaler.py`
    sys.path.insert(0, str(PROJECT_ROOT))


class RuleBasedScaler:
    """Threshold policy over the 5-dimensional autoscaling observation.

    Parameters
    ----------
    scale_up_threshold:
        CPU utilisation above which one pod is added.
    scale_down_threshold:
        CPU utilisation below which one pod is removed.
    cooldown_steps:
        Minimum number of environment steps between two scaling decisions
        (mimics the HPA stabilisation window). ``0`` disables the cooldown.
    """

    SCALE_UP_ACTION = 3    # +1 pod
    SCALE_DOWN_ACTION = 1  # -1 pod
    HOLD_ACTION = 2        # no change

    def __init__(
        self,
        scale_up_threshold: float = 0.75,
        scale_down_threshold: float = 0.25,
        cooldown_steps: int = 0,
    ) -> None:
        if not 0.0 <= scale_down_threshold < scale_up_threshold <= 1.0:
            raise ValueError(
                "Expected 0 <= scale_down_threshold < scale_up_threshold <= 1, got "
                f"{scale_down_threshold} and {scale_up_threshold}"
            )
        self.scale_up_threshold = float(scale_up_threshold)
        self.scale_down_threshold = float(scale_down_threshold)
        self.cooldown_steps = int(max(0, cooldown_steps))
        self._steps_since_change = 0

    def reset(self) -> None:
        """Reset the internal cooldown counter (call at the start of an episode)."""
        self._steps_since_change = 0

    @property
    def _cooldown_elapsed(self) -> bool:
        return self._steps_since_change >= self.cooldown_steps

    def predict(self, observation) -> int:
        """Return an action index in ``{0, 1, 2, 3, 4}`` for a single observation."""
        observation = np.asarray(observation, dtype=np.float64).ravel()
        if observation.size < 1:
            raise ValueError("Observation must contain at least cpu_utilization at index 0")
        cpu = float(observation[0])

        if self._cooldown_elapsed and cpu > self.scale_up_threshold:
            action = self.SCALE_UP_ACTION
        elif self._cooldown_elapsed and cpu < self.scale_down_threshold:
            action = self.SCALE_DOWN_ACTION
        else:
            action = self.HOLD_ACTION

        if action == self.HOLD_ACTION:
            self._steps_since_change += 1
        else:
            self._steps_since_change = 0
        return action


def _demo(steps: int | None, trace: Path | None) -> int:
    """Run the baseline over the workload trace and print a quick summary."""
    from env.cloud_env import CloudAutoscalingEnv

    env = CloudAutoscalingEnv(trace_path=trace, episode_steps=steps, random_start=False)
    scaler = RuleBasedScaler()
    observation, _ = env.reset(seed=42, options={"start_index": 0})

    pods, latencies, rewards, actions = [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = scaler.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        pods.append(info["active_pods"])
        latencies.append(info["latency_ms"])
        rewards.append(reward)
        actions.append(action)

    latencies = np.asarray(latencies)
    print("=" * 68)
    print("Rule-based baseline (HPA-style thresholds)")
    print("=" * 68)
    print(f"  steps               : {len(pods):,} ({env.episode_length:,} horizon)")
    print(f"  mean pods           : {np.mean(pods):6.2f}  (max {max(pods)})")
    print(f"  mean latency        : {latencies.mean():8.2f} ms")
    print(f"  p95 latency         : {np.percentile(latencies, 95):8.2f} ms")
    print(f"  SLA violations      : {100.0 * (latencies > env.sla_latency_ms).mean():6.2f} %")
    print(f"  total reward        : {np.sum(rewards):12.1f}")
    counts = np.bincount(np.asarray(actions), minlength=5)
    print(f"  action counts       : {dict(enumerate(counts.tolist()))}")
    print("=" * 68)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=None,
                        help="Episode horizon (default: the full 8,640-step trace).")
    parser.add_argument("--trace", type=Path, default=None,
                        help="Workload CSV (default: data/workload_trace.csv).")
    args = parser.parse_args(argv)
    return _demo(args.steps, args.trace)


if __name__ == "__main__":
    raise SystemExit(main())
