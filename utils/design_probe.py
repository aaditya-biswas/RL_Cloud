"""Measure how a reward / observation design interacts with the traffic trace.

This is the instrument behind ``docs/DESIGN.md``. It trains nothing; it replays the
trace through the same physics as :class:`env.cloud_env.CloudAutoscalingEnv` and
answers three questions in order:

1. **What does the traffic do?** Hourly demand, burst sizes, and whether the
   ``+-2 pods / step`` action space can physically follow a burst.
2. **What does the reward price?** The per-step cost of one breach expressed in
   pod-steps, and which constant fleet each reward ranks first.
3. **What is a simple policy worth?** Constant fleets, a causal trailing-max
   tracker, an oracle lookahead tracker, and the perfect-foresight DP ceiling.

The numbers printed here are the ones quoted in ``docs/DESIGN.md``.

Usage
-----
    python utils/design_probe.py                    # full report
    python utils/design_probe.py --reward v2        # one reward family
    python utils/design_probe.py --section price    # trace | price | policies | all
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/design_probe.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cloud_env import (  # noqa: E402
    DEFAULT_BASE_LATENCY_MS,
    DEFAULT_CHURN_WEIGHT,
    DEFAULT_COST_WEIGHT,
    DEFAULT_MAX_PODS,
    DEFAULT_MIN_PODS,
    DEFAULT_MU_POD,
    DEFAULT_OVERLOAD_LATENCY_MS,
    DEFAULT_SLA_LATENCY_MS,
    DEFAULT_SLA_WEIGHT,
    load_trace,
)

Policy = Callable[[int, float, int], int]
REWARD_FAMILIES = ("v1", "v2", "v3")


# --------------------------------------------------------------------------- rewards
@dataclass(frozen=True)
class RewardConfig:
    """Reward families compared by the probe.

    ``quadratic`` (v1, the sandbox default)
        ``-(cost*pods + sla_weight*max(0, latency-SLA)^2 + churn*|delta|)``

    ``linear`` (v2/v3, the proposal in ``docs/DESIGN.md``)
        A ramped penalty that reaches ``violation_cost`` exactly at the SLA and
        grows linearly beyond it, optionally capped, with churn charged on the
        *requested* change instead of the realised one.
    """

    name: str = "v1"
    mode: str = "quadratic"
    cost_weight: float = DEFAULT_COST_WEIGHT
    sla_weight: float = DEFAULT_SLA_WEIGHT
    churn_weight: float = DEFAULT_CHURN_WEIGHT
    soft_ms: float = 150.0
    hard_ms: float = DEFAULT_SLA_LATENCY_MS
    violation_cost: float = 200.0
    max_penalty: Optional[float] = None
    churn_on_request: bool = False

    def penalty(self, latency_ms: float) -> float:
        """Penalty term only (no cost, no churn) for a step at ``latency_ms``."""
        return float(self.penalty_array(np.asarray(latency_ms, dtype=float)))

    def penalty_array(self, latency_ms) -> np.ndarray:
        """Vectorised ``penalty`` (used by the DP sweep, where millions of terms occur)."""
        lat = np.asarray(latency_ms, dtype=float)
        if self.mode == "quadratic":
            return self.sla_weight * np.maximum(0.0, lat - self.hard_ms) ** 2
        ramp = self.violation_cost / max(1e-9, self.hard_ms - self.soft_ms)
        penalty = ramp * np.maximum(0.0, lat - self.soft_ms)
        if self.max_penalty is not None:
            penalty = np.minimum(penalty, self.max_penalty)
        return penalty

    def terms(
        self, pods: int, latency_ms: float, realised_delta: int, requested_delta: int = 0
    ) -> dict:
        """Reward split into its three parts (all <= 0), so they can be reported."""
        churn_delta = requested_delta if self.churn_on_request else realised_delta
        return {
            "capacity": -self.cost_weight * pods,
            "latency": -self.penalty(latency_ms),
            "churn": -self.churn_weight * abs(churn_delta),
        }

    def __call__(
        self, pods: int, latency_ms: float, realised_delta: int, requested_delta: int = 0
    ) -> float:
        return sum(self.terms(pods, latency_ms, realised_delta, requested_delta).values())

    def price_of_breach(self, pods: int, rps: float) -> float:
        """Reward paid by one 10 s step at ``rps`` with ``pods``, in reward units."""
        return self.penalty(latency(rps, pods)) + self.cost_weight * pods


def reward_family(name: str) -> RewardConfig:
    """Look up one of the reward designs discussed in ``docs/DESIGN.md``."""
    if name == "v1":
        return RewardConfig(name="v1")
    if name == "v2":
        return RewardConfig(name="v2", mode="linear", churn_weight=4.0, churn_on_request=True)
    if name == "v3":
        return RewardConfig(
            name="v3", mode="linear", churn_weight=4.0, churn_on_request=True, max_penalty=1_000.0
        )
    raise ValueError(f"unknown reward family {name!r}")


# --------------------------------------------------------------------------- physics
def latency(rps: float, pods: int) -> float:
    """Same latency model as ``CloudAutoscalingEnv._latency_ms``."""
    capacity = pods * DEFAULT_MU_POD
    if capacity > rps:
        return min(DEFAULT_OVERLOAD_LATENCY_MS, 1000.0 / (capacity - rps) + DEFAULT_BASE_LATENCY_MS)
    return DEFAULT_OVERLOAD_LATENCY_MS


def latency_array(rps: float, pods: np.ndarray) -> np.ndarray:
    """Vectorised ``latency`` over an array of pod counts."""
    capacity = np.asarray(pods, dtype=float) * DEFAULT_MU_POD
    with np.errstate(divide="ignore", invalid="ignore"):
        stable = np.minimum(DEFAULT_OVERLOAD_LATENCY_MS,
                            1000.0 / (capacity - rps) + DEFAULT_BASE_LATENCY_MS)
    return np.where(capacity > rps, stable, DEFAULT_OVERLOAD_LATENCY_MS)


def safe_utilisation(
    mu_pod: float = DEFAULT_MU_POD, sla_ms: float = DEFAULT_SLA_LATENCY_MS
) -> float:
    """Utilisation at which latency hits the SLA - the edge of the reward cliff.

    ``1000 / (mu_total - rps) + base = sla``  =>  ``mu_total - rps = 1000 / (sla - base)``
    """
    return 1.0 - 1000.0 / ((sla_ms - DEFAULT_BASE_LATENCY_MS) * mu_pod)


def auto_pods(rps: float) -> int:
    """The env's ``initial_pods='auto'`` sizing (about 80 % utilisation)."""
    return int(
        np.clip(
            np.ceil(max(rps, 1e-9) / (0.80 * DEFAULT_MU_POD)), DEFAULT_MIN_PODS, DEFAULT_MAX_PODS
        )
    )


# --------------------------------------------------------------------------- replay
@dataclass
class Metrics:
    total_reward: float
    violations: int
    actions: int
    requests: int
    avg_pods: float
    capacity: float = 0.0
    latency: float = 0.0
    churn: float = 0.0


def simulate(
    trace: np.ndarray,
    policy: Policy,
    reward: RewardConfig,
    start_pods: int,
    max_step: int = 2,
) -> Metrics:
    """Replay ``policy`` over ``trace`` under ``reward`` (mirrors ``env.step``)."""
    pods = int(start_pods)
    total = 0.0
    violations = actions = requests = pod_sum = 0
    capacity = latency_term = churn = 0.0

    for index, value in enumerate(trace):
        rps = float(value)
        want = policy(index, rps, pods)
        want = min(max(want, DEFAULT_MIN_PODS), DEFAULT_MAX_PODS)
        # The tracker's target is not an action; an agent would press the largest
        # envelope step that moves towards it. Charge churn on that intent.
        requested = min(max(want - pods, -max_step), max_step)
        applied = min(max(pods + requested, DEFAULT_MIN_PODS), DEFAULT_MAX_PODS)
        delta = applied - pods
        step_latency = latency(rps, applied)
        parts = reward.terms(applied, step_latency, delta, requested)
        total += parts["capacity"] + parts["latency"] + parts["churn"]
        capacity += parts["capacity"]
        latency_term += parts["latency"]
        churn += parts["churn"]
        violations += step_latency > DEFAULT_SLA_LATENCY_MS
        actions += delta != 0
        requests += want != pods
        pod_sum += applied
        pods = applied

    return Metrics(total, violations, actions, requests, pod_sum / max(1, len(trace)),
                   capacity, latency_term, churn)


# --------------------------------------------------------------------------- policies
def constant_policy(pods: int) -> Policy:
    """Never change the fleet - the baseline the workbook has to beat."""
    return lambda index, rps, current: pods


def util_policy(
    trace: np.ndarray, utilisation: float, forecast: str = "current", window: int = 3
) -> Policy:
    """Size the fleet to ``demand / (utilisation * mu_pod)``.

    ``forecast``
        ``current``  - this step's RPS (no anticipation)
        ``trailing`` - max over the last ``window`` steps (causal, no leak)
        ``oracle``   - max over the next ``window`` steps (leaks the future)
    """
    last = len(trace) - 1

    def policy(index: int, rps: float, current: int) -> int:
        if forecast == "current":
            demand = rps
        elif forecast == "trailing":
            demand = float(trace[max(0, index - window + 1): index + 1].max())
        elif forecast == "oracle":
            demand = float(trace[index: min(last + 1, index + window)].max())
        else:
            raise ValueError(f"unknown forecast {forecast!r}")
        pods = math.ceil(max(demand, 1e-9) / (utilisation * DEFAULT_MU_POD))
        return min(max(pods, DEFAULT_MIN_PODS), DEFAULT_MAX_PODS)

    return policy



# ------------------------------------------------------------------- DP ceiling
def dp_ceiling(trace: np.ndarray, reward: RewardConfig, start_pods: int, max_step: int = 2):
    """Perfect-foresight ceiling under the same ``+-max_step`` ramp limit.

    ``V[t, p]`` is the best return obtainable from step ``t`` with ``p`` pods;
    ``choice[t, p]`` the action that attains it. Returns ``(return, actions, avg_pods)``.
    """
    steps = len(trace)
    pods_grid = np.arange(DEFAULT_MIN_PODS, DEFAULT_MAX_PODS + 1)
    moves = np.arange(-max_step, max_step + 1)
    # (pods, moves) tables, reused every step
    applied = np.clip(pods_grid[:, None] + moves[None, :], DEFAULT_MIN_PODS, DEFAULT_MAX_PODS)
    offset = applied - DEFAULT_MIN_PODS
    realised = applied - pods_grid[:, None]
    churn_delta = moves[None, :] if reward.churn_on_request else realised

    value = np.zeros((steps + 1, pods_grid.size))
    choice = np.zeros((steps, pods_grid.size), dtype=np.int64)

    for t in range(steps - 1, -1, -1):
        rps = float(trace[t])
        step_reward = -(
            reward.cost_weight * applied
            + reward.penalty_array(latency_array(rps, applied))
            + reward.churn_weight * np.abs(churn_delta)
        )
        candidates = step_reward + value[t + 1][offset]
        best = candidates.argmax(axis=1)
        value[t] = candidates[np.arange(pods_grid.size), best]
        choice[t] = moves[best]

    total = float(value[0, start_pods - DEFAULT_MIN_PODS])
    pods = int(start_pods)
    actions = pod_sum = 0
    for t in range(steps):
        applied_pods = int(choice[t, pods - DEFAULT_MIN_PODS]) + pods
        applied_pods = int(np.clip(applied_pods, DEFAULT_MIN_PODS, DEFAULT_MAX_PODS))
        actions += applied_pods != pods
        pod_sum += applied_pods
        pods = applied_pods
    return total, actions, pod_sum / steps


# --------------------------------------------------------------------------- reports
def report_trace(trace: np.ndarray) -> None:
    deltas = np.abs(np.diff(trace))
    print("== 1. the traffic ==")
    print(f"  bins {len(trace):,}   mean {trace.mean():.1f}   p50 {np.median(trace):.1f}   "
          f"p95 {np.percentile(trace, 95):.1f}   peak {trace.max():.1f} @ {int(trace.argmax())}")
    print(f"  |dRPS| per 10 s: p50 {np.percentile(deltas, 50):.1f}   p90 "
          f"{np.percentile(deltas, 90):.1f}   p99 {np.percentile(deltas, 99):.1f}   "
          f"max {deltas.max():.1f}")
    print(f"  steps that outrun 1 pod/step: {int((deltas > DEFAULT_MU_POD).sum()):,}   "
          f"that outrun the +-2 limit: {int((deltas > 2 * DEFAULT_MU_POD).sum()):,}")
    print("  window    mean   peak   pods needed at peak (80% util)")
    for hour in range(0, 24, 3):
        window = trace[hour * 360: (hour + 3) * 360]
        print(f"  {hour:02d}-{hour + 3:02d}h  {window.mean():6.1f} {window.max():6.1f}   "
              f"{auto_pods(window.max()):6d}")


def report_price(trace: np.ndarray) -> None:
    peak_rps = float(trace.max())
    print()
    print("== 2. what the reward prices ==")
    print(f"  safe utilisation u* = {safe_utilisation():.4f}  "
          f"(pods = ceil(rps / {safe_utilisation() * DEFAULT_MU_POD:.1f}))")
    print("  one 10 s breach at the peak (480 rps, 8 pods -> 3000 ms):")
    for name in REWARD_FAMILIES:
        reward = reward_family(name)
        cost = reward.price_of_breach(8, peak_rps)
        print(f"    {name}: {cost:9.0f} reward = {cost / reward.cost_weight:8.0f} pod-steps "
              f"= {cost / reward.cost_weight * 10 / 3600:6.1f} h of one spare pod")
    print()
    print("  constant fleets, return under each reward:")
    print("  pods " + " ".join(f"{name:>12s}" for name in REWARD_FAMILIES))
    start = auto_pods(trace[0])
    best = {name: (-np.inf, 0) for name in REWARD_FAMILIES}
    for pods in range(6, DEFAULT_MAX_PODS + 1):
        cells = []
        for name in REWARD_FAMILIES:
            metrics = simulate(trace, constant_policy(pods), reward_family(name), start_pods=pods)
            if metrics.total_reward > best[name][0]:
                best[name] = (metrics.total_reward, pods)
            cells.append(f"{metrics.total_reward:12.0f}")
        print(f"  {pods:4d} " + " ".join(cells))
    for name in REWARD_FAMILIES:
        print(f"  best constant fleet ({name}) = {best[name][1]} pods "
              f"(return {best[name][0]:.0f})")




def report_policies(trace: np.ndarray, families=REWARD_FAMILIES) -> None:
    print()
    print("== 3. what a simple policy is worth ==")
    print("  (tracker u is the best of 0.70..0.96; DP ceiling is perfect foresight)")
    start = auto_pods(trace[0])
    # (label, forecast, window in steps). The 480 s row matters: memory, not
    # foresight, is what a causal controller is missing on this trace.
    trackers = (
        ("current step", "current", 3),
        ("trailing 30 s", "trailing", 3),
        ("trailing 480 s", "trailing", 48),
        ("oracle 20 s", "oracle", 3),
    )
    for name in families:
        reward = reward_family(name)
        print()
        print(f"  -- reward {name} --")
        print(f"  {'policy':30s} {'return':>10s} {'viol':>5s} {'acts':>6s} {'pods':>6s} "
              f"{'capacity':>10s} {'latency':>9s} {'churn':>8s}")
        rows = []
        for pods in (8, 9, 10, 15):
            rows.append((f"constant {pods} pods",
                         simulate(trace, constant_policy(pods), reward, start_pods=pods)))
        for label, forecast, window in trackers:
            scored = [
                (simulate(trace, util_policy(trace, u / 100.0, forecast, window), reward, start), u)
                for u in range(70, 97)
            ]
            metrics, u = max(scored, key=lambda item: item[0].total_reward)
            rows.append((f"{label} u={u / 100:.2f}", metrics))
        for label, metrics in rows:
            print(f"  {label:30s} {metrics.total_reward:10.0f} {metrics.violations:5d} "
                  f"{metrics.actions:6d} {metrics.avg_pods:6.2f} {metrics.capacity:10.0f} "
                  f"{metrics.latency:9.0f} {metrics.churn:8.0f}")
        ceiling, ceiling_actions, ceiling_pods = dp_ceiling(trace, reward, start)
        print(f"  {'DP ceiling (perfect foresight)':30s} {ceiling:10.0f} {'-':>5s} "
              f"{ceiling_actions:6d} {ceiling_pods:6.2f}")


def report_actuator(trace: np.ndarray, reward_name: str = "v2") -> None:
    """Actuator speed (``max_step``) against information (forecast window).

    Answers the question "should I widen the action space instead of fixing the
    observation?" - a real trade-off in this trace, because 192 steps/day move
    faster than a +-2 actuator.
    """
    reward = reward_family(reward_name)
    start = auto_pods(trace[0])
    trackers = (
        ("current step", "current", 3),
        ("trailing 30 s", "trailing", 3),
        ("trailing 480 s", "trailing", 48),
    )
    print()
    print(f"== 4. actuator speed vs information (reward {reward_name}, +-2 rows in bold) ==")
    print(f"  {'max_step':>8s} {'demand signal':16s} {'best u':>6s} {'return':>10s} "
          f"{'viol':>5s} {'acts':>6s} {'pods':>6s}")
    for max_step in (1, 2, 3, 5, 15):
        for label, forecast, window in trackers:
            scored = [
                (simulate(trace, util_policy(trace, u / 100.0, forecast, window),
                          reward, start, max_step=max_step), u)
                for u in range(70, 97)
            ]
            metrics, u = max(scored, key=lambda item: item[0].total_reward)
            marker = "*" if max_step == 2 else " "
            print(f" {marker}{max_step:>7d} {label:16s} {u / 100:6.2f} "
                  f"{metrics.total_reward:10.0f} {metrics.violations:5d} "
                  f"{metrics.actions:6d} {metrics.avg_pods:6.2f}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trace", type=Path, default=None,
                        help="Trace CSV (default: data/workload_trace.csv).")
    parser.add_argument("--section",
                        choices=["all", "trace", "price", "policies", "actuator"],
                        default="all")
    parser.add_argument("--reward", choices=REWARD_FAMILIES, action="append",
                        help="Restrict the policy comparison to these reward families.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    trace = load_trace(args.trace)
    families = tuple(args.reward) if args.reward else REWARD_FAMILIES

    if args.section in ("all", "trace"):
        report_trace(trace)
    if args.section in ("all", "price"):
        report_price(trace)
    if args.section in ("all", "policies"):
        report_policies(trace, families)
    if args.section == "actuator":
        report_actuator(trace, families[0] if len(families) == 1 else "v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
