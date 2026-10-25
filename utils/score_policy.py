"""Score any provisioning policy on the same day, under all three reward designs.

Every rung of the experiment is compared here, so no comparison depends on two
scripts agreeing about physics:

``--policy hold``    a constant fleet, the baseline (``--pods 9`` reproduces
                    -116,640 exactly: 864 pod-hours x 135 / hour).
``--policy b1``      the 480 s trailing-max tracker, the heuristic to beat.
``--policy model``   a trained checkpoint (11-dim design obs or the stock 5-dim).

Why one scorer for everything
-----------------------------
A policy is a *sequence of decisions*, so the only honest comparison is to replay
the decisions through one simulator and then price the same trajectory three
ways: reward v1 (the stock quadratic), v2 (linear, capped, churn on request) and
v3 (v2 with the 1,000-unit cap that the design settles on). Reporting one column
would invite the question "is the winner only winning because of its reward?", so
all three are printed for every row.

The trick that makes this cheap: the trajectory is recorded once (RPS, pods
before/after, requested delta, latency) and each reward family is then evaluated
on the recorded numbers. One rollout, three prices, zero re-simulation.

Note that ``utils/evaluate.py`` cannot score the 11-dim models: it rebuilds the
*stock* 5-dim environment before loading the ``VecNormalize`` statistics, and
``VecNormalize`` then feeds 11-dim observations into 5-dim statistics. This
scorer builds the matching stack first.

Usage
-----
    python utils/score_policy.py --policy hold --pods 9
    python utils/score_policy.py --policy b1
    python utils/score_policy.py --policy model --model models/sla/r2-lagrangian.zip
    python utils/score_policy.py --policy model --model models/sla/r2-lagrangian.zip \\
        --ablate time --json logs/score_r2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/score_policy.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cloud_env import (  # noqa: E402
    ACTION_DELTAS,
    DEFAULT_COST_WEIGHT,
    DEFAULT_MAX_PODS,
    DEFAULT_MIN_PODS,
    DEFAULT_SLA_LATENCY_MS,
    DEFAULT_STEP_SECONDS,
    DEFAULT_TRACE_PATH,
    CloudAutoscalingEnv,
    load_trace,
)
from env.design_wrappers import OBS_DIM, HeadroomObs  # noqa: E402
from utils.design_probe import REWARD_FAMILIES, reward_family, util_policy  # noqa: E402

# The three lines from ``docs/DESIGN.md`` a candidate has to be read against.
ACCEPTANCE_RETURN = -116_640.0  # constant 9 pods: 216 pod-hours
STRETCH_RETURN = -60_088.0      # trailing max 480 s at u=0.77
CEILING_RETURN = -38_658.0      # perfect-foresight DP


@dataclass
class Record:
    """One replay of one policy, in the units the reward families need."""

    rps: np.ndarray
    pods: np.ndarray        # after the action (what the customer experienced)
    previous: np.ndarray    # before the action (for the realised delta)
    requested: np.ndarray   # the action's delta, before clipping
    latency: np.ndarray

    def __len__(self) -> int:
        return int(self.rps.size)


@dataclass
class Row:
    """Scored summary of a replay, in the columns ``docs/DESIGN.md`` quotes."""

    policy: str
    steps: int
    violations: int
    violation_pct: float
    avg_pods: float
    pod_hours: float
    cost_units: float
    churn_v3: float
    return_v1: float
    return_v2: float
    return_v3: float
    max_latency_ms: float
    p99_latency_ms: float
    actions: int


# -------------------------------------------------------------------- replay
def rollout(env, steps: int, decide: Optional[Callable] = None,
            targets: Optional[np.ndarray] = None) -> Record:
    """Replay one policy over the trace and record what the customers saw.

    Two ways in:

    * ``targets`` - a *desired pod count* per step. The actuator is the same
      ``+-2 pods/step`` envelope the agent has, so a target becomes the largest
      legal move towards it. This is how ``utils/design_probe.simulate`` treats a
      tracker, and it is why the scripted rows here reproduce the DESIGN table
      exactly rather than approximately.
    * ``decide`` - a function from observation to action index, for checkpoints.

    Either way the recorded numbers come from the environment's ``info`` dict,
    never from the intent: pods reflect what was applied, ``requested`` what the
    policy asked for. Churn is charged on the latter, which is the whole point of
    v2/v3.
    """
    obs, _ = env.reset(seed=0)
    pods = int(env.unwrapped.active_pods)
    rps, applied, previous, requested, latency = [], [], [], [], []

    for index in range(steps):
        if targets is not None:
            target = int(np.clip(targets[index], DEFAULT_MIN_PODS, DEFAULT_MAX_PODS))
            action = ACTION_DELTAS.index(int(np.clip(target - pods, -2, 2)))
        else:
            action = int(decide(obs))
        obs, _, terminated, truncated, info = env.step(action)
        rps.append(float(info["request_rate_rps"]))
        applied.append(int(info["active_pods"]))
        previous.append(int(info["previous_pods"]))
        requested.append(int(ACTION_DELTAS[int(info["action"])]))
        latency.append(float(info["latency_ms"]))
        pods = int(info["active_pods"])
        if terminated or truncated:
            break

    return Record(
        rps=np.asarray(rps, dtype=np.float64),
        pods=np.asarray(applied, dtype=np.int64),
        previous=np.asarray(previous, dtype=np.int64),
        requested=np.asarray(requested, dtype=np.int64),
        latency=np.asarray(latency, dtype=np.float64),
    )


def summarise(label: str, record: Record,
              step_seconds: int = DEFAULT_STEP_SECONDS) -> Row:
    """Price one recorded trajectory under every reward family in the design."""
    steps = len(record)
    violations = int(np.sum(record.latency > DEFAULT_SLA_LATENCY_MS))

    returns = {}
    for name in REWARD_FAMILIES:
        cfg = reward_family(name)
        total = 0.0
        for pods, latency, previous, requested in zip(
            record.pods, record.latency, record.previous, record.requested
        ):
            total += cfg(int(pods), float(latency), int(pods - previous), int(requested))
        returns[name] = total

    return Row(
        policy=label,
        steps=steps,
        violations=violations,
        violation_pct=100.0 * violations / max(1, steps),
        avg_pods=float(record.pods.mean()),
        pod_hours=float(record.pods.sum()) * step_seconds / 3600.0,
        cost_units=DEFAULT_COST_WEIGHT * float(record.pods.sum()),
        churn_v3=reward_family("v3").churn_weight * float(np.abs(record.requested).sum()),
        return_v1=returns["v1"],
        return_v2=returns["v2"],
        return_v3=returns["v3"],
        max_latency_ms=float(record.latency.max()),
        p99_latency_ms=float(np.percentile(record.latency, 99)),
        actions=int(np.sum(record.pods != record.previous)),
    )


# ------------------------------------------------------------------ reporting
HEADER = ("policy", "viol", "rate%", "pods", "pod-h", "cost", "churn",
          "ret v1", "ret v2", "ret v3", "max ms", "p99 ms", "acts")
WIDTHS = (32, 5, 6, 6, 8, 10, 8, 10, 10, 10, 8, 8, 6)


def print_table(rows: list[Row]) -> None:
    """Fixed-width table, one row per policy, three reward prices per row."""
    header = f"{HEADER[0]:<{WIDTHS[0]}} " + " ".join(
        f"{head:>{width}}" for head, width in zip(HEADER[1:], WIDTHS[1:])
    )
    print("  " + header)
    for row in rows:
        cells = (
            f"{row.policy[:WIDTHS[0]]:<{WIDTHS[0]}}",
            f"{row.violations:>5d}",
            f"{row.violation_pct:>6.2f}",
            f"{row.avg_pods:>6.2f}",
            f"{row.pod_hours:>8.1f}",
            f"{row.cost_units:>10,.0f}",
            f"{row.churn_v3:>8,.0f}",
            f"{row.return_v1:>10,.0f}",
            f"{row.return_v2:>10,.0f}",
            f"{row.return_v3:>10,.0f}",
            f"{row.max_latency_ms:>8,.0f}",
            f"{row.p99_latency_ms:>8,.1f}",
            f"{row.actions:>6,d}",
        )
        print("  " + " ".join(cells))


def print_verdict(row: Row) -> None:
    """Read the row against the three lines the design is measured on.

    Returns are sums of negative costs, so **less negative is better** and the
    perfect-foresight ceiling is the largest number any policy can reach. A row
    that "beats the ceiling" is therefore not an achievement, it is a bug alarm:
    it can only happen if the row was priced differently or given information the
    DP did not have.
    """
    def compare(name: str, target: float) -> None:
        if abs(row.return_v3 - target) <= 0.5:
            verdict = "TIES "
        elif row.return_v3 > target:
            verdict = "BEATS"
        else:
            verdict = "WORSE"
        print(f"  {verdict}  {name:<32s} line {target:>10,.0f}   this {row.return_v3:>10,.0f}")

    print()
    print("  against docs/DESIGN.md (less negative is better):")
    met = row.violations == 0
    print(f"  {'PASS ' if met else 'FAIL '}  {'violations == 0':<32s} "
          f"{'':>10s}          {row.violations:>10d}")
    compare("acceptance: hold 9 pods", ACCEPTANCE_RETURN)
    compare("stretch: 480 s tracker", STRETCH_RETURN)
    compare("ceiling: perfect foresight", CEILING_RETURN)

    accepted = met and row.return_v3 > ACCEPTANCE_RETURN
    print()
    print(f"  {'ACCEPTED' if accepted else 'NOT ACCEPTED'}: "
          f"{'zero violations' if met else f'{row.violations} violations'} and "
          f"{'cheaper' if row.return_v3 > ACCEPTANCE_RETURN else 'not cheaper'} than the 9-pod hold")


# ---------------------------------------------------------------- model loading
def _resolve_stats_path(model_path: Path, explicit: Optional[str]) -> Optional[Path]:
    """Find the ``VecNormalize`` statistics that belong to ``model_path``.

    Same convention as ``utils/evaluate.py``: ``<stem>_vecnormalize.pkl`` next to
    the model, or an explicit path.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    for name in (f"{model_path.stem}_vecnormalize.pkl", "vecnormalize.pkl"):
        candidate = model_path.parent / name
        if candidate.exists():
            return candidate
    return None


def load_policy(model_path: Path, algo: str = "auto", stats_path: Optional[str] = None,
                ablate: str = "none", trace_path: Optional[Path] = None):
    """Load a checkpoint into ``(decide, label, obs_dim, stats_path)``.

    The observation wrapper is picked from the checkpoint's *own* observation
    space, so the stock 5-dim models and the 11-dim design models both score
    here. ``VecNormalize`` is then loaded against a matching environment - this
    is the step ``utils/evaluate.py`` hardcodes to the stock env, which is why it
    cannot score a design model.

    ``ablate="time"`` zeroes the ``fraction_of_day`` feature before normalisation,
    which is the transfer test: with the time index gone, the policy cannot know
    where in the day it is and must fall back on the trailing peaks.
    """
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")

    candidates = {
        "auto": [(PPO, "ppo"), (DQN, "dqn")],
        "ppo": [(PPO, "ppo")],
        "dqn": [(DQN, "dqn")],
    }
    model = None
    resolved = algo
    errors = []
    for model_class, name in candidates[algo]:
        try:
            model = model_class.load(str(path), device="cpu")
            resolved = name
            break
        except Exception as exc:  # wrong algorithm for this archive
            errors.append(f"{name}: {exc}")
    if model is None:
        raise RuntimeError(f"could not load {path} as {algo}\n" + "\n".join(errors))

    obs_dim = int(np.prod(model.observation_space.shape))
    if ablate == "time" and obs_dim != OBS_DIM:
        raise ValueError(f"--ablate time needs the {OBS_DIM}-dim design observation, "
                         f"this model has {obs_dim}")

    def probe_env():
        env = CloudAutoscalingEnv(trace_path=trace_path, episode_steps=1)
        return HeadroomObs(env) if obs_dim == OBS_DIM else env

    stats_path = _resolve_stats_path(path, stats_path)
    normalizer = None
    if stats_path is not None:
        normalizer = VecNormalize.load(str(stats_path), DummyVecEnv([probe_env]))
        normalizer.training = False
    else:
        print("[score] warning: no VecNormalize statistics found; the checkpoint is fed "
              "raw observations, which is almost certainly wrong for a PPO policy.")

    def decide(observation) -> int:
        raw = np.asarray(observation, dtype=np.float32).copy()
        if ablate == "time":
            raw[OBS_DIM - 1] = 0.0
        features = normalizer.normalize_obs(raw) if normalizer is not None else raw
        action, _ = model.predict(features, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    suffix = ", no time index" if ablate == "time" else ""
    label = f"{path.stem} [{resolved}{suffix}]"
    return decide, label, obs_dim, stats_path


# ----------------------------------------------------------------------- driver
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", choices=("hold", "b1", "model"), required=True,
                        help="Which rung of the ladder to score.")
    parser.add_argument("--pods", type=int, default=9,
                        help="Constant fleet size for --policy hold (default 9).")
    parser.add_argument("--u", type=float, default=0.77,
                        help="Target utilisation for --policy b1 (default 0.77).")
    parser.add_argument("--window", type=int, default=48,
                        help="Trailing window in steps for --policy b1 (48 = 480 s).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Checkpoint for --policy model.")
    parser.add_argument("--algo", choices=("auto", "ppo", "dqn"), default="auto")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Explicit VecNormalize .pkl (default: next to the model).")
    parser.add_argument("--ablate", choices=("none", "time"), default="none",
                        help="time = zero the fraction_of_day feature (transfer test).")
    parser.add_argument("--trace", type=Path, default=None,
                        help="Trace CSV (default: data/workload_trace.csv).")
    parser.add_argument("--episode-steps", type=int, default=None,
                        help="Steps to replay (default: the whole trace).")
    parser.add_argument("--start-pods", type=int, default=None,
                        help="Initial fleet (default: the environment's auto sizing).")
    parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS)
    parser.add_argument("--json", type=str, default=None,
                        help="Write the scored row to this JSON file.")
    args = parser.parse_args(argv)
    if args.policy == "model" and args.model is None:
        parser.error("--policy model requires --model PATH")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    trace = load_trace(args.trace)
    steps = min(int(args.episode_steps or trace.size), int(trace.size))
    trace_path = Path(args.trace) if args.trace else None

    targets = None
    decide = None
    obs_dim = 0
    stats_path = None
    start_pods = args.start_pods if args.start_pods is not None else "auto"

    if args.policy == "hold":
        targets = np.full(steps, args.pods, dtype=np.int64)
        label = f"constant {args.pods} pods"
        # Hold from step 0, otherwise the first decision is a ramp and the row
        # picks up churn the baseline never had.
        start_pods = args.pods
    elif args.policy == "b1":
        tracker = util_policy(trace, args.u, "trailing", args.window)
        targets = np.asarray([tracker(i, float(trace[i]), 0) for i in range(steps)],
                             dtype=np.int64)
        label = f"trailing max {args.window * args.step_seconds:,.0f} s, u={args.u:.2f}"
    else:
        decide, label, obs_dim, stats_path = load_policy(
            args.model,
            algo=args.algo,
            stats_path=args.vecnormalize,
            ablate=args.ablate,
            trace_path=trace_path,
        )

    env = CloudAutoscalingEnv(
        trace_path=args.trace,
        episode_steps=steps,
        random_start=False,
        initial_pods=start_pods,
        seed=0,
    )
    if obs_dim == OBS_DIM:
        env = HeadroomObs(env)

    record = rollout(env, steps, decide=decide, targets=targets)
    row = summarise(label, record, step_seconds=args.step_seconds)

    print("== scoring ==")
    print(f"  policy      {label}")
    print(f"  trace       {Path(args.trace).expanduser() if args.trace else DEFAULT_TRACE_PATH}")
    print(f"  replay      {row.steps:,} steps "
          f"({row.steps * args.step_seconds / 3600.0:.2f} h), "
          f"SLA {DEFAULT_SLA_LATENCY_MS:.0f} ms, cost {DEFAULT_COST_WEIGHT:.2f} / pod-step")
    if obs_dim:
        print(f"  observation {obs_dim} dims, stats {stats_path}")
    print()
    print_table([row])
    print_verdict(row)

    if args.json:
        out = Path(args.json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "policy": args.policy,
                "label": label,
                "trace": str(args.trace or DEFAULT_TRACE_PATH),
                "steps": row.steps,
                "step_seconds": args.step_seconds,
                "sla_ms": DEFAULT_SLA_LATENCY_MS,
                "cost_weight": DEFAULT_COST_WEIGHT,
                "model": str(args.model) if args.model else None,
                "vecnormalize": str(stats_path) if stats_path else None,
                "ablate": args.ablate,
                "utilisation": args.u,
                "window_steps": args.window,
            },
            "rows": [asdict(row)],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


