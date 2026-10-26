#!/usr/bin/env python
"""The cost-safety frontier on a split: what "clean" costs.

``results/scores.csv`` says what the learned rungs and the two trackers scored,
but not what a *clean* policy would have cost, so the report cannot answer the
question a reader always asks: "hold-9 is clean at -116,640; is there a cheaper
policy that is also clean, and how far are the learned rows from it?"

This script fills that gap with the two cheapest families of policy that are not
learned, replayed through exactly the same actuator and priced by exactly the
same reward as the scored rows:

* ``hold-p``  - a constant fleet of ``p`` pods (3..10).
* ``b1-u``    - ``ceil(trailing-48-step max / (u * 60))`` pods, the tracker
  behind ``results/baselines.csv`` (``--policy b1 --window 48``).

Both are replayed via ``utils.make_plots.replay_targets`` so the numbers are
comparable with ``results/scores.csv`` cell for cell.

Usage::

    python utils/frontier.py                       # test split, writes results/frontier.txt
    python utils/frontier.py --days all --out ""
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cloud_env import CloudAutoscalingEnv, load_trace  # noqa: E402
from utils.design_probe import util_policy  # noqa: E402
from utils.score_policy import rollout  # noqa: E402
from utils.score_policy import summarise as price  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "days"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_JSON = RESULTS_DIR / "split.json"


def replay(day: int, *, pods: int | None = None, u: float | None = None,
           window: int = 48) -> dict:
    """Replay one constant fleet or one tracker on one day.

    The env, the actuator, the target construction and the pricing all come from
    ``utils/score_policy.py`` - the module that produced ``results/scores.csv`` -
    so a tracker row here is the same number as the corresponding baseline row.
    """
    path = DATA_DIR / f"day_{day:02d}.csv"
    trace = load_trace(path)
    steps = int(trace.size)
    if pods is not None:
        targets = np.full(steps, pods, dtype=np.int64)
        start_pods: int | str = pods  # hold from step 0, as score_policy does
        label = f"constant {pods} pods"
    else:
        policy = util_policy(trace, float(u), "trailing", window)
        targets = np.asarray([policy(i, float(trace[i]), 0) for i in range(steps)],
                             dtype=np.int64)
        start_pods = "auto"
        label = f"trailing max {window * 10:,.0f} s, u={u:.2f}"

    env = CloudAutoscalingEnv(trace_path=path, episode_steps=steps,
                              random_start=False, initial_pods=start_pods, seed=0)
    row = price(label, rollout(env, steps, targets=targets))
    return {field: getattr(row, field) for field in row.__dataclass_fields__}


def summarise_days(days: list[int], **kwargs) -> dict:
    """Mean over days, plus the pooled violation count (the gate)."""
    rows = [replay(day, **kwargs) for day in days]
    return {
        "days": len(days),
        "violations": sum(r["violations"] for r in rows),
        "days_with_violation": sum(1 for r in rows if r["violations"]),
        "return_v3": float(np.mean([r["return_v3"] for r in rows])),
        "avg_pods": float(np.mean([r["avg_pods"] for r in rows])),
    }


def split_days(name: str) -> list[int]:
    """Days of one split from the frozen ``results/split.json``."""
    split = json.loads(SPLIT_JSON.read_text())["split"]
    if name == "all":
        return sorted(day for days in split.values() for day in days)
    if name not in split:
        raise SystemExit(f"unknown split {name!r}; expected one of {sorted(split)} or 'all'")
    return sorted(split[name])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", default="test",
                        help="'train', 'val', 'test' or 'all' (default: test)")
    parser.add_argument("--out", default="results/frontier.txt",
                        help="write the table here; '' prints only")
    parser.add_argument("--us", default="0.55,0.58,0.60,0.62,0.64,0.65,0.70,0.74,0.77",
                        help="comma-separated tracker utilisations")
    args = parser.parse_args(argv)

    days = split_days(args.days)
    lines = [f"# cost-safety frontier, split={args.days} ({len(days)} days)",
             "| policy | days | violations | days_with_violation | mean_return_v3 | avg_pods |",
             "|---|---|---|---|---|---|"]
    rows = []
    for pods in range(3, 11):
        rows.append((f"hold-{pods}", summarise_days(days, pods=pods)))
    for u in [float(x) for x in args.us.split(",") if x.strip()]:
        rows.append((f"b1-u{u:.2f}", summarise_days(days, u=u)))
    for label, row in rows:
        lines.append(f"| {label} | {row['days']} | {row['violations']} | "
                     f"{row['days_with_violation']} | {row['return_v3']:,.0f} | "
                     f"{row['avg_pods']:.2f} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
