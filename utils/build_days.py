"""Materialise every day of the NASA access log and freeze the 16/6/6 split.

``data/workload_trace.csv`` is a single day (13 Jul 1995). Learning across only
one day cannot distinguish "learned to autoscale" from "memorised Thursday the
13th", so this script replays all 28 available days (1-28 Jul 1995) into
``data/days/day_DD.csv`` and records, for each day, its shape statistics (mean /
p95 / peak RPS, interpolated bins, rescaling factor) and its split.

Every day is rescaled to the same peak (``--target-peak-rps``, default 480 RPS)
so the workload *shape* varies while the difficulty ceiling stays fixed. That is
what keeps the constant-pods baseline fair: the 9-pod fleet serves 540 RPS, so
hold-9 is SLA-clean on every day and costs the same -116,640 per day, i.e. a
fixed yardstick instead of a moving target.

Split (chronological, never shuffled - day 17 follows day 16 in real time):

    train  days 01-16   (16 days)
    val    days 17-22   ( 6 days, used by EvalCallback)
    test   days 23-28   ( 6 days, touched exactly once, in score_policy.py)

Outputs
-------
``data/days/day_DD.csv``      one 8,640-row trace per day
``results/inventory.csv``     one row per day: shape stats + split
``results/split.json``        the split, machine-readable, for trainer and scorer

Usage
-----
    python utils/build_days.py                  # build all 28 days (~60 s)
    python utils/build_days.py --force          # rebuild existing files
    python utils/build_days.py --days 13        # rebuild a single day
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/build_days.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generator import (  # noqa: E402
    DEFAULT_RAW_LOG,
    DEFAULT_TARGET_PEAK_RPS,
    STEPS_PER_DAY,
    build_real_profile,
    write_trace,
)

DAYS_DIR = PROJECT_ROOT / "data" / "days"
RESULTS_DIR = PROJECT_ROOT / "results"

# 1 Jul 1995 was a Saturday, so day % 7 in {1, 2} are weekend days. Native
# weekend traffic is 2-4x quieter; after per-day rescaling the difference shows
# up as a flatter curve rather than a smaller one. They stay in the split on
# purpose: a policy that never saw a weekend day should not be scored on one
# without that being visible in the report.
SPLIT_RANGES = {"train": (1, 16), "val": (17, 22), "test": (23, 28)}
DAY_TOKEN = "{day:02d}/Jul/1995"
DATE_FORMAT = "%d/%b/%Y"


def split_of(day: int) -> str:
    """Name of the split a day belongs to (``train`` / ``val`` / ``test``)."""
    for name, (lo, hi) in SPLIT_RANGES.items():
        if lo <= day <= hi:
            return name
    raise ValueError(f"day {day} is outside the split ranges {SPLIT_RANGES}")


def day_window(split: str) -> list[int]:
    """Inclusive day numbers of one split, in calendar order."""
    lo, hi = SPLIT_RANGES[split]
    return list(range(lo, hi + 1))


def parse_days(spec: str) -> list[int]:
    """Parse ``"1-28"`` / ``"13"`` / ``"1,3,17-19"`` into a sorted day list."""
    days: set[int] = set()
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(part) for part in chunk.split("-", 1))
            days.update(range(lo, hi + 1))
        else:
            days.add(int(chunk))
    if not days:
        raise ValueError(f"no days parsed from {spec!r}")
    return sorted(days)


def describe(path: Path, day: int) -> dict:
    """One inventory row, read back from the CSV the scorer will actually see.

    Reading from disk matters: ``write_trace`` rounds RPS to 3 decimals, so the
    in-memory array is not exactly what downstream code loads.
    """
    frame = pd.read_csv(path)
    values = frame["request_rate_rps"].to_numpy(dtype=float)
    if len(frame) != STEPS_PER_DAY:
        raise RuntimeError(f"{path}: {len(frame)} rows, expected {STEPS_PER_DAY}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{path}: non-finite RPS values")
    stamp = datetime.strptime(DAY_TOKEN.format(day=day), DATE_FORMAT)

    # Logging outages are linearly interpolated by the generator, so a long gap
    # becomes a run of identical values. Recording the longest run and the flat
    # tail keeps that visible instead of letting an outage look like calm traffic.
    flat = np.diff(values) == 0
    longest = run = 1
    for flag in flat:
        run = run + 1 if flag else 1
        longest = max(longest, run)
    tail = len(values) - 1
    while tail > 0 and values[tail] == values[tail - 1]:
        tail -= 1
    trailing = len(values) - 1 - tail

    return {
        "day": day,
        "date": stamp.strftime("%Y-%m-%d"),
        "weekday": stamp.strftime("%a"),
        "split": split_of(day),
        "rows": len(frame),
        "mean_rps": round(float(values.mean()), 2),
        "p95_rps": round(float(np.percentile(values, 95)), 2),
        "peak_rps": round(float(values.max()), 2),
        "min_rps": round(float(values.min()), 2),
        "native_peak_rps": float("nan"),
        "scale_factor": float("nan"),
        "missing_bins": -1,
        "longest_flat_bins": int(longest),
        "trailing_flat_bins": int(trailing),
        "flat_fraction": round(trailing / len(values), 3),
        "peak_pods": int(np.ceil(values.max() / 60.0)),
        "mean_pods": round(float(values.mean()) / 60.0, 2),
        "path": str(path.relative_to(PROJECT_ROOT)),
    }


def build_day(day: int, raw_log: Path, target_peak_rps: float, out_dir: Path) -> dict:
    """Replay one calendar day, write ``data/days/day_DD.csv``, describe it."""
    token = DAY_TOKEN.format(day=day)
    timestamps, rps, meta = build_real_profile(raw_log, token, target_peak_rps)
    path = out_dir / f"day_{day:02d}.csv"
    write_trace(timestamps, rps, path)
    row = describe(path, day)
    row["native_peak_rps"] = round(float(meta["native_peak_rps"]), 3)
    row["scale_factor"] = round(float(meta["scale_factor"]), 3)
    row["missing_bins"] = int(meta["missing_bins"])
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-day traces and freeze the chronological 16/6/6 split.")
    parser.add_argument("--days", default="1-28",
                        help="Day numbers to build, e.g. '1-28' or '13,23-25'.")
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_RAW_LOG,
                        help="NASA access log (default: raw/access_log_Jul95).")
    parser.add_argument("--target-peak-rps", type=float, default=DEFAULT_TARGET_PEAK_RPS,
                        help="Rescale each day so its busiest 10 s bin hits this RPS.")
    parser.add_argument("--out-dir", type=Path, default=DAYS_DIR,
                        help="Where the day_*.csv files go (default: data/days).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Where inventory.csv and split.json go.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild days whose CSV already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_log = Path(args.raw_log).expanduser()
    if not raw_log.exists():
        raise SystemExit(f"raw log not found: {raw_log}")
    days = parse_days(args.days)
    out_dir = Path(args.out_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows = []
    for day in days:
        path = out_dir / f"day_{day:02d}.csv"
        cached = path.exists() and not args.force
        if cached and len(pd.read_csv(path, usecols=["request_rate_rps"])) != STEPS_PER_DAY:
            cached = False  # truncated file: rebuild it
        if cached:
            row = describe(path, day)
            print(f"  day {day:02d}  cached  {path.name}")
        else:
            row = build_day(day, raw_log, float(args.target_peak_rps), out_dir)
            print(f"  day {day:02d}  {row['split']:5s} {row['weekday']}  "
                  f"mean {row['mean_rps']:6.1f}  p95 {row['p95_rps']:6.1f}  "
                  f"peak {row['peak_rps']:6.1f}  x{row['scale_factor']:.2f}  "
                  f"-> {row['path']}")
        rows.append(row)

    inventory = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    inventory_path = results_dir / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_log": (str(raw_log.relative_to(PROJECT_ROOT))
                    if raw_log.is_relative_to(PROJECT_ROOT) else str(raw_log)),
        "target_peak_rps": float(args.target_peak_rps),
        "steps_per_day": STEPS_PER_DAY,
        "step_seconds": 10,
        "split": {name: day_window(name) for name in SPLIT_RANGES},
        "files": {name: [f"data/days/day_{d:02d}.csv" for d in day_window(name)]
                  for name in SPLIT_RANGES},
        "days": {int(r["day"]): {key: r[key] for key in
                                 ("date", "weekday", "split", "mean_rps", "p95_rps",
                                  "peak_rps", "native_peak_rps", "scale_factor")}
                 for r in rows},
    }
    split_path = results_dir / "split.json"
    split_path.write_text(json.dumps(manifest, indent=2) + "\n")

    elapsed = time.time() - started
    print()
    print(f"== built {len(rows)} day(s) in {elapsed:,.1f}s ==")
    for name in SPLIT_RANGES:
        subset = inventory[inventory["split"] == name]
        print(f"  {name:5s} {len(subset):2d} days  "
              f"mean {subset['mean_rps'].mean():6.1f} RPS  "
              f"p95 {subset['p95_rps'].max():6.1f} RPS")
    print(f"  inventory  {inventory_path}")
    print(f"  split      {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
