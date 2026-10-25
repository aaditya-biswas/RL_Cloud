"""Score one policy on many days in parallel and append to ``results/scores.csv``.

``utils/score_policy.py`` scores a policy on a single trace and prints a table.
Comparing a policy with a baseline needs that same measurement on every day of
the split, which is 28 subprocesses per policy - so this script is a thin,
parallel driver around it and nothing more. All numbers still come from
``score_policy.py`` (one code path, one definition of a violation), and the
merged CSV is the only thing the report and the plots read.

The output is *long format*, one row per (run, day)::

    run,policy,model,seed,steps_trained,day,split,date,violations,...return_v3,...

so a run is identified by its label and can be aggregated over days, and every
row keeps the day it came from (needed for the train/val/test comparison).

Usage
-----
    python utils/run_scores.py --run hold9 --policy hold --pods 9 --days all
    python utils/run_scores.py --run b1-u0.77 --policy b1 --u 0.77 --days all
    python utils/run_scores.py --run r1-s2 --policy model \\
        --model models/sla/r1-day16-s2.zip --seed 2 --days test
    python utils/run_scores.py --run r1-s2-ablate-time --policy model \\
        --model models/sla/r1-day16-s2.zip --seed 2 --days test --ablate time
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/run_scores.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.build_days import DAYS_DIR, RESULTS_DIR, SPLIT_RANGES, parse_days  # noqa: E402

RAW_DIR = RESULTS_DIR / "raw"
SCORES_PATH = RESULTS_DIR / "scores.csv"

# Columns copied verbatim from one ``score_policy.py --json`` row.
SCORE_COLUMNS = (
    "violations", "violation_pct", "avg_pods", "pod_hours", "cost_units", "churn_v3",
    "return_v1", "return_v2", "return_v3", "max_latency_ms", "p99_latency_ms",
    "actions", "steps",
)


def resolve_days(spec: str) -> list[int]:
    """``all`` / ``train`` / ``val`` / ``test`` / ``23-28`` -> day numbers."""
    spec = str(spec).strip()
    if spec == "all":
        return sorted(parse_days("1-28"))
    if spec in SPLIT_RANGES:
        lo, hi = SPLIT_RANGES[spec]
        return list(range(lo, hi + 1))
    return parse_days(spec)


def day_split(day: int) -> str:
    for name, (lo, hi) in SPLIT_RANGES.items():
        if lo <= day <= hi:
            return name
    return "unknown"


def build_command(args: argparse.Namespace, day: int) -> tuple[list[str], Path]:
    """The exact ``score_policy.py`` invocation for one day, and its JSON target."""
    trace = Path(args.days_dir).expanduser() / f"day_{day:02d}.csv"
    if not trace.exists():
        raise FileNotFoundError(f"missing trace {trace} (run utils/build_days.py)")
    suffix = f"-{args.ablate}" if args.ablate != "none" else ""
    json_path = RAW_DIR / f"{args.run}{suffix}_day_{day:02d}.json"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "utils" / "score_policy.py"),
        "--policy", args.policy,
        "--trace", str(trace),
        "--step-seconds", str(args.step_seconds),
        "--json", str(json_path),
    ]
    if args.policy == "hold":
        cmd += ["--pods", str(args.pods)]
    elif args.policy == "b1":
        cmd += ["--u", str(args.u), "--window", str(args.window)]
    else:
        cmd += ["--model", str(args.model), "--algo", args.algo,
                "--ablate", args.ablate]
        if args.vecnormalize:
            cmd += ["--vecnormalize", str(args.vecnormalize)]
    if args.start_pods is not None:
        cmd += ["--start-pods", str(args.start_pods)]
    return cmd, json_path

def run_one(args: argparse.Namespace, day: int) -> dict:
    """Score one day in a subprocess and return the merged long-format row."""
    cmd, json_path = build_command(args, day)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{args.run} day {day:02d} failed\n{proc.stdout}\n{proc.stderr}")
    payload = json.loads(json_path.read_text())
    row = payload["rows"][0]
    meta = payload["meta"]

    out = {
        "run": args.run,
        "policy": args.policy,
        "model": str(args.model) if args.model else "",
        "seed": int(args.seed),
        "steps_trained": int(args.steps_trained),
        "day": int(day),
        "split": day_split(day),
        "ablate": args.ablate,
        "u": float(args.u),
        "window": int(args.window),
        "pods": int(args.pods),
        "vecnormalize": meta.get("vecnormalize") or "",
    }
    for column in SCORE_COLUMNS:
        out[column] = row.get(column)
    out["json"] = str(json_path.relative_to(PROJECT_ROOT))
    return out


def merge(existing: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    """Append to the scores file, replacing rows for the same (run, day, ablate).

    Re-running a policy must not double-count a day, so a rerun overwrites its
    earlier row instead of accumulating a second copy.
    """
    frames = [fresh]
    if existing.exists():
        frames.insert(0, pd.read_csv(existing))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["run", "day", "ablate"], keep="last")
    return frame.sort_values(["run", "day"]).reset_index(drop=True)


def print_summary(frame: pd.DataFrame) -> None:
    """One line per run: days, total violations, mean/worst return v3."""
    print()
    print(f"  {'run':28s} {'days':>4s} {'viol':>6s} {'mean ret v3':>12s} "
          f"{'worst ret v3':>13s} {'pods':>6s} {'max ms':>8s}")
    for run, group in frame.groupby("run", sort=True):
        print(f"  {run[:28]:28s} {len(group):4d} {int(group['violations'].sum()):6d} "
              f"{group['return_v3'].mean():12,.0f} {group['return_v3'].min():13,.0f} "
              f"{group['avg_pods'].mean():6.2f} {group['max_latency_ms'].max():8,.0f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one policy across days and append to results/scores.csv.")
    parser.add_argument("--run", required=True,
                        help="Label for this run in the CSV, e.g. r1-s2 or b1-u0.77.")
    parser.add_argument("--policy", required=True, choices=("hold", "b1", "model"))
    parser.add_argument("--days", default="all",
                        help="'all', 'train', 'val', 'test', or day numbers like '23-28'.")
    parser.add_argument("--days-dir", type=Path, default=DAYS_DIR)
    parser.add_argument("--pods", type=int, default=9, help="--policy hold only.")
    parser.add_argument("--u", type=float, default=0.77, help="--policy b1 only.")
    parser.add_argument("--window", type=int, default=48, help="--policy b1 only.")
    parser.add_argument("--model", type=Path, default=None, help="--policy model only.")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Explicit VecNormalize pkl (default: next to the model).")
    parser.add_argument("--algo", choices=("auto", "ppo", "dqn"), default="auto")
    parser.add_argument("--ablate", choices=("none", "time"), default="none")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed of the checkpoint (recorded, not used).")
    parser.add_argument("--steps-trained", type=int, default=0,
                        help="Timesteps behind the checkpoint (recorded, not used).")
    parser.add_argument("--start-pods", type=int, default=None)
    parser.add_argument("--step-seconds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=SCORES_PATH)
    args = parser.parse_args(argv)
    if args.policy == "model" and args.model is None:
        parser.error("--policy model requires --model")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    days = resolve_days(args.days)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"== {args.run}: {args.policy} on {len(days)} day(s) "
          f"({args.days}), {args.workers} workers ==")
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_one, args, day): day for day in days}
        for future in futures:
            day = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep going: one bad day is not a lost run
                errors.append(str(exc))
                print(f"  day {day:02d}  FAILED")
                continue
            rows.append(row)
            print(f"  day {day:02d}  {row['split']:5s} viol {row['violations']:4d}  "
                  f"ret v3 {row['return_v3']:>10,.0f}  pods {row['avg_pods']:5.2f}  "
                  f"max {row['max_latency_ms']:,.0f} ms")

    if not rows:
        print("  nothing scored", file=sys.stderr)
        return 1

    frame = merge(out_path, pd.DataFrame(rows))
    frame.to_csv(out_path, index=False)
    print_summary(frame[frame["run"] == args.run])
    print(f"\n  scores  {out_path}")
    if errors:
        print(f"  {len(errors)} day(s) failed:", file=sys.stderr)
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


