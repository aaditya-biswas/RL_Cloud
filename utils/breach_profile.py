#!/usr/bin/env python
"""Anatomy of the residual breaches: how long, how deep, how far short.

``results/scores.csv`` says *how many* steps each arm breaches. This script says
what those steps look like, for any checkpoint and any split:

* burst length - consecutive violating steps (is the policy recovering, or is the
  day lost?);
* depth - the latency value at the breach, against the ``3,000 ms`` overload
  clamp of the M/M/k model in ``env/cloud_env.py``;
* shortfall - ``ceil(rps / 60)`` is the fleet that keeps utilisation under 1.0, so
  ``pods - ceil(rps / 60)`` says how many pods late the policy was.

It is a diagnostic, not a scorer: the counts it prints must agree with
``results/scores.csv`` for the same (checkpoint, day), which is the check that
keeps it honest.

Usage::

    python utils/breach_profile.py models/sla/r1-day16-s3.zip models/sla/r3-day16-s3.zip
    python utils/breach_profile.py --days all models/sla/r3-day16-s2.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/breach_profile.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cloud_env import (  # noqa: E402
    DEFAULT_MU_POD, DEFAULT_MAX_PODS, DEFAULT_OVERLOAD_LATENCY_MS, DEFAULT_SLA_LATENCY_MS,
    CloudAutoscalingEnv, load_trace,
)
from env.design_wrappers import OBS_DIM, HeadroomObs  # noqa: E402
from utils.score_policy import load_policy, rollout  # noqa: E402
from utils.run_scores import resolve_days  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "days"


def bursts(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, length), ...]`` for the runs of True in ``mask``."""
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def profile(path: Path, days: list[int]) -> None:
    decide, label, obs_dim, _ = load_policy(Path(path))
    total_steps = breached_days = 0
    lengths: list[int] = []
    shortfalls: list[int] = []
    depths: list[float] = []
    print(f"== {label}  ({path.name})")
    for day in days:
        trace_path = DATA_DIR / f"day_{day:02d}.csv"
        env = CloudAutoscalingEnv(trace_path=trace_path, episode_steps=None,
                                  random_start=False, initial_pods="auto", seed=0)
        if obs_dim == OBS_DIM:
            env = HeadroomObs(env)
        record = rollout(env, int(load_trace(trace_path).size), decide=decide)
        bad = np.asarray(record.latency) > DEFAULT_SLA_LATENCY_MS
        if not bad.any():
            continue
        breached_days += 1
        total_steps += int(bad.sum())
        needed = np.ceil(np.asarray(record.rps) / DEFAULT_MU_POD).astype(int)
        for start, length in bursts(bad):
            pods = int(record.pods[start])
            rps = float(record.rps[start])
            latency = float(record.latency[start])
            lengths.append(length)
            shortfalls.append(int(needed[start]) - pods)
            depths.append(latency)
            clock = f"{start * 10 // 3600:02d}:{start * 10 % 3600 // 60:02d}"
            print(f"   day {day:02d} {clock}  {length} step(s)  rps {rps:6.1f}  "
                  f"pods {pods:2d} (needs {needed[start]:2d}, "
                  f"{needed[start] - pods:+d})  latency {latency:7.1f} ms  "
                  f"cap {DEFAULT_MAX_PODS}")
    if not total_steps:
        print("   no breaches\n")
        return
    print(f"   {total_steps} violating step(s) in {len(lengths)} burst(s) on "
          f"{breached_days}/{len(days)} day(s)")
    print(f"   burst length: mean {np.mean(lengths):.2f}, max {max(lengths)}; "
          f"share of 1-step bursts {np.mean(np.asarray(lengths) == 1):.0%}")
    print(f"   shortfall at breach: mean {np.mean(shortfalls):.2f} pod(s), "
          f"max {max(shortfalls)}, share short by 1 {np.mean(np.asarray(shortfalls) == 1):.0%}")
    print(f"   latency at breach: {min(depths):.0f}-{max(depths):.0f} ms, "
          f"share at the {DEFAULT_OVERLOAD_LATENCY_MS:.0f} ms overload clamp "
          f"{np.mean(np.asarray(depths) >= DEFAULT_OVERLOAD_LATENCY_MS):.0%}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path, help="checkpoint .zip files")
    parser.add_argument("--days", default="test",
                        help="'train', 'val', 'test', 'all' or '23-28' (default: test)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    days = resolve_days(args.days)
    print(f"days: {', '.join(str(day) for day in days)}")
    for path in args.models:
        profile(path, days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
