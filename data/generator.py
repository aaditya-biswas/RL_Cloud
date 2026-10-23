"""Generate the 24-hour HTTP request-rate trace used by the autoscaling environment.

Two sources are supported:

``real`` (default)
    Replays one calendar day of the NASA Kennedy Space Center HTTP access log
    (``raw/access_log_Jul95``). Every log line is binned into 10-second slots,
    short logging outages are linearly interpolated, and the resulting RPS
    curve is rescaled from the 1995 traffic volume to a modern, production-like
    peak so that the simulated pod fleet is actually stressed.

``synthetic``
    Reproducible fallback (diurnal sine wave + Gaussian noise + bursty spikes)
    for machines that do not have the 196 MB raw log. Produces the exact same
    CSV schema, so everything downstream is unchanged.

Output schema (8,640 rows = 24 h at 1 sample / 10 s)::

    timestamp,request_rate_rps
    1995-07-13 00:00:00,36.183
    1995-07-13 00:00:10,38.421
    ...

Usage
-----
    python data/generator.py                      # real trace, peak 480 RPS
    python data/generator.py --date 13 --target-peak-rps 480
    python data/generator.py --source synthetic --seed 7
    python data/generator.py --no-scale           # keep native 1995 RPS
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_LOG = PROJECT_ROOT / "raw" / "access_log_Jul95"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "workload_trace.csv"

STEP_SECONDS = 10
STEPS_PER_DAY = 24 * 60 * 60 // STEP_SECONDS  # 8_640
DEFAULT_DATE = "13/Jul/1995"
DEFAULT_TARGET_PEAK_RPS = 480.0
DATE_FORMAT = "%d/%b/%Y"

# The access log stores the timestamp inside square brackets, e.g.
#   199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
# -----------------------------------------------------------------------------------
#                            |<- 20 chars: 01/Jul/1995:00:00:01
_TS_OFFSET_END = 20
_TS_HOUR = slice(12, 14)
_TS_MINUTE = slice(15, 17)
_TS_SECOND = slice(18, 20)


def count_requests_per_bin(raw_log_path: Path, date_token: str) -> np.ndarray:
    """Count request lines per 10-second bin for a single day of the access log.

    Parameters
    ----------
    raw_log_path:
        Path to the raw NASA log file.
    date_token:
        Day prefix as it appears in the log, e.g. ``"13/Jul/1995"``.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of length 8,640 with the raw request count per bin.
    """
    raw_log_path = Path(raw_log_path)
    if not raw_log_path.exists():
        raise FileNotFoundError(
            f"Raw access log not found: {raw_log_path}\n"
            "Place the NASA access log there, or run with --source synthetic."
        )

    counts = np.zeros(STEPS_PER_DAY, dtype=np.int64)
    matched = 0

    with raw_log_path.open("r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            bracket = line.find("[")
            if bracket == -1:
                continue
            # Cheap prefix filter before doing any integer parsing.
            if not line.startswith(date_token, bracket + 1):
                continue
            timestamp = line[bracket + 1: bracket + 1 + _TS_OFFSET_END]
            try:
                hour = int(timestamp[_TS_HOUR])
                minute = int(timestamp[_TS_MINUTE])
                second = int(timestamp[_TS_SECOND])
            except ValueError:  # malformed / truncated log line
                continue
            bin_index = (hour * 3_600 + minute * 60 + second) // STEP_SECONDS
            if 0 <= bin_index < STEPS_PER_DAY:
                counts[bin_index] += 1
                matched += 1

    if matched == 0:
        raise RuntimeError(
            f"No log lines matched date {date_token!r} in {raw_log_path}. "
            "Check the --date argument (format: DD/Mon/YYYY)."
        )
    return counts


def fill_missing_bins(counts: np.ndarray) -> Tuple[np.ndarray, int]:
    """Linearly interpolate bins with no observed requests.

    The 1995 log has genuine collector outages (e.g. ~24 minutes around 19:00 on
    July 13). Treating those holes as "zero traffic" would teach the agent to
    scale to one pod right before a spike, so we interpolate them instead.

    Returns
    -------
    (filled_counts, n_missing_bins)
    """
    observed = counts > 0
    n_missing = int((~observed).sum())
    if not observed.any():
        raise RuntimeError("Trace contains no observed requests; nothing to interpolate.")
    positions = np.arange(counts.size, dtype=np.float64)
    filled = np.interp(positions, positions[observed], counts[observed].astype(np.float64))
    return filled, n_missing


def build_real_profile(
    raw_log_path: Path,
    date_token: str,
    target_peak_rps: float,
    scale: float | None = None,
) -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, float]]:
    """Build the replayed, rescaled RPS profile of one day of the access log."""
    counts = count_requests_per_bin(raw_log_path, date_token)
    filled, n_missing = fill_missing_bins(counts)

    rps = filled / float(STEP_SECONDS)
    native_peak = float(rps.max())
    factor = float(scale) if scale is not None else target_peak_rps / native_peak
    rps = rps * factor

    start = datetime.strptime(date_token, DATE_FORMAT)
    timestamps = pd.date_range(start=start, periods=STEPS_PER_DAY, freq=f"{STEP_SECONDS}s")

    meta = {
        "source": 1.0,  # numeric on purpose: keeps this dict float-only
        "native_peak_rps": native_peak,
        "scale_factor": factor,
        "missing_bins": float(n_missing),
        "observed_bins": float(STEPS_PER_DAY - n_missing),
    }
    return timestamps, rps, meta


def build_synthetic_profile(
    seed: int,
    target_peak_rps: float,
    n_bursts: int = 8,
) -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, float]]:
    """Build a reproducible diurnal sine + noise + burst profile.

    Used as a fallback when the raw access log is unavailable. The shape mirrors
    the July 13 replay: a quiet night, a morning ramp, an afternoon plateau and
    a sharp evening drop, decorated with 3-15 minute traffic bursts.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(STEPS_PER_DAY) * STEP_SECONDS / 3_600.0

    # 0.20 at ~03:00, 0.90 at ~15:00.
    diurnal = 0.55 + 0.35 * np.sin(2.0 * np.pi * (hours - 9.0) / 24.0)
    noise = rng.normal(0.0, 0.05, size=STEPS_PER_DAY)
    profile = np.clip(diurnal + noise, 0.08, None)

    for _ in range(n_bursts):
        start = int(rng.integers(0, STEPS_PER_DAY - 120))
        length = int(rng.integers(18, 90))  # 3-15 minutes at 10 s resolution
        profile[start:start + length] *= float(rng.uniform(1.6, 2.8))

    rps = profile / profile.max() * float(target_peak_rps)
    timestamps = pd.date_range(
        start=datetime(2024, 1, 1), periods=STEPS_PER_DAY, freq=f"{STEP_SECONDS}s"
    )
    meta = {
        "source": 2.0,
        "native_peak_rps": float(rps.max()),
        "scale_factor": 1.0,
        "missing_bins": 0.0,
        "observed_bins": float(STEPS_PER_DAY),
    }
    return timestamps, rps, meta

def write_trace(timestamps: pd.DatetimeIndex, rps: np.ndarray, output_path: Path) -> pd.DataFrame:
    """Persist the trace as CSV with the schema consumed by the environment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "request_rate_rps": np.round(rps, 3),
        }
    )
    frame.to_csv(output_path, index=False)
    return frame


def _print_summary(frame: pd.DataFrame, meta: Dict[str, float], output_path: Path) -> None:
    rps = frame["request_rate_rps"].to_numpy()
    source = "real (NASA KSC access log)" if meta["source"] == 1.0 else "synthetic"
    print("=" * 68)
    print("Workload trace generated")
    print("=" * 68)
    print(f"  source            : {source}")
    print(f"  rows              : {len(frame):,} ({STEP_SECONDS}s resolution)")
    print(f"  window            : {frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]}")
    print(f"  mean RPS          : {rps.mean():8.2f}")
    print(f"  median RPS        : {np.median(rps):8.2f}")
    print(f"  p95 RPS           : {np.percentile(rps, 95):8.2f}")
    print(f"  peak RPS          : {rps.max():8.2f}")
    print(f"  min RPS           : {rps.min():8.2f}")
    print(f"  native peak RPS   : {meta['native_peak_rps']:8.2f}")
    print(f"  scale factor      : {meta['scale_factor']:8.2f}x")
    print(f"  interpolated bins : {int(meta['missing_bins']):,} of {STEPS_PER_DAY:,}")
    print(f"  output            : {output_path}")
    print("=" * 68)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["real", "synthetic"], default="real",
                        help="Trace source (default: real NASA access log).")
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_RAW_LOG,
                        help="Path to the raw NASA access log.")
    parser.add_argument("--date", default=DEFAULT_DATE,
                        help="Day to replay from the log, format DD/Mon/YYYY (default: 13/Jul/1995).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination CSV path.")
    parser.add_argument("--target-peak-rps", type=float, default=DEFAULT_TARGET_PEAK_RPS,
                        help="Rescale the day so its busiest 10s bin hits this RPS value.")
    parser.add_argument("--scale", type=float, default=None,
                        help="Explicit multiplicative scale, overrides --target-peak-rps.")
    parser.add_argument("--no-scale", action="store_true",
                        help="Keep the native 1995 request rate (scale factor 1.0).")
    parser.add_argument("--seed", type=int, default=7,
                        help="RNG seed used by the synthetic generator.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.source == "synthetic":
        timestamps, rps, meta = build_synthetic_profile(args.seed, args.target_peak_rps)
    else:
        scale = 1.0 if args.no_scale else args.scale
        timestamps, rps, meta = build_real_profile(
            raw_log_path=args.raw_log,
            date_token=args.date,
            target_peak_rps=args.target_peak_rps,
            scale=scale,
        )

    frame = write_trace(timestamps, rps, args.output)
    _print_summary(frame, meta, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
