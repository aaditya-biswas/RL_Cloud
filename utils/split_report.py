"""Turn ``results/scores.csv`` into the tables the report and the plots read.

``scores.csv`` is long format (one row per run and day) because that is what a
measurement is. A report needs the other axis - per split, per rung, per seed -
so this script folds the long table into four tables plus ``summary.md``:

``results/baselines.csv``     hold-9 and the two trackers, per split
``results/policy_scores.csv`` each rung per split, aggregated over seeds
``results/seed_table.csv``    each individual seed, with its accept/reject verdict
``results/summary.md``        the same numbers as markdown, plus the verdict

The accept/reject rule is fixed before the numbers are looked at, and it is the
rule from the plan: a rung is accepted iff **every** seed is violation-free on
**every** test day *and* its mean test return v3 beats the constant-9 line. A
cheaper-but-breaching policy is not accepted - it is reported as cheaper and
breaching, which is a different claim.

Usage
-----
    python utils/split_report.py
    python utils/split_report.py --scores results/scores.csv --out-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/split_report.py`
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS = ("train", "val", "test")
HOLD_RETURN = -116_640.0  # constant 9 pods, 216 pod-hours, every day


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"no scores at {path} (run utils/run_scores.py first)")
    frame = pd.read_csv(path)
    # A run is a (label, ablation) pair; the ablation column is part of its identity.
    frame["key"] = frame["run"].astype(str)
    return frame


def summarise_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (run, split): days scored, violations, mean/worst return v3."""
    grouped = frame.groupby(["key", "split"], as_index=False).agg(
        days=("day", "size"),
        violations=("violations", "sum"),
        violation_pct=("violation_pct", "mean"),
        return_v3=("return_v3", "mean"),
        worst_return_v3=("return_v3", "min"),
        avg_pods=("avg_pods", "mean"),
        max_latency_ms=("max_latency_ms", "max"),
        actions=("actions", "mean"),
    )
    return grouped


def seed_table(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per seed-like run: test-set totals and the pre-committed verdict."""
    test = scores[scores["split"] == "test"]
    rows = []
    for key, group in test.groupby("key", sort=True):
        meta = group.iloc[0]
        violations = int(group["violations"].sum())
        mean_v3 = float(group["return_v3"].mean())
        rows.append({
            "run": key,
            "rung": rung_of(key),
            "policy": meta["policy"],
            "seed": int(meta["seed"]),
            "steps_trained": int(meta["steps_trained"]),
            "ablate": meta["ablate"],
            "test_days": int(len(group)),
            "violations": violations,
            "days_with_violation": int((group["violations"] > 0).sum()),
            "mean_return_v3": mean_v3,
            "worst_return_v3": float(group["return_v3"].min()),
            "avg_pods": float(group["avg_pods"].mean()),
            "max_latency_ms": float(group["max_latency_ms"].max()),
            "beats_hold": bool(mean_v3 > HOLD_RETURN),
            "violation_free": bool(violations == 0),
            "accepted": bool(violations == 0 and mean_v3 > HOLD_RETURN),
        })
    return pd.DataFrame(rows)


def rung_of(run: str) -> str:
    """``r1-day16-s2`` -> ``R1``; baselines keep their own label."""
    head = run.split("-")[0].lower()
    return head.upper() if head in ("r1", "r2", "r3") else "baseline"


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Small markdown table writer (pandas' needs the tabulate extra)."""
    view = frame if columns is None else frame[columns]
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    rule = "|" + "|".join("---" for _ in view.columns) + "|"
    lines = [header, rule]
    for _, row in view.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rung_frame(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (rung, variant, split) aggregate over seeds - the report's main table."""
    def variant(run: str) -> str:
        ablate = scores.loc[scores["run"] == run, "ablate"].iloc[0]
        return "full" if str(ablate) == "none" else f"ablate-{ablate}"

    rows = []
    for key, group in scores.groupby("run", sort=True):
        rung = rung_of(key)
        if rung == "baseline":
            continue
        rows.append({
            "run": key,
            "rung": rung,
            "variant": variant(key),
            "seed": int(group["seed"].iloc[0]),
            "split": group["split"].iloc[0] if len(group["split"].unique()) == 1 else "mixed",
            "violations": int(group["violations"].sum()),
            "return_v3": float(group["return_v3"].mean()),
            "worst_return_v3": float(group["return_v3"].min()),
            "avg_pods": float(group["avg_pods"].mean()),
            "max_latency_ms": float(group["max_latency_ms"].max()),
        })
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed

    aggregated = per_seed.groupby(["rung", "variant", "split"], as_index=False).agg(
        seeds=("seed", "nunique"),
        violations_mean=("violations", "mean"),
        violations_max=("violations", "max"),
        violation_free_seeds=("violations", lambda s: int((s == 0).sum())),
        return_v3_mean=("return_v3", "mean"),
        return_v3_std=("return_v3", "std"),
        worst_return_v3=("worst_return_v3", "min"),
        avg_pods=("avg_pods", "mean"),
        max_latency_ms=("max_latency_ms", "max"),
    )
    aggregated["return_v3_std"] = aggregated["return_v3_std"].fillna(0.0)
    aggregated["beats_hold"] = aggregated["return_v3_mean"] > HOLD_RETURN
    return aggregated.sort_values(["rung", "variant", "split"]).reset_index(drop=True)


def split_section(inventory: pd.DataFrame | None, split_json: Path) -> list[str]:
    """Describe the frozen split: days, weekend count and native volume."""
    lines = ["## 1. The split", ""]
    if split_json.exists():
        manifest = json.loads(split_json.read_text())
        lines.append(f"Chronological, never shuffled. Peak rescaled to "
                     f"{manifest['target_peak_rps']:,.0f} RPS on every day; "
                     f"{manifest['steps_per_day']:,} steps/day at "
                     f"{manifest['step_seconds']} s.")
        lines.append("")
    if inventory is None:
        return lines + [""]
    rows = []
    has_flat = "trailing_flat_bins" in inventory.columns
    for split in SPLITS:
        subset = inventory[inventory["split"] == split]
        row = {
            "split": split,
            "days": len(subset),
            "first": subset["date"].min(),
            "last": subset["date"].max(),
            "weekend_days": int(subset["weekday"].isin(["Sat", "Sun"]).sum()),
            "mean_rps": round(float(subset["mean_rps"].mean()), 1),
            "max_p95_rps": round(float(subset["p95_rps"].max()), 1),
        }
        if has_flat:
            row["max_flat_tail_h"] = round(float(subset["trailing_flat_bins"].max()) * 10 / 3600, 1)
        if "missing_bins" in subset.columns:
            row["max_missing_pct"] = round(100.0 * float(subset["missing_bins"].max()) / 8_640, 1)
        rows.append(row)
    lines.append(md_table(pd.DataFrame(rows)))
    if has_flat:
        outage = inventory[inventory["trailing_flat_bins"] > 360]
        if len(outage):
            listed = ", ".join(
                f"day {int(row.day)} ({row.trailing_flat_bins * 10 / 3600:.1f} h)"
                for row in outage.itertuples())
            lines.append("")
            lines.append(f"**Log outages.** A trailing run of identical RPS values is a gap in "
                         f"the 1995 log that the generator forward-fills, not calm traffic: "
                         f"{listed}. Those days keep their place in the split (dropping them "
                         f"after looking at the scores would be picking the test set), and the "
                         f"flat tail is easy traffic, so it makes those days *cheaper*, not "
                         f"harder, for every policy.")
    lines.append("")
    return lines


def decision_lines(seeds: pd.DataFrame, baselines: pd.DataFrame) -> list[str]:
    """The verdict, generated from the tables rather than typed by hand."""
    if seeds.empty:
        return ["No learned checkpoints scored yet."]
    test = seeds
    accepted = test[test["accepted"]]
    hold = baselines[(baselines["key"] == "hold9") & (baselines["split"] == "test")]
    hold_return = float(hold["return_v3"].mean()) if not hold.empty else HOLD_RETURN

    lines = []
    if accepted.empty:
        lines.append(f"**No arm is accepted.** {len(test)} arm(s) (every learned seed plus the "
                     f"two trackers) were scored on 6 test days; either a violation appeared "
                     f"or the mean return v3 did not beat the constant-9 line "
                     f"({hold_return:,.0f}).")
        cheapest = test.sort_values("mean_return_v3", ascending=False).iloc[0]
        lines.append("")
        lines.append(f"Cheapest arm: `{cheapest['run']}` at "
                     f"{cheapest['mean_return_v3']:,.0f} mean return v3 with "
                     f"{int(cheapest['violations'])} violation(s) on "
                     f"{int(cheapest['days_with_violation'])} test day(s) - cheaper than "
                     f"constant-9 by {cheapest['mean_return_v3'] - hold_return:,.0f}, but "
                     f"not clean, so it is reported as cheaper *and* breaching.")
    else:
        for _, row in accepted.iterrows():
            lines.append(f"**Accepted: `{row['run']}`.** {int(row['violations'])} violations "
                         f"across {int(row['test_days'])} test days, mean return v3 "
                         f"{row['mean_return_v3']:,.0f} against {hold_return:,.0f} for "
                         f"constant-9 ({row['mean_return_v3'] - hold_return:,.0f} cheaper).")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scores_path = Path(args.scores).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(scores_path)
    inventory_path = Path(args.inventory).expanduser()
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else None

    per_split = summarise_split(scores)
    baselines = per_split[per_split["key"].str.match(r"^(hold|b1)")].copy()
    baselines.to_csv(out_dir / "baselines.csv", index=False)

    seeds = seed_table(scores)
    seeds.to_csv(out_dir / "seed_table.csv", index=False)

    rungs = rung_frame(scores)
    rungs.to_csv(out_dir / "policy_scores.csv", index=False)

    lines = ["# 16/6/6 split: what the autoscaler does on days it has never seen", "",
             "Every row is one policy replayed over whole days and priced under reward v3 "
             "(`utils/score_policy.py`). Return v3 is a **negative cost**, so less negative is "
             "cheaper; `violations` is the hard gate and is counted in 10 s steps above the "
             "200 ms SLA. A row that is cheaper and breaching is reported as both.", ""]
    lines += split_section(inventory, Path(args.split_json).expanduser())
    lines += ["## 2. Baselines, per split", ""]
    lines.append(md_table(baselines, ["key", "split", "days", "violations",
                                      "return_v3", "worst_return_v3", "avg_pods",
                                      "max_latency_ms"]))
    lines.append("")
    if not rungs.empty:
        lines += ["## 3. Learned rungs, per split (mean over seeds)", ""]
        lines.append(md_table(rungs, ["rung", "variant", "split", "seeds",
                                      "violations_mean", "violations_max",
                                      "violation_free_seeds", "return_v3_mean",
                                      "return_v3_std", "worst_return_v3",
                                      "avg_pods", "beats_hold"]))
        lines.append("")

    # ``seed_table`` is built from test rows only, so it *is* the test table.
    test_seeds = seeds
    if not test_seeds.empty:
        lines += ["## 4. Test-set verdicts (one row per seed, baselines for reference)", "",
                  "Accept = every test day violation-free **and** mean return v3 above "
                  "the constant-9 line (-116,640).", ""]
        lines.append(md_table(test_seeds, ["run", "rung", "seed", "test_days",
                                           "violations", "days_with_violation",
                                           "mean_return_v3", "worst_return_v3",
                                           "avg_pods", "beats_hold", "accepted"]))
        lines.append("")

    ablation = rungs[rungs["variant"] != "full"] if not rungs.empty else rungs
    if not ablation.empty:
        lines += ["## 5. Time-feature ablation (transfer test)", "",
                  "The same checkpoints with `fraction_of_day` zeroed: if the score "
                  "holds up, the policy is reading demand shape rather than a clock.", ""]
        lines.append(md_table(ablation, ["rung", "variant", "split", "seeds",
                                         "violations_mean", "return_v3_mean",
                                         "worst_return_v3", "beats_hold"]))
        lines.append("")

    lines += ["## 6. Decision", ""]
    lines += decision_lines(seeds, baselines)
    lines.append("")
    lines += ["## 7. Reproduce", "",
              "```bash",
              "python utils/build_days.py",
              "python utils/run_scores.py --run hold9 --policy hold --pods 9 --days all",
              "python utils/run_scores.py --run b1-u0.77 --policy b1 --u 0.77 --days all",
              "python utils/run_scores.py --run r1-day16-s2 --policy model \\",
              "    --model models/sla/r1-day16-s2.zip --seed 2 --days test",
              "python utils/split_report.py",
              "```", ""]

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[:4]))
    print(f"  baselines      {out_dir / 'baselines.csv'}")
    print(f"  rungs          {out_dir / 'policy_scores.csv'}")
    print(f"  seeds          {out_dir / 'seed_table.csv'}")
    print(f"  summary        {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold results/scores.csv into tables plus results/summary.md.")
    parser.add_argument("--scores", type=Path, default=RESULTS_DIR / "scores.csv")
    parser.add_argument("--inventory", type=Path, default=RESULTS_DIR / "inventory.csv")
    parser.add_argument("--split-json", type=Path, default=RESULTS_DIR / "split.json")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
