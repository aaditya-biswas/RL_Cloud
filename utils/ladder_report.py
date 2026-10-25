"""Collate every scored run into one markdown table.

Reads ``logs/runs/*.json`` (written by ``utils/score_policy.py``), groups the
checkpoints by experiment rung, and reports mean +/- std across seeds. A rung is
*accepted* only if every seed has zero violations and beats the 9-pod hold.

    python utils/ladder_report.py            # markdown for docs/DESIGN.md §11
    python utils/ladder_report.py > /tmp/t.md
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs" / "runs"

RUNGS = [
    ("B0", "constant 9 pods", ("hold9", {})),
    ("B1", "trailing max 480 s, u=0.77", ("b1", {})),
    ("R1", "reward v3 only (price = 0)", ("r1-", {"r1-reward-only": 42})),
    ("R2", "Lagrangian, price step 50", ("r2-", {"r2-lagrangian": 42})),
    ("R3", "Lagrangian, price step 20,000", ("r3-", {})),
]

FIELDS = [
    ("violations", "viol", "{:.0f}"),
    ("violation_pct", "rate %", "{:.3f}"),
    ("avg_pods", "pods", "{:.2f}"),
    ("pod_hours", "pod-h", "{:.1f}"),
    ("churn_v3", "churn", "{:.0f}"),
    ("return_v3", "ret v3", "{:,.0f}"),
    ("max_latency_ms", "max ms", "{:.0f}"),
    ("actions", "acts", "{:.0f}"),
]


def load(tag):
    path = RUNS / f"{tag}.json"
    if not path.exists():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    return payload["rows"][0]


def collect(prefix, aliases):
    rows = []
    for path in sorted(RUNS.glob("*.json")):
        tag = path.stem
        if path.stem in ("hold9", "b1", "table"):
            continue
        if not tag.startswith(prefix) and tag not in aliases:
            continue
        row = load(tag)
        if row is not None:
            rows.append((tag, aliases.get(tag, int(tag.rsplit("s", 1)[-1])
                                          if "s" in tag else 0), row))
    return rows


print("| rung | policy | seeds | viol | rate % | pods | pod-h | churn | ret v3 | max ms | acts | verdict |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")

for knob, description, (prefix, aliases) in RUNGS:
    if prefix in ("hold9", "b1"):
        row = load(prefix)
        if row is None:
            print(f"| {knob} | {description} | - | (not scored) | | | | | | | | |")
            continue
        cells = [str(row[key]) if key == "violations" else fmt.format(row[key])
                 for key, _, fmt in FIELDS]
        accepted = row["violations"] == 0 and row["return_v3"] > -116_640
        print(f"| {knob} | {description} | 1 | " + " | ".join(cells) + " | "
              + ("accepted" if accepted else "rejected") + " |")
        continue

    rows = collect(prefix, aliases)
    if not rows:
        print(f"| {knob} | {description} | 0 | (running) | | | | | | | | |")
        continue
    values = {key: np.array([row[key] for _, _, row in rows], dtype=float)
              for key, _, _ in FIELDS}
    seeds = sorted({seed for _, seed, _ in rows})
    cells = []
    for key, _, fmt in FIELDS:
        array = values[key]
        if array.size > 1:
            cells.append(f"{fmt.format(array.mean())} ± {fmt.format(array.std(ddof=0))}")
        else:
            cells.append(fmt.format(array.mean()))
    accepted = all(row["violations"] == 0 for _, _, row in rows) and \
        all(row["return_v3"] > -116_640 for _, _, row in rows)
    label = f"{description} (n={len(rows)})"
    print(f"| {knob} | {label} | {','.join(str(s) for s in seeds)} | "
          + " | ".join(cells) + " | " + ("accepted" if accepted else "rejected") + " |")

print()
print("Per-seed detail (ret v3 / violations / pods):")
for path in sorted(RUNS.glob("*.json")):
    row = load(path.stem)
    if row is None:
        continue
    print(f"  {path.stem:<20} {row['return_v3']:>10,.0f}  viol {row['violations']:>3d}  "
          f"pods {row['avg_pods']:5.2f}  churn {row['churn_v3']:6.0f}  "
          f"max {row['max_latency_ms']:6.0f} ms  acts {row['actions']:>4d}")
