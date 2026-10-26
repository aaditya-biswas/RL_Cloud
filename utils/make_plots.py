"""The ten figures for the 16/6/6 split experiment.

Every figure reads from the same two files the report does - ``results/scores.csv``
(one row per run and day) and ``results/inventory.csv`` - plus, for the two
trajectory figures, a replay of the checkpoint itself. Nothing here recomputes a
metric differently from ``utils/score_policy.py``: if a number appears in a plot
it came out of the same ``summarise`` call that produced the table.

Figures
-------
01_split_overview      the 28 days, with the split boundaries drawn
02_baseline_ladder     hold-9 vs the two trackers, per split
03_violations_by_day   which test days breach, for each policy
04_training_curves     train/eval reward against timesteps, per seed
05_seed_spread         per-seed test return, against the baseline lines
06_test_day_trajectory pods and latency on the worst test day
07_state_action_map    where the policy acts, in (demand, headroom) space
08_ablation_time       the same checkpoints with the time feature zeroed
09_score_heatmap       run x test day return, the whole matrix at a glance
10_cost_safety_frontier cost against breaches, and the cost split by term

Usage
-----
    python utils/make_plots.py
    python utils/make_plots.py --only 01 03      # iterate on two figures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/make_plots.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cloud_env import DEFAULT_SLA_LATENCY_MS, CloudAutoscalingEnv  # noqa: E402
from env.design_wrappers import OBS_DIM, HeadroomObs  # noqa: E402
from utils.design_probe import util_policy  # noqa: E402
from utils.score_policy import ACTION_DELTAS, load_policy, summarise  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
PLOTS = PROJECT_ROOT / "plots"
HOLD_RETURN = -116_640.0
SPLITS = ("train", "val", "test")
COLORS = {"hold9": "#444444", "b1-u0.77": "#1f77b4", "b1-u0.70": "#17becf"}
DPI = 150


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })


def load_scores(path: Path = RESULTS / "scores.csv") -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"no scores at {path} (run utils/run_scores.py first)")
    frame = pd.read_csv(path)
    frame["key"] = frame["run"].astype(str)
    return frame


def rl_runs(scores: pd.DataFrame, ablate: str = "none") -> list[str]:
    """Run labels for learned policies (everything that is not a baseline)."""
    runs = [r for r in sorted(scores["run"].unique())
            if re.match(r"^r\d", str(r)) and str(r).startswith(("r1", "r2", "r3"))]
    keep = []
    for run in runs:
        row = scores[scores["run"] == run].iloc[0]
        if str(row["ablate"]) == ablate:
            keep.append(run)
    return keep


def best_run(scores: pd.DataFrame, split: str = "test") -> str | None:
    """The learned run with the highest mean return v3 on ``split``."""
    runs = rl_runs(scores)
    rows = scores[scores["run"].isin(runs) & (scores["split"] == split)]
    if rows.empty:
        return None
    means = rows.groupby("run")["return_v3"].mean().sort_values(ascending=False)
    return str(means.index[0])


def save(fig, name: str, only: list[str]) -> bool:
    """Save ``name`` unless a --only filter excludes its numeric prefix."""
    prefix = name.split("_", 1)[0]
    if only and prefix not in only:
        plt.close(fig)
        return False
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    return True


def tb_series(run: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one scalar series out of the run's TensorBoard events.

    Returns ``(steps, values)``, empty arrays when the series is absent - the
    figure then simply omits that curve instead of failing.
    """
    from tensorboard.backend.event_processing import event_accumulator

    candidates = sorted((PROJECT_ROOT / "logs" / "tb").glob(f"{run}_*"))
    for directory in candidates:
        accumulator = event_accumulator.EventAccumulator(str(directory),
                                                         size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags()["scalars"]:
            events = accumulator.Scalars(tag)
            return (np.array([e.step for e in events], dtype=float),
                    np.array([e.value for e in events], dtype=float))
    return np.array([]), np.array([])


def figure_split_overview(inventory: pd.DataFrame, only: list[str]) -> None:
    """Every day of the log, coloured by split, with the day's own volume."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    colours = {"train": "#9e9e9e", "val": "#1f77b4", "test": "#d62728"}

    for _, row in inventory.iterrows():
        values = pd.read_csv(PROJECT_ROOT / row["path"])["request_rate_rps"].to_numpy()
        # 360 bins of 10 s = one hour, so the 28 curves are comparable by shape.
        hourly = values[: len(values) // 360 * 360].reshape(-1, 360).mean(axis=1)
        hours = np.arange(hourly.size)
        axes[0].plot(hours, hourly, color=colours[row["split"]], lw=0.8, alpha=0.55)
    for split, colour in colours.items():
        axes[0].plot([], [], color=colour, lw=1.5, label=f"{split} "
                     f"({int((inventory['split'] == split).sum())} days)")
    axes[0].axhline(9 * 60, color="#444444", ls="--", lw=1.0, label="9 pods = 540 RPS")
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("request rate (RPS)")
    axes[0].set_title("28 real NASA days, each rescaled to a 480 RPS peak")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].bar(inventory["day"], inventory["mean_rps"],
                color=[colours[s] for s in inventory["split"]], width=0.8)
    axes[1].axvline(16.5, color="#1f77b4", ls=":", lw=1.0)
    axes[1].axvline(22.5, color="#d62728", ls=":", lw=1.0)
    axes[1].set_xlabel("day of July 1995")
    axes[1].set_ylabel("mean RPS")
    axes[1].set_title("daily volume; dotted lines are split boundaries")
    save(fig, "01_split_overview", only)


def figure_baseline_ladder(baselines: pd.DataFrame, only: list[str]) -> None:
    """hold-9 against the two trackers, per split, drawn as cost (positive = worse)."""
    keys = [k for k in ("hold9", "b1-u0.77", "b1-u0.70")
            if k in set(baselines["key"])]
    width = 0.8 / max(1, len(keys))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for index, key in enumerate(keys):
        subset = baselines[baselines["key"] == key].set_index("split")
        # Costs are negative returns; flipping the sign makes "taller = more
        # expensive" true, which is the direction a reader expects from a cost chart.
        heights = [-float(subset.loc[s, "return_v3"]) / 1000.0 if s in subset.index
                   else np.nan for s in SPLITS]
        positions = np.arange(len(SPLITS)) + (index - (len(keys) - 1) / 2) * width
        bars = ax.bar(positions, heights, width=width, label=key,
                      color=COLORS.get(key))
        for bar, height, split in zip(bars, heights, SPLITS):
            violations = int(subset.loc[split, "violations"]) if split in subset.index else 0
            ax.annotate(f"{height:,.0f}k\n{violations} viol",
                        (bar.get_x() + bar.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=7)
    ax.axhline(-HOLD_RETURN / 1000.0, color="#444444", ls="--", lw=1.0,
               label="constant-9 line")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_ylabel("cost (thousands of reward-v3 units)")
    ax.set_title("Baselines per split: shorter is cheaper, zero violations is the bar")
    ax.set_ylim(0, 135)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    save(fig, "02_baseline_ladder", only)


def figure_violations_by_day(scores: pd.DataFrame, only: list[str]) -> None:
    """Where the breaches are, day by day, on the test split."""
    test = scores[scores["split"] == "test"]
    days = sorted(test["day"].unique())
    runs = [k for k in ("b1-u0.77", "b1-u0.70") if k in set(test["run"])]
    runs += [r for r in rl_runs(scores) if r in set(test["run"])]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / max(1, len(runs))
    for index, run in enumerate(runs):
        subset = test[test["run"] == run].set_index("day")
        heights = [int(subset.loc[day, "violations"]) if day in subset.index else 0
                   for day in days]
        positions = np.arange(len(days)) + (index - (len(runs) - 1) / 2) * width
        colour = COLORS.get(run) or plt.cm.viridis(index / max(1, len(runs) - 1))
        ax.bar(positions, heights, width=width, label=run, color=colour)
    ax.set_xticks(np.arange(len(days)),
                  [inventory_label(d) for d in days])
    ax.set_ylabel("SLA violations (steps over 200 ms)")
    ax.set_title("Test days only: a single breached step fails the acceptance rule")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "03_violations_by_day", only)


def inventory_label(day: int) -> str:
    """``23`` -> ``day 23``; kept in one place so axes stay consistent."""
    return f"day {int(day)}"


def figure_training_curves(scores: pd.DataFrame, only: list[str]) -> None:
    """Train and evaluation reward against timesteps, one line per seed."""
    runs = rl_runs(scores)
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = {"R1": "-", "R3": "--", "R2": ":"}

    for run in runs:
        rung = str(run).split("-")[0].upper()
        colour = plt.cm.viridis(runs.index(run) / max(1, len(runs) - 1))
        for axis, tag, scale in ((axes[0], "rollout/ep_rew_mean", 1000.0),
                                 (axes[1], "eval/mean_reward", 1000.0)):
            steps, values = tb_series(run, tag)
            if not steps.size:
                continue
            smooth = pd.Series(values).rolling(5, min_periods=1).mean().to_numpy()
            axis.plot(steps, smooth / scale, styles.get(rung, "-"), color=colour,
                      lw=1.2, label=run)
    for axis, title in zip(axes, ("training episodes (5-point mean)",
                                  "validation day 17 (full 24 h)")):
        axis.axhline(HOLD_RETURN / 1000.0, color="#444444", ls="--", lw=1.0)
        axis.set_xlabel("timesteps")
        axis.set_ylabel("episode return (thousands)")
        axis.set_title(title)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Learning curves: solid = R1 (reward only), dashed = R3 (Lagrangian)")
    save(fig, "04_training_curves", only)


def figure_seed_spread(scores: pd.DataFrame, baselines: pd.DataFrame,
                       only: list[str]) -> None:
    """Per-seed test return, so the spread is visible next to the mean."""
    runs = rl_runs(scores)
    test = scores[scores["run"].isin(runs) & (scores["split"] == "test")]
    if test.empty:
        return
    means = test.groupby("run")["return_v3"].mean()
    rungs = sorted({str(r).split("-")[0].upper() for r in runs})
    rung_colours = {"R1": "#6a3d9a", "R3": "#1b9e77", "R2": "#e31a1c"}
    markers = ("o", "s", "^")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for index, rung in enumerate(rungs):
        labels = [r for r in means.index if str(r).startswith(rung.lower())]
        for offset, run in enumerate(labels):
            rows = test[test["run"] == run]
            values = rows["return_v3"] / 1000.0
            x = index + (offset - (len(labels) - 1) / 2) * 0.18
            ax.scatter(np.full(values.size, x), values, s=30, alpha=0.9,
                       marker=markers[offset % len(markers)],
                       color=rung_colours.get(rung, "#333333"),
                       label=f"{rung} seed {int(rows['seed'].iloc[0])}")
            ax.hlines(values.mean(), x - 0.07, x + 0.07, color="black", lw=1.2)
    for key in ("b1-u0.70", "b1-u0.77"):
        subset = baselines[(baselines["key"] == key) & (baselines["split"] == "test")]
        if not subset.empty:
            ax.axhline(float(subset["return_v3"].iloc[0]) / 1000.0,
                       color=COLORS.get(key, "#1f77b4"), ls="--", lw=1.0, label=key)
    ax.axhline(HOLD_RETURN / 1000.0, color="#444444", ls=":", lw=1.0, label="hold-9")
    ax.set_ylim(-100, -45)
    ax.set_xticks(np.arange(len(rungs)), rungs)
    ax.set_ylabel("mean test return v3 (thousands)")
    ax.set_title("Every seed is a point; black tick = seed mean, dashed lines = baselines")
    ax.text(0.01, 0.02, f"hold-9 = {HOLD_RETURN / 1000.0:,.0f}k (below the axis)",
            transform=ax.transAxes, fontsize=7, color="#444444")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "05_seed_spread", only)


def replay_model(scores: pd.DataFrame, run: str, trace_path: Path,
                 ablate: str = "none") -> dict:
    """Replay a checkpoint and keep the observations, actions and outcomes."""
    row = scores[scores["run"] == run].iloc[0]
    stats = row["vecnormalize"]
    stats = str(stats) if isinstance(stats, str) and stats and stats != "nan" else None
    # One observation at a time: torch's default 12-thread pool is pure overhead
    # here (and starves the training jobs running next to it).
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # torch is a dependency of the scorer, but never assume
        pass
    decide, _, obs_dim, _ = load_policy(Path(row["model"]), stats_path=stats,
                                       ablate=ablate, trace_path=trace_path)
    env = CloudAutoscalingEnv(trace_path=trace_path, episode_steps=None,
                              random_start=False, initial_pods="auto", seed=0)
    if obs_dim == OBS_DIM:
        env = HeadroomObs(env)
    return _walk(env, decide)


def replay_targets(trace_path: Path, targets: np.ndarray) -> dict:
    """Replay a desired-pod trajectory (the baseline actuators).

    The actions are built up front from the same ``+-2 pods/step`` envelope and
    the same ``[1, 15]`` clip the environment applies, so the replay follows the
    target exactly as ``utils/score_policy.rollout`` does.
    """
    actions: list[int] = []
    pods = int(np.clip(int(targets[0]), 1, 15))
    for target in targets:
        delta = int(np.clip(int(target) - pods, -2, 2))
        actions.append(ACTION_DELTAS.index(delta))
        pods = int(np.clip(pods + delta, 1, 15))

    env = CloudAutoscalingEnv(trace_path=trace_path, episode_steps=None,
                              random_start=False, initial_pods=int(targets[0]), seed=0)
    queue = list(actions)
    return _walk(env, lambda _observation: queue.pop(0))


def _walk(env, decide) -> dict:
    """Step ``env`` until the episode ends, recording what the customer saw."""
    obs, _ = env.reset(seed=0)
    record = {"obs": [], "actions": [], "pods": [], "rps": [], "latency": []}
    done = False
    while not done:
        action = int(decide(obs))
        obs, _, terminated, truncated, info = env.step(action)
        record["obs"].append(np.asarray(obs, dtype=float))
        record["actions"].append(action)
        record["pods"].append(int(info["active_pods"]))
        record["rps"].append(float(info["request_rate_rps"]))
        record["latency"].append(float(info["latency_ms"]))
        done = bool(terminated or truncated)
    return {key: np.asarray(values) for key, values in record.items()}


def figure_trajectory(scores: pd.DataFrame, only: list[str]) -> None:
    """One test day, three policies: what the pods did and what it cost."""
    run = best_run(scores)
    if run is None:
        return
    test = scores[(scores["run"] == run) & (scores["split"] == "test")]
    day = int(test.sort_values("return_v3").iloc[0]["day"])
    trace_path = PROJECT_ROOT / "data" / "days" / f"day_{day:02d}.csv"

    best = replay_model(scores, run, trace_path)
    steps = best["rps"].size
    tracker = util_policy(best["rps"], 0.77, "trailing", 48)
    targets = np.asarray([tracker(i, float(best["rps"][i]), 0) for i in range(steps)])
    tracked = replay_targets(trace_path, targets)
    held = replay_targets(trace_path, np.full(steps, 9))

    hours = np.arange(steps) * 10.0 / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    axes[0].fill_between(hours, best["rps"] / 60.0, color="#cccccc", alpha=0.8,
                         label="demand / pod capacity (RPS / 60)", step="post")
    axes[0].step(hours, held["pods"], color=COLORS["hold9"], lw=1.2, label="hold 9 pods")
    axes[0].step(hours, tracked["pods"], color=COLORS["b1-u0.77"], lw=1.2,
                 label="tracker u=0.77")
    axes[0].step(hours, best["pods"], color="#d62728", lw=1.6, label=f"{run} (best test)")
    axes[0].set_ylabel("pods")
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)

    # A trailing run of identical values is a logging outage, not calm traffic:
    # the generator forward-fills the gap. Say so on the figure instead of
    # letting it look like a quiet evening.
    values = pd.read_csv(trace_path)["request_rate_rps"].to_numpy()
    start = len(values) - 1
    while start > 0 and values[start] == values[start - 1]:
        start -= 1
    if len(values) - 1 - start > 360:  # more than an hour of flat tail
        flat_hours = start * 10.0 / 3600.0
        axes[0].axvspan(flat_hours, hours[-1], color="#f2f2f2", zorder=0)
        axes[0].annotate("log outage: value forward-filled",
                         (flat_hours + 0.2, 0.94), xycoords=("data", "axes fraction"),
                         fontsize=8, color="#666666")
        axes[1].axvspan(flat_hours, hours[-1], color="#f2f2f2", zorder=0)
    axes[0].set_title(f"Day {day}: the worst test day for {run}")

    axes[1].plot(hours, held["latency"], color=COLORS["hold9"], lw=1.0, label="hold 9")
    axes[1].plot(hours, tracked["latency"], color=COLORS["b1-u0.77"], lw=1.0, label="tracker")
    axes[1].plot(hours, best["latency"], color="#d62728", lw=1.2, label=run)
    axes[1].set_yscale("log")
    axes[1].axhline(DEFAULT_SLA_LATENCY_MS, color="black", ls="--", lw=1.0,
                    label=f"SLA {DEFAULT_SLA_LATENCY_MS:.0f} ms")
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("latency (ms, log)")
    axes[1].legend(fontsize=8, ncol=4, loc="upper left")
    save(fig, "06_test_day_trajectory", only)


def figure_state_action_map(scores: pd.DataFrame, only: list[str]) -> None:
    """Where the policy acts: a map from (required pods, current pods) to action."""
    run = best_run(scores)
    if run is None:
        return
    test = sorted(scores[(scores["run"] == run) & (scores["split"] == "test")]["day"])
    frames = []
    for day in test[:3]:
        trace_path = PROJECT_ROOT / "data" / "days" / f"day_{day:02d}.csv"
        record = replay_model(scores, run, trace_path)
        obs = record["obs"]
        # Index 8 is peak480: the 480 s trailing peak, normalised by the 480 RPS
        # scale, so ``peak480 * 8`` is the pod count that peak needs.
        frames.append(pd.DataFrame({
            "required_pods": obs[:, 8] * 8.0,
            "pods": record["pods"],
            "action": record["actions"],
        }))
    data = pd.concat(frames, ignore_index=True)

    x_edges = np.arange(0.0, 9.01, 0.2)
    y_edges = np.arange(0.5, 15.51, 0.5)
    counts, _, _ = np.histogram2d(data["required_pods"], data["pods"],
                                  bins=(x_edges, y_edges))
    totals = np.zeros_like(counts)
    for action in range(len(ACTION_DELTAS)):
        subset = data[data["action"] == action]
        values, _, _ = np.histogram2d(subset["required_pods"], subset["pods"],
                                      bins=(x_edges, y_edges))
        totals += values * (action - 2)  # action index -> pod delta
    mean_delta = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    mesh = ax.pcolormesh(x_edges, y_edges, mean_delta.T, cmap="coolwarm",
                         vmin=-2, vmax=2, shading="auto")
    ax.plot(x_edges, x_edges, color="black", lw=1.0, ls=":",
            label="pods = required (no slack)")
    ax.plot(x_edges, np.ceil(x_edges / 0.77), color="#17becf", lw=1.0,
            label="tracker: ceil(required / 0.77)")
    fig.colorbar(mesh, ax=ax, label="mean pod delta (-2 .. +2)")
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 15)
    ax.set_xlabel("pods the 480 s peak needs (peak480 x 8)")
    ax.set_ylabel("pods running")
    ax.set_title(f"State -> action map of {run} on the first 3 test days "
                 f"({len(data):,} decisions)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "07_state_action_map", only)


def figure_ablation(scores: pd.DataFrame, only: list[str]) -> None:
    """Same checkpoints, time feature zeroed: does the score survive?"""
    ablated = scores[scores["ablate"] == "time"]
    if ablated.empty:
        print("  skip 08_ablation_time (no --ablate time rows yet)")
        return

    def keyed(frame: pd.DataFrame) -> dict:
        """(rung, seed) -> its test rows, so a pair means the same checkpoint."""
        out = {}
        for run, group in frame.groupby("run"):
            out[(str(run).split("-")[0].upper(), int(group["seed"].iloc[0]))] = group
        return out

    full = scores[(scores["ablate"] == "none") & scores["run"].astype(str).str.match(r"^r\d")]
    pairs = sorted(set(keyed(full)) & set(keyed(ablated)))
    if not pairs:
        print("  skip 08_ablation_time (no full/ablated pair scored on the same seed)")
        return
    full_map, abl_map = keyed(full), keyed(ablated)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.36
    for index, key in enumerate(pairs):
        base, abl = full_map[key], abl_map[key]
        costs = [-float(base["return_v3"].mean()) / 1000.0,
                 -float(abl["return_v3"].mean()) / 1000.0]
        violations = [int(base["violations"].sum()), int(abl["violations"].sum())]
        for axis, series in ((axes[0], costs), (axes[1], violations)):
            bars = axis.bar([index - width / 2, index + width / 2], series, width=width,
                            color=["#d62728", "#ff9896"])
            for bar, value in zip(bars, series):
                text = f"{value:,.1f}" if axis is axes[0] else f"{int(value)}"
                axis.annotate(text, (bar.get_x() + bar.get_width() / 2, value),
                              ha="center", va="bottom", fontsize=7)
    labels = [f"{rung} s{seed}" for rung, seed in pairs]
    for axis in axes:
        axis.set_xticks(np.arange(len(pairs)), labels)
    axes[0].axhline(-HOLD_RETURN / 1000.0, color="#444444", ls="--", lw=1.0,
                    label="constant-9 line")
    axes[0].set_ylabel("mean test cost (thousands of v3 units)")
    axes[0].set_title("dark = full observation, light = fraction_of_day zeroed")
    axes[0].set_ylim(0, 135)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_ylabel("test violations")
    axes[1].set_title("breaches with and without the time index")
    fig.suptitle("Time-feature ablation: does the policy need the clock?")
    save(fig, "08_ablation_time", only)


def figure_score_heatmap(scores: pd.DataFrame, only: list[str]) -> None:
    """The whole run x test-day matrix, so no single day hides in an average."""
    test = scores[scores["split"] == "test"]
    rows = [r for r in ("hold9", "b1-u0.77", "b1-u0.70") if r in set(test["run"])]
    rows += rl_runs(scores)
    rows = [r for r in rows if r in set(test["run"])]
    if not rows:
        return
    days = sorted(test["day"].unique())
    matrix = np.full((len(rows), len(days)), np.nan)
    for i, run in enumerate(rows):
        subset = test[test["run"] == run].set_index("day")
        for j, day in enumerate(days):
            if day in subset.index:
                matrix[i, j] = float(subset.loc[day, "return_v3"]) / 1000.0

    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(rows) + 2.4))
    mesh = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-120, vmax=-55)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                continue
            value = matrix[i, j]
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_xticks(np.arange(len(days)), [inventory_label(d) for d in days])
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xlabel("test day")
    ax.set_title("Test return v3 by run and day (thousands; greener = cheaper)")
    fig.colorbar(mesh, ax=ax, label="return v3 (thousands)")
    save(fig, "09_score_heatmap", only)


def figure_frontier(scores: pd.DataFrame, only: list[str]) -> None:
    """The trade-off the whole experiment is about: cost against breaches.

    Left panel: the two scripted families (constant fleet, tracker sweep) as lines
    and every learned seed as a point, so the reader can see where the rungs sit
    relative to what a *clean* scripted policy costs. Right panel: the same rows
    with their cost split into the three terms v3 charges, which is why the
    cheapest arms are the ones running the fewest pods.

    The scripted rows come from ``utils.frontier`` (the module behind
    ``results/frontier.txt``), so the figure and the table cannot drift apart.
    """
    from utils.frontier import summarise_days

    days = list(range(23, 29))  # the frozen test split, see results/split.json
    holds = [(pods, summarise_days(days, pods=pods)) for pods in range(5, 11)]
    sweep = [(u, summarise_days(days, u=u)) for u in
             (0.55, 0.58, 0.62, 0.65, 0.70, 0.74, 0.77)]

    test = scores[scores["split"] == "test"]
    runs = [r for r in rl_runs(scores) if r in set(test["run"])]
    per_seed = test[test["run"].isin(runs)].groupby("run").agg(
        return_v3=("return_v3", "mean"), violations=("violations", "sum"),
        avg_pods=("avg_pods", "mean"))
    rung_of = {run: str(run).split("-")[0].upper() for run in per_seed.index}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    # --- left: cost against violations -------------------------------------
    ax.plot([r["return_v3"] / 1000 for _, r in holds],
            [r["violations"] for _, r in holds], "-o", color="#8c8c8c", ms=4,
            lw=1.2, label="constant fleet (5-10 pods)")
    for pods, row in holds:
        ax.annotate(f"{pods}", (row["return_v3"] / 1000, row["violations"]),
                    textcoords="offset points", xytext=(0, -12), ha="center",
                    fontsize=7, color="#555555")
    ax.plot([r["return_v3"] / 1000 for _, r in sweep],
            [r["violations"] for _, r in sweep], "-s", color="#1f77b4", ms=4,
            lw=1.2, label="tracker, u = 0.55 ... 0.77")
    for index, (u, row) in enumerate(sweep):
        # Alternate the label row: 0.70 and 0.74 are ~4k apart and would collide.
        ax.annotate(f"{u:.2f}", (row["return_v3"] / 1000, row["violations"]),
                    textcoords="offset points",
                    xytext=(0, -11 if index % 2 == 0 else -21), ha="center", va="top",
                    fontsize=6.5, color="#1f77b4")
    for rung, colour, marker in (("R1", "#6a3d9a", "o"), ("R3", "#1b9e77", "D")):
        rows = per_seed[[rung_of[r] == rung for r in per_seed.index]]
        if rows.empty:
            continue
        ax.scatter(rows["return_v3"] / 1000, rows["violations"], s=70, marker=marker,
                   color=colour, edgecolor="white", linewidth=0.6, zorder=5,
                   label=f"{rung} (one point per seed)")
        for run, row in rows.iterrows():
            ax.annotate(str(run).split("-")[-1], (row["return_v3"] / 1000, row["violations"]),
                        textcoords="offset points", xytext=(6, -3), fontsize=6.5,
                        color=colour)
    ax.axvline(HOLD_RETURN / 1000, color="#444444", ls=":", lw=1.0)
    ax.axhline(0, color="#2ca02c", ls="--", lw=1.0)
    ax.fill_between([HOLD_RETURN / 1000, -56], -0.9, 0.0, color="#2ca02c", alpha=0.12,
                    zorder=0)

    # The one comparison the experiment exists to make: same zero, 26 % cheaper.
    clean = next(row for u, row in sweep if u == 0.62)
    ax.annotate("", xy=(clean["return_v3"] / 1000, 0.0), xytext=(HOLD_RETURN / 1000, 0.0),
                arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.1,
                            "alpha": 0.8, "shrinkA": 2, "shrinkB": 2}, zorder=1)
    ax.annotate(f"both clean: the tracker does it on {clean['avg_pods']:.2f} pods,\n"
                f"{1 - clean['return_v3'] / HOLD_RETURN:.0%} cheaper than 9",
                xy=((clean["return_v3"] / 1000 + HOLD_RETURN / 1000) / 2, 0.0),
                xytext=(0, 26), textcoords="offset points", ha="center", va="bottom",
                fontsize=6.8, color="#2ca02c",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
    offscale = " / ".join(f"{summarise_days(days, pods=pods)['return_v3'] / 1000:,.0f}k"
                          for pods in (3, 4))
    ax.text(-151.5, 3.0, "the rule's corner is the green\nstrip: no breach and no dearer\n"
            "than hold-9. 3-4 pods are off-\n"
            f"scale to the left ({offscale}).",
            fontsize=6.5, color="#2ca02c", va="bottom")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.9, 3000)
    ax.set_xlim(-152, -56)
    ax.set_xlabel("mean test return v3 (thousands) - to the right is cheaper")
    ax.set_ylabel("SLA violations over 6 test days (steps over 200 ms)")
    ax.set_title("Cost against breaches: scripted frontier and every learned seed")
    ax.legend(fontsize=7, loc="upper center", ncol=2)

    # --- right: where the money goes ---------------------------------------
    ordered = ["hold9", "b1-u0.70", "b1-u0.77"] + runs
    labels, pods_cost, churn_cost, penalty = [], [], [], []
    for run in ordered:
        rows = test[test["run"] == run]
        if rows.empty:
            continue
        penalty.append(float((-rows["return_v3"] - rows["cost_units"]
                              - rows["churn_v3"]).mean()))
        pods_cost.append(float(rows["cost_units"].mean()))
        churn_cost.append(float(rows["churn_v3"].mean()))
        labels.append(run)
    y = np.arange(len(labels))
    scale = 1 / 1000.0
    ax2.barh(y, np.array(pods_cost) * scale, color="#4c72b0", label="pods (1.5 / pod-step)")
    ax2.barh(y, np.array(churn_cost) * scale, left=np.array(pods_cost) * scale,
             color="#dd8452", label="churn (4 / requested pod)")
    ax2.barh(y, np.array(penalty) * scale,
             left=(np.array(pods_cost) + np.array(churn_cost)) * scale,
             color="#c44e52", label="SLA penalty (capped at 1,000 / step)")
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean cost per test day (thousands of reward units)")
    fleet_share = min(pod / (pod + churn + pen) for pod, churn, pen
                      in zip(pods_cost, churn_cost, penalty))
    ax2.set_title(f"Where the cost goes: the fleet bill is {fleet_share:.0%}-100 % "
                  "of every row")
    ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for index, (pod, churn, pen) in enumerate(zip(pods_cost, churn_cost, penalty)):
        ax2.text((pod + churn + pen) * scale, index,
                 f"  {(churn + pen) / (pod + churn + pen):.1%} non-fleet",
                 va="center", fontsize=6.5, color="#333333")

    fig.suptitle("The test set in one picture: the learned arms buy cost with margin, "
                 "and margin is what the gate is about")
    save(fig, "10_cost_safety_frontier", only)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the ten split figures.")
    parser.add_argument("--scores", type=Path, default=RESULTS / "scores.csv")
    parser.add_argument("--inventory", type=Path, default=RESULTS / "inventory.csv")
    parser.add_argument("--baselines", type=Path, default=RESULTS / "baselines.csv")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Figure prefixes to draw, e.g. --only 01 03.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _style()
    only = [str(prefix) for prefix in (args.only or [])]
    scores = load_scores(Path(args.scores).expanduser())
    inventory_path = Path(args.inventory).expanduser()
    if not inventory_path.exists():
        raise SystemExit(f"no inventory at {inventory_path}")
    inventory = pd.read_csv(inventory_path)

    baselines_path = Path(args.baselines).expanduser()
    if baselines_path.exists():
        baselines = pd.read_csv(baselines_path)
    else:
        grouped = scores.groupby(["run", "split"], as_index=False).agg(
            days=("day", "size"), violations=("violations", "sum"),
            return_v3=("return_v3", "mean"), worst_return_v3=("return_v3", "min"),
            avg_pods=("avg_pods", "mean"), max_latency_ms=("max_latency_ms", "max"))
        baselines = grouped.rename(columns={"run": "key"})

    print(f"== figures -> {(PLOTS).relative_to(PROJECT_ROOT)} ==")
    # Filter before drawing, not after: figure 06/07 replay a checkpoint and cost
    # minutes under load, which --only exists to avoid.
    jobs = (
        ("01_split_overview", figure_split_overview, (inventory, only)),
        ("02_baseline_ladder", figure_baseline_ladder, (baselines, only)),
        ("03_violations_by_day", figure_violations_by_day, (scores, only)),
        ("04_training_curves", figure_training_curves, (scores, only)),
        ("05_seed_spread", figure_seed_spread, (scores, baselines, only)),
        ("06_test_day_trajectory", figure_trajectory, (scores, only)),
        ("07_state_action_map", figure_state_action_map, (scores, only)),
        ("08_ablation_time", figure_ablation, (scores, only)),
        ("09_score_heatmap", figure_score_heatmap, (scores, only)),
        ("10_cost_safety_frontier", figure_frontier, (scores, only)),
    )
    for name, function, call_args in jobs:
        if only and name.split("_", 1)[0] not in only:
            continue
        function(*call_args)
    written = sorted(p.name for p in PLOTS.glob("*.png"))
    print(f"  {len(written)} file(s) in {PLOTS.relative_to(PROJECT_ROOT)}: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
