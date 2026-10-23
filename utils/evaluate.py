"""Benchmark the rule-based HPA baseline against a trained RL agent.

Runs one deterministic pass over the full 24 h workload trace for every
available policy and writes:

* ``utils/latency_comparison.png``          - latency vs the 200 ms SLA
* ``utils/pod_provisioning_comparison.png`` - pods vs incoming traffic
* ``utils/metrics.csv``                     - summary metrics table

Usage
-----
    python utils/evaluate.py
    python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo
    python utils/evaluate.py --constant-pods 9
    python utils/evaluate.py --no-baseline --constant-pods 6,9,12,15
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/evaluate.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baseline_scaler import RuleBasedScaler  # noqa: E402
from env.cloud_env import DEFAULT_TRACE_PATH, CloudAutoscalingEnv  # noqa: E402

DEFAULT_POD_COST_PER_HOUR = 0.12  # USD, small general-purpose pod
POLICY_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2")


def hold_policy(_observation: np.ndarray) -> int:
    """Fixed-fleet baseline policy: never change the pod count (action 2 = delta 0)."""
    return 2


def rollout(
    env: CloudAutoscalingEnv,
    policy: Callable[[np.ndarray], int],
    seed: int = 42,
    start_index: int = 0,
    episode_steps: Optional[int] = None,
) -> pd.DataFrame:
    """Run ``policy`` over the trace and collect one row per environment step."""
    options = {"start_index": start_index}
    if episode_steps is not None:
        options["episode_steps"] = episode_steps
    observation, _ = env.reset(seed=seed, options=options)

    rows = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "index": info["index"],
                "request_rate_rps": info["request_rate_rps"],
                "active_pods": info["active_pods"],
                "delta_pods": info["delta_pods"],
                "cpu_utilization": info["cpu_utilization"],
                "ram_utilization": info["ram_utilization"],
                "latency_ms": info["latency_ms"],
                "reward": reward,
                "sla_violation": info["sla_violation"],
            }
        )

    frame = pd.DataFrame(rows)
    first_timestamp = pd.read_csv(env.trace_path, usecols=["timestamp"], nrows=1)["timestamp"].iloc[0]
    frame["timestamp"] = pd.date_range(
        pd.Timestamp(first_timestamp), periods=len(frame), freq=f"{env.step_seconds}s"
    )
    return frame


def summarise(
    frame: pd.DataFrame,
    label: str,
    pod_cost_per_hour: float,
    step_seconds: int,
    sla_ms: float,
) -> Dict[str, object]:
    """Compute the cost / SLA / latency metrics used in the comparison table."""
    hours_per_step = step_seconds / 3_600.0
    return {
        "Policy": label,
        "Total Cost ($)": round(float(frame["active_pods"].sum() * hours_per_step * pod_cost_per_hour), 2),
        "SLA Violation Rate (%)": round(float((frame["latency_ms"] > sla_ms).mean() * 100.0), 2),
        "Average Latency (ms)": round(float(frame["latency_ms"].mean()), 2),
        "P95 Latency (ms)": round(float(frame["latency_ms"].quantile(0.95)), 2),
        "Avg Pods": round(float(frame["active_pods"].mean()), 2),
        "Scale Actions": int((frame["delta_pods"] != 0).sum()),
        "Total Reward": round(float(frame["reward"].sum()), 1),
    }


def resolve_vecnormalize_path(model_path: Path, vecnormalize_path: Optional[str]) -> Optional[Path]:
    """Find the VecNormalize statistics that belong to a saved model.

    The trainer writes ``<model_stem>_vecnormalize.pkl`` next to the model
    (e.g. ``ppo_cloud_autoscaling_vecnormalize.pkl``) so that a PPO and a DQN run
    can coexist in ``models/``. ``vecnormalize.pkl`` is still accepted for older
    checkpoints.
    """
    if vecnormalize_path:
        path = Path(vecnormalize_path).expanduser()
        return path if path.exists() else None
    parent = Path(model_path).expanduser().parent
    for name in (f"{Path(model_path).stem}_vecnormalize.pkl", "vecnormalize.pkl"):
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def load_rl_policy(
    model_path: Path,
    algo: str = "auto",
    vecnormalize_path: Optional[str] = None,
) -> tuple[Callable[[np.ndarray], int], str, str, Optional[object]]:
    """Load a Stable-Baselines3 model and wrap it into a plain ``policy(obs)``."""
    from stable_baselines3 import DQN, PPO

    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")

    candidates = {"ppo": [(PPO, "ppo")], "dqn": [(DQN, "dqn")]}
    candidates["auto"] = [(PPO, "ppo"), (DQN, "dqn")]
    errors = []
    model = None
    resolved = algo
    for model_class, name in candidates[algo]:
        try:
            model = model_class.load(path, device="cpu")
            resolved = name
            break
        except Exception as exc:  # wrong algorithm for this archive
            errors.append(f"{name}: {exc}")

    if model is None:
        raise RuntimeError(
            f"Could not load {path} as one of {list(candidates[algo])}.\n" + "\n".join(errors)
        )

    normalizer = None
    stats_path = resolve_vecnormalize_path(path, vecnormalize_path)
    if stats_path is not None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        normalizer = VecNormalize.load(
            str(stats_path), DummyVecEnv([lambda: CloudAutoscalingEnv()])
        )
        normalizer.training = False
    else:
        print("[evaluate] warning: no VecNormalize statistics found; observations are "
              "passed to the model unnormalised.")

    def policy(observation: np.ndarray) -> int:
        model_observation = np.asarray(observation, dtype=np.float32)
        if normalizer is not None:
            model_observation = normalizer.normalize_obs(model_observation)
        action, _ = model.predict(model_observation, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    label = f"RL agent ({resolved.upper()})"
    return policy, label, resolved, normalizer


def _hours(frame: pd.DataFrame) -> pd.Series:
    return (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds() / 3_600.0


def plot_latency(results: Dict[str, pd.DataFrame], sla_ms: float, out_path: Path) -> None:
    """Latency over the day with the SLA threshold marked."""
    colors = itertools.cycle(POLICY_COLORS)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    for label, frame in results.items():
        ax.plot(_hours(frame), frame["latency_ms"], color=next(colors), lw=0.9,
                label=label, alpha=0.9)

    ax.axhline(sla_ms, color="black", linestyle="--", linewidth=1.2,
               label=f"SLA threshold ({sla_ms:.0f} ms)")
    ax.set_yscale("log")
    ax.set_yticks([15, 50, 100, 200, 500, 1000, 3000])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Request latency (ms, log scale)")
    ax.set_title("Request latency over a 24-hour workload replay")
    ax.set_xlim(0.0, max(float(_hours(frame).max()) for frame in results.values()))
    ax.grid(alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_provisioning(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Active pod count per policy, against the incoming request rate."""
    colors = itertools.cycle(POLICY_COLORS)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    # Traffic goes in first (and underneath) so the pod profiles stay readable.
    traffic = next(iter(results.values()))
    hours = _hours(traffic)
    traffic_axis = ax.twinx()
    traffic_axis.fill_between(
        hours, 0, traffic["request_rate_rps"], color="0.75", alpha=0.45,
        step="post", linewidth=0, zorder=0,
    )
    traffic_axis.plot(
        hours, traffic["request_rate_rps"].rolling(6, min_periods=1).mean(),
        color="0.35", lw=1.1, zorder=1, label="Incoming traffic (1-min mean, right axis)",
    )
    traffic_axis.set_ylabel("Incoming request rate (RPS)")
    traffic_axis.set_ylim(0, float(traffic["request_rate_rps"].max()) * 1.15)
    traffic_axis.set_zorder(ax.get_zorder() - 1)
    traffic_axis.patch.set_visible(False)

    for label, frame in results.items():
        ax.step(_hours(frame), frame["active_pods"], where="post", color=next(colors),
                lw=1.4, label=label, zorder=3)

    ax.set_zorder(ax.get_zorder() + 1)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Active pods")
    ax.set_ylim(0, 16)
    ax.set_yticks(range(0, 17, 2))
    ax.set_xlim(0.0, float(hours.max()))
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    more_handles, more_labels = traffic_axis.get_legend_handles_labels()
    fig.legend(handles + more_handles, labels + more_labels, loc="lower center",
               ncol=min(len(labels) + len(more_labels), 4), frameon=False)
    ax.set_title("Pod provisioning vs incoming traffic")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def _pod_list(value: str) -> list[int]:
    """Parse ``--constant-pods 6,9,12`` into ``[6, 9, 12]``."""
    try:
        pods = [int(part) for part in value.replace(",", " ").split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated pod counts, got {value!r}") from exc
    if not pods or min(pods) < 1:
        raise argparse.ArgumentTypeError("pod counts must be positive integers")
    return pods


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH,
                        help="Workload CSV (default: data/workload_trace.csv).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Trained SB3 model (.zip). Omit to benchmark the baseline only.")
    parser.add_argument("--algo", choices=["auto", "ppo", "dqn"], default="auto",
                        help="Algorithm used for the saved model (auto tries PPO, then DQN).")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="VecNormalize .pkl (default: looks for vecnormalize.pkl next to the model).")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "utils",
                        help="Directory for the benchmark charts.")
    parser.add_argument("--pod-cost-per-hour", type=float, default=DEFAULT_POD_COST_PER_HOUR,
                        help="USD per pod-hour used for the Total Cost column.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the (deterministic) evaluation rollout.")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the HPA baseline and evaluate the RL model only.")
    parser.add_argument("--constant-pods", type=_pod_list, default=[], metavar="LIST",
                        help="Comma-separated pod counts to run as fixed-fleet baselines, "
                             "e.g. 9 or 6,9,12,15 (default: none).")
    parser.add_argument("--hpa-scale-up", type=float, default=0.75,
                        help="CPU threshold for the HPA baseline scale-up action.")
    parser.add_argument("--hpa-scale-down", type=float, default=0.25,
                        help="CPU threshold for the HPA baseline scale-down action.")
    parser.add_argument("--hpa-cooldown", type=int, default=0,
                        help="Minimum steps between HPA baseline scaling decisions.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read the shared environment configuration (SLA + step size) once.
    probe = CloudAutoscalingEnv(trace_path=args.trace, episode_steps=None, random_start=False)
    sla_ms = probe.sla_latency_ms
    step_seconds = probe.step_seconds
    probe.close()

    results: Dict[str, pd.DataFrame] = {}
    table: list[Dict[str, object]] = []

    if not args.no_baseline:
        env = CloudAutoscalingEnv(trace_path=args.trace, episode_steps=None, random_start=False)
        scaler = RuleBasedScaler(
            scale_up_threshold=args.hpa_scale_up,
            scale_down_threshold=args.hpa_scale_down,
            cooldown_steps=args.hpa_cooldown,
        )
        frame = rollout(env, scaler.predict, seed=args.seed)
        env.close()
        results["HPA baseline"] = frame
        table.append(summarise(frame, "HPA baseline", args.pod_cost_per_hour, step_seconds, sla_ms))

    # Fixed-fleet baselines: "hold the pod count constant all day". Cheap to
    # interpret and hard to beat when the SLA penalty dominates the reward.
    for pods in args.constant_pods:
        env = CloudAutoscalingEnv(trace_path=args.trace, episode_steps=None,
                                  random_start=False, initial_pods=int(pods))
        frame = rollout(env, hold_policy, seed=args.seed)
        env.close()
        label = f"constant {pods} pods"
        results[label] = frame
        table.append(summarise(frame, label, args.pod_cost_per_hour, step_seconds, sla_ms))

    if args.model is not None:
        policy, label, _, _ = load_rl_policy(args.model, args.algo, args.vecnormalize)
        env = CloudAutoscalingEnv(trace_path=args.trace, episode_steps=None, random_start=False)
        frame = rollout(env, policy, seed=args.seed)
        env.close()
        results[label] = frame
        table.append(summarise(frame, label, args.pod_cost_per_hour, step_seconds, sla_ms))

    if not results:
        print("[evaluate] nothing to evaluate: pass --model or drop --no-baseline")
        return 1

    table_frame = pd.DataFrame(table)
    print()
    print(table_frame.to_string(index=False))
    print()

    if len(table) == 2:
        baseline, learned = table
        cost_delta = (
            (learned["Total Cost ($)"] - baseline["Total Cost ($)"])
            / max(baseline["Total Cost ($)"], 1e-9) * 100.0
        )
        sla_delta = learned["SLA Violation Rate (%)"] - baseline["SLA Violation Rate (%)"]
        print(f"RL vs HPA baseline: cost {cost_delta:+.2f}% | "
              f"SLA violation rate {sla_delta:+.2f} percentage points")

    latency_path = outdir / "latency_comparison.png"
    pods_path = outdir / "pod_provisioning_comparison.png"
    metrics_path = outdir / "metrics.csv"

    plot_latency(results, sla_ms, latency_path)
    plot_provisioning(results, pods_path)
    table_frame.to_csv(metrics_path, index=False)

    print(f"[evaluate] wrote {latency_path}")
    print(f"[evaluate] wrote {pods_path}")
    print(f"[evaluate] wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
