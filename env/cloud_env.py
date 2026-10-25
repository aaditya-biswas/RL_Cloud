"""Gymnasium environment modelling a Kubernetes-style autoscaling problem.

MDP definition
--------------
State  : [cpu_utilization, ram_utilization, request_rate_rps,
          average_latency_ms, active_pods]
Action : Discrete(5) -> pod deltas (-2, -1, 0, +1, +2), pods clipped to [1, 15]
Reward : -(1.5 * pods + 0.05 * max(0, latency - 200)^2 + 0.5 * |delta_pods|)

Physics
-------
Capacity of the fleet is ``mu_total = active_pods * mu_pod`` (mu_pod = 60 RPS).
CPU utilisation saturates at 1.0, and latency follows a simplified M/M/k
waiting time in the stable region plus a hard breach value when overloaded::

    mu_total > rps : latency = 1000 / (mu_total - rps) + 15.0
    otherwise      : latency = 3000.0

RAM is not specified by the original MDP formulation, so it is modelled as an
exponential moving average of the load-dependent working set::

    ram_t = 0.8 * ram_{t-1} + 0.2 * (0.20 + 0.60 * cpu_t)

Containers do not release memory the instant CPU drops, so RAM lags the load.
Because RAM is part of the observation the MDP remains Markov.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_PATH = PROJECT_ROOT / "data" / "workload_trace.csv"

# Action index -> pod delta.
ACTION_DELTAS = (-2, -1, 0, 1, 2)
ACTION_LABELS = {-2: "scale down x2", -1: "scale down", 0: "hold", 1: "scale up", 2: "scale up x2"}

DEFAULT_MU_POD = 60.0           # requests per second served by one pod
DEFAULT_MIN_PODS = 1
DEFAULT_MAX_PODS = 15
DEFAULT_SLA_LATENCY_MS = 200.0
DEFAULT_OVERLOAD_LATENCY_MS = 3000.0
DEFAULT_BASE_LATENCY_MS = 15.0  # network + processing floor
DEFAULT_STEP_SECONDS = 10

DEFAULT_COST_WEIGHT = 1.5       # $ per pod per step
DEFAULT_SLA_WEIGHT = 0.05       # per (ms over SLA)^2
DEFAULT_CHURN_WEIGHT = 0.5      # per pod scaled

RAM_IDLE_UTILIZATION = 0.20
RAM_LOAD_COEFFICIENT = 0.60
RAM_MEMORY = 0.80               # weight on the previous RAM reading

_TRACE_CACHE: Dict[tuple, np.ndarray] = {}


def load_trace(trace_path: Optional[Path] = None) -> np.ndarray:
    """Load ``request_rate_rps`` from the workload CSV (cached by path + mtime).

    Uses ``numpy.loadtxt`` so the environment has no pandas dependency at run
    time; the generator is the only component that needs pandas.
    """
    path = Path(trace_path).expanduser() if trace_path else DEFAULT_TRACE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Workload trace not found: {path}\n"
            "Run `python data/generator.py` first to create it."
        )

    key = (str(path.resolve()), path.stat().st_mtime)
    cached = _TRACE_CACHE.get(key)
    if cached is not None:
        return cached

    trace = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64, ndmin=1)
    if trace.size < 2:
        raise ValueError(f"Trace must contain at least two rows, got {trace.size}: {path}")
    if not np.all(np.isfinite(trace)):
        raise ValueError(f"Trace contains non-finite values: {path}")
    trace = np.clip(trace, 0.0, None)

    _TRACE_CACHE[key] = trace
    return trace


class CloudAutoscalingEnv(gym.Env):
    """Simulated autoscaling cluster driven by a replayed 24 h traffic trace.

    Parameters
    ----------
    trace_path:
        CSV with a ``request_rate_rps`` column (see ``data/generator.py``).
    episode_steps:
        Episode horizon in environment steps. ``None`` means one full pass over
        the trace (8,640 steps = 24 h). Training usually uses a shorter horizon.
    random_start:
        If True each ``reset`` starts at a random offset, which improves
        coverage during training. Evaluation should keep this False.
    trace_paths:
        Optional list of traces. When given, every ``reset`` draws one of them
        uniformly *for the whole episode* - this is how a train/validation/test
        split by day is expressed (see ``utils/build_days.py``). ``trace_path``
        is then ignored for the replay, but the observation scale ``max_rps`` is
        held at the maximum over all days so the observation does not change
        meaning from one episode to the next.
    initial_pods:
        ``"auto"`` sizes the fleet from the first traffic sample; an int forces
        a cold start at that pod count.
    reward_scale:
        Multiplies the reward. Leave at 1.0 to match the MDP definition; useful
        for exploring reward normalisation strategies.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        trace_path: Optional[Path] = None,
        episode_steps: Optional[int] = None,
        random_start: bool = False,
        mu_pod: float = DEFAULT_MU_POD,
        min_pods: int = DEFAULT_MIN_PODS,
        max_pods: int = DEFAULT_MAX_PODS,
        sla_latency_ms: float = DEFAULT_SLA_LATENCY_MS,
        overload_latency_ms: float = DEFAULT_OVERLOAD_LATENCY_MS,
        base_latency_ms: float = DEFAULT_BASE_LATENCY_MS,
        cost_weight: float = DEFAULT_COST_WEIGHT,
        sla_weight: float = DEFAULT_SLA_WEIGHT,
        churn_weight: float = DEFAULT_CHURN_WEIGHT,
        reward_scale: float = 1.0,
        step_seconds: int = DEFAULT_STEP_SECONDS,
        initial_pods: Any = "auto",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        trace_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        super().__init__()

        self._day_paths: List[Path] = [Path(p) for p in trace_paths] if trace_paths else []
        self._day_traces: List[np.ndarray] = [load_trace(p) for p in self._day_paths]
        if trace_paths and not self._day_paths:
            raise ValueError("trace_paths was given but empty")
        self.trace = self._day_traces[0] if self._day_traces else load_trace(trace_path)
        self.trace_path = (
            self._day_paths[0] if self._day_paths
            else (Path(trace_path) if trace_path else DEFAULT_TRACE_PATH)
        )
        self.day_name = self.trace_path.stem
        self.trace_length = int(self.trace.size)
        self.step_seconds = int(step_seconds)

        self.mu_pod = float(mu_pod)
        self.min_pods = int(min_pods)
        self.max_pods = int(max_pods)
        if self.min_pods < 1:
            raise ValueError("min_pods must be >= 1")
        if self.max_pods < self.min_pods:
            raise ValueError("max_pods must be >= min_pods")

        self.sla_latency_ms = float(sla_latency_ms)
        self.overload_latency_ms = float(overload_latency_ms)
        self.base_latency_ms = float(base_latency_ms)
        self.cost_weight = float(cost_weight)
        self.sla_weight = float(sla_weight)
        self.churn_weight = float(churn_weight)
        self.reward_scale = float(reward_scale)

        # Every day in ``trace_paths`` is 8,640 steps; the shortest one wins so an
        # episode can never run off the end of a trace.
        shortest = min((t.size for t in self._day_traces), default=self.trace_length)
        self.episode_steps = int(episode_steps) if episode_steps else int(shortest)
        if self.episode_steps < 1:
            raise ValueError("episode_steps must be >= 1")
        self.random_start = bool(random_start)
        self.initial_pods = initial_pods
        self.render_mode = render_mode
        self._initial_seed = seed

        # Fixed across episodes on purpose: the observation scale must not drift
        # when the day changes, or "demand_pods = 0.8" means something different
        # on a quiet day than on a busy one.
        self.max_rps = float(max(t.max() for t in self._day_traces)) if self._day_traces \
            else float(self.trace.max())
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, float(self.min_pods)], dtype=np.float32)
        obs_high = np.array(
            [1.0, 1.0, max(1.0, self.max_rps * 1.25), self.overload_latency_ms, float(self.max_pods)],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))

        # Mutable episode state (populated by reset()).
        self._episode_start = 0
        self._episode_length = self.episode_steps
        self._cursor = 0
        self._t = 0
        self._pods = self.min_pods
        self._ram = RAM_IDLE_UTILIZATION

    # ------------------------------------------------------------------ helpers
    def _initial_pods_for(self, rps: float) -> int:
        """Sizing heuristic used by ``initial_pods='auto'`` (~80 % CPU target)."""
        pods = int(np.ceil(max(rps, 1e-9) / (0.80 * self.mu_pod)))
        return int(np.clip(pods, self.min_pods, self.max_pods))

    def _select_day(self) -> str:
        """Draw the episode's trace from ``trace_paths``; no-op for a single trace.

        The day is fixed for the whole episode (picked in ``reset``), so a
        training episode is a coherent 4 h window of one real day rather than a
        stitched sequence of unrelated days.
        """
        if len(self._day_traces) > 1:
            k = int(self.np_random.integers(len(self._day_traces)))
            self.trace = self._day_traces[k]
            self.trace_path = self._day_paths[k]
            self.trace_length = int(self.trace.size)
        self.day_name = self.trace_path.stem
        return self.day_name

    def _latency_ms(self, rps: float, mu_total: float) -> float:
        """Simplified M/M/k latency, capped so it always fits the observation box."""
        if mu_total > rps:
            return min(self.overload_latency_ms, 1000.0 / (mu_total - rps) + self.base_latency_ms)
        return self.overload_latency_ms

    def _state_at(self, index: int) -> np.ndarray:
        """Observation describing ``trace[index]`` under the current pod count."""
        rps = float(self.trace[index])
        mu_total = self._pods * self.mu_pod
        cpu = min(1.0, rps / mu_total)
        latency = self._latency_ms(rps, mu_total)
        return np.array([cpu, self._ram, rps, latency, float(self._pods)], dtype=np.float32)

    # -------------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode; ``options`` may override start index and horizon."""
        seed = seed if seed is not None else self._initial_seed
        self._initial_seed = None
        super().reset(seed=seed)

        day = self._select_day()
        options = options or {}
        start = options.get("start_index")
        horizon = int(options.get("episode_steps", self.episode_steps))

        if start is None:
            if self.random_start:
                max_start = max(1, self.trace_length - min(horizon, self.trace_length))
                start = int(self.np_random.integers(0, max_start))
            else:
                start = 0
        start = int(np.clip(start, 0, self.trace_length - 1))

        self._episode_start = start
        self._episode_length = max(1, min(horizon, self.trace_length - start))
        self._cursor = start
        self._t = 0

        if isinstance(self.initial_pods, str) and self.initial_pods == "auto":
            self._pods = self._initial_pods_for(float(self.trace[start]))
        else:
            self._pods = int(np.clip(int(self.initial_pods), self.min_pods, self.max_pods))

        cpu0 = min(1.0, float(self.trace[start]) / (self._pods * self.mu_pod))
        self._ram = float(np.clip(RAM_IDLE_UTILIZATION + RAM_LOAD_COEFFICIENT * cpu0, 0.0, 1.0))

        info = {"start_index": start, "episode_steps": self._episode_length, "day": day}
        return self._state_at(start), info

    def step(self, action: int):
        """Apply a pod delta and advance the trace by one 10-second slot."""
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{self.action_space.n - 1}")

        index = self._cursor
        rps = float(self.trace[index])

        previous_pods = self._pods
        pods = int(np.clip(previous_pods + ACTION_DELTAS[action], self.min_pods, self.max_pods))
        delta = pods - previous_pods
        mu_total = pods * self.mu_pod

        cpu = min(1.0, rps / mu_total)
        latency = self._latency_ms(rps, mu_total)
        self._ram = float(
            np.clip(
                RAM_MEMORY * self._ram
                + (1.0 - RAM_MEMORY) * (RAM_IDLE_UTILIZATION + RAM_LOAD_COEFFICIENT * cpu),
                0.0,
                1.0,
            )
        )

        cost = self.cost_weight * pods
        sla_penalty = self.sla_weight * max(0.0, latency - self.sla_latency_ms) ** 2
        churn_penalty = self.churn_weight * abs(delta)
        reward = -(cost + sla_penalty + churn_penalty) * self.reward_scale

        self._pods = pods
        self._t += 1
        self._cursor += 1
        truncated = self._t >= self._episode_length

        info = {
            "index": index,
            "request_rate_rps": rps,
            "cpu_utilization": cpu,
            "ram_utilization": self._ram,
            "latency_ms": latency,
            "active_pods": pods,
            "previous_pods": previous_pods,
            "delta_pods": delta,
            "action": action,
            "cost": cost,
            "sla_penalty": sla_penalty,
            "churn_penalty": churn_penalty,
            "reward": float(reward),
            "sla_violation": bool(latency > self.sla_latency_ms),
        }
        observation = self._state_at(min(self._cursor, self.trace_length - 1))
        return observation, float(reward), False, bool(truncated), info

    def render(self):
        """Print a single status line (``render_mode="human"``)."""
        if self.render_mode != "human":
            return None
        observation = self._state_at(min(self._cursor, self.trace_length - 1))
        hour = (self._cursor * self.step_seconds % 86_400) / 3_600.0
        print(
            f"[{hour:05.2f}h] pods={self._pods:2d} rps={observation[2]:7.1f} "
            f"cpu={observation[0]:.2f} ram={observation[1]:.2f} "
            f"latency={observation[3]:7.1f}ms"
        )
        return None

    def close(self) -> None:
        """No resources to release; provided for API completeness."""
        return None

    # ------------------------------------------------------------------ properties
    @property
    def episode_length(self) -> int:
        """Number of steps in the current episode."""
        return self._episode_length

    @property
    def current_index(self) -> int:
        """Trace index of the next step to be served."""
        return min(self._cursor, self.trace_length - 1)

    @property
    def active_pods(self) -> int:
        """Current pod count."""
        return self._pods
