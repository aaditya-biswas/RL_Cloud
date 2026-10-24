"""Opt-in wrappers implementing the design in ``docs/DESIGN.md``.

Both are drop-in ``gymnasium`` wrappers: the default environment and the
``models/`` checkpoints keep their 5-dim observation and quadratic reward, so the
numbers in ``README.md`` stay valid. Enable them only when you want to train a
policy against the redesigned observation / reward:

    env = CloudAutoscalingEnv(episode_steps=1440, random_start=True)
    env = LinearSLAReward(env)   # reward v2: linear, capped, churn on request
    env = HeadroomObs(env)       # 11-dim observation built around headroom

``HeadroomObs`` *replaces* the observation, so its shape changes from ``(5,)`` to
``(11,)`` and a model trained on the old shape will not load. That is intentional:
the two observations are different MDPs.

Why these features
------------------
The stock observation is ``[cpu, ram, rps, latency, pods]``. Two of those are
hard to steer on (``ram`` is a lagged function of ``cpu``; ``cpu`` already
saturates at 1.0), ``rps`` is on an absolute scale that ``VecNormalize`` clips at
the peak, and ``latency`` is a *lagging* signal: it only leaves the floor once the
fleet is already overloaded. The wrapper reports everything in pod units - "demand
is 8.0 pods, I have 9, so 1.0 pod of slack" - plus recent slope and two trailing
peaks, which are the cheapest causal stand-in for a forecast.

The second, slow peak is not decoration. The best causal controller on this trace
provisions to the **8-minute** high-water mark (``docs/DESIGN.md`` section 8:
-60,088 with zero breaches, versus -94,797 for a 30 s window), and a feed-forward
policy cannot compute an 8-minute maximum from a 60 s window - whatever the
network size, the information is not in its input.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.cloud_env import (
    ACTION_DELTAS,
    DEFAULT_COST_WEIGHT,
    DEFAULT_MAX_PODS,
    DEFAULT_MIN_PODS,
    DEFAULT_SLA_LATENCY_MS,
    CloudAutoscalingEnv,
)

OBS_DIM = 11


class HeadroomObs(gym.ObservationWrapper):
    """Eleven features expressed in pod units instead of raw telemetry.

    ====  ==================================================================
    0     ``cpu_utilization`` (unchanged, still bounded by 1.0)
    1     ``ram_utilization`` (unchanged; kept so plots stay comparable)
    2     ``demand_pods`` = ``rps / mu_pod`` - fleet size at 100 % utilisation
    3     ``min(latency / sla, 3)`` - 1.0 is exactly the SLA, 3.0 is ">= 600 ms"
    4     ``pods / max_pods``
    5     ``headroom_pods`` = ``(pods * mu_pod - rps) / mu_pod``, negative = over
    6     ``slope_pods`` - demand change over the last ``slope_steps`` steps,
          in pods, i.e. "how much more fleet this ramp still needs"
    7     ``trailing_pods`` - peak demand over the last ``window`` steps (60 s)
    8     ``peak_pods_slow`` - peak demand over the last ``slow_window`` steps
          (480 s); the statistic the winning causal policy of section 8 uses
    9     ``impulse_pods`` - demand change in the last step, in pods per step;
          the 387 rps jumps in this trace read as +6.5 here (clipped at +-1 pod
          per step, i.e. the largest move the actuator can make)
    10    ``fraction_of_day`` - time-of-day; only useful if you also shift the
          trace per episode (see ``docs/DESIGN.md``), otherwise it is an index
    ====  ==================================================================
    """

    LATENCY_CAP = 3.0
    SLOW_WINDOW = 48  # 8 minutes at 10 s/step

    def __init__(
        self,
        env: gym.Env,
        window: int = 6,
        slope_steps: int = 3,
        slow_window: int = SLOW_WINDOW,
        require_wrapped: bool = True,
    ) -> None:
        super().__init__(env)
        if require_wrapped and not isinstance(env.unwrapped, CloudAutoscalingEnv):
            raise TypeError("HeadroomObs expects a CloudAutoscalingEnv inside")
        self.window = max(1, int(window))
        self.slope_steps = max(1, int(slope_steps))
        self.slow_window = max(self.window, int(slow_window))
        self._last_ram = 0.0

        # dims: cpu, ram, demand, latency, pods, headroom, slope, fast peak,
        #       slow peak, impulse, fraction_of_day. The two signed features
        #       (headroom, slope) and impulse_pods go negative; impulse was
        #       declared [0, 1] before and left the Box whenever demand fell.
        low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32
        )
        high = np.ones(OBS_DIM, dtype=np.float32)
        # Latency is capped at 3x SLA (600 ms): below that the value is informative,
        # above it "very late" and "dead" are the same decision, and the 3,000 ms
        # overload would otherwise be a 15-sigma outlier for VecNormalize.
        high[3] = self.LATENCY_CAP
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _features(self) -> np.ndarray:
        env = self.env.unwrapped
        index = int(env.current_index)
        trace = env.trace
        rps = float(trace[index])
        pods = float(env.active_pods)
        mu_pod = float(env.mu_pod)
        capacity = pods * mu_pod
        scale = max(1.0, float(env.max_rps) / mu_pod)

        cpu = min(1.0, rps / capacity)
        latency = (
            min(env.overload_latency_ms, 1000.0 / (capacity - rps) + env.base_latency_ms)
            if capacity > rps
            else env.overload_latency_ms
        )
        demand = rps / mu_pod
        trailing = float(trace[max(0, index - self.window + 1): index + 1].max()) / mu_pod
        slow = float(trace[max(0, index - self.slow_window + 1): index + 1].max()) / mu_pod
        # Pods per step of recent growth; +-2 is the action limit, so this reads
        # "how many steps of full ramp this burst still needs".
        slope = (rps - float(trace[max(0, index - self.slope_steps)])) / (
            self.slope_steps * mu_pod
        )
        headroom = (capacity - rps) / mu_pod
        impulse = (rps - float(trace[max(0, index - 1)])) / mu_pod

        return np.array(
            [
                cpu,
                self._last_ram,
                min(1.0, demand / scale),
                min(latency / float(env.sla_latency_ms), self.LATENCY_CAP),
                pods / float(env.max_pods),
                float(np.clip(headroom, -1.0, 1.0)),
                float(np.clip(slope, -1.0, 1.0)),
                min(1.0, trailing / scale),
                min(1.0, slow / scale),
                float(np.clip(impulse, -1.0, 1.0)),
                (index % env.trace_length) / float(env.trace_length),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_ram = float(observation[1])
        return self._features(), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self._last_ram = float(observation[1])
        return self._features()


class LinearSLAReward(gym.Wrapper):
    """Reward v2 from ``docs/DESIGN.md``.

    ``-(cost*pods + V*clamp((latency-soft)/(hard-soft), 0, cap_mult) + churn*|requested|)``

    Differences from the stock reward:

    * **Linear instead of quadratic.** The stock ``0.05 * (latency - 200)**2``
      pays 392,012 for a 10 s overload at the peak (480 rps, 8 pods) but only 125
      for a mild 250 ms overshoot. The jump between adjacent pod counts is a
      cliff, the gradient around it is useless, and the optimal learned response
      is "never risk it" - which is why the reference agent sits at 15 pods.
    * **Capped.** ``cap_mult`` bounds a full outage at a few times a mild breach,
      so the policy is not dominated by the worst 20 minutes of the day.
    * **Churn on the requested delta.** The stock reward charges the *realised*
      change, so asking for +2 while already at 15 pods costs nothing: 8,535 of
      the 200k reference agent's requests are exactly that no-op.
    """

    def __init__(
        self,
        env: gym.Env,
        cost_weight: float = DEFAULT_COST_WEIGHT,
        churn_weight: float = 4.0,
        soft_ms: float = 150.0,
        hard_ms: float = DEFAULT_SLA_LATENCY_MS,
        violation_cost: float = 200.0,
        cap_mult: Optional[float] = 5.0,
    ) -> None:
        super().__init__(env)
        self.cost_weight = float(cost_weight)
        self.churn_weight = float(churn_weight)
        self.soft_ms = float(soft_ms)
        self.hard_ms = float(hard_ms)
        self.violation_cost = float(violation_cost)
        self.cap_mult = cap_mult

    def penalty(self, latency_ms: float) -> float:
        """Penalty term alone, so it can be plotted or debugged independently."""
        ramp = self.violation_cost / max(1e-9, self.hard_ms - self.soft_ms)
        penalty = ramp * max(0.0, latency_ms - self.soft_ms)
        if self.cap_mult is not None:
            penalty = min(penalty, self.cap_mult * self.violation_cost)
        return penalty

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        requested = int(ACTION_DELTAS[int(action)])
        reward = -(
            self.cost_weight * float(info["active_pods"])
            + self.penalty(float(info["latency_ms"]))
            + self.churn_weight * abs(requested)
        )
        info = {**info, "reward_v2": reward, "requested_delta": requested}
        return observation, reward, terminated, truncated, info
