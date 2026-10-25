"""Where do the learned policies breach, and did the 480 s tracker know better?

For every step a checkpoint exceeds the SLA, print the traffic, the fleet the
checkpoint ran, the fleet the trailing-max tracker would have run, and whether
the burst was already visible in the previous 480 s. This answers the only
question the ladder leaves open: are the residual breaches *information* the
policy failed to use, or *ramp* no policy could have beaten.

    python utils/breach_diag.py models/sla/r1-reward-only.zip models/sla/r2-lagrangian.zip
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from env.cloud_env import (
    DEFAULT_SLA_LATENCY_MS, CloudAutoscalingEnv, load_trace,
)
from env.design_wrappers import OBS_DIM, HeadroomObs
from utils.design_probe import util_policy
from utils.score_policy import load_policy, rollout

STEPS = 8640
WINDOW = 48
U = 0.77
SLA = DEFAULT_SLA_LATENCY_MS

trace = load_trace(None)
tracker = util_policy(trace, U, "trailing", WINDOW)
targets = np.asarray([tracker(i, float(trace[i]), 0) for i in range(STEPS)], dtype=np.int64)
b1 = rollout(CloudAutoscalingEnv(episode_steps=STEPS, random_start=False, seed=0),
             STEPS, targets=targets)
print(f"tracker baseline: {int((b1.latency > SLA).sum())} violations, "
      f"max {b1.latency.max():.0f} ms, mean {b1.pods.mean():.2f} pods")
print(f"trace: {STEPS} steps, peak {trace.max():.1f} rps at step {int(trace.argmax())}")
print()

CHECKPOINTS = sys.argv[1:] or ["models/sla/r1-reward-only.zip",
                               "models/sla/r2-lagrangian.zip"]
for name in CHECKPOINTS:
    decide, label, obs_dim, _ = load_policy(ROOT / name)
    env = CloudAutoscalingEnv(episode_steps=STEPS, random_start=False, seed=0)
    if obs_dim == OBS_DIM:
        env = HeadroomObs(env)
    rec = rollout(env, STEPS, decide=decide)
    bad = np.flatnonzero(rec.latency > SLA)
    print(f"== {label}: {bad.size} violating steps, "
          f"mean {rec.pods.mean():.2f} pods, "
          f"max |pods - tracker| = {np.abs(rec.pods - b1.pods).max()}, "
          f"mean |pods - tracker| = {np.abs(rec.pods - b1.pods).mean():.2f}")
    for i in bad:
        lo = max(0, i - WINDOW)
        print(f"   step {i:5d} ({i * 10 // 3600:02d}:{i * 10 % 3600 // 60:02d}) "
              f"rps {rec.rps[i]:6.2f}  pods {rec.pods[i]:2d}  lat {rec.latency[i]:7.1f}  | "
              f"tracker pods {b1.pods[i]:2d} lat {b1.latency[i]:6.1f}  | "
              f"max rps prev 480 s {rec.rps[lo:i + 1].max():6.2f}  "
              f"peak rps this hour {trace[i // 360:i // 360 + 360].max():6.2f}")
        print(f"           pods over the previous 12 steps: "
              f"{list(rec.pods[max(0, i - 12):i + 1])}")
        print(f"           tracker over the same steps:    "
              f"{list(b1.pods[max(0, i - 12):i + 1])}")
    # how often the two disagree by more than one pod
    gap = np.abs(rec.pods - b1.pods)
    print(f"   steps where |pods - tracker| >= 2: {int((gap >= 2).sum())}")
    print()
