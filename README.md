# Cloud Infrastructure Autoscaling with Reinforcement Learning

A self-contained research sandbox for **RL-based autoscaling of a cloud service**:
a Gymnasium M/M/k queueing environment driven by a replayed *real* HTTP workload,
a Kubernetes-HPA-style rule-based baseline, a full evaluation harness, and a PPO/DQN
training skeleton designed to be filled in step by step.

The question the project answers:

> Given a 24-hour traffic trace and a 200 ms latency SLA, can a learned policy
> provision pods more cheaply / more safely than a threshold-based autoscaler?

---

## 1. What is in the box

```
.
├── raw/
│   └── access_log_Jul95            # NASA Kennedy Space Center HTTP log (196 MB, not in git)
├── data/
│   ├── generator.py                # raw log -> 8,640 x 10 s RPS trace (or synthetic fallback)
│   └── workload_trace.csv          # generated artefact
├── env/
│   └── cloud_env.py                # CloudAutoscalingEnv (Gymnasium)
├── agents/
│   ├── baseline_scaler.py          # HPA-style rule-based baseline
│   ├── train_rl.py                 # WORKBOOK: four functions for you to implement
│   ├── self_check.py               # the contracts behind --self-check
│   └── solutions/
│       └── train_rl_reference.py   # worked implementation (read it after you try)
├── docs/
│   └── WALKTHROUGH.md              # Sessions 0-5: reasoning, checkpoints, exercises
├── models/                         # trained model + vecnormalize pairs (not in git)
├── utils/
│   ├── evaluate.py                 # baseline vs RL benchmark + charts + metrics.csv
│   ├── latency_comparison.png
│   └── pod_provisioning_comparison.png
├── requirements.txt
└── README.md
```

---

## 2. Data pipeline: from a 1995 web log to a modern traffic trace

`raw/access_log_Jul95` contains **1.89 M requests** logged by the NASA Kennedy Space
Center web server in July 1995. `data/generator.py` turns it into a fixed-length
time series suitable for RL:

1. **Pick one day.** July 13, 1995 is the busiest day in the log: 134,203 requests,
   mean 1.55 RPS, 10-second peak 7.8 RPS.
2. **Bin** every matching log line into 8,640 ten-second slots (24 h × 360 bins/h).
3. **Fill gaps.** 402 of the 8,640 bins have no records (collector outages, e.g. ~24
   minutes around 19:00). They are linearly interpolated instead of being treated as
   "zero traffic", otherwise the agent would learn to scale to one pod right before
   a spike.
4. **Rescale to a production-like load.** 1995 traffic (7.8 RPS peak) would never
   stress a pod fleet sized for 60 RPS/pod. The whole day is multiplied by
   **61.54×** so the busiest bin hits **480 RPS**, which makes the pod count span
   1–11 and pushes the problem into the interesting region of the action space.

Resulting profile:

| metric | value |
|---|---|
| mean RPS | 96.84 |
| median RPS | 73.85 |
| p95 RPS | 246.15 |
| peak RPS | 480.00 |
| min RPS | 6.15 |
| interpolated bins | 402 / 8,640 |

Synthetic fallback (no raw log needed, same CSV schema):

```bash
python data/generator.py --source synthetic --seed 7
```

---

## 3. MDP formulation

### State — `Box(5,)`, `float32`

| index | feature | range | notes |
|---|---|---|---|
| 0 | CPU utilisation | [0, 1] | `min(1, rps / (pods * 60))` |
| 1 | RAM utilisation | [0, 1] | EMA of the load-dependent working set |
| 2 | incoming request rate | [0, 600] RPS | current 10 s bin |
| 3 | average latency | [0, 3000] ms | capped at the overload value |
| 4 | active pods | [1, 15] | current fleet size |

### Action — `Discrete(5)`

`0: -2 pods`, `1: -1 pod`, `2: hold`, `3: +1 pod`, `4: +2 pods`.
Pod counts are clipped to `[1, 15]`, so the agent cannot go negative or exceed the
cluster capacity.

### Reward (exactly the specification)

```
reward = -( 1.5 * active_pods
            + 0.05 * max(0, latency_ms - 200)^2
            + 0.5  * |delta_pods| )
```

* `1.5 × pods` — infrastructure cost per 10 s step.
* `0.05 × (latency − 200)²` — quadratic SLA penalty. Overloaded steps cost
  `0.05 × 2800² = 392,000`, which dominates everything else; this is intentional
  (it is what the spec asks for) and is why the training script normalises rewards.
* `0.5 × |Δpods|` — churn / cold-start penalty.

### Latency physics

`mu_total = active_pods * 60` RPS of capacity.

```
mu_total > rps :  latency = min(3000, 1000 / (mu_total - rps) + 15)   # ms
otherwise      :  latency = 3000                                     # hard overload
```

At `rps = 0.8 * mu_total` latency is 98 ms with one pod and 32 ms with five. The
200 ms SLA is crossed at ~91 % CPU utilisation with a single pod, ~96 % with two and
~97 % with three, so the controller has to hold utilisation near or below 90 % —
and it only becomes *impossible* once `rps > mu_total` (hard 3000 ms overload).

### RAM assumption (documented, not in the spec)

```
ram_t = 0.8 * ram_{t-1} + 0.2 * (0.20 + 0.60 * cpu_t)
```

Containers do not release memory the instant CPU drops, so RAM lags the load. Because
RAM is part of the observation, the MDP stays Markov.

### Episode handling

* Training: `episode_steps=1440` (4 hours) with `random_start=True`.
* Evaluation: full 8,640-step day, `start_index=0`.
* `terminated=False` always; `truncated=True` at the horizon. The workload never
  "ends" — it is a continuing task cut into fixed-length chunks.

---

## 4. Setup (verified on Python 3.12)

```bash
conda create -n cloudrl python=3.12 -y
conda activate cloudrl
pip install --upgrade pip
# CPU wheel keeps the install ~200 MB instead of ~2.5 GB
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Versions used while building and validating this repo:

| package | version |
|---|---|
| Python | 3.12.14 |
| gymnasium | 1.4.0 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| matplotlib | 3.11.2 |
| torch | 2.14.1+cpu |
| stable-baselines3 | 2.9.0 |
| tensorboard | 2.21.0 |

Everything runs on CPU; no GPU is required.

---

## 5. Quick start

```bash
# 1. build the 24 h trace (takes ~10 s: streams the 196 MB log once)
python data/generator.py

# 2. what is still missing? four functions, PASS/FAIL per step (~10 s)
python agents/train_rl.py --self-check

# 3. sanity-check the MDP (check_env + hand-checked reward table + rollout)
python agents/train_rl.py --dry-run

# 4. benchmark the rule-based baseline, the fixed fleets and write charts
python utils/evaluate.py --constant-pods 8,9,15

# 5. train, once all four self-checks pass (see section 7 for the knobs)
python agents/train_rl.py --algo ppo --timesteps 200000

# 6. compare the trained agent against the baseline and the fixed fleets
python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo \
    --constant-pods 9
```

`python agents/baseline_scaler.py` also runs the baseline standalone and prints a
compact summary (useful as a first smoke test).

---

## 6. Reference results (sandbox author's runs)

Everything in this section was produced by the author of this sandbox, on one
machine, with `--seed 42`. Treat it as a measuring stick, not a finish line: the
rows are reproducible with the commands below, and your own numbers go in §6.2.

### 6.1 Full day, deterministic

HPA-style thresholds on CPU: `> 0.75` → `+1 pod`, `< 0.25` → `-1 pod`, otherwise hold.
Rows marked *(fixed fleet)* are not learned policies: they hold the pod count constant
for the whole day, and exist to answer "what would a static deployment have cost?" —
`python utils/evaluate.py --no-baseline --constant-pods 8,9,15`.

| Policy | Total Cost ($) | SLA Violations (%) | Avg Latency (ms) | P95 Latency (ms) | Avg Pods | Scale Actions | Total Reward |
|---|---|---|---|---|---|---|---|
| HPA baseline | 10.07 | 0.38 | 38.60 | 49.21 | 3.50 | 1,865 | −11,027,157.4 |
| constant 8 pods *(fixed fleet)* | 23.04 | 0.01 | 18.13 | 19.28 | 8.00 | 0 | −495,680.0 |
| **constant 9 pods** *(fixed fleet)* | **25.92** | **0.00** | **17.35** | **18.40** | **9.00** | **0** | **−116,640.0** |
| constant 15 pods *(fixed fleet)* | 43.20 | 0.00 | 16.26 | 16.53 | 15.00 | 0 | −194,400.0 |
| PPO, ~4k steps (smoke) | 39.84 | 0.00 | 16.37 | 16.53 | 13.83 | 1,382 | −180,007.5 |
| PPO, 200k steps | 43.19 | 0.00 | 16.26 | 16.53 | 15.00 | 7 | −194,344.0 |
| PPO, 100k steps, `--cost-weight 20` | 32.70 | 0.00 | 16.98 | 18.49 | 11.35 | 53 | −147,175.0 |
| PPO, 100k steps, `--cost-weight 100` | 28.16 | 0.02 | 18.24 | 21.25 | 9.78 | 3,539 | −521,523.9 |

#### Why 8 pods is a cliff and 9 pods is free

Latency is `base + 1000 / (mu_pod × pods − rps)` with `mu_pod = 60` rps per pod and
`base = 15 ms`; anything at or above capacity returns 3,000 ms. The trace averages
97 rps and peaks at **480 rps for exactly one 10 s bin** (index 3,535). So:

| pods | capacity | at the 480 rps spike | violations / day | return / day |
|---|---|---|---|---|
| 8 | 480 | 480 − 480 = 0 → 3,000 ms | 1 step | −(1.5 × 8 × 8,640 + 392,000) = −495,680 |
| 9 | 540 | 1000 / 60 + 15 ≈ 32 ms | 0 | −(1.5 × 9 × 8,640) = −116,640 |
| 15 | 900 | 1000 / 420 + 15 ≈ 17 ms | 0 | −(1.5 × 15 × 8,640) = −194,400 |

Everything else in the day is served by 1–3 pods (mean 97 rps), which is why the
average latency of the fixed 9-pod fleet is 17.4 ms and its P95 is 18.4 ms — far below
the 200 ms SLA. One overloaded step costs `0.05 × (3000 − 200)² = 392,000`; one extra
pod for the whole day costs `1.5 × 8,640 = 12,960`. **Thirty days of a spare pod are
cheaper than one second of saturation**: the 8 → 9 cliff is a 379,040-point swing
(−495,680 → −116,640) for a 12,960 cost increase. That single ratio is the whole story
of this sandbox, and it is why 9 pods (just above the peak) is the cheapest fleet with
no violations, while 15 pods (the cap) is the *reward-optimal* one.

Reading the HPA row:

* The baseline is **cheap but laggy**. It adds one pod per 10 s step and only after
  CPU has already crossed 75 %, so short spikes (the trace contains several 10–60 s
  bursts) still breach the SLA: 33 of 8,640 steps.
* Those 33 steps account for almost the entire reward: ~330k per violation versus
  ~5 per step of infrastructure cost. **Reward is SLA-dominated by design.**
* 1,865 scaling actions over the day is a lot of churn — the CPU signal oscillates
  around the thresholds. `RuleBasedScaler(cooldown_steps=...)` adds an HPA-style
  stabilisation window if you want a calmer baseline:

```bash
python utils/evaluate.py --hpa-cooldown 3
```

The RL agent's opportunity is therefore: **pre-empt the ramp** (react to rising RPS
instead of current CPU), tolerate short bursts without scaling, and keep churn low. None
of the trained policies below realised that opportunity — see lesson 1.

### What these runs teach

Every row is one pass of `utils/evaluate.py` over the complete 8,640-step day. The
evaluator always builds `CloudAutoscalingEnv()` with the **default** reward weights and
reward normalisation **off**, so `Total Reward` is the raw default-objective return of
every policy — including rows that were *trained* with a modified `--cost-weight`. That
is the honest comparison: how well does a differently-trained policy score on the
objective we wrote down first.

1. **A static 9-pod fleet beats every policy in this table.** $25.92, zero violations,
   P95 18.4 ms, return **−116,640** — better than the 200k-step PPO (−194,344), the
   4k smoke run (−180,008) and the cost-aware `--cost-weight 20` run (−147,175). The
   learned policies are not "bad at autoscaling"; they are optimising the reward we
   specified, and that reward's optimum is not the cost optimum. This number is the
   bar the workbook asks you to beat, and it is why the project ends with a challenge
   (Session 5 of `docs/WALKTHROUGH.md`) rather than a victory lap.
2. **The reward objective is learned almost immediately.** Even the 4k-step smoke
   run drops the return from −11.03 M to −0.18 M simply by eliminating the
   392k-per-violation term. The scalarised reward *is* the objective.
3. **Longer training made the policy more extreme, not cheaper.** 200k steps ends at
   **15.00 pods for the entire day**. Two details make that readable: the "7 scale
   actions" are the climb (the episode starts at 1 pod and reaches the cap at step 6), and
   the greedy policy asks for more pods on **8,542 of 8,640** decisions — 8,535 of those
   requests are clipped to no-ops at `max_pods`. The flat line in
   `utils/pod_provisioning_comparison.png` is therefore the finding, not a plotting bug.
   It is the exact optimum of the specified reward.
4. **The default weights make over-provisioning rational.** The cliff table above is the
   arithmetic: avoiding one 3,000 ms step is worth 392,000, a spare pod for the whole day
   costs 12,960, so paying ~149k to remove 12.9 M of violation penalty is optimal and a
   reward-maximising agent *must* over-provision. The break-even cost weight is
   `392,000 / 8,640 ≈ 45`.
5. **A cost-heavier reward finds a strictly better policy.** Training with
   `--cost-weight 20` cuts the fleet from 15.00 to **11.35 pods** and the cost from
   $43.19 to **$32.70 while keeping the SLA violation rate at 0.00 %**. Scored on the
   *original* reward it returns **−147,175**, which beats the 200k-step run's
   −194,344 by 47k: the default-weight agent is stuck in a "pin at `max_pods`" local
   optimum, and reshaping the reward was what kicked it out. Shaping is an
   exploration tool, not only an objective change — though it still loses to a static
   fleet of 9, which is the point of lesson 1.
6. **Past a point, the trade-off turns against you.** `--cost-weight 100` squeezes the
   fleet to **9.78 pods / $28.16** but reintroduces SLA breaches (0.02 % of steps) and,
   more importantly, **3,539 scaling actions** instead of 53 — the policy starts
   flapping. Its default-reward return collapses to −521,524. Between the two ablation
   points the honest reading is: *the pod price buys the fleet size you ask for, and the
   churn term is what stops the agent from paying for it with flapping.*
7. **Violation cost is severity, not count.** The penalty is
   `0.05 × max(0, latency − SLA)²`, so a step that misses the SLA by 5 ms costs 1.25
   while one that misses it by 2,800 ms costs 392,000. That is why the `cost-weight 100`
   policy can log a 0.02 % violation rate for almost free — and why any SLA target you
   actually care about must be encoded as a hard constraint or a *linear* per-step
   penalty, not only as this quadratic term.

`cost_weight`, `sla_weight` and `churn_weight` are constructor arguments of
`CloudAutoscalingEnv`, exposed on the trainer as `--cost-weight / --sla-weight /
--churn-weight` (defaults read from `DEFAULT_COST_WEIGHT = 1.5`,
`DEFAULT_SLA_WEIGHT = 0.05`, `DEFAULT_CHURN_WEIGHT = 0.5`). The two ablation rows are
one flag apart from the reference command:

```bash
# 0. what still needs implementing? (~10 s, no training)
python agents/train_rl.py --self-check

# 1. the reference row: default weights, 200k steps (~20 min on CPU)
python agents/train_rl.py --algo ppo --timesteps 200000

# 2. the ablation rows (~6 min each on CPU)
python agents/train_rl.py --algo ppo --timesteps 100000 --cost-weight 20
python agents/train_rl.py --algo ppo --timesteps 100000 --cost-weight 100

# 3. score a pair, and score the fixed fleet it has to beat
python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo
python utils/evaluate.py --no-baseline --constant-pods 9
```

Every run writes `models/ppo_cloud_autoscaling.zip` **and**
`models/ppo_cloud_autoscaling_vecnormalize.pkl`, so move the pair into its own directory
between ablations (`mkdir -p models/cw20 && mv models/ppo_cloud_autoscaling* models/cw20/`)
and point `--model` at the copy; `utils/evaluate.py` finds the statistics from the model
filename and keeps the frozen eval env (`norm_reward=False`). The author's pairs live in
`models/` (200k, default weights), `models/cw20/` and `models/cw100/` — those are what
the ablation rows were scored from.

### 6.2 Your results

The same table, your numbers. Add one row per command you actually ran and keep the
command next to it — a result you cannot reproduce with one line is not a result.

| Policy | Total Cost ($) | SLA Violations (%) | Avg Latency (ms) | P95 Latency (ms) | Avg Pods | Scale Actions | Total Reward |
|---|---|---|---|---|---|---|---|
| HPA baseline (`python utils/evaluate.py`) | | | | | | | |
| constant 9 pods (`--no-baseline --constant-pods 9`) | | | | | | | |
| my PPO run (`--algo ppo --timesteps 200000`) | | | | | | | |

The bar to beat is the second row: **−116,640 / $25.92 / 0.00 %**. Session 5 of
`docs/WALKTHROUGH.md` explains why it is hard and what counts as a win.

---

## 7. Training guide: the workbook

`agents/train_rl.py` is a workbook. The environment, evaluator, callbacks, reward
plumbing and CLI are done; **four functions are yours to write**.

```bash
python agents/train_rl.py --self-check   # PASS/FAIL per step, ~10 s, no training
python agents/train_rl.py --dry-run      # watch the MDP act, without any RL
```

| Step | Function | You decide | Session |
|---|---|---|---|
| 1 | `build_vector_env` | what to wrap, what to normalise, train vs eval env | 1 |
| 2 | `build_model` | PPO or DQN, rollout/batch, network, seeds | 2 |
| 3 | `make_callbacks` | eval cadence, checkpointing, sync ordering, metrics | 3 |
| 4 | `save_artifacts` | how a model and its normalisation statistics travel together | 4 |

`docs/WALKTHROUGH.md` pairs every step with the reasoning, a command to run, a
checkpoint ("you should now see ...") and a knowledge check. Session 5 is the
acceptance test: **beat the constant 9-pod fleet's −116,640 Total Reward with 0.00 %
SLA violations**, scored by `utils/evaluate.py` on the default weights. Churn is already
priced into the reward (0.5 per action), so there is no separate action budget — but
note that a *perfect-foresight* oracle resizes 4,185 times and still scores only
−31,254, so frequent small adjustments are the expected shape of a good policy.

A worked implementation is in `agents/solutions/train_rl_reference.py` (4/4 on
`--self-check`). Reach for it when a step refuses to work, not as a copy-paste source —
the difference between the two files is the exercise.

The subsections below describe what that reference does, so they double as an answer key.

### Step 1 — Environment (`build_vector_env`)

`CloudAutoscalingEnv` → `Monitor` → `DummyVecEnv(n_envs)` → `VecNormalize`.

* **Observation normalisation is mandatory** here: RPS spans 6–480 while CPU/RAM live
  in [0, 1]. Without it the value function cannot fit the return scale.
* **Reward normalisation** (`norm_reward=True, clip_reward=10`) is the default answer
  to Q3 below.
* Training env: `episode_steps=1440`, `random_start=True`.
  Evaluation env: full 8,640-step day, no reward normalisation (so rewards stay
  comparable with the baseline table).

### Step 2 — Algorithm (`build_model`)

`MlpPolicy` with `net_arch=[128, 128]`, TensorBoard logging into `logs/tb/`.

### Step 3 — Callbacks (`make_callbacks`)

* `EvalCallback` — full-day deterministic evaluation every `--eval-freq` env steps,
  saves `models/best/best_model.zip`.
* `SyncVecNormalizeCallback` — copies the training env's `obs_rms` / `ret_rms` into
  the frozen eval env before every evaluation. Without it the eval env would
  normalise observations with its own fresh statistics and the evaluation would be
  measuring a different policy input distribution than the one being trained.
* `CheckpointCallback` — periodic snapshots in `models/checkpoints/`.
* `MetricsCallback` — logs `autoscaling/mean_pods`, `autoscaling/p95_latency_ms`,
  `autoscaling/sla_violation_rate` from the env `info` dict.

### Step 4 — Learn and save (`main`)

Saves the model **and** `models/<algo>_cloud_autoscaling_vecnormalize.pkl` (e.g.
`models/ppo_cloud_autoscaling_vecnormalize.pkl`, so a PPO and a DQN run never
overwrite each other). The normalisation statistics are
part of the policy: loading the model without them silently destroys performance
(the evaluator warns you if it cannot find them).

### Q1 — PPO or DQN?

| | PPO | DQN |
|---|---|---|
| Policy | on-policy, stochastic | off-policy, deterministic + ε-exploration |
| Replay buffer | no | yes (100k) |
| Stability on this reward | high | medium — the 392k penalty destabilises TD targets unless rewards are normalised |
| Sample efficiency | lower | higher per env step |
| Sensitive to | learning rate, `n_steps` | reward scale, `exploration_fraction`, `target_update_interval` |

**Start with PPO.** Once PPO is stable, run DQN with `--algo dqn` and compare
`rollout/ep_rew_mean` at equal wall-clock time.

### Q2 — Hyperparameters worth thinking about

| knob | default | effect on this problem |
|---|---|---|
| `--learning-rate` | `3e-4` | `1e-4` if the policy collapses to "always 15 pods"; `1e-3` for faster but noisier learning |
| `--gamma` | `0.99` | effective horizon ≈ 100 steps ≈ 17 min. `0.995` ≈ 33 min, better for pre-emptive ramps |
| `--timesteps` | `200k` | 200k–500k is the useful band; watch `rollout/ep_rew_mean` plateau. PPO always finishes the current rollout, so the real count is rounded up to a multiple of `n_steps × n_envs` (e.g. `--n-steps 2048 --n-envs 4` trains in 8,192-step chunks) |
| `--n-steps` | `2048` | PPO rollout length per env; must satisfy `n_steps * n_envs >= batch_size` |
| `--batch-size` | `256` | 64–512; larger = smoother gradients |
| `--n-envs` | `4` | more parallel episodes = better coverage of the daily profile |
| `--episode-steps` | `1440` | shorter episodes (e.g. 720) = more diverse starts per timestep |
| `--eval-freq` | `10000` | each eval is a full 8,640-step day, so don't set it too low |

Example runs:

```bash
python agents/train_rl.py --algo ppo --timesteps 300000 --gamma 0.995
python agents/train_rl.py --algo ppo --timesteps 300000 --learning-rate 1e-4 --n-envs 8
python agents/train_rl.py --algo dqn --timesteps 300000 --learning-rate 1e-4 --n-envs 1
```

### Q3 — Reward handling (the most important decision)

A single overloaded step contributes `0.05 * 2800² = 392,000`, i.e. **five orders of
magnitude** more than a pod costs (`1.5`). Three ways to deal with it:

1. **`VecNormalize(norm_reward=True, clip_reward=10)`** — default, keeps the MDP
   reward exactly as specified while feeding the learner sane targets.
2. **`env.reward_scale=1e-3`** — rescales the reward at the source; changes nothing
   about the optimal policy but makes raw episode returns readable.
3. **Clip latency inside the reward** — e.g. cap the SLA term at 400 ms; changes the
   optimal policy (a 3 s stall would become as bad as a 0.4 s stall). Use only if you
   deliberately want a softer SLA.

Do not mix 1 and 2 without re-tuning `learning_rate`; comparing them in TensorBoard is
the intended exercise.

### Q4 — Reading TensorBoard

```bash
tensorboard --logdir logs/tb --port 6006
# then open http://localhost:6006
```

| scalar | healthy behaviour |
|---|---|
| `rollout/ep_rew_mean` | rises quickly, then plateaus (normalised scale) |
| `train/approx_kl` (PPO) | < 0.05; if it explodes, lower `learning_rate` |
| `train/explained_variance` | > 0.5 means the critic understands the returns |
| `train/entropy_loss` | should shrink slowly, not collapse to 0 |
| `eval/mean_reward` | the number to compare across runs |
| `autoscaling/mean_pods` | should settle near 3–4 (baseline is 3.50) |
| `autoscaling/sla_violation_rate` | target: below the baseline's 0.0038 |
| `autoscaling/p95_latency_ms` | target: below 200 ms |

Compare runs by `tb_log_name` (`ppo_42`, `ppo_42_1`, ...): each invocation creates a
new numbered run, so you never overwrite an experiment.

---

## 8. Evaluation harness

```bash
python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo
```

What it does:

1. Rolls out the HPA baseline and the RL policy over the **full 8,640-step day**,
   deterministically, from `start_index=0`.
2. Applies the matching `*_vecnormalize.pkl` manually when feeding observations to the
   model (so evaluation matches training exactly). `--vecnormalize PATH` overrides the
   lookup; if the file is missing you get an explicit warning.
3. Writes `utils/metrics.csv` and prints the comparison table:

```
      Policy  Total Cost ($)  SLA Violation Rate (%)  Average Latency (ms)  P95 Latency (ms)  Avg Pods  Scale Actions  Total Reward
HPA baseline           10.07                    0.38                 38.60             49.21      3.50           1865   -11027157.4
 RL agent (PPO)          ...                     ...                   ...               ...       ...            ...           ...
RL vs HPA baseline: cost +x.xx% | SLA violation rate -y.yy percentage points
```

4. Writes two 150 DPI charts:

| file | what to look for |
|---|---|
| `latency_comparison.png` | latency vs the dashed 200 ms SLA line. The baseline spikes to 3,000 ms on short bursts; a good agent stays under 200 ms. **Log y-axis** so both 20 ms and 3,000 ms are readable. |
| `pod_provisioning_comparison.png` | pods per policy (step plot) against the incoming RPS on the right axis. A good agent tracks the *ramp*, not the CPU aftermath. |

Useful flags:

```bash
python utils/evaluate.py --no-baseline                 # RL only
python utils/evaluate.py --hpa-cooldown 3              # calmer baseline
python utils/evaluate.py --pod-cost-per-hour 0.24      # your cloud price
python utils/evaluate.py --outdir /tmp/bench --seed 7  # custom output dir
```

`Total Cost ($)` is a reporting-only metric: `pods × 10 s × $0.12/pod-hour`. The
training reward uses the specification's `1.5 × pods` term, not dollars.

For a design-level view — what a policy family is worth, and what a reward actually prices —
run `python utils/design_probe.py` and read [docs/DESIGN.md](docs/DESIGN.md). It trains
nothing: it replays the same physics to produce a ladder (constant 9 → trailing-max tracker →
20 s oracle → perfect-foresight ceiling) so you can tell a bad policy apart from a bad reward
before spending another 200k steps.

---

## 9. Design decisions and assumptions

| decision | rationale |
|---|---|
| July 13, 1995 as the replay day | busiest day in the log (134k requests); best stress test available |
| 10-second bins | matches the spec; 8,640 steps is a manageable episode |
| interpolate 402 empty bins | they are logging outages, not real zero-traffic periods |
| rescale ×61.54 to 480 RPS peak | 1995 volume (7.8 RPS) never stresses a 60 RPS/pod fleet |
| `mu_pod = 60` RPS | spec value; keeps pods in 1–11 over the day |
| latency capped at 3,000 ms | spec overload value; also keeps the observation inside `Box` bounds |
| SLA violation = bin latency > 200 ms | latency is per-bin, so the violation *rate* is measured over time bins, not over individual requests |
| RAM = EMA of `0.20 + 0.60·cpu` | RAM is undefined in the spec; this keeps it observable and Markov |
| `terminated=False`, `truncated=True` | autoscaling is a continuing task |
| `initial_pods="auto"` (~80 % CPU target) | avoids an artificial cold-start penalty at random episode offsets |
| reward exactly as specified | the 392k SLA term is intentional; normalisation is handled in the learner |
| eval env frozen + stats synced from train env | keeps evaluations comparable while the training statistics evolve |
| reward/cost/churn weights exposed as constructor args | lets you ablate `cost_weight`, `sla_weight`, `churn_weight` without editing the env |

Known limitations (good material for a report):

* Queueing is a deterministic fluid approximation — no burstiness, no queue length,
  no request-level tail latency.
* Pods start serving instantly; real clusters need 10–60 s to schedule + warm up.
* One homogeneous service; no node capacity, no bin-packing, no spot pricing.
* The trace is rescaled 1995 traffic: shapes are realistic, absolute volumes are not.
* Reward normalisation changes the effective reward scale, so `Total Reward` in
  `metrics.csv` is not directly comparable to a run trained with `norm_reward=False`.

---

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `FileNotFoundError: Workload trace not found` | run `python data/generator.py` first |
| `No log lines matched date` | use `--date 13/Jul/1995` (format `DD/Mon/YYYY`) |
| `ModuleNotFoundError: stable_baselines3` | activate the env: `conda activate cloudrl` |
| `NotImplementedError: Step n ...` | expected in the workbook: implement that function, then `--self-check` |
| `--self-check` says "blocked - step 1 has to pass first" | fix steps in order; each check builds on the previous one |
| policy performs terribly after loading | missing `*_vecnormalize.pkl`; pass `--vecnormalize` explicitly |
| `ep_rew_mean` stays at −1e8 | reward not normalised; keep `norm_reward=True` or lower `--learning-rate` |
| training runs longer than `--timesteps` | expected: PPO completes the current rollout, so steps round up to `n_steps × n_envs` |
| plot window does not open | evaluation forces the headless `Agg` backend and saves PNGs |
| evaluation feels slow | it is 8,640 steps × 2 policies ≈ a few seconds; lower `--eval-freq` in training instead |
| TensorBoard shows no scalars | check `logs/tb/<run>/` exists and point `--logdir` at `logs/tb` |

---

## 11. Ideas to extend

The honest fix for lesson 1 — deliberately **not** shipped in the default environment,
because it would remove the bar the workbook is built around. It *is* implemented as opt-in
wrappers and measured end to end in **[docs/DESIGN.md](docs/DESIGN.md)**:

```bash
python utils/design_probe.py     # reproduces every number quoted here
```

The four items below, in short: a linear-and-capped SLA penalty plus churn charged on the
*requested* delta moves the price of one 10 s breach from 726 hours of a spare pod to 21
hours — 1.9 hours once the ramp is capped — and an observation built on headroom + a
multi-minute trailing peak lets a 10-line heuristic beat constant 9 pods **by 48 % with zero
violations** (−60,088 vs −116,640; a 30 s peak gets only 18 % and still breaches 4 times) —
and by 61 % with a 20 s lookahead (−44,996, zero violations). The shipped env still scores
the old objective, so §6.1 stays valid and remains the "before" table.

* Penalise the *requested* delta rather than the applied one, so `+2` at `max_pods` is
  no longer a free no-op, and the "always push up" policy stops being stable.
* Add a `capacity_headroom = (pods × mu_pod − rps) / (pods × mu_pod)` observation
  channel: the agent currently has to infer headroom from utilisation, which is exactly
  the signal that makes the 8→9 cliff invisible until it is crossed.
* Add a *linear* SLA term (`sla_weight_linear × max(0, latency − SLA)`) next to the
  quadratic one: the quadratic term makes a 3 s stall worth 392,000 and a 5 ms miss
  worth 1.25, so "one bad second" and "a slightly laggy day" are not comparable.
* Curriculum: start episodes at 9 pods (or at the analytically safe fleet) so the
  agent sees the non-violating region early instead of discovering it by accident.

All four are implemented in `env/design_wrappers.py` (`LinearSLAReward`, `HeadroomObs`) and
measured in [docs/DESIGN.md](docs/DESIGN.md) §3–§6 — the numbers there supersede the
hand-waving here. The measured version of the curriculum item is *peak-heavy episode
starts* rather than a fixed 9-pod start: the 09:49 spike carries more reward than the whole
rest of the day and appears in only 17 % of uniformly sampled 4 h episodes.

Beyond the reward design:

* Add a pod warm-up delay (`pods_pending`) and make the action space include a
  "reserve" action — much harder, more realistic credit assignment.
* ~~Train on multiple days from the log (Jul 1–28) and test on a held-out day to
  measure generalisation across daily shapes.~~ **Done** — `utils/build_days.py`
  materialises all 28 days and freezes a chronological 16/6/6 split, the
  environment samples a day per episode (`--train-days` + `--split-json`), and
  every arm is scored on the six held-out days. Results: `results/summary.md`,
  figures in `plots/`, method and limits in [docs/DESIGN.md](docs/DESIGN.md) §12.
* Replace the synthetic latency with a queue simulation (`simpy`) and log real
  request-level p99.
* Action masking for `pods == 1` (cannot scale below) to remove wasted actions.
* Compare against a predictive autoscaler (linear trend / ARIMA) as a stronger
  baseline than threshold HPA.
* Try SAC or TD3 with a continuous "target pod count" action.

---

## 12. Data

`raw/access_log_Jul95` — NASA Kennedy Space Center HTTP server log, July 1995
(publicly available log research dataset). It is **excluded from git** (196 MB);
drop it into `raw/` or use `--source synthetic`.

The synthetic generator is deterministic for a given `--seed`, so all results in
this repository can be reproduced without the raw log — but the reported baseline
numbers were produced from the real July 13 replay.
