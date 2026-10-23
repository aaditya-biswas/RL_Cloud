# Walkthrough: build the autoscaler training loop yourself

This is the workbook that goes with `agents/train_rl.py`. Six sessions, roughly one
afternoon of work plus one training run. By the end you will have written the four
functions that turn a Gymnasium environment into a trained, reproducible PPO policy —
and you will have found out why "trained" is not the same as "good".

**You are given** the environment (`env/cloud_env.py`), the rule-based baseline
(`agents/baseline_scaler.py`), the evaluator (`utils/evaluate.py`), two callbacks, the
reward plumbing and the CLI.

**You write** four functions, each with a `CONTRACT` block in its docstring:

| Step | Function | Session |
|---|---|---|
| 1 | `build_vector_env` — environment wrappers and normalisation | 1 |
| 2 | `build_model` — PPO/DQN instantiation | 2 |
| 3 | `make_callbacks` — evaluation, checkpoints, metrics | 3 |
| 4 | `save_artifacts` — model + statistics, saved as a pair | 4 |

**You are measured on** Session 5, not on the self-check. The self-check only proves your
plumbing works; the acceptance test is whether your policy beats a static 9-pod
deployment on the default objective. The reference solution itself does not pass that
test — that is the honest state of the project, and it is the most interesting thing in
it.

Vocabulary used throughout: *step* = one 10 s decision, *episode* = 1,440 steps (4 h) for
training and 8,640 steps (24 h) for evaluation, *violation* = a step whose latency
exceeds the 200 ms SLA.

---

## Session 0 — Run the sandbox and meet the objective (~30 min)

Nothing to implement yet. The goal is to be able to predict what the environment will
pay for a policy before you train one.

```bash
conda activate cloudrl
python data/generator.py                       # trace, ~10 s
python agents/train_rl.py --dry-run            # check_env + 8 hand-checked steps
python utils/evaluate.py --no-baseline --constant-pods 8,9,15
```

The last command runs three *fixed-fleet* baselines: hold the pod count constant all day.
You should get exactly:

```
 constant 8 pods           23.04   0.01   18.13   19.28   8.0   0   -495680.0
 constant 9 pods           25.92   0.00   17.35   18.40   9.0   0   -116640.0
constant 15 pods           43.20   0.00   16.26   16.53  15.0   0   -194400.0
```

**Exercise 0.1 — predict before you run.** With `mu_pod = 60` rps per pod,
`latency = 1000 / (mu_total − rps) + 15` (and 3,000 ms at or above capacity), reward
`−1.5·pods − 0.05·max(0, latency − 200)² − 0.5·|Δpods|`, and a trace that averages 97 rps
and peaks at 480 rps:

1. How many pods are needed to serve the 480 rps peak without violating the SLA?
2. What does the answer to (1) say about constant 8 pods — how many violations, and how
   much does each one cost?
3. Why is constant 9 pods *cheaper* in reward than constant 15 pods?

<details><summary>Answers</summary>

1. Nine. Capacity is `60 × pods`; 8 pods = 480 rps = exactly the peak, so the latency
   formula divides by zero and the env returns 3,000 ms. 9 pods = 540 rps gives
   `1000/60 + 15 ≈ 32 ms` at the spike.
2. Exactly one violation, at the single trace bin that reaches 480 rps. It costs
   `0.05 × (3000 − 200)² = 392,000`, which is why constant 8 scores −495,680 while
   constant 9 scores −(1.5 × 9 × 8,640) = −116,640.
3. Both have zero violations, so the only difference is pod-hours: 15 pods cost
   `1.5 × 15 × 8,640 = 194,400` versus 116,640. **The reward-optimal fleet is the
   cheapest fleet that does not violate the SLA** — for a constant policy. Keep that
   sentence; Session 5 is about whether a learned policy can do better than a constant
   one.

</details>

**Checkpoint.** `python agents/train_rl.py --self-check` prints four `[FAIL]` blocks.
That is correct at this stage — it is your to-do list.

**Knowledge check.** Why does the evaluator use `norm_reward=False` and the default
weights rather than whatever the trainer used?

<details><summary>Answer</summary>

Because the question is "how does this policy score on the objective we specified", not
"how well did it optimise the objective it was given". A policy trained with
`--cost-weight 20` is scored with `cost_weight=1.5`, so reward-normalised values, scaled
rewards and tuned weights cannot make a run look better than it is. Report the same
number for every row or the table is meaningless.

</details>


---

## Session 1 — Step 1: `build_vector_env` (~1.5 h)

### What you are building

```
CloudAutoscalingEnv → Monitor → DummyVecEnv(n_envs) → VecNormalize
```

Four wrappers, each earning its place:

| wrapper | why |
|---|---|
| `Monitor` | records episode return/length and merges env `info` into the rollout (the `MetricsCallback` reads it) |
| `DummyVecEnv` | SB3 requires a vectorised env; `n_envs=4` gives four independent 4 h episodes per rollout |
| `VecNormalize(norm_obs=True)` | RPS is 6–480, CPU/RAM are 0–1, pods are 1–15; without it one channel owns the gradient |
| `VecNormalize(norm_reward=...)` | the overload penalty is 392,000 against a pod cost of 1.5 — off for evaluation, on for training |

The subtle part is that **the same function must build two different environments**:

| | training env | eval env |
|---|---|---|
| `episode_steps` | `args.episode_steps` (1,440) | `None` → the whole 8,640-step day |
| `random_start` | `True` (random hour) | `False` (always hour 0) |
| `n_envs` | `args.n_envs` (4) | `1` |
| `norm_reward` | `True` | `False` |
| `training` flag | `True` | set to `False` by `main` — frozen statistics |

That is why every one of those is a *parameter*, not a constant. If you hard-code
`norm_reward=True` or `episode_steps=args.episode_steps`, the self-check will tell you;
if you hard-code them and skip the self-check, you will train happily and then evaluate
on a distribution you never trained for.

Seeding: env `rank` gets `seed + rank`. Identical seeds across parallel envs would give
you four copies of the same episode, which is worse than `n_envs=1`.

### Do it

1. Open `agents/train_rl.py`, read the `STEP 1 - CONTRACT` block.
2. Implement `build_vector_env` only. Leave the other three raising.
3. Run `python agents/train_rl.py --self-check`. Step 1 must go `[PASS]`; steps 2–4 stay
   blocked, which is expected at this point.

### When it fails

| self-check message | cause |
|---|---|
| `build_vector_env must return a VecNormalize, got DummyVecEnv` | you returned the vec env before wrapping it |
| `each sub-env must be wrapped in Monitor` | you passed the raw env into `DummyVecEnv` |
| `norm_reward is ...` on the eval env | you hard-coded `norm_reward=True` |
| `the evaluation env must run the full day` | `episode_steps` is not forwarded (`None` must fall through to the env) |
| `env cost_weight=... but the CLI asked for ...` | you forgot to forward the reward weights |
| `normalised observations exceed clip_obs=10.0` | you built `VecNormalize` but passed `norm_obs=False`, or normalised the raw env twice |

### Exercise 1.1 — the flat line is the finding

The sandbox ships the author's 200k-step model in `models/`. Evaluate it, then audit what
it actually does:

```bash
python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo \
    --constant-pods 9
```

The RL row is 15.00 average pods, P95 16.53 ms, 7 scale actions, reward −194,344 — i.e.
*pinned at `max_pods` for the entire day*, slightly cheaper than constant 15 only because
it starts at 1 pod and climbs. `utils/pod_provisioning_comparison.png` shows a flat line
at 15. Before you call it a plotting bug, count the actions the greedy policy emits:

```python
import numpy as np
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path.cwd()))  # run this from the repository root
from env.cloud_env import CloudAutoscalingEnv
from utils.evaluate import resolve_vecnormalize_path

model_path = Path("models/ppo_cloud_autoscaling.zip")
model = PPO.load(model_path, device="cpu")
stats = VecNormalize.load(str(resolve_vecnormalize_path(model_path, None)),
                          DummyVecEnv([CloudAutoscalingEnv]))
stats.training = False

env = CloudAutoscalingEnv(episode_steps=None, random_start=False)
observation, _ = env.reset(seed=42)
actions, done = [], False
while not done:
    action, _ = model.predict(stats.normalize_obs(observation), deterministic=True)
    actions.append(int(np.asarray(action).reshape(-1)[0]))
    observation, _, terminated, truncated, _ = env.step(actions[-1])
    done = terminated or truncated

values, counts = np.unique(actions, return_counts=True)
print(dict(zip(values.tolist(), counts.tolist())))
# {2: 98, 3: 1956, 4: 6586}   ->  2 = hold, 3 = "+1 pod", 4 = "+2 pods"
```

**8,542 of the 8,640 decisions ask for more pods** (6,586 × `+2`, 1,956 × `+1`). The fleet
reaches the 15-pod cap at **step 6** and never leaves it: 8,634 steps are spent at 15 pods,
and 8,535 of the requests are clipped to no-ops by `max_pods`. The "7 scale actions" in the
table are the climb. The flat plot is the policy saying "up" 8,542 times to a ceiling that
stops answering.

1. Why is "always `+2`" a stable strategy under this reward? (Hint: what does the action
   *cost* when the fleet is already at the cap?)
2. What would the reward have to penalise to make that strategy unstable?

<details><summary>Answers</summary>

1. The churn penalty is `0.5 × |delta_pods|` on the **clipped** delta. At `max_pods` the
   clip makes it 0, so `+2` costs exactly nothing; meanwhile any downward action risks a
   392,000 violation. An argmax that always pushes up is risk-dominant, and once it is at
   the cap it is also free.
2. The *requested* delta, before clipping — or a hard budget on scale requests. That is
   the first item in README §11 "Ideas to extend", and it is deliberately not fixed here:
   fixing it removes the bar that Session 5 is built around.

</details>

### Exercise 1.2 — the 379,040-point cliff

Use the formulas from Session 0 to fill in the blanks, then check them against the
constant rows in `utils/metrics.csv`:

| fleet | pods/day | pod cost | violations | violation cost | return |
|---|---|---|---|---|---|
| constant 8 | 8 × 8,640 | 103,680 | 1 | 392,000 | −495,680 |
| constant 9 | 9 × 8,640 | 116,640 | 0 | 0 | −116,640 |

The difference is **379,040 reward points for a 12,960 cost increase** ($1.56 for the
day). One extra pod, one avoided saturation second. Before you train anything, ask
yourself which side of that cliff a policy that has never seen the 3,000 ms penalty would
land on — and how it could ever learn the difference, given that both 8 and 9 pods look
fine for 8,639 of the 8,640 steps.

**Checkpoint.** `--self-check` shows `[PASS] Step 1`, and you can explain why the training
env normalises rewards while the eval env does not.

**Knowledge check.** The evaluation env is created with `n_envs=1` and then `main` sets
`eval_env.training = False`. What would go wrong if you skipped that line?

<details><summary>Answer</summary>

`VecNormalize` would keep updating `obs_rms` from the evaluation episodes, so the
statistics used for normalisation would drift toward full-day observations and the policy
would be evaluated on a moving target. `SyncVecNormalizeCallback` (step 3) overwrites the
eval statistics with the training ones before each evaluation anyway; freezing the eval
wrapper is what makes that copy the *only* source of truth.

</details>

---

## Session 2 — Step 2: `build_model` (~45 min)

### The decision

| | PPO | DQN |
|---|---|---|
| policy | on-policy, stochastic | off-policy, deterministic + ε-exploration |
| replay buffer | no | yes (100k transitions) |
| data efficiency | lower per env step | higher per env step |
| stability on this reward | high | medium — TD targets are built from 392,000-scale returns unless rewards are normalised |
| knobs that matter | `learning_rate`, `n_steps`, `n_epochs` | `learning_starts`, `exploration_fraction`, `target_update_interval` |

Start with PPO. It is the forgiving choice on a reward with a 392,000 spike, and it is
what the reference rows in README §6 use. Move to DQN only after you have a PPO baseline
to compare against, otherwise you will not know which change did what.

Two constraints that are easy to miss:

* **Build order.** `build_model` must receive the `VecNormalize`, not the raw env. SB3
  reads the observation/action spaces from the object it is handed, so building the model
  before the wrappers compile is a silent mismatch.
* **`n_steps × n_envs ≥ batch_size`.** PPO needs at least one full batch per rollout.
  `parse_args` already rejects the combination with a clear error; that guard is your
  friend, not an obstacle. With the defaults (2,048 × 4) one rollout is 8,192 env steps,
  which is why a `--timesteps 200000` run actually stops at 204,800.

### Do it

Implement `build_model`, run `--self-check`, and confirm steps 1–2 pass. The check
verifies the class matches `--algo`, both spaces match the env, PPO's `n_steps` /
`batch_size` / `gamma` were wired through, and a TensorBoard log target exists.

| self-check message | cause |
|---|---|
| `build_model returned DQN for --algo ppo` | you ignored `args.algo` |
| `model observation space ... does not match the env` | you passed the raw env instead of the vec env |
| `tensorboard_log is empty` | you dropped `tensorboard_log` — the run would be unreadable |
| `PPO n_steps=2048, expected 1024` | you hard-coded it instead of using `args.n_steps` |

**Checkpoint.** `--self-check` passes steps 1–2. `python agents/train_rl.py --timesteps 8192`
now trains for one rollout (steps 3–4 still raise, so it will stop before saving — that is
expected).

**Knowledge check.** `--n-envs 8 --batch-size 2048` is accepted, `--n-envs 1 --batch-size
4096 --n-steps 2048` is not. Why?

<details><summary>Answer</summary>

PPO collects `n_steps` transitions per env before it computes one gradient step, i.e.
`n_steps × n_envs` samples per rollout. A batch of 4,096 cannot be drawn from a rollout of
2,048, so SB3 would either error or silently pad. The default 4 × 2,048 = 8,192 gives four
full batches of 2,048. Raising `--n-envs` also makes each *call* to the callback cheaper to
schedule: `EvalCallback` counts per-env steps, hence the `// n_envs` in step 3.

</details>

---

## Session 3 — Step 3: `make_callbacks` (~1 h)

### The two ordering traps

`make_callbacks` is where a run that trains fine still produces meaningless evidence. Both
traps are silent — nothing crashes, the numbers are just wrong.

**Trap 1 — callback cadence is per env, not global.** `EvalCallback` and
`CheckpointCallback` count `n_calls`, which advances once per env step of the *vec* env. With
`n_envs=4`, `eval_freq=10_000` fires every 40,000 environment steps. Divide by the number
of envs (`args.eval_freq // n_envs`, floored at 1) or your "every 10k steps" evaluation
happens every 40k.

**Trap 2 — sync must run before eval.** `SyncVecNormalizeCallback` copies `obs_rms` /
`ret_rms` from the training env into the frozen eval env. If `EvalCallback` runs first, the
evaluation is scored with the statistics from the *previous* sync — one full evaluation
interval stale. Early in training that is the difference between "learning" and "collapsed",
and the eval curve will disagree with the final `utils/evaluate.py` numbers. Order in the
`CallbackList` is execution order:

```
[SyncVecNormalizeCallback, EvalCallback, CheckpointCallback, MetricsCallback]
```

### Do it

Implement `make_callbacks`; `--self-check` verifies the list type, the four entries, the
sync/eval order, `EvalCallback.eval_env is eval_env` (identity — pass the object you were
given), `deterministic=True`, the divided `eval_freq`, and the checkpoint path/prefix.

| self-check message | cause |
|---|---|
| `expected exactly one EvalCallback, found 0` | you returned a bare callback instead of a `CallbackList` |
| `SyncVecNormalizeCallback must run *before* EvalCallback` | list order is reversed (trap 2) |
| `eval_freq=10000; use args.eval_freq // n_envs (=2500)` | trap 1 |
| `EvalCallback must be given the frozen eval env you built` | you constructed a second eval env inside the callbacks |

**Checkpoint.** `--self-check` passes steps 1–3 and a short run
(`--timesteps 16384 --eval-freq 8192`) writes `logs/eval/evaluations.npz` plus at least one
checkpoint in `models/checkpoints/`.

**Knowledge check.** You have two policies with identical `eval/mean_reward` in TensorBoard
but they score differently in `utils/evaluate.py`. Which callback ordering bug would explain
that, and why does it only show up in one direction?

<details><summary>Answer</summary>

Sync running *after* eval: TensorBoard's `eval/mean_reward` is computed on observations
normalised with stale statistics, while `utils/evaluate.py` loads the final
`*_vecnormalize.pkl` and normalises correctly. The TensorBoard number can be better or worse
depending on how far the statistics moved, but the discrepancy grows over a run — early
evaluations look almost right, late ones do not.

</details>

---

## Session 4 — Step 4: `save_artifacts` (~30 min)

### Why a model is two files

The policy was trained on observations normalised by a *specific* `VecNormalize`
instance. Its running mean/variance (`obs_rms`) is as much a part of the policy as the
weights: feed raw observations to a network that learned on z-scores and every action is
garbage. So the contract is a pair, with names `utils/evaluate.py` can find on its own:

```
models/ppo_cloud_autoscaling.zip
models/ppo_cloud_autoscaling_vecnormalize.pkl
```

`resolve_vecnormalize_path()` looks for `<model_stem>_vecnormalize.pkl` next to the model,
then falls back to the legacy `vecnormalize.pkl`. If neither exists it prints a warning and
evaluates with raw observations — a run that "trained fine" and then scores like a random
policy is almost always this.

`save_artifacts` also takes `models_dir` (default `models/`) so the self-check can save
into a temporary directory instead of polluting your `models/`.

### Do it

Implement `save_artifacts`, run `--self-check` — it saves, then reloads both files with
`PPO.load` / `VecNormalize.load` and checks the statistics still match the env.

| self-check message | cause |
|---|---|
| `missing ..._vecnormalize.pkl - a model without its VecNormalize statistics cannot be evaluated` | you saved only the model |
| `save_artifacts returned model path ...` | wrong filename; the evaluator's auto-detect expects `<algo>_cloud_autoscaling` |
| `step 4 check crashed: ...` | usually a missing `mkdir(parents=True, exist_ok=True)` |

### Exercise 4.1 — break it on purpose

With the shipped model, hide the statistics and re-evaluate:

```bash
mv models/ppo_cloud_autoscaling_vecnormalize.pkl /tmp/stats.pkl
python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo --no-baseline
mv /tmp/stats.pkl models/ppo_cloud_autoscaling_vecnormalize.pkl
```

You get a warning (`no VecNormalize statistics found`) and a different world: the same
weights now average **1.00 pod**, violate the SLA on **67.5 %** of steps, and return
**−2.1e9**. Nothing crashed, nothing complained about a broken model, and the number is
meaningless — which is exactly why the self-check reloads the pair instead of just checking
that two files exist.

**Checkpoint.** `--self-check` reports **4/4** and exits 0. Now do one real run:

```bash
python agents/train_rl.py --algo ppo --timesteps 200000
```

**Knowledge check.** A colleague sends you `my_agent.zip` and says it "gets 15 pods, P95
16.5 ms". What single question determines whether you can reproduce that number?

<details><summary>Answer</summary>

"Where is the `_vecnormalize.pkl` that was saved with it?" Without the matching statistics,
the observation normalisation is unknown and the policy is not reproducible — not badly
performing, *undefined*. The same applies to the reward weights it was trained with, but that
only changes what it optimised, not whether it runs.

</details>

---

## Session 5 — Acceptance: beat a constant 9-pod fleet (~2 h, mostly training)

### The bar

Scored on the **default** weights, over the full day, by `utils/evaluate.py`:

```bash
python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo \
    --constant-pods 9
```

| | Total Reward | Cost | SLA violations | Avg pods |
|---|---|---|---|---|
| bar: constant 9 pods | **−116,640** | $25.92 | 0.00 % | 9.00 |
| your policy | must be **> −116,640** | | 0.00 % | |

"Better" means a strictly higher default-objective return with zero SLA violations. Churn
is already priced in at 0.5 per action, so there is no separate action budget.

Two calibration points, so you know where the ceiling is:

* **Perfect foresight** — an oracle that resizes every step to the smallest SLA-satisfying
  fleet (a dozen lines of numpy over the trace) averages 2.22 pods and 4,185 scale actions,
  scoring **−31,254**. That is the practical ceiling: 1–3 pods at night, 9 pods for the
  single 480 rps bin.
* **The author's best** — `--cost-weight 20`, 100k steps, 11.35 pods, 53 actions, scores
  **−147,175** on the default objective. Still worse than doing nothing. The default 200k
  run scores −194,344.

So the honest statement is: **this sandbox's reference solution does not pass its own
acceptance test.** Beating a static 9-pod fleet means scaling *down* to 1–3 pods for most
of the day and *up* for the spike — while the reward makes "pin at the cap" locally optimal,
which is exactly the policy PPO finds.

### A plan of attack

Work in this order; each step is a controlled experiment, not a guess.

1. **Verify the plumbing.** 4/4 self-check, then `--timesteps 16384`, and confirm
   `logs/eval/evaluations.npz` exists. If the plumbing is wrong, no hyperparameter will save
   you.
2. **Reproduce the reference row.** `--algo ppo --timesteps 200000`. If you do not land near
   15 pods / −194k, your step 3 or 4 differs from the reference — find out how before
   changing anything else.
3. **Change one knob at a time**, always re-scoring with `utils/evaluate.py` on the default
   weights (never trust `eval/mean_reward` once `--cost-weight` is not 1.5 — it is a
   different objective):

   | change | expected effect |
   |---|---|
   | `--cost-weight 20…40` | fleet shrinks toward 9; watch for flapping as you approach the break-even weight ≈ 45 |
   | `--gamma 0.995` | horizon 33 min instead of 17 min; better pre-emption of ramps |
   | `--episode-steps 720` | more diverse starting hours per timestep |
   | `--n-envs 8` | better coverage of the daily profile per rollout |
   | `--learning-rate 1e-4` | slower, less likely to collapse into the pin-at-cap basin |

4. **Read the three autoscaling scalars, not just reward.** `autoscaling/mean_pods` should
   fall below 9, `autoscaling/sla_violation_rate` must stay 0, `autoscaling/p95_latency_ms`
   should stay under 200. A run that improves reward by accepting violations has not passed.
5. **Write it down**, pass or fail, with the command and the matching `utils/metrics.csv`
   row.

### What a passing result looks like

Add it to README §6.2 with the command. The shape of the row, with placeholder numbers:

```
| PPO, 200k, --cost-weight 30 | 21.40 | 0.00 | 17.10 | 18.20 | 7.43 | 214 | -108900.0 |
```

(Those values are illustrative, not a measured run — replace them with yours.)

### If you cannot beat it

That is a legitimate outcome, and the more instructive one. Write a paragraph answering:

* Did the policy learn the peak (one bin at 480 rps, index 3,535), or only the daily
  average? Compare `utils/pod_provisioning_comparison.png` against the trace.
* Did it spend the day over-provisioned (safe, expensive) or under-provisioned (cheap,
  violating)?
* Which reward change from README §11 would have made *your* failure mode impossible —
  requested-delta churn, a headroom observation, a linear SLA term, or a start-at-9
  curriculum? Argue it with the numbers you already have.

**Knowledge check.** You are allowed one change to the *environment* to make this problem
easier for a learner. Which do you pick, and what does it cost?

<details><summary>Answer</summary>

There is no free change. Every option in README §11 either removes the cliff (curriculum,
headroom observation), removes the free no-op that makes pinning stable (requested-delta
churn), or changes what "good" means (linear SLA term) — and any of them invalidates
comparison with the reference table in §6.1, because the objective is no longer the one those
rows were scored on. That trade-off — a better learner versus a comparable benchmark — is the
actual design decision in RL infrastructure work.

</details>

---

## Where to go next

* `README.md` §6.2 — put your numbers in, or they do not exist.
* `README.md` §11 — the reward redesign, if you want to fix the finding instead of beating it.
* `agents/solutions/train_rl_reference.py` — diff it against your file. The differences are
  your notes.
