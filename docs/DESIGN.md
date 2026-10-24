# Designing a reward that fits the traffic

Companion to [§11 of the README](../README.md#11-ideas-to-extend) and to Session 5 of the
[walkthrough](WALKTHROUGH.md). The workbook asks you to beat a static 9-pod deployment;
the reference PPO solution does not (it costs $43.19/day at 15 pods versus $25.92 for 9).
This document is the *why*, plus a concrete design that does.

Everything below is measured, not asserted. One command reproduces every number:

```bash
conda activate cloudrl
python utils/design_probe.py                      # the whole report, ~40 s
python utils/design_probe.py --section trace      # the traffic only
python utils/design_probe.py --reward v2 --section policies
```

The report replays the trace through the same physics as `env/cloud_env.py`
(`mu_pod = 60` rps, `latency = 15 + 1000 / (capacity - rps)` or 3,000 ms when full) and
prints three things: what the traffic does, what a reward prices, and what a policy can
be worth. The proposed reward and observation are implemented as opt-in wrappers in
`env/design_wrappers.py`, so the shipped environment, its 5-dim observation and the
existing `models/` checkpoints are untouched.

The short version, in four sentences:

1. The fleet that is actually needed swings from **1 pod at 03:00 to 10 pods at 09:49**
   and back, while the bill is paid all day — that gap is the only reason to use RL here.
2. The traffic jumps faster than the `±2 pods / step` actuator can follow (192 steps a day
   do), so **the agent has to remember a burst, not react to it**.
3. The stock reward prices one 10-second breach at **726 hours of a spare pod**, so the
   rational policy is "buy the whole cap and never look back" — which is exactly what PPO
   learned (15 pods, zero scale-downs).
4. Fix the price (linear, capped, churn on request) *and* give the agent headroom plus a
   multi-minute peak, and a **10-line heuristic** — buy `demand / 0.77` pods from the last
   8 minutes of traffic — already beats constant 9 by **48 % with zero breaches**; with a
   20 s lookahead it beats it by 61 %, still at zero violations.

---

## 1. Read the traffic before you touch the reward

`--section trace` prints this. Every design decision later is a response to one of these
three facts.

| window | mean RPS | peak RPS | pods needed at the peak (80 % util) |
|---|---|---|---|
| 00–03 | 54.9 | 141.5 | 3 |
| 03–06 | 42.6 | 141.5 | 3 |
| 06–09 | 121.9 | 381.5 | 8 |
| **09–12** | **198.9** | **480.0** | **10** |
| 12–15 | 133.1 | 443.1 | 10 |
| 15–18 | 106.8 | 301.5 | 7 |
| 18–21 | 56.0 | 227.7 | 5 |
| 21–24 | 60.4 | 190.8 | 4 |

**Fact 1 — the shape is a ramp, not noise.** Day mean 96.8 rps, p50 73.8, p95 246.2, and the
busiest bin is 480.0 at index 3,535 (09:49). Night traffic needs one pod; the morning ramp
needs ten. The demand-weighted optimum (the DP ceiling) averages **2.2–2.7 pods**
depending on how movement is priced, while a flat 9-pod deployment averages 9.00. That 4×
gap is the prize.

**Fact 2 — the actuator is slower than the traffic.** Step-to-step demand changes:
p50 24.6 rps, p90 73.8, p99 141.5, **max 387.7**. One pod serves 60 rps and the action space
moves at most 2 pods per step:

* 1,463 of 8,639 steps grow by more than one pod's worth,
* **192 steps grow by more than the ±2 limit can absorb**,
* the worst step needs 6.5 extra pods *in ten seconds*.

So a policy that only looks at the present is not merely suboptimal, it is **physically
unable to be safe**: `--section policies` shows a perfect "size it to today's usage"
tracker taking 16 breaches even after tuning its target utilisation.

**Fact 3 — the peak carries all the risk, the night carries all the cost.** One saturated
step at 09:49 costs 392,012 reward; holding 9 pods for the entire day costs 116,640, i.e. one
bin is worth 3.4× the whole day's capacity. The learning signal lives in *one bin out of
8,640* — an episode sampled uniformly from the day contains it only 17 % of the time. Keep
that in mind for §6.


---

## 2. Why the stock reward cannot work

The stock reward is

```python
reward = -(1.5 * pods + 0.05 * max(0, latency - 200) ** 2 + 0.5 * abs(delta_pods))
```

Four separate defects, in order of importance.

### 2.1 It is a step function, not a gradient

Substitute the latency model. The SLA is crossed at a fixed **utilisation**, not at a
particular load:

```
15 + 1000 / (capacity - rps) = 200   =>   capacity - rps = 5.405 rps
u* = 1 - 5.405 / 60 = 0.9099         =>   safe fleet = ceil(rps / 54.6)
```

Below `u*` the penalty is exactly zero; above it the penalty is quadratic in a quantity
that is hyperbolic in the pod count. At the 480 rps spike:

| pods | capacity | latency | stock penalty | reward that step |
|---|---|---|---|---|
| 8 | 480 | **3,000 ms** (full) | **392,012** | −392,024 |
| 9 | 540 | 31.7 ms | 0 | −13.5 |
| 10 | 600 | 23.3 ms | 0 | −15.0 |

One pod separates "worst possible outcome" from "best possible outcome". A value network
sees a 392,000 discontinuity between two adjacent, otherwise identical states, and the
advantage estimate for *scale up* is dominated by it on a handful of steps and by noise
elsewhere. The learned response is not "learn the pattern", it is "stay away from the
edge" — i.e. sit at `max_pods`. That is what the reference agent does: 15.00 average pods,
0 violations, 68 % worse than a flat 9.

### 2.2 The price is quoted in the wrong unit

`--section price` converts the penalty into the unit the agent actually trades in:
**pod-steps** (one pod for one 10 s step costs 1.5).

| reward | cost of one 10 s breach at the peak | in pod-steps | in hours of one spare pod |
|---|---|---|---|
| v1 (stock, quadratic) | 392,012 | 261,341 | **725.9 h** |
| v2 (linear) | 11,412 | 7,608 | **21.1 h** |
| v3 (linear + capped) | 1,012 | 675 | **1.9 h** |

"One bad ten seconds is worth a month of spare capacity" is not a design, it is a corner
solution. The interesting number is v2: a full saturation at the peak is priced like **21
hours of one spare pod** — still large, but two orders of magnitude below v1, and a *mild*
breach (a step that just touches the 200 ms line) costs 22 minutes. Capping the ramp (v3)
brings even the worst case down to 1.9 hours. What that buys is a reward whose gradient near
the threshold is usable: the agent is told "slightly late is bad, very late is worse",
instead of "slightly late and catastrophically late are both infinity".

### 2.3 Churn is charged on the wrong quantity

`delta_pods` in `env.step` is the *realised* change, so requesting `+2` while at
`max_pods` costs nothing. The 200k reference policy's action histogram over one full day is
`{hold: 98, +1: 1,956, +2: 6,586}` — it asks for more capacity on **98.9 %** of steps and
asks to scale down on **0 %**, and 8,535 of those requests are clipped no-ops. The
evaluator reports `Scale Actions = 7` because the agent hits the cap in the first minute and
then pushes against a wall. A policy that cannot express "down" cannot track traffic.

Charge the *request*: `churn_weight * abs(ACTION_DELTAS[action])`.

### 2.4 The observation hides the decision

`[cpu, ram, rps, latency, pods]` contains the answer, but not in a usable form:

* `ram` is a deterministic EMA of `cpu`, so 20 % of the input vector buys a lag and no
  information.
* `cpu` is `min(1, rps / capacity)`: the whole safe region lives in `[0, 0.91]` and **every
  overloaded state — one pod short or ten — reads exactly 1.0**. The dangerous region has no
  resolution at all, and it is the only region the reward cares about.
* `rps` is on an absolute 6–480 scale, so the distance between "8 pods' worth" and "9 pods'
  worth" is 60 units out of a 474-unit range. (The peak is not an outlier — measured
  `max |z|` over the day is 5.3 for `rps` and 2.1 for `cpu`, well inside `clip_obs=10`. The
  problem is resolution and relevance, not clipping.)
* `latency` is a **lagging** indicator: it reads 3,000 ms only after the fleet is already
  overloaded, and it is 3,000 for *every* overloaded state, so it cannot say how far under
  you are. It tells the agent what it should have done one step ago.

There is no history in the observation at all — no slope, no recent peak — so Fact 2 is not
merely hard, it is *unobservable*. That is the deeper reason a PPO run ends at 15 pods: the
policy cannot tell a 480 rps spike from a 480 rps plateau, so it buys insurance for both.


---

## 3. Reward v2: price the failure, not the catastrophe

`env/design_wrappers.py::LinearSLAReward` implements this:

```python
penalty = violation_cost * max(0.0, latency - soft_ms) / (hard_ms - soft_ms)   # ramp
penalty = min(penalty, cap_mult * violation_cost)                              # cap
reward  = -(cost_weight * pods + penalty + churn_weight * abs(requested_delta))
```

with `soft_ms=150`, `hard_ms=200`, `violation_cost=200`, `cap_mult=5`, `churn_weight=4`
and `cost_weight=1.5` unchanged.

`LinearSLAReward` ships with `cap_mult=5`, which is the **v3** column in the tables that
follow; pass `cap_mult=None` for the uncapped **v2**. v2 is quoted most often because it
isolates the effect of linearising the penalty; v3 is what you should train with.

Every constant is derived from the trace, not guessed:

| knob | value | why this value |
|---|---|---|
| `soft_ms` | 150 ms | 50 ms of warning band below the SLA, so latency has a gradient *before* the cliff instead of a binary death signal. |
| `violation_cost` | 200 | A step that just touches the SLA costs 133 pod-steps ≈ 22 min of a spare pod; a full 3,000 ms overload costs 7,608 (21 h, or 1.9 h with the cap). Still ≫ a one-step fix (≈ 6), so breaches remain worth avoiding. |
| `cap_mult` | 5 | A full outage (3,000 ms) pays 1,012 = 1.9 h of a spare pod, not 726 h. Bad, not apocalyptic — so the policy does not sacrifice the other 23 hours to dodge it. |
| ramp shape | linear | The marginal cost of the 250th millisecond equals the marginal cost of the 300th. |
| `churn_weight` | 4.0 | `+2` then `-2` costs 16 (≈ 10 pod-steps) — priced, but a genuine scale event (4–8) is not. The stock 0.5 made up-then-down cheaper than holding still. |
| churn base | requested | `+2` against `max_pods` is intent, and intent is what we want to stop. |

Two properties are worth checking explicitly, because they are what "calibrated" means.

**(a) The reward no longer changes the price of capacity.** Constant fleets with zero
breaches score *identically* under v1, v2 and v3 — the redesign only reprices failure:

| constant fleet | v1 (quadratic) | v2 (linear) | v3 (linear + cap) |
|---|---|---|---|
| 6 pods | −17,335,559 | −586,960 | −129,360 |
| 7 pods | −3,629,540 | −195,381 | −100,720 |
| 8 pods | −495,680 | −115,080 | −104,680 |
| **9 pods** | **−116,640** | **−116,640** | **−116,640** |
| 10 pods | −129,600 | −129,600 | −129,600 |
| 15 pods | −194,400 | −194,400 | −194,400 |
| best constant | 9 pods | 8 pods | 7 pods |

**(b) The dynamic range collapses.** Under v1 one step spans `1.5 → 392,012` (a factor of
260,000); under v2 it spans `1.5 → 1,012` (a factor of 675). The critic is fitting a bounded
function instead of one whose return is dominated by a single tail, so `norm_reward` becomes
a convenience rather than a precondition.

### The one thing v2 gets "wrong" on purpose

Under v2 the cheapest *constant* fleet is 8 pods, and under v3 it is 7. That is not a bug,
it is arithmetic: holding a 9th pod for 8,640 steps costs 12,960, while paying for the
single 09:49 breach costs 11,412. Every scalar reward has a region where breaching is
cheaper than provisioning — the question is *where* that region sits:

* If your SLO is an **error budget** ("≤ 0.1 % of steps may breach", i.e. ≤ 8.6 steps/day),
  4–5 breaches at 0.05 % is exactly right, and v2 is calibrated correctly.
* If your SLO is a **hard constraint** ("never"), a scalar reward cannot express it and you
  need a constrained objective — see §7. You can also raise `violation_cost` until 9 pods
  wins, and the probe says exactly how far: **V > 230** uncapped (57 × 230 = 13,110 > 12,960)
  or **V > 2,600** with the cap. But that walks the cliff back in through the front door —
  at V = 2,600 a mild breach is already 2,600, which is v1 with a different constant.

The comparison that matters is therefore **not** "which constant fleet does the reward
rank first", it is "which *policy* does the reward make cheapest" — because unlike a
constant fleet, a policy can pay ≈ 3 reward to fix a breach instead of 12,960. That is §8.


---

## 4. Observation v2: give the agent the pattern, in pod units

`env/design_wrappers.py::HeadroomObs` replaces the 5-dim vector with 11 features:

| # | feature | what it tells the agent |
|---|---|---|
| 0 | `cpu_utilization` | unchanged (still the HPA's own signal) |
| 1 | `ram_utilization` | unchanged, kept only so plots stay comparable |
| 2 | `demand_pods = rps / 60` | "this step needs 8.0 pods" — the decision in its own units |
| 3 | `min(latency / 200, 3)` | 1.0 = at the SLA, 3.0 = "600 ms or worse" |
| 4 | `pods / 15` | the actuator state, normalised |
| 5 | **`headroom_pods = (capacity - rps) / 60`** | "I have 1.0 pod of slack"; negative = already over |
| 6 | **`slope_pods`** | demand growth over the last 30 s, in pods *per step* — how long the ramp still needs |
| 7 | **`trailing_pods`** | max demand over the last 60 s — the fast peak |
| 8 | **`peak_pods_slow`** | max demand over the last 480 s — the regime peak (§8 shows this is the one that wins) |
| 9 | `impulse_pods` | demand change in the last step, in pods (the 387 rps jumps read as +6.5, clipped) |
| 10 | `fraction_of_day` | time-of-day (see the caveat below) |

The four bold ones are the design. `headroom_pods` replaces the flat 1.0 of `cpu` with a line
that keeps its gradient across the overload region; `slope_pods` is expressed in "pods per
step", the same unit as the action, so a (`slope`, `headroom`) pair directly implies an
action; and the two peaks cover the two horizons that matter — 60 s for "this burst is still
growing", 480 s for "this is the level I must hold through the next dip". `latency` is capped
at 3× SLA because beyond 600 ms "very late" and "dead" are the same decision.

**Why two peaks and not one.** The single best causal controller in §8 provisions to the
8-minute peak; the 30 s peak alone leaves 4 breaches and 36 % less reward (−94,797 vs
−60,088). A feed-forward MLP cannot integrate 8 minutes out of a 60 s window, at any width —
the information is simply not in its input. If you use a recurrent policy, one peak is
enough; if you use `MlpPolicy` (the default here), supply both.

Wire it in by wrapping inside `build_vector_env` — the rest of the trainer needs no change:

```python
from env.design_wrappers import HeadroomObs, LinearSLAReward

def build_vector_env(args, episode_steps, random_start, seed, n_envs=None, norm_reward=True):
    def make(rank: int):
        env = CloudAutoscalingEnv(
            trace_path=args.trace, episode_steps=episode_steps,
            random_start=random_start, seed=seed + rank,
        )
        env = LinearSLAReward(env)   # reward v2; supersedes --cost/sla/churn-weight
        env = HeadroomObs(env)       # 5-dim -> 11-dim observation
        return Monitor(env)

    venv = DummyVecEnv([lambda rank=rank: make(rank) for rank in range(n_envs or args.n_envs)])
    return VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                        clip_obs=10.0, gamma=args.gamma)
```

Three practical notes, all verified:

* **The 200k checkpoint will not load** — its observation space is `(5,)`. This is a new
  MDP, not a new feature: retrain, and keep the old model as the "before" row.
* **`--self-check` still passes**, because it compares against
  `train_env.observation_space.shape` rather than a hard-coded 5 (see `agents/self_check.py`).
  A 200-step PPO smoke run over two envs with both wrappers trains and predicts cleanly.
* **Every feature stays inside its declared bounds** (raw `max |obs| = 3.00`, the latency
  cap), and the largest z-score over a full day is **5.25 σ** (`demand_pods`) against
  `clip_obs=10`; the slow peak sits at 3.07 σ. That is not the selling point — the selling
  point is resolution: the stock `cpu` is a flat 1.0 for every overloaded state, while
  `headroom_pods` keeps a usable slope there.

**Caveat on `fraction_of_day`.** With one trace replayed, that feature is the trace index in
disguise: the agent can memorise "at 09:49 buy 10 pods" instead of learning the ramp. Two
honest ways to keep it:

* rotate the trace per episode (`start_index` drawn anywhere, and let the index wrap by
  replaying the day twice), so the same demand level appears at many times of day;
* or drop the feature and let `trailing_pods`/`slope_pods` carry the timing.

Dropping it makes the model strictly more transferable — which is what §6 measures.


---

## 5. The action space is not the problem; memory is

It is tempting to fix Fact 2 by letting the agent jump more than two pods. `python
utils/design_probe.py --section actuator` makes that trade-off visible under reward v2
(best u of 0.70–0.96 per row):

| `max_step` | current step | trailing 30 s | trailing 480 s |
|---|---|---|---|
| 1 | −805,377 (76) | −226,475 (16) | −86,646 (3) |
| **2** | **−186,496 (11)** | **−94,797 (4)** | **−60,088 (0)** |
| 3 | −99,545 (4) | −79,355 (3) | −50,403 (0) |
| 5 | −62,360 (1) | −55,472 (0) | −50,401 (0) |
| 15 | −50,979 (0) | −45,164 (0) | −50,401 (0) |

(return, then violations. Full day from index 0, start at `auto_pods`.)

Read the diagonal: **one extra pod of actuator speed buys about what 30 s of history buys
(11 → 4 breaches), and three extra pods buy about what 8 minutes of history buys.** But the
two levers are not equally honest. Real pods take 20–60 s to become Ready, so `max_step=15`
is fantasy; and a wide action space deletes the problem instead of solving it. The 480 s
column also *saturates*: −60,088 at ±2 versus −50,401 at ±15, because once the policy knows
the 8-minute peak it is capacity-bound, not speed-bound. Take the memory.

The reassuring measurement is that the ramp limit is cheap *when you can see ahead*: with
perfect foresight the DP reaches −31,346 under ±2, versus −31,254 for an unconstrained oracle
that can jump straight to the target (4,185 resizes). The limit costs 92 reward across 8,640
steps — while costing 16 breaches when the agent is blind. **The constraint is not the
difficulty. Not knowing what the next minutes bring is.**

So: keep `Discrete(5)`, keep ±2, and put a *long* trailing peak in the observation — 30 s of
history is not enough (§8 measures the window directly). If you do want a wider action space,
the realistic version is a "target pod count" action plus a `pods_pending` queue — a
different, harder environment (README §11).

---

## 6. Training so the gradient can find the pattern

The reward and observation above make the problem learnable; these settings make it
learnable *in 200k steps* on a laptop.

**Episodes and the peak-heavy curriculum.** `--episode-steps 1440 --n-envs 4` with
`random_start=True` gives 4 h episodes. Uniformly sampled starts put the 09:49 spike in
`1440 / 8640 = 17 %` of episodes, and that single bin is worth 392,012 — 3.4× the entire
day's infrastructure cost. Most episodes therefore teach "pods are expensive" and almost
none teach "this is the moment to spend". Oversample the ramp:

```python
# 50 % of resets inside 05:00-10:00, where every breach in the trace lives
if rng.random() < 0.5:
    options["start_index"] = int(rng.integers(1_800, 3_600))
```

That single line changes the frequency of the informative state from 17 % to ~80 % of
episodes. It is also the likeliest reason the reference runs plateau at "hold 15": they do
not see the ramp often enough for it to beat the base rate.

**Discounting.** A step is 10 s, so `gamma` is a horizon in minutes:

| gamma | effective horizon | sees |
|---|---|---|
| 0.99 | 100 steps ≈ 17 min | burst response (enough for Fact 2) |
| 0.995 | 200 steps ≈ 33 min | a full ramp segment |
| 0.999 | 1,000 steps ≈ 2.8 h | the morning ramp |

Use `gamma=0.995` with `gae_lambda=0.95`, and watch `train/explained_variance` in
TensorBoard: below ~0.5 the critic cannot represent the return, which is what v1's 260,000×
spread tends to produce and what v2's bounded range is meant to fix.

**Exploration.** Five actions, and four of them are wrong most of the time. `ent_coef=0.01`,
and log the action histogram from `info` — if one action exceeds ~90 % of the day you have
either the "pressed against the wall" pathology (§2.3) or an entropy collapse. The reference
agent's `{hold: 98, +1: 1,956, +2: 6,586}` is the former.

**Everything else.** `n_steps=1024–2048`, `batch_size=256`, `lr=3e-4`, `clip_range=0.2`,
`--eval-freq 10000` with `n_eval_episodes=1` (the eval is the full day; there is no
variance to average), and `SyncVecNormalizeCallback` — without it the eval env normalises
with its own statistics and every number is a lie.

**Three seeds, median, spread.** One 200k run is an anecdote. Report
`median [min–max]` over `--seed 42/43/44`; day-level returns of two runs with identical
hyperparameters routinely differ by more than the gap between constant 9 and constant 10.

**Check transfer, cheaply.** Only one real day exists, but
`python data/generator.py --source synthetic --seed N` produces structurally similar days.
The winning causal family (peak-hold, reward v2) was replayed on four of them, once frozen at
the real day's parameters and once with (window, u) re-tuned on that day:

| trace | peak | best window + u on that trace | frozen at 480 s, u = 0.77 | loss |
|---|---|---|---|---|
| real (Jul 13 1995) | 480 | 480 s, 0.77 → −60,088 (0 viol) | −60,088 (0 viol) | — |
| synthetic seed 7 | 480 | 120 s, 0.72 → −69,053 (2 viol) | −80,392 (3 viol) | 16 % |
| synthetic seed 11 | 480 | 60 s, 0.85 → −28,182 (0 viol) | −32,334 (0 viol) | 13 % |
| synthetic seed 23 | 480 | 900 s, 0.72 → −59,784 (1 viol) | −77,412 (3 viol) | 30 % |
| synthetic seed 29 | 480 | 240 s, 0.70 → −66,454 (1 viol) | −86,042 (3 viol) | 30 % |

Two honest readings:

* **The shape transfers, the margin does not.** With `u` re-tuned per day and the window held
  fixed, the cost of the wrong window is 1.4–21 % (30 s) and 2.9–14 % (480 s); no single
  window wins on every day, but 2–8 minutes is never far off. Freeze *both* parameters and a
  new day costs 13–30 % and re-introduces breaches — the synthetic days want a fatter margin
  (u = 0.70–0.74) than the real one. Do not tune `u` on the evaluation day and call it
  transfer.
* **Every day is beatable from a constant.** On all four synthetic days the tuned causal
  policy is 1.7–4× better than that day's best constant (`−69,053` vs `−116,640`, `−28,182`
  vs `−115,080`, …), so the question is not whether the ladder exists off-trace — it does —
  but whether your agent climbs it.

That is also the recipe for testing the agent: train on a synthetic trace, evaluate on
`data/workload_trace.csv`, and report the *violation count* next to the return — a policy that
wins by breaching has learned the reward, not the traffic.

---

## 7. If the SLO is hard, use a constraint, not a bigger penalty

Every scalar reward has a "cheaper to breach" region (§3). If the requirement is literally
zero violations, the honest formulation is a constrained MDP: keep the **cost** signal
separate from the **reward**, and let a multiplier find the price.

```python
reward = -(cost_weight * pods + churn_weight * abs(requested_delta))  # what you buy
cost   = float(latency > sla_ms)                                     # what you must not buy
```

The environment does not emit a `cost` key today — add one line next to `sla_violation` in
`env.step` (and in the wrapper): `info["cost"] = float(latency > self.sla_latency_ms)`.
Keeping the two signals in separate fields is the whole point; folding the violation straight
into the reward is what §2 was about.

Lagrangian PPO is plain PPO plus dual ascent on `λ`:

```python
class LagrangianCallback(BaseCallback):
    """Raise the price of a violation until the budget is met, lower it when it is."""

    def __init__(self, budget: float = 0.001, eta: float = 1e-3, verbose: int = 0):
        super().__init__(verbose)
        self.budget, self.eta, self.lam = float(budget), float(eta), 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        mean_cost = float(np.mean([float(i.get("cost", 0.0)) for i in infos])) if infos else 0.0
        self.lam = max(0.0, self.lam + self.eta * (mean_cost - self.budget))
        self.logger.record("constraint/lambda", self.lam)
        self.logger.record("constraint/cost", mean_cost)
        return True
```

and the environment subtracts `lam * cost` inside its reward (pass `λ` in, or wrap the
training env in a small `RewardWrapper` that reads it from a shared object). The mechanism
matters more than the plumbing: `λ` rises while violations exceed the budget and falls when
they do not, so the *effective* price of a breach is discovered rather than guessed.

Two warnings from this trace:

* `budget = 0` drives `λ → ∞` and reproduces the corner solution of §2.1. Set a real budget
  (0.1 % of steps = 8.6 steps/day) or accept that "never" is not a reward, it is a constraint
  you enforce outside the learner.
* Report `λ` and the cost curve next to the reward. A run that satisfies the budget by
  sitting at 15 pods is visible immediately: runaway `λ`, zero scale-downs.

For this sandbox, prefer the cheaper route. Under v2 the 20 s-lookahead tracker already
reaches **zero violations at −44,996** — 2.6× better than constant 9 — so with a good
observation the gentle reward is sufficient, and no constrained machinery is needed. That is
the design rule worth remembering: **a better observation lets you use a gentler reward.**

---

## 8. The reference ladder, and what to aim at

`python utils/design_probe.py --reward v2 --section policies` prints this, including where
the reward goes. Full day, from index 0, same ±2 actuator, start at `auto_pods`:

| policy | return | violations | avg pods | actions | capacity | latency | churn |
|---|---|---|---|---|---|---|---|
| constant 8 pods | −115,080 | 1 | 8.00 | 0 | −103,680 | **−11,400** | 0 |
| constant 9 pods | −116,640 | 0 | 9.00 | 0 | −116,640 | 0 | 0 |
| "size it to this step" (u = 0.71) | −186,496 | 11 | 2.82 | 4,940 | −36,484 | **−125,400** | −24,612 |
| trailing max 30 s (u = 0.82) | −94,797 | 4 | 3.06 | 2,087 | −39,681 | **−45,600** | −9,516 |
| **trailing max 480 s (u = 0.77)** | **−60,088** | **0** | **4.58** | **159** | −59,376 | **0** | −712 |
| **oracle 20 s ahead (u = 0.92)** | **−44,996** | **0** | **2.79** | **1,959** | −36,116 | **−144** | −8,736 |
| DP ceiling (perfect foresight) | −38,658 | 0 | 2.66 | 963 | — | — | — |

Read the last three columns, not the return:

1. **A causal policy loses to latency, not to pods.** "Size it to this step" spends 36,484 on
   capacity and **125,400 (3.4×)** on being late; the 30 s version spends 45,600 on latency.
   The oracle spends *less* on capacity than the 30 s version (36,116) and **144** on latency.
   Autoscaling under this objective is not a cost-minimisation problem, it is a **timing**
   problem: the money is not in buying the right number of pods, it is in buying them before
   the ramp arrives.
2. **The 480 s row is the realistic first target for PPO** — 159 actions over the whole day,
   zero breaches, and it closes **73 %** of the gap between constant 9 and the ceiling
   (56,552 of 77,982) at 4.58 average pods. It is ~10 lines of numpy. If your agent cannot
   match it, the problem is reward/observation/curriculum, not sample count.
3. **What is left is worth less than it looks.** Twenty seconds of foresight (the oracle row)
   buys another 15,092 — 19 % of the gap — and perfect foresight over the entire day (the DP
   row) only 6,338 more (8 %). The expensive mistake is not "no forecast"; it is
   **too short a memory**.

### Why the window is the whole story

Sweeping the window at the best u for each (`simulate` with `util_policy(..., "trailing", w)`):

| window | best u | return | violations | actions | avg pods |
|---|---|---|---|---|---|
| 30 s | 0.82 | −94,797 | 4 | 2,087 | 3.06 |
| 60 s | 0.88 | −79,911 | 3 | 1,060 | 3.16 |
| 120 s | 0.82 | −62,509 | 1 | 604 | 3.74 |
| 240 s | 0.81 | −65,438 | 1 | 297 | 4.07 |
| **480 s** | **0.77** | **−60,088** | **0** | **159** | **4.58** |
| 600 s | 0.77 | −61,429 | 0 | 133 | 4.69 |

Monotone to ~8 minutes, then it starts to cost. The 30 s window is not "a bit worse": it is
the reason the causal rung looked hopeless (4 breaches, −94,797) next to a forecaster. Give a
feed-forward policy an 8-minute peak and it stops needing a forecast. That is why
`HeadroomObs` carries `peak_pods_slow` (§4): **the information has to be in the input, because
an MLP has no memory of its own.**

The transfer caveat is measured in §6: the *window* is stable (2–8 min wins on every trace
tested), the *utilisation margin* is not (u lands between 0.70 and 0.85 depending on the day).

Under the stock reward the same ladder collapses: trailing 30 s −1,608,870, oracle −37,208,
DP −31,346 — four breaches cost 1.5M there. But note the 480 s row scores **−59,465 under
v1 too**, because it takes no breaches at all: a policy that stops breaching is better under
*either* reward. Reproduced in the real environment, not just the probe:

```
CloudAutoscalingEnv, 8,640 steps, trailing-480 s u=0.77 replay:
  env-v1 return −59,465   SLA violations 0   final pods 3
```

So the reward redesign is not what makes "better than the constant" true — it is what makes
the gap *visible to the optimiser*. Under v1 the same solution is buried under a 1.5M cliff.

With the cap (`cap_mult=5`, the shipped default) the reward-optimal causal policy changes:
the probe prints `trailing 480 s u=0.95 → −51,848 with 1 breach`, because the cap makes being
late cheaper than an extra pod. The zero-breach version of that policy is still −60,088. That
is the error-budget knob of §3 doing exactly what it says: **if you require zero breaches, the
optimum under v2 and v3 is the same policy; if you accept a breach, the capped reward buys
one.**

**What "not learned yet" looks like:** any full-day run worse than −116,640, or one that
averages more than ~5 pods, or one that beats the constant *by taking breaches*. Those are
insurance, not traffic-reading.


---

## 9. Checklist

Reward:
- [ ] No quadratic in latency; the penalty is linear above a soft threshold.
- [ ] The penalty is capped, so one outage cannot dominate the episode.
- [ ] Churn is charged on the **requested** delta and is worth ≈ 2–4 pod-steps.
- [ ] One breach costs tens — not hundreds of thousands — of pod-steps (here: 133 at the SLA,
      675 at the 3,000 ms cap).
- [ ] You have decided whether the SLO is a budget or a hard constraint (§7).

Observation:
- [ ] Headroom, not only utilisation: `(capacity − rps) / mu_pod`.
- [ ] 30–60 s of history: trailing max and slope, in pod units.
- [ ] **A multi-minute peak too** (here 480 s). A feed-forward policy cannot integrate what
      you do not hand it; 30 s of history alone costs 36 % of the reward on this trace (§8).
- [ ] Latency normalised by the SLA, so the cliff is at a known value.
- [ ] Nothing saturates at the peak under `VecNormalize` (check `|obs|` against `clip_obs`).
- [ ] No trace-index feature unless the trace is rotated per episode.

Training:
- [ ] 4 h episodes, `random_start`, and a peak-heavy start distribution.
- [ ] `gamma` chosen as a horizon: 0.995 ≈ 33 min.
- [ ] `VecNormalize(norm_reward=True)` for training, `False` for eval, plus
      `SyncVecNormalizeCallback`.
- [ ] Action histogram logged; no action above ~90 % of the day.
- [ ] Three seeds, median and spread.

Evaluation:
- [ ] Compared against constant 9, the peak-hold tracker (last 8 min), the oracle tracker and
      the DP ceiling — same full day, deterministic.
- [ ] Violations reported next to cost: a reward win that adds breaches is not a win.
- [ ] One transfer run: synthetic trace → real day.

---

## 10. Reproduce

```bash
conda activate cloudrl

python utils/design_probe.py                      # every number in this document
python utils/design_probe.py --section price      # the pricing table
python utils/design_probe.py --reward v2 --section policies
python utils/design_probe.py --section actuator   # actuator speed vs memory (§5)

# stock objective, for the "before" rows
python utils/evaluate.py --no-baseline --constant-pods 8,9,15

# train with the redesign (after wiring the two wrappers into build_vector_env)
python agents/train_rl.py --algo ppo --timesteps 200000 \
    --episode-steps 1440 --n-envs 4 --gamma 0.995 --seed 42

python utils/evaluate.py --model models/ppo_cloud_autoscaling.zip --algo ppo
```

The probe and the wrappers are validated against each other: over a full 8,640-step day (a
deliberately bad "hold" policy, so most steps are overloaded), `LinearSLAReward` reproduces
the capped reward formula exactly — `max |wrapper − RewardConfig(v3)| = 0` — in either wrapper
order, with the 11-dim observation always inside its declared `Box`, and PPO trains and
predicts on the pair under `DummyVecEnv` + `VecNormalize`. The one deliberate mismatch is
against the *uncapped* v2 formula, which differs by up to 10,400 on saturated steps: that is
the cap doing its job.

The causal rung of §8 is also replayed through the *real* environment (not the probe's copy of
the physics) as a policy script: 8,640 steps, reward v1, 0 SLA violations, −59,465. The probe
is a mirror; that run is the mirror being checked.

One caveat on the numbers above: the probe replays physics, it does not train. It gives you
the *ladder* (what a policy family is worth) and the *price* (what the reward pays), which is
what you need to tell "my agent is bad" apart from "my reward is bad". Whether PPO finds the
ladder is still an empirical question, and it is the one claim this document deliberately does
not make. The falsifiable version: 200k steps, reward v2, `HeadroomObs`, one seed — **beat
−116,640 with zero SLA violations**. The heuristic proves that is possible with 159 actions a
day; if PPO lands on −60,088 ± 10 % it found the same statistic, if it sits at −116,640 with
~9 pods it is pinned, and if it beats −116,640 *while breaching* you have taught it the
reward's error budget instead of the traffic.

