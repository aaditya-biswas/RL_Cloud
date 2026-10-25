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

### What a constant fleet is actually paying for

The capacity term is linear in mean pods — 1.5 per pod-step = **540 per pod-hour = 12,960 per
pod-day** — so the bill is exactly `avg pods × 24 h × 540`, and the `--section policies` table
prints both halves of that product.

| policy | pod-hours paid | breaches | capacity | total |
|---|---|---|---|---|
| constant 6 | 144 | 54 | −77,760 | −586,960 |
| **constant 9** | **216** | **0** | **−116,640** | **−116,640** |
| trailing 480 s, u = 0.77 | 110 | 0 | −59,376 | −60,088 |
| DP ceiling | 64 | 0 | −34,434 | −38,658 |

Constant 9 buys 216 pod-hours to serve traffic whose *instantaneous* requirement averages
**2.31 pods** (55 pod-hours) and whose ±2-feasible optimum is **2.66 pods** (64 pod-hours):
**161 of its 216 pod-hours — 74 % of the bill — are surplus**, and it holds ≥ 2× what the SLA
needs on 93 % of the day, ≥ 3× on 83 %. The 3.9× peak/mean ratio is the reason: a constant has
to be right at 09:49, so it is wrong everywhere else. It is not even knife-edge at the peak —
480 rps against 540 rps of capacity, 1.0 pod of margin, never above 89 % utilisation — which
is what makes 9 a *fair* baseline rather than a strawman: it is the cheapest constant that
survives the day (8 breaches at the peak).

The caveat, because this is easy to over-read: mean pods is the *cost axis*, not the
objective. Constant 6 has the smallest capacity bill of any constant and the worst return of
any constant, because it breaches 54 times; "size it to this step" averages 2.82 pods and
loses to every constant (−186,496, 11 breaches). What holds is domination **at zero
breaches**: the 480 s tracker sits at (4.58 pods, 0 breaches) and strictly beats (9.00, 0),
recovering 4.42 of constant 9's 6.34 pods of excess over the 2.66-pod floor. The 1.92 pods it
leaves on the table are exactly what 20 s of foresight would have bought (§8).


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
**pod-steps** (one pod for one 10 s step costs 1.5, i.e. **540 per pod-hour**).

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
reward's error budget instead of the traffic. §11 reports which of those three happened,
with three seeds per rung.

## 11. What the rig measured

§10 left one falsifiable question: at 200k steps, does PPO clear −116,640 **with zero
violations**, and does it land near −60,088? Six learner runs (three seeds × two routes)
plus the scripted rungs answer it. Every row below is produced by replaying a *finished*
checkpoint through the real environment and pricing that single trajectory under v1, v2
and v3 (`utils/score_policy.py --policy model`), so no row is a probe estimate and no two
rows were priced by different code.

### 11.1 The scripted rungs, including the margin sweep

| policy | viol | pods | pod-h | churn | ret v3 | max ms | acts |
|---|---|---|---|---|---|---|---|
| constant 9 pods | 0 | 9.00 | 216.0 | 0 | −116,640 | 32 | 0 |
| trailing max 480 s, u=0.65 | 0 | 5.35 | 128.4 | 832 | −70,152 | 58 | 178 |
| trailing max 480 s, u=0.70 | 0 | 5.05 | 121.3 | 848 | −66,340 | 67 | 192 |
| trailing max 480 s, u=0.72 | 0 | 4.84 | 116.2 | 784 | −63,538 | 67 | 170 |
| trailing max 480 s, u=0.74 | 0 | 4.79 | 114.8 | 784 | −62,803 | 67 | 169 |
| **trailing max 480 s, u=0.77** | **0** | **4.58** | **110.0** | **712** | **−60,088** | **67** | **159** |

The sweep matters because it prices *safety*: the same rule, 10 % more conservative, is
still clean at −66,340. That is inside the learned policies' scatter, and cheaper than the
mean of the run where the price actually bit (−71,394). **The safety the learner lacked was
available at a cost the learner was already paying.**

### 11.2 The learned rungs (3 seeds, 200k steps, reward v3 + `HeadroomObs`)

| rung | seeds | viol | rate % | pods | pod-h | churn | ret v3 | max ms | acts |
|---|---|---|---|---|---|---|---|---|---|
| R1 reward only (`--lambda-lr 0`) | 2,3,42 | 2 ± 1 (1–3) | 0.023 | 4.56 ± 0.64 | 109.3 | 1,960 | −62,996 ± 6,726 | 3,000 | 388 |
| R2 Lagrangian, price step 50 | 2,3,42 | 2 (2–3) | 0.027 | 4.41 ± 0.51 | 105.8 | 3,781 | −63,261 ± 5,916 | 3,000 | 747 |
| R3 Lagrangian, price step 20,000 | 2,3,42 | 1.7 (1–2) | 0.019 | 4.75 ± 0.60 | 113.9 | 8,205 | −71,394 ± 7,522 | 3,000 | 1,707 |

Per seed, cost against violations: R1 −54,891/3, −71,360/1, −62,737/2; R2 −54,916/3,
−67,968/2, −66,898/2; R3 −63,390/2, −81,464/1, −69,329/2.

**Zero of nine checkpoints reached zero violations.** No seed of any route was accepted.

### 11.3 The claim in §10, checked honestly

* **Cost: met.** R1's mean −62,996 sits inside the predicted ±10 % band around −60,088
  (−66,097 … −54,079), and its pod-hours (109.3) are *below* the tracker's (110.0).
* **Safety: missed.** Every checkpoint breaches 1–3 steps a day, all with the same
  signature: a 1-step spike to 3,000 ms latency.
* **The reward redesign is legible where it matters.** Priced under v1, the same checkpoints
  score −461,086 … −1,224,870 (r1-s2, r1-s3) against −59,465 for a policy that never
  breaches. r1-s3's three 3,000 ms steps are worth 1.17 M of that gap and 9,000 under v3 —
  ≈390 k per breach, the same order §8 quotes for the four-breach script. The cliff is real
  on *learned* policies, not only on scripts.

So the redesign did what it claimed for the *price* and not for the *constraint*: PPO found
the heuristic's economics and not its safety.

### 11.4 Why the price did not close the gap

Three measurements, in increasing order of how much they rule out.

1. **The dual step was three orders of magnitude too small.** R2 moved λ from 0 to 6.24 in
   200k steps, because the update is `η · violation_rate` and the rate was 0.001:
   `50 × 0.001 = 0.05` per update, against a pod that costs 1.5 per step. λ never became a
   barrier, so R2 ≈ R1 by construction.
2. **A price that does bite still does not comply.** R3 raised η to 20,000; λ hit its
   ceiling of 1,000 on the *first* update and stayed pinned for the rest of the run, in all
   three seeds. Compliance did not improve (0.019 % against R1's 0.023 %), cost fell a
   further 13 % behind, churn quadrupled, and two seeds thrashed (2,487 and 2,523 realised
   changes against R1's 243–630). "Make the price bigger" is now measured, not argued.
3. **The residual breaches are a ±1-pod accuracy failure at a cliff, not a budget failure.**
   The trace jumps by more than 100 rps 333 times a day (171 up, 162 down, 12 over 200) and
   the learner absorbs all but one to three of them. It misses the ones where its fleet sits
   one or two pods below what that rate needs, and the ramp is not the difference — both
   policies take the same +2 step, from a different starting point: at 4,572 (301.54 rps) R1
   ramps `3→5` and takes 3,000 ms while the tracker ramps `5→7` and clears at 23 ms; at 4,877
   (443.08 rps) R1 ramps `5→7` and takes 3,000 ms while the tracker ramps `6→8` and clears at
   42 ms; at 6,148 — the same 301.54 rps, a value the day holds for 24 steps and the *same*
   policy clears 23 times — R3 ramps `4→5` and breaches while the tracker ramps `4→6` and
   clears at 32 ms. Latency on these edges is a step, not a slope: 42 ms at eight pods,
   3,000 ms at seven, for the same 443 rps. The learner is not poorer — it spends 109.3
   pod-hours against the tracker's 110.0 — it sits two or more pods away from the tracker's
   fleet on 674 of 8,640 steps (R1) and is unlucky on 1–2 of them. The rate is in the
   observation when the decision is made: both policies see 301.54, and only one buys the
   margin.

That is what makes this a resolution limit rather than a pricing limit. λ = 1,000 puts a
1,000-unit charge on those states against the 1.5 units an extra pod-step costs — a factor of
600 — and the pod was still not bought. What 200k steps produced is a policy that maps rps to
pods within one pod; the SLA is unforgiving at the cliff, where one pod is the whole
difference. A per-step indicator price cannot repair that: it is one scalar on the 1–2 wrong
states, and the hundreds of states where the policy is *about* to be one pod low are priced
exactly like the states where it is safely high. A shaped charge — paid for running *near* the
SLA, or for occupancy above a threshold — puts a slope on those hundreds of states and turns
"be one pod short sometimes" into a cost. That is a hypothesis, and §11.5 says how to test it.

### 11.5 What this rules in and out

* **Ruled out.** A larger per-step price, or more steps at this reward, as the fix. R3 is
  the direct test of that hypothesis, with three seeds.
* **Ruled out.** That the 11-dim observation hides the decision: the rate that requires the
  extra pod is in the observation *at* the decision step — both policies read 301.54, and
  only one buys the sixth pod.
* **Ruled in.** The reward stack and the rig are sound: `--design-check` is 6/6, the ladder
  is reproduced cell-for-cell, and v1/v2/v3 are priced from one trajectory. The learned rows
  are therefore a statement about PPO's optimum, not about a mispriced wrapper.
* **Ruled in, as the next experiment.** Shape the price so it has a slope *before* the
  breach — an occupancy or headroom charge above ~100 ms, or a constraint on the maximum (or
  a high quantile) of latency over the episode — and keep the 480 s feature. §11.1 bounds
  what that is worth: a clean policy costs between −60,088 and −66,340, so a shaped run has
  ~6,000 units of margin to spend and still beat the acceptance line.

One caveat, stated rather than hidden: this is 200k steps, one PPO configuration, one trace,
three seeds. It is enough to reject the specific claim in §10 and to reject a bigger λ; it
is not enough to claim that PPO cannot find the margin under a shaped reward — that is the
experiment §11.5 proposes.

### 11.6 Reproduce

```bash
# the gate, then the two routes (200k steps each; ~10 min on 12 cores)
python agents/train_sla.py --design-check
python agents/train_sla.py --timesteps 200000 --tag r1 --lambda-lr 0
python agents/train_sla.py --timesteps 200000 --tag r2

# the price that actually bites, and the seeds that turn a run into a result
python agents/train_sla.py --timesteps 200000 --tag r3 \
    --lambda-lr 20000 --lambda-max 1000 --lambda-freeze-after 1000 --seed 42

# score a finished checkpoint on one rollout, priced under v1/v2/v3
python utils/score_policy.py --policy model --model models/sla/r1.zip
python utils/score_policy.py --policy b1 --u 0.70     # the margin sweep
```

TensorBoard records the constraint controller: `sla/price`, `sla/violation_rate`,
`sla/mean_pods`, `sla/frozen`. A flat `sla/price` at 0 means you are looking at the
reward-only route; a price pinned at `--lambda-max` means it saturated, and per §11.4
saturation is not compliance.

## 12. One day is not a benchmark: the 16/6/6 split

§11 ends with a caveat: 200k steps, one PPO configuration, **one trace**, three seeds. The
single trace is the weakest part of that list. Day 13 alone cannot separate "learned to
autoscale" from "memorised Thursday the 13th", and it cannot fail a policy on a day whose
ramp is steeper. The log holds 28 days (1-28 Jul 1995), so this section trains on 16 of
them and scores on 6 that no run has ever seen.

### 12.1 What changed, and why each change was needed

* **`utils/build_days.py`** replays every day to `data/days/day_DD.csv` and freezes a
  **chronological** split (train 01-16, val 17-22, test 23-28) in `results/split.json`.
  Chronological, not shuffled: day 17 follows day 16 in real time, so a shuffled split
  would leak the shape of the following day into training.
* **Every day is rescaled to the same 480 RPS peak.** Shape varies, difficulty ceiling does
  not, so the constant-9 row stays a fixed yardstick (−116,640 on every day) instead of a
  moving target.
* **`CloudAutoscalingEnv(trace_paths=[...])`** draws one day per `reset()` and keeps it for
  the whole episode. `max_rps` is *not* recomputed per day: freezing the observation scale
  to the pool maximum keeps `peak480 = 0.8` meaning the same traffic on a quiet day and on
  a busy one. A single-trace env behaves exactly as before.
* **`train_sla.py --train-days --split-json --eval-trace`** take the train list from the
  manifest, so a directory that happens to contain the test days cannot leak into training.
  The eval env replays the **whole** validation day (not the 4 h training window): scoring a
  seed on 00:00-04:00 of one day ranks seeds by night-time behaviour.
* **`CheckpointCallback(save_vecnormalize=True)`**, which was missing: without the matching
  `VecNormalize` statistics a mid-training checkpoint is fed raw observations, and every
  number scored from it is wrong.
* **`--design-check` is now 8/8**: the day split is seed-reproducible, every train day is
  visited, every observation on every day is inside the `Box`, and the eval env never draws
  a training day.

### 12.2 The scripted rungs on unseen days

| policy | split | days | violations | return v3 | worst day | avg pods |
|---|---|---|---|---|---|---|
| constant 9 pods | train / val / test | 16 / 6 / 6 | 0 / 0 / 0 | −116,640 | −116,640 | 9.00 |
| trailing max 480 s, u=0.77 | test | 6 | **4** | −71,583 | −83,200 | 5.39 |
| trailing max 480 s, u=0.70 | test | 6 | **3** | −76,975 | −90,116 | 5.81 |
| trailing max 480 s, u=0.77 | train | 16 | 9 | −72,210 | −92,198 | 5.45 |
| trailing max 480 s, u=0.70 | train | 16 | 8 | −78,443 | −100,559 | 5.92 |

The row that matters is the third column: **the tracker is not clean either.** On day 13 it
was (0 violations, −60,088); across 28 days it breaches on 4 test days and 9 training days.
The same one-step signature as the learner: the trailing 480 s window is a step behind a
ramp, and one pod short at the cliff is 3,000 ms. §11's "the safety was available at a cost
the learner was already paying" is therefore too generous — at u=0.77 the heuristic was
buying that safety on *that* day, not in general. What survives is the cost claim: the
tracker is ~45,000 cheaper than hold-9 on average and ~44,000 cheaper on test.

### 12.3 The learned arms, six unseen days, three seeds

| rung | seeds | violations (test) | days with a breach | return v3 (test) | worst day | avg pods |
|---|---|---|---|---|---|---|
| R1 reward only (`--lambda-lr 0`) | 2,3,42 | 5 / 18 / 12 | 3 / 6 / 5 | −65,791 ± 5,821 | −88,714 | 4.66 |
| R3 Lagrangian, price step 20,000 | 2,3,42 | 3 / 3 / 2 | 2 / 3 / 2 | −79,485 ± 6,429 | −87,366 | 5.62 |

Both rungs are cheaper than hold-9 on every one of the six test days — the §11 cost result
replicates out of sample, which is the one thing a single trace could not establish. The
safety result replicates too: **no seed of either rung is violation-free on any test day's
worth of traffic**, and the breaches are still single 3,000 ms steps.

The price does buy something, just not compliance: R3 breaches 2.7 steps per six days
against R1's 11.7, at a cost of ~13,700 (~21 %) more. That is the first measurement in this
document where the Lagrangian route changes the safety/cost trade-off in the direction it
was designed to — and it still cannot clear zero.

### 12.4 The decision rule, applied

The rule was fixed before the run: a rung is accepted iff **every seed is violation-free on
every test day** and its mean test return v3 beats the constant-9 line.

| arm | violations (test, all seeds) | violations == 0? | cheaper than −116,640? | accepted |
|---|---|---|---|---|
| R1 reward only | 35 | no | yes (−65,791) | **no** |
| R3 Lagrangian | 8 | no | yes (−79,485) | **no** |
| trailing max 480 s, u=0.77 | 4 | no | yes (−71,583) | **no** |
| constant 9 pods | 0 | yes | — (it *is* the line) | — |

**No arm is accepted.** Cheapest is R1 seed 3 at −60,830 — 55,810 cheaper than hold-9 and 18
breaches; safest is R3 seed 42 at 2 breaches and −72,152. On 6 unseen days the ranking is
"R1 cheapest, R3 safest, neither clean", and the scripted tracker sits between them.

That is the honest summary of this experiment: with 16× the training days, the cost claim
survives out of sample, and the safety claim fails on every arm including the heuristic.

### 12.5 The time-feature ablation

Scoring the same checkpoints with `fraction_of_day` zeroed (`--ablate time`, the transfer
test from §5 of the reward design):

| rung | violations, full | violations, no clock | return v3, full | return v3, no clock |
|---|---|---|---|---|
| R1 | 11.7 | 27.0 | −65,791 | −70,193 |
| R3 | 2.7 | 6.3 | −79,485 | −80,026 |

Removing the clock roughly doubles breaches in both rungs while moving cost by ~4,500 (R1)
and ~500 (R3). Read together: the policy does use the time index, but as a *safety* cue, not
as the mechanism behind its cost advantage. A policy that still beats hold-9 with the clock
removed is not memorising the daily schedule; it is one that gets worse at the cliff without
it. That is a narrower and more useful claim than "the model learned the diurnal shape".

### 12.6 What this section does and does not establish

* **Established.** The cost result generalises: every seed of both rungs is cheaper than
  hold-9 on all 6 unseen days, with one PPO configuration and the same reward stack.
* **Established.** Day 13 was not representative. The tracker's clean 0-violation row is a
  day-13 property: on 28 days it breaches 14 times (u=0.77) and 11 times (u=0.70).
* **Not established.** That PPO cannot reach zero violations with more capacity. 16 days in
  200k steps is ~12,500 steps per day-shape, less per shape than the single-day runs had; the
  rungs are undertrained by construction and §11.5's shaped-reward experiment is still the
  next test.
* **Known limits.** (i) Day 28 is 49 % interpolated: the log stops around 13:20 and the
  generator forward-fills 10.5 h of flat 240 RPS. It is kept in the test set and flagged in
  `results/inventory.csv` — dropping it after seeing the scores would be choosing the test
  set. (ii) 28 days is 28 samples: the seed spread on R1 (±5,821) is not much smaller than
  the gap between rungs (~13,700). (iii) Checkpoint selection uses one validation day
  (day 17); the test set is six days, touched once.

### 12.7 Reproduce

```bash
# 28 days, the frozen split, the gate
python utils/build_days.py
python agents/train_sla.py --design-check --train-days data/days \
    --split-json results/split.json --eval-trace data/days/day_17.csv   # 8/8

# baselines on every day, then the learned rungs on the test days
python utils/run_scores.py --run hold9 --policy hold --pods 9 --days all
python utils/run_scores.py --run b1-u0.77 --policy b1 --u 0.77 --days all
python agents/train_sla.py --timesteps 200000 --tag r1-day16-s2 --seed 2 --lambda-lr 0 \
    --train-days data/days --split-json results/split.json \
    --eval-trace data/days/day_17.csv
python agents/train_sla.py --timesteps 200000 --tag r3-day16-s2 --seed 2 \
    --lambda-lr 20000 --lambda-max 1000 --lambda-freeze-after 1000 \
    --train-days data/days --split-json results/split.json \
    --eval-trace data/days/day_17.csv
python utils/run_scores.py --run r1-day16-s2 --policy model \
    --model models/sla/r1-day16-s2.zip --seed 2 --days test
python utils/run_scores.py --run r1-ablate-time-s2 --policy model \
    --model models/sla/r1-day16-s2.zip --seed 2 --days test --ablate time

# tables, figures
python utils/split_report.py      # results/scores.csv -> results/summary.md + 3 tables
python utils/make_plots.py        # plots/01..09
```

The artefacts are `results/scores.csv` (one row per run and day), `results/baselines.csv`,
`results/policy_scores.csv`, `results/seed_table.csv`, `results/summary.md`,
`results/inventory.csv`, `results/split.json` and `plots/01..09_*.png`. Every number in the
tables above is one `score_policy.py` replay of one checkpoint on one day, priced under v1,
v2 and v3 from that single trajectory.



