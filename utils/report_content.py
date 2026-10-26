"""Text of the two-page submission report, as data.

The report is short, so the text lives here rather than in a document format that
needs a parser: ``utils/make_report.py`` walks :data:`BLOCKS` in order and turns
each entry into a flowable. Inline markup is ReportLab's mini-HTML (``<b>``,
``<i>``, ``<sub>``, ``<super>``, ``<font color=...>``), so the strings can be
edited without touching the layout.

Block kinds
-----------
``("abstract", text)``     shaded paragraph at the top of column 1
``("h2", "3. State space")`` section heading; the number is coloured
``("p", text)``            body paragraph
``("eq", text)``           centred display equation
``("table", key)``         one of the tables in ``TABLES``
``("fig", path, caption)`` figure + caption, kept on one page
``("refs", [entries])``    the reference list

Every number in the text is traceable to ``results/scores.csv``,
``results/frontier.txt`` or ``docs/RESULTS.md``; nothing here is recomputed.
"""

from __future__ import annotations

META = {
    "kicker": "AI COURSE · MINI PROJECT · ASSIGNMENT 2 (PHASE 3) · 26 OCTOBER 2026",
    "title": "Cost-First Cloud Autoscaling with PPO and a Soft SLA Price",
    "subtitle": ("A learned policy gets the fleet size right on real HTTP traffic — 28–48 % "
                 "cheaper than a safe static fleet — and the reward it maximises explains, "
                 "exactly, why it will not buy the one pod that makes it safe."),
    "authors": "Aaditya Biswas  ·  Satya Prakash Yadav",
    "url": "github.com/aaditya-biswas/RL_Cloud",
    "note": "code, trained checkpoints, report source",
}

ABSTRACT = (
    "<b>Abstract.</b> Autoscaling must follow a bursty request rate without paying for idle "
    "capacity. We train PPO on 28 days of the NASA Kennedy Space Center HTTP log, rescaled to "
    "480 RPS daily peaks and replayed as 8,640 ten-second steps per day against an M/M/k queue. "
    "The agent observes 11 headroom-based features and moves the fleet by at most two pods per "
    "step. Trained on days 1–16 and scored once on days 23–28, all six checkpoints (two reward "
    "designs × three seeds) are 28–48 % cheaper than a safe 9-pod fleet, yet each breaches the "
    "200 ms SLA on 2–18 of 51,840 steps. The reason is arithmetic, not training: a breached step "
    "is capped at 1,000 units while a pod-day costs 12,960, so a <i>standing</i> pod must prevent "
    "13 breaches a day to pay for itself. The policy is right about <i>how many</i> pods and wrong "
    "about <i>when</i> — 40 of 43 breaches are single 10-second steps at the 3,000 ms clamp, one "
    "pod short of <i>ceil</i>(rps/60), at eight traffic instants. A <i>conditional</i> pod, "
    "applied only there, is estimated at ≈200 units a day."
)


BLOCKS: list = [
    ("h2", "1. Problem and approach"),
    ("p",
     "Threshold autoscalers such as the Kubernetes HPA react to measured utilisation: by the "
     "time CPU crosses the target the queue is already deep, and the replica arrives after the "
     "spike. A learned policy can in principle act on the <i>shape</i> of the recent trace "
     "instead. We ask a narrow question that a table can answer: on a real 24-hour traffic trace "
     "with a hard 200 ms latency SLA, can a PPO policy beat (i) a safe constant fleet and (ii) a "
     "tuned threshold tracker on cost — and at what price in violations? Every number below comes "
     "from replaying a policy through the same environment and pricing it with the same function "
     "(<font face='Mono'>utils/score_policy.py</font>), under an acceptance rule fixed before the "
     "test days were scored."),

    ("h2", "2. Environment and dynamics"),
    ("p",
     "<b>Traffic.</b> 28 daily traces built from 1.89 M requests of the July 1995 log, binned into "
     "10-second slots and rescaled so that every day peaks at 480 RPS (the real peak is 7.8 RPS). "
     "Collector outages are linearly filled, never treated as zero traffic. The split is "
     "chronological and never shuffled: days 1–16 train, 17–22 validate, 23–28 test."),
    ("p",
     "<b>Queue model.</b> One pod serves μ = 60 RPS, so a fleet of <i>k</i> pods has capacity "
     "60<i>k</i> RPS. Latency in a step is"),
    ("eq",
     "L<sub>t</sub> = min(3000, 1000 / (60<i>k</i><sub>t</sub> − λ<sub>t</sub>) + 15) ms"
     "&nbsp;&nbsp; if &nbsp;60<i>k</i><sub>t</sub> &gt; λ<sub>t</sub>, &nbsp;&nbsp; else 3000 ms"),
    ("p",
     "Two properties of this model drive the whole result. Latency reaches 200 ms at 91 % "
     "utilisation, so the cliff starts <i>before</i> the fleet saturates; and because every day "
     "peaks at exactly 480 = 8 × 60 RPS, the fleet <i>k</i> = <i>ceil</i>(λ/60) has exactly 480 "
     "RPS of capacity at the peak — which the model scores as a 3,000 ms clamp. “Just enough” pods "
     "is therefore a breach <i>precisely at the peaks</i>, and nowhere else. That is why hold-8 "
     "breaches once a day and hold-9 is the cheapest clean constant fleet."),
    ("p",
     "<b>Actuator and cost.</b> The agent changes the fleet by a bounded delta, "
     "<i>k</i><sub>t+1</sub> = clip(<i>k</i><sub>t</sub> + <i>a</i><sub>t</sub>, 1, 15), and pays "
     "1.5 reward units per pod per step — a pod-day is 12,960 units. Training episodes are "
     "1,440-step (4-hour) windows with a random start, so the agent sees every hour of the day; "
     "evaluation replays whole days."),

    ("h2", "3. State space: 11 dimensions"),
    ("p",
     "The stock environment exposes five raw telemetry values (CPU, RAM, RPS, latency, pods). We "
     "replace them with eleven <i>pod-scaled</i> features — how many pods the traffic needs, how "
     "many I have, how much slack that is — because the action is denominated in pods. Features 6–9 "
     "are the causal stand-in for a forecast: a feed-forward policy cannot compute an eight-minute "
     "maximum from a 60-second window, so the window is handed to it."),
    ("table", "state"),

    ("h2", "4. Action space: five discrete moves"),
    ("p",
     "<b>Discrete(5):</b> <i>a</i><sub>t</sub> ∈ {−2, −1, 0, +1, +2} pods applied to the running "
     "fleet, then clipped to [1, 15]. A <i>delta</i> action rather than an absolute target, because "
     "it encodes the actuator's rate limit inside the action set, keeps the branching factor at "
     "five, and makes churn a first-class quantity — the reward charges the <i>requested</i> delta, "
     "so an oscillation is priced even when the clip swallows it. The action set is not the "
     "bottleneck: a one-pod forecast error is a single +1 move, and what the policy gets wrong is "
     "<i>when</i> to make it."),

    ("h2", "5. Reward function"),
    ("p", "The reward is a negative cost, paid every step (λ is a Lagrangian price, zero for R1):"),
    ("eq",
     "r<sub>t</sub> = −1.5 <i>k</i><sub>t</sub> − 4.0 |Δ<i>k</i><sub>t</sub>| − "
     "min(1000, 4.0 · max(0, L<sub>t</sub> − 150)) − λ · <b>1</b>[L<sub>t</sub> &gt; 200]"),
    ("p",
     "<b>Incentive design.</b> Three terms: <i>fleet rent</i>, linear in pods and paid every step; "
     "<i>churn</i>, charged on the request so that a jittery policy pays even when the actuator "
     "clamps; and a <i>latency ramp</i> that starts at 150 ms, reaches 200 units at the 200 ms SLA, "
     "hits its 1,000-unit cap at 400 ms, and is flat from there to the 3,000 ms overload clamp. The "
     "design is deliberately cost-first: across the scored runs the fleet term is <b>81–100 %</b> of "
     "total cost and the penalty is a sliver. Two consequences follow, and both are the point of "
     "Figure 1. Inside a step, depth is free above 400 ms, so a 410 ms blip and a 3,000 ms stall "
     "cost the same and the agent has no gradient toward “less deep”. Across steps the price is "
     "linear at 1,000 per breach, so <i>N</i> breaches cost <i>N</i> × 1,000 — one more breached "
     "step is never free."),
    ("p",
     "Comparing the two numbers gives the whole result. A pod-day costs 12,960, so a <i>standing</i> "
     "pod must prevent <b>13 breach steps a day</b> to pay for itself. Near the safe end the "
     "<i>marginal</i> breach costs 11,900–18,800 units, i.e. 12–19 × the cap: no <i>static</i> move "
     "towards safety is worth making for a v3 maximiser. R3 charges λ = 1,000 per breached step on "
     "top of v3 (frozen after 1,000 updates) — the same magnitude as the cap, which doubles the "
     "effective price and moves the arm along the frontier (Section 8)."),
    ("fig", "report/fig_reward.png",
     "<b>Figure 1.</b> The reward's geometry, which is the experiment's mechanism. <i>(a)</i> One "
     "step's penalty against its latency: a 4 units/ms ramp from 150 ms, 200 units at the SLA, and a "
     "1,000-unit cap at 400 ms — flat to the 3,000 ms clamp, so depth past the cap is invisible to "
     "the agent. <i>(b)</i> Total penalty against the number of breached steps: linear, which puts "
     "the break-even with a standing pod (12,960 units) at 13 breaches per day."),

    ("h2", "6. Algorithm and training protocol"),
    ("p",
     "PPO (Schulman et al., 2017) via Stable-Baselines3: an MLP policy with two 128-unit tanh "
     "layers, γ = 0.99, GAE λ = 0.95, clip 0.2, learning rate 3 × 10<super>−4</super>, 4 parallel "
     "environments × 2,048-step rollouts, minibatch 256, <b>200k steps</b> (≈15 min on CPU per "
     "run), seeds 2, 3 and 42. Observations and rewards are normalised by <i>VecNormalize</i> "
     "(observation clip 10), but <i>clip_reward</i> stays at infinity so the return the acceptance "
     "rule compares on keeps its raw scale; the evaluation environment inherits the training "
     "<i>obs_rms</i>, without which every evaluation number is noise. Checkpoints are selected on "
     "day 17 only and then frozen; the test days are scored once, after the rule was fixed."),
    ("p",
     "PPO rather than DQN: the reward is dense with a per-step scale of order 10<super>4</super> and "
     "a churn term of order 4 — exactly the regime where PPO's advantage normalisation is a free "
     "stabiliser — the action set is five discrete moves, and on-policy rollouts avoid a replay "
     "buffer of near-identical 10-second transitions."),

    ("h2", "7. Baselines"),
    ("p",
     "All three families are replayed through the same actuator and priced by the same function. "
     "<b>hold-<i>k</i></b>: a constant fleet, <i>k</i> ∈ {3, …, 10} — the naive safe policy, and "
     "the line the acceptance rule uses (hold-9 costs 116,640 units a day). <b>b1(<i>u</i>)</b>: "
     "the HPA shape, <i>k</i><sub>t</sub> = <i>ceil</i>(max<sub>480 s</sub> λ / (60<i>u</i>)), a "
     "trailing-maximum tracker whose utilisation target is swept over <i>u</i> = 0.55, …, 0.77; this "
     "family contains the cheapest clean policy on the test split. <b>Perfect-foresight DP</b> over "
     "the same dynamics (<font face='Mono'>utils/design_probe.py</font>) is used as a cost "
     "lower bound; it needs the whole day in advance and is not deployable."),

    ("h2", "8. Results on the six unseen days"),
    ("table", "results"),
    ("p",
     "<b>The cost claim generalises, and it is not close.</b> All six learned checkpoints beat the "
     "constant 9-pod fleet on all six unseen days, by 28–48 % (Figure 2). The two designs do what they were "
     "asked: R1 minimises v3 and lands cheapest (mean −65,791 a day, 11.7 breaches), while R3 pays a "
     "second, per-step price for breaches and lands 13,694 units dearer with 2.7 breaches — the "
     "Lagrangian moved the arm along the frontier and stopped short of zero. Seed noise (standard "
     "deviation 5,821 for R1, 6,429 for R3) is about half that 13,694 gap, so “R3 is dearer and "
     "safer than R1” survives, while “seed 3 is the best seed” does not."),
    ("p",
     "<b>The clean corner was affordable.</b> The tracker at <i>u</i> = 0.62 runs 6.53 average pods "
     "and takes <i>zero</i> breaches on the six days for −85,898 — 26 % below hold-9. So the learned "
     "arms' failure is not “clean is too expensive”; clean is <i>cheaper</i> than the safe constant "
     "fleet. Against that reference the arms are 1,700 (R3 seed 42) to 25,100 (R1 seed 3) units "
     "cheaper, and the difference is 1.2–2.3 average pods. Under the pre-registered rule — "
     "<i>zero</i> breaches on every test day <i>and</i> cheaper than hold-9 — nothing is accepted: "
     "the RL arms fail the first clause, and so do the two trackers used as scoring baselines. A "
     "rule allowing three breaches would accept R3 seed 42."),

    ("fig", "plots/10_cost_safety_frontier.png",     "<b>Figure 2.</b> The whole experiment on one axis. <i>x</i> = mean return v3 on the six test "
     "days (right = cheaper); <i>y</i> = breached steps on a symlog scale. Grey = constant fleet of "
     "5–10 pods, blue = the tracker sweep, purple/green = every learned seed; the green band is the "
     "acceptance corner. The constant ladder turns <i>up</i> after 7 pods and has no clean point "
     "before 9; the tracker reaches the axis at <i>u</i> = 0.62 and only gets dearer from there; "
     "every learned dot sits below and to the right of clean — cheaper, and breaching.", 0.97),
    ("fig", "plots/06_test_day_trajectory.png",
     "<b>Figure 3.</b> The worst test day (day 28), three policies. <i>Top:</i> grey area = "
     "demand / 60, the pods the traffic needs; black = hold-9, blue = the clean tracker, red = a "
     "learned arm. <i>Bottom:</i> the same day's latency on a log axis, 200 ms SLA dashed. Every "
     "point where the red line <i>meets</i> the grey one is a breach — 07:21, 08:13, 12:11 — one "
     "10-second step each. Cheap because the red line is low, breaching because it is not one pod "
     "higher.", 0.93),

    ("h2", "9. Why the breaches happen"),
    ("p",
     "<b>1. The controller is competent; its margin is one pod (Figure 3).</b> Eight of the 51,840 steps carry "
     "27 of the 43 breaches (63 %), and every one is a step where demand jumped by a full pod between "
     "samples: the 480 RPS peak or the 420 RPS shoulder. At those steps the fleet sits at exactly "
     "<i>ceil</i>(λ/60) — 8 pods for 480 RPS — and the shortfall is never more than one pod. Forty of "
     "the 43 stall at the 3,000 ms clamp; the other three are 207, 665 and 665 ms, one pod <i>just</i> "
     "short rather than two. Averaged over a day the arms are not thin at all: against instantaneous "
     "demand they carry +2.2 to +4.3 pods of slack and satisfy <i>k</i> ≥ <i>ceil</i>(λ/60) + 1 on "
     "98.5–100 % of steps. The failure is confined to the ~1 % of steps where the fleet sits exactly "
     "on that boundary."),
    ("p",
     "<b>2. The reward cannot tell a near-miss from an outage.</b> Because the cap binds at 400 ms, "
     "the 207 ms breach and a 3,000 ms stall are priced identically, and both fail the gate. The "
     "training signal contains no gradient toward “shallower”, only toward “fewer” — and fewer is "
     "worth 1,000 units against a 12,960-unit pod."),
    ("p",
     "<b>3. The clock is a safety crutch, not the source of savings.</b> Zeroing "
     "<i>fraction_of_day</i> at scoring time, so that the policy cannot know the hour, costs 6.7 % "
     "(R1) and 0.7 % (R3) of cost but multiplies breaches by 2.3–2.4 (11.7 → 27.0 for R1, 2.7 → 6.3 "
     "for R3). Without the hour the policy can no longer start climbing <i>before</i> the morning "
     "ramp, so it arrives late at every peak. The cost advantage does not depend on memorised "
     "schedules; the safety does."),

    ("h2", "10. Limitations and next step"),
    ("p",
     "PPO learns the <i>level</i> of the fleet almost immediately — every seed crosses the hold-9 "
     "line during training and stays above it — and never learns the timing margin that costs 12,960 "
     "a day and prevents 1,000-unit breaches. Three limits bound the claim. (i) The queue model is "
     "analytic — one pod short is a deterministic 3,000 ms, with no cold start and no queueing "
     "delay — so a real cluster would punish the same policy more at the peaks and less at the "
     "shoulders. (ii) The test split is six days of one month, all rescaled to the same 480 RPS "
     "peak, so “unseen” means unseen <i>shape</i>, never unseen <i>level</i>. (iii) The clean "
     "tracker's corner was located on the test days, so it is that split's own frontier rather than "
     "a validated policy."),
    ("p",
     "The next experiment is not longer training — both rungs are flat at 200k steps — but a "
     "<i>conditional</i> guard: a floor of <i>ceil</i>(max<sub>480 s</sub> λ / 60) + 1 pods on the "
     "policy's own action. That is one pod, only at the instants that matter, ≈200 units a day "
     "rather than 12,960, and it covers every one of the 43 breaches because the shortfall is never "
     "two. More generally, a per-breach price small enough to keep the fleet lean cannot also buy "
     "the last pod: safety has to enter as a constraint (a higher λ, a CVaR over step latency), not "
     "as a term in the bill."),

    ("refs", [
        "[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, “Proximal Policy "
        "Optimization Algorithms,” <i>arXiv:1707.06347</i>, 2017.",
        "[2] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus and N. Dormann, "
        "“Stable-Baselines3: Reliable Reinforcement Learning Implementations,” <i>Journal of Machine "
        "Learning Research</i> 22(268), 2021.",
        "[3] R. S. Sutton and A. G. Barto, <i>Reinforcement Learning: An Introduction</i>, 2nd ed., "
        "MIT Press, 2018.",
        "[4] J. Achiam, D. Held, A. Tamar and P. Abbeel, “Constrained Policy Optimization,” "
        "<i>ICML</i>, 2017.",
        "[5] NASA Kennedy Space Center WWW server access logs, <i>access_log_Jul95</i>, July 1995 "
        "(1.89 M requests; the traffic corpus of this report).",
        "[6] Kubernetes documentation, “Horizontal Pod Autoscaling,” 2024.",
    ]),
]


TABLES = {
    "state": {
        "caption": (
            "<b>Table 1.</b> Observation vector, <i>dim</i> = 11 "
            "(<font face='Mono'>env/design_wrappers.py</font>). Every entry is scaled to "
            "[0, 1] except the three signed ones (5, 6, 9), which are clipped to [−1, 1]; "
            "<i>VecNormalize</i> then standardises the vector and clips at 10."
        ),
        "header": ["dim", "feature", "what it measures"],
        "widths": [18, 72, 153],
        "align": ["right", "left", "left"],
        "mono_col": 1,
        "rows": [
            ["0", "cpu_utilization", "λ/(60k), capped at 1.0"],
            ["1", "ram_utilization", "lagged memory proxy, kept so plots stay comparable"],
            ["2", "demand_pods", "λ/60 scaled by the corpus peak — fleet size at 100 % load"],
            ["3", "latency_ratio", "min(L/200 ms, 3): 1.0 is exactly the SLA, 3.0 the clamp"],
            ["4", "fleet_fraction", "k/15"],
            ["5", "headroom_pods", "(60k − λ)/60 in pods; negative = short"],
            ["6", "slope_pods", "demand change over 3 steps, in pods — the ramp still to come"],
            ["7", "peak_60s", "max demand over the last 60 s, in pods"],
            ["8", "peak_480s", "max demand over the last 480 s (8 min), in pods"],
            ["9", "impulse_pods", "Δλ of the last step, clipped to ±1 pod"],
            ["10", "fraction_of_day", "t/8,640 — time of day (a safety cue, see Section 9)"],
        ],
    },
    "results": {
        "caption": (
            "<b>Table 2.</b> One row per checkpoint, scored once on days 23–28 (51,840 steps). "
            "Return v3 is a negative cost, so <i>less negative is cheaper</i>; “saving” is relative "
            "to hold-9; a breach is one 10-second step above 200 ms. R1 = reward v3 only, R3 = v3 "
            "plus the Lagrangian price. Last four rows are scripted references."
        ),
        "header": ["arm", "reward", "breaches", "days", "return v3", "saving", "pods"],
        "widths": [46, 30, 34, 24, 44, 26, 30],
        "align": ["left", "left", "right", "right", "right", "right", "right"],
        "rows": [
            ["r1 day16 s3", "v3", "18", "6", "−60,830", "48 %", "4.24"],
            ["r1 day16 s42", "v3", "12", "5", "−64,344", "45 %", "4.48"],
            ["r1 day16 s2", "v3", "5", "3", "−72,199", "38 %", "5.26"],
            ["r3 day16 s42", "v3 + λ", "2", "2", "−72,152", "38 %", "5.35"],
            ["r3 day16 s3", "v3 + λ", "3", "2", "−82,155", "30 %", "5.14"],
            ["r3 day16 s2", "v3 + λ", "3", "2", "−84,149", "28 %", "6.38"],
            ["hold-9", "static", "0", "0", "−116,640", "—", "9.00"],
            ["b1 u = 0.62", "tracker", "0", "0", "−85,898", "26 %", "6.53"],
            ["b1 u = 0.70", "tracker", "3", "3", "−76,975", "34 %", "5.81"],
            ["b1 u = 0.77", "tracker", "4", "3", "−71,583", "39 %", "5.39"],
        ],
        "highlight": {3: "#eef4ee"},   # the best cost/safety trade-off
        "muted": (6, 7, 8, 9),         # scripted references
    },
}
