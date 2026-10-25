# 16/6/6 split: what the autoscaler does on days it has never seen

Every row is one policy replayed over whole days and priced under reward v3 (`utils/score_policy.py`). Return v3 is a **negative cost**, so less negative is cheaper; `violations` is the hard gate and is counted in 10 s steps above the 200 ms SLA. A row that is cheaper and breaching is reported as both.

## 1. The split

Chronological, never shuffled. Peak rescaled to 480 RPS on every day; 8,640 steps/day at 10 s.

| split | days | first | last | weekend_days | mean_rps | max_p95_rps | max_flat_tail_h | max_missing_pct |
|---|---|---|---|---|---|---|---|---|
| train | 16 | 1995-07-01 | 1995-07-16 | 6 | 97.700 | 264.800 | 0.000 | 12.200 |
| val | 6 | 1995-07-17 | 1995-07-22 | 1 | 93.400 | 265.300 | 0.000 | 11.600 |
| test | 6 | 1995-07-23 | 1995-07-28 | 1 | 97.100 | 240.000 | 10.500 | 48.700 |

**Log outages.** A trailing run of identical RPS values is a gap in the 1995 log that the generator forward-fills, not calm traffic: day 28 (10.5 h). Those days keep their place in the split (dropping them after looking at the scores would be picking the test set), and the flat tail is easy traffic, so it makes those days *cheaper*, not harder, for every policy.

## 2. Baselines, per split

| key | split | days | violations | return_v3 | worst_return_v3 | avg_pods | max_latency_ms |
|---|---|---|---|---|---|---|---|
| b1-u0.70 | test | 6 | 3 | -76,975 | -90,116 | 5.810 | 3,000 |
| b1-u0.70 | train | 16 | 8 | -78,443 | -100,559 | 5.924 | 3,000 |
| b1-u0.70 | val | 6 | 0 | -79,423 | -88,198 | 6.034 | 69.762 |
| b1-u0.77 | test | 6 | 4 | -71,583 | -83,200 | 5.393 | 3,000 |
| b1-u0.77 | train | 16 | 9 | -72,210 | -92,198 | 5.447 | 3,000 |
| b1-u0.77 | val | 6 | 1 | -72,773 | -80,579 | 5.516 | 3,000 |
| hold9 | test | 6 | 0 | -116,640 | -116,640 | 9.000 | 31.667 |
| hold9 | train | 16 | 0 | -116,640 | -116,640 | 9.000 | 31.667 |
| hold9 | val | 6 | 0 | -116,640 | -116,640 | 9.000 | 31.667 |

## 3. Learned rungs, per split (mean over seeds)

| rung | variant | split | seeds | violations_mean | violations_max | violation_free_seeds | return_v3_mean | return_v3_std | worst_return_v3 | avg_pods | beats_hold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | ablate-time | test | 3 | 27.000 | 50 | 0 | -70,193 | 7,088 | -89,421 | 4.424 | True |
| R1 | full | test | 3 | 11.667 | 18 | 0 | -65,791 | 5,821 | -88,714 | 4.661 | True |
| R3 | ablate-time | test | 3 | 6.333 | 10 | 0 | -80,026 | 4,203 | -98,303 | 5.343 | True |
| R3 | full | test | 3 | 2.667 | 3 | 0 | -79,485 | 6,429 | -87,366 | 5.625 | True |

## 4. Test-set verdicts (one row per seed, baselines for reference)

Accept = every test day violation-free **and** mean return v3 above the constant-9 line (-116,640).

| run | rung | seed | test_days | violations | days_with_violation | mean_return_v3 | worst_return_v3 | avg_pods | beats_hold | accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| b1-u0.70 | baseline | 0 | 6 | 3 | 3 | -76,975 | -90,116 | 5.810 | True | False |
| b1-u0.77 | baseline | 0 | 6 | 4 | 3 | -71,583 | -83,200 | 5.393 | True | False |
| hold9 | baseline | 0 | 6 | 0 | 0 | -116,640 | -116,640 | 9.000 | False | False |
| r1-ablate-time-s2 | R1 | 2 | 6 | 5 | 3 | -73,530 | -83,880 | 5.123 | True | False |
| r1-ablate-time-s3 | R1 | 3 | 6 | 26 | 6 | -62,052 | -70,605 | 4.227 | True | False |
| r1-ablate-time-s42 | R1 | 42 | 6 | 50 | 6 | -74,997 | -89,421 | 3.924 | True | False |
| r1-day16-s2 | R1 | 2 | 6 | 5 | 3 | -72,199 | -88,714 | 5.265 | True | False |
| r1-day16-s3 | R1 | 3 | 6 | 18 | 6 | -60,830 | -69,717 | 4.240 | True | False |
| r1-day16-s42 | R1 | 42 | 6 | 12 | 5 | -64,344 | -77,648 | 4.478 | True | False |
| r3-ablate-time-s2 | R3 | 2 | 6 | 7 | 4 | -83,570 | -89,026 | 5.879 | True | False |
| r3-ablate-time-s3 | R3 | 3 | 6 | 10 | 5 | -81,124 | -98,303 | 4.416 | True | False |
| r3-ablate-time-s42 | R3 | 42 | 6 | 2 | 2 | -75,383 | -84,016 | 5.734 | True | False |
| r3-day16-s2 | R3 | 2 | 6 | 3 | 2 | -84,149 | -87,366 | 6.380 | True | False |
| r3-day16-s3 | R3 | 3 | 6 | 3 | 2 | -82,155 | -84,998 | 5.140 | True | False |
| r3-day16-s42 | R3 | 42 | 6 | 2 | 2 | -72,152 | -87,229 | 5.353 | True | False |

## 5. Time-feature ablation (transfer test)

The same checkpoints with `fraction_of_day` zeroed: if the score holds up, the policy is reading demand shape rather than a clock.

| rung | variant | split | seeds | violations_mean | return_v3_mean | worst_return_v3 | beats_hold |
|---|---|---|---|---|---|---|---|
| R1 | ablate-time | test | 3 | 27.000 | -70,193 | -89,421 | True |
| R3 | ablate-time | test | 3 | 6.333 | -80,026 | -98,303 | True |

## 6. Decision

**No arm is accepted.** 15 arm(s) (every learned seed plus the two trackers) were scored on 6 test days; either a violation appeared or the mean return v3 did not beat the constant-9 line (-116,640).

Cheapest arm: `r1-day16-s3` at -60,830 mean return v3 with 18 violation(s) on 6 test day(s) - cheaper than constant-9 by 55,810, but not clean, so it is reported as cheaper *and* breaching.

## 7. Reproduce

```bash
python utils/build_days.py
python utils/run_scores.py --run hold9 --policy hold --pods 9 --days all
python utils/run_scores.py --run b1-u0.77 --policy b1 --u 0.77 --days all
python utils/run_scores.py --run r1-day16-s2 --policy model \
    --model models/sla/r1-day16-s2.zip --seed 2 --days test
python utils/split_report.py
```

