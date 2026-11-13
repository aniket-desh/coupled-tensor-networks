# Degree as a difficulty axis under a constant-biased leaf init

When each leaf of the eight-leaf TTN starts with a fraction w of its variance on the constant feature, the time to learn a product of D patches rises steeply with D, and faster at larger w. The pre-registered criteria for "supported" are met. In a log-linear fit over D = 1 to 6, log t_half rises by 0.60 per patch (95% interval 0.47 to 0.73) at w = 0.25 and by 1.21 (1.09 to 1.33) at w = 0.5, about 1.8x and 3.4x per patch. Under the isotropic init (w = 1/9), D = 1 and D = 8 show no detectable difference (Mann-Whitney p = 0.97, 16 seeds each), as the basis-permutation symmetry predicts. The quantitative prediction fails, though: it gives slopes of 0.425 and 0.901, both measured slopes are 35 to 41% steeper, and neither interval contains its prediction.

All of this is at a horizon of t = 3e4, not the planned 1e5. The training had to be restarted to fit the worker's time cap (see [Judgement calls](#judgement-calls)).

![Learning time per seed against D (left) and log median against the prediction (right)](figures/degree.pdf)

*Left: t_half of each of the 16 seeds per group. Triangles at the top are runs that had not reached half the target by t = 3e4. Right: log median t_half (solid) against the prediction (dashed), with the prediction anchored at the D = 1 median. n = 16 seeds per group, the same 16 initial conditions at every D for a given w.*

| w | D | learned / 16 | median t_half | 95% interval |
|---|---|---|---|---|
| 1/9 (isotropic) | 1 | 13 | 9388 | 4825 to 15463 |
| 1/9 (isotropic) | 8 | 13 | 7419 | 3733 to 17136 |
| 0.25 | 1 | 16 | 158 | 107 to 292 |
| 0.25 | 2 | 16 | 337 | 154 to 822 |
| 0.25 | 4 | 16 | 1830 | 1106 to 3726 |
| 0.25 | 6 | 14 | 4356 | 2081 to 8906 |
| 0.25 | 8 | 11 | 17681 | 9169 to censored |
| 0.5 | 1 | 16 | 7.6 | 5.9 to 12.2 |
| 0.5 | 2 | 16 | 44 | 21 to 81 |
| 0.5 | 4 | 16 | 638 | 425 to 1190 |
| 0.5 | 6 | 15 | 4350 | 1976 to 7708 |
| 0.5 | 8 | 4 | censored | censored |

"Learned" means the coefficient on the teacher product reached half its target (t_half finite) by t = 3e4. Runs that didn't get there count as +inf in the medians, their bootstrap intervals and the Mann-Whitney test. "Censored" in the table means the median or the interval's upper end is +inf. Times are gradient-flow time, t = eta x steps with eta = 0.05.

| criterion | measured | predicted | result |
|---|---|---|---|
| symmetry control, w = 1/9, D = 1 vs 8 | Mann-Whitney p = 0.97 | p > 0.01 | passed |
| slope at w = 0.25, D in {1, 2, 4, 6}, 62 runs | 0.600 (0.469 to 0.730) | 0.425 | above 0; prediction outside the interval |
| slope at w = 0.5, D in {1, 2, 4, 6}, 63 runs | 1.214 (1.092 to 1.333) | 0.901 | above 0; prediction outside the interval |
| w = 0.5 slope larger than w = 0.25 slope | 1.21 vs 0.60 | yes | yes |

The slope is the least-squares slope of log t_half on D over individual runs, using the D values where at least 12 of 16 seeds learned (D = 8 misses that bar at both w). Its interval comes from 4000 bootstrap resamples of runs. The fit is log-linear, but the growth isn't uniform across D: at w = 0.5 the slope between adjacent D values falls from 1.77 (D = 1 to 2) to 1.34 (2 to 4) to 0.96 (4 to 6). The prediction is d log t_half / dD = (13/15) x 0.5 x log(8w / (1 - w)), in natural log per patch. It assumes that each active patch multiplies the initial coefficient by sqrt((1 - w) / (8w)), and that the escape time scales as |c0|^-(13/15). That second part is the depth-15 law that the init-scale experiment was designed to test, but its training was lost, so the law is still untested in this network.

## How it was done

- **Model and teacher.** The torch eight-leaf TTN in `ctn/ttn.py`, with bond dimension r = 8, d = 8 inputs per patch (so each leaf feature phi = (1, x) with x in {+-1}^8 has 9 entries), alpha = 1, float64, and 10 outputs of which only output 1 is trained. The teacher is y = prod_{j < D} x^(j)_1, with the constant on the other 8 - D patches (`teachers.degree(D)`), at strength s = 1.
- **Init.** Each leaf row has variance w on phi_0 and (1 - w)/8 on each x coordinate, so its second moment stays 1, and w = 1/9 is the isotropic init. Fusion and root weights have std 1/8. Every group of a given w uses the same 16 inits (seed 3000).
- **Training.** Full-batch gradient descent at eta = 0.05 on the exact population loss, all 12 groups in parallel on one A100-SXM4-80GB, until every run converged (|c/s - 1| < 1e-3 and loss < 1e-8) or t = 3e4. No group converged in all 16 runs, so every group stopped at t = 3e4. No run diverged.
- **Measurement.** The coefficient c on the teacher product is computed exactly from the population moments every 10 steps (Δt = 0.5) and recorded whenever it has moved by more than a threshold (adaptive logging). t_half is the time c reaches s/2, interpolated linearly between records.

The definition, with its commands and amendments, is in [EXPERIMENT.md](EXPERIMENT.md). The results are in `results/rerun/` (one `.json`, `.npz` and `_params.pt` per group, plus `analysis.json` and a `source/` snapshot), and the figure comes from `plot.py`, which reads only those files.

## How it was checked

- **Correctness checks** (`studies.ttn.checks`, `results/checks.json`) all passed on the worker before training. Among them is the alignment check that failed the first attempt (now 1, needs > 0.999) and the basis-permutation symmetry (6.1e-20). The first attempt's failure and diagnosis are in `attempt1/`, and the fix is in `ctn.ttn.alignment`.
- **The symmetry control** checks, in full-length runs on the worker, that degree has no detectable effect under the isotropic init: 13/16 learned in each group, medians 9388 vs 7419, p = 0.97. Its reach is limited. At w = 1/9 the biased init reduces to the isotropic one, so the control can't detect an error in how w is split between the constant and the x coordinates. It also can't detect a step-size artefact, because gradient descent at any step size preserves the basis-permutation symmetry exactly.
- **Censored runs in the slope fit.** `analyze.py` fits the slope on finite t_half only, so the 2 unlearned runs at w = 0.25, D = 6 and the 1 at w = 0.5, D = 6 drop out of the fit. Setting them to t = 3e4 instead gives slopes of 0.648 and 1.236, so dropping them makes the fitted slopes shallower. The "above 0" conclusion is conservative, and the excess over the prediction is, if anything, understated.
- **A fact-checking pass** compared this write-up with the result files, the job logs and the code. It corrected the reasoning about what the symmetry control can catch, a mislabelled step time and several timings in the restart account. It also corrected the exploratory section, which had mixed in D = 8 runs and proposed a mechanism that the numbers don't support.

## Exploratory: splitting the slope into two factors

This section was not planned and comes from looking at the saved initial coefficients after the main result. It uses the learned runs at D in {1, 2, 4, 6}.

The prediction multiplies two factors, and each can be fitted separately. The first is how fast the initial coefficient |c0| shrinks per active patch. Per-run regression of log|c0| on D gives -0.55 (-0.78 to -0.34) at w = 0.25, against a predicted -0.49, and -1.10 (-1.33 to -0.88) at w = 0.5, against a predicted -1.04. Both are consistent with the prediction, though the intervals are about ±0.22 wide. The second factor is how t_half depends on |c0|. At w = 0.5 the slope of log t_half on log|c0| is -0.86 (-0.96 to -0.77), which contains the depth-15 value of -0.867. At w = 0.25 it is -0.64 (-0.78 to -0.48), which excludes it. Pooled over both w it is -0.82 (-0.90 to -0.74, 125 runs).

Multiplying the two fitted factors gives a slope in D of about 0.95 at w = 0.5 and 0.35 at w = 0.25, short of the 1.21 and 0.60 measured directly. So t_half depends on D in a way that |c0| doesn't capture. Regressing log t_half on log|c0| and D together gives a D coefficient of 0.87 per patch at w = 0.5 and 0.40 at w = 0.25, at fixed |c0|. At w = 0.5 both factors agree with the prediction on their own, yet their product falls short. So the excess steepness isn't carried by the initial coefficient, and this analysis doesn't identify where it comes from.

## Not yet checked

- **The planned horizon of t = 1e5.** It would very likely add D = 8 to the w = 0.25 fit: 11/16 learned by 3e4 here, and the first training job's copy of that group (identical per seed on the shared learned runs) had 12/16 by t = 3.4e4. It could add D = 8 at w = 0.5 if 8 more of its 12 unlearned seeds learn by 1e5. It also pins down the isotropic control's tail (3/16 unlearned per group). It needs about 7 h of an A100 for all 12 groups at once, about $15, which is over this definition's cap. The direction and ordering of the slopes rest on the D = 1 to 6 groups, where 14 to 16 of 16 seeds learned, so a longer horizon is unlikely to reverse them.
- **Whether the steeper-than-predicted slope comes from discretization.** At w = 0.5, D = 1, t_half is about 150 steps and the coefficient is evaluated every Δt = 0.5. A rerun of the low-D groups at eta = 0.025 (under an hour) would show whether those times, and so the anchor of the slope, move.
- **A second init seed set.** All groups of one w share the same 16 inits. That is the design, since it gives a paired comparison across D, but it means the slope intervals cover seed-to-seed variation within one set of 16 inits only.

## Judgement calls

- **Training was restarted with all 12 groups at once and the horizon cut to t = 3e4.** The first job ran 8 groups at a time and queued the other 4. The isotropic control was among the queued groups. Each group sets its step budget from a quick timing at startup, there is no wall-clock stop, and a group saves only when it ends. No group had saved 1.75 h into that job (23:30 UTC), so I timed the real training loop on the worker. The w = 0.25, D = 8 group saved a few minutes later. Its startup timing of 23 ms per step had given it a short step budget, and it actually ran at 9.1 ms per step. The other seven first-wave groups had budgeted 2e6 steps on startup timings of 5.6 to 7.1 ms per step, so at that rate they would have ended near the worker's hard cap, about 02:50 UTC against a cap at about 03:14. One queued group (w = 0.5, D = 6) had just started but couldn't have finished before the cap. The other three, including the isotropic control, would have started too late to train. I killed the job at 23:36 and restarted all 12 groups together with `--t-max 3e4`, so every group, the control included, is censored at the same t. The alternative was to let the job run, which risked losing every group to the cap and could never produce the control. The cost is censoring between t = 3e4 and 1e5. At 3e4 the D = 8 groups fall short of the 12-of-16 bar (11 and 4 learned), so they are out of both fits. w = 0.25 would have passed it by t = 3.4e4, and whether w = 0.5 would have by 1e5 is unknown. The amendment in [EXPERIMENT.md](EXPERIMENT.md) has the details.
- **The first job's w = 0.25, D = 8 group** (t up to 3.5e4) is kept in `results/` but not used for the criteria, because its horizon differs from the rerun's. It does meet the 12-of-16 bar (12/16 learned by t = 3.38e4). As an unregistered check, a w = 0.25 fit that includes it over D in {1, 2, 4, 6, 8} gives 0.57 (0.48 to 0.66, 74 runs), which still excludes the predicted 0.425.

## What this means

Under the biased init the constant feature makes degree a real difficulty axis. The learning time rises steeply with the number of patches the target uses, by about 1.8x per patch at w = 0.25 and 3.4x at w = 0.5 over D = 1 to 6, and faster the more the init favours ignoring patches. The symmetry control is consistent with this coming from the bias rather than from D itself: under the isotropic init, D = 1 and D = 8 show no detectable difference. The prediction (the depth-15 law with the per-patch coefficient factor) gets the ordering and the rough ratio of the two slopes right (2.0 measured, 2.1 predicted), but both measured slopes are 35 to 41% steeper than predicted. Even at a fixed initial coefficient, t_half still rises with D. The cheapest next step is to rerun the low-D groups at eta = 0.025, which tests whether discretization explains part of the gap.
