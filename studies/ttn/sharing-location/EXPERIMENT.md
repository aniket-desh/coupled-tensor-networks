---
name: sharing-location
title: "Where and how much computation two outputs share: agreement sets in the 15-core TTN"
status: done
gpu: NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe
count: 1
hours: 5.5
budget_usd: 12.05
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-12T18:37:28Z
result: failed
finished_utc: 2026-11-13T00:09:12Z
writeup: studies/ttn/sharing-location/README.md
---

# Where and how much computation two outputs share: agreement sets in the 15-core TTN

## Question

In the jointly trained TTN, does a second output learn faster when part of its computation is shared with an output being learned at the same time? Does that depend on how much is shared, and on whether the shared part is a whole subtree or scattered leaves?

Here "shared" is defined by what the dynamics can see. Two outputs share leaf j when they use the same basis index there. By the basis-permutation symmetry (checked in `studies/ttn/checks.py`), the constant phi_0 is not special and product degree is irrelevant. So the agreement set S, and where it sits in the tree, is the whole description.

## Hypothesis

Teachers come from `teachers.agreement(S)`, with s = (1, 0.5, 0.75):

- y1 = prod_j x^(j)_1;
- y2 uses x_1 on the leaves in S and x_2 elsewhere;
- y3 = prod_j x^(j)_3, which shares nothing with either.

y1, y3 and the inits are identical in every family, so runs pair by seed. At the teacher, y1 and y2 share exactly the tree nodes whose leaves all lie in S:

| family | S (0-indexed leaves) | shared cores |
|---|---|---|
| none | {} | 0 |
| leaf | {0} | 1 |
| split2 | {0, 2} | 2 |
| quarter | {0, 1} = q1 | 3 |
| split4 | {0, 1, 4, 5} | 6 |
| half | {0, 1, 2, 3} = left half | 7 |

Predictions:

- **Reuse speeds y2.** Seed-paired, t_half(c2) is shorter than in `none` for quarter and half. The speed-up grows with the number of shared cores.
- **Location (exploratory).** At equal extent, a subtree (quarter, half) beats scattered leaves (split2, split4) by more than one extra shared core alone would explain. No numeric prediction.
- **Mechanism.** Under gradient flow, dc2/dt = K21 e1 + K22 e2 + K23 e3 + rest. The rest is the pull of the student's coefficients on all non-teacher basis functions.
  - CPU validation (21 escapes): own about 4 to 8x the rise, rest about -3 to -7x, cross terms about 0.
  - A rise that comes after y1 is learned has e1 ~ 0, so the cross term vanishes whatever K21 is, and "cross is small" can't discriminate.
  - The prediction is therefore that sharing speeds y2 by enlarging its own mobility K22, mostly in the shared subtree (q1 for quarter). K22 is measured at equal c2 (c2/s2 = 0.02) and paired by seed against none.
- **Pilot.** A 16-seed CPU pilot (exact population GD, this init seed) on quarter vs none showed little difference by t = 5000: c2 was learned in 3/16 vs 2/16 seeds. By t = 2e4 the difference was clear: 11/16 vs 3/16, with c1 at 12/16 vs 9/16 and c3 at 3/16 vs 4/16. Escapes are heavy-tailed, so many runs may still be censored at t = 1e5.
- **Against:** the paired h2 isn't faster in quarter or half than in none (sign test p > 0.05), or there's no monotone trend in shared cores.

## Method

- **Model and init.** As in `ctn/ttn.py`: r = 8, d = 8, 10 outputs (4 to 10 masked), alpha = 1 (leaf std 1/3, fusion and root std 1/8). Float64 throughout.
- **Training.** Full-batch GD at eta = 0.05 on the exact population loss (Gram contractions, no samples), with t = eta x steps. All three outputs are trained from t = 0.
- **Seeds.** 16 per family, with the same init seed (2000) in every family.
- **Finite-data replicate.** The quarter family again, on N = 2^17 sampled +-1 inputs per run: full batch, sampled squared error, same init seed. Its coefficients are measured on a 16384-sample probe (sd ~ 0.004).
- **Measurements.**
  - Exact c, recorded whenever it changes.
  - The mobility K_ij by subtree group (q1, q2, q3, q4, L, R, root), at every record while a component rises.
  - Alignment and spectra on a coarse schedule.
- **Stopping.** A group stops when every run has |c_k/s_k - 1| < 1e-3 and loss < 1e-8, or at t = 1e5, or at the 4.5 h deadline. All 7 groups run as parallel processes on the GPU.
- **Half-time** is the first time c_k reaches s_k/2; censored runs count as +inf.
- **Drive decomposition.** At every K record the exact velocity v = <grad c, -grad L> is also logged (checked against a finite-difference GD step).
  - For each run that learns c2, `studies/ttn/analysis.py:drive` integrates over c2's rise, from c2/s2 = 0.02 up to its half-time, where records are dense: K21 e1 (cross from y1), K22 e2 (own), K23 e3, and rest = v - sum_j K2j e_j.
  - Each is reported as a fraction of the actual rise. The integral of v over the actual change ("resolved") checks the time resolution.

## Inputs

None. Inputs are generated (seeded independent +-1). No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
uv run --frozen python -m studies.ttn.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/ttn/sharing-location/run.py --out "$EXP_DIR/results" --hours 4.5
PYTHONPATH=. uv run --frozen python studies/ttn/sharing-location/analyze.py "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/ttn/sharing-location/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero if any value is out of tolerance. Run the training as its own `orch exec` step from a background command.

## Expected outputs

- `results/checks.json`.
- `results/{none,leaf,split2,quarter,split4,half,quarter_sampled}.{npz,json}` plus `_params.pt`.
- `results/analysis.json`.
- `results/source/`.
- `figures/*.pdf`.
- `README.md`: the write-up, with the per-family table from `analyze.py`.

## Controls

- `none` is the no-sharing control. It has the same y1, y3, strengths and inits, differing only in y2's agreement with y1.
- **Pairing by seed** across families: the same init, and y1 and y3 are identical.
- **Equal-extent pairs** separate location from amount: quarter vs split2, and half vs split4.
- **Correctness checks** before training.
- The **sampled replicate** shows whether finite data changes the answer.

## Success and failure criteria

Computed by `analyze.py`. Ties, including both censored, are excluded from sign tests; censored runs count as +inf.

- **Supported:**
  - for both quarter and half, t_half(c2) is shorter than in `none` in at least 12 of the untied paired seeds, with sign-test p < 0.01;
  - and the Spearman correlation between shared cores and the median t_half(c2), over the 6 families, is <= -0.8.
- **Partly supported:** exactly one of these holds. State which.
- **Not supported:** neither quarter nor half beats `none` at p < 0.05.
- **Mechanism (descriptive):**
  - The seed-paired sign test of K22 at c2/s2 = 0.02 against none, and K22's share by subtree.
  - The drive fractions (own, cross from y1, cross from y3, rest) over c2's rise, trusted only if the median "resolved" is within 0.9 to 1.1.
  - Supporting the mechanism requires larger K22 in quarter and half at p < 0.05, with the shared subtree's share above none's.
- **Location (exploratory):** the paired sign tests quarter vs split2 and half vs split4, reported with their p-values. No support claim.
- **Sampled replicate:** consistent if the paired sampled/population ratios of median t_half(c1) and t_half(c2) have 95% intervals inside [0.8, 1.25]. Report it either way.
- **Inconclusive:** fewer than 8 of 16 seeds learn c2 in quarter, or in half, or in none, so medians are infinite.
- **Run failed:** checks fail, CUDA is unavailable, or no population family ran.

## Time and cost

- **Setup:** about 10 minutes.
- **Population groups:** 6 groups of 16 runs, concurrent, each up to 2e6 steps at an estimated 2–5 ms (GPU not measured yet). The pilot suggests t = 1e5 is enough for most seeds; any still training are censored at the deadline.
- **Sampled group:** 16 runs x 2^17 inputs in fp64, roughly 30–100 ms/step. It reaches only t ~ 8e3–3e4 by the deadline, so it will be partly censored; that is reported.
- **Cost:** training is capped at 4.5 h, inside the 5.5 h worker cap. The worst case is 5.5 h x $2.19/h = $12.05 on the A100 SXM (the PCIe type, at $1.79/h, was out of stock at launch).

## If something fails

- **Checks fail or CUDA unavailable:** stop, finish the worker, mark failed. Don't switch to CPU.
- **A group crashes:** the others continue. Rerun only that group once into `results/rerun`, with `--groups <name> --hours <left - 0.75>`, and note an amendment.
- **Out of memory in the sampled group:** rerun it alone with `--seeds 8` into `results/rerun` as an amendment.
- **More than 25% of a group diverges:** rerun that group once with `--eta 0.025` into `results/rerun` as an amendment.
- **Time running out:** everything finished is saved. Write up what exists.
- **No A100 in stock:** the launcher waits and retries. If neither type is found, report that and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Which loss?** The user chose population training plus a sampled replicate; the replicate is the quarter family at N = 2^17.
- **Strengths and init?** The user agreed to s = (1, 0.5, 0.75), alpha = 1 and 16 seeds. eta = 0.05 is stable in CPU probes, while 0.3 was not.
- **Budget?** The user set $10 per worker and $50 for the batch. At launch the A100 PCIe was out of stock and the SXM cost $2.19/h, so the user raised the per-worker budget to $12.05 (5.5 h), $48.20 for the batch.

## Review

Reviewed by the experiment-reviewer. The draft compared "quarter vs half vs separate" left features.

- **P0, the K12 grid couldn't resolve c1's escape** (a rise about 5 time units wide; K was taken every t = 50). K is now recorded at every change during rises (adaptive logging).
- **P0, the timescale rested on one seed, and the censoring definition was wrong.** A 16-seed pilot now sets expectations: t_max = 1e5, runs are censored per run, and the sampled group is expected to be partly censored.
- **P1, the order criterion was dominated by escape-time noise.** The primary measure is now the seed-paired t_half(c2) against `none`, by sign test; c2-before-c3 is dropped from the criteria.
- **P1, |K12| dropped the sign.** Replaced by the signed drive decomposition (K21 e1 vs K22 e2).
- **P1, K12 doesn't drive c2.** That is now an explicit mechanism criterion ("reuse, not coupling").
- **P2s.** Location claims are exploratory. Checks now fail the run. The code is committed and snapshotted. Outcome classes are complete. `analyze.py` is written before the run.

A sibling review found the basis-permutation symmetry. It made the old quarter and half teachers exactly equivalent and the old "separate" teacher share q2, which is why this redesign uses agreement sets.

## Amendments

- **2026-11-12 18:55 UTC, checks rerun with a thread cap (supervisor).** The first `studies.ttn.checks` step ran 11 minutes without finishing on the worker. The container has a CPU quota of 13.6 cores (`cpu.max` 1360000/100000) but `nproc` reports 128, and `checks.py` imports torch before `ctn.io` sets `OMP_NUM_THREADS=1`, so torch ran 193 threads on tiny CPU tensors under CFS throttling. The checks were killed and rerun unchanged except for `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` in front of the command. This changes no computed value, only the thread count.
- **2026-11-12 22:26 UTC, sampled replicate stopped to protect the population groups (supervisor).** At 3.5 h of training no population group had finished, although the startup step times (5.95–8.30 ms, 2e6 steps) put split2 at about 22:13. Five population groups have a step budget above the 2e6 steps they want, so the 4.5 h deadline doesn't shorten them, `train()` has no wall-clock check, and a group saves only when it ends: an overrun past the worker's hard cap (about 00:11 UTC) would lose every unfinished family. `nvidia-smi pmon` showed the `quarter_sampled` process (32 GB, 261 ms/step) taking about 66% of SM time while the six population processes got about 16% each when scheduled: without MPS the processes time-slice the GPU. The sampled group's deadline budget only reached t <= 3097, where the pilot had c2 learned in about 3/16 runs even by t = 5000, so its h2 ratio would be almost entirely censored. The `quarter_sampled` process was killed; the finite-data replicate is not reported. The population families are unchanged. If a population group is still running at 23:45 UTC it will be killed too, so that the saved groups can be analysed and published before the cap.
- **2026-11-12 22:52 UTC, population groups restarted with a deadline that fits the cap (supervisor).** At 22:51 (3.9 h of training, 1.34 h left on the worker cap) still no group had saved. The worker grants no ptrace, so step counters can't be read. The startup step times were evidently taken before the sampled group began its heavy kernels, so the population groups ran at an unknown fraction of their measured speed (at least 1.2x slower for split2, likely 2–3x given the 66% time-slice share); at that rate they would end around or after the cap and lose everything, since a group saves only when it ends. The run was killed and the six population groups (`--groups none,leaf,split2,quarter,split4,half`) restarted into `results/rerun` with `--hours 0.85`, in the definition's rerun form. Each group's step budget is then set by the deadline (the six groups alone match the startup timing conditions), so every group saves by about 23:47 UTC, leaving about 25 minutes for analysis, plots and publishing before the cap. The definition's crash-rerun margin of 0.75 h would leave only 0.58 h; 0.85 h was chosen because the remaining steps (analysis, plots, `orch finish`) take about 15 minutes. Consequence: groups reach roughly t ~ 2e4 instead of 1e5, so far more runs are censored than planned; the pilot had quarter vs none at 11/16 vs 3/16 c2 learned by t = 2e4. analyze.py and plot.py run on `results/rerun`. The 4 h of the first attempt are lost.
- **2026-11-13 00:02 UTC, rerun stopped to publish before the cap (supervisor, recorded after acting).** The rerun overran its 0.85 h deadline as well: leaf ran 11.6 ms/step against a startup timing of 8.6 ms. Leaf saved at about 00:01. The other five groups hadn't saved by 00:02, and the worker's own last automatic publish (at 23:56:51) had come before leaf saved. The job was stopped and `orch finish` run at 00:02 with about 9 minutes left on the cap, so that leaf's results were kept. Nothing else was saved.

## Outcome

**Failed: no answer.** Only the leaf family saved; none of the definition's outcome classes can be evaluated (its "Run failed" class doesn't strictly apply, because the checks passed and one population family ran). Write-up: [README.md](README.md).

- Supported / partly / not supported (quarter and half against none, Spearman over the 6 families): not evaluable, because none, quarter and half have no results.
- Mechanism (K22 against none, drive fractions): not evaluable against none. Leaf alone gave own 5.48, cross from y1 0.00, cross from y3 0.00, rest -4.48 and resolved 1.00, from the 2 seeds that learned c2.
- Location (quarter vs split2, half vs split4): not evaluable.
- Sampled replicate: stopped at 3.5 h (amendment), no results.
- Inconclusive (fewer than 8/16 learn c2): not evaluable for quarter, half and none. Leaf learned c2 in 2/16 by t = 1.78e4.
- Checks: all passed.
- Cost and time: one A100 SXM at $2.19/h for about 5.33 h, about $11.7 against the $12.05 budget, inside the 5.5 h cap.
