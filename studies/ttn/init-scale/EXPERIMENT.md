---
name: init-scale
title: "Init scale: activation variance, small-init plateaus, and the depth law in the 15-core TTN"
status: approved
gpu: NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe
count: 1
hours: 5.5
budget_usd: 12.05
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-12T18:37:27Z
---

# Init scale: activation variance, small-init plateaus, and the depth law in the 15-core TTN

## Question

Three questions about the variance-controlled initialization of the eight-leaf TTN:

1. Does it keep activations at unit scale through the tree?
2. How long are the plateaus before a planted product is learned, at that init and as all 15 cores are scaled down by alpha? This is the practical calibration the other TTN experiments need.
3. Does the escape follow the depth-15 product law we use for the small coupled models, judged per seed against each seed's own initial overlap?

## Hypothesis

- **Part A (descriptive).** At alpha = 1 every layer's second moment is about 1. The spread across inits grows with depth and shrinks with r.
- **Small initial coefficients.** Even at alpha = 1 the teacher coefficients start near 9^-4 ~ 1e-4, because unit output variance is spread over 9^8 basis functions. The CPU pilot measured |c(0)| ~ 1e-4 exactly.
- **Escape timescale.** The plateau is long and its length varies a lot across seeds. A 16-seed CPU pilot on this teacher had learned c1 in 4/16 seeds by t = 3000 and 8/16 by t = 5000.
- **Homogeneity (a sanity check, not a test).** The output is homogeneous of degree 15 in the parameters, and every alpha group uses the same inits. Scaling alpha therefore rescales time exactly by alpha^-13 as alpha -> 0, so the median half-time exponent is about 13 whatever the escape mechanism. Deviations at alpha near 1 come from the alpha^15 self-term.
- **Depth law (the test).** At alpha = 1, a depth-n product escaping from its own initial coefficient c1(0) predicts per-seed log t_half(c1) = const - (1 - 2/n) log|c1(0)|. That is a slope of -13/15 = -0.867 for n = 15, -0.6 for n = 5, and -0.33 for n = 3.
- **Against the depth-15 escape:** a 95% interval for the slope that excludes -0.867. An interval that also excludes -0.6 rules out an effective depth of 5 or more.

## Method

- **Model.** `ctn/ttn.py`: eight-leaf binary TTN, d = 8 coordinates per patch, phi_j = (1, x^(j)), bond dimension r = 8, 10 outputs. Every contraction goes through the cores; the coefficient tensor is never formed. Everything runs in float64. At small alpha, float32 drops most updates: at alpha = 0.6, 72% of per-step updates fall below one ulp.
- **Init.** Entries are N(0, (alpha sigma)^2), with sigma = 1/3 for the leaves and 1/r for the fusions and the root.
- **Part A (no training).** 64 inits per (r, alpha), with r in {4, 8, 16, 32} and alpha in {1, 0.9, ..., 0.5}, evaluated on 8192 random +-1 inputs.
- **Teacher.** `teachers.agreement('quarter')`, with s = (1, 0.5, 0.75):
  - y1 = prod_j x^(j)_1;
  - y2 agrees with y1 on leaves 1 and 2 and uses x_2 elsewhere;
  - y3 = prod_j x^(j)_3.

  (By the basis-permutation symmetry, product degree doesn't matter; see `studies/ttn/teachers.py`.)
- **Training.** Full-batch GD on the exact population loss 1/2 sum_o ||T_o - T*_o||^2, which equals the expected squared error for independent +-1 inputs. It is computed through the cores by Gram contractions. Outputs 4 to 10 are masked. eta = 0.05 and t = eta x steps.
- **Groups.** alpha in {1, 0.95, 0.9, 0.85, 0.8}, plus alpha = 1 at eta = 0.025 (discretization control). Each group has 32 seeds, and the same 32 inits (seed 1000) are used in every group. Each group's horizon is t_max = 1e5 x alpha^-13. The six groups run as parallel processes on the one GPU, each limited by the 4.5 h training deadline. A group stops when every run has |c_k/s_k - 1| < 1e-3 and loss < 1e-8, or at t_max or the deadline. Runs still training are censored.
- **Logging.** Exact c is evaluated every 10 steps and recorded whenever it changes, by 1% in c/s or 0.1 in log|c|. The mobility K (by subtree) is recorded at every record while a component is rising, so each escape is resolved. Alignment and spectra on a 16384-sample probe are recorded on a coarse schedule.
- **Half-time** is the first time c_k reaches s_k/2. A run that never reaches it is censored and counts as +inf in medians, never dropped.

## Inputs

None. Inputs are generated (seeded independent +-1). No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
uv run --frozen python -m studies.ttn.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/ttn/init-scale/run.py --out "$EXP_DIR/results" --hours 4.5
PYTHONPATH=. uv run --frozen python studies/ttn/init-scale/analyze.py "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/ttn/init-scale/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero if any value is outside its tolerance (listed in `studies/ttn/checks.py`), which stops the job. Run the training command as its own `orch exec` step from a background command, because it takes hours. It prints each group's measured ms/step and step budget first, so feasibility shows within minutes.

## Expected outputs

- `results/checks.json`.
- `results/activations.json`.
- `results/alpha{a}_eta{e}.{npz,json}` plus `_params.pt`, one set per group (each npz is under ~100 MB).
- `results/analysis.json`.
- `results/source/`: the exact code used.
- `figures/*.pdf`.
- `README.md`: the write-up.

## Controls

- **Correctness checks before training**, which fail the run if any is out of tolerance:
  - the dense-T and Gram contractions;
  - autograd against finite differences;
  - exact teachers;
  - the population loss against Monte Carlo;
  - the mobility decomposition and first-order dc;
  - freezing and masking;
  - the basis-permutation symmetry.
- **Discretization.** eta/2 against eta, paired by seed.
- **Same inits in every alpha group**, so groups differ only in scale.

## Success and failure criteria

Computed by `analyze.py`.

- **Depth law, the primary criterion.**
  - Supported if, at alpha = 1 with at least 20 of 32 seeds learning c1, the 95% bootstrap interval of the slope of log t_half(c1) on log|c1(0)| contains -0.867.
  - Not supported if it excludes -0.867. Report also whether it excludes -0.6.
  - Inconclusive if fewer than 20 seeds learned c1, or the interval is wider than 0.6.
- **Discretization.** The median paired eta/2-to-eta ratio of t_half(c1) must lie within [0.98, 1.02]. Otherwise every time in the batch carries a step-size error: report it, and the depth-law result becomes inconclusive.
- **Descriptive, with no pass or fail:**
  - Part A;
  - the alpha exponent (expected to tend to 13);
  - the number of seeds learned by the deadline per alpha (the practical calibration);
  - c2-before-c3 at alpha = 1.
- **Run failed:** checks fail, CUDA is unavailable, or the alpha = 1, eta = 0.05 group did not run.

## Time and cost

- **Setup:** about 10 minutes.
- **Step time:** the GPU step time is not measured yet. A 16-run fp64 step takes about 5.6 ms on one CPU thread. The GPU step is launch-bound, likely 2–5 ms, and the groups run concurrently.
- **Horizon:** alpha = 1 reaches t = 1e5 (2e6 steps) in about 1–3 h. Smaller alpha will be censored at the deadline, and that is reported.
- **Cost:** training is capped at 4.5 h, inside the 5.5 h worker cap. The worst case is 5.5 h x $2.19/h = $12.05 on the A100 SXM (the PCIe type, at $1.79/h, was out of stock at launch).

## If something fails

- **Checks fail, or CUDA is unavailable:** stop, finish the worker, and mark failed. The scripts refuse to run the grid on CPU.
- **A group crashes:** `run_parallel` keeps the others running and prints the failure. Rerun only that group once:

  ```bash
  ... run.py --out "$EXP_DIR/results/rerun" --groups <name> --hours <hours left - 0.75>
  ```

  Then note it under Amendments.
- **More than 25% of a group's runs diverge:** rerun that group once with `--eta 0.025` into `results/rerun`, and record an amendment.
- **Measured ms/step makes alpha = 1 unable to reach t = 2e4 within the deadline:** continue anyway (the deadline censors). Report it, since that affects the depth-law criterion.
- **Time running out:** finished groups and partial logs are saved. Write up what exists and never extend the worker cap.
- **No A100 in stock:** the launcher waits and retries. If neither A100 type is found at launch, report that and stop. Don't switch to a GPU type the budget doesn't cover.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Which loss?** Population for all groups, plus a sampled replicate in sharing-location. The user chose this.
- **Teacher strengths and the alpha grid?** The user agreed to s = (1, 0.5, 0.75) and alpha in {1, 0.9, 0.8, 0.7, 0.6}. After review, the grid is narrowed to {1, 0.95, 0.9, 0.85, 0.8}, for two reasons:
  - homogeneity makes small alpha a pure time rescaling, so it can't test anything new;
  - alpha <= 0.7 can't finish in the budget (about 100x the alpha = 1 time).
- **Degree slot?** Replaced by a constant-biased-init degree experiment. The user chose this; see the degree definition.
- **Budget?** The user set $10 per worker and $50 for the batch. At launch the A100 PCIe was out of stock and the SXM cost $2.19/h, so the user raised the per-worker budget to $12.05 (5.5 h), $48.20 for the batch.

## Review

Reviewed by the experiment-reviewer on the first draft. Findings and changes:

- **P0, fp32 drops updates at small alpha.** Everything now runs in float64.
- **P0, the exponent test can't fail (homogeneity).** The exponent is now a descriptive sanity check. The primary test is the per-seed depth-law slope at fixed alpha.
- **P1, the time split starved small alpha.** Groups now run as parallel processes, each with the full deadline, and the measured step budget is printed per group.
- **P1, Part A criteria hold by construction.** Part A is now descriptive.
- **P1, checks couldn't fail.** Tolerances were added; the script exits non-zero.
- **P1, the ordering was counted repeatedly over alpha and had a weak control.** It is now reported once, at alpha = 1, as descriptive; sharing is tested in sharing-location.
- **P1, code wasn't committed or snapshotted.** Everything is committed at approval, and run.py copies its sources into `results/source`.
- **P2, nanmedian dropped censored runs.** Censored runs now count as +inf.
- **P2, no plan for a missing A100.** A100 PCIe was added, at the same price.

A sibling review also established the basis-permutation symmetry and the slow, heavy-tailed escapes. Both are reflected above.

## Amendments

None.

## Outcome

Not run yet.
