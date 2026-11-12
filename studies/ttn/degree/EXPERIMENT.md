---
name: degree
title: "Degree as a difficulty axis under a constant-biased leaf init"
status: approved
gpu: NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe
count: 1
hours: 5.5
budget_usd: 12.05
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-12T18:37:28Z
---

# Degree as a difficulty axis under a constant-biased leaf init

## Question

With phi_j = (1, x^(j)) and an isotropic leaf init, product degree is invisible to population GD, because permuting a leaf's basis, including the constant, is an exact symmetry (checked in `studies/ttn/checks.py`). The reason for keeping the constant feature is that a network can ignore patches. Degree becomes a real difficulty axis only if the init favours ignoring: each leaf starts mostly on the constant.

So the question is: when a fraction w of each leaf's init variance sits on the constant, how does the time to learn a product grow with the number of patches D it uses? Does that growth match the depth-15 escape law?

## Hypothesis

- **Teacher.** One output, y = prod_{j < D} x^(j)_1, with the constant on the remaining patches (`teachers.degree(D)`), s = 1, D in {1, 2, 4, 6, 8}.
- **Init.** The TTN leaf rows have variance w on phi_0 and (1 - w)/8 on each x coordinate. The leaf second moment stays 1. w = 1/9 is the isotropic default; w = 0.25 and w = 0.5 are biased.
- **Prediction.** Each active patch scales the initial coefficient by sqrt((1 - w)/(8w)). If escape time scales as c0^-(13/15) (the depth-15 law that init-scale tests), then

  d log(t_half) / dD = (13/15) x 0.5 x log(8w / (1 - w))

  This gives 0 at w = 1/9, 0.425 at w = 0.25, and 0.901 at w = 0.5, in natural log per patch.
- **Symmetry control.** At w = 1/9, D = 1 and D = 8 have the same t_half distribution.
- **Against:** no growth with D under bias, growth under the isotropic init, or slopes far from the prediction.

## Method

- **Model.** As in `ctn/ttn.py`: r = 8, d = 8, 10 outputs (only output 1 trained; the rest masked), alpha = 1. Fusion and root std 1/8; leaves as above. Float64.
- **Training.** Full-batch GD at eta = 0.05 on the exact population loss, t = eta x steps.
- **Groups.** (w, D) for w in {0.25, 0.5} and D in {1, 2, 4, 6, 8}, plus w = 1/9 at D in {1, 8}: 12 groups. Each has 16 seeds, with the same init seed (3000) for every group of a given w. Groups run as parallel processes on the GPU (8 at a time, the rest queued).
- **Stopping.** A group stops at |c/s - 1| < 1e-3 and loss < 1e-8 for every run, or at t = 1e5, or at the 4.5 h deadline. Runs still training are censored and count as +inf.
- **Logging.** Adaptive (exact c on every change), as in the other TTN experiments.

## Inputs

None. Inputs are generated (seeded independent +-1). No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
uv run --frozen python -m studies.ttn.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/ttn/degree/run.py --out "$EXP_DIR/results" --hours 4.5
PYTHONPATH=. uv run --frozen python studies/ttn/degree/analyze.py "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/ttn/degree/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero if any value is out of tolerance. Run the training as its own `orch exec` step from a background command.

## Expected outputs

- `results/checks.json`.
- `results/w{w}_D{D}.{npz,json}` plus `_params.pt`, for the 12 groups.
- `results/analysis.json`.
- `results/source/`.
- `figures/*.pdf`.
- `README.md`: the write-up, with a table of learned/16 and median t_half (95% interval) per (w, D), and the fitted slopes against the predictions.

## Controls

- **Isotropic w = 1/9 at D = 1 and 8:** the symmetry says these match, so a difference would expose a bug or a finite-step effect.
- **The same inits across D** within each w.
- **Correctness checks** before training, including the basis-permutation symmetry.

## Success and failure criteria

Computed by `analyze.py`.

- **Symmetry control.** The w = 1/9, D = 1 vs D = 8 Mann-Whitney test must give p > 0.01. Otherwise report a failed control and treat the slopes as inconclusive.
- **Supported:** for both w = 0.25 and w = 0.5, using the D values where at least 12 of 16 seeds learned (at least 3 such D values), both of these hold:
  - the per-run regression of log t_half on D has a 95% bootstrap interval above 0;
  - the w = 0.5 slope exceeds the w = 0.25 slope.
- **Quantitative agreement (reported, not required for support):** each slope's interval contains its predicted value (0.425 and 0.901).
- **Not supported:** either slope's interval contains 0.
- **Inconclusive:** fewer than 3 learned D values for either w. A high-D censored tail is expected at w = 0.5.
- **Run failed:** checks fail, CUDA unavailable, or neither D = 1 group ran.

## Time and cost

- **Setup:** about 10 minutes.
- **Groups:** 12 groups of 16 runs, about 8 concurrent. The single-output population step is cheaper than in the three-output teachers.
- **Expected tail:** large D at w = 0.5 is predicted to be about e^(0.9 x 7) ~ 500x slower than D = 1. It will be censored, and that is reported.
- **Cost:** training is capped at 4.5 h, inside the 5.5 h worker cap. The worst case is 5.5 h x $2.19/h = $12.05 on the A100 SXM (the PCIe type, at $1.79/h, was out of stock at launch).

## If something fails

- **Checks fail or CUDA unavailable:** stop, finish the worker, mark failed.
- **A group crashes:** the others continue. Rerun only that group once into `results/rerun`, with `--groups <name> --hours <left - 0.75>`, and note an amendment.
- **More than 25% of a group diverges:** rerun that group once with `--eta 0.025` into `results/rerun` as an amendment.
- **Time running out:** everything finished is saved. Write up what exists, with censoring per group.
- **No A100 in stock:** the launcher waits and retries. If neither type is found, report that and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **What should the degree slot become, given that degree is a symmetry under the isotropic init?** The user chose the constant-biased init.
- **Which loss?** Population. The user chose population plus a replicate; the replicate is in sharing-location.
- **Bias values w?** The user chose the constant-biased init but didn't pick values. I chose 0.25 and 0.5 so that the predicted slowdowns fit the budget; w = 0.9 would predict about 1e6x at D = 8.
- **Budget?** The user set $10 per worker and $50 for the batch. At launch the A100 PCIe was out of stock and the SXM cost $2.19/h, so the user raised the per-worker budget to $12.05 (5.5 h), $48.20 for the batch.

## Review

The draft (a shared-vs-separate degree sweep under the isotropic init) was reviewed. The findings and changes:

- **P0, degree is an exact symmetry under the isotropic init.** The experiment is redefined around a constant-biased init (the user's choice), with the isotropic init kept as a symmetry control and the symmetry added to the checks.
- **P1, censored c2 was dropped from ratios.** It now counts as +inf everywhere, and the shared/separate ratio criterion is gone.
- **P1, no uncertainty.** Bootstrap intervals on the slopes and a Mann-Whitney test for the control.
- **P2, every group was labelled censored.** Censoring is now per run and the stop reason per group.

## Amendments

None.

## Outcome

Not run yet.
