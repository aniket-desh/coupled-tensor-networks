---
name: sequential
title: "Sequential transfer: add output 2 after output 1, freeze vs update vs fresh tree"
status: approved
gpu: NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe
count: 1
hours: 5.5
budget_usd: 12.05
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-12T18:37:29Z
---

# Sequential transfer: add output 2 after output 1, freeze vs update vs fresh tree

## Question

After the TTN has learned output 1, does a new output that reuses part of output 1's computation learn faster than it would from scratch? Which subtrees carry that transfer? And does learning the new output disturb the old one?

Freezing quarter 1, or the whole left subtree, against updating everything or training a fresh tree, separates two explanations: the reused computation is already in place, or the network re-learns it.

## Hypothesis

Teachers come from `teachers.agreement(S, s=(1, s2), third=False)`:

- y1 = prod_j x^(j)_1, identical in every family;
- y2 uses x_1 on the leaves in S and x_2 elsewhere.

Families: `none` (S = {}), `quarter` (S = q1 = {0, 1}) and `half` (S = left half = {0, 1, 2, 3}), with s2 in {0.5, 1.0}. By the basis-permutation symmetry, a fresh tree faces the same problem for every family's y2, so the fresh baseline is pooled per s2.

Predictions for the time to learn y2 (phase-2 half-time):

- **Transfer.** In quarter and half, `all` learns y2 faster than in `none`, paired by phase-1 seed. Both quarter and half also beat the pooled fresh tree.
- **Where transfer lives:**
  - quarter: `freeze_q1` stays close to `all`, because q1 already computes the shared part. `freeze_left` blocks learning, because W_L and q2 must build y2's different left-half feature.
  - half: `freeze_left` stays close to `all`, because the whole left half is already right.
  - none: both freezes block learning. This is a representability sanity check, not evidence.
- **Interference.** While y2 is learned, c1 moves in `all`. The peak |c1/s1 - 1| is larger in `all` than in the arms that freeze the shared subtree.
- **Against:** no paired speed-up over `none` in quarter and half, or freeze_q1 slowing quarter as much as freeze_left.

## Method

- **Model and init.** As in `ctn/ttn.py`: r = 8, d = 8, 10 outputs, alpha = 1, float64.
- **Training.** Full-batch GD at eta = 0.05 on the exact population loss, t = eta x steps.
- **Phase 1, run once.**
  - Output 1 alone from 16 inits (seed 4000). Output 2's loss is masked, not given a zero target.
  - It runs until every seed has |c1 - 1| < 1e-3 and loss < 1e-8, or t = 1e5, or 45% of the 4.5 h deadline. Escapes are heavy-tailed: a CPU pilot had 1 of 4 seeds escape by t = 2500.
  - A seed passes the gate if |c1 - 1| < 1e-2 at the end of phase 1. Seeds that fail are excluded from every arm and counted.
  - The parameters are saved (`phase1_params.pt`).
- **Phase 2.** For each (family, s2), four arms per seed branch from the saved state:
  - `all`: every core trains.
  - `freeze_q1`: A1, A2 and W1 frozen.
  - `freeze_left`: A1 to A4, W1, W2 and W_L frozen.
  - `fresh`: a new independent tree (seed 4500) trained on y2 alone.
  - Each group has 64 runs, and the 6 groups run as parallel processes for the remaining time.
- **Measurements.**
  - Exact c is recorded on every change, and K by subtree is recorded densely during rises. Frozen cores are excluded from K.
  - The phase-2 half-time of c2 is counted from the phase-2 start.
  - Interference is the peak |c1/s1 - 1| over phase 2.

## Inputs

None. Inputs are generated (seeded independent +-1). No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
uv run --frozen python -m studies.ttn.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/ttn/sequential/run.py --out "$EXP_DIR/results" --hours 4.5
PYTHONPATH=. uv run --frozen python studies/ttn/sequential/analyze.py "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/ttn/sequential/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero if any value is out of tolerance. Run the training as its own `orch exec` step from a background command.

## Expected outputs

- `results/checks.json`.
- `results/phase1.{npz,json}` and `phase1_params.pt`.
- `results/{none,quarter,half}_s2{0.5,1.0}.{npz,json}` plus `_params.pt`, with run index = arm x 16 + seed and arms in the order all, freeze_q1, freeze_left, fresh.
- `results/analysis.json`.
- `results/source/`.
- `figures/sequential.pdf`.
- `README.md`: the write-up, with the per-(family, s2, arm) table from `analyze.py`.

## Controls

- `none` is the no-sharing control, compared seed by seed against quarter and half through the shared phase 1.
- The `fresh` tree is the no-transfer baseline, pooled per s2.
- All arms branch from the identical phase-1 state.
- Correctness checks before training. They include that frozen cores don't move and drop out of K, and that masked outputs don't affect the gradient.

## Success and failure criteria

Computed by `analyze.py` over seeds that passed the phase-1 gate; censored runs count as +inf, and ties are excluded from sign tests.

- **Transfer, primary.** Supported if, for quarter and for half at both s2, `all` learns y2 faster than `none` in at least 12 of the untied paired seeds, with sign-test p < 0.01.
- **Location of transfer:**
  - quarter: the median freeze_q1 / all ratio of t_half(c2) is <= 1.5, and freeze_left is >= 3x all or more than half censored;
  - half: freeze_left / all <= 1.5.
  - Report these per s2. They are part of "supported" only if the transfer criterion holds.
- **Partly supported:** transfer holds, but a location prediction fails (state which), or the reverse.
- **Not supported:** neither quarter nor half beats `none` at p < 0.05 at either s2.
- **Inconclusive:** fewer than 10 of 16 seeds pass the phase-1 gate, or more than half of `none`'s `all` arm is censored at both s2 (no baseline to compare against).
- **Descriptive:** transfer against the pooled fresh tree, interference per arm, and K12 by subtree at phase-2 onset.
- **Run failed:** checks fail, CUDA unavailable, or phase 1 didn't run.

## Time and cost

- **Setup:** about 10 minutes.
- **Phase 1:** 16 runs for up to 45% of the deadline (about 2 h), likely 2–5 ms/step (GPU not measured).
- **Phase 2:** 6 groups of 64 runs, concurrent, for the remaining about 2.5 h. Frozen arms that can't learn will run to the deadline and be censored, as expected.
- **Cost:** training is capped at 4.5 h, inside the 5.5 h worker cap. The worst case is 5.5 h x $2.19/h = $12.05 on the A100 SXM (the PCIe type, at $1.79/h, was out of stock at launch).

## If something fails

- **Checks fail or CUDA unavailable:** stop, finish the worker, mark failed.
- **Phase 2 crashes, or one group fails:** rerun only the phase-2 groups from the saved phase 1 into `results/rerun`, with `--groups <names> --hours <left - 0.75>`. This reuses `results/phase1_params.pt`; copy `phase1.json` and `phase1_params.pt` into `results/rerun` first. Note an amendment.
- **Phase 1 crashes:** rerun everything once into `results/rerun` with `--hours <left - 0.75>`.
- **More than 25% of a group diverges:** rerun that group once with `--eta 0.025` into `results/rerun` as an amendment.
- **Time running out:** everything finished is saved. Write up what exists.
- **No A100 in stock:** the launcher waits and retries. If neither type is found, report that and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Which loss?** Population. The user chose population plus a replicate; the replicate is in sharing-location.
- **Grid?** The user agreed to alpha = 1, 16 seeds and eta = 0.05. Here s1 = 1 and s2 in {0.5, 1.0}.
- **Arms?** Update-all, freeze quarter 1, freeze the left subtree, and a separate fresh tree, as the user specified, with masked inactive outputs.
- **Budget?** The user set $10 per worker and $50 for the batch. At launch the A100 PCIe was out of stock and the SXM cost $2.19/h, so the user raised the per-worker budget to $12.05 (5.5 h), $48.20 for the batch.

## Review

The first draft was reviewed. Findings and changes:

- **P0, the commands broke after code changes.** The code is ported and smoke-tested, and the Method matches it.
- **P0, the "separate" family shared q2 and "quarter" shared the whole left half** (the constant is a shared feature under the symmetry). Families are now explicit agreement sets: none, quarter and half.
- **P0, the time plan was contradicted by a pilot, and phase 1 ran four times.** Phase 1 now runs once with a 45% share and a per-seed gate, and its parameters are saved for branching and reruns.
- **P1, censored seeds were dropped.** They now count as +inf.
- **P1, the family contrast was unpaired and had no margin.** It is now paired by phase-1 seed, with a sign test, and the fresh tree is pooled per s2.
- **P1, gate-failing seeds still entered phase 2.** They are now excluded from every arm and counted.
- **P2s:**
  - interference is measured as peak |c1 - 1|, both directions;
  - freeze_left has a criterion;
  - the "none" freeze test is labelled a sanity check;
  - reruns are possible with `--groups`, `--eta`, saved parameters and `results/rerun`.

## Amendments

None.

## Outcome

Not run yet.
