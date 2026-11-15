---
name: hmm-belief-geometry
title: Belief-state geometry of hidden Markov processes in a transformer, a chain and a tree tensor network
status: done
gpu: NVIDIA RTX A6000,NVIDIA RTX PRO 4500 Blackwell,NVIDIA RTX PRO 4500 Blackwell Server Edition,NVIDIA RTX 6000 Ada Generation
count: 1
hours: 8
budget_usd: 10
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-14T23:44:19Z
result: inconclusive
finished_utc: 2026-11-15T08:58:13Z
writeup: studies/hmm/hmm-belief-geometry/README.md
---

# Belief-state geometry of hidden Markov processes in a transformer, a chain and a tree tensor network

## Question

Shai et al. (2024, arXiv 2405.15943) found the Bayesian belief state of a hidden Markov process laid out linearly in a
transformer's residual stream: the Mess3 fractal, and the 36 belief states of RRXOR. The two RHM experiments
(`rhm-belief-geometry`, `rhm-ttn-belief-geometry`) apply their probe to a grammar. This experiment goes back to their
processes and asks three things on the same data and with the same probes:

1. Does our pipeline reproduce their transformer result on Mess3 and RRXOR?
2. Do tensor networks trained on the same next-token task carry the belief, and in what form? Exact filtering is
   u_{t+1} = u_t T^(x_t), η_t = u_t / Σu_t, which is bilinear in (state, token) and then normalized.
   - A chain (matrix product state read left to right) has exactly this update, h_t = W(h_{t−1}, token). If its state is
     a linear image of the unnormalized message u_t, the belief is a ratio of linear functions of the state, not an
     affine function of it.
   - A tree's merge can multiply transfer matrices. Up to the midpoint only the left top bond has seen tokens. Past it,
     η_t ∝ η_H M_R is a product of something the left bond knows and something the right bond knows.
3. On processes with product structure, which belief is recoverable: the joint, or only the factors' marginals?

It runs concurrently with the RHM experiments and reuses their probe code, so results can be read side by side.

## Hypothesis

All four are judged on held-out prefixes at eligible positions (defined under Method), in (process, model) cells that
learn. A cell learns if, in all three seeds, its excess test loss is below min(0.02, half the excess of the optimal
predictor that sees only the last two tokens). On mess3, factored and coupled that short-context predictor is already
within about 0.015 nats/token of Bayes, so a fixed 0.02 would pass a model that has not learned to filter.

**How margins are measured here.** A probe "beats" a control if it leaves at most half of the control's unexplained
variance: (1 − R²_probe) ≤ 0.5 × (1 − R²_control). The RHM experiments used a fixed margin of 0.1 in R². That cannot
work on Mess3, where the pilot found the belief almost linear in the one-hot prefix (token readout R² 0.96–0.97 on
held-out prefixes) and almost additive in the two halves, so the controls leave less than 0.1 to gain.

**Saturated controls.** If the initial network or the token readout already scores R² ≥ 0.99 on a criterion's
positions in some seed, the cell is "not testable (control saturated)" for that hypothesis and leaves the count.
Halving an error that small asks for less than the probe's own noise. The gate is 0.99, above the 0.9 used for the
ceilings below, because Mess3's controls sit at 0.96–0.975 and a trained probe can still be told apart there (0.998 in
the pilot). The pilot's `coins3` chain is the case this rule is for: its identity-init chain scored 0.998.

- **H1, reproduction (transformer; Mess3 and RRXOR).** The affine probe on the last layer recovers the belief with
  pooled R² ≥ 0.8 and beats both the same probe on the initial network and the token readout. On RRXOR it also beats
  the next-token ceiling at the eligible positions where that ceiling is below 0.9.
  - Why: this is the published result. RRXOR is the process where beliefs differ although the next-token distributions
    agree (after two random bits, states A0 and A1 both predict a fair bit).
  - Against: R² < 0.8, or no better than half the error of a control, or on RRXOR no better than half the error of
    the next-token ceiling.
- **H2, chain: the state determines the belief through a projective map.** The projective probe b̂ = q/Σq,
  q = W h_t + c, on the chain's state:
  - reaches pooled R² ≥ 0.8;
  - beats the projective probe on the identity-init chain and the token readout;
  - beats the next-token ceiling at the eligible positions where that ceiling is below 0.9.
  - Why: the chain's update has the form of the unnormalized filter, so a state that is a linear image of the filter
    carries the belief projectively.
  - What this does not assume: the chain is trained through softmax(U h_t), so its state need not be the unnormalized
    filter. The hand-built exact chain of `hmm_check.py` (projective R² 1.000000, affine 0.32) validates the probe; it
    does not show that a trained chain uses that code.
  - Against: the projective probe fails the threshold, a control or the next-token ceiling.
  - **Form of the code, a label fixed before the run and not part of the verdict:** the chain's code is called
    "projective" if the projective probe beats the affine probe on the same positions (same ratio rule), and "affine"
    otherwise.
- **H3a, tree, first half.** At eligible positions t ≤ T/2, the projective probe on the top bonds [h_L, h_R] reaches
  R² ≥ 0.8 and beats the same probe on the identity-init tree and the token readout.
- **H3b, tree, second half: the product of the top bonds recovers the belief where the bonds alone, an additive
  fit and the model's own prediction cannot.** At positions t > T/2 eligible for the product features
  [h_L, h_R, h_L ⊗ h_R], the projective probe on them:
  - reaches R² ≥ 0.8;
  - beats the projective probe on the top bonds alone, on the same positions;
  - beats the exact additive ceiling, at the positions where that ceiling is below 0.9;
  - beats the next-token ceiling, at the positions where that ceiling is below 0.9;
  - beats the same probe on the identity-init tree, the token readout, and the token readout with cross-half
    products where that one is eligible.
  - The additive ceiling is the best predictor of the form f(left half) + g(right prefix), computed exactly from the
    process. Every affine probe on bonds from disjoint spans is bounded by it.
  - Where this can be tested. The process has to remember its left half, and the position has to be eligible for the
    product features. The exact additive ceilings, which do not depend on the model, are:
    - `mess3`: 0.975 and 0.970 at t = 9 and 10, then 0.991 rising to 0.9999 at t = 15. Recent tokens nearly determine
      the belief, so Mess3 is not judged on H3b. That is a property of the process, not of the tree.
    - `coins3`: 0.957 at t = 9, falling to 0.897 at t = 15, and below 0.9 only at t = 14 and 15. The posterior over
      the coin is the normalized product of the two halves' likelihoods.
    - `rrxor`: 0.62 at t = 9, rising to 0.80 at t = 15, so it leaves the most room.
    - `coins3` and `rrxor` have at most 2^t distinct prefixes and never reach the 10,890 fit prefixes that the 1,088
      product features of an r = 32 tree need. So the r = 32 tree has no H3b cell, and the tree is also run at r = 8
      (80 product features, 810 fit prefixes needed). There the candidates are `rrxor` at t = 13–15 and `coins3` at
      t = 14 and 15.
  - Why: a tree built by hand to multiply transfer matrices scores projective R² 1.0000 with the product features at
    every position, and 0.955–0.992 without them past the midpoint (`hmm_check.py`). As for the chain, that validates
    the probe, not the trained tree's code.
  - Against: the product probe fails the threshold, the top bonds alone, a ceiling or a control.
- **H4, product structure (descriptive, no verdict).** On `factored` and `coupled`, each model's primary probe is fit
  to the joint belief (9 states) and to the two factor marginals, and both R² are reported. Simplex's later work
  (arXiv 2602.02385) reports that networks prefer the factored code. `coupled` is the case where the marginals do not
  determine the joint.
- **What a supported H1–H3 shows, and where it cannot show more than the model's own prediction.**
  - A supported hypothesis shows that the hidden state determines the belief through the stated probe on prefixes it
    never saw.
  - Every model's probed features contain its next-token prediction by construction: the transformer's logits are a
    linear readout of the normalized last layer, the chain's are U h_t, and the tree's are linear in h_L ⊗ h_R. So
    wherever the belief is a function of the exact next-token distribution, recovering it shows no more than that
    the model predicts well.
  - That is the case on four of the six processes. On `mess3`, `factored` and `coupled` the belief is an exact linear
    image of the next-token distribution (the emission matrix is square and full rank), and on `coins3` it is a
    one-to-one function of it. Their next-token ceiling is 1. Results there, and all of H4, say how the prediction is
    laid out in the hidden state, not that the model holds more than its prediction.
  - `rrxor` is the process that separates the two: its next-token ceiling is 0.45–0.57 at t = 5–15. That is
    why H1 on rrxor, H2 and H3b carry the next-token clause, and why the rrxor cells carry the experiment's claim
    about beliefs beyond prediction.
  - With few distinct beliefs per position (RRXOR, Z1R, coins3), a probe that identifies the belief state fits any
    labelling of the states, so the scrambled-target score and the number of distinct beliefs are printed next to
    every verdict. Mess3's beliefs are all distinct, so there the probe has to get the geometry right.

## Method

- **Processes** (`studies/hmm/processes.py`; T[x, s, s'] = P(emit x, move to s' | s); start from the stationary
  distribution unless noted), context T = 16:
  - `mess3`: x = 0.05, α = 0.85, the process of Shai et al. 3 states, 3 tokens. Its kernel matches the matrix printed
    in their appendix.
  - `rrxor`: two fair random bits, then their XOR. 5 states, 2 tokens, 36 distinct belief states (checked).
  - `z1r`: zero, one, random. 3 states, 2 tokens. It has too few distinct prefixes to be eligible anywhere, so it is
    figure-only, as in Shai et al.
  - `coins3`: a mixture of three biased coins, P(1) = 0.2, 0.5, 0.8, uniform prior. Non-ergodic: the belief is the
    posterior over the coin.
  - `factored`: independent mess3(0.05, 0.85) and mess3(0.15, 0.6). 9 joint states, 9 tokens.
  - `coupled`: as `factored`, but with probability ε = 0.3 the second factor emits token a and jumps to state a, where
    a is the first factor's current state:
    T[(xa, xb), (a, b), (a', b')] = T_A[xa, a, a'] × ((1 − ε) T_B[xb, b, b'] + ε 1[xb = a] 1[b' = a]).
- **Exact targets:** the belief η_t = P(state | x_<t) by filtering, the next-token distribution η_t T^(x) 1, and the
  Bayes loss.
- **Models**, four arms with 3 seeds each (the seed sets the init and the data stream), 6 × 4 × 3 = 72 runs
  (`hmm_run.py`):
  - `transformer`: the model of `rhm-belief-geometry` (4 layers, d 128, 4 heads, MLP 512, BOS). This is Simplex's
    RRXOR configuration; their Mess3 model was smaller (d 64, 1 head).
  - `tree`: `ctn.tree.TreeTN`, r = 32, untied, identity init, on the masked prefix task of `rhm-ttn-belief-geometry`.
  - `tree8`: the same tree at r = 8, so that the product features can be probed on processes with few prefixes.
  - `chain`: `ctn.tree.ChainTN`, r = 32, identity init, read causally (logits_t = U h_t, h_t the state after t
    tokens), so one pass trains every position like the transformer.
- **Training:** fresh sequences every update, mean cross-entropy over sequences and the 16 positions, Adam, peak LR
  1e-3, cosine to 0.1 × peak (`studies.rhm.train.lr_at`; warmup 500 updates for the transformer and 100 for the tensor
  networks, as in the RHM runs), batch 512, 30,000 updates, float32. Checkpoints at 0, 300, 1000, 3000, 10000, 30000.
  Per-position test loss on 8,192 fixed sequences is logged next to the exact Bayes loss.
  - Deviations from Shai et al.: Adam instead of SGD, 30,000 updates instead of about 10⁶, context 16 instead of 10
    (the tree needs a power of 2), a BOS token, GELU.
- **Hidden states** (`hmm_analyze.py`), at each position t = 1 .. 15:
  - transformer: `L4`, the last block's output (primary), and `concat`, all six sets;
  - chain: `state`, h_t;
  - tree: `top` = [h_L, h_R] (primary), `top_bil` = [h_L, h_R, h_L ⊗ h_R], and `all` bonds.
- **Probes**, a separate fit per position:
  - affine, b̂ = W a + c, least squares (every checkpoint);
  - projective, b̂ = q/Σq with q = W a + c (first and last checkpoint), for `L4`, `state`, `top` and `top_bil`. It is
    fit by L-BFGS from two starts, the affine solution and an algebraic solution, keeping the better fit.
  - For the product processes the same probes are also fit to the factor marginals (each factor normalized
    separately).
- **One map for all positions** (the Simplex convention; transformer and chain): a single affine map fit on rows from
  every position, reported next to the per-position fits. It is descriptive.
- **Analysis set and splits:** 60,000 sequences with a fixed seed, the same for every model of a process.
  - Sequence split (30,000 / 30,000): used for the figures.
  - Prefix split: at each position the distinct prefixes are assigned at random to fit or test. Used for all criteria.
  - Eligible positions: at least 10 × (features + 1) distinct fit prefixes. That is 1,290 for `L4`, 330 for `state`,
    650 for `top` and 10,890 for `top_bil` at r = 32, and 170 and 810 for the tree at r = 8.
- **Position groups** that the criteria pool over: `e` (eligible for the primary set); `nt` (`e` with next-token
  ceiling < 0.9); for the tree `h1` (`e`, t ≤ 8), `h2b` (eligible for `top_bil`, t > 8), `h2c` (`h2b` with an exact
  additive ceiling below 0.9), `h2n` (`h2b` with a next-token ceiling below 0.9) and `h2t` (`h2b` with an eligible
  token-product readout).
- **Figures** (`hmm_plot.py`, PDFs from saved results only), seed 0:
  - `triangle_mess3.pdf`, `triangle_coins3.pdf`, `triangle_z1r.pdf`: the belief simplex with ground truth, transformer,
    chain and the two trees side by side, trained above and the initial network below, all positions overlaid, colored by the
    true belief as RGB.
  - `pca_rrxor.pdf`, `pca_factored.pdf`, `pca_coupled.pdf`: the same layout on the top two PCs of the true beliefs,
    with per-belief centroids for RRXOR.
  - `positions.pdf`, `summary.pdf`, `training.pdf`.
- **Verdict:** `hmm_summary.py` applies the criteria and writes `results/summary.json`.
- **CPU pilot** (seed 0; 300–1,500 updates of batch 64, 8,000 analysis sequences, the tree at r = 16 in place of
  r = 32; R² on held-out prefixes, pooled over the eligible positions of that small analysis set). It ran end to end
  (train, analyze, summary, plot) on the final code for every arm on mess3 and rrxor, and for one arm on coins3,
  factored and z1r. These are previews, not results.
  - `mess3`, transformer (600 updates, excess 0.0019 against a gate of 0.0073): affine 0.998, initial network 0.975,
    token readout 0.967, one map for all positions 0.996. Next-token ceiling 1.000.
  - `mess3`, chain (1,500, excess 0.0014): affine 0.990, projective 0.992, initial chain projective 0.964, token
    readout 0.966. Form label: affine.
  - `mess3`, tree r = 16 (1,000, excess 0.0030): first half projective 0.975 (initial 0.958, token readout 0.964);
    second half product projective 0.981, where the additive ceiling is above 0.99.
  - `mess3`, tree r = 8 (800, excess 0.0074, just above the gate): first half projective 0.957; second half product
    projective 0.980.
  - `rrxor`, transformer (600, excess 0.044, not yet learned): affine 0.560, next-token ceiling 0.468.
  - `rrxor`, chain (1,500, excess 0.013): affine 0.899, projective 0.950, initial 0.155, token readout 0.053, next-token
    ceiling 0.473, scrambled target 0.743 with 17 distinct beliefs per position. Form label: projective.
  - `rrxor`, tree r = 16 (1,000, excess 0.018): no position eligible at this analysis size. At t = 10–15 the product
    projective probe scored 0.82–0.88 against an additive ceiling of 0.66–0.80, and the projective probe without the
    product 0.31–0.59.
  - `rrxor`, tree r = 8 (800, excess 0.090, not yet learned): product projective 0.245 against an additive ceiling of
    0.772 and a next-token ceiling of 0.461. A longer run of the same arm (5,000 updates of batch 128, training only)
    reached excess 0.013, under the 0.02 gate, so the r = 8 tree can learn `rrxor`.
  - `coins3`, chain (600, excess 0.0005): affine 0.941, projective 0.999, initial chain projective 0.998, token readout
    0.608, next-token ceiling 1.000. The summary marks the cell "control saturated". Form label: projective.
  - `coins3`, tree r = 8 (800): second half, top bonds affine 0.831 and projective 0.872, product affine 0.941 and
    projective 0.968, initial 0.784, token readout 0.601. At t = 14 and 15: product projective 0.972 against the
    additive ceiling 0.897.
  - `factored`, chain (300, excess 0.016 against a gate of 0.0078, not yet learned): joint belief affine 0.835 and
    projective 0.838 against a token readout of 0.917; factor marginals 0.937 and 0.945 against 0.971.
  - `z1r`, tree r = 8 (400): at most 37 distinct fit prefixes at any position, so no position is eligible, as
    expected. The analysis, summary and triangle figure run.

## Inputs

None. All data is generated from the processes. The code imports, read-only, `studies/belief_common.py`, and from
`studies/rhm/rhm-belief-geometry/` and `studies/rhm/rhm-ttn-belief-geometry/` the transformer, the prefix-task
masking, the bond extraction and the plot style.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
free -g
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
PYTHONPATH=. uv run --frozen python -m studies.hmm.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_check.py
```

Training, four runs at a time on the GPU, as its own `orch exec` step. The order is transformer, tree at r = 8,
chain, then the r = 32 tree, whose rrxor, coins3 and z1r runs come last because they enter no verdict. A run is skipped
if the file `runs/hmm-belief-geometry/STOP` exists when its turn comes:

```bash
set -euo pipefail
mkdir -p runs/hmm-belief-geometry
{
  for m in transformer tree8 chain; do
    for p in mess3 rrxor coins3 factored coupled z1r; do
      for s in 0 1 2; do echo "$p $m $s"; done
    done
  done
  for p in mess3 factored coupled rrxor coins3 z1r; do
    for s in 0 1 2; do echo "$p tree $s"; done
  done
} | xargs -P 4 -n 3 sh -c 'if [ -e runs/hmm-belief-geometry/STOP ]; then echo "SKIPPED $0 $1 s$2"; exit 0; fi; OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_run.py "$0" "$1" --seed "$2" --out runs/hmm-belief-geometry > runs/hmm-belief-geometry/train_$0_$1_s$2.log 2>&1 || echo "FAILED $0 $1 s$2"'
```

Analysis, on the worker's CPU while training runs. This command analyzes, two at a time, every run that has a
`result.json` and no `analysis.json` yet. Run it as a separate `orch exec` step in batches during training, starting
a batch only after the previous one has finished (an unfinished analysis has no `analysis.json`, so an overlapping
batch would repeat it). The r = 32 tree's runs (`<process>_tree_s<seed>`) are left out of these batches: their
product-feature fits are unmeasured on a CPU and could take an hour or more each on the product processes, and the
GPU step after training would have to wait for them.

```bash
set -euo pipefail
mkdir -p "$EXP_DIR/results"
for d in runs/hmm-belief-geometry/*_s[012]; do
  r=$(basename "$d")
  case "$r" in *_tree_s[012]) continue ;; esac   # the r = 32 tree waits for the GPU step
  if [ -e "$d/result.json" ] && [ ! -e "$EXP_DIR/results/$r/analysis.json" ]; then echo "$d"; fi
done | xargs -r -P 2 -I{} sh -c 'r=$(basename {}); CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_analyze.py {} "$EXP_DIR/results/$r" > runs/hmm-belief-geometry/analyze_$r.log 2>&1 || echo "FAILED $r"'
```

After training has ended, the same command once more without `CUDA_VISIBLE_DEVICES=` (so the remaining analyses use
the GPU), then the verdict and figures:

```bash
set -euo pipefail
for d in runs/hmm-belief-geometry/*_s[012]; do
  r=$(basename "$d")
  if [ -e "$d/result.json" ] && [ ! -e "$EXP_DIR/results/$r/analysis.json" ]; then echo "$d"; fi
done | xargs -r -P 2 -I{} sh -c 'r=$(basename {}); OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_analyze.py {} "$EXP_DIR/results/$r" > runs/hmm-belief-geometry/analyze_$r.log 2>&1 || echo "FAILED $r"'
PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_summary.py "$EXP_DIR/results" 72 | tee "$EXP_DIR/results/summary.txt"
PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The analysis fits are float64 least squares on either device; only the models' float32 forward pass and the linear
algebra library differ. The sibling `rhm-ttn-belief-geometry` run analyzes on the CPU during training in the same way.

## Expected outputs

- `results/checks.json`.
- `runs/hmm-belief-geometry/<process>_<arm>_s<seed>/`: `result.json` plus 6 checkpoints (0.03 to 3.3 MB each), with
  the training and analysis logs next to them. About 260 MB in total.
- `results/<process>_<arm>_s<seed>/`: `analysis.json` for every run (about 0.1 MB each), and for seed 0 only
  `proj_ckpt0.npz` and `proj_ckpt30000.npz`, which hold the true beliefs and the one probe drawn per model
  (0.2 to 6 MB per run, the largest on the 9-state processes). About 65 MB in total.
- `results/summary.json` and `results/summary.txt`.
- `figures/*.pdf`: 9 files (three triangles, three PCA figures, `positions.pdf`, `summary.pdf`, `training.pdf`).
- `README.md`, the write-up.

## Controls

- **Initial network:** checkpoint 0 is probed identically, with both probe families. For the tensor networks it is
  the identity init, whose states are close to an additive function of the tokens, so it is a strong control: in the
  pilot the identity-init chain already scored projective R² 0.964 on Mess3 and 0.998 on `coins3`.
- **Token readout:** the affine probe on the one-hot prefix.
- **Next-token ceiling:** the best any function of the exact next-token distribution can do, E[belief | next-token
  distribution]. It is estimated from all analysis rows, not from the fit prefixes only, so that it is a true ceiling:
  the mean belief of all rows with the same next-token distribution, or an affine function of it, whichever has the
  lower error. It is 1 on mess3, factored, coupled and coins3, and well below 1 on rrxor.
- **Short-context predictors:** the excess loss of the optimal predictors from only the last two and the last three
  tokens, computed exactly from the process on the training runs' test set. They set the "learned" gate and show how
  much of each process a model can predict without filtering.
- **Exact additive ceiling** (tree, t > T/2): the population value of the best f(left half) + g(right prefix), from
  the exact joint table. It is computable for the four processes with 2 or 3 tokens. For `factored` and `coupled` the
  table has 9^t entries, so there H3b is not judged and the numbers are descriptive. H3b uses only positions where
  the ceiling is below 0.9, so that there is something for the product to add.
- **Token readout with cross-half products** (tree, t > T/2), where it meets the eligibility rule.
- **Scrambled target:** the affine probe fit to a random 8-vector per distinct belief, with the number of distinct
  beliefs per position.
- **Shuffled labels:** expect R² ≈ −p/n.
- **Learning check:** excess test loss over the exact Bayes loss.
- **Correctness checks before training:**
  - `studies.hmm.checks` (29 checks): kernels and stationarity, the Mess3 matrix of Shai et al., the 36 RRXOR belief
    states, filtering against brute-force enumeration of hidden paths, the sampler, the product processes, and the
    short-context predictors (the one-token predictor against enumeration; a full-length context gives the Bayes
    loss).
  - `hmm_check.py` (7 checks): causality of each model's logits; a hand-built exact chain (projective R² 1, affine
    0.32) and a hand-built exact tree (product features needed past the midpoint), which also check that states and
    beliefs are aligned by position; the exact additive ceiling against a sampled least-squares fit (0.9701 against
    0.9701); the next-token ceiling (1 on Mess3 and coins3, below 0.9 somewhere on RRXOR).
  - The hand-built models validate the probes. They are not evidence about what a trained model does.
  - Both pass on the coordinator.

## Success and failure criteria

Computed by `hmm_summary.py` from `results/*/analysis.json` at the final checkpoint, on the prefix split. "Beats"
means (1 − R²_probe) ≤ 0.5 × (1 − R²_control). A (process, model) cell is judged only if all three seeds finished
with status `finished`; otherwise it is censored. Censored cells are listed separately from cells that did not learn.

- **Learned:** all three seeds have excess test loss < min(0.02, half the excess of the optimal predictor from the
  last two tokens). The gate is about 0.007 for mess3 and coupled, 0.008 for factored, 0.018 for coins3 and 0.02
  for rrxor and z1r. Both short-context excesses (two and three tokens) are saved with the results and printed next to
  each run's excess.
- **Saturated control:** a cell whose initial-network or token-readout control scores R² ≥ 0.99 on a hypothesis's
  positions in some seed is "not testable (control saturated)" for that hypothesis and is left out of its count.
- **H1 per cell** (transformer on mess3 and on rrxor; needs a position in `e`): supported if in all three seeds
  - R²(`L4`) ≥ 0.8 on `e`,
  - `L4` beats the initial network's `L4` and the token readout on `e`, and
  - on rrxor only: `nt` is not empty and `L4` beats the next-token ceiling on `nt`.
- **H2 per cell** (chain, every process with a position in `e`): supported if in all three seeds
  - R²(`state`, projective) ≥ 0.8 on `e`,
  - it beats the initial chain's projective probe and the token readout on `e`, and
  - it beats the next-token ceiling on `nt`, when `nt` is not empty.
  - The form label is "projective" if the projective probe beats the affine one on `e` in all three seeds, "affine"
    if it does in none, and "mixed" otherwise.
- **H3a per cell** (`tree` and `tree8`, every process with a position in `h1`): supported if in all three seeds the
  projective probe on `top` reaches R² ≥ 0.8 on `h1` and beats the initial tree's same probe and the token readout
  there.
- **H3b per cell** (`tree` and `tree8`; needs positions in `h2b` and in `h2c`): supported if in all three seeds
  - R²(`top_bil`, projective) ≥ 0.8 on `h2b`,
  - it beats the projective probe on `top` on `h2b`,
  - it beats the exact additive ceiling on `h2c`,
  - it beats the next-token ceiling on `h2n`, when `h2n` is not empty,
  - it beats the initial tree's same probe and the token readout on `h2b`, and
  - it beats the token-product readout on `h2t`, when `h2t` is not empty.
- **Overall, per hypothesis, over the judged cells:** supported if at least 75% of them support it, not supported if
  at most 25% do, partly supported in between, inconclusive if no cell can be judged.
  - The number of judged cells is small by design and is printed with each verdict. H1 has at most 2. H3b has at
    most 2, both at r = 8: `rrxor` (eligible at t = 13–15) and `coins3` (t = 14 and 15). The r = 32 tree has none.
- **Descriptive, no verdict:** H4; the form label; the affine probe next to the projective one; the
  one-map-for-all-positions scores; the scrambled-target scores; the transformer on the processes other than mess3
  and rrxor.
- **Counted, but read narrowly:** the cells on mess3, coins3, factored and coupled enter the H2, H3a and H3b counts
  (they are H3a's only cells), and a pass there is read as showing the layout of the prediction, as described under
  "What a supported H1–H3 shows".
- **The rrxor cells are reported on their own.** With coins3 saturated, H2 has four judged cells, so the other three
  passing while rrxor fails would still pool to "supported". The summary prints every cell, and the write-up must
  state the rrxor result of H1, H2 and H3b separately from the pooled verdicts, because that cell carries the claim
  about beliefs beyond the prediction.
- **Run failed:** the checks fail; no CUDA; fewer than 62 of the 72 runs finish training and analysis; or the
  shuffled-label pooled R² exceeds 0.05 in any run.

## Time and cost

- **Setup:** about 10 minutes.
- **Training:** 72 runs × 30,000 updates at T = 16, four at a time.
  - Transformer: the same model at T = 16 took about 17 minutes for 50,000 updates with four runs sharing the RTX PRO
    4500 (`rhm-belief-geometry`), so about 10 minutes per run here. 18 runs, about 45 minutes.
  - Tree at r = 32: the same tree at T = 16 took about 17 minutes for 30,000 updates with four sharing the GPU
    (`rhm-ttn-belief-geometry`). 18 runs, about 75 minutes.
  - Chain at r = 32 and tree at r = 8: not measured on a GPU. They are much smaller, so an estimated 5–10 minutes per
    run, about 30 minutes for each arm.
  - About 3–3.5 h in total.
- **Analysis:** the beliefs take seconds. The cost is the projective fits: 60 per run for the transformer and the
  chain, 120 for a tree, and three times that on the two product processes, where the factor marginals are fit too.
  Each fit is a linear solve plus two L-BFGS runs.
  - The largest case is the r = 32 tree's product features on the 9-state processes, a 9,801-dimensional solve.
  - On the coordinator's shared CPU, with 8,000 sequences, a whole chain or transformer analysis took 1–2 minutes and
    an r = 16 tree about 2 minutes. With 60,000 sequences an estimated 3–8 minutes per run on the worker's CPU, and
    10–20 for the six r = 32 tree runs on the product processes (not measured).
  - The first three arms (54 runs) are analyzed on the CPU while training continues, so they add little wall-clock
    time. The r = 32 tree trains last, so most of its 18 analyses run after training, on the GPU: about 1 h with two
    in parallel.
- **Total:** about 4.5–5 h expected, with an 8 h cap.
- **Cost:** worst case $1.09/h (RTX 6000 Ada) × 8 h = $8.72, inside the $10 budget. Expected about $3.5 at
  $0.69–0.72/h.

## If something fails

- **A training or analysis run fails** (FAILED in the step output): read its log. May be decided without asking:
  - A bug in `studies/hmm/`: fix it and rerun only the failed runs, noted under Amendments. Files outside
    `studies/hmm/` belong to other experiments and must not be changed; copy what needs changing into this folder.
  - Out of memory in analysis: rerun that run alone (`-P 1`), noted under Amendments. A smaller analysis set is not
    allowed without asking, because it changes which positions are eligible.
  - A run that diverges or collapses to the uniform output: report it as such, don't retune. Its cell is censored.
  - A projective fit that returns a non-finite value falls back to the affine solution by construction; if an
    analysis still crashes there, rerun it once, then report the run as failed.
- **Time running short:**
  - At 4 h after launch, if training is still running, create `runs/hmm-belief-geometry/STOP`. Runs already started
    finish; runs whose turn has not come are skipped and print SKIPPED. The last in the order are the r = 32 tree's
    rrxor, coins3 and z1r runs, which enter no verdict.
  - Then analyze what exists, and run the summary and the figures. If the analysis is the slow part, analyze seed 0
    of every cell first, then seeds 1 and 2.
  - Skipped and unfinished runs are reported as censored, never dropped silently.
- **Checks fail or there is no CUDA:** stop, delete the worker, report.
- **Must wait for the user:** any change to the processes, the models or the success criteria; any spend above the
  budget.

## Questions asked before launch

None needed. The user's go-ahead for this batch (2026-11-14) was "might as well add in the rest of these", after the
theory agent's list of processes. The budget is the project's default worker budget ($10, `orch.toml`). Choices made
without asking, all recorded above: the six processes and their parameters, the construction of `coupled`, context
16, r = 32 for the tensor networks plus a second tree at r = 8, error-ratio margins in place of fixed ones, a
second starting point for the projective fit, and, after the review, the learned gate tied to the short-context
predictor, the saturated-control rule, the next-token clause on H2 and H3b, and the training order.

## Review

experiment-reviewer, first pass: "needs revision", no P0, four P1s and three P2s.

- **P1. On four of the six processes the belief is a function of the model's own prediction target.** On mess3,
  factored and coupled the belief is an exact linear image of the next-token distribution, on coins3 a one-to-one
  function of it, and every model's probed features contain its prediction by construction. The next-token ceiling
  was also underestimated: it read 0.73–1.00 on coins3, where the truth is 1, because its groups came from the fit
  prefixes only.
  - Fixed:
    - H2 and H3b now carry the clause H1 has on rrxor: beat the next-token ceiling where it is below 0.9.
    - The ceiling is estimated from all analysis rows, and `hmm_check.py` checks that it is 1 on mess3 and coins3.
    - "What a supported H1–H3 shows" names the cells that cannot show more than the prediction (mess3, coins3,
      factored, coupled, and H4) and rrxor as the process that separates the two.
- **P1. A control that is already saturated makes "halve its error" unreachable.** The pilot's coins3 chain had an
  identity-init control at 0.998.
  - Fixed: a cell whose initial-network or token control scores ≥ 0.99 is "not testable (control saturated)" and
    leaves the 75%/25% count. The definition says why this gate is higher than the ceilings' 0.9.
- **P1. A fixed "learned" gate of 0.02 would pass a model that does not filter.** The optimal predictor from the
  last two tokens is within about 0.015 nats/token of Bayes on mess3, factored and coupled.
  - Fixed: learned means excess loss below min(0.02, half that predictor's excess). The two- and three-token
    excesses are computed exactly from the process, checked in `studies/hmm/checks.py`, saved with the results and
    printed next to each run's excess. The reviewer called these the 2-gram and 3-gram predictors; the code names
    them by context length, and its values match the reviewer's (mess3 0.0162 and 0.0033 against 0.0158 and 0.0034).
- **P1. H2 and H3b claimed a form that the criteria did not test.**
  - Fixed:
    - The chain's code is labelled "projective" or "affine" by a rule fixed before the run (the projective probe
      beats the affine one, or not), outside the verdict.
    - H3b now requires the product probe to beat the projective probe on the top bonds alone.
    - H2 and H3b are retitled to what is tested, and the definition says that the chain is trained through
      softmax(U h_t), so its state need not be the unnormalized filter, and that the hand-built models of
      `hmm_check.py` validate the probes, not a trained model's code.
- **P2. Training order and the cap.**
  - Fixed: the arms train in the order transformer, tree at r = 8, chain, r = 32 tree, with the r = 32 tree's rrxor,
    coins3 and z1r runs last. The training wrapper skips a run when a STOP file exists, and the cut rule is stated in
    those terms. The sentence about H3b's candidate cells is corrected: the r = 32 tree has none, and rrxor at r = 8
    is eligible at t = 13–15.
- **P2. Analysis should not wait for training to end.**
  - Fixed: the analysis command runs on the worker's CPU in batches during training, as the sibling TTN run does, and
    the remaining runs are analyzed on the GPU afterwards. The time estimate is updated.
- **P2. Saved predictions were larger than the figures need.**
  - Fixed: predictions are saved only for seed 0 and only for the probe each figure draws. The size estimate is
    corrected.

experiment-reviewer, second pass: all four P1s judged addressed in the definition and the code, no new P0; "ready to
run". Three notes, none blocking, all taken:

- **The pooled H2 verdict can hide the rrxor cell.**
  - Fixed: the criteria now require the rrxor cells to be reported separately from the pooled verdicts.
- **One sentence contradicted the code**: it listed the mess3, coins3, factored and coupled cells under "no verdict",
  while the summary counts them.
  - Fixed: they are described as counted and read as showing the layout of the prediction.
- **The r = 32 tree's analyses should stay out of the CPU batches.**
  - Fixed: the CPU-batch command skips `<process>_tree_s<seed>` runs, which are analyzed in the GPU step.

## Amendments

- **2026-11-15 00:35Z, CPU analysis batches run one at a time with 3 threads** (`xargs -P 1`, `OMP_NUM_THREADS=3`
  instead of `-P 2` and 8 threads), from the second batch on. Decided by the supervising session, nobody asked.
  - Reason: the worker's container has a CPU quota of 7.65 cores (`/sys/fs/cgroup/cpu.max` = 765000 100000), not the
    96 that `nproc` reports. The first batch's two analyses ran 16 threads next to the four training processes, and
    the scheduler gave every thread about 0.36 of a core. Training slowed from 41 ms to about 85 ms per update (the
    first four transformer runs took 1,226 s for 30,000 updates; the next four were at update 16,000 after 1,342 s).
    At that rate training would not finish before the 4 h STOP mark.
  - With one analysis on 3 threads there are 7 busy threads for 7.65 cores, so training keeps a full core per run.
  - Nothing about the method changes: the same script, analysis set and fits, on the CPU as before. Fewer runs are
    analyzed during training, and more are left for the GPU step after it, which is unchanged.
  - The first batch (mess3 transformer seeds 0–2, rrxor transformer seed 0) ran as written and was left to finish.
- **2026-11-15 01:52Z, the 4 h STOP mark is no longer fixed, and the budget ceiling is $15.** Authorized by the user
  on 2026-11-15 01:50 UTC (message relayed by the planning session): the budget ceiling is raised from $10 to $15,
  and the definition's time-based cutoffs may be moved so that every planned run is trained and analyzed.
  - Reason: training is slower than estimated. The GPU's time is split between the four training processes, and
    every transformer run took 41 ms per update (about 20.5 minutes per run, against the 10 estimated). The r = 8
    tree runs at about 33 ms per update. Training was projected to end near 05:00Z, not 03:15Z, so a STOP at 03:44Z
    (4 h after the launch at 23:44Z) would have skipped most of the r = 32 tree arm.
  - New rule: the STOP file is created only if the projected end of training and analysis leaves less than 45
    minutes before the worker's hard time cap for the summary, the figures and publishing. The cap is 07:44Z until
    the planning session confirms an extension. The decision is made at 03:44Z from the measured rates and recorded
    here.
  - The worker's worst-case cost at the current cap stays $5.52 ($0.69/h × 8 h).
- **2026-11-15 02:18Z, no STOP at the 4 h mark: all 72 runs are trained and analyzed.** Authorized by the user on
  2026-11-15 01:50 UTC. The planning session extended the worker's hard time cap from 07:44Z to 14:00Z (`orch ls`
  shows "cap in 11.70h" at 02:17Z); the worst-case cost at the new cap is $9.84, inside the $15 ceiling.
  - Training is projected to end near 05:00Z and the analysis by about 07:30Z. The STOP file is created only if the
    work would otherwise run past 13:00Z, which leaves an hour before the cap for the summary, the figures and
    `orch finish`.
- **2026-11-15 02:55Z, CPU analysis batches limited to eight runs (recorded on 2026-11-15 after the run, not before
  acting).** From the fifth batch on, the list of pending runs was passed through `head -n 8` before `xargs`, so that
  a batch took about an hour and the GPU step could start as soon as training ended without overlapping a batch.
  Decided by the supervising session.
  - Under `set -o pipefail` the listing loop is killed by SIGPIPE once `head` has eight lines, so the fifth, sixth
    and seventh batch steps ended with exit code 141. Their eight analyses each ran to the end; no analysis log has an
    error and all 72 `analysis.json` files exist.
- **2026-11-15 05:59Z, the last command block ran as two steps:** the GPU analysis of the remaining 18 runs first,
  then the summary and the figures, so that the analyses could be checked before the verdict was computed. The
  commands are the definition's, unchanged.
- **2026-11-15, after the run: PNG copies of the nine figures** (`figures/png/`, 200 dpi) made on the coordinator by
  `hmm_figures_png.py`, which calls the plotting functions of `hmm_plot.py` on the saved results, and a research note,
  `NOTE.md`. Both were requested by the user on 2026-11-15 01:50 UTC. The PDFs are unchanged.

## Outcome

Result recorded: **inconclusive**, which stands for a mixed outcome: H1 is supported and H2, H3a and H3b are each partly
supported. Write-up: [README.md](README.md). Research note: [NOTE.md](NOTE.md). Supervisor report:
[SUPERVISOR.md](SUPERVISOR.md).

- **Run failed?** No. Both check scripts passed (29 of 29 and 7 of 7), CUDA was present, 72 of 72 runs finished
  training and analysis, and the shuffled-label pooled R² was between −0.011 and 0.005 in every run.
- **Learned:** all 24 cells, in all three seeds. No cell was censored.
- **H1, transformer reproduction: supported, 2 of 2 cells.** Mess3 0.9997; RRXOR 0.865–0.872 against a next-token
  ceiling of 0.474.
- **H2, chain, projective probe: partly supported, 2 of 4 judged cells.** Mess3 and RRXOR pass; `factored` and
  `coupled` fail against the token readout; coins is not testable (control saturated). Form label: projective on
  RRXOR, affine on Mess3, `factored` and `coupled`, mixed on coins.
- **H3a, tree, first half: partly supported, 2 of 6 judged cells.** Mess3 passes at r = 32 and r = 8; `factored` and
  `coupled` fail at both.
- **H3b, tree, second half: partly supported, 1 of 2 judged cells.** Coins at r = 8 passes; RRXOR at r = 8 fails.
- **The RRXOR cells on their own:** H1 pass, H2 pass, H3b fail.
- **H4 (descriptive):** every model reads the factor marginals at 0.97 or more; only the transformer reads the joint
  belief better than half the token readout's error.
- **Cost and runtime:** 8 h 57 min on one RTX A6000 at $0.69/h, about $6.18. The definition's budget was $10 and its
  cap 8 h; the user raised the ceiling to $15 and the cap was extended to 14:00 UTC during the run (see Amendments).
