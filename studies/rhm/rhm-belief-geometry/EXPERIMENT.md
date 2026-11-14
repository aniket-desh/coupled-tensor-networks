---
name: rhm-belief-geometry
title: Belief-state geometry of the RHM in a transformer's residual stream
status: approved
gpu: NVIDIA RTX A6000,NVIDIA RTX PRO 4500 Blackwell,NVIDIA RTX PRO 4500 Blackwell Server Edition,NVIDIA RTX 6000 Ada Generation,NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB
count: 1
hours: 8
budget_usd: 20
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-14T18:29:39Z
---

# Belief-state geometry of the RHM in a transformer's residual stream

## Question

Shai et al. (2024, arXiv 2405.15943) trained small transformers on data from hidden Markov processes and found the
Bayesian belief states over the hidden states laid out linearly in the residual stream. Their models were trained on
Mess3, whose belief states form a fractal, and RRXOR, which has 36 discrete belief states. Read left to right, the
Random Hierarchy Model is also a hidden Markov process. Its hidden state after t tokens is the stack of pending
subtrees, meaning the subtrees whose symbols are fixed but none of whose leaves have been emitted. The exact belief
state b_t = P(pending symbols | x_<t) is computed by `grammar.stack_belief`, which is checked against brute-force
enumeration.

Does a transformer trained by next-token prediction on RHM data carry b_t linearly in its residual stream? How does
that depend on the grammar's parameters: synonym number m, depth L, vocabulary v and branching s? This is a
curiosity-driven replication on a new process, and its main product is the Simplex-style figures. The run is cheap
because the code exists and has been piloted on CPU.

## Hypothesis

- **H1, the belief is linearly decodable from the last layer on prefixes the probe never saw, beyond trivial
  explanations.** Take each grammar the transformer learns (excess test loss < 0.02 nats/token) and that has eligible
  positions (defined under Method). There, an affine probe from the last block's output recovers b_t on held-out
  prefixes with pooled R² ≥ 0.8. It also beats both cheap explanations by at least 0.1 in R²:
  - the same probe on the randomly initialized network;
  - the best affine function of the one-hot prefix tokens.
  - Why: the RHM's next-token distribution is a linear function of b_t (`next_token_from_belief`), so a network
    predicting near Bayes must compute something that determines it. Simplex's result suggests the network lays out
    the whole belief linearly. If so, prefixes with equal beliefs map to the same point, and the probe generalizes to
    unseen prefixes.
  - Against: R² < 0.8, or no 0.1 margin over random init and the token readout, in a majority of the testable
    grammars.
- **H2, more than the next-token output (descriptive).** At eligible positions where the belief is not a function of
  the exact next-token distribution (next-token ceiling R² < 0.9), does the probe exceed that ceiling? These cases are
  reported as counts, with no verdict threshold.
  - The pilot found the ceiling is 1 at most positions of small grammars: at the last position of a binary tree, b_t
    is exactly P(x_t | x_<t). So H1 alone is not evidence that the network represents more than its output.
- **Descriptive:** which layer carries the belief (last layer alone vs all layers concatenated, as in Shai et al.
  Fig. 7E), and how R² evolves over training.

## Method

- **Grammars:** 17 configurations, named vVmMsSLL, with n = v root classes. Each runs with seeds 0, 1 and 2, and the
  seed sets both the grammar's rules and the network's init, giving 51 runs.
  - `v3m2s2L3`, `v3m2s2L4`, `v3m2s2L5`, `v3m2s2L6` (T = 64), `v3m3s3L2`, `v3m3s3L3`: 3 symbols per node, so each node's
    marginal belief lives on a 2-simplex and gives the literal Simplex triangle.
  - `v8m2s2L3`, `v8m4s2L3`, `v8m6s2L3`: a synonym sweep at fixed depth.
  - `v8m2s2L4`, `v8m4s2L4`, `v8m2s2L5`: depth 4 (and 5) against depth 3.
  - `v6m3s3L2`: ternary branching.
  - `v4m2s2L5`: depth 5, T = 32.
  - `v16m2s2L4`, `v16m4s2L4`, `v16m8s2L4`: a synonym sweep at the main grammar of the other RHM studies (m = 4).
    Position t = 1 is skipped in these grammars and in v8m2s2L5, because the joint there has more than 4,096 entries.
  - The smallest grammars have too few distinct prefixes for any eligible position (see below), so they are
    figure-only: v3m2s2L3, v3m3s3L2, v6m3s3L2 and v8m2s2L3, by the reviewer's count. `v3m2s2L5` and `v3m3s3L3` were
    added so the triangle family also has testable members.
- **Model** (`model.py`, the pre-LN blocks of `studies/lm/models.py`, GPT-2 init):
  - 4 layers, d_model 128, 4 heads, MLP 512, GELU, learned positions.
  - Input [BOS, x_0 .. x_{T−2}], targets x_0 .. x_{T−1}. The residual at input position t has seen x_<t.
  - This matches the Simplex RRXOR configuration (d 128, 4 heads, MLP 512, from their `rrxor/train_config.json`),
    except for the BOS token (as in Shai et al. 2026, "Transformers learn factored representations") and GELU in place
    of ReLU.
- **Training** (`run.py`):
  - Fresh sequences sampled from the grammar every update (online, as in Simplex), cross-entropy over all T positions.
  - Adam, peak LR 1e-3, linear warmup over 500 updates, then cosine to 0.1 × peak (`studies.rhm.train.lr_at`).
  - Batch 512, 50,000 updates, float32.
  - Checkpoints at updates 0, 300, 1000, 3000, 10000, 30000 and 50000.
  - Every log point records per-position test loss on 8,192 fixed sequences, next to the exact Bayes loss
    −log P(x_<=t) + log P(x_<t) computed from `grammar.likelihoods`.
  - The LR comes from the CPU pilots, which reached 0.0003 nats/token excess on v3m2s2L3 after 2,000 updates and 0.006
    on v8m4s2L3 after 3,000. It is not swept, because the probe only needs a network near Bayes, and excess loss is
    reported per run.
- **Probe** (`analyze.py`, the method of Shai et al. 2024 §2.3):
  - Data: 60,000 analysis sequences sampled with a fixed seed. Sampling makes each prefix count in proportion to its
    probability (the weighting of Riechers et al. 2025, without deduplication).
  - Positions and fit: at each t = 1 .. T−1 with D_t ≤ 4096, a separate affine least-squares map is fit, because D_t
    changes with t. The ridge is 1e-8 relative to the mean diagonal and only guards rank deficiency.
  - Activation sets: the embedding, the four block outputs (L1 to L4, before the final LayerNorm), the final LayerNorm
    output, and all six concatenated.
  - Two splits:
    - **Sequence split** (the Simplex convention): a 30,000 / 30,000 split by sequence, so test prefixes also occur in
      the fit half. It is used for the geometry figures.
    - **Prefix split**, used for the criteria: at each position the distinct prefixes x_<t are assigned at random to
      fit or test, so the probe is scored only on prefixes it never saw. This split is needed because, when a grammar
      has fewer distinct prefixes than features, any injective embedding of the prefix lets an affine probe
      interpolate any function of it. The pilot made this concrete: on v3m2s2L3 the random-init network reached R²
      0.976 on the sequence split and −4.9 on the prefix split. A network that maps prefixes to an affine image of
      their belief passes the prefix split.
  - **Eligible positions** (for the primary probe on L4, 128 + 1 features): at least 10 × 129 = 1,290 distinct fit
    prefixes. Below that, the probe is underdetermined for any network. Concat (769 features) has its own eligible set
    (≥ 7,690 fit prefixes) and is descriptive.
  - Metrics:
    - R² = 1 − Σ‖b − b̂‖² / Σ‖b − mean_fit b‖², pooled over positions as 1 − ΣSSE / ΣSST.
    - RMSE = sqrt(mean ‖b − b̂‖²), the Riechers convention.
    - The trained network, random init, the token readout and the next-token ceiling are always pooled over the same
      positions.
  - Checkpoints: every checkpoint is probed.
- **Pairwise distances** (Shai Fig. 7D): R² between probe-output distances and true-belief distances, and between
  probe-output distances and next-token distances, on the sequence split.
- **Figures** (`plot.py`, PDFs from saved results only):
  - `geometry_<cfg>.pdf`: ground-truth beliefs on their top-2 PCs (probability-weighted), then the probes from all
    layers and from the last layer, the shuffled-label probe, the random-init probe, and a raw PCA of the last-layer
    residual.
    - Rows are 4 positions, chosen to prefer a next-token ceiling < 0.9 and then the most distinct beliefs. Each row
      prints its ceiling.
    - Points are colored identically across panels, and each probe panel also shows the per-belief centroid.
    - Figure-only grammars carry a note that a random network also fits them.
  - `triangles_<cfg>.pdf`: the v = 3 grammars, with rows for truth, the trained probe and the random-init probe.
  - `distances_<cfg>.pdf`.
  - `summary.pdf`: pooled R² by layer on the prefix split.
  - `positions.pdf`: per-position R², with ineligible positions shaded.
  - `training.pdf`.
- **Verdict:** `summary.py` applies the criteria and writes `results/summary.json`.

## Inputs

None. All data is generated from the seeded grammars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
free -g
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-geometry/check_probe.py
```

Training, four runs at a time on the GPU, largest grammar first, as its own `orch exec` step:

```bash
set -euo pipefail
mkdir -p runs/rhm-belief-geometry
for c in v16m4s2L4 v16m8s2L4 v16m2s2L4 v3m2s2L6 v4m2s2L5 v8m2s2L5 v8m4s2L4 v8m2s2L4 v3m2s2L5 v3m3s3L3 v3m2s2L4 v8m6s2L3 v8m4s2L3 v8m2s2L3 v6m3s3L2 v3m3s3L2 v3m2s2L3; do
  for s in 0 1 2; do echo "$c $s"; done
done | xargs -P 4 -n 2 sh -c 'OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-geometry/run.py "$0" --seed "$1" --out runs/rhm-belief-geometry > runs/rhm-belief-geometry/train_$0_s$1.log 2>&1 || echo "FAILED $0 s$1"'
```

Analysis, two at a time (memory: the T = 32 and T = 64 grammars hold 6–12 GB of float32 activations each), then the
verdict and the figures:

```bash
set -euo pipefail
ls -d runs/rhm-belief-geometry/*_s[012] | xargs -P 2 -I{} sh -c 'r=$(basename {}); OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-geometry/analyze.py {} "$EXP_DIR/results/$r" > runs/rhm-belief-geometry/analyze_$r.log 2>&1 || echo "FAILED $r"'
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-geometry/summary.py "$EXP_DIR/results" 51 | tee "$EXP_DIR/results/summary.txt"
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-geometry/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

## Expected outputs

- `results/checks.json`.
- `runs/rhm-belief-geometry/<cfg>_s<seed>/`: `result.json` plus 7 checkpoints, each 0.4 to 3.3 MB. Training and analysis
  logs sit next to them. The total is about 800 MB.
- `results/<cfg>_s<seed>/`, under 200 MB in total:
  - `analysis.json`, with all metrics, the belief geometry and the eligible positions;
  - `proj_ckpt0.npz` and `proj_ckpt50000.npz`;
  - `dists.npz`.
- `results/summary.json` and `results/summary.txt`.
- `figures/*.pdf`, about 55 files.
- `README.md`, the write-up.

## Controls

- **Randomly initialized network** (Riechers 2025, Piotrowski 2025): checkpoint 0 is probed identically, on both
  splits.
- **Linear in tokens:** the same probe applied to the one-hot encoding of the full prefix. If this matches the trained
  network, the belief is just a linear readout of the input.
- **Prefix split and eligibility:** these rule out probe interpolation over a small set of distinct prefixes, as
  described under Method.
- **Shuffled labels** (Shai 2024): the fit half's belief rows are permuted across sequences, and the error is measured
  against the true beliefs. Expect R² ≈ −p/n.
- **Next-token ceiling:** the best any function of the exact next-token distribution can do, measured as the test R² of
  predicting b_t by the mean belief of fit rows with the same next-token distribution. It separates "the network
  encodes its output" from "the network encodes the belief".
- **Learning check:** excess test loss over the exact Bayes loss, per position and overall.
- **Correctness checks before training:**
  - `studies.rhm.checks` covers `stack_belief` and next-token-from-belief against brute-force enumeration.
  - `check_probe.py` checks that a linear image of b_t gives R² ≈ 1, shuffled labels give ≈ 0, the ceiling is ≤ 1,
    and the affine fit keeps beliefs summing to 1.
- **Independent check of the Bayes loss:** in the pilot the likelihood-based Bayes loss of v8m4s2L3 was 1.4729
  nats/token, against 1.473 from the belief-state route.

## Success and failure criteria

Computed by `summary.py` from `results/*/analysis.json` at the final checkpoint, on the prefix split, pooled over the
eligible positions (L4, ≥ 1,290 distinct fit prefixes), with seeds counted separately. A grammar is judged only if all
three of its seeds finished; otherwise it is censored and reported as such.

- **Learned grammar:** all three seeds have excess test loss < 0.02 nats/token. A grammar failing this is reported as "not
  learned"; that is a finding, not a failure.
- **Testable grammar:** all three seeds have at least one eligible position. The others are figure-only, and their
  sequence-split numbers are reported but carry no verdict.
- **H1 per grammar:** supported if, in all three seeds:
  - R²(L4) ≥ 0.8; and
  - R²(L4) − max(R²(random init, L4), R²(linear in tokens)) ≥ 0.1.
- **H1 overall, over learned and testable grammars:**
  - supported if at least 75% support it;
  - not supported if at most 25% do;
  - partly supported in between;
  - inconclusive if fewer than 4 grammars are both learned and testable.
- **H2:** reported descriptively as the number of eligible (grammar, seed, position) cases with next-token ceiling
  < 0.9, and how many of those have R²(L4) > ceiling + 0.05.
- **Descriptive:** concat on its own eligible positions; the sequence-split numbers (Simplex convention); R² over
  checkpoints.
- **Run failed:**
  - the checks fail;
  - no CUDA;
  - fewer than 44 of the 51 runs finish both training and analysis;
  - or the shuffled-label pooled R² exceeds 0.05 in any run, which would mean a probe bug, not a result.

## Time and cost

- **Setup:** about 10 minutes.
- **Training:** 51 runs × 50,000 updates. On CPU it took 106 ms/update at batch 256. On these GPUs these small models
  should be launch-bound at roughly 5–10 ms/update (an estimate, not measured; T = 64 is slower), so each run takes
  5–10 minutes. With 4 at a time that is about 1.5–2 h.
- **Analysis:** belief computation runs on the CPU. On the coordinator, at 3,000–4,000 sequences, it took 34 s for
  v16m4s2L4 and 16 s for v16m8s2L4, so the v16 grammars need roughly 5–8 minutes each at 60,000 sequences, and the
  others need seconds to a minute or two. Seven checkpoints × 2 splits are probed per run on the GPU. That makes about
  1.5–2 h with 2 in parallel.
- **Total:** about 3.5–4.5 h expected, with an 8 h cap.
- **Cost:**
  - Worst case: $2.19/h (A100-SXM4, the most expensive GPU listed, used only if the cheaper ones are out of stock) ×
    8 h = $17.5, inside the $20 budget.
  - Expected: about $3 on an RTX A6000 at $0.69/h.

## If something fails

- **A training or analysis run fails** (FAILED in the step output): read its log. May be decided without asking:
  - A bug in the code: fix it and rerun only the failed runs, noted under Amendments.
  - Out of memory in analysis: rerun that run alone, or with a smaller third argument to `analyze.py` (e.g. 20000),
    noted under Amendments.
  - A grammar that doesn't learn: report it, don't retune.
- **Time running short:**
  - After 5 h, stop starting new training runs, then analyze and plot what exists. Seed 2 of the smallest grammars is
    the first to go, because the training order puts the largest grammars first.
  - Unfinished runs are reported as censored, never dropped silently.
- **Checks fail or there is no CUDA:** stop, delete the worker, report.
- **Must wait for the user:** any change to the grammar list, the model, or the success criteria; any spend above the
  budget.

## Questions asked before launch

- **Launch as defined (13 grammars × 2 seeds, $4.5 cap)?**
  - The user (2026-11-14): "expand the budget a bit so you can expand the experiment scale. $20. launch in
    tmux/batch".
  - Applied:
    - 3 seeds instead of 2;
    - 4 more grammars (v3m2s2L6, v8m2s2L5, v16m2s2L4, v16m8s2L4);
    - 50,000 updates instead of 30,000, with a checkpoint added at 30,000;
    - 60,000 analysis sequences instead of 40,000;
    - an 8 h cap, a $20 budget, and A100s as fallback GPUs.
  - Supporting code change: the concatenated activation set is built per position, to keep T = 64 within memory.
  - The design and the criteria are otherwise unchanged from the reviewed version, with "both seeds" now reading "all
    three seeds".
  - The launch is run by a supervising tmux session (`orch experiment supervise`).

## Review

experiment-reviewer, first pass: "needs revision".

- **P0. Random init interpolates the belief in the small grammars on the sequence split, so the H1 margin was
  unreachable there and the v = 3 triangles would show nothing learned.**
  - The reviewer measured random-init concat R² of 0.97–0.995 in v3m2s2L3, v3m3s3L2, v8m2s2L3 and v6m3s3L2. The CPU
    pilot found the same.
  - Fixed:
    - H1 now uses a prefix split, pooled only over positions with ≥ 10 × (features + 1) distinct fit prefixes, with L4
      as the primary probe.
    - Grammars without such positions are figure-only, and their figures say so.
    - Thresholds are restated over testable grammars.
    - v3m2s2L5 and v3m3s3L3 were added so the triangle family has testable members.
    - The triangle figure shows the random-init probe as a third row.
- **P1. The geometry rows would mostly show positions where the belief is a function of the next-token
  distribution.** Fixed: rows prefer positions with a next-token ceiling < 0.9, and each row prints its ceiling. H2's
  text says H1 alone is not evidence beyond the output.
- **P2. Four analyses at once may exceed RAM.** Fixed: activations are stored as float32, the analysis runs two at a
  time, and `free -g` is logged.
- **P2. The main grammar would be the one censored under time pressure.** Fixed: the largest grammars train first.
- Second pass: the P0 and P1 were judged fixed, with no new P0. The reviewer's one open item was the user's consent to
  the added grammars, which the user then gave together with the expansion above.
- The expansion (more seeds, grammars, updates and analysis sequences) was not re-reviewed: it scales the reviewed
  design without changing its logic.

## Amendments

None.

## Outcome

Not run yet. Afterwards: one line per criterion with what was observed, the cost and runtime against the budget, and a link to the write-up. The write-up holds the detail; don't repeat it here.
