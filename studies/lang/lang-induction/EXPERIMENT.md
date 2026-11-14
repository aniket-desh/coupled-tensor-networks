---
name: lang-induction
title: "Storing associations before the query: key-value retrieval and multi-match induction in a TreeTN across bond dimension 16-128"
status: approved
gpu: NVIDIA RTX A6000,NVIDIA L40S,NVIDIA A40
count: 1
hours: 10
budget_usd: 12
disk_gb: 120
data:
secrets:
approved_utc: 2026-11-13T23:52:39Z
---

# Storing associations before the query: key-value retrieval and multi-match induction in a TreeTN across bond dimension 16-128

## Question

Induction-style prediction, [a][b] … [a] → [b], asks a network to keep associations before it knows which one will be queried.

In a TreeTN, a subtree that doesn't contain the query must pass up a summary that stays useful for every possible query. The candidate summary is its table of adjacent (key, successor) pairs plus its boundary tokens. Because the root is linear in each child's bond, that table bounds the bond dimension the subtree needs.

The sampled centered-logit ranks (`studies/lang/results/rank.json`) make this concrete:

- **Retrieval, with the key set held fixed:** rank 8 for one pair, 15 for two, and 29 for the four pairs in the left half (4(V − 1) + 1). Letting the key set vary can only raise these.
- **Multi-match induction:** the exact rank reaches about 440 at level 2. Only about 75 singular values are above 1% of the largest, and 2–29 above 10%.

The experiment asks:

1. How does final quality depend on width (r = 16, 32, 64, 128), and what is the smallest stable width that reaches a fixed Bayes gap?
2. With unambiguous retrieval, are targets whose pair meets the query only at the root (pairs 0–3) harder than those near the query (pair 6), and does that gap close with width? This is a direct capacity test.
3. Does alignment still matter? Adjacent key-value pairs sit inside level-1 subtrees of the aligned tree, but are split across subtrees by a permutation.
4. Saxe-style staging: does association storage in the left half become decodable before the loss drop, together with it, or after it?

We call this "learning induction behavior through hierarchical associative representations". The canonical transformer induction circuit needs two attention layers: a previous-token head, then an induction head (Olsson et al. 2022; Singh, Moskovitz, Hill, Chan and Saxe 2024, arXiv:2404.07129, on how it forms). A TreeTN solving the task doesn't mean it forms that circuit, and tree depth is not attention depth. So the measured question is whether binding (pairs inside a subtree) and query-dependent retrieval (at the root) appear in a staged way.

## Hypothesis

- **H1 (retrieval width).** Final test KL at r = 128 is lower than at r = 16 in 3 of 3 seeds. We also report the smallest stable width: every seed at r and 2r finished and not collapsed, median KL < 0.01 nats, and doubling r gains < 0.002 nats. The alternatives are "none", or "unresolved (still improving)" if the widest runs haven't converged.
- **H2 (stratified capacity, retrieval).** At r = 16, only one bond can't hold what its span must pass up: the left half (node 3_0, sampled rank ≥ 29). Nodes 2_2, 2_3 and 3_1 need 15, 13 and 11.
  - Capacity therefore predicts this pattern: accuracy on mid targets (pairs 4–5, which meet the query at level 3) within 0.02 of the near target (pair 6, level 2), and both > far targets (pairs 0–3, only the root) + 0.05.
  - Distance alone would instead predict a graded near > mid > far, which is reported.
  - The pattern must hold in every seed that learned at r = 16 (near accuracy ≥ 0.8 × Bayes; others are reported as "did not learn"). The mid − far gap at r = 128 must be below half its r = 16 value in all but at most one of those seeds.
  - Against: no capacity pattern, or a graded pattern.
- **H3 (alignment, retrieval).** At equal width (r = 16 and 64), the aligned tree has lower final KL than the permuted tree in ≥ 5 of 6 (width, seed) pairs. Mechanism: the permutation splits every key from its value at level 1, so each binding must be formed higher up, where the bonds also have to carry other tokens.
- **H4 (multi-match width).** KL at r = 128 is lower than at r = 16 in 3 of 3 seeds, and the median KL at r = 128 stays above 0.01 nats, because the exact rank (about 440) is far above 128.
  - The second clause counts only if the r = 128 runs have converged: diagnostic KL over the last 10,000 updates fell by < 5% or < 0.002 nats. A run still improving would also end above 0.01. About 75 singular values exceed 1% of the largest, so an approximation within r = 128 reaching KL < 0.01 is possible.
  - Supported if both clauses hold with convergence. Partly supported if only the first holds, or if the gap remains while the runs are still improving.
- **H5 (staging; descriptive, two-sided).** For learned, finished retrieval tree runs, the step at which the left half's dictionary R² (node 3_0) reaches half its final gain is compared with the step at which the diagnostic loss falls halfway. Each run is classed as "dict first", "together" (same checkpoint interval), "loss first", or "no gain" (final R² ≤ step 0). Linear checkpoints between 500 and 8,000 (r ≤ 32) give the resolution.
- **Descriptive.**
  - tree vs chain at matched parameter count;
  - multi-match tree vs perm;
  - KL by number of matches;
  - collapses by width.

## Method

- **Processes** (`studies/lang/processes.py`), V = 8, input vocabulary 9 (the blank token), target noise ε = 0.05:
  - **`retrieval`:**
    - Leaves (2m, 2m+1), m = 0..6, hold 7 key-value pairs. The keys are distinct (drawn without replacement from 8 symbols) and the values uniform.
    - Leaf 14 is the query, the key of a uniformly chosen pair. Leaf 15 is the blank.
    - y = that pair's value.
    - Strata = the target pair's index. Pair 6 shares the level-2 node with the query, pairs 4–5 the level-3 node, and pairs 0–3 only the root.
  - **`multimatch`:**
    - x_0..x_13 are iid uniform. A planted i ~ U{0..12} sets the query x_14 = x_i, and leaf 15 is the blank. y = x_{i+1}.
    - The Bayes posterior is a uniform vote over all matches M = {j ≤ 12 : x_j = x_14}. Strata = min(|M|, 3).
    - Bayes accuracy is about 0.57, because accidental matches are common.
- **Arms**, per process:
  - `tree{r}`: untied tree, r ∈ {16, 32, 64, 128}; 61,696, 471,552, 3.71 M and 29.5 M parameters.
  - `perm{r}`: r ∈ {16, 64}, with the leaf permutation seeded per seed and redrawn until no (2k, 2k+1) pair are model siblings.
  - `chain{r}`: widths 39 and 155, the parameter matches to the r = 16 and r = 64 trees.
  - Identity init, ε_init = 0.1.
- **Grid:** 2 processes × 8 arms × seeds 0–2 = 48 runs. Seed 0 of everything runs first.
- **Training:** online batches of 256, Adam, 60,000 updates, warmup 100, then a cosine schedule to 0.1 × peak.
- **LR:**
  - Piloted per (process, arm, width) on seed 9, over 12,000 updates of the 60,000-update schedule. The run is long because the r = 64 RHM runs collapsed only after update 3,900.
  - A pilot run that doesn't finish (for example, out of memory) stops the job rather than silently dropping its LR.
  - Grids: {1e-4, 3e-4, 1e-3, 3e-3} for r ≤ 32 and the width-39 chain; {3e-5, 1e-4, 3e-4, 1e-3} for r ≥ 64 and the width-155 chain.
  - The lowest diagnostic loss wins, and an LR whose pilot learned and then returned within 0.1 nats of log V is excluded.
- **Measures:**
  - Diagnostic set of 20,000 examples, logged through training.
  - Test set of 50,000, with per-stratum accuracy, loss and KL.
  - t = the first logged update with diagnostic accuracy ≥ 0.9 × Bayes accuracy.
  - The collapse flag: learned to below max(log V / 2, log V − 0.5), so it can fire for multi-match, whose Bayes loss is 0.87 nats, then returned within 0.1 nats of log V.
- **Checkpoints** (tree and perm): about 25 for r ≤ 32 (log-spaced, plus linear ones at 500–8,000, plus the shared RHM steps scaled to 60,000) and about 15 for r ≥ 64.
- **Interpretation** (tree and perm runs, every checkpoint):
  - `dict`: the adjacent-pair table of a span as a 64-dimensional ridge target (binary for retrieval, counts for multi-match).
  - `query`: x_14.
  - `ret`: the value associated with the query among the span's pairs; for multi-match, the successor-vote vector within the span.
  - Baselines: step 0, shuffled labels, and an additive one-hot probe.
  - Invariance (retrieval): shuffle the complete pairs within a span.
  - Patching: the dict, query or ret subspace, against random subspaces of the same dimension and the whole bond. A regression probe's subspace is the fewest top ridge directions holding 90% of the predicted variance, at most half the bond, so the random control isn't the whole bond. k and d are reported. Only donor splices that remain valid inputs are used; the count is reported.
  - Forward passes run in float32; whitening, probes and KLs run in float64.

### Comparison rules

These are as in `studies/lang/analysis.py`: better(·) on final test KL with a tie within max(0.002, 10%), and faster(·) with the logging-interval margin and factor 1.25. Only finished pairs enter, and dropped pairs are counted. Collapsed runs stay in and are listed.

## Inputs

None.

## Commands

```bash
set -euo pipefail
mkdir -p "$EXP_DIR/results" "$EXP_DIR/figures"
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python -m studies.lang.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python -m studies.lang.rank "$EXP_DIR/results/rank.json" --configs lang-induction
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-induction/run.py --out runs/lang-induction --hours 4.5
PYTHONPATH=. uv run --frozen python -m studies.lang.interpret_all runs/lang-induction --procs 3 --hours 3 --first retrieval
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-induction/analyze.py runs/lang-induction "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/lang/lang-induction/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
for d in runs/lang-induction/*128_s*/; do [ -f "$d/interpret.json" ] || continue; last=$(ls "$d"ckpt_*.pt | sort -V | tail -n1); find "$d" -name 'ckpt_*.pt' ! -name ckpt_0.pt ! -path "$last" -delete; done
```

Run the training and interpretation steps from a background command. The checks exit non-zero on failure. The training deadline (4.5 h) counts from the end of setup and the pilots. Interpretation takes retrieval runs first within each seed, since H5 depends on them.

The last line removes the intermediate checkpoints of each r = 128 run (about 118 MB each), but only once that run has an `interpret.json`. It keeps step 0 and the run's last checkpoint, which keeps the published data to a few GB.

## Expected outputs

- `results/checks.json` and `results/rank.json`.
- `runs/lang-induction/<process>_<arm><r>_s<seed>/`: `result.json`, checkpoints and `interpret.json`. Plus `lr.json`, `meta.json`, `planned.json`, `status.json`, `pilot/` and `source/`. About 3–5 GB after pruning.
- `results/analysis.json`, `results/curves.json` and `results/interp.json`.
- `figures/width.pdf`, `figures/strata.pdf`, `figures/staging.pdf` and `figures/curves.pdf`.
- `README.md`.

## Controls

- **Pairing.** Same data stream and process instance across arms per (process, seed).
- **Alignment.** perm at equal width isolates alignment. The chain is matched on parameter count, which is a separate question.
- **Within-model stratification.** The H2 comparison is inside each model (far vs near targets), so the model, data and training are identical across the strata.
- **Rank as context.** The sampled rank sets expectations for width, but as a lower bound on sampled contexts, not a certificate.
- **Probe baselines:** step 0, shuffled targets, the additive one-hot probe, random-subspace patches, and whole-bond patching (equal to splicing).
- **Correctness checks** run first.

## Success and failure criteria

- **H1:** supported if r = 128 beats r = 16 on final KL in 3 of 3 seeds; not supported otherwise. The selected width (or none) and collapse counts per width are reported with it.
- **H2:**
  - A seed counts as learned if its near accuracy at r = 16 is ≥ 0.8 × Bayes. Other seeds are reported as "did not learn".
  - supported if every learned seed shows the capacity pattern (|acc(4–5) − acc(6)| ≤ 0.02 and both > acc(0–3) + 0.05), and the mid − far gap at r = 128 is below half its r = 16 value in all but at most one learned seed;
  - partly supported if only the pattern holds;
  - not supported otherwise;
  - inconclusive with fewer than 2 learned seeds.
  - Whether each seed is graded (near > mid > far) is reported.
- **H3:** supported if the tree beats perm on final KL in ≥ 5 of 6 pairs with no losses; "opposite direction" if the mirror holds; inconclusive with fewer than 4 usable pairs; not supported otherwise. Speed is reported with it.
- **H4:**
  - supported if r = 128 beats r = 16 in 3 of 3 seeds and the median KL at r = 128 is > 0.01 with the r = 128 runs converged (diagnostic KL over the last 10,000 updates fell by < 5% or < 0.002 nats);
  - partly supported if only the first clause holds, or if the gap remains while the runs are still improving (reported as "partly supported (r = 128 still improving)");
  - not supported if r = 128 doesn't beat r = 16 in 3 of 3;
  - inconclusive with fewer than 2 usable pairs.
  - The width rule reports "unresolved (still improving)" in the same case.
- **H5:** descriptive counts.
- **Inconclusive:** more than a quarter of the planned runs didn't finish (this overrides every verdict), or a comparison has fewer usable pairs than its rule above needs.
- **Run failed:** the checks fail, CUDA is unavailable, or no tree run finished.

## Time and cost

- **Setup:** about 10 minutes, plus about 5 minutes for checks and rank.
- **Pilots:** 16 keys × 4 LRs × 12,000 updates = 768,000 updates.
  - On the coordinator's CPU, r = 16 trees ran at about 8–14 ms/update and the width-39 chain at about 76–140 ms.
  - On a GPU, the r = 128 tree costs about 45 GFLOP per update and the width-155 chain about 30 GFLOP, so a few ms each on an A6000/L40S/4090. The narrow arms are launch-bound.
  - The pilots should take well under 1 h with 6 at a time.
- **Training:** 48 × 60,000 updates, about 1.5–3 h. The deadline is 4.5 h.
- **Interpretation:** 36 runs in float32 forwards, deadline 3 h.
- **Total:** at most about 0.5 + 1 + 4.5 + 3 + 0.2 ≈ 9.2 h, under the 10 h cap.
- **GPUs:** the 24 GB RTX 4090 is excluded. Six concurrent processes with r = 128 evaluation chunks need more memory; the other three cards have 48 GB.
- **Cost:** at most $10.90 at $1.09/h, inside $12. Expected about $2–5.
- **Disk:** checkpoints of the r = 128 runs (about 1 GB per run before pruning), within the 120 GB disk.

## If something fails

- **Checks fail or CUDA is unavailable:** stop and mark the experiment failed.
- **Out of GPU memory with 6 concurrent runs** (the r = 128 runs are the largest): rerun the remaining runs with `--procs 3` (or `--procs 2`) into the same `--out`. Finished runs are not rerun.
- **A run crashes:** rerun it once into `runs/lang-induction-rerun`, after copying `lr.json` there, with `--only <names>`. Record it as an amendment and treat it as descriptive.
- **Every LR in a pilot is excluded as collapsed:** `pilot` exits. Rerun with `--lr <key>=<the lowest grid LR>` for that key into the same `--out`, and record it as an amendment.
- **A pilot run doesn't finish (for example, out of memory):** `pilot` exits. Rerun `run.py` into the same `--out` with `--procs 3` (or 2). Finished pilot and grid runs are reused.
- **The deadline skips runs:** seed 0 runs first, so later seeds are what gets skipped. Rerun them if the cap allows; otherwise H1–H4 use the available pairs under the pre-set "fewer than 2 usable pairs" rule.
- **An arm collapses in more than a quarter of its runs:** rerun that arm once at the next lower grid LR into `runs/lang-induction-rerun` as an amendment, and report both.
- **Interpretation times out:** analyze what finished, and report how many runs H5 rests on. The pruning loop already skips any r = 128 run without an `interpret.json`.
- **No GPU in stock:** wait, then report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Batch, budget and interpretation depth:** as for lang-source. The user chose 4 experiments with a 10 h cap and $12 per worker, then said "no budget, just work until finished", a standing go-ahead for the set. They chose probes plus invariance plus patching at every checkpoint.
- **Width:** the user's guidance was to sweep 16, 32, 64 and 128 for induction, with the LR piloted per width, pilots past 10,000 updates for wide trees, and stability checked over the full horizon. The width is chosen as the smallest stable width with a Bayes gap < 0.01 nats where doubling gains < 0.002. The same width is used for tree and perm, and the chain is matched on parameters separately.
- **Design from the user's review:**
  - induction is split into unambiguous retrieval and the multi-match process;
  - query placement is the capacity comparison, implemented as within-model stratification by where the target pair meets the query;
  - V² = 64 is a guide, not an established threshold;
  - "induction behavior through hierarchical associative representations" rather than a claim of induction-head formation, after the user's manager raised the two-layer induction-circuit result.

## Review

An independent review returned "needs revision" (no P0). Findings and changes:

- **Dict subspace patching equals whole-bond patching at r ≤ 64 (P1).** A regression probe's subspace is now capped at the top ridge directions holding 90% of the predicted variance, at most half the bond, with the random control at the same k. This is shared code (`interpret.fit`), so it also affects the other experiments' regression probes.
- **H2 mixed capacity with distance, and scored an unlearned run against itself (P1).** It now tests the capacity pattern (mid ≈ near > far, since only node 3_0 exceeds r = 16) against a graded distance pattern, and only on seeds that learned.
- **H4's second clause would pass for an undertrained r = 128 (P1).** It now needs a plateau (KL change over the last 10,000 updates < 5% or < 0.002). Otherwise the outcome is "still improving", and the width rule likewise.
- **Pruning could delete uninterpreted checkpoints (P1).** It now prunes per directory only when `interpret.json` exists, and keeps the run's last checkpoint.
- **Interpretation too slow in FP64 and ordered against H5 (P1).** Forwards now run in float32. Retrieval runs go first. Training has 4.5 h and interpretation 3 h.
- **4090 out of memory in pilots, with failed pilots silently dropped (P1).** The 4090 is removed from the list, and `pilot` exits when any pilot run didn't finish.
- **The width rule ignored "stable" (P1).** All seeds at r and 2r must be finished and not collapsed.
- **P2s:**
  - collapse is now relative to max(log V / 2, log V − 0.5), so it can fire for multi-match;
  - H5 gets linear checkpoints from 500 to 8,000, a "no gain" class, and node 3_0 in staging.pdf;
  - missing pairs give inconclusive, and H2 uses finished runs only;
  - the pilot is now 12,000 updates;
  - the checkpoint counts are corrected.
- **Re-check.** It confirmed these fixes and found two remaining P1s, both fixed:
  - the pruning loop's sort kept the wrong checkpoint (now `sort -V`, verified by the reviewer to keep ckpt_60000.pt);
  - the criteria section still had the old H2 and H4 rules (now the revised rules, matching `analyze.py`).
  - H3's minimum of 4 usable pairs is stated.


## Amendments

- **2026-11-14 00:12 UTC, pilots out of GPU memory (covered by "If something fails").** The worker is an A40 (44.4 GiB usable; L40S and A6000 had no stock). With 6 concurrent processes, four retrieval pilot runs at r = 64/128 died with CUDA OOM during evaluation. Rather than let the remaining pilots run on at 6 at a time (more r = 128 runs would OOM and `pilot` would exit anyway), the supervisor stopped the training job and reran `run.py` into the same `--out` with `--procs 3`, as the definition prescribes. Finished pilot runs are reused; nothing else changes. The training deadline (4.5 h) still counts from the end of the pilots.

## Outcome

Not run yet.
