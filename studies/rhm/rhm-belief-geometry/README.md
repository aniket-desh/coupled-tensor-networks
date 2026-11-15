# Belief-state geometry of the RHM in a transformer's residual stream

An affine probe on the last layer of a 4-layer transformer, trained by next-token prediction on Random Hierarchy Model
(RHM) data, recovers the exact Bayesian belief state with R² ≥ 0.8 on prefixes the probe never saw in 10 of the 12
grammars that were both learned and testable. By the pre-registered rule, which needs 75%, H1 is **supported**. The
two failures are the deepest grammar and a synonym-rich one. At depth 6 (v3m2s2L6), last-layer R² is 0.52 to 0.84
across seeds, and at v = 16 with m = 4 synonyms it is 0.68 to 0.72. So both fail H1 on the R² ≥ 0.8 bar: all three
seeds at v16m4s2L4 and two of three at v3m2s2L6. The m = 8 grammar at v = 16 was not learned, with 0.040 nats/token
of excess loss after 50,000 updates. For the second hypothesis (descriptive), consider the 414 eligible
(grammar, seed, position) cases where the lookup next-token ceiling is below 0.9. In 383 of them the last-layer probe
beats that ceiling by more than 0.05. Almost all the exceptions are in the m ≥ 4 grammars, where the probe beats it
in only 8 of 37 cases. The ceiling is a lookup estimate, so this count is descriptive (see "Not yet checked").

## The evidence

**Last-layer probe on unseen prefixes, final checkpoint, pooled over eligible positions.** Each cell gives the median
over 3 seeds, with the range in brackets. "Eligible" means at least 1,290 distinct fit prefixes, ten times the 129
probe parameters per output dimension. R² is 1 − ΣSSE/ΣSST over all belief coordinates and positions. The next-token
ceiling here is the prefix-split ceiling pooled over the same positions.

| grammar | learned | eligible positions | R², trained L4 | R², random init | R², linear in tokens | next-token ceiling | H1 (all 3 seeds) |
|---|---|---|---|---|---|---|---|
| v3m2s2L4 | yes | 5 | 0.98 [0.95, 0.98] | 0.63 [0.63, 0.77] | 0.51 [0.47, 0.65] | 0.61 [0.52, 0.89] | pass |
| v3m2s2L5 | yes | 21 | 0.94 [0.92, 0.97] | 0.54 [0.33, 0.57] | 0.45 [0.26, 0.50] | 0.60 [0.54, 0.77] | pass |
| v3m2s2L6 | yes | 53 | 0.79 [0.52, 0.84] | 0.24 [0.17, 0.29] | 0.22 [0.13, 0.29] | 0.52 [0.29, 0.53] | **fail** (passes in 1 of 3 seeds) |
| v3m3s3L3 | yes | 14 | 0.95 [0.92, 0.97] | 0.58 [0.50, 0.62] | 0.39 [0.36, 0.51] | 0.60 [0.51, 0.66] | pass |
| v4m2s2L5 | yes | 21–22 | 0.95 [0.94, 0.98] | 0.50 [0.47, 0.59] | 0.49 [0.46, 0.55] | 0.61 [0.60, 0.63] | pass |
| v8m2s2L4 | yes | 7 | 0.97 [0.94, 0.98] | 0.51 [0.50, 0.51] | 0.50 [0.45, 0.52] | 0.76 [0.71, 0.78] | pass |
| v8m2s2L5 | yes | 23 | 0.87 [0.87, 0.88] | 0.33 [0.32, 0.37] | 0.41 [0.37, 0.46] | 0.57 [0.56, 0.61] | pass |
| v8m4s2L3 | yes | 3 | 0.97 [0.97, 0.98] | 0.33 [0.30, 0.39] | 0.28 [0.27, 0.34] | 0.99 [0.99, 0.99] | pass |
| v8m4s2L4 | yes | 11 | 0.90 [0.84, 0.91] | 0.19 [0.17, 0.20] | 0.19 [0.16, 0.19] | 0.92 [0.91, 0.92] | pass |
| v8m6s2L3 | yes | 3 | 0.93 [0.92, 0.94] | 0.27 [0.26, 0.32] | 0.24 [0.23, 0.27] | 1.00 [1.00, 1.00] | pass |
| v16m2s2L4 | yes | 7 | 0.89 [0.88, 0.89] | 0.40 [0.39, 0.42] | 0.51 [0.46, 0.52] | 0.68 [0.67, 0.70] | pass |
| v16m4s2L4 | yes | 11 | 0.68 [0.68, 0.72] | 0.13 [0.13, 0.14] | 0.15 [0.14, 0.16] | 0.79 [0.79, 0.80] | **fail** (R² < 0.8) |
| v16m8s2L4 | **no** (excess 0.040) | 12 | 0.43 [0.42, 0.45] | 0.08 [0.08, 0.08] | 0.08 [0.08, 0.09] | 0.98 [0.98, 0.98] | not judged |

Grammar names read vVmMsSLL: v symbols per level (and v root classes), m synonymous productions per symbol, branching
s, depth L, so a sequence has T = s^L tokens. Wherever a grammar fails H1, it fails on the 0.8 threshold. The margin
over the better of the two controls is at least 0.22 in every run of every testable grammar.

Four grammars (v3m2s2L3, v3m3s3L2, v6m3s3L2, v8m2s2L3) have no eligible position and are figure-only, as the
definition planned. On the sequence split, which lets test prefixes repeat fit prefixes, the trained network reaches
an all-layer R² of 1.00 there, but the random-init network reaches 0.95 to 1.00 as well. These grammars have at most
about 1,000 distinct prefixes per position, and any injective embedding of that many prefixes can be fit, so these
numbers carry no evidence of learning.

[`figures/geometry_v8m4s2L4.pdf`](figures/geometry_v8m4s2L4.pdf) shows what the numbers mean, for seed 0 on the
sequence split. The first column plots the true beliefs on their top two principal components at four positions. The
ceilings printed on the rows (0.61 to 0.86) are per-position, sequence-split ceilings, not the pooled prefix-split
number in the table. The trained all-layer probe reproduces the arrangement, and the last-layer probe reproduces it
more loosely. The random-init all-layer probe matches it only at t = 3, where the 54 beliefs are few enough to
memorize. At t = 7, 11 and 13 it collapses into a blob or a few clusters (R² 0.23 to 0.26), against 0.79 to 0.94 for
the trained all-layer probe and 0.60 to 0.67 for the trained last layer. Shuffled labels give a point mass.

The other `geometry_*.pdf` files show the same panels for every grammar (seed 0). `distances_*.pdf` plots
probe-output distances against true-belief and next-token distances, and `triangles_*.pdf` exists for the six v = 3
grammars. [`figures/summary.pdf`](figures/summary.pdf) shows pooled R² by layer for the trained and random-init
networks, with the next-token ceiling and the token readout as horizontal lines.
[`figures/positions.pdf`](figures/positions.pdf) gives R² per position, with ineligible positions shaded, and
[`figures/training.pdf`](figures/training.pdf) shows R² and excess loss over training.

**Descriptive results.**

- **Which layer.** Of the 36 runs in the 12 learned, testable grammars, the best single layer is L4 in 24 and the
  post-LayerNorm output in 12. On the same L4-eligible positions, concatenating all six activation sets beats L4 in
  all 36 runs. The gain is largest at v16m4s2L4 (+0.16 to +0.18). At v3m2s2L6 it is +0.04 to +0.08, no more than in
  passing grammars such as v8m2s2L5 and v8m4s2L4, and seed 2 stays at 0.59.
- **Over training.** In the same 36 runs, the median last-layer R² on eligible positions was 0.38 at initialization,
  then 0.56, 0.71, 0.73, 0.89, 0.92 and 0.93 at updates 300, 1,000, 3,000, 10,000, 30,000 and 50,000. The probe keeps
  improving after the loss is within 0.01 nats/token of Bayes, which these 36 runs first reach at a median of update
  2,000 (range 1,000 to 15,000).
- **Trends.** These come from reading the table, not from a planned test. R² is lower for deeper grammars at fixed
  (v, m): it is 0.98, 0.94 and 0.79 for v3m2 at L = 4, 5, 6, and 0.97 and 0.87 for v8m2 at L = 4, 5. It is also lower
  with more synonyms at v = 16, L = 4: 0.89, 0.68 and 0.43 for m = 2, 4, 8. At v = 8, L = 3 the m = 4 and m = 6
  grammars are both ≥ 0.92, but they have only 3 eligible positions each.
- **Below the next-token ceiling.** In v8m4s2L3, v8m6s2L3, v8m4s2L4 and v16m4s2L4, the exact next-token distribution
  predicts the belief better than the last-layer probe does (ceiling 0.79 to 1.00, against 0.68 to 0.97), while the
  network's output is near Bayes (excess ≤ 0.0011 nats/token). This suggests that the last layer holds the belief in
  a form that the unembedding and softmax decode, but that is not fully affine in the belief.

## How it was done

The RHM is a random context-free grammar of depth L. Each of the v symbols at a level has m synonymous productions
into s symbols of the level below, and the leaves are the tokens. Read left to right, it is a hidden Markov process.
The hidden state after t tokens is the stack of pending subtrees, meaning the subtrees whose symbol is fixed but none
of whose leaves have been emitted. The belief b_t = P(pending symbols | x_<t) is computed exactly by
`grammar.stack_belief` as a joint over the pending stack, of dimension D_t. Positions with D_t > 4,096 are skipped,
which removes t = 1 in the v16 grammars and v8m2s2L5.

- **Model and training.** 4 pre-LN blocks, d_model 128, 4 heads, MLP 512, learned positions, and a BOS token, so the
  residual at input position t has seen x_<t. Each run trains for 50,000 Adam updates at batch 512, with fresh
  sequences every update, peak LR 1e-3, and a cosine decay. Seeds 0, 1 and 2 set both the grammar's rules and the
  initialization, so the three seeds of a grammar name are three different random grammars with the same parameters.
  That gives 17 configurations × 3 seeds = 51 runs.
- **Learning check.** Test loss on 8,192 fixed sequences against the exact Bayes loss from the grammar's likelihoods.
  All 48 runs outside v16m8s2L4 end with at most 0.0010 nats/token of excess.
- **Probe.** At each position, an affine least-squares map from activations to b_t is fit on the fit half of 60,000
  sampled sequences, so prefixes are weighted by their probability. The **prefix split**, which the criteria use,
  assigns each distinct prefix x_<t to fit or test at random, so the probe is scored only on prefixes it never saw.
  Without this, a grammar with fewer distinct prefixes than features lets any injective embedding interpolate the
  belief, and the CPU pilot measured R² 0.976 on the sequence split and −4.9 on the prefix split for a random network.
  The **sequence split** (30,000/30,000 sequences, as in Shai et al. 2024) is used for the geometry figures only.
- **Controls, scored on the same positions as the trained probe.** The same probe on the randomly initialized
  network (checkpoint 0). The best affine function of the one-hot prefix tokens. The next-token ceiling: the test R²
  of predicting b_t by the mean belief of the fit rows that have the same exact next-token distribution. Shuffled
  labels.

The exact commands, the criteria and every amendment are in [`EXPERIMENT.md`](EXPERIMENT.md).

## How it was checked

- **Correctness checks before training passed.** `studies.rhm.checks` passed 28 of 28, including `stack_belief` and
  next-token-from-belief against brute-force enumeration (max error 3e-16); these are in `results/checks.json`.
  `check_probe.py` passed at all seven positions of v8m4s2L3 on synthetic activations: a linear image of b_t gives
  R² ≥ 0.99999, shuffled labels give R² between −0.06 and −0.03, the ceiling is ≤ 1, and the affine fit keeps beliefs
  summing to 1 within 2e-11 (worker log).
- **Shuffled-label control.** Pooled R² is between −0.023 and 0.001 in all 51 runs, well under the 0.05 bound that
  would have signalled a probe bug.
- **Run-failed criteria.** None was triggered: checks passed, CUDA was present, and all 51 runs finished both
  training and analysis.
- **Analysis reruns.** Four analyses failed with CUDA out of memory when two analyses shared the 32 GB GPU:
  v16m2s2L4 seeds 0 and 2, and v8m2s2L4 seeds 1 and 2. All were rerun with the same script and data. v16m2s2L4_s0
  failed a second time next to a v8m4s2L4 analysis and was then rerun alone. The first summary was written before it
  had a result, so it read "9 of 11" with v16m2s2L4 censored. The final summary has all 51 runs.
- **Fact check.** An independent fact-checking pass compared a draft of this write-up against `results/summary.json`,
  the per-run `analysis.json` and `result.json` files, the worker logs, `plot.py` and `EXPERIMENT.md`. It found 19
  claims that did not fully hold. Most were overstatements: the opening summary, the H2 framing, the comparison of
  all-layer against last-layer probes, the population behind the training numbers, and the account of the reruns.
  It also found a rounding slip (0.85 for 0.84), mis-described figure contents, and the scope of `check_probe.py`.
  All 19 were corrected.

## Not yet checked

- **Is the next-token ceiling an upper bound on unseen prefixes?** It is a lookup table from exact next-token
  distributions to mean beliefs. On the prefix split it falls back to the global mean for any test distribution that
  never occurs among the fit prefixes. When many distinct distributions occur, this understates what a smooth function
  of the next-token distribution could do, so the H2 count (383 of 414) may be inflated. That count is also uneven:
  146 of the 414 cases come from v3m2s2L6, and 6 from the unlearned v16m8s2L4. A kernel or nearest-neighbour
  regression from P(x_t | x_<t) to b_t on the same split would settle it. That is a CPU job of minutes on the saved
  data plus the belief computation.
- **Seed versus grammar.** Each seed draws a new grammar and a new initialization together. So the large spread at
  v3m2s2L6 (R² 0.52, 0.79, 0.84, all with excess ≤ 0.0001) cannot be attributed to either. Retraining one grammar
  under several init seeds would separate them, at about an hour of GPU time per T = 64 run.
- **Longer training for v16m8s2L4.** Its excess loss was still falling, but slowly: in seed 0 it went from 0.0407 to
  0.0403 nats/token over the last 5,000 updates. The definition forbids retuning, so it stays "not learned", and a
  longer run would be a new experiment.

## Judgement calls

- **Analysis scheduling.** The definition ran analyses two at a time. The worker was a 32 GB RTX PRO 4500, because
  the A6000 was out of stock at launch (launch output, recorded in the Amendments). After the first out-of-memory
  failures, the v16 analyses ran one at a time. The T ≤ 9 grammars ran alongside them, and later so did some T = 16
  and T = 32 grammars. Those caused three more out-of-memory failures, which were rerun. This changed only the order
  of the work.
- **Collection without a merge.** The probe projections (`results/*/proj_ckpt*.npz`) total 2.1 GB, ten times the
  definition's estimate, because they store coordinates for every test row at every position. They sit on disk in
  this folder but are kept out of git by its `.gitignore`. The analysis JSONs, distance samples, summary and figures
  are committed. `plot.py` needs the projection files to redraw the figures.
- **The triangle figures do not show learning.** `triangles_*.pdf` uses the sequence split at early positions,
  where a pending node's marginal takes only a few values. There the random-init network matches the trained one,
  and the figure's caption says so. The geometry figures at later positions are the ones that separate trained from
  random.

## What this means

On this process the Shai et al. picture mostly holds. In 10 of 12 learned grammars, a network trained near Bayes lays
out the belief over the hidden stack affinely in its last layer, with R² ≥ 0.8 on unseen prefixes. Its margin over
both a random network and a linear readout of the tokens is 0.2 to 0.7 in R². It degrades at depth 6 and at v = 16,
m = 4. At v = 16, m = 4 the all-layer probe recovers much of the gap (0.86 against 0.68); at depth 6 it adds no more
than it does elsewhere. The two cheap follow-ups are the smooth next-token ceiling, which decides whether H2's count
holds up, and the init-seed replicate at v3m2s2L6.
