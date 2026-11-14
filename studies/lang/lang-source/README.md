# lang-source: selecting and combining source tokens in a TreeTN

The untied tree reached threshold on a two-token conjunction y = f(x_i, x_j) at a median of 130 updates whether the two tokens first meet at level 1, level 3 or the root, with every run within one logging interval of that. The time to threshold was 130 updates in 5 of the 6 level-1 and level-3 runs (the other took 140), and 120–140 over all 18 two-source tree runs, against a logging interval of 10, so the pre-registered level comparison (H3) is "unresolved at n = 3" with every pair tied, and no level effect larger than about one logging interval (8%) is visible. The parameter-matched chain, by contrast, depends strongly on where the sources sit, but not in the way H2 predicted. It learns a single source at either end of the sequence in 56–90 updates and one in the middle in 800–1,090, so reading the source last is no faster than reading it first (H2 not supported). The mechanism checks on the tree came out as expected (H5 consistent): the source token is carried up the tree, f is decodable from the meeting bond and every bond above it, and patching the f subspace moves the prediction. The predicted fall in how well distractor tokens can be decoded (H6) did not happen at the pre-registered level, with a median change of 0.00.

![Updates to threshold per condition and arm, one point per seed](figures/t.pdf)

*Updates until diagnostic accuracy first reaches 0.9 × Bayes accuracy, one point per seed (n = 3 per condition and arm; tree32 only on three conditions). No run was censored. Fixed-source conditions `fixed_j{j}` have y = σ(x_j). Two-source conditions `two_{i}_{j}` have y = f(x_i, x_j), with the tokens meeting at level 1 for (0,1), level 2 for (1,2), level 3 for (3,4) and the root for (7,8), (0,8) and (0,15).*

## Setup

Each example has 16 iid uniform tokens over V = 8 symbols. The target is either a seeded random permutation of one token (fixed source, j ∈ {0, 7, 8, 15}) or a seeded random 8 × 8 table of two tokens (two-source). With probability 0.05 the target is replaced by a uniform symbol, so Bayes accuracy is 0.956. There are four arms:

- **tree:** the untied TreeTN at bond dimension r = 16 (61,440 parameters).
- **tied:** the same tree with cores shared within each level (14,464 parameters).
- **chain:** a shared-core MPS with output at the right end, at width 39 so its parameter count is close to the tree's (59,982 parameters).
- **tree32:** the untied tree at r = 32 (471,040 parameters), run on three conditions as a capacity check.

Every arm starts from the identity init and trains with Adam on fresh batches of 256 for 20,000 updates. Arms with the same condition and seed see the same data stream and the same process instance, so comparisons are paired. The learning rate was piloted per arm on `two_3_4` with seed 9, which no evaluation run uses. The pilot picked 1e-3 for tree, tied and chain, and 1e-4 for tree32 ([`lr.json`](../../../runs/lang-source/lr.json)).

The time to threshold t is the first logged update at which diagnostic accuracy (20,000 fixed examples) reaches 0.9 × Bayes accuracy. "Faster" means the gap exceeds the logging interval and the ratio exceeds 1.25. Final quality is KL to the Bayes posterior on 50,000 test examples. Interpretation fits held-out linear probes on whitened bonds (6,000 examples) at every checkpoint, for the 69 tree, tied and tree32 runs. The full method and rules are in [`EXPERIMENT.md`](EXPERIMENT.md).

## Results against the pre-registered criteria

All 99 planned runs finished. None collapsed and none was censored, so no health override applies.

| Hypothesis | Verdict | Evidence (n = 3 seeds per condition unless stated) |
|---|---|---|
| H1, tree symmetry control | holds | Fixed-source median t: 79, 79, 79 and 70 for j = 0, 7, 8, 15, a spread of 9 against a seed range of 13. Root pairs: 130, 130, 130. |
| H2, chain reads last faster than first | not supported | Chain t at j = 15: 56, 60, 79. At j = 0: 60, 56, 90. All 3 pairs tied. |
| H3, tree level 1 vs level 3 | unresolved at n = 3 | t at level 1: 130, 130, 130. At level 3: 130, 140, 130. All 3 pairs tied. Level 1 vs root (confounded with the root core's init): all 3 tied. Spearman ρ of t with meeting level over the 18 two-source tree runs: −0.14 (p = 0.57). |
| H4, root pairs (descriptive) | see below | Tree median t = 130 at all three root pairs. Chain: 980 at (7,8), 2,550 at (0,8), 140 at (0,15). |
| H5, mechanism (tree, 30 runs) | consistent | (a) 108/108 ancestor bonds decode the source at ≥ 0.95. (b) 18/18 bonds at or above the meeting point decode f at ≥ 0.95, with a median gain of 0.61 over step 0 at the meeting bond. (c) An f-subspace patch moves the prediction in 100% of cases, against 33% for random subspaces of the same size (n = 9 meeting bonds). |
| H6, distractor decodability falls | not supported | Median change from step 0 to the final checkpoint: 0.00 over 123 (run, bond) cases (quartiles −0.43, 0.00, +0.02), against the criterion ≤ −0.2. |

### The chain depends on distance from the nearer end, not from the output

H2 assumed the chain's difficulty grows with the number of shared-core multiplications between the source and the output. If that were right, j = 15 would be fast and j = 0 slow. Instead both ends are fast and the middle is slow by an order of magnitude: the median t is 60 at j = 0, 810 at j = 7, 910 at j = 8 and 60 at j = 15. The two-source conditions follow the same pattern. The median chain t rises as the pair moves inward, from 150 for (0,1) to 300 for (1,2), 550 for (3,4) and 980 for (7,8). It reaches 2,550 for (0,8), where one source sits at the start and the other in the middle, and drops back to 140 for (0,15), where both sit at the ends. So the chain learned (0,15) faster than (7,8) in 3 of 3 seeds, the opposite of the slowdown with sequence distance that H4 described.

This pattern was found after looking. One explanation that fits it, but which this experiment does not test, is about position. The chain applies the same core at every position, so it can only tell position 7 apart from position 6 or 8 by building a counter in its state. Positions 0 and 15 are marked for free, by the learned start vector and by the output map that reads the last state.

### The tree's mechanism checks

![Probe accuracy against update for the tree arm](figures/probes.pdf)

*Tree arm, one line per (seed, level) for each two-source condition: probe accuracy for f (top row, at levels that contain both sources), for source x_i (middle) and for the distractor probe (bottom), as the median over all nodes at that level, most of which hold no source. Root pairs have no f row because the root is not a bond.*

H5 was framed in advance as a consistency check, since a learned task forces the meeting bond to determine f, and every part of it passed by a wide margin. The source probes reach 1.0 on every ancestor bond. At the level-2 and level-3 meeting bonds the f probe goes from its step-0 value (0.29–0.31 at level 3) to 1.0 between updates 100 and 150, as the loss drops. At the level-1 meeting bond it starts at 0.60–0.72 and reaches 1.0 by update 40–60, while the diagnostic loss is still 1.94–2.05 against 2.08 at step 0, so at a level-1 meeting point f is decodable well before the loss has dropped much. The additive one-hot probe of the two tokens, which can't represent a conjunction, reads 0.46–0.53 on the same bonds, so the final 1.0 is a property of the bond and not of the data. The tied arm passes in the same way (gain 0.63, random patches 33%).

tree32 is "partly consistent". It fails (b) only because its level-1 meeting bond already decodes f at 1.0 at step 0 in all three seeds, so the gain is 0.00. A 32-dimensional bond over two 8-symbol tokens has enough room to hold f linearly from the identity init. That reflects the probe's capacity, not missing learning. Its random-subspace patches moved the prediction in only 5% of cases, against 33% for the r = 16 tree, because a subspace of the same size covers less of a wider bond.

Split by level (not pre-registered), the level-1 cases are saturated: there the distractor is the source's sibling token, and its probe reads 1.0 at step 0 and at the end in all 42 cases. One explanation is that a 16-dimensional bond has room for both tokens additively (7 + 7 centered dimensions), and nothing in the loss pushes the sibling out. Higher up the change is mixed. The median change is −0.12 at level 2 (n = 42, quartiles −0.67 and +0.03) and −0.32 at level 3 (n = 39, quartiles −0.48 and +0.27), and 35 of the 81 cases at these levels rose. The 42 zero changes at level 1 put the pooled median at 0.00, and without them it is −0.22. In the tied arm the pooled median change is −0.13.

### Other comparisons (descriptive)

- **Tree against tied:** the tree was faster in 27 of 30 paired runs and slower in none. Both arms used the same learning rate, so the cumulative-learning-rate count (30 of 30) differs from the update count only because the warmup turns the 3 ties into wins. The tree had lower final KL in 7 and higher in none. The tied arm's root pairs (7,8) and (0,15) ended at KL 0.008–0.019 nats, against at most 0.001 for the tree.
- **Tree against chain:** the tree was faster in 21 of 30 pairs, tied in 6 and slower in 3. All three slower cases have the source at an end of the chain (j = 0 in seeds 0 and 1, j = 15 in seed 0). The tree had lower final KL in 9 of 30 and higher in none.
- **Capacity:** tree32 and tree tie on final KL in all 9 paired runs (medians 0.0006–0.0009 against 0.0004–0.0006 nats). So on these three conditions r = 16 does not limit the final KL, as the sampled rank (at most 8 at every cut) predicted. tree32 takes more updates to reach threshold (130–280 against 70–130), but its learning rate is 10× smaller.

## How it was checked

- **Correctness checks:** `studies.lang.checks` passed 47/47 on the worker before training ([`results/checks.json`](results/checks.json)). These include Bayes predictors against generation frequencies, closed-form cross-entropies, locality of summaries, invariance moves, and patching equal to splicing (max difference 0).
- **Rank diagnostic:** the centered-logit rank was at most 8 at every cut, on the five seed-0 configurations it covers ([`results/rank.json`](results/rank.json)).
- **Symmetry control (H1):** it held, which is the evidence that the tree's training and timing code treats leaves symmetrically.
- **Health:** 99/99 runs finished, with no collapses and no censoring. All 69 tree-type runs were interpreted.
- **Fact-check:** a fact-checker agent checked about 115 claims against `results/`, `runs/lang-source/`, the analysis and plotting code and the experiment log. It corrected seven: the f-probe timing at the level-1 meeting bond, the description of the distractor row in the probe figure, the claim that finer logging could change H3's verdict, and four overstatements (the opening sentence, the H6 breakdown, the cumulative-learning-rate count and the scope of the capacity check).

## Not checked

- **Finer timing for H3.** Every two-source tree run reached threshold between 120 and 140 updates, so a level effect smaller than one logging step (10 updates) would not show. Logging every update over the first 300, which is a CPU-scale rerun of 18 tree runs, would show whether a small effect exists. It can't change the pre-registered verdict, because "faster" also needs a ratio above 1.25, and the level-1 and level-3 runs all crossed threshold between updates 121 and 140, a ratio of at most 1.16.
- **The chain's position effect.** The end-versus-middle pattern rests on 3 seeds at four positions and was not predicted. Sweeping j over all 16 positions, and running a chain with a position-dependent core, would test the counting explanation.
- **Rank for the other seeds.** The rank diagnostic covers seed 0 and five of the ten processes.

## Judgement calls

- **The tree32 learning rate.** The pilot picked 1e-4, the bottom of the grid. The definition says to proceed and report this, so tree32's speed in updates is not comparable to the tree's. Its diagnostic loss at 1e-4 and 3e-4 differed by 0.0002 nats. The tree's pilot was similarly close between 3e-4 and 1e-3 (a difference of 0.00005 nats), and it picked 1e-3.
- **H6 breakdown by level.** It is exploratory and does not change the pre-registered verdict.

## Files

- [`results/analysis.json`](results/analysis.json): every verdict with its pairs, and per-run t, KL and learning rate.
- [`results/interp.json`](results/interp.json): probe trajectories and the probe and patching details at each meeting bond.
- [`results/curves.json`](results/curves.json): diagnostic curves. [`figures/curves.pdf`](figures/curves.pdf) plots them.
- `runs/lang-source/`: per-run results, checkpoints and `interpret.json` (0.6 GB, data directory).
- [`run.py`](run.py), [`analyze.py`](analyze.py), [`plot.py`](plot.py): training, analysis and figures.

The worker was an A40 at $0.59/h, run from 2026-11-13 23:52 to 2026-11-14 02:38 UTC (2 h 47 m, about $1.64), against a 10 h cap and a $12 budget.
