# lang-parity: parity of k bits in a TreeTN, across degree, init scale and input bias

Parity of k bits is a function that the TreeTN can represent with a scalar bond, but finding it is hard, and how hard depends on the degree k and on the scale of the initialization. With unbiased inputs and the identity init, the median number of updates to reach 0.9 × Bayes accuracy grows from 50–220 at k = 2 to 128–4,500 at k = 4 (50–100 and 128–4,500 at ε ≤ 0.3). At k = 8 and k = 16 the small-init runs almost never get off the uniform-output plateau in 60,000 updates (11 of the 12 bias-0 runs at ε ≤ 0.1 are censored). Raising the initial noise scale ε from 0.01 to 0.3 shortens the plateau in every tested pair (6 of 6 at k ∈ {4, 8}), but ε = 1.0 breaks the trend and is slower than ε = 0.3 at k ≤ 4. In the 18 pre-registered pairs (k ∈ {4, 8}, ε ≤ 0.3), a 0.2 input bias, which gives low-order correlations with the target, shortens the plateau in 17 with 1 tie and no losses, timed on unbiased inputs. At k = 2 it is slightly slower in every cell, and at ε = 1.0 it is mixed (3 wins, 2 losses, 1 tie). All four pre-registered verdicts (H1–H4) are "supported", with n = 3 seeds per cell. When a bias-0 run learns (36 runs), its output at the end of training depends causally on each subtree's partial parity: patching that one-dimensional summary from a donor flips the prediction at every one of 203 internal nodes, against a median of 0.17 for random subspaces. That causal use appears all at once, at the loss drop, rather than level by level (21 of 24 runs "simultaneous", 0 bottom-up), but consecutive checkpoints are a factor of 1.1–1.8 apart in update count, so "simultaneous" means within that resolution.

![Updates to threshold against init noise ε, one panel per degree k](figures/t_grid.pdf)

*Updates until diagnostic accuracy first reaches 0.9 × the Bayes classifier's accuracy on the timing set (0.876–0.879; nominally 0.8775), one point per seed (n = 3 per ε, bias and k). Blue: bias 0. Orange: bias 0.2, timed on an unbiased diagnostic set. Open circles above the dotted line at 60,000: never reached (censored). Points at the same ε are offset slightly so the two biases don't overlap.*

![Diagnostic loss against update for every bias-0 run](figures/curves.pdf)

*Diagnostic loss (nats, log scale) for the 12 main bias-0 runs per k (3 seeds × 4 ε). Dotted line: ln 2, the loss of a uniform output. Every learned run sits on or near the ln 2 plateau and then drops to the Bayes loss (0.117 in expectation, 0.111–0.124 on the individual diagnostic sets). The 12 censored runs stay at ln 2 for all 60,000 updates. Where logging is every 10 updates, the drop at ε ≤ 0.3 takes under about 100 updates. After 3,000 updates logging is every 1,000, so drops there aren't resolved more finely. The ε = 1.0 runs start at about 690–2,450 nats, first fall to the plateau, and at k ≤ 4 take 400–2,400 updates from leaving it to reach 0.15 nats.*

## Setup

Each example has 16 iid bits x_t with P(x_t = 1) = 1/2 + bias, and the target is y = (x_0 + … + x_{k−1}) mod 2, replaced with probability 0.05 by a uniform random bit (a flip with probability 0.025), so Bayes accuracy is 0.975. The first k leaves form an aligned subtree for k = 2, 4, 8 and 16. With bias 0, every proper subset of those k bits is independent of y, so no low-order statistic points toward the answer. With bias 0.2 the subsets do carry signal, of size (2·bias)^{k−1}.

The model is the untied TreeTN on 16 leaves at bond dimension r = 16 (58,368 parameters). The representation is not the bottleneck: the sampled rank of the centered logit is 1 at every cut for all four k ([`results/rank.json`](results/rank.json)), because each subtree only needs the product of its ±1 bits. Every fusion core below the root starts as the identity pass-through plus ε × (variance-init noise), every leaf as e_0 + ε × noise, and the root core keeps its variance init at every ε, with ε ∈ {0.01, 0.1, 0.3, 1.0}. At ε = 0 the network would be constant.

Training uses Adam on fresh batches of 256 for 60,000 updates, with peak learning rate 1e-3 fixed in advance for every run (warmup 100, cosine to 0.1 × peak), so ε, k and bias are the only factors that differ. Every run with the same (k, bias, seed) sees the same data stream. The main grid is k ∈ {2, 4, 8, 16} × ε × bias ∈ {0, 0.2} × 3 seeds = 96 runs. Two descriptive arms, at ε = 0.1 and bias 0, add 9 runs of a mod-4 sum (k ∈ {2, 4, 8}) and 6 runs at r = 2 (k ∈ {4, 8}), 111 runs in all.

The time t is the first logged update at which accuracy on a fixed set of 20,000 examples reaches 0.9 × Bayes accuracy. Runs with bias are timed on a separate unbiased set of 20,000. On biased inputs a low-degree function of the count of ones already passes the threshold, so only the unbiased set requires the parity itself. Logging is every 10 updates up to 2,000, every 50 up to 3,000 and every 1,000 after, plus log-spaced updates throughout. "Faster" means the gap exceeds the logging interval and the ratio exceeds 1.25, and a finite t beats a censored one. A pair in which both runs are censored is uninformative and is excluded from the counts. The full method is in [`EXPERIMENT.md`](EXPERIMENT.md).

## Results against the pre-registered criteria

All 111 planned runs finished their 60,000 updates and none collapsed, so the "more than a quarter unfinished" override does not apply. Censored runs are counted below and in the table, never dropped.

| Hypothesis | Criterion | Verdict | Evidence |
|---|---|---|---|
| H1, larger k is not faster (bias 0) | For each ε: no adjacent (k, 2k) pair with 2k faster in ≥ 2 seeds, and k = 2 beats k = 16 in ≥ 2 informative seeds with none the other way | supported (all 4 ε) | No adjacent pair has 2k faster in any seed. Of 36 adjacent pairs, 30 have 2k slower, 1 is a tie (ε = 1.0, k = 8 vs 4, seed 2) and 5 are both censored. k = 2 beats k = 16 in 3 of 3 seeds at every ε. |
| H2, ε = 0.3 faster than ε = 0.01 (bias 0, k ∈ {4, 8}) | ≥ 3 informative pairs, ε = 0.3 wins ≥ 80% and loses none | supported | 6 of 6 pairs informative, 6 wins (3 at k = 4, 3 at k = 8). |
| H3, bias 0.2 faster than bias 0 (k ∈ {4, 8}, ε ∈ {0.01, 0.1, 0.3}), timed on unbiased inputs | ≥ 6 informative pairs, bias wins ≥ 75% and loses ≤ 1 | supported | 18 of 18 pairs informative, 17 wins, 1 tie (k = 4, ε = 0.3, seed 1), 0 losses. |
| H4, causal use of the partial parity (learned bias-0 runs, final checkpoint) | Median "moved" ≥ 0.8 for the partial-parity subspace patch, at least 0.4 above random subspaces of the same dimension | supported | Partial-parity patch: moved = 1.0 at all 203 internal nodes whose span meets the k bits, in 36 learned runs. Random subspaces: median 0.17 (quartiles 0.05–0.41, range 0–0.56). |
| H5, order of discovery (descriptive) | Classify each learned run with k ≥ 4 | 21 simultaneous, 3 other, 0 bottom-up, 0 with a level never causal | See [Order of discovery](#order-of-discovery). |

### Time to threshold by cell

Median t over 3 seeds, with the per-seed values. A dash (–) is a censored run, and a median is censored when at least two of its three seeds are.

| k | ε | bias 0 | bias 0.2 (unbiased timing) |
|---|---|---|---|
| 2 | 0.01 | 100 (100, 100, 100) | 110 (106, 110, 110) |
| 2 | 0.1 | 70 (70, 73, 60) | 88 (88, 88, 80) |
| 2 | 0.3 | 50 (50, 50, 40) | 57 (70, 57, 42) |
| 2 | 1.0 | 220 (220, 220, 88) | 280 (280, 480, 100) |
| 4 | 0.01 | 4,500 (4,500, 7,715, 3,037) | 120 (120, 120, 120) |
| 4 | 0.1 | 530 (530, 710, 530) | 106 (106, 110, 106) |
| 4 | 0.3 | 128 (128, 120, 140) | 100 (100, 100, 88) |
| 4 | 1.0 | 550 (550, 510, 1,195) | 640 (760, 640, 620) |
| 8 | 0.01 | censored (–, 27,000, –) | 1,720 (1,570, 1,720, 2,650) |
| 8 | 0.1 | censored (–, –, –) | 1,480 (1,940, 1,480, 1,400) |
| 8 | 0.3 | 6,402 (6,402, 2,450, 6,402) | 790 (960, 790, 670) |
| 8 | 1.0 | 1,200 (1,050, 3,659, 1,200) | 1,010 (823, 1,770, 1,010) |
| 16 | 0.01 | censored (–, –, –) | censored (–, –, –) |
| 16 | 0.1 | censored (–, –, –) | censored (–, –, –) |
| 16 | 0.3 | 21,000 (21,000, 18,000, –) | 13,000 (16,000, 9,000, 13,000) |
| 16 | 1.0 | 5,313 (4,000, 5,313, 6,000) | 3,291 (2,150, 3,291, 4,000) |

Three patterns in this table go beyond the criteria and were not predicted in this form:

- **ε = 1.0 is slower than ε = 0.3 at k ≤ 4 in every seed, and faster by median at k ≥ 8 (in 5 of 6 seeds).** The median initial diagnostic loss is 0.693 nats at ε = 0.01, 0.697 at 0.1, 0.817 at 0.3 and 762 at 1.0 (median over 12 runs each), so the ε = 1.0 runs first have to shrink huge logits: their diagnostic loss takes 185–1,220 updates to fall below 0.75 nats (curves above), and at k ≤ 4 that costs more than the escape itself. At k ≥ 8 their median t is lower than at any smaller ε: ε = 1.0 beats ε = 0.01 in 6 of 6 pairs at k ∈ {4, 8} and beats ε = 0.1 in 4 of 6 (1 loss, 1 tie). The definition kept ε = 1.0 out of H2 for this reason.
- **Bias speeds up k ∈ {4, 8} at ε ≤ 0.3 and slightly slows k = 2.** At k = 2, the bias-0.2 median is slightly higher in every ε cell (for example 88 against 70 at ε = 0.1). At k = 4, ε = 0.01 it cuts the median from 4,500 to 120 updates, and at k = 8 it rescues every censored cell. At k = 16 and ε ≤ 0.1, where the low-order signal is (0.4)^15 ≈ 1e-6, all 6 bias runs stay censored. At ε = 1.0 the bias comparison is mixed (3 wins, 2 losses, 1 tie at k ∈ {4, 8}).
- **The biased runs did learn the parity, not the low-degree shortcut.** For bias runs at k = 8 and ε ≤ 0.3, the time on the biased set equals the time on the unbiased set in all 9 runs. At k = 4, ε = 0.01 the two times are also equal. At k = 4, ε ∈ {0.1, 0.3} the biased set is reached 0–12 updates earlier, and at ε = 1.0 130–250 updates earlier.

### Order of discovery

For each learned bias-0 run with k ≥ 4, the causal gap at a node is the fraction of predictions flipped by patching its partial-parity subspace minus the fraction flipped by a random subspace. For each tree level, the analysis finds the first checkpoint at which the median gap over that level's nodes (those holding at least 2 of the k bits) reaches 0.5. In 21 of 24 runs every level and the run's t fall in the same checkpoint interval. In the other 3 runs (k = 4 at ε = 1.0 seed 1 and ε = 0.1 seed 2, k = 8 at ε = 1.0 seed 0), levels 2 and 3 cross 0.5 one checkpoint before level 1, the reverse of bottom-up. No run is bottom-up.

The checkpoints are at updates 0, 32, 57, 101, 150, 181, 323, 450, 577, 1,030, 1,500, 1,841, 3,291, 4,500, 5,882, 10,511, 15,000, 18,785, 30,000, 33,573 and 60,000. Consecutive checkpoints are therefore a factor of 1.1–1.8 apart in update count, and a bottom-up order inside one interval would be classed as simultaneous. Because the loss drop itself is shorter than one checkpoint interval wherever logging resolves it (under about 100 updates at k ≤ 4, ε ≤ 0.3), and the late intervals are thousands of updates wide, this measurement can't separate the levels of the late-learning runs.

### Found after looking: partial parities in the runs that never learned

This is exploratory. I looked at one quantity, the descriptive partial-parity probe (a linear probe on the whitened bond, which the definition reports but does not use as a criterion), in the censored runs. In all 12 censored main bias-0 runs at k ∈ {8, 16}, the level-3 probe (8-bit partial parities, the full parity at k = 8) ends at 0.49–0.57. In 11 of them the level-2 probe for the 4-bit partial parities starts at 0.53–0.60 and rises to 1.0 in 10 and to 0.78 in one (k = 8, ε = 0.01, seed 2). In the twelfth (k = 16, ε = 0.3, seed 2) it is already 1.0 at initialization. The ε ≤ 0.1 panels of [`figures/probes.pdf`](figures/probes.pdf) show this for seed 0. So the stuck runs have made the 4-bit partial parities linearly decodable but not the 8-bit ones. Since whitening removes scale, this says nothing about whether those directions are large enough to matter to the output, and it isn't evidence for a bottom-up order in the learned runs.

### Descriptive arms

- **mod-4 sum** (y = sum of k symbols in {0..3} mod 4, ε = 0.1, bias 0): k = 2 learns in 100–120 updates, k = 4 in 2,900, 2,900 and 41,322, and k = 8 is censored in all 3 seeds (median final KL to the Bayes posterior 1.19 nats). Compared with parity at the same ε, k = 4 is about 5× slower (2,900 against 530) and k = 8 is censored in both.
- **r = 2 bond** (parity, ε = 0.1, bias 0, 184 parameters): k = 4 learns in 25,000–36,000 updates against 530–710 at r = 16, 35–68× slower by seed (57× by median), though a scalar bond suffices to represent it. k = 8 is censored at both widths.

## How it was checked

- **Correctness checks** ([`results/checks.json`](results/checks.json)): all 47 passed before training. They include a check that no proper subset of the k bits correlates with y at bias 0 (maximum |corr| = 0.0035), a closed-form check of the parity Bayes loss (0.1169 against 0.1169), a check that patching equals splicing in the tree, and summary-locality and resampling checks for the parity process with and without bias.
- **Rank check** ([`results/rank.json`](results/rank.json)): the sampled centered-logit rank is 1 at every cut for parity at all four k, and 3 at the cuts inside the span of the k = 4 mod-4 sum (1 elsewhere; k = 2 and 8 weren't measured), as expected.
- **Run health:** 111 of 111 runs finished, none collapsed. All 81 learned parity runs (main and r = 2) end with test accuracy equal to the Bayes classifier's accuracy on the same 50,000 test examples (0.973–0.977; the nominal Bayes accuracy is 0.975).
- **Figures:** I regenerated `curves.pdf` with a log loss axis after the first version, on a linear axis, showed only the ε = 1.0 runs. No number changed.
- **Fact-check:** a separate fact-checking agent compared about 80 claims with the result files, logs and code. 16 didn't fully hold and are corrected here. The ones that changed the substance: the label noise is a 0.05 replacement by a uniform bit (a 0.025 flip), not a 0.05 flip; the root core keeps its variance init at every ε; ε = 1.0 breaks the ε trend; the bias effect is limited to the pre-registered cells; and there are 12 censored k ≥ 8 bias-0 runs, not 11.

## Not yet checked

- **The H5 classification at finer resolution.** Re-running interpretation on denser checkpoints around each run's loss drop would say whether the causal use really appears at once or level by level within an interval. That needs retraining a few runs with extra checkpoints (minutes each on a GPU, well under $1). Only the H5 description depends on it.
- **How long the censored runs would take.** All censored runs were stopped at 60,000 updates. A longer horizon for k = 8 at ε ≤ 0.1 would turn the censored cells into numbers and show whether the growth with k is a steep rise or a wall. The verdicts don't depend on it, since they exclude both-censored pairs.
- **Seeds.** Every cell rests on 3 seeds. The verdicts are clean (no losses in H2 or H3, no violations in H1), but the medians in the table have wide spreads at k = 8 and 16 (for example 2,450 to 6,402 at k = 8, ε = 0.3).
- **Whether the level-2 probe pattern in stuck runs reflects causal use.** The patching measure exists for these runs in `runs/lang-parity/*/interpret.json` but I didn't analyse it for the censored runs.

## Judgement calls

- **One fixed learning rate, no pilot.** The definition fixed 1e-3 for every run, so ε, k and bias are the only factors that differ. A per-cell LR might shorten the hard cells, and a different LR could move the ε = 1.0 crossover, but it would confound the comparison.
- **The plot fix.** I changed only the loss axis of `curves.pdf` to log scale (and corrected the `plot.py` docstring, which described bias-0.2 points as squares). This was a presentation change made after looking at the figure.
- **The exploratory probe section** reports one quantity I looked at after the run, in the censored runs only. I didn't search other measures or levels for patterns.
