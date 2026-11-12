# Byte-level language modelling on TinyStories: TTN against n-grams, MLP, chain and transformer

The task is to predict the next byte of TinyStories text from the 16 bytes before it. On this task the balanced tree tensor network (TTN) beats Kneser–Ney counting with up to 5 bytes of context, but loses to counting with 6 or 7 bytes, and it doesn't win overall. The untied r = 32 TTN reaches 1.286 test bits/byte, with all 3 seeds between 1.2857 and 1.2867. That is far below the best Kneser–Ney n-gram of order ≤ 4 (1.815), so the pre-registered hypothesis H1 is **supported**. The same TTN loses to two references:

- an order-8 Kneser–Ney n-gram counted on the full 100 MB training set, at 1.158;
- a parameter-matched shared-core chain (an MPS with one core reused at every position), at 1.213.

Against the parameter-matched transformer, the TTN comes out "equal" under the pre-registered rule: 1.286 against 1.301, a gap of 0.015, which is below the 0.02 margin. It is clearly better than the matched MLP (1.619) and the level-tied TTN (1.425). It is also much better than the same TTN with its fusion cores frozen at their starting values (2.228). Training the fusion cores, the tensors that combine pairs of child bonds inside the tree, is worth 0.94 bits/byte. It takes the TTN from between KN orders 3 and 4 (frozen, 2.228) to better than order 6. Training is not converged at 40k updates: a single run with four times as many updates reaches 1.211.

All 40 planned runs finished. None were skipped or diverged. The worker ran for 1.91 h on one A40 at $0.59/h, about $1.13 against a $10 budget.

## Results

Test bits/byte on all 5,000,223 targets of `test.bin`. Uniform guessing gives 8.0. Every neural run saw 10.24M training targets (40,000 updates of 256 windows), except the 160k run, and runs with the same seed saw the same batches. Values are the median and the range over seeds.

| model | params | seeds | test bits/byte, median [range] | validation | verdict against TTN r = 32 (pre-registered rule) |
|---|---:|---:|---|---|---|
| **TTN untied r = 32** | 851,968 | 3 | **1.2860** [1.2857, 1.2867] | 1.2772 | |
| chain r = 93 (matched) | 852,066 | 3 | 1.2127 [1.2115, 1.2127] | 1.2047 | chain better |
| transformer d = 128, 4 layers | 861,184 | 3 | 1.3009 [1.2992, 1.3028] | 1.2930 | equal |
| MLP, 192 ReLU units | 836,032 | 3 | 1.6186 [1.6136, 1.6215] | 1.6114 | TTN better |
| TTN level-tied r = 32 | 368,640 | 3 | 1.4253 [1.4241, 1.4256] | 1.4161 | untied better |
| TTN r = 32, fusion cores frozen | 851,968 (393,216 trained) | 3 | 2.2285 [2.2280, 2.2311] | 2.2196 | TTN better |
| TTN r = 32, variance init | 851,968 | 2 | 1.3665 [1.3427, 1.3903] | 1.3557 | identity init better |
| TTN untied r = 16 | 188,416 | 3 | 1.6339 [1.6331, 1.6373] | 1.6252 | |
| TTN untied r = 64 | 4,980,736 | 2 | 1.1757 [1.1756, 1.1758] | 1.1682 | |
| TTN untied r = 128 | 34,078,720 | 1 | 1.1583 | 1.1508 | |
| TTN untied r = 32, 160k updates (40.96M targets) | 851,968 | 1 | 1.2107 | 1.2016 | "not converged at 40k" |
| chain r = 32 (small) | 49,184 | 3 | 1.5510 [1.5500, 1.5525] | 1.5428 | |

The second set of baselines is interpolated modified Kneser–Ney (KN) n-grams. These are deterministic, so each is a single value. Order m conditions on the previous m − 1 bytes.

| KN order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| counted on 100 MB (100M targets) | 4.418 | 3.299 | 2.395 | 1.815 | 1.521 | 1.352 | 1.236 | **1.158** |
| counted on the first 10.24M bytes (same count as the neural targets) | 4.419 | 3.299 | 2.397 | 1.821 | 1.535 | 1.377 | 1.277 | 1.215 |

The TTN median beats the n-grams up to order 6 and loses from order 7 on. The order-8 n-gram beats it whether it is counted on all 100 MB (by 0.128) or only on as many bytes as the TTN saw as targets (by 0.070).

Two figures go with these tables:

- [`figures/curves.pdf`](figures/curves.pdf) has two panels. The left one shows the LR pilots. The right one shows validation curves for every group on a log axis, as the median with a band for the seed range, with KN orders 4 and 8 drawn in.
- [`figures/params.pdf`](figures/params.pdf) plots final test bits/byte against parameter count, with one marker per seed and the KN ladder as horizontal lines.

In both figures, `ttn_rR` is the untied TTN at bond dimension R, `ttn_tied_r32` the level-tied TTN, `ttn_frozenW_r32` the frozen-fusion TTN, `ttn_var_r32` the variance-init TTN, `ttn_r32_long` the 160k run, and `chain_rR` the chain.

### What each comparison says

The matched chain beats the TTN by 0.073 bits/byte, and every chain seed is below every TTN seed. The chain reuses one core at all 16 positions, so it has none of the tree's hierarchy. Here the tree's hierarchy is not an advantage, at least at about 850k parameters with this budget of targets and this LR.

The planned caveat about starting positions doesn't rescue the tree. Under identity init, the chain starts out responding to a byte the same way at every position, while the untied TTN starts with a separate embedding per position. That head start favours the TTN. Even so, the two are level for the first 1,000 updates, and the chain pulls ahead after that:

| median validation bits/byte at update | 500 | 1k | 2k | 5k | 10k | 40k (full validation) |
|---|---|---|---|---|---|---|
| TTN untied r = 32 | 2.529 | 2.082 | 1.795 | 1.591 | 1.474 | 1.277 |
| chain r = 93 | 2.557 | 2.076 | 1.736 | 1.528 | 1.411 | 1.205 |
| TTN level-tied r = 32 | 2.938 | 2.344 | 1.947 | 1.715 | 1.600 | 1.416 |
| transformer | 2.714 | 2.413 | 2.096 | 1.797 | 1.601 | 1.293 |
| MLP | 3.488 | 2.762 | 2.470 | 2.069 | 1.834 | 1.611 |
| TTN r = 32, fusion cores frozen | 3.211 | 2.754 | 2.551 | 2.406 | 2.335 | 2.220 |
| TTN r = 32, variance init | 7.853 | 7.938 | 7.929 | 7.519 | 2.043 | 1.356 |

The early points come from 2^17 fixed validation positions, and the last column from the full validation set.

The level-tied TTN loses to the untied one by 0.139. Here the caveat about starting positions does apply. The tied TTN starts position-blind and trails by 0.41 at update 500, but the gap is already down to 0.13 by update 10k and stays there. Tying also cuts the parameters from 852k to 369k. So this comparison can't separate the cost of sharing weights from the smaller capacity and the different starting point.

The transformer result runs against my prior that the transformer would win. All three TTN seeds are below all three transformer seeds, but the median gap of 0.015 is under the 0.02 margin, so the rule calls it "equal". With this many targets and this LR, a 4-layer transformer of the same size does no better than the TTN.

Freezing the fusion cores costs 0.94 bits/byte (2.228 against 1.286). Freezing the cores at their identity-init values is meant to keep the network near a sum of per-position responses read out through the root. How close it stays wasn't measured. That model lands between KN orders 3 and 4, so what lifts the full TTN past KN order 6 comes from training the fusion cores.

The variance-init arm was expected to start slowly or stall, and it started slowly. Both seeds sat near 7.9 bits/byte for the first 2,000 updates and first dropped below the KN unigram (4.415) at updates 7,500 and 7,000. They then trained to 1.343 and 1.390. Identity init is better by 0.081, and its seeds are about 50 times more consistent (a seed range of 0.001 against 0.048). The CPU pilots, which stopped at 5,000 updates, could not tell this slow start from a stall.

Bond dimension r helps steadily, with diminishing returns: 1.634 at r = 16, 1.286 at r = 32, 1.176 at r = 64 and 1.158 at r = 128. The r = 128 run comes closest to the order-8 n-gram. It narrowly loses to it on test (1.1583 against 1.1577) and narrowly beats it on validation (1.1508 against 1.1513), using 40 times the parameters of the r = 32 TTN. The r = 64 TTN is 0.018 above the order-8 n-gram on test, which the pre-registered rule would also call "equal". The r = 64 and r = 128 points rest on 2 seeds and 1 seed.

## How it was done

The full method, the commands and the pre-registered criteria are in [`EXPERIMENT.md`](EXPERIMENT.md).

- **Data.** The UTF-8 bytes of TinyStories (V2, GPT-4 stories), with a 0x00 byte between stories. The splits are disjoint by story: 100 MB of training data, and 5 MB each of validation and test. A training window is the 16 bytes before a uniformly sampled target. With 10.24M targets drawn from 100M positions, each training window is seen about 0.1 times on average, so only about 5% of the drawn windows are drawn more than once. The gap between validation and training bits/byte is 0.007 to 0.015 per seed for every non-variance group (group medians 0.007 to 0.012). (The variance-init arm's gap is 0.001 to 0.003.)
- **Models.** `TreeTN` and `ChainTN` from `ctn/tree.py`, with no biases and no nonlinearities. Every TTN and chain except the variance arm uses `identity_init(eps = 0.1)`. The MLP acts on the flattened one-hot context. The transformer is pre-LN and causal, and reads out only at the last position, so it gets one target per window like every other model.
- **Training.** Identical for every neural model: batch 256, Adam with no weight decay, 1,000 linear warmup steps, then a cosine decay to 0.1 × the peak LR, gradient-norm clipping at 1.0, and the same batch stream for the same seed.
- **LR.** Chosen per family from {3e-4, 1e-3} by a 5,000-update pilot on a 10 MB subset of the training data. Every family chose 1e-3 (pilot validation numbers are in `results/lr_choice.json`).
- **Metric.** Bits/byte is cross-entropy divided by ln 2, computed on every target of the validation and test splits.
- **Comparison rule, fixed before the run, with MARGIN = 0.02 bits/byte.**
  - "A better": every A seed is below every B seed and the medians differ by at least MARGIN.
  - "Equal": the medians differ by less than MARGIN and neither seed range is wider than MARGIN.
  - "Unresolved": anything else.

## How it was checked

- **Pre-training checks** (`studies.lm.checks`, `results/checks.json`) all passed on the worker before any training:
  - data hashes against `meta.json`;
  - window extraction;
  - uniform = 8.0 bits exactly;
  - parameter counts against their formulas for the TTN at r = 16, 32 and 64, the tied TTN, the matched chain and the MLP (the transformer only to within 5% of 850k);
  - the LR schedule endpoints;
  - finite per-level activations and gradients under both inits;
  - channel 0 carrying the constant under identity init;
  - a small instance of every family overfitting 128 random pairs;
  - vectorised n-gram counts equal to a naive count, with every KN distribution summing to 1 (largest error 4e-16).

  These rule out the bookkeeping errors that would make the comparison meaningless. They say nothing about whether the hypothesis holds.
- **Seed consistency.** For every 3-seed group, the seed range is 0.008 bits/byte or less (the MLP's is the widest; the others are 0.0042 or less). That is much smaller than every gap called a win between 3-seed groups (0.073 or more). The identity-vs-variance verdict is weaker: the variance arm's two seeds span 0.048, against a gap of 0.081.
- **Validation against test.** Test bits/byte is 0.007 to 0.011 above validation for every neural run (0.003 to 0.008 for the n-grams), and the neural groups rank the same on both splits. The one swap is r = 128 against KN order 8: r = 128 narrowly beats it on validation and narrowly loses to it on test.
- **Figure fix.** After the run, the legend in `params.pdf` covered the frozen-fusion markers, so I moved it outside the axes in `plot.py` and redrew both figures from the saved results. No numbers changed.
- **Fact check.** A separate agent checked about 100 claims against the result files, the code and the audit log. It found 15 that didn't fully hold, and all of them are corrected here. The most important were: the first draft said training the fusion cores accounts for "most" of what the TTN learns, when it is 0.94 bits/byte (14% of the gain over uniform, 30% of the gain over the KN unigram); it said only r = 128 ties the order-8 n-gram, when r = 64 is also within the margin; it overstated the pre-training checks; and it underestimated the cost of a clipping rerun.

## Not yet checked

- **A higher LR.** Every family chose 1e-3, the top of a two-point grid, so none of the models is known to be at its best LR. A pilot at 3e-3 for each family would take a few GPU-minutes. It could change the TTN-vs-transformer verdict, which rests on a 0.015 gap, and the TTN-vs-chain gap. The MLP had the largest pilot difference between its two LRs (2.62 against 2.24), so it is the most likely to be under-tuned.
- **The effect of clipping.** Clipping at 1.0 binds on most steps for most models. The clipped fraction at the end was 0.98 for the r = 32 TTN, 1.0 for the chain, the tied TTN, r = 16 and the variance arm, 0.77 for the transformer and 0 for the MLP, so most gradients get rescaled to unit norm. Adam largely undoes a constant gradient scale, but a per-step rescaling isn't constant, so clipping changes the optimiser differently for each family. A no-clip or clip-at-10 rerun of the matched models (about 40 minutes on one A40 with three processes) would show whether the ranking depends on it. The definition kept clipping for every family because the r = 32 TTN diverged without it in the CPU pilots, which used the variance init.
- **Convergence beyond one seed.** The verdict "not converged at 40k" rests on a single 160k run. Its 0.075 lead over the 40k seeds is 70 times the 40k seed range, so it is unlikely to be seed noise. Whether the chain's lead also holds at 160k was not tested, and that is the comparison that would say whether the chain is merely faster to train.
- **The checkpoints.** The eight checkpoints of the r = 32 TTN, seed 0, were saved to `artifacts/lm-bytes/ttn_r32_s0/` on the network volume for later interpretation and have not been looked at.

## Judgement calls

- **Reading the transformer comparison.** The rule says "equal", and that is the verdict reported. The consistent ordering of the seeds (TTN lower in all 9 pairs) is reported alongside it, but not upgraded to "TTN better", because the gap is under the margin fixed before the run.
- **One LR for the whole TTN family.** As planned, the r = 32 pilot's LR was used for r = 16, 64 and 128, the long run, the variance arm and the frozen-fusion TTN. The larger r and the frozen variant might prefer a different LR.

## What this means

On real text, the TTN's learned features clearly beat local counting at contexts up to 5 bytes (KN order 6), and they beat a same-size MLP by a wide margin. The TTN also matches a same-size transformer at this budget. But the tree's hierarchy gives it no advantage here: a chain with one shared core does better at the same size, at this LR and this budget of targets. A strong n-gram with 7 bytes of context still beats every model here by more than the 0.02 margin, except the 5M- and 34M-parameter TTNs. Those come within 0.018 and 0.0006 of it, and neither beats it on test. The next comparisons worth making are the chain against the TTN at 160k updates and at a higher LR. Those would show whether the chain's advantage is real or comes from faster optimisation.
