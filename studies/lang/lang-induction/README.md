# lang-induction: key-value retrieval and multi-match induction in a TreeTN across bond dimension 16–128

Width decides retrieval: final test KL falls from a median of 0.33 nats at r = 16 to 0.0037 at r = 128 (3 of 3 seeds), and r = 64 is the smallest width that meets the pre-set "stable and within 0.01 nats" rule. Two caveats change how to read this. Every run at r ≥ 32 except multi-match r = 32 was still improving when training ended, and retrieval r = 64 and r = 128 pass the convergence test only through its 0.002-nat absolute clause. A parameter-matched chain also does better than the tree at small width, so the tree's hierarchy isn't what makes the task learnable. Against the pre-registered criteria:

| | Criterion | Result |
|---|---|---|
| H1 | r = 128 beats r = 16 on retrieval KL in 3/3 seeds | **supported** (3/3); smallest stable width r = 64 |
| H2 | capacity pattern at r = 16 (mid ≈ near > far) and the gap closing at r = 128 | **not supported**: graded near > mid > far in 3/3 seeds; the gap does close at r = 128 |
| H3 | aligned tree beats leaf-permuted tree on retrieval KL in ≥ 5/6 (width, seed) pairs | **supported** (6/6, also faster in 6/6) |
| H4 | r = 128 beats r = 16 on multi-match KL in 3/3, and median KL at r = 128 stays > 0.01 after converging | **partly supported (r = 128 still improving)**: 3/3 wins, median 0.056, but still falling 8–9% over the last 10,000 updates |
| H5 | staging of the left-half dictionary vs the loss drop (descriptive) | 3 "dict first" (all r = 128), 6 "together", 3 "loss first", 0 "no gain" |

All 48 planned runs finished and none collapsed. The correctness checks passed 47 of 47. The worker ran 6.6 h on an A40 at $0.59/h, about $3.87 of the $12 budget.

## The evidence

**Figure 1** ([figures/width.pdf](figures/width.pdf)): final test KL (nats, log scale) on 50,000 test examples, one dot per seed (n = 3 per point). Chain dots are placed at the tree width whose parameter count they match. The dotted line is the 0.01-nat threshold of the width rule. No run collapsed.

Final test KL in nats, min–max over seeds 0–2 (Bayes test accuracy is 0.956 for retrieval and 0.564–0.567 for multi-match):

| arm | params | retrieval KL | multi-match KL |
|---|---|---|---|
| tree r = 16 | 61,696 | 0.277–0.339 | 0.651–0.663 |
| tree r = 32 | 471,552 | 0.032–0.034 | 0.299–0.305 |
| tree r = 64 | 3.71 M | 0.0039–0.0045 | 0.088–0.089 |
| tree r = 128 | 29.5 M | 0.0037–0.0037 | 0.055–0.056 |
| perm r = 16 | 61,696 | 0.663–0.938 | 0.754–0.766 |
| perm r = 64 | 3.71 M | 0.024–0.127 | 0.308–0.343 |
| chain width 39 (matches r = 16) | 60,021 | 0.068–0.073 | 0.302–0.312 |
| chain width 155 (matches r = 64) | 3.73 M | 0.0027–0.0028 | 0.020–0.024 |

The three seeds agree closely. The exception is the permuted tree, whose seeds also differ in which leaf permutation they use.

### H1: retrieval width

r = 128 beats r = 16 in all three seeds (0.0037 against 0.28–0.34 nats). The width rule takes the smallest r where every seed at r and 2r finished without collapsing, the median KL is below 0.01, and doubling gains less than 0.002 nats. That's r = 64: median 0.0041 against 0.0037 at r = 128, a gain of 0.00035.

The rule calls r = 64 and r = 128 converged, but only because KL is so small. Over the last 10,000 updates their diagnostic KL fell 23–28%, from about 0.005 to 0.004, which is under the 0.002-nat clause and over the 5% relative clause (`figures/curves.pdf` shows them still dropping on a log scale). So "r = 64 reaches 0.004" is a statement about this 60,000-update budget, not an asymptote. It also means r = 128 may have more to gain than r = 64. The r = 128 pilot chose 3e-5, the bottom of its grid, so a lower LR is untested there.

### H2: does a target pair that meets the query only at the root cost more?

**Figure 2** ([figures/strata.pdf](figures/strata.pdf)): retrieval test accuracy by the index of the queried pair, seed median (n = 3), with the Bayes accuracy dashed. Pairs 0–3 meet the query only at the root, pairs 4–5 at level 3, and pair 6 at level 2. The r = 32, 64 and 128 lines lie on top of each other: at those widths the argmax matches the Bayes argmax on the test set (except one example in each of two r = 32 seeds), so the seed-median per-pair accuracies are identical.

At r = 16 every seed learned (near accuracy ≥ 0.8 × Bayes), and every seed is graded near > mid > far:

| seed | far (pairs 0–3) | mid (pairs 4–5) | near (pair 6) | near − mid | mid − far, r = 16 | mid − far, r = 128 |
|---|---|---|---|---|---|---|
| 0 | 0.849 | 0.923 | 0.951 | 0.028 | 0.075 | 0.003 |
| 1 | 0.882 | 0.926 | 0.952 | 0.027 | 0.043 | −0.004 |
| 2 | 0.860 | 0.920 | 0.951 | 0.031 | 0.060 | 0.004 |

The capacity hypothesis predicted mid within 0.02 of near and both more than 0.05 above far. It fails in all three seeds: near − mid is 0.027–0.031, and in seed 1 mid is only 0.043 above far. So the pre-set criterion is not met. Width does close the gap: at r = 128 the mid − far difference is within ±0.004 in all three seeds, below half its r = 16 value.

One exploratory observation, made after looking: the per-pair medians at r = 16 are 0.855, 0.858, 0.865 and 0.863 for pairs 0–3, then 0.921 and 0.926 for pairs 4–5, and 0.951 for pair 6. The big step is between the root-only pairs and the rest (about 0.06). The step from mid to near is about half that size. This is closer to "the left half is the bottleneck, plus some distance cost" than to a smooth distance gradient. But the criterion was fixed in advance, and this reading wasn't.

### H3: alignment

The aligned tree has lower final KL than the leaf-permuted tree in 6 of 6 (width, seed) pairs, and reaches 90% of Bayes accuracy sooner in 6 of 6. At r = 16 the permuted tree never reached 0.9 × Bayes accuracy in any seed (accuracy 0.53–0.65), while the aligned tree got there between update 15,210 and 22,000 (final accuracy 0.88–0.90). At r = 64 the gap is 0.004 against 0.024–0.127 nats. The same holds for multi-match (descriptive): 6 of 6.

### H4: multi-match width

r = 128 beats r = 16 in all three seeds (0.055–0.056 against 0.651–0.663 nats), and the median at r = 128, 0.056, is above 0.01. The r = 128 runs were still improving: diagnostic KL fell 8–9% (about 0.005 nats) over the last 10,000 updates, which fails the convergence rule. So the verdict is "partly supported (r = 128 still improving)", and the width rule's answer is "unresolved". The gap that remains at r = 128 may be undertraining, not a capacity limit. The parameter-matched chain at width 155 reaches 0.020–0.024 with 3.73 M parameters, below the 29.5 M-parameter r = 128 tree, which shows this process can be approximated much better than the tree manages here.

KL by number of matches in the input (seed median) reverses with width. At r = 16 it is 1.03 nats with one match, 0.70 with two and 0.46 with three or more. At r = 128 it is 0.032, 0.042 and 0.077. A narrow tree does worst when a single exact match must be found. A wide one does worst when it has to tally several matches.

### H5: staging of association storage

**Figure 3** ([figures/staging.pdf](figures/staging.pdf)): for each retrieval tree run, the held-out ridge R² of the left half's adjacent-pair table, decoded from node 3_0's bond (solid), and the diagnostic loss as a fraction of its initial value (dashed), against update. One line per seed. Vertical lines mark where R² reaches half its final gain.

A run is "dict first" if its R² reached half its final gain at an earlier checkpoint interval than the loss fell halfway, and "together" if both happened in the same interval:

| width | checkpoints | dict first | together | loss first |
|---|---|---|---|---|
| r = 16 | 27 (linear 500–8,000) | 0 | 2 | 1 |
| r = 32 | 27 (linear 500–8,000) | 0 | 1 | 2 |
| r = 64 | 15 | 0 | 3 | 0 |
| r = 128 | 15 | 3 | 0 | 0 |

At the widths with fine checkpoints (r ≤ 32), storage never leads: it rises with the loss drop or after it. At r = 128 it leads in every seed: half-gain at update 273–450, against a loss half-drop at 630–710, which is one checkpoint interval in seeds 1 and 2 and two in seed 0. With only 15 checkpoints at r ≥ 64, "together" at r = 64 means the same interval (803 to 1,500), and the r = 128 lead is one to two coarse intervals wide.

The decoded table is real, and at large width it is causal:

- **Decodability.** Final R² at node 3_0 is 0.20–0.21 (r = 16), 0.44 (r = 32), 0.88–0.90 (r = 64) and 0.97 (r = 128). The baseline probe, which reads the span's tokens additively, gets 0.168, and shuffled targets give −0.003 to −0.05. At r = 16 the bond holds only a little more than the additive baseline, which fits r = 16 being below the sampled rank of 29 at this node.
- **Patching.** A donor's bond is patched into a recipient only within the probe's top subspace (8, 16, 32 and 53 directions at r = 16, 32, 64, 128). Among the 84 of 196 valid donor–recipient pairs whose counterfactual answer differs from the recipient's, the prediction moves to the counterfactual answer in 98–100% of pairs at r = 128. Random subspaces of the same size give 25–28%, and the whole bond gives 100%. At r = 16 the subspace gives 37–42% against 25–36% random and 87–88% whole bond. The subspace's advantage over random grows with width (r = 32: 39–49% against 30–33%; r = 64: 71–74% against 48–53%). Only at r = 128, where the probe uses 53 of 128 directions instead of the cap of half the bond, does it carry nearly the whole effect. Whole-bond patching equals running the spliced input, so its 100% at r ≥ 32 says nothing about the table in particular.

### Descriptive: tree against chain

At matched parameter counts the chain beats the tree on final KL in 3 of 3 seeds at the r = 16 match, for both processes: retrieval 0.068–0.073 against 0.28–0.34, and multi-match 0.30–0.31 against 0.65–0.66. At the r = 64 match, retrieval is a tie within the comparison margin (0.0027–0.0028 against 0.0039–0.0045), and the chain wins multi-match (0.020–0.024 against 0.088–0.089). At the r = 16 match the chain also reaches 90% of Bayes accuracy sooner, in 3 of 3 seeds for both processes. At the r = 64 match the tree is faster on retrieval (3 of 3), and the chain on multi-match (2 of 3, one tie). At matched size the chain is never worse on final KL. Both wide-chain comparisons rest on runs that hadn't plateaued, though: the width-155 chain's diagnostic KL fell 30–35% (retrieval) and 16–20% (multi-match) over its last 10,000 updates, more than the tree it is compared with.

## How it was done

The definition, with commands and full method, is [EXPERIMENT.md](EXPERIMENT.md). In brief:

- **Processes**, both with vocabulary V = 8 plus a blank and target noise ε = 0.05. *Retrieval* puts 7 key-value pairs with distinct keys on leaves (2m, 2m+1). Leaf 14 holds the key of one of the pairs, chosen uniformly, and the target is its value. *Multi-match* has iid tokens on leaves 0–13 and plants a query x_14 = x_i, with target x_{i+1}. The Bayes posterior votes over every accidental match.
- **Models**: an untied 16-leaf TreeTN at r = 16, 32, 64, 128; the same tree with a leaf permutation that separates every key from its value at level 1 (r = 16, 64); and a chain (MPS) with its width matched to the r = 16 and r = 64 trees' parameter counts. Identity initialization.
- **Training**: seeds 0–2, online batches of 256, Adam, 60,000 updates, warmup then a cosine schedule. The peak LR was piloted per (process, arm, width) on seed 9 over the first 12,000 updates of the schedule, and the lowest diagnostic loss won. Each (process, seed) shares one data stream and test set across arms.
- **Metric**: KL from the Bayes posterior on a 50,000-example test set. A diagnostic set of 20,000 was logged through training.
- **Interpretation**: ridge probes, invariance and subspace patching at every checkpoint of the 36 tree and perm runs, on 6,000 examples per checkpoint.
- **Comparisons** use the project's rules (`studies/lang/analysis.py`): "better" on final test KL with ties within max(0.002, 10%).

Results are in `results/` (`analysis.json` holds every verdict and per-run row, `curves.json` the training curves, `interp.json` the probe trajectories), and the runs with their checkpoints and `interpret.json` files are in `runs/lang-induction/`.

## How it was checked

- **Correctness checks** (`results/checks.json`): all 47 passed before training. They cover Bayes posteriors against generation counts, closed-form Bayes losses, the locality and resampling of every summary the probes use, and patching the whole bond equals running the spliced input exactly.
- **Sampled ranks** (`results/rank.json`) reproduced the numbers the definition was built on: 29 at node 3_0 for retrieval, and up to 440–442 at level 2 for multi-match (nodes 2_1 and 2_2; 258 and 119 at 2_0 and 2_3).
- **Interpretation output**: interpretation finished in 18 minutes against its 2.67 h budget (the definition's 3 h, capped by the amendment), so I opened an `interpret.json` before trusting it. The files have probes at every internal node and checkpoint (27 at r ≤ 32, 15 at r ≥ 64), invariance at the five retrieval nodes where resampling is defined, and patching wherever at least 50 valid donor pairs exist, with the counts reported. Patching is skipped for the left-half dictionary in retrieval perm16 and perm64, seeds 0 and 1 (47–49 valid pairs).
- **Convergence**: I recomputed the KL drop over the last 10,000 updates from `curves.json` for every tree and wide chain run. That's where the "still falling 23–28%" for retrieval r = 64/128 comes from. The definition's rule passes them on the absolute clause, and the write-up says so instead of calling them converged.
- **Fact-check**: a separate fact-checking pass compared about 115 claims against `analysis.json`, `curves.json`, the run files and the logs. It found eleven that didn't fully hold, and all were corrected. The main ones: the r = 128 staging lead is one to two checkpoint intervals, not one. "Causally used" is now limited to r = 128. The patching result at r = 32 and 64 is added. The convergence caveat now covers every run at r ≥ 32 that was still improving.

## Not yet checked

- **A lower LR and a longer schedule at r ≥ 64.** Retrieval r = 128, retrieval chain 155 and multi-match r = 128 all chose 3e-5, the bottom of their grid, and all were still improving. This would decide whether multi-match keeps a gap above 0.01 at r = 128 (H4's second clause) and whether r = 64 and r = 128 truly differ on retrieval. It would take roughly 2–3 GPU-hours for the six r = 128 runs at 120,000 updates.
- **Finer checkpoints at r ≥ 64 for H5.** The only "dict first" runs are at r = 128, where the lead is one to two coarse checkpoint intervals. Rerunning the three r = 128 retrieval seeds with the linear 500–8,000 checkpoints would confirm or remove that lead. That's under an hour on one GPU.
- **Why the chain wins at small width.** A likely reason is that the chain carries the running state left to right and can bind each key to its value as it passes, so its bottleneck wouldn't be the left half's bond. Probing the chain's bond at the cut between leaf 7 and leaf 8 would test that. The interpretation code only supports trees today.

## Judgement calls

- **Three runs at a time instead of six.** With 6 concurrent processes, four of the 16 r = 64/128 retrieval pilots ran out of memory on the 44 GB A40 (the L40S and A6000 had no stock). The definition prescribes rerunning with `--procs 3`, and I did that. Finished pilots were reused. This changed nothing in the method except wall time.
- **Interpretation deadline tied to the worker's cap.** The pilots took 84 minutes in the rerun, after 3.5 minutes of setup and 13 minutes lost to the first attempt, against a plan of under an hour. So I capped interpretation to end by 08:45 UTC instead of a fixed 3 h, to leave time to publish before the 10 h cap. That gave it 2.67 h, and it needed only 18 minutes, so nothing was cut.
- Both are recorded under "Amendments" in [EXPERIMENT.md](EXPERIMENT.md).

## What this means

On unambiguous retrieval, a TreeTN needs a bond wide enough to pass up the left half's whole association table. At r = 16, below the sampled rank of 29, the left half's table is barely decodable, targets stored there are answered worst, and the permuted tree, which has to bind every pair higher up, does worse still. From r = 32 all strata sit at Bayes accuracy, from r = 64 the table is decodable at R² ≈ 0.9, and at r = 128 patching its probe subspace moves the prediction in 98–100% of pairs (71–74% at r = 64, against about 50% for random subspaces). That fits the capacity story in direction. The sharp mid ≈ near prediction didn't hold, though, and a chain of the same size learns the same task far better at small width. Multi-match is harder at every width and not settled at r = 128. The next step that would change the conclusions is the lower-LR, longer r ≥ 64 rerun.
