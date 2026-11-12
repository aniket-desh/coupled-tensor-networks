# RHM feature reuse: pretrained lower subtrees transfer to unseen root productions

The test was whether a TreeTN's lower subtrees, pretrained on 48 of the 64 root productions of a Random Hierarchy Model grammar, carry level-1 symbol codes that a fresh readout can reuse on the 16 held-out productions. They do, and the pre-registered hypothesis H1 is supported in both of its variants. With the subtrees frozen and only a fresh root core trained, unseen-production accuracy reached 0.9 after a median of 42 updates (range 42 to 50, n = 9 units). Training the whole network from scratch first logged 0.9 at update 148 in every unit (it crosses somewhere after update 126), so the frozen arm was faster in 9 of 9 pairs (sign test p = 0.002, 3 of 3 in each grammar). An oracle that is handed the true level-1 symbols took 36 (31 to 36). Two caveats bound the claim. Unseen productions are new pairs of side symbols that pretraining already saw, so the result shows that per-side symbol codes can be reused, not that the network recombines conjunctions it has never seen. And every threshold crossing happens within the first 150 updates, where logged steps at the crossings are 4 to 22 updates apart.

## Results

[Phase-2 accuracy curves: figures/phase2.pdf](figures/phase2.pdf)

`figures/phase2.pdf`: accuracy on the diagnostic set of unseen productions (left, the productions trained on in phase 2) and of seen productions (right) against phase-2 updates, log scale. There is one line per unit (3 grammars × 3 init seeds = 9 units per arm). Solid lines are C variants and dashed lines WC variants; the dotted line is the 0.9 threshold. Chance is 1/16 = 0.0625.

The arms, all from the same init and pretraining within a unit, and all trained in phase 2 on the same stream of unseen-production trees:

- `scratch`: the whole network trained from the init.
- `finetune`: the whole network trained from the pretrained state.
- `frozen_learned_C`: leaves and fusion levels 1 to 3 frozen at the pretrained values, with a fresh root core C. `_WC` freezes only levels up to 2 and also trains fresh level-3 cores.
- `frozen_other`: the same freezing, with subtrees pretrained on a different grammar (100 + g). This is the trained floor.
- `frozen_random`: the same freezing, at the untrained init.
- `oracle_C`: a fresh C on a fixed dense code √r·Q·e_a of the true level-1 symbols, read off the parse. This is the ceiling.

Steps to 0.9 unseen accuracy, final accuracies after 10,000 phase-2 updates, and speed, as median [min, max] over 9 units (from `results/analysis.json`):

| arm | trainable params | updates to 0.9 | cumulative LR to 0.9 | final unseen acc | final seen acc | ms/update |
|---|---|---|---|---|---|---|
| scratch | 65,536 | 148 [148, 148] | 0.30 | 1.00 | 0.060 [0.044, 0.077] | 10.6 |
| finetune | 65,536 | 75 [68, 75] | 0.086 [0.070, 0.086] | 1.00 | 0.994 [0.985, 0.998] | 10.8 |
| frozen_learned_C | 4,096 | 42 [42, 50] | 0.027 [0.027, 0.038] | 1.00 | 0.070 [0.028, 0.092] | 3.1 |
| frozen_other_C | 4,096 | never (9 of 9) | | 0.103 [0.089, 0.119] | 0.062 | 3.4 |
| frozen_random_C | 4,096 | never (9 of 9) | | 0.157 [0.141, 0.173] | 0.063 | 3.1 |
| oracle_C | 4,096 | 36 [31, 36] | 0.020 [0.015, 0.020] | 1.00 | 0.081 [0.039, 0.105] | 4.8 |
| frozen_learned_WC | 12,288 | 50 [50, 58] | 0.038 [0.038, 0.051] | 1.00 | 0.067 [0.040, 0.075] | 6.1 |
| frozen_other_WC | 12,288 | never (9 of 9) | | 0.189 [0.158, 0.229] | 0.064 | 5.8 |
| frozen_random_WC | 12,288 | never (9 of 9) | | 0.253 [0.233, 0.305] | 0.065 | 5.8 |

### Against the pre-registered criteria

All 9 units passed the pretraining gate (seen accuracy 1.0 in every unit, against a gate of 0.95). The other-grammar pretraining also reached 1.0 on its own grammar in every unit. All 81 phase-2 arms finished their 10,000 updates, so no pair was dropped.

- **Scratch must learn: met.** The median final unseen accuracy for scratch is 1.0 and its final unseen loss is 8e-7 nats, far from log 16.
- **H1, C variant: supported.** `frozen_learned_C` was faster than scratch in 9 of 9 pairs with no ties (one-sided sign test p = 0.002), and won 3 to 0 in each grammar. The margin was 8 updates against gaps of 98 to 106. Its median of 42 updates is 1.17 times the oracle's 36, within the factor of 2.
- **H1, WC variant: supported.** It was faster than scratch in 9 of 9 pairs (p = 0.002, 3 to 0 per grammar). Its median of 50 is 1.39 times the oracle's 36.
- **Other-grammar floor (descriptive, expected censored): as expected.** Neither `frozen_other` variant reached 0.9 in any unit. Their final unseen accuracies were 0.10 (C) and 0.19 (WC).
- **Learned vs random (descriptive).** `frozen_random` never reached 0.9 either, finishing at 0.16 (C) and 0.25 (WC). This only shows that the learned subtrees beat the near-additive map of the tokens that an untrained identity-init subtree computes.
- **Finetune (descriptive).** It was faster than scratch in 9 of 9 pairs (75 against 148) and slower than `frozen_learned_C` in 9 of 9 (75 against 42). It barely forgot the seen productions: seen accuracy finished at 0.994 [0.985, 0.998] while it learned the unseen ones.

### What the numbers mean

The scratch time of exactly 148 in all 9 units comes from the logging grid, not from identical runs. The logged points near the threshold are 126 and 148. At update 126 scratch's unseen accuracy was between 0.82 and 0.90 in every unit, and at 148 it was between 0.93 and 0.97. So scratch crosses 0.9 somewhere in (126, 148], about three times later than the frozen learned arm, which was at 0.94 to 0.99 by update 50.

All learned-subtree and oracle crossings, and finetune's, fall inside the 100-update linear warmup. Scratch crosses just after it. Since the LR is still rising during the warmup, the cumulative LR to threshold separates the arms more than the update count does: at the logged crossings it is 0.027 for `frozen_learned_C` against 0.30 for scratch, about 11-fold (between roughly 8 and 15 once the logging grid is allowed for), compared with 3.5-fold in updates (3.0 to 4.1). An update of a frozen-C arm also costs about a third as much (3.1 against 10.6 ms on the A40), so the wall-clock gap is larger again.

Pretraining alone does not solve the unseen productions. During pretraining, accuracy on the held-out productions stayed at or below chance (0.02 to 0.11 over training and 0.02 to 0.08 at the end, against a chance level of 0.0625; `figures/pretrain.pdf`), so the pretrained root core does not classify the new pairs. The new readout has to be learned, and on frozen learned features it learns almost as fast as on the oracle code. The fresh-C arms stay near chance on the seen productions throughout phase 2 (final 0.03 to 0.10, with the oracle peaking at 0.13), far below finetune's 0.99, which is expected because their C never saw those productions. Finetune keeps them because it starts from the pretrained C.

Before the scale normalization, the level-3 bonds had RMS about 2.1 for learned subtrees, 0.82 to 0.91 for other-grammar subtrees and 0.36 to 0.38 for untrained ones (`rescale.rho` in each arm's `result.json`, units g*_s0). The normalization then gave every fresh core unit-RMS inputs (C in the C arms, and the level-3 cores in the WC arms, whose C then received RMS 0.95 to 1.05), so the arms differ in what the bonds code, not in their scale.

## How it was done

The grammar is the root task with v = n = 16, m = 4, s = 2 and L = 4, for grammars 0 to 2. The model is an untied TreeTN with r = 16, float32 and the identity init (eps 0.1). In each grammar, one of each class's 4 root productions is held out, giving 16 unseen and 48 seen. The split is redrawn until every level-1 symbol on a side of an unseen production also occurs on that side in a seen one (0 or 1 redraws were needed). Pretraining ran for 5,000 online updates of 256 fresh trees on seen productions. Phase 2 ran for 10,000 updates per arm on unseen productions, with Adam, a 100-update warmup and cosine decay to 0.1 × peak. The peak LR was 3e-3, chosen by the pilot. Accuracy was measured on fixed diagnostic sets of 20,000 seen and 20,000 unseen trees per grammar. The full method, comparison rules and commands are in [EXPERIMENT.md](EXPERIMENT.md).

It ran on one NVIDIA A40 (CA-MTL-1, $0.59/h; the L40S and RTX A6000 were out of stock) from 21:44 to 22:10 UTC on 2026-11-12. That was 27 minutes and about $0.26, against a 4 h cap and a $4.50 budget. Training took 22 minutes of its 3 h deadline.

Files: `results/checks.json` (correctness checks), `results/analysis.json` (per-unit results, pairs, margins, per-grammar counts, the table above), `results/curves.json`, `figures/phase2.pdf` and `figures/pretrain.pdf` (made by `plot.py` from the results). Per-arm logs, splits, rescale factors and checkpoints are in `runs/rhm-transfer/` on the coordinator (about 70 MB, git-ignored, published as pub-001 of job `20261112-214407-rhm-transfer-vqmjc0z8futms1`). The code is commit `22d3075`.

## How it was checked

- All 26 checks in `studies/rhm/checks.py` passed on the worker before training. They cover frozen cores staying bitwise unchanged under both freezings, the normalization giving unit-RMS bonds, the oracle's parse-based code being classifiable by a lookup C, the split and restricted sampler, and checkpoint round trips.
- The scratch time being 148 in every unit looked suspicious, so I read the logged curves. It is the grid quantization described above.
- In some units the oracle's accuracy stays flat between steps 25 and 27 while its loss keeps falling. This is expected: the oracle's prediction depends only on the pair of true symbols, so its accuracy moves in steps of one production's share.
- The log copy on the coordinator was missing the last two arms when the job exited. On the worker, `status.json` listed all 9 units as finished and each arm folder held its `result.json` and final checkpoint.
- A fact-check of this write-up against `results/`, `runs/rhm-transfer/`, the job logs and the experiment log found no claim that changes a verdict. It corrected eight sentences that stated grid-quantized step counts and "at chance" levels more precisely than the logs support, and it pointed out the margin definition noted under "Not yet checked".

## Not yet checked

- **One learning rate.** Every arm used the pilot's 3e-3, which is the top edge of the grid {1e-4, 3e-4, 1e-3, 3e-3}. A higher LR might speed scratch up more than the frozen arms, or the reverse. Rerunning phase 2 at 1e-2 and at 1e-3 would take under an hour on the same GPU for well under $1. The size of the 3.5-fold ratio depends on this, but the sign of H1 is unlikely to: scratch has to learn the subtrees as well as the readout.
- **Time resolution.** Steps to threshold are only as fine as the logging grid (4 to 8 updates between 31 and 58, and 22 just before 148). Dense logging over the first 200 updates would give exact crossings. The H1 verdicts don't depend on it: their logged gaps are 90 to 106 updates against margins of 2 to 8, and even at the grid's worst case scratch crosses at least 69 updates after the slowest learned arm. Note that `analysis.py`'s margin uses the gap from the threshold step to the next logged step (2 updates after 148), not the 22-update interval before it, which the definition's wording allows. With a 22-update margin every pair in every H1, finetune and floor comparison is still a win.
- **Recombination of unseen conjunctions.** This design can't test it, as stated in its scope. A split that holds out side symbols as well as pairs would be needed.

## Judgement calls

The pilot's choice of 3e-3 at the grid edge was kept as the definition specifies, and it is recorded in `results/analysis.json` (`lr.at_grid_edge`). No other choice was needed: no unit failed a gate, no arm crashed or was truncated, and there were no amendments.
