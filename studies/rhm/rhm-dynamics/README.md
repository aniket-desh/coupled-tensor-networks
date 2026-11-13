# RHM learning dynamics: TreeTN across widths, grammars and seeds

A balanced binary tree tensor network (TreeTN: no biases, no nonlinearities, identity init) learns the Random Hierarchy Model by backprop. At bond dimension r = 16 it classifies the root symbol perfectly (test accuracy 1.0 on 3 of 3 grammars) and predicts the last token to within 0.006 nats of the Bayes-optimal predictor (median test KL 0.0040 over 9 runs, range 0.0036–0.0055, against a Bayes cross-entropy of 0.14–0.18 nats), so H1 is supported. Width mostly changes speed, not the end point. Widths r = 8, 16 and 32 end at Bayes accuracy in every run, all six r = 64 runs reached it before three of them collapsed, and wider networks cross the 0.9 × Bayes accuracy threshold earlier (median 600, 250, 150 and 128 updates for r = 8, 16, 32 and 64 on the root task). Both width hypotheses are conditional on a single shared learning rate, 3e-3, the top of the pilot grid. At that rate 3 of the 6 r = 64 runs learned and then collapsed to a uniform output. A rerun of all six r = 64 runs at 1e-3, added as an amendment, had no collapse (one run per grammar and task).

| criterion (fixed in advance) | verdict | evidence |
| --- | --- | --- |
| H1: r = 16 learns (root acc ≥ 0.99, last KL ≤ 0.05 nats) | supported | root acc 1.0, 1.0, 1.0; last KL 0.0040, 0.0036, 0.0055 nats (grammars 0, 1, 2) |
| H2a: r = 16 ends better than r = 8 in ≥ 5 of 6 pairs | partly supported, conditional on the shared LR | 3 wins, 0 losses, 3 ties; all three ties are root pairs where both widths have zero test error |
| H2b: r = 64 reaches threshold earlier than r = 16 in ≥ 5 of 6 pairs | supported, conditional on the shared LR | 6 of 6 pairs, 2 per grammar; 3 of those r = 64 runs later collapsed to chance |
| r = 32 against r = 16 (descriptive) | faster in 6 of 6 pairs | same rules as H2b |
| Variance init stays at chance (descriptive) | yes, 6 of 6 | test accuracy 0.062–0.077 (chance 0.0625), loss ≈ log 16 = 2.773 |
| Run health | 48 of 48 planned runs finished | none truncated, diverged or skipped |

Except in the m = 8 arm, the grammar has v = 16 symbols per level, n = 16 root classes, m = 4 productions per symbol, binary branching and depth 4, so each input is 16 tokens. The 'root' task maps all 16 tokens to the root symbol (Bayes: loss 0, accuracy 1). The 'last' task masks token 15 and predicts it from tokens 0–14 (Bayes cross-entropy 0.137–0.182 nats and accuracy 0.872–0.903, depending on the grammar).

## Width: r = 8 to 32 end at Bayes, wider is faster

![widths](figures/widths.pdf)

*[figures/widths.pdf](figures/widths.pdf): diagnostic loss, KL from Bayes and accuracy against updates, m = 4, init seed 0, one line per grammar (3 per width). Dashed lines are the Bayes references. The r = 64 lines that jump back to log 16 are the collapsed runs.*

Medians and ranges over runs (3 grammars, plus init seeds 1 and 2 at r = 16), from `results/analysis.json`. "Updates to threshold" is the first logged update at which diagnostic accuracy reaches 0.9 × Bayes accuracy, which means 0.9 for 'root' and about 0.80 for 'last'.

| task | r | runs | updates to threshold | test KL from Bayes (nats) | test error | ms/update (6 runs sharing the GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| root | 8 | 3 | 600 (550–600) | < 1e-7 | 0 | 6.3 |
| root | 16 | 9 | 250 (219–250) | < 1e-7 | 0 | 6.4 |
| root | 32 | 3 | 150 (128–150) | 1e-7 (1e-8 to 5e-6) | 0 | 6.6 |
| root | 64 | 3 | 128 (107–128) | 0.0004 (0 to 2.77) | 0.0001 (0 to 0.937) | 19.5 |
| last | 8 | 3 | 1150 (800–1200) | 0.0077 (0.0068–0.0108) | 0.106 (0.099–0.127) | 6.5 |
| last | 16 | 9 | 400 (313–449) | 0.0040 (0.0036–0.0055) | 0.107 (0.097–0.128) | 6.6 |
| last | 32 | 3 | 219 (200–250) | 0.0043 (0.0038–0.0055) | 0.108 (0.098–0.128) | 6.9 |
| last | 64 | 3 | 183 (150–200) | 2.62 (0.0066 to 2.64) | 0.926 (0.128 to 0.928) | 19.9 |

The Bayes error rate on 'last' is 0.097–0.128 depending on the grammar, so every width except the collapsed r = 64 runs ends within 0.002 of the Bayes error rate.

**H2a, r = 8 against r = 16.** The definition predicted that r = 8, below the 16 latent symbols per level, would be too narrow. That is not what happened in accuracy. r = 8 solves 'root' perfectly and matches the Bayes error on 'last'. The three wins for r = 16 are the three 'last' pairs. Two of them (grammars 0 and 2) come from the loss, after the error rates tie, and on grammar 1 r = 16's error rate is lower by 0.00104, just past the 0.001 tolerance. Test KL is about half that of r = 8 (0.0036–0.0055 against 0.0068–0.0108 nats). All three 'root' pairs tie at zero error and float32-zero loss. r = 8 is also 2–3 times slower to threshold. So r = 8 is enough to represent both tasks at this grammar size, and the cost of narrowness shows up as speed and a small residual KL.

**H2b, r = 64 against r = 16.** r = 64 crossed the threshold first in all 6 pairs, with margins well beyond the tie rule. On 'root' it took 107–128 updates against 219–250, and on 'last' 150–200 against 350–449. The tie margin, the larger of the logging interval and the r = 16 seed range, was 1–62 updates. The ordering is monotone in width. r = 32 also beat r = 16 in 6 of 6 pairs, and r = 8 was slowest everywhere. All widths share the same LR schedule, so the cumulative LR to threshold (in `analysis.json`) orders the runs exactly as the update counts do and adds no separate information.

## The r = 64 runs are unstable at the shared LR

![rerun](figures/rerun_r64.pdf)

*[figures/rerun_r64.pdf](figures/rerun_r64.pdf): the six r = 64 runs at the shared peak LR 3e-3 (solid) and at 1e-3 (dashed), one run per grammar and LR. The dotted line is log 16, the loss of a uniform output.*

At 3e-3, three r = 64 runs (last on grammars 0 and 1, root on grammar 2) reached near-Bayes diagnostic accuracy by update 3,000 and then fell to the uniform output (diagnostic loss 2.773). The two 'last' runs first collapsed around updates 3,900–5,500, recovered by update 6,000–8,000, and collapsed again at update 9,000. The root run collapsed at update 6,000. All three ended at chance, with test accuracy 0.06–0.07. A fourth run (root, grammar 1) touched log 16 at update 9,000 and recovered to test accuracy 0.9999. None of these losses became non-finite, so the definition's divergence rule did not apply. Under its comparison rules, all three collapsed runs still count as finished, and their early threshold crossings count toward H2b. The pilot's own r = 64 runs (grammar 0, seed 9) did not collapse within their 4,000 updates, although one grid run (last, grammar 1) collapsed at update 3,873, so a short pilot window is not the whole explanation. In the pilot the r = 64 runs clearly preferred 3e-4 (diagnostic loss 2e-6 against 0.13 at 3e-3 on root, 0.19 against 0.29 on last). The r = 8 root pilot preferred 1e-3 by only 1e-6 nats (0.0080293 against 0.0080304), so H2a's "conditional" label rests on a near-tie, while H2b's rests on a clear preference.

As a descriptive amendment that does not enter H2a or H2b, all six r = 64 runs were rerun once at peak LR 1e-3 with the same data stream. No run collapsed: the highest diagnostic loss after update 1,000 was 0.36 nats, root test accuracy was 1.0 on all three grammars, and last-token test KL was 0.0036, 0.0033 and 0.0044 nats (grammars 0, 1, 2). The rerun still crossed the threshold before r = 16 at 3e-3, in 128 updates on every 'root' grammar and 183–219 on 'last' (`results/rerun_r64.json`). The rerun also lowered the LR, which should slow learning, so it is the more conservative test of H2b's direction. That direction survives, but it rests on one rerun per grammar at a different LR, not on the pre-registered comparison.

## Init seeds, synonyms and the variance init

**Init and data-order spread at r = 16** ([figures/seeds.pdf](figures/seeds.pdf), 3 seeds × 3 grammars). The seed changes both the initialization and the online data stream. Threshold times varied by at most 31 updates within a 'root' grammar (219–250) and by at most 62 within a 'last' grammar (313–375 on grammar 0, 375–400 on grammar 1, 449 for every seed on grammar 2). Final last-token KL stayed within 0.0036–0.0055 nats across all 9 runs. The grammar matters more than the seed.

**m = 8, more synonyms** ([figures/m8.pdf](figures/m8.pdf), r = 16, 3 grammars). With 8 productions per symbol, 'root' was still solved (test accuracy 1.0), but the network took 1500–1650 updates to threshold against 219–250 at m = 4, about 6.5 times longer. 'Last' did not reach the threshold within 40,000 updates in any of the 3 grammars. Test accuracy ended at 0.43–0.44 against a Bayes accuracy of 0.54–0.57, and test KL from Bayes at 0.36–0.40 nats. The diagnostic loss was still falling at update 40,000, so this measures learning speed under a fixed 40,000-update schedule, not a plateau.

**Variance init** ([figures/init.pdf](figures/init.pdf), 6 runs, 5,000 updates). With the TreeTN's default variance init, all 6 runs stayed at chance on the GPU (test accuracy 0.062–0.077, final diagnostic loss 2.7726–2.7736 against log 16 = 2.7726). This matches the CPU probes before launch, where the loss also stayed at log 16. The identity init is what makes the TreeTN trainable here.

## Interpretation (descriptive, feeds `studies/rhm/interpret.py` analysis)

The interpretation step ran `python -m studies.rhm.interpret` on every run's checkpoints, identity-init r = 16 runs first, 4 at a time, under a 3.5 h timeout. 38 runs finished with all of their checkpoints, plus the 6 variance-init runs, which have 13 checkpoints because they stop at update 5,000. The timeout cut off 4 of the 6 r = 64 runs (last g2: 13 of 15 checkpoints, root g0: 14, root g1: 7, root g2: 4); their `interpret.json` files are partial. Every identity-init r = 16 run is complete, and `plot_interp` made one figure set per (task, m) from them: [interp_root_m4](figures/interp_root_m4/), [interp_last_m4](figures/interp_last_m4/), [interp_root_m8](figures/interp_root_m8/) and [interp_last_m8](figures/interp_last_m8/). Because the timeout ended the step, it never wrote its status into `analysis.json`; `results/rerun_r64.json` lists the checkpoints each run covers.

The figures were looked at but not analyzed further, so what follows is a first reading. In `interp_root_m4/interp_training.pdf` (9 runs), held-out probes decode the latent symbols at tree levels 2 and 3 from the whitened bonds almost perfectly from about update 300. That happens together with the drop in root loss, and mean-ablating the top LDA subspace of a bond raises the loss more than ablating a random subspace of the same size. On 'last' (`interp_last_m4/interp_training.pdf`), level-2 decoding rises only slowly, to about 0.4 of its ceiling, until about update 1,500 and reaches the Bayes ceiling only by update 10,000, long after most of the loss drop. Level-3 decoding ends at about 0.6 of its ceiling. The definitions of these measures are in `studies/rhm/interpret.py`.

## How it was done

- **Model and data.** TreeTN from `ctn/tree.py` with untied cores in float32, built by `studies/rhm/common.py`, with the tree's leaf order matching the data tree. Data are online, 256 fresh trees per update, drawn on the GPU with a generator seeded by (grammar, init seed) only, so runs that differ in width see the same trees.
- **Training.** Cross-entropy, Adam, batch 256, 40,000 updates. The LR warms up linearly for 100 updates, then follows a cosine to 0.1 × peak.
- **LR pilot.** Grammar 0, init seed 9, 4,000 updates, peak LR in {1e-4, 3e-4, 1e-3, 3e-3}. The r = 16 pilot chose 3e-3 for both tasks, the top edge of the grid, and every run used it. Diagnostic loss at update 4,000 was 1.9e-4 (root) and 0.223 (last).
- **Evaluation.** A fixed 20,000-tree diagnostic set per grammar is evaluated at 60 log-spaced updates, every 50 updates up to 3,000, every 1,000 updates after that, and at every checkpoint. A separate 50,000-tree test set is evaluated at the end. KL is measured from the exact Bayes posterior.
- **Comparisons.** Only finished runs enter. A pair is a tie if the difference is within max(logging interval, the r = 16 seed range) for steps, and within max(1e-3, the r = 16 seed range) for the error rate and then the loss. A "supported" verdict needs wins in at least 2 grammars and losses in none.
- **Hardware.** One RTX A6000 (EU-SE-1), 6 training runs at a time.

The exact commands, grid order and rules are in [EXPERIMENT.md](EXPERIMENT.md). The per-run results, checkpoints and source snapshot are in `runs/rhm-dynamics/`, and the rerun is in `runs/rhm-dynamics-rerun/`; both are data folders, not in git.

## How it was checked

- `studies/rhm/checks.py` passed 26 of 26 checks before training (`results/checks.json`). They cover the Bayes posteriors against brute force on a tiny grammar, parsing and task masks, checkpoint round trips, a training step that lowers the loss, and frozen parameters staying bitwise fixed. `studies/rhm/check_interpret.py` also passed all of its checks: gauge invariance of the interpretation measures, probes on a synthetic separable bond, ablation masks, and the first-order coupling limit.
- The r = 64 collapse was found by reading the run logs after training, and confirmed in the per-run diagnostic curves before any write-up. It led to the rerun amendment and to the caveat on H2b. Without it, H2b would have read as clean support.
- I recomputed the H2a pair outcomes by hand from the per-run test errors, losses and seed ranges. For example, on 'last' grammar 0, r = 8 has the lower error rate (0.1061 against 0.1080), but the difference is within the tie tolerance (0.0019, the r = 16 seed range). r = 16 then wins on loss, 0.1539 against 0.1561 nats, beyond a tolerance of 0.001. The recomputation agreed with `analysis.json`.
- A fact-checking pass compared about 105 claims in this write-up against `results/`, the run files, the job logs and the definition. It found 11 that did not fully hold. The main ones: the headline KL bound (first written as "within 0.004" although the worst run is 0.0055), "every width reaches Bayes" (three r = 64 runs end at chance), the H2a wins (one of the three is decided on error rate, not loss), the timing of the r = 64 collapses (two collapsed twice, one within 4,000 updates), and an unsupported explanation of the grammar-to-grammar spread. It also flagged that the r = 8 pilot preference is a near-tie. All of these are corrected above.

## Not yet checked

- **H2b at per-width tuned LRs.** The pre-registered comparison holds the LR fixed at the r = 16 choice, which sits at the top of the grid, so r = 16 might do better at a higher LR, and r = 64 is known to do worse there. A pilot over a wider grid, followed by a comparison at each width's best LR, would show whether "wider is faster" survives tuning. It would take about 1 GPU-hour, roughly $0.70 on an A6000.
- **How often r = 64 collapses.** It happened in 3 of 6 runs at 3e-3 and 0 of 6 at 1e-3, one seed per grammar. Init seeds 1 and 2 at r = 64 would give a collapse rate and show whether 1e-3 is reliably safe. That is 12 runs, about 30 minutes.
- **m = 8 'last' with a longer schedule.** The loss was still falling at 40,000 updates, so whether the TreeTN reaches the Bayes predictor at m = 8 is open.
- **Interpretation of the r = 64 runs.** 4 of the 6 have only partial interpretation files.

## Judgement calls

- **Counting the collapsed r = 64 runs in H2b.** The definition's rules treat them as finished runs with finite threshold times, so H2b counts their early crossings. An alternative would drop runs that ended at chance, but that rule wasn't fixed in advance and would have reduced H2b to 3 usable pairs, making it inconclusive (fewer than 4 pairs). I kept the pre-registered rule and stated the caveat next to the verdict.
- **The rerun LR.** I chose 1e-3, the next grid value below the shared LR, which mirrors the definition's own rule for diverged runs. 3e-4, the LR the r = 64 pilot preferred, was the other option and would probably have been slower. The rerun shows that a modest LR reduction removes the collapse, not that 1e-3 is the best LR for r = 64.

## What this means

The TreeTN with identity init is a working model of the RHM, so the checkpoints here are a reasonable basis for the interpretation analysis and for the follow-up experiments that reuse this training code. At this grammar size width mostly buys speed: r = 8 matches the Bayes error rate on both tasks, though its last-token KL is about twice that of r = 16. The r = 64 instability means later experiments that use wide trees should pilot over the full training length, or at least past update 10,000, rather than 4,000 updates, or use a lower LR for wide trees.

Cost: one RTX A6000 for 4.63 h at $0.69/h, about $3.19, against an 8 h cap and a $9 budget. That covers setup, pilots and training (about 50 minutes), interpretation (3.5 h) and the rerun (about 15 min).
