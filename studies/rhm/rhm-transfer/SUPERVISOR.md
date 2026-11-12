# rhm-transfer: supervisor report

The experiment ran to completion as defined, with no failures and no amendments. H1 is supported in both variants, and the worker is deleted. The write-up is [README.md](README.md) and the verdict is in the definition's Outcome.

## What ran

Worker `vqmjc0z8futms1` was one NVIDIA A40 in CA-MTL-1 at $0.59/h, under job `20261112-214407-rhm-transfer-vqmjc0z8futms1`. The RTX A6000 create call was refused for lack of capacity and the L40S was out of stock outside US-NC-1, so the launcher took the A40, the second type on the definition's list. The definition's six commands each ran as their own `orch exec` step, in order, all with exit code 0:

1. `nvidia-smi` and the CUDA assertion.
2. `studies.rhm.checks`: 26/26 passed.
3. `run.py --hours 3`: 22 min. The LR pilot picked 3e-3, at the top edge of the grid. All 9 units passed the pretraining gate (seen accuracy 1.0, other-grammar 1.0), and all 81 phase-2 arms finished.
4. `analyze.py`: 9 units passed, 0 failed, 0 missing. Both H1 variants came out supported.
5. `plot.py`: wrote `figures/phase2.pdf` and `figures/pretrain.pdf`.

`orch finish` published 357 data files to `runs/rhm-transfer/` (git-ignored, 71 MB, pub-001), merged the folder, and deleted the worker. `orch ls` then showed no rhm-transfer worker.

## Result against each criterion

- **Scratch must learn: met.** Its median final unseen accuracy is 1.0.
- **H1, C variant: supported.** `frozen_learned_C` reached 0.9 at median 42 updates [42, 50]. Scratch first logged 0.9 at update 148 in every unit, having crossed after 126. That is 9 of 9 pairs, sign test p = 0.002, 3–0 in every grammar. The ratio to the oracle (36 [31, 36]) is 1.17, within the 2× limit.
- **H1, WC variant: supported.** It took 50 [50, 58] and won 9 of 9 pairs (p = 0.002, 3–0 per grammar). The ratio to the oracle is 1.39.
- **Other-grammar floor (descriptive): censored, as expected.** Neither variant reached 0.9 in any unit.
- **Descriptive.** Random frozen subtrees never reached 0.9. Finetune took 75 updates, faster than scratch and slower than `frozen_learned_C` in 9 of 9 pairs each, and kept seen accuracy at 0.994.

## Cost

The worker ran from 21:44 to 22:10 UTC, 27 min, for about $0.26. The cap was 4 h and the budget $4.50. No API spend beyond this session and its one fact-checker agent.

## Decisions I made

- **Logging resolution.** Scratch's time to threshold was 148 in all 9 units. I checked the curves and confirmed this is the logging grid (logged steps 126 and 148), not a bug. The README states the crossing as being in (126, 148].
- **Oracle plateau.** The oracle's accuracy is flat between steps 25 and 27 in some units. I judged this expected, since its prediction depends only on the symbol pair, and said so in the README.
- **Log copy.** The coordinator's copy of the training log was missing the last two arms at exit. I confirmed on the worker, through `status.json` and the arm folders, that all 81 arms had finished before running the analysis.
- **Fact-check.** The fact-checker found eight overstatements, all of them grid-quantized numbers or "at chance" wording, and one omission. I applied every correction. The omission: `studies/rhm/analysis.py` computes the margin's logging-interval term as the gap to the next logged step (2 updates after 148), not the 22-update interval before it. The definition's wording allows either reading, and with a 22-update margin every pair is still a win. I documented this in the README and did not change the analysis.

## What you need to do

- **Nothing is required.** The experiment log `EXPERIMENTS.md` was updated by `orch experiment done`, but I left it uncommitted along with the other sessions' changes, as instructed.
- **Optional follow-up.** H1 rests on one LR, 3e-3, at the grid's top edge. A phase-2 rerun at 1e-2 and 1e-3 would cost under $1 and would show whether the 3.5-fold speed-up depends on it.
- **Your call on the margin.** You may want to decide which reading of the margin in `studies/rhm/analysis.py` is intended for the other RHM experiments that use it.
