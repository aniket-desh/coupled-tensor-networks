# rhm-dynamics: supervisor report

The experiment ran to completion inside its cap and budget, and the result is recorded as supported. The r = 16 TreeTN learns both RHM tasks to near-Bayes, H2a is partly supported and H2b is supported, with both width verdicts conditional on the shared LR. I added one amendment, a descriptive rerun of the r = 64 runs at a lower LR, after 3 of them collapsed at the shared LR. The write-up is [README.md](README.md), and the verdict lines are in the definition's Outcome.

## What ran

Everything ran on one worker, `aniket-ctn-rhm-dynamics` (RTX A6000, EU-SE-1, $0.69/h), from 2026-11-12 21:43:54 to 2026-11-13 02:21:27 UTC. Each command in the definition ran in order as its own `orch exec` step:

1. **GPU and CUDA checks.** Both passed.
2. **`studies.rhm.checks` and `studies.rhm.check_interpret`.** `checks` passed 26 of 26, and `check_interpret` passed all of its checks.
3. **Training (`run.py --hours 3`).** The LR pilots took 220 s. All 48 grid runs finished in about 45 minutes, well inside the 3 h training deadline, so nothing was skipped.
4. **`analyze.py` and `plot.py`.** These wrote `results/analysis.json`, `curves.json` and the four training figures.
5. **Interpretation (`analyze.py --interpret-only`, 3.5 h timeout).** It ran until the timeout (exit 124, recorded by the definition's `|| echo`). All identity-init r = 16 runs completed; 4 of the 6 r = 64 runs have partial `interpret.json` files.
6. **Final `plot.py` with the runs folder.** It wrote the four `interp_<task>_m<m>/` figure sets.
7. **Amendment: `run.py --only r64 --lr 1e-3 --out runs/rhm-dynamics-rerun`.** All 6 runs finished in about 15 minutes.

`orch finish` published 1032 data files (3.16 GB under `runs/`) and merged the experiment folder. `orch ls` afterwards showed no worker of mine.

## Result against each criterion

| criterion | verdict |
| --- | --- |
| H1 | supported: root test acc 1.0 on 3 of 3 grammars; last-token test KL 0.0036–0.0055 nats against the 0.05 bar |
| H2a | partly supported, conditional on the shared LR: 3 wins, 0 losses, 3 ties (the root pairs, both at zero error) |
| H2b | supported, conditional on the shared LR: 6 of 6 pairs; 3 of those r = 64 runs collapsed to chance after reaching threshold |
| r = 32 against r = 16 (descriptive) | faster in 6 of 6 |
| Inconclusive triggers | none: 48 of 48 finished, 6 usable pairs per comparison, the r = 16 reference learned |
| Variance init (descriptive) | at chance in 6 of 6 |

## Cost

The worker ran 4.63 h at $0.69/h, so it cost about $3.19, against the 8 h cap and the $9 budget. The fact-checking subagent was the only other paid call.

## Decisions I made

- **Rerun of all 6 r = 64 runs at peak LR 1e-3.** It is recorded under Amendments before it ran. At 3e-3, three r = 64 runs learned and then collapsed to the uniform output with finite loss. The definition's divergence rule covers only non-finite losses, so the collapse was outside it. I kept the pre-registered rules for H2b, so the collapsed runs count with their early threshold times, and I added the rerun as descriptive evidence that stays out of H2a and H2b. I chose 1e-3, the next lower grid LR, to mirror the definition's rule for diverged runs. It cost about 15 minutes of worker time (about $0.17).
- **Running the rerun after interpretation, not alongside it.** Sharing the GPU would have slowed the interpretation step, which has a hard timeout, and more runs would have been cut off.
- **A new script, `rerun.py`, in the experiment folder.** It reads only saved result files. It writes `results/rerun_r64.json` and `figures/rerun_r64.pdf`, and it records the checkpoints each run's interpretation covers. That status is listed in the definition's expected outputs, but `analyze.py` never wrote it, because the timeout ended the step before it could. I did not modify `analyze.py`, `plot.py` or anything outside the folder.
- **The overall result is "supported".** The main question (H1) is supported and H2b is supported. H2a is partly supported, and both width verdicts are conditional on the shared LR; the note given to `orch experiment done` and the Outcome say so.
- **Fact-check corrections.** The fact-checker found 11 claims in the first draft that did not fully hold, plus one omission. I applied all of them. The README's "How it was checked" section lists the main ones.

## What the user needs to do

- **Experiment log.** `EXPERIMENTS.md` was left for the planning session to update, as instructed.
- **Merge branch.** `orch finish` also left a branch, `results/20261112-214354-rhm-dynamics-blghb2am5xkc6n`. Its contents are already merged into `main` (commit 0febd4f), so you can delete it whenever you like.
- **Data under `runs/`.** The checkpoints and per-run results (3.16 GB) are in `runs/rhm-dynamics/` and `runs/rhm-dynamics-rerun/`, which are data folders and not in git.
- **Decisions that are yours:**
  - whether to test H2b at per-width tuned LRs, since the r = 16 choice sits at the top of the grid;
  - whether to measure how often r = 64 collapses with more seeds;
  - whether to run m = 8 'last' longer, since its loss was still falling at 40,000 updates.

  The sister experiments that reuse this training code and LR pilot (`rhm-controls`, `rhm-sample-efficiency`, `rhm-transfer`) should know that a 4,000-update pilot can miss a collapse that happens later in training.
- **Commits.** Nothing was pushed. My commits on `main` touch only `studies/rhm/rhm-dynamics`.
