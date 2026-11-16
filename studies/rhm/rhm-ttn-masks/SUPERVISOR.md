# Supervisor's report: rhm-ttn-masks

The experiment ran to completion and its result is recorded as **inconclusive**, the verdict the definition's rule gives for its primary hypothesis: a tree trained on mixed masks passes the earlier experiment's test of the merge (H4) in 0 of 4 grammars, and it learned the varied task in only 2 grammars, where the definition needs 3 before a failed test counts as "not supported". The secondary hypothesis is not supported. All 36 runs trained and were analysed, nothing was censored, no amendment was needed, and the worker cost about $3.21 over 5 h 26 m. The write-up is [README.md](README.md) and the research note is [NOTE.md](NOTE.md).

## What ran

All times are UTC, 15–16 November 2026. The worker was `yliwtwndt6diaa`, an NVIDIA A40 in EU-SE-1 at $0.59/h, the third GPU type on the definition's list.

| time | step | outcome |
| --- | --- | --- |
| 19:17 | launch from the approved definition | worker ready at 19:19 |
| 19:19–19:21 | checks | CUDA present, `studies.rhm.checks` 28 of 28, `msg_check.py` 38 lines all `PASS`, `mask_run.py --check` passed |
| 19:21–23:29 | training, four runs at a time, seed by seed | 36 of 36 runs with status `finished`, no rerun; 4 h 08 m against an estimated 2.7 h |
| 23:29–00:16 | analysis, summary and figures of the `prefix` and `mixed` arms | 24 of 24 analysed, `run failed: no`; 47 minutes |
| 00:16–00:40 | analysis of the `varied` arm, three-arm summary and figures | 12 of 12 analysed, `run failed: no`; 24 minutes |
| 00:43 | `orch finish` | 216 data files and 154 files of the folder collected and merged, worker deleted |
| 00:44–01:10 | tables, PNGs, write-up, note, fact check, `orch experiment done`, commit | see below |

`orch ls` showed no worker of this project after 00:43.

## Result against each criterion

| criterion | result |
| --- | --- |
| Run not failed (checks, at least 31 of 36 trained, at least 86% analysed, prefix arm within 0.005 nats/token of `rhm-ttn-messages`) | met: 36 of 36 trained and analysed; largest difference in prefix excess loss 1.1e-8 |
| Prefix gate (prefix excess loss < 0.02 in all three seeds) | 4 of 4 grammars in `prefix` and `mixed`; 3 of 4 in `varied` (`v16m4s2L4` at 0.022–0.024) |
| H_A supported (`mixed` passes H4 in 3 of 4 grammars, `prefix` does not) | no: `mixed` 0 of 4 (0 of 12 runs), `prefix` 0 of 4 |
| H_A partly supported (H4 in 1 or 2 grammars, or the gap over `prefix` closed in all seeds of 3 grammars) | no: the gap is closed in 0 grammars, in 1 run of 12 |
| H_A not supported (neither, and `mixed` learned the varied task in 3 grammars) | no: the varied task was learned in 2 grammars (`v3m2s2L4`, `v8m2s2L4`); `v8m4s2L4` ends at 0.025–0.030 and `v16m4s2L4` at 0.090–0.098 |
| **H_A** | **inconclusive** |
| **H_B** (invalid read-out mass lower in `mixed` in 10 of 12 pairs, median paired ratio at most 0.5) | **not supported**: lower in 6 of 12, median ratio 1.10, range 0.53 to 1.74 |

The both-informative R² that H4 is judged on is 0.22–0.78 over the 12 `mixed` runs and 0.18–0.55 over the 12 `prefix` runs, against a bar of 0.8.

## Cost

About $3.21 for 5 h 26 m of one A40, against a $10 budget, a 10-hour cap and an estimate of $4.30 to $5.00 on a faster and dearer GPU. Training took 1.5 times the estimated time and the analysis about a third of it.

## Decisions I made

1. **Let the launch fall through to the A40.** The definition lists three GPU types and says to try the next when one is out of stock. The launcher's output was:

   ```
   rp-up: NVIDIA RTX PRO 4500 Blackwell: not in stock outside US-NC-1
   rp-up: NVIDIA A40: not in stock outside US-NC-1
   rp-up: NVIDIA RTX A6000: 400 There are no longer any instances available with the requested specifications. Please refresh and try again.
   rp-up: nothing placed; retrying in 30s
   (the A6000 line and the retry twice more)
   rp-up: created yliwtwndt6diaa (aniket-ctn-rhm-ttn-masks, NVIDIA A40 in EU-SE-1), $0.59/hr, worst case $5.90
   ```

   The A40 was reported out of stock on the first pass and placed on the fourth. All of the definition's timings were measured on the RTX PRO 4500, so I applied the 4-hour margin it sets for the other GPUs.
2. **Kept four training runs at a time although the A40 was slower.** Each run took 21 to 29 minutes against the estimated 18, with the GPU at 100% utilization and 7.8 GB of 46 GB in use. Changing the parallelism is a change to the commands that the definition allows only for an out-of-memory failure, and the slower pace still left room under the cap.
3. **Started the `varied` arm's analysis.** The first analysis step ended at 00:16 with 5.0 hours left before the cap, above the 4 hours the definition requires on this GPU.
4. **Finished the worker as soon as the second summary printed**, after confirming 36 `result.json` files with status `finished` and that `runs` comes back as a data folder, since the follow-up the definition fixes needs the checkpoints.
5. **Regenerated `figures/summary.pdf` and `figures/training.pdf` on the coordinator from the saved results.** The worker's `training.pdf` had its two row labels overlapping and no gate line in the varied row. I changed `mask_plot.py` (labels on two lines, the 0.02 line in both rows, an optional fourth argument that also writes 200-dpi PNGs) and re-ran it. `summary.pdf` has the same content as the worker's. The figures under `figures/<arm>/` are the worker's, untouched; the two production-table PNGs in `figures/png/` were rendered with `msg_plot.py`'s existing PNG option, its PDFs going to a scratch folder.
6. **Added `mask_table.py`**, which prints the README's tables from the saved results, so the tables can be regenerated. It recomputes nothing except `mask_summary.py`'s gap rule and per-seed differences.
7. **Recorded "inconclusive" although H4 fails in every run.** That is the definition's rule, and the write-up says in its first paragraph that the numbers lean negative. I did not change a criterion.
8. **Reported two things the definition did not ask for, marked as found after the run:** the gain in both-informative R² is in the two grammars where the varied task was not learned to the gate, and the left-masked pairs stay far below their ceiling in every arm.
9. **Reported that the prefix arm's analysis differs from `rhm-ttn-messages` by up to 0.027 in both-informative R² and 0.11 in the invalid read-out ratio**, on runs whose prefix loss agrees to 1.1e-8. The run-failed criterion covers only the loss, so this is not a failure; I did not investigate the cause.
10. **One fact-check pass.** The `fact-checker` agent recomputed the tables from the per-run files (all held) and returned 11 items in the prose of the two documents, all of which I corrected. The most consequential were a sentence saying the merge "does not improve" in the two grammars where the varied task was learned (it is unchanged in one and changes with both signs in the other) and one saying the gain stops at the additive ceiling (true of `v16m4s2L4` only). I did not run a second pass on the corrected text.

No amendment was recorded, because nothing in the method, the criteria or the budget changed.

## What you need to do

- **Commit the experiment log.** `orch` wrote this run's entry and its result into `EXPERIMENTS.md`, which I left uncommitted as instructed. `orch.toml` and `.vscode/` also show as changed or untracked; neither is mine.
- **Nothing is pushed.** `orch finish` made commit `1b7ada5` (the worker's results) and I made `77316fd` and the commit that adds this report, all on `main`, all touching only `studies/rhm/rhm-ttn-masks`.
- **Decide on the follow-up the definition fixed:** the projective probe from a bond to its normalized message on bonds from varied-pattern inputs. It needs a new analysis script and the checkpoints only, which are in `runs/rhm-ttn-masks/` on the coordinator (274 MB, git-ignored) and in `runs.tar` under `/workspace-global/results/20261115-191730-rhm-ttn-masks-yliwtwndt6diaa/pub-001/`.
- **Decide whether to rerun the two grammars with 4 productions per symbol with a longer schedule or a wider tree.** The primary hypothesis cannot return "not supported" until the `mixed` arm meets the varied gate in a third grammar, and `v8m4s2L4` missed it by 0.005 to 0.010 nats/token.
- **Read the corrected sentences once.** The fact check ran on the first draft; the corrections were applied by me and not re-checked by a second pass.
