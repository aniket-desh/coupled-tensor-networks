# Supervisor's report: rhm-ttn-messages

The experiment ran to completion and its result is recorded as **not supported**: a tree tensor network trained on the Random Hierarchy Model does not implement the grammar's message passing, although its bonds are close to a function of what the task needs from each subtree. All 30 runs trained and were analysed, nothing was censored, and the worker cost about $5.26 over 7 h 18 m. The write-up is [README.md](README.md), the research note is [NOTE.md](NOTE.md), and the definition with its amendments and outcome is [EXPERIMENT.md](EXPERIMENT.md).

## What ran

All times are UTC, 14–15 November 2026. The worker was `6mpbypts00dn26`, an RTX PRO 4500 Blackwell in EU-CZ-1 at $0.72/h; the first GPU on the definition's list, the RTX A6000, was out of stock.

| time | step | outcome |
| --- | --- | --- |
| 21:40 | launch from the approved definition | worker ready after about a minute |
| 21:42–21:44 | checks | CUDA present, `studies.rhm.checks` 28 of 28, `msg_check.py` all 38 lines pass |
| 21:44–00:16 | training, four runs at a time | 30 of 30 runs with status `finished`, 2 h 32 m against an estimated 2–2.5 h |
| 00:16–04:57 | analysis, verdict and figures | 30 of 30 analysed, no failure, 4 h 41 m against an estimated 1.5–2.5 h |
| 04:58 | `orch finish` | results collected and merged, worker deleted |
| 05:00–06:05 | write-up, note, three fact-check passes | see below |

`orch ls` showed no worker of this experiment after 04:58.

## Result against each criterion

A cell is one grammar at one width with three seeds, and a cell supports a hypothesis only if all three seeds pass.

| criterion | result |
| --- | --- |
| Run not failed (checks, CUDA, at least 26 of 30 runs, shuffled-label R² ≤ 0.05) | met: 30 of 30 runs, shuffled-label R² at most 0.011 |
| Learned (excess loss < 0.02 nats/token in all seeds) | 9 of 10 cells; `v16m4s2L4` at r = 16 did not learn (0.043–0.057) and is not judged |
| H1, bonds hold the partially observed messages | not supported: 0 of 5 cells at r = 32, 0 of 4 at r = 16; per-cell median R² 0.08–0.75 against 0.8 |
| H2, a bond keeps what the task needs and little else | supported: 4 of 5 cells at r = 32, 3 of 4 at r = 16; ve 0.81–0.96 in the 25 passing runs, identity init 0.08–0.41 |
| H3, the cores reproduce the production rules | partly supported at r = 32 (2 of 5 cells), not supported at r = 16 (1 of 4) |
| H4, the merge is the grammar's update | not supported: 0 of 5 and 0 of 4 cells; per-cell median R² on the both-informative pairs 0.17–0.58 against 0.8 |

One exploratory result was added after the run and has no criterion: setting the invalid child pairs to zero in the saved read-out tables raises the merge's R² on Dirichlet pairs by 0.25–0.91 per cell at r = 32, which is 37–79% of the shortfall on those pairs (one node per level).

## What it cost

About $5.26 for 7 h 18 m of one worker (the experiment log's figures), inside the original $10 budget and 8 h cap. The user raised the ceiling to $15 and the cap to 10:00 UTC during the run; neither was needed.

## Decisions I made

1. **Raised the analysis from 2 to 6 runs at a time (00:42).** The first two analyses took about 25.7 minutes each against an estimated 5–10, which projected the last analysis past the 8 h cap. I recorded the amendment first, then sent `SIGUSR1` to the running `xargs`, so nothing was restarted. Four signals sent at once registered as one, so I added the rest one at a time until six analyses were running. The throughput rose by about 1.4 times. During the analysis `nvidia-smi` showed 99–100% GPU utilization with 2 and with 6 processes, and GPU memory stayed under 15 GB of 32, which is why I did not go higher than six.
2. **Left the analysis order as it ran.** The definition's command passes both width globs to one `ls`, which sorts them together, so the order was by grammar name with r = 16 first, and not r = 32 first as intended. Fixing it meant restarting and losing two nearly finished analyses, and it mattered only if some runs were left unanalysed. None were.
3. **Ran every command of the definition unchanged.** The only difference is quoting: I passed each multi-line step to `orch exec` through a shell variable.
4. **Recorded the overall result as not supported, not inconclusive.** The definition names H4 as the test that only the message algebra passes; H4 and H1 fail in every cell. H2 holds, and any code that identifies the extended message can pass it.
5. **Made the PNGs by re-running `msg_plot.py` on the coordinator.** I added an optional third argument to the script that also saves each figure as a 200-dpi PNG, and sent the re-rendered PDFs to the scratchpad, so the worker's 24 PDFs in `figures/` are exactly as produced. Only the five PNGs that the README and the note embed are in `figures/png/`.
6. **Added `msg_table.py`.** It prints every table of the README from `results/summary.json` and the analysis files, so the tables can be regenerated and checked.
7. **Added `msg_readout_ablation.py` and one exploratory table.** The fact-check of the note pointed out that the write-up inferred, without testing, that the invalid pairs' read-out mass explains H4's failure. The saved read-out tables allow a direct test in seconds on CPU, so I ran it on all 30 runs and reported it in both documents as exploratory.
8. **Applied the fact-checkers' corrections by rewriting the sentences.** Two passes changed 16 statements in the README and 13 in the note; a third, narrower pass checked the rewritten passages and the exploratory section, confirmed the new script and its numbers, and changed 11 more wordings. The corrections that change the reading are listed in the README under "How it was checked".
9. **Wrote the note from the research-note skill's `SKILL.md` and the write-technical-prose skill only**, as instructed, without reading the skill's reference files.

## Things that did not go as planned

- **The analysis took 4 h 41 m against an estimated 1.5–2.5 h.** A single analysis took about 20 to 80 minutes with six sharing the GPU. A future definition that reuses `msg_analyze.py` should budget for this or speed the probe fits up first.
- **I ran `orch log` once while checking what `orch finish` had merged.** It rebuilds `EXPERIMENTS.md`, which `orch finish` had already modified. I did not commit that file.
- **The note's mathematics was not rendered.** No Markdown or TeX renderer is installed on the coordinator. I checked that every `$` and `$$` delimiter and every brace is balanced and followed the conventions of the sibling note (`\lt` for the less-than sign, display blocks on their own lines), but I did not see the page on GitHub.

## What you need to do

- **Commit the experiment log.** `EXPERIMENTS.md` has this run's entry and result in the working tree, uncommitted, as agreed for the planning session. `orch finish` made its own commit of the collected results (`a1cdf5c`). My two commits (`aed0230` for the write-up, note, figures and scripts, and the one that adds this file) touch only `studies/rhm/rhm-ttn-messages/`. Nothing was pushed.
- **Look at `NOTE.md` on GitHub once**, to confirm the mathematics and the five embedded figures render.
- **Decide on the follow-ups**, all from the saved checkpoints in `runs/rhm-ttn-messages/` (226 MB, on the network volume, not in git): M2 with the random partner at the same number of observed leaves, which the H2 cells near 0.8 depend on; a decoder that is not linear in the bond; and the check, raised by the fact-check and not repeated by me, of whether training inputs already fix the cores' values on invalid pairs.
- **Fix the analysis command before reusing it**: list the r = 32 runs and the r = 16 runs with two separate `ls` calls so that r = 32 really goes first.
