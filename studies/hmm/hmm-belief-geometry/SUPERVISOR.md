# Supervisor's report: hmm-belief-geometry

The experiment ran to completion and its result is recorded as **inconclusive**, which here stands for a mixed outcome under the pre-set rule: the transformer reproduction (H1) is supported, and the chain (H2) and both tree hypotheses (H3a, H3b) are each partly supported. All 72 runs trained and were analyzed, none failed or was censored, and every cell learned. The worker ran 8 h 57 min and cost about $6.18. The write-up is [`README.md`](README.md), the research note is [`NOTE.md`](NOTE.md), and both were fact-checked.

## What ran

All times are UTC, 14–15 November 2026. The worker was `e1dlkqg8emkuoj`, an RTX A6000 in EU-SE-1 at $0.69/h, the first GPU on the definition's list.

| time | step | outcome |
| --- | --- | --- |
| 23:44 | launch from the approved definition | worker ready after about a minute |
| 23:45–23:47 | checks | CUDA present, `studies.hmm.checks` 29 of 29, `hmm_check.py` 7 of 7 |
| 23:47–05:58 | training, four runs at a time | 72 of 72 runs with status `finished`, 6 h 11 min against an estimated 3–3.5 h |
| 00:09–04:29 | eight CPU analysis batches during training | the 54 runs of the transformer, chain and r = 8 tree arms analyzed |
| 05:59–08:38 | GPU analysis of the 18 r = 32 tree runs | 72 of 72 analyzed, no failure, 2 h 39 min against an estimated 1 h |
| 08:38–08:40 | verdict and figures | `results/summary.txt`, `results/summary.json`, nine PDFs |
| 08:40–08:41 | `orch finish` | results collected and merged, worker deleted |
| 08:42–09:25 | PNG figures, write-up, note, two fact-check passes, result recorded | see below |

`orch ls` reported no running worker for this project after 08:41.

## Result against each criterion

A cell is one process and one model with three seeds, and it supports a hypothesis only if every clause holds in all three seeds. Scores are R² on prefixes the probe never saw.

| criterion | result |
| --- | --- |
| Run not failed (checks, CUDA, at least 62 of 72 runs, shuffled-label R² ≤ 0.05) | met: 72 of 72 runs, shuffled-label R² between −0.011 and 0.005 |
| Learned (excess loss under the gate in all seeds) | 24 of 24 cells; largest excess 0.0044 nats/token against a gate of 0.02 |
| H1, transformer reproduction | supported, 2 of 2: Mess3 0.9997; RRXOR 0.865–0.872 against a next-token ceiling of 0.474 |
| H2, chain, projective probe | partly supported, 2 of 4: Mess3 0.9999 and RRXOR 0.994–0.997 pass; `factored` (0.844–0.845) and `coupled` (0.956–0.960) fail against the token readout (0.919, 0.928); coins not testable, control saturated |
| H3a, tree, first half | partly supported, 2 of 6: Mess3 passes at r = 32 and r = 8; `factored` and `coupled` fail at both |
| H3b, tree, second half, product of the top bonds | partly supported, 1 of 2: coins at r = 8 pass (0.991–0.994); RRXOR at r = 8 fails (0.733, 0.767, −4.55 against an additive ceiling of 0.772) |
| RRXOR cells reported on their own | H1 pass, H2 pass, H3b fail |
| H4, product processes (descriptive) | every model reads the factor marginals at 0.97 or more; only the transformer halves the token readout's error on the joint |
| Form of the chain's code (label) | projective on RRXOR; affine on Mess3, `factored` and `coupled`; mixed on coins |

## Cost

The worker ran from 23:44 to 08:41, 8 h 57 min at $0.69/h, about $6.18 by the toolkit's record. The definition's budget was $10 and its cap 8 h. The user raised the ceiling to $15 at 01:50 UTC, and the planning session extended the worker's cap to 14:00 UTC, so the run ended 5 h 19 min before the cap and $8.82 under the ceiling. No other paid service was used.

## Decisions I made

1. **CPU analysis one at a time on 3 threads, from the second batch on** (00:35). The worker's container had a quota of 7.65 CPU cores although `nproc` reported 96. The first batch's two 8-thread analyses, as the definition specified, left each training process about a third of a core, and training slowed from 41 ms to about 85 ms per update. I recorded the change under Amendments before the second batch. The first batch ran as written; four transformer runs trained more slowly during it (68.5 ms per update) with no effect on their results.
2. **No STOP at the 4-hour mark.** Training was running slower than estimated in every arm, and the definition's rule would have stopped it with 56 runs started. Before the user's message I had not changed the rule. After the message relayed at 01:50 UTC (budget $15, time-based cutoffs may move) I recorded an amendment that made the STOP conditional on the hard cap, and after the planning session confirmed the cap at 14:00 UTC I recorded that no STOP would be created. `orch ls` showed the new cap before I relied on it.
3. **CPU batches limited to eight runs, from the fifth batch on** (02:55), with `head -n 8` in the definition's command, so that a batch took about an hour and the GPU step could start when training ended. **I did not record this before acting**; it is in the Amendments now, marked as recorded afterwards. Its side effect is that the fifth, sixth and seventh batch steps ended with exit code 141: under `pipefail` the listing loop is killed by SIGPIPE once `head` has eight lines. The analyses of those batches ran to the end; all 72 analysis logs are complete and without errors. The experiment log shows those three steps as non-zero exits.
4. **The last command block ran as two steps**, the GPU analysis and then the summary and figures, so that I could confirm 72 analyses and no failure before the verdict was computed. The commands are unchanged.
5. **A preview of the summary at 04:30**, on a copy of the 54 results then analyzed, in a temporary folder on the worker that I removed. It wrote nothing into the experiment folder.
6. **Result label "inconclusive".** The tool accepts supported, not-supported, inconclusive or failed. With one "supported" and three "partly supported", "supported" would overstate H2 and H3, so I recorded inconclusive and put the per-hypothesis verdicts in the note. The planning session may prefer another label.
7. **PNG figures from the plot script.** `pdftoppm` is not installed, so I added `hmm_figures_png.py`, which calls the functions of `hmm_plot.py` on the saved results and writes the nine figures at 200 dpi to `figures/png/`. The md5 sums of the nine PDFs are the same before and after.
8. **The research note was written without the skill's reference files.** They are outside the working directories, and the planning session told me not to read there. I used the research-note skill's own instructions, the write-technical-prose skill and the form of the sibling RHM notes. For the same reason I saved nothing to the memory directory.
9. **Exploratory numbers are reported and marked.** The r = 32 tree's product features on RRXOR are not eligible under the definition's rule, but its affine product probe scores 0.934–0.982 on unseen prefixes at t = 13–15, above the additive ceiling. The README and the note report this as outside the criteria.
10. **I ran `orch log` once** while inspecting the merged folder. It rebuilds `EXPERIMENTS.md`, which `orch finish` had already modified. I have not committed that file, as instructed.

## How the write-ups were checked

- **README.** A fact-checker compared about 290 claims with the result files, the run logs, the worker's job records and the definition, and found 12 that did not fully hold; all are corrected. The ones that changed the reading: the chain scores above the token readout on the `coupled` joint and fails only the halving rule; the r = 8 tree does not halve the token readout's error on the factor marginals; the r = 8 tree on RRXOR is above the next-token ceiling in two seeds and fails on the 0.8 threshold and the additive ceiling; keeping the 4-hour rule would have left 56 runs and censored all three r = 32 cells of H3a.
- **Note.** A second fact-checker checked about 120 claims and the derivations against the code and asked for 11 rewordings; all are applied. The largest: the RRXOR next-token ceiling is 0.45–0.60 on held-out prefixes at t = 5–15, not the definition's pre-run 0.45–0.57, and the hand-built tree check covers four positions of an 8-token tree.
- **Not done:** neither file was fact-checked again after its corrections went in. The note's mathematics has not been seen rendered; no Markdown renderer is on the coordinator. I removed LaTeX thin spaces, moved the kernel formulas into display equations, and checked that every `$` and `$$` and every brace is balanced and that the five image paths exist.

## What the user must do

- **Commit `EXPERIMENTS.md`** from the planning session. It is modified in the working tree by `orch finish`, `orch experiment done` and my `orch log`, and I left it uncommitted.
- **Look at `NOTE.md` on GitHub once**, to confirm that the mathematics and the five embedded figures render.
- **Decide whether "inconclusive" is the label you want** for a result with one supported and three partly supported hypotheses.
- **Decide on the follow-ups** listed under "Not yet checked" in the README. The one that bears most on the conclusions is RRXOR with a context of 32 tokens, which would make a larger tree's product features eligible and test the exploratory observation. A projective probe that keeps its denominator away from zero, and a probe with products of the chain's state coordinates on `factored`, need only the saved checkpoints.
- **The run data** (`runs/hmm-belief-geometry/`, 0.57 GB of checkpoints and logs) is on the coordinator and not in git.
- Nothing is left running: the worker is deleted and nothing was pushed.
