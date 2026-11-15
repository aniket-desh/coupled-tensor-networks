# Supervisor's report: belief-probe-controls

The experiment ran to completion and its result is recorded as **inconclusive**, which stands here for the definition's "partly supported": on `factored` the chain and the r = 32 tree both hold exact factor marginals plus part of the joint, the outcome the definition expected. Every run was produced, no check or reproduction rule failed, and the worker cost about $1.46 of its $10 budget. The write-up is [`README.md`](README.md) and the research note is [`NOTE.md`](NOTE.md).

## What ran

All times are UTC on 15 November 2026. The worker was `g0ldf6s884roj2`, an RTX PRO 4500 Blackwell in EU-RO-1 at $0.72/h, the first GPU on the definition's list.

| time | step | outcome |
| --- | --- | --- |
| 19:38 | launch from the approved definition | worker ready in about two minutes |
| 19:40–19:41 | setup and checks | CUDA present, 2.1 GB of checkpoints unpacked to `/root/ckpt`, `studies.hmm.checks` 29 of 29, `studies.rhm.checks` 28 of 28 |
| 19:41–20:41 | parts 3 and 1, then the Mess3 and RRXOR runs of part 2 | `rrxor_tree.py` done, 24 of 24 runs of `factor_product.py`, 24 of 24 runs of `lowdim.py`, first summary |
| 20:42–21:36 | Random Hierarchy Model runs of part 2, three queues per family, then the plots | tree 30 of 30; transformer 48 of 51, three runs out of GPU memory |
| 21:36–21:39 | rerun of the three failed runs, one process at a time, then summary and plots again | transformer 51 of 51; summary and three figures cover all 105 runs |
| 21:40 | `orch finish` | results published, collected and merged (commit `27568c6`, made by the toolkit), worker deleted |
| 21:42–22:02 | summary figure and PNG copies, floor formula, write-up, note, fact-check, result recorded | commit `b8c9dcb` |

`orch ls` showed no worker of this experiment after 21:40. The one worker still listed, `aniket-ctn-rhm-ttn-masks`, belongs to another supervisor.

## Result against each criterion

| criterion | result |
| --- | --- |
| Run not failed (project checks, split assertions, reproduction within 0.01, enough runs) | met: checks 29 of 29 and 28 of 28; no assertion failed; largest reproduction difference 8.2e-08 in part 1 and 2.3e-05 in part 2; 24 of 24 runs in part 1 (22 needed), and in part 2 24 of 24, 30 of 30 and 51 of 51 (22, 27 and 46 needed) |
| H1, factor code (primary) | **partly supported**, as expected. Chain and r = 32 tree: "marginals plus part of the joint" in all three seeds (product of the model's marginal probes 0.999 and 0.996 against 0.967 from token marginals; direct joint probe 0.078–0.080 and 0.049–0.064 above the floor of 0.755–0.756). Transformer "joint readable" (0.976–0.982). r = 8 tree "marginals not exact" (product 0.980–0.981) |
| `coupled` (descriptive) | not read: its residual carries 0.10–0.11% of the joint's variance, below the required 1% |
| H2, Mess3 and RRXOR | dominant in 5 of 10 judged learned cells, and the same 5 compact: Mess3 in the chain, the transformer and the first half of both trees; RRXOR in the chain. RRXOR in the transformer is neither (reverse score 0.10, 0.10, 0.43) |
| H2, Random Hierarchy Model, transformer | dominant in 7 of 12, compact in 0 of 12 |
| H2, Random Hierarchy Model, r = 32 tree | dominant in 2 of 15, compact in 4 of 15, none both; no second-half cell has either label |
| H3 (a), the r = 8 seed at −4.55 | rule not met, as in the CPU pilot: 146 of 31,129 rows carry 99% of the squared error, and without them the seed scores 0.839 against 0.673 and 0.715. The affine probe gives 0.751 |
| H3 (b), the r = 32 tree with reduced product features | fail in two of three seeds, as in the CPU pilot: 0.884, 0.883, 0.895 against a required 0.886; no interval |

## Cost

The worker existed from 19:38 to 21:40, 2 h 02 min at $0.72/h, about $1.46 by the toolkit's record. The definition estimated 4 to 5 hours and about $3.50, with a 7-hour cap and a $10 budget. Part 2 on the Random Hierarchy Model was the part that ran faster than estimated: 57 minutes including the rerun, against an estimated 2 to 3 hours.

## Decisions I made

1. **Rerun of three out-of-memory runs** (21:36). `v3m2s2L6_s1`, `v4m2s2L5_s0` and `v4m2s2L5_s2` failed with CUDA out of memory while three processes shared the card. None is one of the 4,096-wide grammars the definition put in the serial queue. I reran them one at a time, as "If something fails" prescribes, and ran `lowdim_summary.py` and `plot.py` once more afterwards so that they cover all 51 runs. That extra summary and plot is the only addition to the written commands. I recorded it under "Amendments" before starting the step; I first stamped the entry 21:40 and corrected it to 21:36 after the fact-checker noticed.
2. **The diverged projective fit in part 3 was not treated as a reproduction failure.** For the r = 8 seed at t = 13 the fit scores −15.45 here, −15.42 in the CPU pilot and −15.26 in the saved analysis. The definition's 0.01 rule names part 1's direct probe and part 2's full probe, `rrxor_tree.py` enforces none, and the row count, the trimmed score and the affine score agree to three decimals. The alternative was to stop part 3 and report nothing from it.
3. **Result label "inconclusive".** The tool accepts supported, not-supported, inconclusive or failed. The definition's label is "partly supported", and it reserves "not supported" for inexact marginals or a readable joint in the chain or the r = 32 tree. The earlier `hmm-belief-geometry` run recorded its mixed outcome the same way. The planning session may prefer another label.
4. **A summary figure and PNG copies, added to `plot.py` after the run.** `figures/summary.pdf` shows H1 on `factored` and the two H2 scores for every run, and `plot.py lowdim png` writes 200-dpi copies to `figures/png/` for the note, since `pdftoppm` is not installed. The three figures drawn on the worker are unchanged in git.
5. **A closed form for the factor-code floor, `floor_formula.py`**, run on the coordinator's CPU after the worker was gone. It gives 0.7554–0.7559 on an independent sample against the measured 0.7552–0.7564, and its output is `results/floor_formula.json`. It enters no rule; the note uses it to derive the floor.
6. **The Random Hierarchy Model step was passed to `orch exec` from a scratch file**, because its command contains single quotes. The command text is the definition's.
7. **Commit by pathspec.** I committed with `git commit -- studies/hmm/belief-probe-controls`, so nothing another supervisor had staged could enter my commit. Nothing was pushed, and I did not touch `EXPERIMENTS.md`, `orch.toml` or any file outside the folder.
8. **The research-note skill's style reference was read** from the skills folder under `/root/.claude`, as that skill instructs. Nothing was written there.

## How the write-ups were checked

One fact-checking pass covered `README.md` and `NOTE.md` together, about 720 claims against the result files, the scripts, the worker's logs and the three earlier experiments. Every table cell held. It asked for 17 rewordings and 5 additions, and all are applied. The ones that changed the reading are listed at the end of the README: the opening no longer says all three readings "were settled", "Mess3 in all three architectures" became "the chain, the transformer and the first half of the tree", and "the second half of the tree cannot be dominant by construction" was softened because one run passes the per-run rule there.

Not done: neither file was fact-checked again after the corrections went in, and the note's mathematics has not been seen rendered, since no Markdown renderer is on the coordinator. I checked that every link in the three Markdown files resolves and that the `$` delimiters are balanced.

## What the user must do

- **Commit `EXPERIMENTS.md`** from the planning session. `orch finish` and `orch experiment done` modified it, and I left it uncommitted as instructed.
- **Look at `NOTE.md` on GitHub once**, to confirm that the mathematics and the three embedded figures render.
- **Decide whether "inconclusive" is the label you want** for a primary hypothesis that came out "partly supported", as expected.
- **One statement in the definition turned out too strong.** It says the tree's second half "cannot be `dominant` by construction". No second-half cell is dominant, but one run is (`v4m2s2L5` seed 0, reverse 0.87, full probe 0.88). The write-ups say "not expected" instead.
- **Decide on the follow-ups** under "Not yet checked" in the README. The cheapest is a bootstrap over prefixes for H3 (b), which two seeds miss by 0.002. The one that bears on the most labels is repeating part 2 for the chain and the trees after fixing the gauge of the bonds. The RRXOR transformer's reverse score rests on the one or two principal directions that hold 90% of its last layer's variance, and what those directions encode was not examined.
- Nothing is left running for this experiment. The published results are on the global volume under `results/20261115-193815-belief-probe-controls-g0ldf6s884roj2`, and the toolkit merged them into `main` as commit `27568c6`.
