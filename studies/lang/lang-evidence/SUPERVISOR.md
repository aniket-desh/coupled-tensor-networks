# Supervisor report: lang-evidence

The experiment ran to completion without amendments. All 8 commands from the definition ran in order, all 48 planned runs finished, all 36 tree-type runs were interpreted, and the worker was deleted after publishing. H1, H4 and H5 are supported, H2 is unresolved at n = 3, and H3 and H6 are descriptive. It cost about $1.06 of the $12 budget, using 1.8 h of the 10 h cap. The full write-up is [README.md](README.md), and the definition's Outcome section has the one-line verdicts.

## What ran

| Step (UTC, 2026-11-13 to 14) | What happened |
|---|---|
| 23:52 launch | Launched with `orch experiment launch lang-evidence`. The A6000 and L40S had no stock, so the worker was an A40 in EU-SE-1 at $0.59/h (worker `l0lbo55eou995u`). |
| 23:53 setup | `nvidia-smi`, the CUDA check, `studies.lang.checks` (47/47 passed) and `studies.lang.rank`. |
| 23:58 train | `run.py --hours 5`. Setup and the 32 LR pilots took 7 minutes, and the 48 runs trained from 00:05 to 01:22 (1.3 h, against a 5 h deadline). No crashes, no collapses, no skipped runs. |
| 01:22 interpret | `interpret_all --procs 4 --hours 2`: 36/36 runs were interpreted by 01:38. |
| 01:38 analyze and plot | `analyze.py` and `plot.py` wrote `results/analysis.json`, `curves.json`, `interp.json` and the three figures. |
| 01:39 finish | `orch finish` published 926 run files (0.73 GB, into `runs/lang-evidence/`) and 8 folder files. It merged the folder as commit 915a6aa and deleted the worker. `orch ls` then showed no lang-evidence worker. |

## Result against each criterion

- **H1: supported.** The maximum tree final KL is 0.00164 over 9 runs, against a threshold of 0.02. tree32 against the tree is 0 wins and 9 ties. The ties were forced, because the 0.002 KL tie tolerance is larger than every masked-bigram tree KL.
- **H2: unresolved at n = 3.** p = 5 against p = 8 ties on steps to threshold and on final KL in all 3 seeds. Both measures sat at their floor.
- **H3: descriptive.** Tree against perm is 9 ties on speed and 9 on KL; the raw KL is lower for the tree in 8 of 9 pairs. Pooled by meeting level, the KL medians run 0.00097, 0.00113, 0.00124 and 0.00151 for levels 1, 2, 3 and root, all within the tolerance.
- **H4: supported** (tree32), with all three parts holding:
  - (a) 27 of 27 one-sided cases have R² ≥ 0.9.
  - (b) 9 of 9 both-neighbour bonds are at or above the ceiling − 0.05.
  - (c) The meeting-node subspace patch gives a median KL of 0.21, against 8.83 unpatched and 3.99 for random subspaces (n = 6).
- **H5: supported** (tree32), with both parts holding:
  - (c) The flag patch moves a median of 1.00 of affected pairs, against 0.00 for random subspaces.
  - (b) The level ≥ 2 invariance has a median of 0.0008, against 0.47 at step 0.
- **H6: descriptive.** All trigger arms reach final KLs of 0.0010–0.0029.
- **Inconclusive override:** not triggered (0 of 48 runs unfinished, and interpretation files exist for H4 and H5).

## Findings beyond the pre-registered criteria (post hoc, in the README)

- **A meeting-level lag in the learning curves.** p = 8 reaches each diagnostic-KL level later than p = 5. That holds in 15 of 15 seed–level comparisons at r = 32 and 14 of 15 at r = 16, and the permuted and chain controls don't show it. This is the main lead for a follow-up. The pre-registered H2 measures couldn't see it.
- **Where the flag patch fails.** It fails, moving at most 1% of pairs, at the 3 internal nodes whose span holds leaf 15, in every trigger tree run. There the flag must be held jointly with the current token. The median criterion passes anyway.
- **Gradient-flow time changes four speed readings.** The arms were piloted to different LRs. Recomputed in gradient-flow time, tree against chain goes from 9 tree wins to 3 wins and 6 ties, and tree against perm from 9 ties to 1 tree win and 8 ties. tree32 against the tree reverses from 9 tree wins to 9 tree32 wins, and tree16's speed lead over tree32 on the trigger also reverses. No criterion depended on these readings.

## Decisions I made

- **Pilot LRs at the grid edge were kept.** The masked-bigram tree32 chose 1e-4 and the trigger tree16 chose 3e-3, both at the edge of the grid. The definition says nothing about grid-edge picks, no arm collapsed, and only descriptive speed readings could change, so I didn't rerun the pilots. It's noted in the README's judgement calls.
- **The H5(c) median was read as written.** I reported it as supported and stated the leaf-15 split next to the verdict, rather than downgrading.
- **The overall result was recorded as `supported`.** The three hypotheses with decision rules that could be met (H1, H4, H5) are supported, and H2 is explicitly unresolved, not negative.
- **Gradient-flow comparisons were added.** The project's rules give time as η × steps, so I computed them as supplementary descriptive numbers. The pre-registered rules use updates, and the verdicts follow those.
- **Fact-check.** The fact-checker ran twice. The first pass found the meeting-level lag (my draft had called the mask positions indistinguishable), the forced KL ties, the extra gradient-flow reversals, a wrong probes-figure caption and some small numerical errors. The second pass found five remaining overstatements. All are corrected.
- **The cost estimate is mine.** It comes from the worker's creation time (23:52:16) and deletion time (about 01:39:40) at $0.59/h, which is about $1.06. I didn't read the provider's bill.

## Design issues to know about

- The KL tie tolerance of max(0.002, 10%) is larger than nearly every final KL in this experiment. So every pre-registered KL comparison among the tree arms (H1 capacity, H2 quality and H3 tree against perm quality) was bound to tie. H6 tree16 against tree32 nearly was too. The definition's review didn't catch it. A follow-up should scale the tolerance to the KLs actually reached.
- The definition expected the combined-evidence probe subspace to have about 15 dimensions. The code caps it at half the bond and gives 8 of 16 at r = 16 and 9–11 of 32 at r = 32. That doesn't affect the H4 verdict, which is decided on r = 32 as planned.

## What you need to do

- **Experiment log.** `EXPERIMENTS.md` has uncommitted changes in the working tree; `orch experiment done` may have updated it. I left it uncommitted, as instructed, for the planning session.
- **Run data.** Run directories with checkpoints are in `runs/lang-evidence/` (0.73 GB, not in git). The published copy on the global volume is removed after `keep_days` = 14.
- **Follow-up.** If you want one, pre-register steps to a fixed diagnostic-KL level as the H2 measure, with a KL tolerance below 0.001 nats. Rerunning this grid costs about $1.
- No workers of mine are left running. The three other running workers belong to lang-source, lang-parity and lang-induction.
