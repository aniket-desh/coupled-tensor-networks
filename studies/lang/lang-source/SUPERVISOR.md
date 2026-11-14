# lang-source: supervisor report

The experiment ran to completion as defined, with no failures and no amendments, in 2 h 47 m for about $1.64. It produced a clean result. The symmetry control holds and the mechanism consistency check passes. The two directional predictions (H2 chain distance, H6 distractor fall) are not supported, and the tree's meeting-level comparison (H3) is unresolved at n = 3, with every pair tied. The overall result is recorded as **not-supported**. The full write-up is [README.md](README.md).

## What ran

All commands from `EXPERIMENT.md` ran in order on one A40 worker (`ns9y2lw65uf2jk`, EU-SE-1, $0.59/h), as four `orch exec` steps:

1. **Checks (7 min):** `nvidia-smi`, the CUDA assertion, `studies.lang.checks` (47/47 passed) and `studies.lang.rank` (centered-logit rank ≤ 8 at every cut).
2. **Training (1 h 22 m):** 16 LR pilots, then 99/99 runs finished, with 0 crashed, 0 collapsed, 0 censored and 0 skipped by the deadline. The 5 h training deadline was never close.
3. **Interpretation (1 h 13 m):** 69/69 tree, tied and tree32 runs interpreted, inside the 2 h deadline.
4. **Analysis and plots (under 1 min):** `results/analysis.json`, `curves.json` and `interp.json`, plus `figures/curves.pdf`, `t.pdf` and `probes.pdf`.

The worker was then finished with `orch finish`, which published the results, merged the folder and deleted the pod. `orch ls` shows no lang-source worker left.

## Result against each criterion

| Criterion | Verdict | Key numbers (n = 3 seeds) |
|---|---|---|
| H1, symmetry control | holds | Tree median t 79/79/79/70 at j = 0/7/8/15 (spread 9, against a seed range of 13). Root pairs 130/130/130. |
| H2, chain j = 15 faster than j = 0 | not supported | 3/3 tied (median 60 and 60). Middle positions took 810–910 (not predicted). |
| H3, tree level 1 vs level 3 | unresolved at n = 3 | 3/3 tied (130 against 130–140). Level 1 vs root (confounded): 3/3 tied. |
| H4, descriptive | n/a | Tree 130 at every root pair. Chain 980 / 2,550 / 140 at (7,8) / (0,8) / (0,15). |
| H5, mechanism | consistent | Source 108/108, f 18/18 with a gain of 0.61, sub-patch 1.00 against random 0.33. |
| H6, distractor fall | not supported | Median change 0.00 (n = 123), against the criterion ≤ −0.2. |

## Cost

The worker ran 2 h 47 m at $0.59/h, about $1.64, against a 10 h cap and a $12 budget. The fact-checker subagent used about 140k tokens.

## Decisions I made

- **GPU:** the launcher took an A40 after the L40S and RTX A6000 were out of stock. This is the definition's second choice, so it needed no decision.
- **Steps:** I combined the definition's first five lines (setup, `nvidia-smi`, the CUDA check, checks and rank) into one `orch exec` step under `set -euo pipefail`, so any failure stops it. Analysis and plotting were also one step. The definition allows one step per line but doesn't require it.
- **tree32 learning rate at the grid edge:** the pilot picked 1e-4. Per "If something fails", I proceeded and reported it in the write-up. The pilot losses at 1e-4 and 3e-4 differed by 0.0002 nats.
- **Overall result label:** I chose `not-supported` over `inconclusive`. The two predictions that could fail (H2, H6) failed cleanly on complete data, H3 is unresolved by its pre-set rule, and the control and the consistency check passed. The run did answer its questions, so it isn't `failed`. `inconclusive` would also be defensible, given H3 was the central question.
- **Exploratory analysis in the write-up, marked as not pre-registered:**
  - the chain's slowdown at the middle positions, and a counting explanation that this experiment doesn't test;
  - the H6 breakdown by tree level. Level 1 is saturated at 1.0, levels 2–3 show a median change of −0.22, and 35 of those 81 cases rose.
- **Fact-check:** the fact-checker flagged 7 of about 115 claims, and I applied all 7 corrections. Three were wrong: the f-probe timing at the level-1 meeting bond (f becomes decodable by update 40–60, before the loss drops), the probe figure's distractor row (it shows the median over all nodes at a level), and a claim that finer logging could change H3's verdict (it can't, since the 1.25 ratio rule can't be met). Four were overstated: the opening sentence, the H6 breakdown, the cumulative-LR count for tree vs tied, and the scope of the capacity check.

## What the user must do

- **Experiment log:** `EXPERIMENTS.md` has an uncommitted entry for this run, written by `orch finish` and `orch experiment done`. I left it, as instructed, for the planning session to commit.
- **Run data:** `runs/lang-source/` (0.6 GB of per-run results, checkpoints and `interpret.json`) is on the coordinator and git-ignored. The global volume keeps the published tar for 14 days (`results.keep_days`).
- **Possible follow-ups, if wanted:**
  - a chain position sweep over all 16 positions, plus a chain with a position-dependent core, to test the counting explanation for the middle-position slowdown;
  - finer early logging for H3, which can bound a small level effect but can't change the verdict.
- **Commits:** none were pushed. The commits are on `main`: 886acbe (results, made by orch), 9a6d589 (write-up and outcome) and this report.
