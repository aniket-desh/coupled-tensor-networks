# Supervisor report: rhm-belief-geometry

This session supervised the experiment unattended from launch to write-up, on 2026-11-14 and 15 (UTC). All 51 runs
trained and were analyzed. H1 is **supported**: 10 of 12 learned, testable grammars pass in all three seeds. It cost
about $4.98 against a $20 budget, and no worker of this experiment is left running. The write-up is
[README.md](README.md), and the definition's Outcome and Amendments are filled in.

## What ran

- **Worker.** Pod `dl1dad9eotnowo`, an RTX PRO 4500 Blackwell (32 GB) in EUR-IS-1 at $0.72/h. The A6000, first in the
  definition's list, had no stock at launch, so the worker got the second choice. It was created at 18:30 and deleted
  at 01:25 UTC, 6.92 h against the 8 h cap, about $4.98.
- **Checks (18:32).** `studies.rhm.checks` passed 28/28 and `check_probe.py` passed. CUDA was present.
- **Training (18:33 to 22:38).** All 51 runs (17 grammars × 3 seeds) ran as defined, four at a time, with no
  failures. Training finished at about 4.1 h, before the 5 h point where the definition stops starting new runs, so
  nothing was cut.
- **Analysis (22:38 to 01:22).** All 51 runs were analyzed, after rescheduling for GPU memory (see Decisions). Then
  `summary.py` and `plot.py` ran on all 51 results.
- **Collection (01:25).** `orch finish --no-merge`, then a selective commit (see Decisions).

## Result against each criterion

| criterion | observed |
|---|---|
| Run failed (checks, CUDA, ≥ 44 of 51 runs, shuffled R² ≤ 0.05) | not failed: checks passed, CUDA present, 51/51 runs, shuffled R² ≤ 0.001 |
| Learned (excess < 0.02 in all seeds) | 16 of 17 grammars; v16m8s2L4 not learned (0.040) |
| Testable (an eligible position in all seeds) | 13 of 17; four small grammars are figure-only, as planned |
| H1 per grammar | 10 of 12 pass; v3m2s2L6 (1 of 3 seeds) and v16m4s2L4 (R² 0.68 to 0.72) fail |
| H1 overall (≥ 75%) | **supported**, 10/12 = 83% |
| H2 (descriptive) | 383 of 414 low-ceiling cases beat the lookup ceiling by > 0.05; may be inflated, see README |

## Decisions I made

1. **Analysis rescheduled after GPU out-of-memory.** The definition runs two analyses at a time, but two v16
   analyses need more than this card's 32 GB, so v16m2s2L4_s0 and _s2 failed. The definition allows "rerun that run
   alone". I stopped the step and ran the large grammars (T ≥ 16) one at a time, with the T ≤ 9 grammars in a second
   queue alongside.
2. **Two extra queues to stay inside the cap.** Serial v16 analyses took about 10 minutes each, which projected the
   end near 01:55 against a 02:31 cap. So I added a reverse-order queue over the smaller T = 16 to 32 grammars
   (`analyze3`), and later another (`analyze4`). Running next to v16 analyses, these caused three more out-of-memory
   failures: v8m2s2L4_s1 and _s2, and v16m2s2L4_s0 a second time. All three were rerun, the last one alone with
   nothing else on the GPU (`analyze5`). The script, data, seeds and criteria never changed. Each step is recorded
   under Amendments in `EXPERIMENT.md`.
3. **No automatic merge of the results.** The probe projection files (`results/*/proj_ckpt*.npz`) total 2.1 GB,
   against the definition's estimate of under 200 MB. The automatic merge would have committed 72 of them, about
   590 MB, to git. I finished with `--no-merge` and extracted the job's folder into the working tree. I added
   `studies/rhm/rhm-belief-geometry/.gitignore` so the projections stay on disk but out of git, and copied the 30
   projection files over 20 MB from `.orch/incoming/` into `results/`. The commit holds the analysis JSONs, distance
   samples, summary, figures and checks, 108 MB in all.
4. **Restored the amendments after extraction.** Extracting the job's folder overwrote `EXPERIMENT.md` with the
   worker's copy, which lacked my amendments. I rewrote them from this session's record before committing; no other
   file in the folder changed.
5. **Write-up corrections.** The fact-checker found 19 claims in the first draft that did not fully hold, mostly
   overstatements. All were corrected before the result was recorded.

## What it cost

About $4.98 of GPU time (6.92 h × $0.72/h), plus this session's API usage.

## What you need to do

- **The experiment log is yours to update.** `EXPERIMENTS.md` was left alone, as instructed. `orch experiment done`
  has recorded the result in the definition's front matter.
- **An unmerged results branch remains:** `results/20261114-183036-rhm-belief-geometry-dl1dad9eotnowo`. It holds the
  job's folder, including about 590 MB of projection files. Everything useful from it is already in the folder.
  Delete it with `git branch -D results/20261114-183036-rhm-belief-geometry-dl1dad9eotnowo` when you're satisfied.
- **A duplicate copy of the large files remains.** The 30 large projection files (1.5 GB) are also in
  `.orch/incoming/20261114-183036-rhm-belief-geometry-dl1dad9eotnowo/large-repo-files/`. That copy can go once you've
  confirmed the copy in `results/`.
- **Nothing was pushed.** The commits are on local `main` only: results, then write-up and outcome, then this report.
- **Possible follow-ups** (from the write-up):
  - A smooth next-token ceiling, which decides whether H2's count holds; minutes of CPU.
  - An init-seed replicate of v3m2s2L6, to separate grammar from initialization; about an hour of GPU time per run.
