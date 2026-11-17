# Supervisor's report: rhm-forced-synonyms

No rule or criterion was changed. The definition has no amendment: every command ran as written, at the first
attempt, and nothing under "If something fails" was needed.

The experiment ran to completion on the coordinator and its result is recorded as **supported**: H1 to H4 are all met
on the polished trees, as computed by `analyze.py`. The one result a reader should not take from the verdict alone is
H3. It holds on the 37 exact width-2 runs that Adam found, as the criterion is worded, but 12 of those runs have one
forbidden response below 0.5, and the three width-2 runs that Adam left short of the exact predictor each nearly
silence one forbidden pair. The write-up is [README.md](README.md) and the research note is [NOTE.md](NOTE.md).

## What ran

All times are UTC, 17 November 2026, on the coordinator's 2 vCPUs with `--procs 2`, beside the session of
`rhm-crossed-rank`. No worker was created and no `orch` command that needs Runpod was called.

| time | step | outcome |
| --- | --- | --- |
| 22:34:23 | start of the definition's commands, as one script | |
| 22:34:40 | `studies.rhm.toy_checks` | 29 of 29 passed |
| 22:34:44 | `analyze.py --check` | 8 of 8 passed |
| 22:34:44 to 22:58:52 | `run.py --procs 2`, 160 runs | 160 of 160 written, no `.err` file, no non-finite loss; 24 m 08 s |
| 22:59:10 | `analyze.py` | 160 finite runs, `run_complete` true, 0 diverged |
| 22:59:17 | `plot.py` | four PDFs |
| 22:59 to 23:20 | tables, PNGs, README, note, fact check and its corrections | see below |
| 23:20:28 | `orch experiment done … --result supported` | definition marked done |
| 23:20:49 | commit `e6520ba` of this folder | 24 files |

## Result against each criterion

| criterion | threshold | observed | verdict |
| --- | --- | --- | --- |
| Run not failed | checks pass; at least 150 of 160 finite trees | both checks pass; 160 of 160 | met |
| Gate for H2 to H4 | at least 10 exact width-2 runs | 38 of 40 (20 variance, 18 identity) | met |
| H1: no width-1 run below 1e-5 nats | 1e-5 | smallest of 40 is 8.39e-3 | supported |
| H2: synonyms collapse, codes compose (38 exact runs) | every measure below 1e-3; compositions 6 of 6 | swap effect at most 9.0e-5, within-symbol share at most 1.7e-12, composition error at most 1.35e-6, 6 of 6 in all 38 | supported |
| H3: no forbidden pair silenced (37 exact runs found by Adam; needed 10) | median of per-run smallest response at least 0.5; no response below 0.01 | median 0.826; smallest 0.086 | supported |
| H4: the response is fixed by (x, y) (38 exact runs) | 1e-3 for both | left and right (x, y) agree to 5.8e-6; responses match the formulas to a relative 9.1e-7 | supported |
| **Overall** | H1 to H4 all supported | | **supported** |

Counts the definition asked to be reported with the verdicts:

- **Censored:** the polish stopped at its limit in 31 runs, 28 at width 1 and 3 at width 2 with the identity init
  (seeds 2005 and 2014, which are the two non-exact width-2 runs at 3.1e-3 nats, and seed 2017, which is exact at
  8.0e-14 and is the worst run for every H2 measure but one).
- **Found by the polish, not by Adam:** 1 width-2 run (identity init, seed 2006, from 3.09e-3 to 2.2e-16). It is
  exact, enters H2 and H4, and is excluded from H3.
- **Between the thresholds (1e-10 to 1e-5):** 0 runs.
- **Descriptive arm:** all 80 runs at widths 3 and 4 are exact and both widths count as answered. The synonym is kept
  in 8 of 40 runs at width 3 and 36 of 40 at width 4; 0 of 240 forbidden responses are below 0.01. None of the 38 exact
  width-2 runs is within 0.1 of x = y = −1.

## Runtime and cost

24 m 54 s from the first command to the last figure, against the definition's estimate of about 23 minutes and its cap
of 3 hours. A run took 13.0 to 24.3 s (mean 18.0 s) against the 13 s measured before launch, because two experiments
shared the two CPUs. That is 1.4 times the estimate, inside the 1.5 the definition allowed for, and nothing was
dropped. No hardware was rented: $0. The whole session, including the write-up and the fact check, took about 50
minutes.

## Other decisions I made

1. **Ran the five commands as one script with a `date` line between steps**, so the step times above exist. The
   commands themselves are the definition's, unchanged. The script and its log were in the session's scratch folder.
2. **Watched for the 5-minute stall** that the definition names as its hang signal, with a monitor on `train.log`. It
   never fired; the longest run took 24 s.
3. **Added `tables.py`.** It reads the saved results and prints the breakdowns the write-up quotes that
   `summary.json` does not hold (censored runs per cell, the three plateau runs, per-init forbidden responses, the
   widths 3 and 4 counts by side, the trajectories). Its output is `results/tables.txt`. It computes no criterion.
4. **Added an optional third argument to `plot.py`** that also writes each figure as a 200-dpi PNG, as in the last two
   batches. I re-rendered into the scratch folder and checked by checksum that the four PDFs in `figures/` are the
   ones the run produced. Two of the figures have a label that overlaps a neighbouring panel slightly; I left the
   layout as it was.
5. **Reported the plateau beside H3.** Adam left three identity-init width-2 runs at 3.1e-3 nats, each with one
   forbidden response of 2e-4 to 5e-4, below the definition's 0.01 for "silenced". The criterion is on exact runs and
   so is met, but H3's stated alternative was that training nearly silences one pair, and 3 of 40 width-2 runs end
   Adam in that state without being solutions. The fact checker asked for this to be stated next to the verdict; it
   is in the README's lead, in its H3 section and in the definition's Outcome.
6. **Did not rerun the two censored non-exact runs with a longer polish.** It would take under a minute, but it
   changes the method after seeing the result. It is listed under "Not yet checked".
7. **Recorded an error in the definition's Theory text without editing that text.** It says the smallest forbidden
   response ranges over (0, 1.43]; the upper end is 1.466, found numerically (an optimizer from 400 starts and a fine
   grid agree). No criterion uses it and the largest observed value is 1.30. The correction is in the README, the
   note and the Outcome.
8. **Fact check.** The `fact-checker` agent compared both documents with the result files, the logs, the code and the
   earlier experiments. It found every number, count and verdict correct and returned 17 corrections of wording,
   scope and proof steps. I applied all of them; the README ends with the list. The main ones: Adam alone reaches the
   exact predictor in 37 runs, not 38; the plateau statement above; "the collapse stops at widths 3 and 4" became the
   counts (32 of 40 runs still collapse at width 3); the earlier experiments used widths 16 and 32; three proof steps
   in the note were completed.
9. **`provenance.json` says `dirty: true`.** The tracked files that differed from commit `b960bb2` when the analysis
   ran were `orch.toml` and the project's experiment log. Nothing under `studies/rhm` or `ctn` differed, so the code
   that produced the results is the committed code.

## For the user

- Nothing is required to close this experiment.
- The Theory section of `EXPERIMENT.md` still says (0, 1.43]; change it to (0, 1.466] if that text is reused.
- The per-run checkpoints (7 MB, 160 files) are in `runs/rhm-forced-synonyms/` on the coordinator and are not in
  git. `results/polished_trees.pt` in git holds every polished tree, which is enough to recompute every number except
  the trajectories.
- Three follow-ups are costed in the README's "Not yet checked": a longer polish for the two stuck seeds (under a
  minute), the same 40 width-2 seeds under another optimizer or learning rate (about 15 minutes), and widths 3 and 4
  started from a perturbed one-hot tree (needs a definition). The second one decides how much of H3's median of 0.83
  belongs to Adam.
