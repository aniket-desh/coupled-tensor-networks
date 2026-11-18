# Supervisor's report: rhm-crossed-rank

No rule or criterion was changed. The definition has no amendment: every command ran as written, at the first
attempt, and nothing under "If something fails" was needed.

The experiment ran to completion on the coordinator and its result is recorded as **supported**: H1, H2 and H3 are
met, and H4 is "attained", as computed by `analyze.py`. Two things a reader should not take from the verdict alone.
The pre-stated trainability reading is weak: it says the crossed tree is harder to train at its bound on `sym` by one
run (p = 0.047) and finds no detectable difference on `mod`, and at the default learning rates the two verdicts swap.
And "the crossed bound of 4 is attained" rests on ten trained trees, not on a construction. The write-up is
[README.md](README.md) and the research note is [NOTE.md](NOTE.md).

## What ran

All times are UTC, 17 to 18 November 2026, on the coordinator's 2 vCPUs with `--procs 2`. No worker was created and
no `orch` command that needs Runpod was called.

| time | step | outcome |
| --- | --- | --- |
| 22:34:26 | start of the definition's commands, as one background script | |
| 22:34 to 22:59:08 | the script's wait for `rhm-forced-synonyms` to write its summary | 24 m 42 s of the hour allowed |
| 22:59:20 | `studies.rhm.toy_checks` | 29 of 29 passed |
| 22:59:22 to 00:51:32 | `run.py --procs 2`, 720 runs (480 main grid, then 240 rate arm) | 720 of 720 written, no `.err` file, no non-finite loss; 112 minutes |
| 00:51:55 | `analyze.py`, `plot.py` | summary, verdicts, two PDFs; script exit 0 |
| 00:52 to 01:10 | PNGs, `posthoc.py`, README, note, fact check and its corrections | see below |
| 01:10:32 | `orch experiment done … --result supported` | definition marked done |
| 01:10 | commit `64491d5` of this folder | 19 files |

## Result against each criterion

| criterion | threshold | observed | verdict |
| --- | --- | --- | --- |
| Run not failed | checks pass; both width-3 `mod` cells have 20 runs | 29 of 29; 20 and 20 | met |
| H1: no run below its bound is exact (7 cells, 140 runs) | none below 1e-5 nats | smallest 1.15e-3 (`mod` crossed, r = 3) | supported |
| H2: on `mod` at r = 3, aligned has an exact run and crossed has none | crossed all above 1e-5 | aligned 20 of 20 exact; crossed 0 of 20, 1.15e-3 to 4.19e-3 | supported |
| H3: on `sym` at r = 2 both groupings have an exact run | at least 1 each | aligned 19 of 20; crossed 14 of 20 | supported |
| H4: on `mod` crossed at r = 4, an exact run with logit error below 1e-4 | 1e-4 | 10 of 20 exact (all variance init); best run 1.8e-7, largest weight 2.73 | attained |
| Trainability at the bound, `sym` (best rate per init) | Fisher p < 0.05 | 20 of 20 aligned, 15 of 20 crossed, p = 0.047 | crossed harder to train |
| Trainability at the bound, `mod` (best rate per init) | Fisher p < 0.05 | 20 of 20 aligned (r = 3), 17 of 20 crossed (r = 4), p = 0.23 | no detectable difference |
| **Overall** | H1, H2, H3 all supported | | **supported** |

Counts the definition asked to be reported with the verdicts:

- **Censored:** the polish stopped at its limit in 102 of the 480 default-rate runs (76 of them in the 7 cells below
  their bound, 17 of them exact anyway) and in 32 of the 240 rate-arm runs. None was dropped.
- **Between the thresholds (1e-10 to 1e-5):** 2 runs of the main grid, both `sym` crossed at r = 3 with the variance
  init (2.3e-8 and 3.5e-8), and 1 of the rate arm (`mod` aligned, r = 3, rate 3e-2, 3.6e-10). All three have a
  capped polish and count as neither exact nor floor.
- **Found by Adam:** 318 of the 321 exact default-rate runs were below 1e-6 nats when Adam stopped. One exact run
  (`sym` aligned, r = 2, identity init, seed 2006) was at 3.1e-3 and was found by the polish.
- **Trainability at the default rates**, given beside the reading as the definition asks: `sym` 19 against 14 of
  20 (p = 0.091, no detectable difference); `mod` 20 against 10 of 20 (p = 0.0004, crossed harder). The rate arm is
  complete, so the best-rate comparison carries the reading.
- **Descriptive items:** the smallest exact width is 2, 2, 3, 4 (`sym` aligned, `sym` crossed, `mod` aligned, `mod`
  crossed), equal to the bound in each. Floors as a share of the uniform predictor's excess: `sym` r = 1, 51 %;
  `mod` aligned 76 % and 12 % at r = 1, 2; `mod` crossed 75 %, 30 %, 7 % at r = 1, 2, 3.

## Runtime and cost

Training took 112 minutes, against the definition's estimate of 80 and its cap of 3 hours from the start of training.
A run took 13 to 27 s (median 19 s) against the 13 s measured before launch. I did not establish why: the other
experiment's training had ended before this one began, but the host's load average was about 50 when I looked at
23:05 and two other Claude sessions were active on the coordinator. That is 1.4 times the estimate, inside the 1.5 the definition allowed for, and nothing was dropped; the rate arm is complete.
From the start of the script to the last figure: 2 h 17 min, of which 25 minutes were the planned wait. No hardware
was rented: $0. The whole session, with the write-up and the fact check, took about 2 h 40 min.

## Decisions I made

1. **A watchdog for the definition's 5-minute rule.** Beside the script I ran a loop that would have reported if
   `train.log` gained no line for 5 minutes while `run.py` was alive. It never fired.
2. **`posthoc.py`, three descriptive checks added after seeing the results.** The ten crossed width-4 runs that
   missed the exact predictor on `mod` ended on the width-3 floor to seven digits, which asked for a look. The script
   reads the saved trees and reports (a) the largest weight and logit error of every exact run of the four cells at
   their bound, (b) for `mod` crossed at r = 3 and 4, each run's distance from the best width-3 function on the legal
   prefixes and the singular values of its logits on the 4 × 4 block that forces width 4, (c) the bound of every
   single-token and pair bond in both grammars. No criterion rests on it, and the write-up marks all of it as observed
   after the run. My first version of (b) measured the rank across the crossed cut over all 81 prefixes, legal or
   not; that was uninformative because the tree is unconstrained on illegal prefixes, so I replaced it before
   writing anything from it.
3. **`plot.py` changed after the run, and `capacity.pdf` regenerated.** The legend of the capacity figure covered
   points in the `sym` panel, so I moved it below the panels, and I added an optional third argument that writes
   200-dpi PNG copies for the Markdown files. The plotted data are unchanged; `curves.pdf` was regenerated with the
   same content.
4. **The fact-check pass and what it changed.** The `fact-checker` agent recomputed the tables and the note's
   derivations and found no wrong verdict, count, p-value or derivation. I applied all its corrections: one table
   cell (a capped count of 0 that is 2) and two double-rounded cells; the first paragraph, which called both widths
   theorems when only the lower bounds are proved; an overstatement about the aligned tree's dependence on the init;
   a source-snapshot paragraph that was wrong in two places; an untested cause for Adam's endpoint; a description of
   the earlier experiments that was stronger than their write-ups; and time estimates with no source. The list is at
   the end of the README.
5. **The worker rerun.** While this run was going, another session ran `rhm-toy-rerun`, which executed the same
   commands on a rented CPU worker and committed its results at 23:40 UTC, 71 minutes before this run's last tree. I
   did not read them during the run. Afterwards the fact-checker pointed out that its `runs.json`, `summary.json`,
   `summary.txt`, `curves.json` and `best_trees.pt` have the same SHA-256 as this folder's; I verified that and
   cited it in the README as a reproduction check. The criteria were committed at 22:33 UTC, before either run.
6. **The result word.** "Supported" is the definition's rule (H1, H2, H3). H4 is reported beside it as attained and
   the trainability reading as stated, with its weakness in the first paragraphs of the README, not only in a table.

## What the user must do

- **The experiment log has no entry for this run.** `orch experiment done` marked the definition done but
  `EXPERIMENTS.md` lists only the worker rerun; the definition says the planning session adds the entry. I did not
  edit or commit `EXPERIMENTS.md`, which has uncommitted changes from other sessions.
- **`rhm-toy-rerun`'s README still calls the comparison with this experiment "pending".** It can be completed now:
  the five result files are byte-identical.
- **One number in the definition is wrong and was left as written**: a tree at r = 6 has 1,596 parameters, not "at
  most 1,400" (noted in the Outcome). Nothing depends on it.
- **The trained trees** are in `runs/rhm-crossed-rank/` on the coordinator (720 files, 36 MB), outside git. They are
  needed to rerun `posthoc.py` or `analyze.py`, and can be regenerated from the seeds in about two hours.
- **Follow-ups, if wanted** (none started): a seed and init-scale grid on the two crossed cells at their bound, to
  settle the trainability reading; a closed-form width-4 crossed tree for `mod` read off from the saved witnesses;
  the same rank bound on the larger grammars of `rhm-controls` and `rhm-belief-architectures`.
- Nothing was rented, no pod exists because of this experiment, and nothing was pushed.
