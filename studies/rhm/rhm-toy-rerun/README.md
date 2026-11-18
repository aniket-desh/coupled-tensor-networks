# rhm-toy-rerun: the two toy-grammar experiments rerun on a Runpod CPU worker

**Result: supported for `rhm-forced-synonyms`; the comparison for `rhm-crossed-rank` is pending.** The question was
whether the verdicts of the two experiments, first run on the coordinator's 2 vCPUs, come out the same on a rented
worker. For `rhm-forced-synonyms` they do, and the agreement goes further than the definition expected: all four
verdicts are equal, and so is every saved number of all 160 runs except the wall-clock time, bit for bit. For `rhm-crossed-rank` the worker's
run is complete (720 of 720 runs, all four hypotheses supported), but the coordinator's run was still going when this
was written, so its verdicts cannot be compared yet. The 266 coordinator runs that existed at 23:43 UTC are bit for bit
equal to the worker's.

**Update, 2026-11-18 01:15 UTC, by the planning session: the `rhm-crossed-rank` comparison is complete and agrees.**
The coordinator's run ended at 00:51 UTC. `compare.py` finds every verdict equal (H1 to H4, both trainability
readings, the smallest exact widths), 513 of 720 runs exact on both machines and none on one machine only, and
`compare_runs.py` finds all 720 run files equal in every saved entry except the wall-clock time
(`results/compare.txt`, `results/compare_runs.txt`). The rest of this file is as the rerun's supervisor wrote it at
23:50 UTC, when that comparison was still pending.

Because the two machines produced identical numbers, the rerun shows that the results reproduce. It does not show how
sensitive the verdicts are to numerical noise, which is what a rerun on different hardware would have tested if the
arithmetic had differed.

The rerun took 38 minutes of worker time and cost about $0.31 of a $3 budget. The scientific write-ups are the two
original experiments' ([`rhm-forced-synonyms`](../rhm-forced-synonyms/README.md); `rhm-crossed-rank` has none yet).
This file reports only the rerun.

## The comparison

"Exact" means an excess loss below 1e-10 nats after the L-BFGS polish, the threshold both original definitions use.

| | `rhm-forced-synonyms` | `rhm-crossed-rank` |
| --- | --- | --- |
| Runs on the worker | 160 of 160 | 720 of 720 |
| Runs on the coordinator | 160 of 160 | 266 of 720 at 23:43 UTC, still running |
| Verdicts equal (`compare.py`) | yes, all 4 | pending: the coordinator has no `summary.json` yet |
| Exact runs, coordinator / worker | 118 / 118 of 160 | not compared yet |
| Runs exact on one machine only | 0 of 160 | not compared yet |
| Run files equal in every saved entry (`compare_runs.py`) | 160 of 160 | 266 of 266 present on both |

The last row compares the files that training writes, one per run. Each holds the configuration, the training log, the
checkpoints, the polished tree, the two excess losses and the polish report; `compare_runs.py` tests every one of
these for exact equality. The only entry that differs is `seconds`, the run's wall-clock time. For
`rhm-forced-synonyms` the files that the analysis derives from the runs are identical as well: `summary.json`,
`runs.json`, `trajectories.json`, `polished_trees.pt`, `summary.txt` and `diverged.json` each have the same SHA-256
on both machines.

The 266 coordinator runs of `rhm-crossed-rank` are all from the main grid; none of the 240 runs of its
learning-rate arm had finished there, so the trainability readings below rest on runs that are not covered yet.

### Verdicts of `rhm-forced-synonyms`, equal on both machines

| Hypothesis | Verdict | Supporting number (identical on both machines) |
| --- | --- | --- |
| H1, width 1 cannot reach Bayes | supported | smallest excess loss over 40 width-1 runs is 8.4e-3 nats (criterion: none below 1e-5) |
| H2, synonyms collapse and codes compose | supported | 38 of 40 width-2 runs are exact; worst swap effect 9.0e-5 (criterion: below 1e-3) |
| H3, no forbidden pair is silenced by training | supported | over the 37 exact runs that Adam found, the median of the per-run smallest response is 0.83 and the smallest is 0.086 (criteria: at least 0.5 and 0.01) |
| H4, the forbidden response is fixed by (x, y) | supported | largest relative error 9.1e-7 (criterion: below 1e-3) |

What these hypotheses mean is explained in the original write-up; the numbers match the ones it reports.

### Verdicts of `rhm-crossed-rank` on the worker

These are the rerun's own verdicts. They have no counterpart to compare with yet, and their interpretation belongs to
the original experiment's write-up. A cell is one (grammar, grouping, width) with 20 runs; `sym` and `mod` are the two
grammars, and "aligned" and "crossed" are the two ways the tree groups the four input tokens.

| Hypothesis | Verdict | Supporting number |
| --- | --- | --- |
| H1, no run below its bound is exact | supported | smallest excess loss over the 140 runs in the 7 cells below their bound is 1.15e-3 nats (criterion: none below 1e-5) |
| H2, alignment saves width on `mod` | supported | at width 3, 20 of 20 aligned runs are exact and 0 of 20 crossed runs are (smallest crossed excess 1.15e-3) |
| H3, on `sym` both groupings have an exact run at width 2 | supported | 19 of 20 aligned, 14 of 20 crossed |
| H4, the crossed bound of 4 is attained on `mod` | supported | 10 of 20 runs exact at width 4, all 10 with the variance init and none with the identity init; the best run's largest centered-logit error is 1.8e-7 (criterion: below 1e-4) |

The smallest width with an exact run equals the proved bound in all four (grammar, grouping) pairs: 2 for `sym`
aligned, 2 for `sym` crossed, 3 for `mod` aligned and 4 for `mod` crossed.

The trainability reading compares the two groupings at their own bounds by the number of exact runs of 20, with a
two-sided Fisher exact test at p < 0.05. The reading that counts uses, for each init, the learning rate with the most
exact runs among the four of the rate arm (10 runs per init and rate); the default rates are given beside it.

| Grammar | Rates | Aligned | Crossed | Fisher p | Reading |
| --- | --- | --- | --- | --- | --- |
| `sym` | best per init (counts) | 20/20 at width 2 | 15/20 at width 2 | 0.047 | the crossed tree is harder to train at its bound |
| `sym` | default | 19/20 | 14/20 | 0.091 | no detectable difference |
| `mod` | best per init (counts) | 20/20 at width 3 | 17/20 at width 4 | 0.23 | no detectable difference |
| `mod` | default | 20/20 | 10/20 | 0.00044 | the crossed tree is harder to train at its bound |

On both grammars the reading at the default rates is the opposite of the reading that counts, and the `sym` reading
that counts sits just under the threshold (p = 0.047). Of the 480 main-grid runs and 240 rate-arm runs, none errored
or diverged. Three ended between the two thresholds (1e-10 and 1e-5 nats): two main-grid runs, both in the `sym`
crossed cell at width 3, and one rate-arm run (`mod` aligned, width 3, variance init, rate 0.03, at 3.6e-10 nats).

The per-cell table is in [`results/crossed-rank/summary.txt`](results/crossed-rank/summary.txt), and the figures are in
[`figures/crossed-rank/`](figures/crossed-rank/) (`capacity.pdf`, `curves.pdf`) and
[`figures/forced-synonyms/`](figures/forced-synonyms/) (`widths.pdf`, `collapse.pdf`, `forbidden.pdf`,
`not_forced.pdf`).

## How it was done

One 16-vCPU `cpu3c` worker in EUR-IS-1 ran the two original definitions' `run.py`, `analyze.py` and `plot.py`
unchanged, with only the output folders redirected into this experiment: the same grids (160 and 720 trees), seeds
2000 and up, 20,000 Adam updates and the L-BFGS polish, the same thresholds and verdict code. Training used 16
processes with one thread each. The worker's code was a copy of the coordinator's working tree at commit `cc0fcc7`.
The coordinator's `rhm-forced-synonyms` run records commit `b960bb2`; the only change between the two commits is this
experiment's own definition and `compare.py`. Both runs started from a working tree with uncommitted changes. The
training and analysis code (`toy.py` and both experiments' `run.py` and `analyze.py`) is the same as in commit
`b960bb2` on both machines. The code change on record at the worker's launch is an optional PNG export in
`rhm-forced-synonyms/plot.py`, which the commands here do not use; the toolkit saved only a hash of the diff, so this
cannot be shown to be the only one. The copied tree also held the coordinator's `rhm-forced-synonyms` results and
figures, not yet committed then, which the worker's commands do not read; the coordinator's run files under `runs/`
are not copied to a worker. The worker copy has no git repository, so its `provenance.json` has an empty commit field;
the commit is recorded by the toolkit in the experiment log instead.

Both machines ran torch 2.11.0+cu128 and numpy 2.5.4 in float64. The coordinator is an AMD EPYC 9655 (192 logical
CPUs); the worker's host reported 256 logical CPUs, so it is a different machine, but its CPU model was not recorded
before the worker was deleted.

## How it was checked

- **Install checks, run first on the worker.** The 29 toy-grammar checks and the 8 measure checks all passed, and
  their printed output is byte-for-byte the output of the same checks on the coordinator. A broken install on the
  worker would have shown here.
- **Verdicts and exactness (`compare.py`, from the definition).** No verdict and no exactness flag differs for
  `rhm-forced-synonyms`.
- **Run-by-run equality (`compare_runs.py`, added after the run).** Added because `runs.json` came out identical,
  which the definition did not expect: it allowed runs near a saddle to end differently. The script found no
  differing entry in 160 + 266 run files. It was added by an amendment and does not change any criterion.
- **Completeness.** `analyze.py` printed no `RUN FAILED` for either grid, and the run counts match the grids.
- **Fact check.** The fact-checker agent checked this write-up against the result files, the saved trees, the job
  logs, the experiment log and the git history, and recomputed the run-by-run equality and the Fisher tests. Six
  corrections are applied: the count of runs between the thresholds (3 over all 720 runs; `summary.json` counts 2
  because it covers the main grid only), the range of the worker's per-run time, the wall-clock exception in the
  first paragraph, the account of what was uncommitted at launch, and two statements written ahead of their event.

## Not yet checked

- **The `rhm-crossed-rank` verdict comparison.** It needs the coordinator's run to finish (454 runs were left at
  23:43 UTC, at about 19 s per run on 2 processes, so a little over an hour) and its `analyze.py` to write
  `summary.json`. Then rerun the two comparison commands in [`EXPERIMENT.md`](EXPERIMENT.md) ("Commands" and
  "Amendments"); it takes under a minute and costs nothing. If the remaining runs are also bit for bit equal, as the
  first 266 are, the verdicts will be equal. That is a prediction, not a result.
- **Whether the verdicts survive numerical noise.** The rerun cannot say, because nothing differed. The verdict
  closest to its threshold is the `sym` trainability reading: p = 0.047 with 15 of 20 exact crossed runs, and one
  more exact run (16 of 20) would give p = 0.11 and the other reading. A check would need new seeds or a deliberately
  different arithmetic, which changes the method and is the original experiments' call.
- **The figures.** They were made on the worker from results that are identical to the coordinator's for
  `rhm-forced-synonyms`, but the PDFs themselves were not compared with the coordinator's.

## Judgement calls

- **Reporting `rhm-crossed-rank` as pending.** The definition says to judge on what exists and name the pending
  comparison. The alternative was to keep this session waiting for the coordinator's run; the worker was already
  deleted, so waiting would have cost nothing on Runpod, but the planning session is set up to make that comparison.
- **Adding the run-by-run comparison.** It reads the coordinator's run folders (it writes nothing there) and gives a
  stronger statement than the planned comparison. It also covers the part of `rhm-crossed-rank` that exists, which
  the planned comparison skips entirely.
- **Overall result recorded as supported.** One of the two comparisons is complete and agrees; the other is pending
  and its partial evidence agrees. Recording "inconclusive" would also have been defensible; the definition's own
  wording ("judged on what exists") decided it.

## Runtime and cost

| | Planned | Observed |
| --- | --- | --- |
| Worker | 16 vCPU `cpu3c`, about $0.48/hr | 16 vCPU, 16 usable, 32 GB memory limit, $0.48/hr, EUR-IS-1 |
| Setup | about 5 minutes | 4 m 17 s (pod 171 s, code copy 30 s, environment 56 s) |
| Time per run on one core | 13 s | median 33 s (range 19 to 44 s over all 880 runs); 19 s on the coordinator |
| Training and analysis | about 12 minutes | 31 minutes: checks 25 s, `rhm-forced-synonyms` 5 m 32 s, `rhm-crossed-rank` 25 m 09 s |
| Worker time | under 1.5 hours | 38 minutes, 23:02 to 23:40 UTC on 2026-11-17 |
| Cost | under $0.30 expected, $0.72 at the cap | about $0.31 (hourly rate times time; Runpod's bill had not posted) |

Each of the worker's 16 processes ran about 2.5 times slower than the estimate and about 1.7 times slower than a
coordinator core. The host was shared and busy: its 1-minute load average was between 22 and 71 at the seven times
the supervisor looked during the run, on 256 logical CPUs (read from `uptime`, not saved to a file). The definition planned no reduction for a slowdown of this size, because the time cap left a factor of
four, and nothing was dropped.

## What this means

The results of `rhm-forced-synonyms` reproduce exactly on a second machine, so its write-up needs no change, and the
launch, run and finish through the toolkit's port to Runpod's REST v2 API worked without a toolkit error. When the
coordinator's `rhm-crossed-rank` run finishes, the planning session should rerun the two comparison commands and
record the result here. The worker's `rhm-crossed-rank` results are complete and can be used for that experiment's
write-up now if the coordinator run is not wanted for anything else.
