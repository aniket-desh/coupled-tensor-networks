# Supervisor's report: rhm-toy-rerun

No rule or criterion was changed. The one amendment added an output (a run-by-run equality check,
`compare_runs.py`); it changed no rule, criterion, method or budget.

## What ran

One 16-vCPU `cpu3c` worker (`ks65j3qxopsyrj`, EUR-IS-1, $0.48/hr) from 23:02 to 23:40 UTC on 2026-11-17. The
hardware line gave 16 usable CPUs and a 32 GB memory limit, which is what the definition assumed (16 processes at
about 400 MB each), so nothing was scaled down. The three steps ran in order, each to exit code 0:

1. **Checks** (25 s): 29 of 29 toy-grammar checks and 8 of 8 measure checks passed.
2. **`rhm-forced-synonyms`** (5 m 32 s): 160 of 160 runs, analysis and four figures.
3. **`rhm-crossed-rank`** (25 m 09 s): 720 of 720 runs, analysis and two figures.

`orch finish` published 880 run files and 22 result files, merged the folder and deleted the worker. `orch ls` shows
no worker left. On the coordinator I then ran `compare.py` and, after the amendment, `compare_runs.py`.

## Result against each criterion

Recorded as **supported**, judged on what exists. The write-up is [`README.md`](README.md).

- **`rhm-forced-synonyms`: supported.** All four verdict strings equal the coordinator's. 118 of 160 runs are exact
  on both machines and none on one machine only. Beyond the criterion, all 160 run files are equal in every saved
  entry except the wall-clock time.
- **`rhm-crossed-rank`: pending.** The worker's run is complete, with H1 to H4 supported. The coordinator's run had
  266 of 720 runs at 23:43 UTC and no `summary.json`, so the verdicts cannot be compared yet. Those 266 runs are bit
  for bit equal to the worker's.
- **Run failed: no.** Both checks passed and `analyze.py` printed no `RUN FAILED` for either grid.

Since the two machines produced identical numbers, the rerun shows reproducibility and says nothing about how
sensitive the verdicts are to numerical noise.

## Cost

About $0.31 of the $3 budget (38 minutes at $0.48/hr; Runpod's bill had not posted when this was written), against
"under $0.30" expected and $0.72 at the cap. Training and analysis took 31 minutes against the 12 estimated: a run
took a median of 33 s on the worker against the 13 s assumed, on a shared host under heavy load. The definition's
plan for a slow worker was to drop nothing, and nothing was dropped.

## Other decisions

- **Recorded "supported" with one comparison pending.** The definition says to judge on what exists and name the
  pending comparison. "Inconclusive" would also have been defensible.
- **Added `compare_runs.py`** (amendment of 23:43 UTC, committed before it ran). `runs.json` came out identical
  between the machines, which the hypothesis did not expect, so I checked how far the agreement goes. The script
  reads the coordinator's two run folders and writes only into this experiment's `results/`.
- **Watched progress by counting run files over SSH.** The definition's hang rule is "no new `train` line for 5
  minutes", but its commands pipe training output through `tail -3`, so the job log shows nothing until training
  ends. The file count grew at every 5-minute check, so the rule never applied.
- **Did not wait for the coordinator's `rhm-crossed-rank` run.** Its own supervisor is running it, and the
  definition gives the later comparison to the planning session.
- **Fact check.** The fact-checker agent's six corrections to the write-up are applied. The one with content: 3 runs
  ended between the two thresholds over all 720 `rhm-crossed-rank` runs, where `summary.json` reports 2 because it
  counts the main grid only.

## For the user

- **Finish the `rhm-crossed-rank` comparison** once the coordinator's run has ended and its `analyze.py` has written
  `studies/rhm/rhm-crossed-rank/results/summary.json`. From the project root:

  ```bash
  python3 studies/rhm/rhm-toy-rerun/compare.py studies/rhm/rhm-toy-rerun/results/compare.json | tee studies/rhm/rhm-toy-rerun/results/compare.txt
  PYTHONPATH=. uv run --frozen python studies/rhm/rhm-toy-rerun/compare_runs.py studies/rhm/rhm-toy-rerun/results/compare_runs.json | tee studies/rhm/rhm-toy-rerun/results/compare_runs.txt
  ```

  Then update the "pending" lines in `README.md` and in the definition's Outcome.
- **The worker's CPU model was not recorded** before the worker was deleted. The host reported 256 logical CPUs
  against the coordinator's 192 (AMD EPYC 9655), so it is a different machine, but I can't say whether it is a
  different CPU family. If "other hardware" was meant to include different arithmetic, this rerun did not test it.
- **The time estimate for CPU workers was low by a factor of 2.5.** A 16-vCPU `cpu3c` worker gave 33 s per run where
  one coordinator core gives 19 s; plan future CPU runs on that figure.
- The toolkit's port to Runpod's REST v2 API handled launch, exec, finish and delete without an error.
