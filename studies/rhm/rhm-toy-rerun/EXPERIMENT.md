---
name: rhm-toy-rerun
title: Rerun of rhm-forced-synonyms and rhm-crossed-rank on a Runpod CPU worker
status: done
cpu: 16
count: 1
hours: 1.5
budget_usd: 3
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-17T23:01:58Z
result: supported
finished_utc: 2026-11-17T23:52:22Z
writeup: studies/rhm/rhm-toy-rerun/README.md
---

# Rerun of rhm-forced-synonyms and rhm-crossed-rank on a Runpod CPU worker

## Question

`rhm-forced-synonyms` and `rhm-crossed-rank` were run on the coordinator's 2 vCPUs on 2026-11-17, because Runpod
workers could not be launched that day (Runpod retired REST v1 on 2026-11-15 and the toolkit's pod calls still used it;
the Runpod key had also been swapped for another account's). Both are fixed: the old key is back in the project's
`.env` and the toolkit's pod calls are on REST v2 (toolkit branch `rest-v2-pods`, checked live with a test worker).
The user asked for the experiments to be rerun with worker access back. This experiment reruns both grids from
scratch on one CPU worker, with the code, seeds and thresholds of the two approved definitions unchanged, and asks:
do the verdicts of both experiments come out the same on other hardware? It is also the first full launch, run and
finish through the ported toolkit.

No GPU is used: every run is a full-batch problem with 24 inputs and at most 1,400 parameters, so a GPU would be
slower than a CPU core. A 16-vCPU worker runs the 880 trees in about 15 minutes.

## Hypothesis

The rerun reproduces the verdicts of the coordinator runs: for `rhm-forced-synonyms`, H1 to H4 as that definition
states them; for `rhm-crossed-rank`, H1 to H4 and the trainability readings. The two machines use the same code, seeds
and float64 arithmetic but different CPUs, so individual runs near a saddle may end differently; agreement is expected
on verdicts, smallest exact widths and, within a few runs, the exact counts per cell.

- Supported: every verdict string of both `summary.json` files equals the coordinator's (`compare.py`).
- Not supported: any verdict differs. That would mean a verdict of the original run rests on a handful of runs, and
  the write-up says which and by how many runs.
- When this experiment finishes, the coordinator's `rhm-crossed-rank` run may not have finished (it takes about 2.5
  hours there). The comparison for it is then made later by the planning session; this experiment's own outcome is
  judged on what exists, and says which comparison is pending.

## Method

The two approved definitions, `studies/rhm/rhm-forced-synonyms/EXPERIMENT.md` and
`studies/rhm/rhm-crossed-rank/EXPERIMENT.md`, with their `run.py`, `analyze.py` and `plot.py` unchanged: the same
grids (160 and 720 trees), seeds 2000 and up, 20,000 Adam updates and the L-BFGS polish, the same thresholds and
verdict code. Only the output locations differ: trees go to `runs/rhm-toy-rerun/<experiment>/`, results and figures
to `results/<experiment>/` and `figures/<experiment>/` in this folder, so the coordinator runs' folders are not
touched. `compare.py` (run on the coordinator after the worker's results are merged) compares verdicts and per-run
exactness with the coordinator runs.

## Inputs

None. The grammars are written out in `studies/rhm/toy.py`.

## Commands

From the project root on the worker. Checks:

```bash
set -euo pipefail
export PYTHONPATH=. OMP_NUM_THREADS=1
mkdir -p "$EXP_DIR/results/forced-synonyms" "$EXP_DIR/results/crossed-rank"
uv run --frozen python -m studies.rhm.toy_checks "$EXP_DIR/results/toy_checks.json" | tee "$EXP_DIR/results/toy_checks.txt"
uv run --frozen python studies/rhm/rhm-forced-synonyms/analyze.py --check | tee "$EXP_DIR/results/measure_checks.txt"
```

`rhm-forced-synonyms`:

```bash
set -euo pipefail
export PYTHONPATH=. OMP_NUM_THREADS=1
uv run --frozen python studies/rhm/rhm-forced-synonyms/run.py --out runs/rhm-toy-rerun/forced-synonyms --procs "$ORCH_CPUS" | tail -3
uv run --frozen python studies/rhm/rhm-forced-synonyms/analyze.py runs/rhm-toy-rerun/forced-synonyms "$EXP_DIR/results/forced-synonyms" | tee "$EXP_DIR/results/forced-synonyms/summary.txt"
uv run --frozen python studies/rhm/rhm-forced-synonyms/plot.py "$EXP_DIR/results/forced-synonyms" "$EXP_DIR/figures/forced-synonyms"
```

`rhm-crossed-rank`:

```bash
set -euo pipefail
export PYTHONPATH=. OMP_NUM_THREADS=1
uv run --frozen python studies/rhm/rhm-crossed-rank/run.py --out runs/rhm-toy-rerun/crossed-rank --procs "$ORCH_CPUS" | tail -3
uv run --frozen python studies/rhm/rhm-crossed-rank/analyze.py runs/rhm-toy-rerun/crossed-rank "$EXP_DIR/results/crossed-rank" | tee "$EXP_DIR/results/crossed-rank/summary.txt"
uv run --frozen python studies/rhm/rhm-crossed-rank/plot.py "$EXP_DIR/results/crossed-rank" "$EXP_DIR/figures/crossed-rank"
```

After `orch finish`, on the coordinator (system python, no extra packages):

```bash
python3 studies/rhm/rhm-toy-rerun/compare.py studies/rhm/rhm-toy-rerun/results/compare.json | tee studies/rhm/rhm-toy-rerun/results/compare.txt
```

## Expected outputs

- `results/toy_checks.json`, `results/toy_checks.txt`, `results/measure_checks.txt`.
- `runs/rhm-toy-rerun/forced-synonyms/` (160 files) and `runs/rhm-toy-rerun/crossed-rank/` (720 files), under 100 MB
  in all, published as data.
- `results/forced-synonyms/` and `results/crossed-rank/`: the files the two original definitions list (`runs.json`,
  `summary.json`, `summary.txt`, `provenance.json`, the saved trees, and `trajectories.json` or `curves.json`).
- `figures/forced-synonyms/*.pdf`, `figures/crossed-rank/*.pdf`.
- `results/compare.json`, `results/compare.txt`.
- `README.md`: a short write-up. The scientific write-ups are the two original experiments'; this one reports the
  rerun's verdict tables, the comparison, and the runtime and cost.

## Controls

The coordinator runs are the comparison. The checks (`toy_checks`, the measure checks) run first on the worker, so a
difference in verdicts cannot come from a broken install.

## Success and failure criteria

- Supported, not supported or pending per experiment, as under Hypothesis.
- The run failed if a check fails, or `analyze.py` prints `RUN FAILED` for either grid.

## Time and cost

13 s per run on one core of the coordinator, so 880 runs take about 12 minutes on 16 vCPUs if the worker's cores are
comparable; setup (copying the code, building the environment) adds about 5 minutes. A 16-vCPU `cpu3c` worker is
quoted at about $0.48/hr, so the expected cost is under $0.30 and the worst case at the 1.5-hour cap is $0.72, inside
the $3 budget. Each process uses about 400 MB; `--procs "$ORCH_CPUS"` follows the worker's real CPU quota. If the
worker is 1.5 times slower than estimated nothing is dropped: the cap leaves a factor of four.

## If something fails

- **No 16-vCPU CPU worker can be placed:** amend `cpu` to 8 (the run then takes about 25 minutes) and relaunch; if
  that fails too, record the result as failed with the launch error. The coordinator runs are unaffected.
- **A check fails:** stop, finish the worker, record failed.
- **A step is interrupted:** rerun it; `run.py` skips finished runs.
- **`train` prints no new line for 5 minutes:** stop the job and rerun it with `--procs` halved (a killed pool worker
  hangs the pool).
- **A toolkit command fails in a way that looks like the REST v2 port** (an HTTP 404 or 422 from `orch`, a pod that
  is not listed): do not edit the toolkit. Make sure no worker is left (`orch ls`; `orch down <id>` if one is),
  record the result as failed with the exact error, and stop.
- Nothing here needs the user.

## Fixed rules

- The code, grids, seeds and thresholds of the two original definitions are not changed.
- Nothing is written into `studies/rhm/rhm-forced-synonyms/` or `studies/rhm/rhm-crossed-rank/`.

## Questions asked before launch

None asked. The user said on 2026-11-17: "restore the old key and rerun the experiments with the gpu access back",
which covers approval. Choices made without asking: a CPU worker, since these runs gain nothing from a GPU; one
worker for both grids; a separate folder, so that the coordinator runs and their write-ups stay intact and the two
can be compared.

## Review

skipped: a rerun of two approved designs, each reviewed once, with no change to method, grid or criteria.

## Amendments

- **2026-11-17 23:43 UTC, an added output; no rule, criterion, method or budget changed.** After the worker was
  finished, `compare.py` reported the `rhm-forced-synonyms` verdicts and all 160 exactness flags equal, and
  `runs.json` turned out to be equal value for value, which is more than the hypothesis expected (it allowed runs to
  differ in the last digits). To say how far the agreement goes, the supervisor added `compare_runs.py`: it loads
  every run file present on both machines and tests each saved entry for exact equality, and reports the per-run
  wall-clock time on each machine. It runs on the coordinator, reads the two coordinator run folders, and writes
  `results/compare_runs.json` and `results/compare_runs.txt` in this folder only. For `rhm-crossed-rank` it covers
  the coordinator runs finished when it is run; the verdict comparison for that experiment stays pending as the
  Hypothesis section says. Command:
  `PYTHONPATH=. uv run --frozen python studies/rhm/rhm-toy-rerun/compare_runs.py studies/rhm/rhm-toy-rerun/results/compare_runs.json | tee studies/rhm/rhm-toy-rerun/results/compare_runs.txt`

<!-- Each change made during the run, with the time and the reason, recorded and committed before acting on it. -->

## Outcome

Recorded as **supported**, judged on what exists, as the Hypothesis section says. Write-up: [`README.md`](README.md).

- **`rhm-forced-synonyms`, supported.** Every verdict string of the worker's `summary.json` equals the coordinator's
  (H1 to H4 supported on both). 118 of 160 runs are exact on both machines, none on one machine only, and all 160 run
  files are equal in every saved entry except the wall-clock time.
- **`rhm-crossed-rank`, pending.** The worker's run is complete: 720 of 720 runs, H1 to H4 supported, smallest exact
  widths 2, 2, 3, 4 equal to the bounds, trainability readings "the crossed tree is harder to train at its bound" on
  `sym` and "no detectable difference" on `mod`. The coordinator's run had 266 of 720 runs at 23:43 UTC and no
  `summary.json`, so `compare.py` reports it missing. The 266 runs present on both machines are bit for bit equal.
  The planning session makes the verdict comparison when the coordinator's run ends.
- **Run failed?** No. Both checks passed (29 of 29 toy checks, 8 of 8 measure checks) and `analyze.py` printed no
  `RUN FAILED` for either grid.
- **Cost and runtime.** About $0.31 of the $3 budget and 38 minutes of the 1.5-hour cap (23:02 to 23:40 UTC,
  2026-11-17). Training and analysis took 31 minutes against an estimate of 12, because a run took a median of 33 s
  on the worker against the 13 s assumed; nothing was dropped.
