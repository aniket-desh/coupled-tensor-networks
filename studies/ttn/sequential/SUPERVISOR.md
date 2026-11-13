# Supervisor report: sequential

The run failed to answer its question. Phase 1 finished, but none of the six phase-2 groups finished before the worker's 5.5 h cap, so no phase-2 data exists. It cost about $11.90 of the $12.05 budget. No worker is left running. The write-up is [README.md](README.md), and the definition's Outcome is filled in.

## What ran

Worker `jche6exleq6oe1`, an A100-SXM4-80GB in EUR-IS-1 at $2.19/h, ran from 18:41 to 00:08 UTC (5 h 26 m). Times below are UTC on 12–13 Nov 2026.

1. **Setup and checks, 18:43.** `nvidia-smi` and the CUDA check passed (torch 2.11.0+cu128). `studies.ttn.checks` stalled for about 11 minutes from CPU thread oversubscription (see decision 1). It was killed, and the job ended with exit 143.
2. **Checks rerun with `OMP_NUM_THREADS=8`, 18:55.** All checks passed in 8 s on the worker.
3. **Training, 18:56 to 00:08.**
   - **Phase 1** ran 7,697 s and stopped on its 45% share at t = 67,077. 15 of 16 seeds passed the gate.
   - **Phase 2** started at about 21:04. Six groups of 64 runs ran in parallel and were still running when stopped at 00:08.
4. **Not run:** `analyze.py` and `plot.py` both need phase-2 files.
5. **`orch finish` at 00:07.** It merged the phase-1 results, which the worker had already published automatically at 23:57, and deleted the worker.

## Result against each criterion

| Criterion | Result |
|---|---|
| Transfer, primary | Not evaluated (no phase-2 data) |
| Location of transfer, quarter and half | Not evaluated |
| Inconclusive: fewer than 10 seeds pass the phase-1 gate | Not triggered: 15 of 16 passed |
| Inconclusive: `none` baseline more than half censored | Not evaluated |
| Descriptive: fresh tree, interference, K12 | Not produced |
| Run failed: checks fail, CUDA unavailable, or phase 1 didn't run | None of these occurred |

I recorded the result as `failed`, because no answer was produced. The definition's own "Run failed" conditions don't name this case.

## Cost

About $11.90 (5 h 26 m at $2.19/h), against the $12.05 budget. I launched no other workers.

## Decisions I made

1. **Thread cap (recorded as an amendment).** The container reports 255 CPUs but has a 27.2-CPU cgroup quota. Torch started about 320 threads, and the container was throttled in 6,436 of 7,118 scheduler periods. I killed the stalled checks and ran the checks and training with `OMP_NUM_THREADS=8`. I didn't change `studies/ttn/common.py`, which still sets each phase-2 group to `cpu_count // procs` = 42 threads. That is shared code outside this folder, and I judged it harmless for GPU-bound groups. It may in fact have contributed to the phase-2 slowdown.
2. **Checked for GPU contention at 21:08 and kept going.** An extra process timed the bare gradient step at 5.0 ms/step while all six groups were running, which matched the groups' own estimates. I took that as evidence they would finish around 23:30. They didn't: their real cost was at least 7.4–7.7 ms/step. The cause wasn't measured; the two candidates are measurement overhead and the CPU oversubscription above.
3. **Let phase 2 run to the cap rather than stopping it early.** The groups save only at the end, so stopping them at any point would have lost everything. Waiting past the planned end cost about $1.40.
4. **Stopped the training processes at 00:08,** four minutes before the cap, so `orch finish` could run; it refuses while a job is running. Nothing more could have been saved by then.
5. **Didn't rerun.** The definition's rerun clause needs time left on the same worker, and none was left. A new worker would exceed this experiment's budget, so that's the user's decision.
6. **Added `criteria.py`.** It computes the freeze/all half-time ratios that the location criteria need and that `analyze.py` doesn't report. It was tested only on a synthetic group file.
7. **Fact-check.** The fact-checker found five wrong timing or cost figures and several overstatements in the first draft of the README. All were corrected before the result was recorded; the phase-1 numbers were unaffected.

## What you need to do

- **Decide whether to rerun phase 2** on a new worker from the saved `results/phase1.json` and `results/phase1_params.pt`. The command is in the README. Before rerunning, fix how phase 2 sizes its step count, or it will likely overrun again:
  - make `ctn.ttn.train` / `studies.ttn.common.run_group` check the wall clock and save when the deadline arrives;
  - or pass a `--t-max` that fits at about 8 ms/step. Reaching t_max = 1e5 would take at least 4.3 h of phase 2 alone.

  Capping each group's torch threads to the container's quota would also test the oversubscription explanation.
- **The thread oversubscription affects every experiment on these workers.** `studies.ttn.checks` stalls without `OMP_NUM_THREADS`, and `run_parallel` sizes threads from `os.cpu_count()`, which ignores the cgroup quota. It's worth fixing in shared code, which I left alone.
- **The experiment log (`EXPERIMENTS.md`) and other shared files are untouched,** as instructed. `orch experiment done` recorded the result for the next `orch log` rebuild.
