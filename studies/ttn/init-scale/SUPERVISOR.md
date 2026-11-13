# Supervisor report: init-scale

The run failed. All six training groups were still running when the worker had to be finished, and since a group saves only at the end, no training data exists. The depth-law test, the discretization control and the plateau calibration are all unanswered. What did come back is the passing correctness checks, the Part A activation results, the source snapshot and every job log. The worker cost about $11.70 of the $12.05 budget, and no worker of this experiment is left (`orch ls` at 00:21 UTC lists only rhm-dynamics and degree). The write-up is [README.md](README.md), and the verdict per criterion is in the definition's Outcome.

## What ran

All times are UTC.

1. **18:39, launch: nothing placed.** Neither A100 type was in stock outside US-NC-1, because the sibling experiments had just taken them.
2. **18:53, second launch.** It placed worker `mh13yvbrmty9uz`, an A100-SXM4-80GB in US-MO-1 at $2.19/h.
3. **18:55, `nvidia-smi` and the CUDA check passed.**
4. **18:55, checks hung.** They ran 27 minutes at about 13.5 cores and 0% GPU, then were killed (exit 143).
5. **19:22, checks rerun with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.** They passed in seconds.
6. **19:23, training (`run.py --hours 4.25`).** Part A finished first, then six groups started, printing 6.1–10.2 ms/step.
7. **23:41, training past its deadline.** The supervisor found it still running with nothing saved, and it was still that way at 00:12.
8. **00:13, the end.** Training was killed, and the queued analyze and plot steps ran and failed for lack of group files. `orch finish` failed because the worker's global-volume mount had dropped (Errno 107). `results/` and the job logs were copied over SSH, and the worker was deleted with `orch down`.
9. **Coordinator, afterwards.** `analyze.py` and `plot.py` were run on the coordinator for Part A only, the write-up was drafted, the fact-checker ran (11 of 60 claims corrected), the result was recorded as failed, and the folder was committed.

## Result against each criterion

- **Run failed:** yes. Checks passed and CUDA was available, but the alpha = 1, eta = 0.05 group didn't run to completion.
- **Depth law (primary):** not evaluated.
- **Discretization:** not evaluated.
- **Part A (descriptive):** at alpha = 1 the mean second moment of every layer is within 5% of 1 for r ≥ 8. At r = 4 the output mean is 1.35, with a wide spread over inits. The spread grows with depth and shrinks with r.
- **Alpha exponent, calibration, c2-before-c3:** not evaluated.

## Cost

The single A100-SXM4-80GB ran 5.35 h at $2.19/h, about $11.70, against a budget of $12.05 and a cap of 5.5 h. About 1 h of that went to failures: the checks hang took 0.45 h, and the last 0.57 h was training that couldn't be saved. The coordinator-side work cost nothing extra.

## Decisions

1. **Retried the launch once after 10 minutes** instead of stopping at the first "no A100". The launcher had retried for only about a minute, and the definition said it "waits and retries". A second failure would have meant reporting and stopping.
2. **Reran the checks with a thread cap** (amendment 1). The sharing-location supervisor had already diagnosed the same hang on an identical worker. The same prefix went on the later commands.
3. **Cut the training deadline from 4.5 h to 4.25 h** (amendment 2), to leave time for analysis and publishing before the cap after the late start. The estimate at the time was a 20-minute margin; the records show it would have been about 30.
4. **Left training alone when it overran, and finished at about 00:13** (amendment 3). The progress of each group couldn't be read (no ptrace), and processes couldn't be matched to groups, so there was no informed way to pause some to speed up others.
5. **Killed the training processes so the job would end**, because `orch finish` refuses while a job runs, and a `--checkpoint` publish would have kept the worker alive.
6. **Copied results over SSH and deleted the worker with `orch down`** after `orch finish` failed on the dropped volume mount. Waiting for the worker to delete itself at the cap would have lost the files, since its 00:08 auto-publish had already failed on the same mount.
7. **Added guards to `analyze.py` and `plot.py`** (3 and 2 lines) so they write the Part A outputs and exit when no group results exist. Nothing else in the analysis changed.
8. **Recorded the result as `failed`, not `inconclusive`**, because the definition's run-failed criterion names the alpha = 1, eta = 0.05 group.

## What you need to do

- **Fix the training deadline in shared code before any rerun of this or a sibling experiment that uses `run_group`.** `train` in `ctn/ttn.py` should stop at a wall-clock deadline, and a group should save on a timer, so an overrun costs minutes instead of everything. `run_group` in `studies/ttn/common.py` should measure step time with all groups running, and set per-group threads from the CPU quota (`/sys/fs/cgroup/cpu.max`) rather than `os.cpu_count()`, which reports 128 on these workers. The other TTN experiments in this batch use the same code, so check their results for the same overrun.
- **Decide how to rerun.** At the measured 6–10 ms/step, a rerun that honours the clock still censors every group. alpha = 1 reaches t ≈ 8.7e4 and the eta/2 control only t ≈ 4.8e4. Dropping the alpha < 1 groups would give the two alpha = 1 groups more of the GPU.
- **Merge or discard the experiment log.** `orch experiment done` added the init-scale result to `EXPERIMENTS.md`, which this session left uncommitted as instructed. The log also still shows the pre-finish state, because `orch finish` never completed. Run `orch log` to rebuild it if needed.
- **The global-volume mount dropped on this worker** (Errno 107 at the 00:08 auto-publish and at `orch finish`). A run with large results couldn't have been rescued over SSH as easily, so this may be worth raising with the toolkit.
- **`studies/ttn/init-scale/results/` is committed through git** (6.2 MB, mostly the source snapshot's `uv.lock` and job logs). It never came back through the toolkit's data channel.
