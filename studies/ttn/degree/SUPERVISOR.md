# Supervisor report: degree rerun (2026-11-12 21:43 to 2026-11-13 02:00 UTC)

The rerun produced a supported result, but only after I killed the first training job and restarted it with a shorter horizon (t = 3e4 instead of 1e5) so that it could save before the worker's hard cap. The degree slopes are positive and ordered as predicted. Their size is 35 to 41% above the prediction, so the quantitative-agreement criterion is not met. The worker is deleted, and it cost about $8.81 of the $12.05 budget.

## What ran

All on worker `qmwn3avebevbu8` (A100-SXM4-80GB, $2.19/h), job `20261112-214354-degree-qmwn3avebevbu8`.

1. **Pre-training steps** (21:44 to 21:45 UTC): `nvidia-smi`, the CUDA check, and `studies.ttn.checks`. Every check passed, including the alignment check that failed the first attempt (now 1).
2. **First training job** (`20261112T214538-train`, 21:45 to 23:36), as the definition wrote it: 12 groups, 8 at a time, t_max 1e5, 4.5 h deadline. I killed it at 23:36 after one group (w = 0.25, D = 8) had saved. The reasons are under the first decision below.
3. **Restarted training** (`20261112T233729-train-rerun`, 23:37 to 01:44): all 12 groups at once with `--t-max 3e4 --hours 4 --procs 12`, writing to `results/rerun`. Every group reached t = 3e4 in 7533 to 7577 s, about 12.6 ms per step, and no run diverged.
4. **Analysis and plots** on `results/rerun` (01:44), then `orch finish` (about 01:46). `orch ls` showed no worker of mine left.
5. **Write-up** in `README.md`, then a fact-checker pass, which found 16 claims to correct. All 16 were applied before `orch experiment done degree --result supported` and the Outcome section were filled in.

I also ran four small diagnostic jobs on the worker while training ran. Two (`probe`, `probe2`) timed the training step under load, and two (`peek`, `peek3`) looked at GPU use and the saved files. Two more (`pyspy`, `pyspy2`) tried to read the training processes' step counters with py-spy, which failed because the container doesn't allow ptrace. These jobs wrote only to `/tmp` on the worker, not into the results.

## Result against each criterion

| criterion | result |
|---|---|
| Pre-training checks | passed |
| Symmetry control, w = 1/9, D = 1 vs 8, p > 0.01 | passed: p = 0.97, 13/16 learned in each |
| Supported: slope interval above 0 at both w, and the w = 0.5 slope larger, over D with at least 12/16 learned (at least 3 such D) | met: D = 1, 2, 4, 6 qualify at both w; 0.600 (0.469 to 0.730) at w = 0.25, 1.214 (1.092 to 1.333) at w = 0.5 |
| Quantitative agreement (reported only) | not met: predicted 0.425 and 0.901 are outside both intervals |

D = 8 is out of both fits, because by t = 3e4 only 11/16 seeds had learned at w = 0.25 and 4/16 at w = 0.5. The full table and the exploratory analysis are in `README.md`.

## Cost

4 h 01 m of the A100 at $2.19/h, about $8.81 of the $12.05 budget, inside the 5.5 h cap. About 1.8 h (about $4) went to the killed first training job, and the only thing it produced is the w = 0.25, D = 8 group in `results/`, which isn't used for the criteria. The first attempt of this definition (failed checks) cost a further $0.12 earlier.

## Decisions I made

1. **Killed the first training job and restarted all 12 groups at once with t_max = 3e4** (amendment recorded in `EXPERIMENT.md` before acting). Each group sets its step budget from a timing taken at startup, `train()` never checks the clock, and a group saves only when it ends. The one group that finished ran at 9.1 ms per step, against the 5.6 to 7.1 ms that the other first-wave groups had budgeted 2e6 steps on. At that rate those groups would have ended around 02:50 UTC, minutes before the cap at about 03:14, which risked losing everything. The four queued groups, including the isotropic control, could not have trained in time. The sharing-location and init-scale supervisors hit the same problem tonight, and init-scale lost all its training data to it. Running the 12 groups together gives every group, the control included, the same censoring point, and setting t_max (instead of relying on the startup timing) made that point predictable. I picked 3e4 so that 6e5 steps at an expected ~14 ms per step would end around 02:00 with margin. They ended at 01:44.
2. **Analysis and plots on `results/rerun`.** The first job's w = 0.25, D = 8 group has a different horizon (3.5e4), so I kept it in `results/` for reference and left it out of the criteria. The write-up mentions, as an unregistered check, that including it leaves the w = 0.25 slope's conclusion unchanged.
3. **Did not run the queued groups alongside the first job.** Earlier, I considered starting the control groups early in a separate process. I didn't, because `run.py` would later have run them again and overwritten them. The restart in decision 1 made the question moot.
4. **Added an exploratory section** to the write-up that splits the slope into the per-patch shrinkage of the initial coefficient and the dependence of escape time on that coefficient. It's labelled as exploratory. Its main finding is that t_half still rises with D at a fixed initial coefficient.

## What the user should do

- **Commit the experiment log.** `orch` updated `EXPERIMENTS.md` (the entry for `20261112-214354-degree-qmwn3avebevbu8` now says "supported"). I left it uncommitted, as instructed. `orch finish` committed the results and figure (`cb25cc9`), and the write-up, amendment and outcome are in `4347b47`. Nothing is pushed.
- **Fix the training budget in shared code before the next long TTN run.** `studies/ttn/common.py` sets step budgets from a startup timing that can be 1.5 to 4 times off when several groups share a GPU, and `ctn/ttn.py` `train()` has no wall-clock check and saves nothing until the end. A wall-clock stop inside `train()`, or periodic saves, would have avoided both restarts tonight. I didn't touch shared code.
- **Optional follow-ups,** both in `README.md` under "Not yet checked". Rerunning the low-D groups at eta = 0.025 (under an hour) would test whether discretization explains part of the gap to the prediction. The planned t = 1e5 horizon (about 7 h, about $15, over this definition's cap) would very likely bring w = 0.25, D = 8 into the fit.
- **Note on the depth-15 law.** The prediction assumes the escape law that init-scale was meant to test, but init-scale lost its training data, so that law is still untested in this network.
