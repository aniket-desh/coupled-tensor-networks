# Init scale: activation variance, small-init plateaus, and the depth law in the 15-core TTN

**This run failed, so it doesn't answer the depth-law question or the plateau calibration.** None of the six training groups finished. The training loop overran its wall-clock deadline because it never checks the clock, and a group saves only at the end, so all training data was lost when the worker had to be finished before its 5.5 h cap. The definition counts a run as failed when the alpha = 1, eta = 0.05 group didn't run, and it didn't run to completion. The part that did finish, the activation scale at init (Part A, descriptive), shows that the variance-controlled init keeps the mean second moment of every layer within 5% of 1 for r ≥ 8. Individual inits still vary widely at the output (5–95% band 0.29–2.22 at r = 8). It cost about $11.70 of the $12.05 budget.

## What finished: activation scale at init (Part A)

![Activation second moments](figures/activations.pdf)

Each number is the second moment of a layer's activations over 8192 random ±1 inputs, computed separately for each of 64 inits and then summarised over those inits (`results/activations.json`).

| r | leaf | quarter | half | output (mean) | output, 5% / median / 95% over inits |
|---|---|---|---|---|---|
| 4 | 1.00 | 1.04 | 1.07 | 1.35 | 0.17 / 0.75 / 4.20 |
| 8 | 0.99 | 0.98 | 0.97 | 0.97 | 0.29 / 0.75 / 2.22 |
| 16 | 1.00 | 0.99 | 0.97 | 0.96 | 0.49 / 0.86 / 1.50 |
| 32 | 1.00 | 0.99 | 0.99 | 0.97 | 0.67 / 0.95 / 1.44 |

The table is at alpha = 1. The mean stays within 5% of 1 at every layer for r ≥ 8, and within 3% everywhere except the r = 16 output (0.956). At r = 4 it drifts up toward the output, to 1.35, and the spread across inits is wide. The spread part of the definition's descriptive hypothesis holds at every r: the spread grows with depth (for r = 8 the 5–95% band of the leaf moment is 0.89–1.11 and of the output 0.29–2.22) and shrinks with r. Its other part, that every layer is about 1, holds for r ≥ 8 but not at r = 4. The median output moment sits below the mean (0.75 at r = 4 and r = 8), so the distribution over inits is right-skewed, and a typical init starts somewhat below unit output scale.

Scaling every core by alpha scales the layer moments as alpha^(2m), where m is the number of cores feeding the layer: 1 for a leaf, 3 for a quarter, 7 for a half and 15 for the output. This is an identity, not a measurement agreeing with a prediction: Part A uses the same inits for every alpha at a given r, and each layer is homogeneous in its cores. At r = 8 the alpha = 0.9 output moment divided by the alpha = 1 moment equals 0.9^30 to about 1e-16. The measured moments at alpha = 0.9 (0.80, 0.52, 0.22 and 0.041) differ slightly from 0.9^(2m) itself (0.81, 0.53, 0.23 and 0.042) only because the alpha = 1 moments aren't exactly 1. So at alpha = 0.8 the output starts at about 1e-3 of its alpha = 1 scale, and at alpha = 0.5 at about 1e-9.

## What didn't run

None of the three training questions has an answer:

- **Depth law (the primary criterion):** not evaluated. No per-seed half-times exist.
- **Discretization (eta/2 against eta):** not evaluated.
- **Plateau calibration per alpha, the alpha exponent, and c2-before-c3:** not evaluated.

The six groups (alpha = 1, 0.95, 0.9, 0.85, 0.8 at eta = 0.05, and alpha = 1 at eta = 0.025) started together at 19:24 UTC with a 4.25 h deadline. Each printed a measured step time of 6.1–10.2 ms and a step budget meant to end at the deadline (`results/worker-logs/jobs/20261112T192339-train/out.log`). For example, the alpha = 1, eta = 0.05 group measured 8.76 ms/step and budgeted 1.75e6 steps, which would reach t = 8.7e4. At 23:41 UTC, 4 h 18 min in, all six processes were still running at full GPU use and none had saved. They were still running with nothing saved at 00:13 UTC, 4 h 50 min in, when they were stopped so the worker could be finished before its cap at about 00:23 UTC.

The cause is in the deadline handling. `run_group` (`studies/ttn/common.py`) turns the deadline into a fixed step count from a 3-step timing taken at the start of each group, with 15% added for logging. `train` (`ctn/ttn.py`) then runs that many steps without looking at the clock. The real steps were at least about 14% slower than the printed per-step estimate, because the groups ran 4.83 h against a 4.25 h budget. That's at least about 31% slower than the raw 3-step timing before the 15% allowance. How much slower is unknown, because nothing reports progress, and the cause wasn't measured. Likely contributors are six groups contending for one GPU and the mobility measurement, which runs at every logged point while any of the 32 seeds is mid-escape. CPU thread oversubscription is also untested: each group calls `torch.set_num_threads(cpu_count // 6)`, which is about 21 threads per group and 126 in all on the container's 13.6-core quota. That's the same kind of oversubscription that hung the first checks run, although each process showing 99% CPU in `ps` argues against heavy throttling. The container blocks ptrace, so the loop counter couldn't be read either. Since a group writes its `.npz`/`.json` only when it finishes, a group that overruns the worker cap loses everything.

## How it was checked

- **Correctness checks passed** (`results/checks.json`). All 29 values that have a tolerance in `studies/ttn/checks.py` are inside it, and 21 more (init second moments) are recorded without one. The first attempt ran 27 minutes without finishing, using about 13.5 cores and no GPU. The sharing-location supervisor diagnosed the same hang on an identical worker: torch starts about 190 CPU threads against a 13.6-core quota. The rerun with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` finished in seconds. The thread cap changes no computed value beyond floating-point summation order.
- **CUDA was available** on the A100-SXM4-80GB (torch 2.11.0+cu128).
- **Part A was computed on the GPU in float64** with the committed code. `results/source/` is the snapshot that `run.py` copied at startup.
- **The Part A figure was regenerated** on the coordinator by `plot.py` from the saved `activations.json`. The worker's analyze and plot steps ran after training was stopped, and both failed for lack of group files; the plot step had written the Part A figure first. Only `results/` and the job logs were copied back from the worker. The figure wasn't inspected visually; the table above comes from the same file.
- A fact-checker agent checked this write-up against the result files, the logs and the definition. It found 11 of 60 claims that didn't fully hold, and they're corrected here. The corrections are: the layer means are within 5% of 1, not 3%; the r = 4 means aren't about 1; there are 29 checks with tolerances, not 51; the cause of the slowdown wasn't measured; the alpha^(2m) agreement is an identity; and the time margins. It also added the point that even a clock-honouring rerun would be censored.

## Judgement calls

- **The training deadline was cut from 4.5 h to 4.25 h** before training started. Launching took a second attempt 10 minutes after the first found no A100 in stock, and the hung checks used another 27 minutes. The estimate at the time was that 4.5 h would leave only about 20 minutes for Part A, analysis and publishing before the cap. From the worker's records it would have been about 30, and 4.25 h left about 45. In the end this didn't matter, since the groups didn't stop at either deadline.
- **Training was left alone once it overran**, rather than paused selectively. Pausing the descriptive groups (alpha < 1) would have given the two alpha = 1 groups more of the GPU, but there was no way to tell which process ran which group, and no way to tell whether even a 3x speed-up would get them done in time.
- **The worker was finished 10 minutes before its cap**, which meant killing training. That kept the Part A results, checks and logs, which a self-delete at the cap might not have published.
- **Results were copied over SSH.** `orch finish` failed because the worker's mount of the shared results volume had dropped ("Transport endpoint is not connected"). `results/` and the job logs were copied straight to the coordinator with tar over SSH, and the worker was deleted with `orch down`.
- **`analyze.py` and `plot.py` now stop cleanly after Part A** when no group results exist, instead of raising on an empty list. That's a three-line guard in `analyze.py` and a two-line guard in `plot.py`, and nothing else in the analysis changed.

## What to do next

Before rerunning, `train` should check the wall clock and stop at the deadline, and a group should save on a timer, so that an overrun loses minutes rather than the whole run. Both changes are in shared code (`ctn/ttn.py`, `studies/ttn/common.py`), so they're for the user to make. The step-time measurement should also be taken once all groups are running, and the per-group thread count should respect the container's CPU quota rather than `os.cpu_count()`.

Even with those fixes, a rerun of the definition as written would stop every group at its deadline. The measured 6.1–10.2 ms/step is 2–5 times the definition's 2–5 ms estimate, and every group's printed step budget was below the steps it wanted. The alpha = 1, eta = 0.05 group could reach t ≤ 8.7e4 of its t_max = 1e5, and the eta/2 discretization control only t ≤ 4.8e4. That's probably enough for the depth-law test if most seeds escape early (the CPU pilot had 8/16 learned by t = 5000). The discretization pairing would cover only seeds that escape before t ≈ 4.8e4. Before rerunning, decide whether to accept that, or to drop the small-alpha groups so the two alpha = 1 groups get the GPU.
