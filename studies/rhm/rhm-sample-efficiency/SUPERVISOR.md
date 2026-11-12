# Supervisor report: rhm-sample-efficiency

The experiment ran to completion as defined. All 42 planned runs finished, both hypotheses are supported, and it cost about $0.63 of the $8 budget. There were no amendments, no reruns and no failures to handle, and no worker of this experiment is left running.

## What ran

Worker `jed7ufv5yzfyr5` was an NVIDIA A40 in EU-SE-1 at $0.59/h, created at 21:43 UTC and deleted at 22:48 by `orch finish`. The steps came from the definition's Commands, in order, each as its own `orch exec` step:

1. `nvidia-smi` and the CUDA check passed (torch 2.11.0+cu128 on the A40).
2. `studies.rhm.checks` passed all 26 checks ([results/checks.json](results/checks.json)).
3. The training step (`run.py --hours 5.5`) took 59 minutes. Both tasks' LR pilots chose 3e-3, the top of the grid, which the run records as a grid-edge choice. All 42 runs then finished: 22 early-stopped after fitting their training data, 14 ran the full 100,000 updates, and the 6 online references ran to their 20,000-update cap. None diverged, crashed or was skipped.
4. `analyze.py` and `plot.py` wrote results/analysis.json, results/curves.json, figures/vs_N.pdf and figures/curves.pdf.

`orch finish` published the run directories (381 files, 0.12 GB, under `runs/rhm-sample-efficiency/`, which is not in git) and committed the results folder as b844cf2.

## Result against each criterion

- **H1, root-task sample complexity near P*: supported.** The median root test accuracy over 3 grammars is 0.084 at 2^10 (below 0.5) and 0.99996 at 2^14 (above 0.9). At 2^12 = P* it is still 0.152, so the threshold lies between P* and 4 P*.
- **H2, large-N failures are data-limited: supported.** None of the 18 runs with N ≥ 2^16 is optimization-limited, because every one of them learned. All 12 runs that failed (N ≤ 2^12, both tasks) fit their training sets.
- **Run failed?** No. The checks passed, CUDA was available, and all 6 N = 2^20 runs finished.

The details, caveats and next steps are in [README.md](README.md). The most important caveats are that the threshold is resolved only to a factor of 4, that both LRs sit at the edge of the pilot grid, and that "learned" is a loose bar for the last-token task.

## Cost

One A40 ran for 1 h 04 min and cost about $0.63, against a cap of 7 h and $8. The experiment log has the exact entry.

## Decisions I made

- **GPU.** The launcher placed the run on an A40, the second GPU type in the definition's list, because L40S had no stock outside the coordinator's data center and RTX A6000 returned "no instances available". I accepted it, since it is in the definition's list and well within budget.
- **Running the checks.** I ran the GPU, CUDA and checks steps from one background command chained with `&&`, so a failure would stop the chain. Training ran as its own step, as the definition asks.
- **Monitoring.** While training ran, I read the training log four times with small `orch exec ... --name peek` jobs (`tail` of the log, plus `nvidia-smi` once). These appear as extra jobs in the experiment log, and they did not touch the run.
- **No reruns.** Nothing crashed, diverged or was skipped, so none of the definition's failure branches applied.
- **Write-up.** The fact-checker found 11 claims that didn't fully hold, and I corrected all of them. The README's "How it was checked" lists what they were.
- **Plots.** I left `plot.py` unchanged. Two of its features are explained in the README's caption instead of fixed: it clips training KLs below 10^-5 to 10^-5, and its Bayes line shows only grammar 0's values.

## What you need to do

- The experiment log (`EXPERIMENTS.md`) was updated by `orch`, but I didn't commit it, because the instructions leave it to the planning session. `orch experiment done` has recorded the result there, and committing it is yours.
- Optional: the natural follow-up is a finer root-task grid at N = 2^12.5, 2^13 and 2^13.5 to locate the threshold, plus an LR pilot extended to 1e-2, since both tasks picked the top of the current grid. Each is well under an hour of GPU time.
