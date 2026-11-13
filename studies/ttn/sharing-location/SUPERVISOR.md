# Supervisor report: sharing-location

The experiment produced no answer. Only one of its six population families (leaf) saved before the worker's 5.5 h cap. The no-sharing control and the two key families (quarter, half) have no results, so no success criterion could be evaluated. It's recorded as `failed`. The correctness checks passed, the worker is deleted, and the spend was about $11.7 of the $12.05 budget. The write-up is [README.md](README.md) and the verdict per criterion is in the "Outcome" section of [EXPERIMENT.md](EXPERIMENT.md).

## What ran

All times are UTC on the worker `ime06c8z4vsp5u` (one A100 SXM 80GB in US-MD-1, $2.19/h, created 2026-11-12 18:41:52, hard cap at about 00:11:50).

1. **18:42, the GPU and CUDA check:** passed, torch 2.11.0+cu128.
2. **18:43, correctness checks, first attempt:** no output after 11 minutes. Killed (decision 1).
3. **18:54, correctness checks, rerun with `OMP_NUM_THREADS=8`:** all passed in about 4 minutes, giving `results/checks.json`.
4. **18:55, training as defined:** seven groups in parallel, `--hours 4.5`. The sampled group was killed at 22:26 (decision 2) and the whole attempt at 22:52 (decision 3). Nothing was saved.
5. **22:52, rerun of the six population groups:** `--hours 0.85` into `results/rerun`. Leaf saved at about 00:01. The job was stopped at 00:02 (decision 4).
6. **00:02, `orch finish`:** published, collected and merged `studies/ttn/sharing-location` (commit 7701009), then deleted the worker.
7. **Coordinator:** `analyze.py` and `plot.py` ran on `results/rerun`, giving `analysis.json` and `figures/families.pdf`. Then the write-up, a fact-check, `orch experiment done --result failed` and the Outcome section, committed in fb9f587.

## Result against each criterion

- **Supported, partly supported, not supported:** not evaluable, because none, quarter and half have no results.
- **Mechanism (K22 against none, drive fractions):** not evaluable against none. For the 2 leaf seeds that learned c2, the drive fractions were own 5.48, cross from y1 0.00, cross from y3 0.00, rest -4.48 and resolved 1.00. That matches the CPU validation pattern, but it's descriptive only.
- **Location (quarter vs split2, half vs split4):** not evaluable.
- **Sampled replicate:** stopped on purpose, so there are no results.
- **Inconclusive:** not evaluable for the three named families. Leaf learned c2 in 2 of 16 seeds by t = 1.78e4.
- **Run failed (as the definition defines it):** doesn't strictly apply, since the checks passed and a population family ran. `failed` was used in the toolkit's sense of no answer.

## Cost

The worker ran about 5.35 h at $2.19/h, about $11.7, against the $12.05 budget and inside the 5.5 h cap. API spend for this session (the fact-checker agent) isn't metered here.

## Decisions I made

Each of these is also under "Amendments" in EXPERIMENT.md.

1. **Checks rerun with a thread cap (18:54).** The container has a 13.6-CPU quota but reports 128 CPUs, and `checks.py` imports torch before `ctn.io` sets `OMP_NUM_THREADS=1`. So torch ran 193 threads and stalled. I reran the checks with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. This changes only the thread count.
2. **Killed the sampled replicate (22:26).** At 3.5 h no population group had saved, although split2 was due at about 3.3 h. `nvidia-smi pmon` showed the sampled process taking about 66% of SM time. Its deadline budget only reached t ≈ 3100, where c2 would have been almost entirely censored. I chose to sacrifice it for the primary families.
3. **Killed the whole first attempt and restarted the population groups with `--hours 0.85` (22:52).** Still nothing had saved at 3.9 h, and the worker allows no ptrace, so progress couldn't be read. `train()` has no wall-clock check, and a group saves only at its end. Assuming a 2–3x slowdown, every family would have run past the cap and been lost.
   - In hindsight, with the 1.34x slowdown the rerun later measured, split2, split4 and leaf would probably have finished the first attempt at t = 1e5, though still without the `none` control. This is the decision most open to question.
4. **Stopped the rerun and published at 00:02.** The rerun also overran its step-count deadline: leaf ran 11.6 ms per step against 8.6 ms measured at startup. Leaf saved at about 00:01. The worker's own final automatic publish (23:56:51) had come before that, and the pod deletes itself at the cap. So I stopped the job with about 9 minutes left so leaf's results would survive. At leaf's speed, the other groups would have needed until about 00:17–00:31.
5. **Analysis and plots on the coordinator.** I ran `analyze.py` and `plot.py` on the coordinator rather than the worker, to save worker time at the end. They read only saved files, so the outputs are the same.
6. **Outcome label `failed`.** None of the definition's classes fits a run where only a non-control family saved. The loop instructions say to record `failed` after two unsuccessful attempts at an uncovered failure, and the toolkit defines `failed` as "no answer".

## What you need to do

- **Decide whether to rerun, and fix the runner first.** The rerun needs three changes before it is worth spending on:
  - a wall-clock deadline check inside `ctn/ttn.py:train()`, so a group saves at its deadline;
  - periodic checkpoints of the training log;
  - the sampled replicate on its own worker, or with MPS, so it doesn't share time slices with the population groups.

  Even then, 2e6 steps at 11.6 ms per step is about 6.4 h for six concurrent groups, so either split the families across two workers or lower t_max to at most about 7e4. These are changes outside this folder, so I didn't make them.
- **Fix `studies/ttn/checks.py` (optional).** Import `ctn.io` before torch, or set the thread count, so the checks don't stall on many-core workers. This is also outside this folder.
- **Update the experiment log.** I didn't edit `EXPERIMENTS.md`, as instructed. `orch experiment done` may have touched it, and it shows as modified in the working tree alongside other supervisors' changes.
- Nothing is left running: `orch ls` shows no sharing-location worker.
