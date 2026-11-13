# Sequential transfer: add output 2 after output 1

**Result: failed. The run produced no answer to the transfer question.** Phase 1 finished: 15 of 16 seeds learned output 1. But none of the six phase-2 groups (none, quarter and half, each at s2 = 0.5 and 1.0, with 64 runs per group) had finished when they were stopped at 00:08 UTC, four minutes before the worker's 5.5 h cap. A group saves its results only when it finishes, so no phase-2 data exists.

The transfer and location criteria could not be evaluated: there is no `analysis.json`, no `figures/sequential.pdf` and no per-arm table. Phase 1 passed its gate (15 of 16, against the inconclusive floor of 10). The saved phase-1 state (`results/phase1.json` and `results/phase1_params.pt`) is enough to rerun phase 2 alone on a new worker.

## What exists

| Output | Status |
|---|---|
| `results/checks.json` | All checks within tolerance (`failed: []`), including the ones for frozen cores and masked outputs |
| `results/phase1.{json,npz}`, `results/phase1_params.pt` | Complete |
| `results/{none,quarter,half}_s2{0.5,1.0}.*` | Not produced |
| `results/analysis.json`, `figures/sequential.pdf` | Not produced, because both need the phase-2 files |

**Phase 1** trained output 1 alone (y1 = prod_j x^(j)_1) from 16 inits (seed 4000) with exact population GD at eta = 0.05 on an A100 SXM:

- It stopped on its wall-clock share (45% of the 4.5 h training deadline) at t = 67,077, short of t_max = 1e5. That took 7,697 s, or 5.74 ms/step, against the 5.43 ms/step estimate its budget was sized from.
- **Gate:** 15 of 16 seeds ended with |c1 - 1| < 1e-2, and the largest deviation among them was 2.3e-4. Seed index 8 never escaped: it ended with c1 = 0 and loss 0.5, so it would have been excluded from the all, freeze_q1 and freeze_left comparisons.
- **Half-time of c1** over the 16 seeds, with the unescaped seed counted as +inf: median 10,291, bootstrap 95% interval [4,351, 25,521]. Across the 15 escaped seeds it ranged from 1,801 to 40,066. Escape times are heavy-tailed, as the CPU pilot suggested: 3 of the 15 escaped after t = 37,000.
- The runner's stop rule (|c1 - 1| < 1e-3 and loss < 1e-8) counted 11 of 16 runs as censored. That reflects the strict loss threshold: 10 of the 11 had escaped and passed the gate, with final losses between 1.5e-8 and 7.4e-8.

## Why phase 2 didn't finish

The phase-2 groups started at about 21:04 UTC and ran together on the one GPU. Each group fixes its step count at the start: it divides the remaining wall-clock time (about 2.36 h, since phase 1 overran its share) by a step cost measured from three gradient steps, plus a 15% margin for logging. Those timings, taken while all six groups were starting at once, gave 5.72 to 5.96 ms/step and budgets of 1.43e6 to 1.49e6 steps.

The training loop (`ctn.ttn.train`) then runs that fixed number of steps and never checks the clock. When the groups were stopped at 00:08 UTC, about 3.05 h in, none had finished. So the real cost was at least 7.4 to 7.7 ms/step, at least 29% over each group's estimate, margin included. Phase 1 also overran its share, by 5.6% (7,697 s against about 7,290 s).

The cause wasn't measured. One timing taken while all six groups shared the GPU argues against GPU contention: at 21:08, a seventh process doing the same bare gradient step (200 steps) ran at 5.0 ms/step. That matches the groups' own start-up timings with the 15% margin removed (5.0 to 5.2 ms/step). Two explanations remain:

1. **Measurement cost during training.** The groups record c every 10 steps, and K by subtree (the coupling between the two outputs' gradients) densely while an output is rising. With 64 runs per group rising at different times, those dense measurements could be running for much of phase 2.
2. **CPU oversubscription.** The thread-cap amendment set `OMP_NUM_THREADS=8`, but each phase-2 group process overrides it with `torch.set_num_threads(cpu_count // procs)` (`studies/ttn/common.py:179`). On this container that is 255 // 6 = 42 threads per group, 252 in total, against a 27.2-CPU quota. Throttling was measured only at the start of phase 1, not during phase 2.

## How it was checked

- **Correctness checks** passed on the worker (`results/checks.json`). The first attempt stalled for about 11 minutes: the container reports 255 CPUs but has a 27.2-CPU quota, so torch started about 320 threads, and the container was throttled in 6,436 of 7,118 scheduler periods. Rerunning with `OMP_NUM_THREADS=8` took 8 s on the worker. This is recorded under the definition's Amendments and doesn't affect any value.
- **Training was alive throughout.** Spot checks at 21:05, 21:11 and 22:08 showed the six group processes running and the GPU at 99–100% utilization. The training job was still running at every per-minute poll from 23:34 until it was stopped. During a 10 s window at the start of phase 1, the throttle count (`nr_throttled`) didn't increase.
- **Fact-check:** a fact-checking agent compared this write-up against the result files, the job logs, the experiment log and the session transcript. It found five wrong timing or cost figures: phase 1's step time, the time left for phase 2, the overrun percentage, the stall length, and the cost of waiting. It also found several overstated claims. All were corrected, and it suggested adding the CPU-oversubscription explanation above. None of the phase-1 numbers changed.

## Judgement calls

- **Letting phase 2 run to the cap rather than stopping it early.** The groups hold their logs in memory, so stopping them at any point loses all phase-2 data. Waiting was the only option with a chance of producing results. The cost of waiting was about $1.40 of worker time after the planned end of about 23:30.
- **Stopping the training processes at 00:08 UTC,** four minutes before the cap. `orch finish` refuses to run while a job is running, and nothing new could have been saved and published in the remaining minutes. Stopping them let `orch finish` merge the phase-1 results, already published by the automatic publish at 23:57, and delete the worker four minutes early.
- **No rerun.** The definition's rerun clause covers a crash on the same worker with time left (`--hours <left - 0.75>`). No time was left, and the worker had used $11.90 of its $12.05 budget, so a rerun on a new worker is the user's decision.
- **Added `criteria.py`.** `analyze.py` reports each arm's median half-time and the paired family sign test, but not the freeze/all ratios that the location criteria need. `criteria.py` computes them as the median of per-seed paired ratios, with censored runs counted as +inf and seeds where both arms are censored dropped. It also reports the ratio of medians and the censored fraction. It was tested on a synthetic group file and has not been run on real data.

## What to do next

Rerunning phase 2 from the saved phase 1 needs a new worker:

```bash
PYTHONPATH=. uv run --frozen python studies/ttn/sequential/run.py --out "$EXP_DIR/results" \
  --groups none_s20.5,none_s21.0,quarter_s20.5,quarter_s21.0,half_s20.5,half_s21.0 --hours H
```

This needs `results/phase1.json` and `results/phase1_params.pt` in the output folder. The slowest group cost at least 7.7 ms/step, so reaching phase-2 t_max = 1e5 (2e6 steps) would take at least 4.3 h. The runner still sizes its budget from a 3-step timing, so a rerun could overrun again. Either of two changes fixes that, in shared code (`ctn.ttn.train` / `studies.ttn.common.run_group`):

- make training check the wall clock and save when the deadline arrives;
- or pass a `--t-max` that fits at about 8 ms/step.

Capping each phase-2 group's torch threads at the container's quota (about 27 / 6) would also test the oversubscription explanation.
