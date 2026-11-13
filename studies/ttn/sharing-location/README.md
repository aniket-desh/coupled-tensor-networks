# Sharing location: run incomplete, no criterion could be evaluated

The run didn't answer its question. Of the six population families (agreement sets none, leaf, split2, quarter, split4 and half), only `leaf` saved before the worker's 5.5 h cap. The no-sharing control `none` isn't among the saved results, so none of the seed-paired comparisons, the shared-cores trend, the mobility test or the location tests can be computed. The finite-data replicate was stopped on purpose and has no results either. The correctness checks passed. The one saved family is described below as a record. It is not evidence for or against the hypothesis.

None of the definition's outcome classes applies. Its "Run failed" class needs failed checks, no CUDA, or no population family, and here the checks passed and one family finished. Every other class needs none, quarter or half. The result is recorded as `failed` in the toolkit's sense: the run produced no answer.

## What exists

`results/rerun/leaf.{json,npz}` holds the leaf family: S = {0}, one shared core, 16 seeds, exact population gradient descent at eta = 0.05. It trained to t = 1.78e4 (3.55e5 steps), the step budget its startup timing allotted for the 0.85 h deadline, but it actually ran 4106 s, past that deadline. `results/rerun/analysis.json` and `figures/families.pdf` come from the unchanged `analyze.py` and `plot.py`, run on the coordinator after collection. The paired figure was skipped because there's no `none` group.

| leaf, 16 seeds, t <= 1.78e4 | c1 (s = 1) | c2 (s = 0.5) | c3 (s = 0.75) |
|---|---|---|---|
| seeds that reached s_k / 2 | 10 | 2 | 5 |
| median half-time (95% bootstrap interval) | 5635 (2952 to inf) | inf (fewer than half learned) | inf (interval from 8061) |

The half-times are censored at t = 1.78e4, far short of the planned t = 1e5. The pilot already expected many seeds to need longer than 2e4. At the end of training, 15 of the 16 seeds had learned exactly one of the three components and one seed had learned c1 and c3; no seed had learned all three.

For the two seeds that learned c2, the drive decomposition over c2's rise (from c2 / s2 = 0.02 to the half-time, as a median fraction of the actual rise) gave:

- own mobility K22 e2: 5.48;
- cross term from y1: 0.00;
- cross term from y3: 0.00;
- the pull of non-teacher coefficients: -4.48;
- resolved: 1.00, inside the 0.9 to 1.1 band the definition requires before trusting these fractions.

This matches the pattern seen in the CPU validation, which had own at about 4 to 8x the rise, the rest at about -3 to -7x and cross terms near 0. With n = 2 and no control, it says nothing about sharing. The median own mobility K22 at c2 / s2 = 0.02 was 3.9e-3, spread almost evenly over the four quarter subtrees (17–21% each).

## Why the run didn't finish

The first training attempt ran all seven groups, the six population families plus the sampled replicate, concurrently on one A100, each in its own process. Each group sets its step budget from a timing at startup: three gradient evaluations plus a flat 15% for logging and measurements. Nothing checks the clock during training, and a group saves only when it ends.

The population groups ran at least 1.2 times slower than their startup timing of 6 to 8 ms per step: split2, due at about 3.3 h, hadn't saved at 3.96 h. `nvidia-smi pmon` showed the sampled group (2^17 inputs per run, 261 ms per step) taking about 66% of SM time, so it was killed at 3.5 h. At 3.9 h still nothing had saved. Groups wanting the full 2e6 steps might have ended after the hard cap, and any group still running then would have lost its results, so that attempt was killed too.

The six population groups were rerun with a 0.85 h deadline, which gave budgets of t = 1.78e4 to 2.53e4. The rerun had no sampled group, but it also ran slow: leaf took 11.6 ms per step against a startup timing of 8.6 ms, about 1.34 times slower. So how much of the first attempt's slowdown the sampled group caused isn't known. The adaptive logging and mobility records during rises are a likely cause of the general underestimate, but that wasn't measured. Leaf saved at about 00:01 UTC. The job was stopped at about 00:02 so the results could be published before the cap at about 00:11, and the other five groups hadn't saved by then.

The checks rerun, the sampled group's stop and the restart are recorded under "Amendments" in [EXPERIMENT.md](EXPERIMENT.md), along with the stop at 00:02.

## How it was checked

- **Correctness checks** (`results/checks.json`): all passed on the worker. These cover:
  - the core contraction against the dense tensor;
  - autograd against finite differences;
  - mobility summing to the full inner product;
  - the population loss against Monte Carlo;
  - the basis-permutation symmetry.

  The first attempt at the checks was killed after 11 minutes, before it produced output, because of thread oversubscription. It was rerun with `OMP_NUM_THREADS=8`, which changes only the thread count, not what is computed.
- **Fact-check:** the fact-checker agent compared this write-up against the result files, the code snapshot, the job logs and EXPERIMENT.md. It corrected the outcome label, the causal account of the slowdown, the time leaf saved, and an unsupported claim that the other groups were minutes from saving. Its corrections are applied.

## Judgement calls

- **Killing the sampled replicate at 3.5 h.** It could only have reached t ≈ 3100 by its deadline, where the pilot learned c2 in about 3 of 16 seeds even by t = 5000, so its main ratio would have been nearly all censored. Keeping it would have left it taking about 66% of SM time from the population groups.
- **Restarting rather than waiting at 3.9 h.** I had no way to read progress, because the worker allows no ptrace, and I assumed a slowdown of 2–3x, at which every group would have run past the cap.
  - At the rerun's measured 1.34x, split2, split4 and leaf would probably have finished the first attempt before the cap at the full t = 1e5, while quarter, none and half would not.
  - So the restart traded three likely families at t = 1e5, still without the `none` control, for an attempt at all six at t ≈ 2e4, which also didn't finish.
- **Stopping the rerun at 00:02.** The worker makes one automatic publish 15 minutes before its cap (23:56:51 here), before leaf had saved, and doesn't publish again. So the manual publish was the only way leaf's results survived. Nothing in the saved data shows the other groups would have saved before the cap: at leaf's 11.6 ms per step, their larger budgets would have needed 5000–5900 s and ended after it.
- **Running `analyze.py` and `plot.py` on the coordinator** after collection, instead of on the worker. They read only saved results, so the outputs are the same.

## What to do next

A rerun needs three code changes before it can use this design:

- **A wall-clock check inside `train()`.** Each group should stop and save at its deadline instead of relying on a startup step count.
- **Periodic checkpoints of the log.** A cap or crash then keeps everything finished so far.
- **The sampled replicate run separately from the population groups.** That means its own worker or a later slot, or MPS enabled, so it can't take SM time from them.

Even then, the population families alone don't fit the planned t_max in 4.5 h. The leaf family took 11.6 ms per step with six groups running together, so 2e6 steps would take about 6.4 h. So either t_max = 1e5 needs more than one worker, or t_max has to drop to at most about 7e4 (1.4e6 steps in 4.5 h), less with a safety margin.
