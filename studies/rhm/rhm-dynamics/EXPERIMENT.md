---
name: rhm-dynamics
title: "RHM learning dynamics: TreeTN on root and last-token tasks across widths, grammars and seeds"
status: done
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 8
budget_usd: 9
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-12T21:42:22Z
result: supported
finished_utc: 2026-11-13T02:30:28Z
writeup: studies/rhm/rhm-dynamics/README.md
---

# RHM learning dynamics: TreeTN on root and last-token tasks across widths, grammars and seeds

## Question

Can a balanced binary tree tensor network (TreeTN, no biases or nonlinearities) learn the Random Hierarchy Model by backprop? How do its loss and accuracy curves depend on the bond dimension r, and how close does it get to the Bayes-optimal predictor? The checkpoints feed the interpretation analysis (`studies/rhm/interpret.py`), which asks whether and when bonds come to encode the grammar's latent symbols.

This is the main RHM experiment. The controls, sample-efficiency and transfer experiments reuse its training code and LR pilot.

## Hypothesis

Grammar: v = 16 tokens and symbols per level, n = 16 root classes, m = 4 productions per symbol, binary (s = 2), depth L = 4 (16 tokens). The tree's leaf order matches the data tree. Tasks: 'root' (all 16 tokens → root symbol; Bayes: loss 0, accuracy 1) and 'last' (tokens 0–14 plus a mask token at leaf 15 → token 15; Bayes cross-entropy ≈ 0.14 nats, accuracy ≈ 0.89 on grammar 0).

- **H1, learnable at r = 16.** At r = 16 (= v, enough for a one-hot code of each latent symbol), init seed 0, m = 4, the network reaches root test accuracy ≥ 0.99 and last-token test KL from Bayes ≤ 0.05 nats on all 3 grammars. Against: root accuracy < 0.9 or last KL > 0.2 nats on 2 or more grammars of a task.
- **H2a, r = 8 is too narrow.** r = 8 < v cannot carry 16 latent symbols as orthogonal codes, so r = 16 ends with better final quality (test error rate, then test loss, beyond the tie tolerances below) in at least 5 of the 6 (task, grammar) pairs. Against: r = 16 is worse in 3 or more pairs, or no better than r = 8 on balance.
- **H2b, wider learns faster.** r = 64 reaches threshold (diagnostic accuracy ≥ 0.9 × Bayes accuracy) earlier than r = 16, beyond the margin below, in at least 5 of 6 pairs. r = 32 is reported the same way, as a descriptive result. Against: r = 64 is slower in 3 or more pairs. All widths share the r = 16 pilot's LR. If the r = 8 or r = 64 pilot prefers another LR, H2a and H2b are reported as conditional on the shared LR.
- Descriptive, no prediction:
  - the init and data-order spread at r = 16 (3 seeds × 3 grammars);
  - the m = 8 grammar (more synonyms; Bayes last-token cross-entropy ≈ 0.74 nats, accuracy ≈ 0.57 on grammar 0), where we expect slower learning;
  - the variance-init arm.

All results are for the TreeTN with the identity init described below.

## Method

- **Model** (`ctn/tree.py` TreeTN, built by `studies/rhm/common.py`): untied cores, no biases or nonlinearities, float32. Leaf vocabulary 16 ('root') or 17 ('last', with the mask token). Parameters at r = 16: 65,536 ('root').
- **Initialization: identity init** (`ctn.tree.identity_init`, cfg `init='identity'`, eps = 0.1), for every run except the variance-init arm.
  - Bond channel 0 carries a constant. Leaf embeddings are e_0 + 0.1 × N(0, 1). Fusion cores are 0.1 × N(0, 1/r²) plus T, with T[0,0,0] = 1 and T[a,a,0] = T[a,0,a] = 1. The network therefore starts near its additive part while staying multilinear. The root core C keeps N(0, 1/r²).
  - Why: the TreeTN's default variance init does not learn the RHM. On the coordinator's CPU at r = 16, on both tasks and at both LRs tried, Adam shrank every core toward 0 within about 300 updates. That is the uniform-output saddle of a multilinear network: the diagnostic loss stayed at log 16 = 2.773, and a root run was still at chance at update 20,000, with logit std 8.5e-3.
  - With the identity init the same model learned both tasks. Under the 100-update warmup used here, r = 16 root accuracy went from 0.08 to 0.94 between updates 100 and 500, and last-token accuracy reached 0.86 (Bayes 0.885) by update 1,250.
- **Data**: online, 256 fresh trees per update from the grammar. They are drawn on the GPU with a torch generator seeded by (grammar, init seed) only, so runs that differ in width see the same trees.
- **Training** (`studies/rhm/train.py`): cross-entropy, Adam, batch 256, 40,000 updates.
  - LR schedule: linear warmup to the peak over 100 updates, then cosine to 0.1 × peak at update 40,000.
  - The warmup is short so that steps to threshold measure learning rather than the LR ramp. With a 1,000-update warmup the root task was solved inside the ramp.
  - The cumulative LR to threshold is reported next to the steps.
- **LR pilot**:
  - Setup: grammar 0, init seed 9 (no evaluation run uses it), each task, peak LR in {1e-4, 3e-4, 1e-3, 3e-3}, 4,000 updates of the 40,000-update schedule. The lowest diagnostic loss at update 4,000 wins.
  - The r = 16 choice per task is used for every run, unless the r = 8 or r = 64 pilot diverged (non-finite diagnostic loss) at it: then the shared LR steps down the grid until no width's pilot diverged, before any grid run starts (`shared_lr` in run.py, recorded as `shared` in lr.json). This keeps one LR for all widths, so H2b stays a comparison of widths.
  - The same pilot also runs at r = 8 and r = 64, only to record which LR those widths prefer.
  - Results, and whether a choice sits at an edge of the grid, are in `runs/rhm-dynamics/lr.json` and `results/analysis.json`.
  - `--lr` overrides the pilot; reruns go to a separate `--out` and are excluded from H2a and H2b.
- **Grid, in priority order** (48 runs; a run whose turn comes after the training deadline is recorded as skipped):
  1. m = 4, r = 16, init seed 0, grammars 0–2, both tasks (6);
  2. m = 4, r ∈ {8, 32}, init seed 0 (12);
  3. m = 4, r = 16, init seeds 1, 2 (12);
  4. variance-init arm: m = 4, r = 16, grammars 0–2, init seed 0, both tasks, with the TreeTN's default variance init, capped at 5,000 updates of the same 40,000-update schedule (6). It is expected to stay at chance and records the collapse on the GPU.
  5. m = 4, r = 64, init seed 0 (6);
  6. m = 8, r = 16, init seed 0, grammars 0–2 (6).
- **Evaluation**:
  - Diagnostic set: a fixed 20,000-tree set per grammar (`common.diag_trees`, CPU-seeded, saved as trees in `runs/rhm-dynamics/data/diag_m<m>_g<g>.pt`). It is evaluated at 60 log-spaced updates, every 50 updates up to 3,000, every 1,000 updates and at every checkpoint: loss, accuracy, and KL(p_Bayes ‖ q).
  - Test set: a separate 50,000-tree set, evaluated at the end.
  - The Bayes posterior is exact for 'last' (`grammar.posterior_token`, float64, once per grammar). For 'root' it is the point mass on the true root.
- **Checkpoints**: at updates 0, 100, 300, 1,000, 3,000, 10,000, 20,000 and 40,000 (`common.CHECKPOINTS`), plus 50, 150, 200, 400, 500, 700 and 1,500; 15 in all.
  - The extra ones bracket the transition under the 100-update warmup. In a CPU probe at r = 16 on grammar 0, root accuracy was 0.08, 0.21, 0.40, 0.74 and 0.94 at updates 100, 200, 300, 400 and 500. Last-token accuracy was 0.12, 0.41, 0.52, 0.62 and 0.70 at updates 50, 100, 150, 200 and 300, and reached 0.80 near 700.
  - Updates 700 and 1,500 cover the slower settings: r = 8 root was at 0.56 at update 1,000 and 0.87 at 1,500.
- **Concurrency**: runs execute 6 at a time as separate processes sharing the GPU. Per-run ms/update is recorded and tabled per width.
- **Interpretation**: a separate step, after the training figures exist, with its own time limit.
  - `analyze.py --interpret-only` runs `python -m studies.rhm.interpret <run dir> --out <run dir>/interpret.json --diag runs/rhm-dynamics/data/diag_m<m>_g<g>.pt --device cuda` (cpu when no GPU), so it uses the training diagnostic trees.
  - Order: identity-init r = 16 runs first, then the other widths and the variance-init arm. 4 processes at a time; no run starts after 3 h.
  - `plot.py` then runs `python -m studies.rhm.plot_interp` on the interpret JSONs of the identity-init r = 16 runs, one figure set per (task, m).

### Comparison rules (`studies/rhm/analysis.py`)

- **Which runs enter.** Only runs with status finished enter a comparison. A pair where either run was truncated, diverged, skipped or failed, or has a non-finite metric, is dropped. Dropped pairs are counted and never count as ties.
- **Run health** is counted against the planned run list (`planned.json`), not against the result files present.
- **Steps to threshold.** r = 64 is faster only if t_64 < t_16 − margin. The margin is max(the logging interval at the later of the two, the r = 16 init/data-order seed range of t on that (task, grammar), seeds 0–2). Otherwise the pair is a tie. A run that finished without reaching the threshold has t = +inf: finite beats inf, and inf vs inf is a tie.
- **Final quality** (H2a). Test error rate decides first. A difference within max(1e-3, the r = 16 seed range of the error rate) is a tie, and then test loss decides, with a tie within max(1e-3 nats, the r = 16 seed range of the loss). Float32 log_softmax gives exactly 0 loss beyond a ~17-nat margin, so losses below 1e-3 nats are not distinguished.
- **Ties** count for neither side and are excluded from sign tests. They are reported with per-grammar counts next to every pooled count.
- **Consistency.** A "supported" verdict also needs wins > losses in at least 2 of the 3 grammars and losses > wins in none.
- **Reference must learn.** A comparison is inconclusive if the r = 16 reference didn't learn on a task. That means median test accuracy < 2 × chance (0.125), or median final diagnostic loss within 0.05 nats of log 16.

## Inputs

None. All data is generated from the seeded grammars. No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.check_interpret
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-dynamics/run.py --out runs/rhm-dynamics --hours 3
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-dynamics/analyze.py runs/rhm-dynamics "$EXP_DIR/results"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-dynamics/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
OMP_NUM_THREADS=8 PYTHONPATH=. timeout 3.5h uv run --frozen python studies/rhm/rhm-dynamics/analyze.py runs/rhm-dynamics "$EXP_DIR/results" --interpret-only --interpret-hours 3 --interpret-procs 4 || echo "interpretation stopped: $?"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-dynamics/plot.py "$EXP_DIR/results" "$EXP_DIR/figures" runs/rhm-dynamics
```

The interpretation step calls, for each run directory, 4 at a time with `OMP_NUM_THREADS=2` each (output in `<run dir>/interpret.log`):

```bash
python -m studies.rhm.interpret <run dir> --out <run dir>/interpret.json --diag runs/rhm-dynamics/data/diag_m<m>_g<g>.pt --device cuda
```

The final `plot.py` call runs `python -m studies.rhm.plot_interp <interpret.json ...> "$EXP_DIR/figures/interp_<task>_m<m>"` for each (task, m) group of identity-init r = 16 runs.

Notes:
- Both check scripts exit non-zero on any failure, which stops the job.
- `--hours 3` is the training deadline, counted from the end of setup and the LR pilots.
- Run the training as its own `orch exec` step from a background command.
- `OMP_NUM_THREADS=8` avoids the cgroup thread oversubscription seen on the sequential worker.

## Expected outputs

- `results/checks.json`.
- `runs/rhm-dynamics/<task>_m<m>_g<g>_r<r>_s<seed>[_varinit]/`, about 2 GB in total, mostly the r = 64 checkpoints (15 MB each). Each holds:
  - `result.json`: config, LR, status, log, test metrics, Bayes references, ms/update;
  - up to 15 checkpoints `ckpt_<step>.pt`;
  - `interpret.json` and `interpret.log` where the interpretation ran.
- `runs/rhm-dynamics/`:
  - `data/diag_<key>.pt` and `data/bayes_<key>.pt`;
  - `lr.json`, `planned.json`, `status.json`, `clock.json`;
  - `pilot/`, `source/`.
- `results/analysis.json`: criteria with pair counts, per-grammar directions, dropped pairs and margins; the per-width table with ms/update; per-run summaries; LR choices and grid-edge flags; interpretation status.
- `results/curves.json`.
- `figures/widths.pdf`, `figures/seeds.pdf`, `figures/m8.pdf`, `figures/init.pdf`, and `figures/interp_<task>_m<m>/` (`interp_training.pdf`, `interp_invariance_coupling.pdf`).
- `README.md`: the write-up.

## Controls

- The Bayes-optimal predictor on the same diagnostic and test sets: every loss is also reported as KL from Bayes.
- Widths share the online data stream and the LR, so they differ only in r. The r = 8 and r = 64 LR pilots show whether sharing the LR favours one width.
- Three grammars and, at r = 16, three init seeds separate grammar-to-grammar variation from init and data-order variation. The r = 16 seed ranges set the tie margins.
- The variance-init arm (6 runs, 5,000 updates) shows what the identity init changes, under the same schedule and data.
- Architecture controls (permuted leaves, tied cores, chain) are a separate experiment, `rhm-controls`.
- Correctness checks run before training: `studies/rhm/checks.py` and `studies/rhm/check_interpret.py`.
  - posteriors against brute force on a tiny grammar
  - parsing and the task masks
  - checkpoint round trips
  - a training step that lowers the loss
  - frozen parameters staying bitwise fixed
  - the interpretation measures

## Success and failure criteria

Computed by `analyze.py` with the comparison rules above.

- **H1.** Supported if all 6 (task, grammar) runs at r = 16, seed 0, m = 4 finished and meet their bar (root test accuracy ≥ 0.99, last test KL ≤ 0.05 nats). Not supported if root accuracy < 0.9 or last KL > 0.2 on 2 or more grammars of a task. Otherwise partly supported. Inconclusive if any of the 6 runs is missing or unfinished.
- **H2a.** Supported if r = 16 has better final quality than r = 8 in at least 5 of 6 pairs and the consistency rule holds. Not supported if r = 8 is better in 3 or more, or r = 16 wins no more pairs than it loses. Otherwise partly supported.
- **H2b.** The same counts for r = 64 against r = 16 on steps to threshold, with the margin. The r = 32 comparison is reported the same way, as a descriptive result.
- **Conditional.** H2a and H2b are marked "conditional on the shared LR" if the r = 8 or r = 64 pilot preferred another LR than r = 16.
- **Inconclusive:**
  - a comparison has fewer than 4 usable pairs (the number dropped is reported);
  - the r = 16 reference didn't learn;
  - more than a third of the planned runs didn't finish (this overrides every verdict).
- **Descriptive:**
  - per (task, m, r): medians and ranges of final test KL, error rate, steps and cumulative LR to threshold, and ms/update;
  - the init and data-order spread;
  - m = 8 against m = 4;
  - whether the variance-init runs stay at chance (test accuracy < 2 × chance), per task and grammar.
- **Run failed:** checks fail, CUDA is unavailable, or no r = 16 run finished.

## Time and cost

- **Setup:** about 10 minutes (`uv sync`, both checks, Bayes posteriors for 6 grammars).
- **LR pilots:** 24 runs × 4,000 updates (3 widths × 2 tasks × 4 LRs), about 10–15 minutes with 6 at a time. The training deadline starts after them.
- **Training:** about 1.8 M updates (42 runs × 40,000 + 6 × 5,000).
  - Measured on this coordinator's CPU (one thread, including sampling): r = 8 at 1.9 ms/update, r = 16 at 5.6 ms, r = 32 at 22 ms, r = 64 at 125 ms.
  - On the GPU each run should be launch-bound at roughly 2–5 ms/update (not measured), so the grid takes 1–2.5 h serially, and less with 6 runs sharing the GPU.
  - The training deadline is 3 h.
- **Interpretation:** its authors measured about 5–10 minutes per r = 16 run for 8 checkpoints on a GPU, so about 9–19 minutes for the 15 here. The 24 identity-init r = 16 runs, 4 at a time, take about 1–2 h. No new run starts after 3 h, and `timeout 3.5h` ends the step.
- **Total:** about 0.3 h + 3 h + 0.1 h + 3.5 h ≈ 6.9 h at most, under the 8 h cap.
- **Cost:** at up to $1.09/h (L40S, secure) the 8 h cap costs at most $8.72, inside the $9 budget.

## If something fails

- **Checks fail or CUDA is unavailable:** stop, finish the worker, mark failed.
- **A run crashes:** the others continue; the failure is in `status.json`.
  - Rerun the failed runs once with `--only <name,...> --hours <left - 4>` into a separate `--out runs/rhm-dynamics-rerun` (first `mkdir -p runs/rhm-dynamics-rerun && cp runs/rhm-dynamics/lr.json runs/rhm-dynamics-rerun/`, so the rerun uses the original LRs), so the original results are kept.
  - Report reruns separately: they don't enter H2a and H2b. Note an amendment.
- **More than a quarter of runs diverge:** rerun those once with `--lr <the next lower grid LR>` into `runs/rhm-dynamics-rerun`, as an amendment. They are reported but don't enter H2a and H2b.
- **The first wave's ms/update projects past the deadline:** let the grid run (lower-priority groups are skipped and counted). If time remains on the worker cap, rerun the skipped groups with `--only` into a separate `--out`.
- **The identity-init r = 16 runs don't learn either (root accuracy < 0.2 at update 40,000):** analyze and write up as is. That is a result, not a reason to change the method on the worker.
- **The interpretation module is missing or fails on a run:** it is skipped with a note, and the training results and figures (already written) stand.
- **Time running out:** everything finished is saved; write up what exists.
- **None of the GPU types in stock:** the launcher waits and retries; if none is found, report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Widths, grammars, seeds?** The user set r ∈ {8, 16, 32, 64}, grammars 0–2, init seed 0 everywhere plus seeds 1 and 2 at r = 16, and 40,000 online updates.
- **Scale-up?** The user allowed adding m = 8 at r = 16, 3 grammars, both tasks.
- **LR?** Pilot at r = 16 on grammar 0, pick by diagnostic loss. After review the grid is {1e-4, 3e-4, 1e-3, 3e-3} on a non-evaluation seed, also run at r = 8 and r = 64 for the record.
- **Initialization?** The user chose the identity init (eps 0.1) as the main init, after the CPU probes showed the variance init collapsing. The user also asked for a documented variance-init arm (r = 16, both tasks, 3 grammars, seed 0).
- **Warmup?** The user set a 100-update warmup for every RHM run, so steps to threshold don't measure the warmup ramp, and asked that the dense checkpoints bracket the transition.
- **Interpretation?** The user fixed the command form `python -m studies.rhm.interpret <run dir> --out <run dir>/interpret.json`. They added `--diag` (this run's diagnostic set) and `--device cuda`, `check_interpret` before training, and `plot_interp` figures per (task, m) over the r = 16 runs.

## Review

An independent review returned "needs revision". Findings and changes:

- **Unfinished runs could enter comparisons, and health was counted from the result files present.**
  - Comparisons now use only pairs where both runs finished. Dropped pairs are counted, and NaN metrics are missing rather than ties.
  - Health is counted against `planned.json`, and the "more than a third unfinished" flag overrides every verdict.
- **Final loss on 'root' saturates in float32.** H2a now compares the test error rate first and then the loss, with tie tolerances max(1e-3, the r = 16 seed range).
- **Steps-to-threshold wins had no margin.** A win now needs more than max(the logging interval, the r = 16 init/data-order seed range). The cumulative LR to threshold is reported next to the steps.
- **One grammar could carry a pooled verdict.** Per-grammar directions are reported next to every count, and "supported" needs the consistency rule.
- **No rule for a reference that didn't learn.** A comparison is now inconclusive if the r = 16 reference's median test accuracy is < 2 × chance, or its final diagnostic loss is within 0.05 nats of log 16.
- **LR pilot.**
  - 1e-4 was added; the pilot runs on seed 9, not an evaluation seed; grid-edge choices are recorded.
  - The pilot also runs at r = 8 and r = 64. H2 is reported as conditional if they prefer another LR.
  - `--lr` overrides exist, reruns go to a separate `--out` and are excluded from H2a and H2b.
- **Ordering.**
  - The variance-init arm now runs before r = 64 and is capped at 5,000 updates.
  - `analyze.py` and `plot.py` produce the training results before the interpretation step, which has its own time limit.
  - "Seed spread" is now "init and data-order spread".
- **Budget.** The deadline starts after setup and pilots, ms/update is tabled per width, and the cap and budget assume secure prices up to $1.09/h. A rerun rule for a deadline overrun was added.

## Amendments

- **2026-11-12 ~23:00 UTC, supervisor: descriptive rerun of the r = 64 runs at peak LR 1e-3.** At the shared LR 3e-3 (the r = 16 pilot choice; the r = 64 pilot preferred 3e-4 but did not diverge at 3e-3, so `shared_lr` kept 3e-3), 3 of the 6 r = 64 runs (last g0, last g1, root g2) learned to near-Bayes diagnostic accuracy by update 3,000 and then collapsed to the uniform output (diagnostic loss 2.773 = log 16) by update 10,000, ending at chance. Their losses stayed finite, so this is not the "diverge" case the definition covers, and the 4,000-update pilot ended before the collapse. The definition's rules are applied unchanged: these runs are status finished and enter H2b with their measured steps to threshold, and the write-up flags that they collapsed later. To show whether the collapse comes from the shared LR, after the interpretation step the 6 r = 64 runs are rerun once with `--only r64 --lr 1e-3` (the next lower grid LR, mirroring the divergence rule) into `runs/rhm-dynamics-rerun` (with the original `lr.json` copied in first). They are reported separately and do not enter H2a or H2b. Cost: about 15 minutes of worker time (~$0.20), inside the cap and budget.

## Outcome

Result: supported (H1), with the width hypotheses conditional on the shared LR. Write-up: [README.md](README.md).

- **H1:** supported. r = 16, seed 0: root test accuracy 1.0, 1.0, 1.0; last-token test KL 0.0040, 0.0036, 0.0055 nats (bar 0.05).
- **H2a:** partly supported, conditional on the shared LR. r = 16 beat r = 8 in 3 pairs, lost none, tied 3 (all root, both at zero error). r = 8 reaches the Bayes error rate; its last-token KL is about twice r = 16's.
- **H2b:** supported, conditional on the shared LR. r = 64 reached threshold first in 6 of 6 pairs (107–200 against 219–449 updates), but 3 of the 6 r = 64 runs collapsed to chance after reaching it. r = 32 was faster than r = 16 in 6 of 6 (descriptive).
- **Conditional:** the r = 64 pilot preferred 3e-4 and the r = 8 root pilot preferred 1e-3 (by 1e-6 nats); the r = 16 choice, 3e-3, is at the top of the grid.
- **Inconclusive triggers:** none. All 48 planned runs finished, every comparison had 6 usable pairs, and the r = 16 reference learned on both tasks.
- **Descriptive:** the variance init stayed at chance in 6 of 6 runs; at m = 8, root was solved about 6.5 times slower and last-token did not reach threshold in 40,000 updates (KL 0.36–0.40 nats). The amended r = 64 rerun at 1e-3 had no collapse. Interpretation completed for all identity-init r = 16 runs; 4 of the 6 r = 64 runs were cut off by the timeout.
- **Cost and time:** one RTX A6000, 4.63 h at $0.69/h, about $3.19, against an 8 h cap and a $9 budget.
