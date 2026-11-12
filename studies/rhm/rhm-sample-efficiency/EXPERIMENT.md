---
name: rhm-sample-efficiency
title: "RHM sample efficiency: r = 16 TreeTN on fixed training sets of 2^10 to 2^20 trees"
status: done
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 7
budget_usd: 8
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-12T21:42:22Z
result: supported
finished_utc: 2026-11-12T22:54:50Z
writeup: studies/rhm/rhm-sample-efficiency/README.md
---

# RHM sample efficiency: r = 16 TreeTN on fixed training sets of 2^10 to 2^20 trees

## Question

How many training examples does an r = 16 TreeTN with the identity init need to learn the Random Hierarchy Model? When it fails, is that because the data is insufficient or because optimization fails?

Deep CNNs need about P* = n m^L = 16 × 4^4 = 4,096 examples on this grammar (Cagnetta et al.). N is reported in units of P*.

## Hypothesis

Same grammar and tasks as `rhm-dynamics` (v = n = 16, m = 4, s = 2, L = 4; 'root' and 'last'). All results are for the TreeTN with the identity init.

- **H1, root-task sample complexity near P*.** The median (over 3 grammars) root test accuracy is below 0.5 at N = 2^10 = P*/4, and reaches 0.9 at some N ≤ 2^16 = 16 P*. Against: it needs 2^18 or more, or never reaches 0.9 even at 2^20.
- **H2, at large N, failures are data-limited, not optimization-limited.** Among the runs with N ≥ 2^16 (18 planned), none is optimization-limited: a run that doesn't learn ends with a training loss at least 0.1 nats above the Bayes loss on the same training trees despite running all its updates. Against: more than a quarter of those 18 runs are optimization-limited.
- Descriptive: last-token test KL from Bayes vs N; the gap between training-set and test loss vs N; and the online references (N → ∞).

## Method

- **Model**: r = 16 untied TreeTN, init seed 0, float32, identity init (`ctn.tree.identity_init`, cfg `init='identity'`, eps = 0.1).
  - Bond channel 0 carries a constant, so the network starts near its additive part.
  - The TreeTN's default variance init collapses on the RHM. In CPU probes at r = 16, on both tasks and both LRs tried, Adam drove every core toward 0 within about 300 updates, and the diagnostic loss stayed at log 16 = 2.773.
  - With the identity init the probes learned: r = 16 root accuracy reached 0.94 by update 500 with the 100-update warmup.
- **Training sets**: for each grammar, one seeded sample of 2^20 trees (CPU generator, seed 40,000 + grammar seed). A set of size N is its first N trees, so the sets are nested across N. Batches of 256 are drawn uniformly with replacement.
- **Grid** (42 runs, priority order):
  1. N ∈ {2^10, 2^14, 2^18, 2^20} (the mentor's sizes) × grammars 0–2 × both tasks;
  2. N ∈ {2^12, 2^16};
  3. 6 online reference runs (2 tasks × 3 grammars): fresh trees every update, the same model, init and 100,000-update schedule, capped at 20,000 updates.
- **Training**: cross-entropy, Adam, up to 100,000 updates. The LR warms up linearly over 100 updates, then follows a cosine to 0.1 × peak at update 100,000.
  - Peak LR per task: from the same online pilot as `rhm-dynamics` (r = 16, grammar 0, init seed 9, which no evaluation run uses; {1e-4, 3e-4, 1e-3, 3e-3}; 4,000 updates of a 40,000-update schedule; lowest diagnostic loss wins).
  - Grid-edge choices are recorded.
  - `--lr` overrides the pilot; reruns go to a separate `--out`.
- **Validation and selection**: the fixed 20,000-tree diagnostic set of each grammar, independent of the training set, is the validation set. It is evaluated at 60 log-spaced updates, every 50 updates up to 3,000, and every 1,000. Two early-stopping selections are kept:
  - the checkpoint with the highest diagnostic accuracy, used for H1, "learned" and the regimes;
  - the checkpoint with the lowest diagnostic loss, used for test loss and KL.
- **Early stop**: a run stops early only when the diagnostic loss hasn't improved for 25,000 updates and the training-set loss is below 1e-3 nats (it has fit the data). Otherwise it runs the full 100,000 updates.
- **Training-set fit**: at every evaluation, the loss, accuracy and KL to Bayes on the first min(N, 20,000) trees of the training set. For 'last', the Bayes posterior of these trees is computed exactly. For 'root' the KL equals the loss.
- **Regime per run**, at the accuracy selection; the primary result table counts these per (task, N):
  - learned: test accuracy ≥ 0.9 × Bayes accuracy;
  - data-limited: not learned, and the final training loss exceeds the Bayes loss on the same training trees by < 0.1 nats (it fit its training data; memorizing gives an excess ≤ 0). The KL to Bayes can't measure fit: a memorizing model puts its mass on the observed token while Bayes spreads it, so its KL grows as it fits;
  - optimization-limited: not learned, the run ended normally (finished or early-stopped), and the final excess training loss over Bayes is ≥ 0.1 nats;
  - unfinished: truncated, diverged, skipped, failed or missing. Never counted as a regime.
- **Medians** per (task, N) need all 3 grammars finished. A size with fewer is missing for H1. Run health is counted against the planned list (`planned.json`).

## Inputs

None. All data is generated from the seeded grammars. No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-sample-efficiency/run.py --out runs/rhm-sample-efficiency --hours 5.5
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-sample-efficiency/analyze.py runs/rhm-sample-efficiency "$EXP_DIR/results"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-sample-efficiency/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero on any failure, which stops the job. `--hours 5.5` is the training deadline, counted from the end of setup and the LR pilot. Run the training as its own `orch exec` step from a background command.

## Expected outputs

- `results/checks.json`.
- `runs/rhm-sample-efficiency/<task>_g<g>_N<N>/` and `<task>_g<g>_online/`, each with:
  - `result.json`: the log with diagnostic and training-set metrics, both selections' steps and metrics, and the final test metrics;
  - checkpoints at the scaled steps (0, 250, 750, 2,500, 7,500, 25,000, 50,000, 100,000, plus the stopping step), about 0.3 MB each.
- Also in `runs/rhm-sample-efficiency/`: `lr.json`, `planned.json`, `status.json`, `clock.json`, `data/`, `pilot/`, `source/`. About 100 MB in total.
- `results/analysis.json`: health, regimes per (task, N), criteria, the per-(task, N) table with ms/update, and per-run summaries.
- `results/curves.json`.
- `figures/vs_N.pdf`, `figures/curves.pdf`.
- `README.md`: the write-up.

## Controls

- The Bayes predictor on the same test set.
- The 6 online references in this run, which share the model, LR and schedule, give the N → ∞ end.
- The excess training loss over Bayes (and the training-set KL) next to the test numbers for every run separates fitting from generalizing, as the project rules require.
- Nested training sets make the N comparison paired within each grammar.
- Both selections use the diagnostic set, never the test set.
- The correctness checks run first (`studies/rhm/checks.py`).

## Success and failure criteria

Computed by `analyze.py`. Medians over the 3 grammars, valid only when all 3 runs finished; a size without a valid median is missing.

- **H1** is decided on the sizes present:
  - Supported if some N in {2^12, 2^14, 2^16} reaches median root accuracy ≥ 0.9 and 2^10 is below 0.5.
  - Partly supported if one of those sizes reaches 0.9 but 2^10 is already ≥ 0.5; if 2^10 is missing, this is reported as "supported at N = …, low end untested".
  - Not supported if no present size up to 2^16 reaches 0.9 and 2^16 is present. The reason is stated: the smallest of 2^18 or 2^20 that reaches it, or not reached at any present size.
  - Inconclusive only if 2^16 is missing and no smaller present size decides.
- **H2:**
  - Supported if none of the N ≥ 2^16 runs is optimization-limited.
  - Not supported if more than 18/4 = 4.5 of them are.
  - Otherwise partly supported, with the runs named.
  - Inconclusive if fewer than 9 of the 18 finished.
- **Descriptive:** regime counts per (task, N); last-token KL from Bayes vs N; train/test gap; online references; ms/update.
- **Run failed:** checks fail, CUDA is unavailable, or no N = 2^20 run finished.

## Time and cost

- **Setup:** about 10 minutes (checks, Bayes posteriors, and the 2^20-tree sets per grammar in each worker process).
- **LR pilot:** 8 runs × 4,000 updates, a few minutes.
- **Training:** at most 36 × 100,000 + 6 × 20,000 ≈ 3.7 M updates. Small N may stop early after fitting; large N run to 100,000.
  - On the coordinator's CPU, r = 16 takes 5.6 ms/update (one thread).
  - On the GPU it should be launch-bound at roughly 2–5 ms (not measured): about 2–5 h serially, and about 1–2 h with 6 runs sharing the GPU.
  - The training deadline is 5.5 h after the pilot. Unstarted runs are recorded as skipped, the online references and the intermediate sizes first.
- **Total:** at most about 0.3 h + 5.5 h + 0.1 h ≈ 5.9 h, under the 7 h cap.
- **Cost:** at up to $1.09/h (L40S, secure), the 7 h cap costs at most $7.63, inside the $8 budget.

## If something fails

- **Checks fail or CUDA is unavailable:** stop, finish the worker, mark failed.
- **A run crashes:** the others continue. Rerun the failed runs once with `--only <names> --hours <left - 0.75>` into a separate `--out runs/rhm-sample-efficiency-rerun` (first `mkdir -p runs/rhm-sample-efficiency-rerun && cp runs/rhm-sample-efficiency/lr.json runs/rhm-sample-efficiency-rerun/`, so the rerun uses the original LRs), so the original results are kept, and note an amendment.
- **Out of GPU memory** (unlikely; the largest training set is 2^20 × 16 tokens): rerun the failed runs with `--procs 3` into the separate `--out`.
- **Runs diverge:** rerun those once with `--lr <the next lower grid LR>` into the separate `--out`, as an amendment.
- **The first wave's ms/update projects past the deadline:** let the grid run (lower-priority runs are skipped and counted). If the worker cap leaves time, rerun the skipped runs with `--only` into a separate `--out`.
- **None of the GPU types in stock:** the launcher waits and retries; if none is found, report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Sizes?** The mentor asked for N ∈ {2^10, 2^14, 2^18, 2^20}. The user allowed adding 2^12 and 2^16, at r = 16, 3 grammars, both tasks, init seed 0.
- **Training length?** The user asked to train long enough to separate optimization failure from insufficient data, for example up to 100,000 updates with early stopping on held-out data from the diagnostic distribution, reporting the final training loss next to the test loss. We use the diagnostic set as that held-out set, stop early only once the training set is fit, and measure fit as the excess training loss over the Bayes loss on the same training trees.
- **Initialization and warmup?** The user chose the identity init (eps 0.1) as the main init for all RHM experiments, after the CPU probes showed the variance init collapsing, and a 100-update warmup for every RHM run.
- **Online references?** Added after review, at the end of the priority list.

## Review

An independent review returned "needs revision". Findings and changes:

- **P0: H1 and the regimes used the loss-selected checkpoint.** On an accuracy question that can pick a less accurate model. Accuracy-based tests now use the checkpoint with the highest diagnostic accuracy; loss and KL keep the loss-selected one.
- **"Fit" for the last-token task was the raw training loss, which can't go below the Bayes cross-entropy.** Fit was then changed to the KL to Bayes on the first 20,000 training trees, which the re-check showed grows as a model memorizes (memorizing runs were labelled optimization-limited); fit is now the excess training loss over the Bayes loss on those trees (≤ 0 when memorizing, ~0 at Bayes; for 'root' it equals the training loss).
- **H2's denominator was all runs, where small-N runs fit trivially.** H2 is now stated over N ≥ 2^16 (18 planned runs), with the threshold on that denominator. Per-(task, N) regime counts are the primary table.
- **H1 depended on every size being present.** H1 now decides on the sizes present (inconclusive only if none decides), and medians need all 3 grammars finished.
- **Wording.** The comparison with CNNs no longer says "or fewer"; N is given in units of P*.
- **No N → ∞ end in this run.** 6 online reference runs were added at the end of the priority list, capped at 20,000 updates.
- **Scope.** The text now says results are for the TreeTN with the identity init.
- **Shared fixes.**
  - Health is counted against `planned.json`.
  - Unfinished runs never count as a regime.
  - The LR grid now includes 1e-4, the pilot runs on seed 9, and grid-edge choices are recorded.
  - `--lr` overrides exist and reruns go to a separate `--out`.
  - The deadline starts after the pilot, and ms/update is tabled.
  - The cap and budget assume secure prices up to $1.09/h.

## Amendments

None.

## Outcome

Supported. Write-up: [README.md](README.md).

- **H1, supported.** The median root test accuracy (3 grammars) is 0.084 at 2^10, 0.152 at 2^12 and 0.99996 at 2^14, and stays at least 0.9999 up to 2^20. The threshold lies between P* and 4 P*.
- **H2, supported.** 0 of 18 runs with N ≥ 2^16 are optimization-limited, because all 18 learned. All 12 runs that failed (N ≤ 2^12, both tasks) fit their training sets and are data-limited.
- **Descriptive.** The last-token test KL to Bayes falls from 2.45 nats at 2^10 to 0.012 at 2^20. The online references reach 0.9995 root accuracy and 0.027 nats last-token KL at 20,000 updates. Each run took 7.3–9.8 ms per update with 6 runs sharing the GPU.
- **Health.** 42 of 42 planned runs finished, with no amendments and no reruns.
- **Cost.** One A40 at $0.59/h, 1 h 04 min, about $0.63, against a cap of 7 h and $8.
