# Supervisor report: lang-induction

The experiment ran to completion: all 48 runs finished, none collapsed, and all 36 tree and perm runs were interpreted. Two of the four hypotheses are supported (H1 retrieval width, H3 alignment), H2 (the capacity pattern by target position) is not supported, and H4 (multi-match width) is partly supported because the r = 128 runs were still improving. It cost about $3.87 of the $12 budget. The results are in [README.md](README.md), and the one-line verdicts are in the definition's "Outcome".

## What ran

The worker was f0xfuziyvp8uk7, an A40 at $0.59/h in CA-MTL-1. The L40S was out of stock outside US-NC-1 and the A6000 launch failed. It ran from 2026-11-13 23:53 to 2026-11-14 06:26 UTC, 6.56 h.

1. **Setup** (23:54–23:57 UTC). nvidia-smi and the CUDA check, `studies.lang.checks` (47/47 passed), and `studies.lang.rank` (29 at node 3_0 for retrieval, up to 442 at level 2 for multi-match, as the definition expected).
2. **Training, first attempt** (23:57–00:11). Pilots at the default 6 processes. Four of the r = 64/128 retrieval pilots died with CUDA out-of-memory. I stopped the job (see decisions).
3. **Training with `--procs 3`** (00:11–06:04). The 64 pilots finished at 01:35, reusing the 17 that had already finished. Then the 48-run grid ran, with a 4.5 h deadline at 06:05. Every run started before the deadline, so none was skipped.
4. **Interpretation** (06:04–06:23). All 36 runs in 18 minutes, against a 2.67 h allowance.
5. **Analysis, plots and pruning** (06:23). `analyze.py` and `plot.py` wrote `results/` and the four figures. The pruning loop then removed the intermediate checkpoints of the r = 128 runs.
6. **`orch finish`** (06:26). Published 5.22 GB of runs and the results, merged this folder, and deleted the worker. `orch ls` then showed no running workers for this project.

## Result against each criterion

| Criterion | Verdict | Key number |
|---|---|---|
| H1 retrieval width | supported | KL 0.0037 (r = 128) against 0.28–0.34 (r = 16), 3/3 seeds; smallest stable width r = 64 |
| H2 stratified capacity | not supported | graded near > mid > far in 3/3 seeds; near − mid 0.027–0.031, above the 0.02 the pattern allows |
| H3 alignment | supported | tree beats perm in 6/6 (width, seed) pairs |
| H4 multi-match width | partly supported (r = 128 still improving) | 0.056 against 0.66, 3/3; r = 128 KL still falling 8–9% over the last 10,000 updates |
| H5 staging | descriptive | 3 dict first (all r = 128), 6 together, 3 loss first |
| Inconclusive override | did not apply | 0 of 48 runs unfinished |

## Cost

The run took 6.56 h at $0.59/h, about $3.87, against the definition's 10 h and $12 caps. The definition expected $2–5.

## Decisions I made

- **Stopped the first training job early instead of letting `pilot` exit on its own.** Once four pilots had run out of memory, `pilot` was certain to exit at the end of its pool. Letting it continue at 6 processes would have spent another 30–40 minutes on pilots, with more r = 128 runs failing the same way. I recorded the amendment and committed it before acting. Then I killed the job and reran `run.py` into the same `--out` with `--procs 3`, as the definition's "If something fails" prescribes.
- **Capped interpretation to end by 08:45 UTC instead of a fixed 3 h.** The pilots overran the plan, so a full 3 h of interpretation from about 06:15 would have finished too close to the worker's self-publish (09:38) and deletion (09:53). It turned out not to matter, since interpretation took 18 minutes. This is recorded as the second amendment and committed before it took effect.
- **Checked the interpretation output before trusting it.** It ran about 10× faster than the definition's estimate. I opened an `interpret.json` and confirmed it has probes, invariance and patching results at each checkpoint before finishing the worker.
- **Recorded the overall result as `supported`.** `orch experiment done` takes one label and the verdicts are mixed. I chose `supported` because the experiment answered its main width and alignment questions as predicted. The note and the Outcome section give the per-hypothesis split, including the H2 failure. `inconclusive` would have misstated a run that produced clear answers. `not-supported` would have hidden H1 and H3.
- **Reported convergence more strictly than the rule.** The definition's convergence rule calls retrieval r = 64 and r = 128 converged through its absolute 0.002-nat clause, although their KL was still falling 23–28%. The verdicts follow the rule as written. The write-up states the caveat next to the selected width.
- **Applied every fact-check correction.** The fact-checker found 11 of about 115 claims that didn't fully hold, and I corrected all of them in the README. The README's "How it was checked" lists the main ones.

## What you need to do

- **`EXPERIMENTS.md` is modified but not committed.** `orch experiment done` wrote this experiment's row to the shared experiment log. You asked me to leave the log to the planning session, so the change is sitting in the working tree for you or that session to commit.
- **Optional follow-ups**, from the README's "Not yet checked": a lower-LR, longer rerun at r ≥ 64, which would settle H4's second clause and the r = 64 vs 128 retrieval comparison (about 2–3 GPU-hours). And finer checkpoints for the r = 128 retrieval runs, to confirm the "dict first" staging (under 1 GPU-hour).
- Nothing was pushed.
