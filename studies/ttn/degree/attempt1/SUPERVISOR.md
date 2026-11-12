# Supervisor report: degree

The experiment failed before training. A pre-training correctness check failed on the worker, the definition says
that stops the run, so I finished the worker after about 3 minutes (about $0.12) and recorded the result as failed.
The cause is a non-reproducible least-squares solver in shared measurement code, and fixing it is the user's call
because it lives outside this experiment's folder.

## What ran

Worker `9lppud6dqcd8my` (A100-SXM4-80GB in US-KS-2, $2.19/h; the A100 PCIe was out of stock), job
`20261112-184153-degree-9lppud6dqcd8my`, launched 18:41 UTC on 2026-11-12. The definition's first three commands ran
as separate `orch exec` steps:

1. **nvidia-smi:** passed, showing the A100-SXM4-80GB.
2. **CUDA check:** passed (torch 2.11.0+cu128 saw the GPU).
3. **`python -m studies.ttn.checks`:** exited 1. `alignment_at_teacher_min` was 0.0 against a required > 0.999, and
   every other check passed (`results/checks.json`).

The training, analysis and plotting commands did not run. I finished the worker with `orch finish`, which merged
`results/checks.json` back and deleted the pod. `orch ls` afterwards listed only the sequential and sharing-location
workers, which belong to other supervisors.

## Result against each criterion

- **Checks:** failed, so the experiment counts as "Run failed" under the definition.
- **Symmetry control (w = 1/9, D = 1 vs 8):** not run.
- **Supported / not supported (slopes at w = 0.25 and 0.5):** not run.
- **Quantitative agreement with 0.425 and 0.901:** not run.

## Why the check failed

At the teacher every TTN node has exactly-zero activation columns, so the least-squares fits in `ctn.ttn.alignment`
are rank-deficient, and `torch.linalg.lstsq`'s default CPU driver (gelsy, MKL) returns wrong values (0, 1.2e-5 or
slightly above 1) for entries that change between calls and between identical runs. Over 3 runs × 3 calls on the
coordinator, the check would have failed in 4 of 9 calls with gelsy and 0 of 9 with gelsd. The details and the
reproduction script (`diagnose_alignment.py`) are in [README.md](README.md). The alignment is a logged diagnostic
that this experiment's criteria don't use, so the network and the training loss are not implicated.

## Cost

About 3 minutes of the worker, an estimated $0.12 against the $12.05 budget and the 5.5 h cap. The fact-checker
and the diagnosis ran on the coordinator, which costs no extra worker time.

## Decisions I made

- **Stopped on the failed check without trying a workaround.** The definition says "Checks fail or CUDA unavailable:
  stop, finish the worker, mark failed", and the failure was covered by that rule even though it turned out to be in
  a diagnostic. Patching the check or skipping it would have changed shared code and the experiment's contract
  without the user.
- **Finished the worker before diagnosing,** so the diagnosis ran on the coordinator instead of on billed GPU time.
- **Did not apply the fix.** It belongs in `ctn/ttn.py`, outside `studies/ttn/degree`, and other supervisors use the
  same file at the same time.
- **Added `diagnose_alignment.py` and `results/diagnose_alignment.txt` to the folder** so the diagnosis can be
  reproduced and the fix tested with one command.
- **Rewrote the write-up's diagnosis after the fact-checker pass.** The first draft said gelsy's errors followed call
  order and that only node q1 was affected. The fact-checker's reruns showed both were wrong, I confirmed that on the
  coordinator, and the README now reports the full alignment matrix over repeated runs.
- **Left EXPERIMENTS.md uncommitted.** It shows as modified (orch regenerates it), but the instructions leave the log
  to the planning session.

## What the user must do

1. **Fix `ctn.ttn.alignment`:** pass `driver='gelsd'` to `torch.linalg.lstsq` on the CPU (CUDA only has `gels`,
   which needs full rank, so a GPU path would need a pseudo-inverse). Then check it with
   `PYTHONPATH=. python studies/ttn/degree/diagnose_alignment.py` and `python -m studies.ttn.checks`.
2. **Rerun degree unchanged** once the checks pass. The definition needs no amendment. Its status is now done/failed,
   so a rerun will probably need it set back to approved, or a new definition copied from it.
3. **Watch the other experiments.** init-scale, sequential and sharing-location run the same `studies.ttn.checks`
   before training, and gelsy failed it in 4 of 9 calls on the coordinator, so they may stop the same way. Their supervisors will
   report that themselves; I didn't touch them.
4. **Check alignment plots from earlier TTN runs** if any conclusions rest on them, because gelsy values at a
   rank-deficient node can be spurious.
5. **Commit EXPERIMENTS.md** from the planning session when convenient.
