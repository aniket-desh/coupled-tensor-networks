# Degree under a constant-biased leaf init: run failed before training

This run gives no answer to the question of how the time to learn a product grows with the number of patches D when
each leaf's init favours the constant. The pre-training correctness checks failed on the worker, and the definition
says a failed check stops the experiment, so none of the 12 training groups ran and there are no t_half values,
slopes or symmetry-control results to report. The failing check traces to a non-reproducible least-squares solver in
a diagnostic measurement, not to the network or the training loss; it needs a one-line fix in shared code before
this experiment can be rerun unchanged.

## What failed

One check in `studies/ttn/checks.py` was out of tolerance; every other check with a tolerance passed (the full output
is in `results/checks.json`).

| Check | Value on the worker | Required |
|---|---|---|
| `alignment_at_teacher_min` | 0.0 | > 0.999 |

This check sets an r = 4 TTN to the exact teacher for the three-output `agreement('quarter')` teacher and asks, for
each of the 3 teacher components and each of the 6 internal nodes (q1 to q4, L, R), what fraction of the component's
part inside the node's subtree lies in the span of that node's activations. The fraction is an R^2 from a
least-squares fit (`ctn.ttn.alignment`), and at the teacher it should be 1 for all 18 entries. The check records only
the minimum over the 18, so which entry was 0 on the worker is not saved.

## Cause

At the teacher every node uses fewer than its r = 4 activation channels, so every least-squares problem in the check
is rank-deficient: q1 has two exactly-zero activation columns and q2 to q4, L and R one each (column norms 64, 64, 0,
0 and 64, 64, 64, 0 over 4096 samples). `torch.linalg.lstsq` on the CPU defaults to the `gelsy` driver (MKL here,
torch 2.11.0), and on these matrices its answer is not reproducible: the same call on the same matrices returns 1 for
some entries and 0, 1.2e-5 or values slightly above 1 for others, and which entries go wrong changes between calls and
between identical runs. The fact-checker saw this with a single thread as well. Forcing the `gelsd` driver (SVD
based) gave 1 for every entry in every call made.

`studies/ttn/degree/diagnose_alignment.py` computes the full 3 x 6 matrix three times with each driver; three fresh
runs on the coordinator's CPU are saved in `results/diagnose_alignment.txt`. Over those 9 calls per driver:

| Driver | Calls with min < 0.999 (check fails) | Calls with any entry off 1 by > 1e-6 | Lowest min |
|---|---|---|---|
| `gelsy` (default) | 4 of 9 | 7 of 9 | 0 |
| `gelsd` | 0 of 9 | 0 of 9 | 0.99999999999992 |

The wrong gelsy entries were spread over components and nodes (for example component 3 at q1 = 1.2e-5, component 2
at q1 = 0, component 1 at q4 = 1.002), and the check's second value, the alignment of an unrelated feature, stayed at
4.9e-4 with both drivers, well under its 0.02 limit.

The alignment is a diagnostic that the training loop logs (`ctn/ttn.py`); the degree experiment's criteria use only
half-times (`analyze.py`), so the failure does not point to a bug in the training or the population loss. The
definition treats any failed check as a stop, and that was followed.

## Likely fix (not applied)

Pass `driver='gelsd'` to `torch.linalg.lstsq` in `ctn.ttn.alignment` when the tensors are on the CPU (on CUDA only
`gels` is available, which needs full rank, so a GPU path would need a pseudo-inverse instead). Then rerun
`python -m studies.ttn.checks` and this experiment unchanged. The fix is in shared code outside this experiment's
folder, so it was left for the user.

## How it was checked

- The failure reproduces on the coordinator's CPU (MKL, torch 2.11.0) with the same seeds, which rules out a worker
  hardware fault.
- The whole alignment matrix of the check, with lstsq forced to gelsd, passed in 9 of 9 calls over 3 runs (table
  above); the check itself was not rerun with the change, since that means editing shared code.
- The worker's GPU and CUDA were working (`nvidia-smi` showed the A100-SXM4-80GB; torch 2.11.0+cu128 saw it).
- A fact-checking pass on the first draft found that it described gelsy's errors as a fixed function of call order
  (first call wrong, repeat right), said only q1 was rank-deficient, and said every other entry returned 1. Reruns
  showed all three were wrong; the Cause section and the diagnosis script were rewritten to report the full matrix
  over repeated runs.

## Not yet checked

- Whether the gelsy path has distorted alignment curves already logged by other TTN experiments. Their training is
  unaffected, but any alignment plot computed at a rank-deficient node could show spurious values, either drops toward
  0 or values slightly above 1. Checking means recomputing the alignment from saved parameters, not retraining.
- Why gelsy's answer varies between calls and between identical runs on a rank-deficient matrix. The diagnosis stops
  at showing that it does and that gelsd did not in the runs made.

## Cost

The worker (A100-SXM4-80GB, $2.19/h) existed about 3 minutes, an estimated $0.12 against the $12.05 budget.
