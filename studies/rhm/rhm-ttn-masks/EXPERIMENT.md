---
name: rhm-ttn-masks
title: Does training a TreeTN on varied observation masks, not only prefixes, make its merges compose uncertain messages?
status: done
gpu: NVIDIA RTX PRO 4500 Blackwell,NVIDIA RTX A6000,NVIDIA A40
count: 1
hours: 10
budget_usd: 10
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-15T19:15:43Z
result: inconclusive
finished_utc: 2026-11-16T00:59:22Z
writeup: studies/rhm/rhm-ttn-masks/README.md
---

# Does training a TreeTN on varied observation masks, not only prefixes, make its merges compose uncertain messages?

## Question

`rhm-ttn-messages` found that a TreeTN trained on RHM prefix prediction does not implement the grammar's message
passing: on uncertain child messages its merge does not reproduce the exact update (H4 failed in 9 of 9 learned cells,
both-informative R² 0.17–0.58 against a 0.8 bar), and the invalid child pairs, which the exact update sets to zero,
carry as much read-out mass as the valid ones (ratio 0.6–2.3 at r = 32).

One explanation is the training distribution. Under prefix masks a parent's right child sees tokens only once its left
child is fully observed, so a parent never receives two uncertain children, and nothing in the loss constrains what
its core does with jointly inconsistent child symbols. This experiment tests that explanation directly: with the
model, grammars, seeds, sequence stream and optimizer held fixed, does adding observation patterns in which both
children are informative make the merge compose uncertain messages correctly?

A pass is the informative outcome: the earlier failure is then a property of the prefix task, and the learned
algorithm follows the observation patterns seen in training. A failure is weaker evidence, for the reasons under
"What a failure does and does not show" below.

## Hypothesis

- **H_A (primary), mixed masks give the grammar's update.** In the `mixed` arm, H4 of `rhm-ttn-messages` holds:
  over parents of tree level ≥ 2, the merge of mixture bonds decodes to the exact normalized update with R² ≥ 0.8 over
  independently paired empirical messages and R² ≥ 0.8 on the both-informative pairs, closing the gap there over the
  identity-init tree and the additive ceiling (1 − R² ≤ 0.5 × (1 − max(control, 0))), in all three seeds of at least 3
  of the 4 grammars, while the `prefix` arm does not reach that.
  - Against: the `mixed` arm passes H4 in no grammar and does not close the gap over the `prefix` arm, although it
    learned the varied task (varied excess loss < 0.02 nats/token in all three seeds) in at least 3 grammars.
- **H_B (secondary), invalid pairs are suppressed.** The invalid pairs' read-out mass ratio (`m3_invalid_q`, the
  median nonnegative read-out mass of invalid child pairs over that of valid pairs; 0 for exact messages) is lower in
  the `mixed` arm than in the `prefix` arm in at least 10 of the 12 (grammar, seed) pairs, and the median of the paired
  ratios mixed / prefix is at most 0.5. Only pairs where both arms pass the prefix gate count. A sign test alone would
  pass on a drop from 1.25 to 1.20, against an exact value of 0 and an earlier range of 0.6–2.3.
- **What a pass does and does not show.** In the `mixed` and `varied` arms the both-informative combinations occur in
  training, so a pass on them is in-distribution: it shows that a TreeTN can learn calibrated composition when the
  loss asks for it, which the prefix task did not. It would not show that prefix-trained trees compose messages.
  The M4 inputs are still mixtures of class-mean bonds, which no arm is trained on.
- **What a failure does and does not show.** A failed H4 in the `mixed` arm does not by itself count against the
  training-distribution explanation, for three reasons the review established:
  - M4 never sees a bond from a varied row. `msg_analyze.py` collects every row on prefix-pattern inputs, and M4's
    inputs are mixtures of fully observed class means decoded by a probe fit on those rows. A tree that composes real
    uncertain bonds correctly in another code fails it, and M4a does not separate the two cases.
  - The varied loss does not ask exactly for the message code. Logits are multilinear in every bond and the Bayes
    log-odds are log-ratios of linear forms in an uncertain message. The reviewer computed, with no network, the least
    excess loss of any readout affine in the right half's message on rows with an uncertain right half
    (`v3m2s2L4`, `v8m2s2L4`, `v8m4s2L4`, seed 0): 0.000–0.011 nats per row for the normalized message and 0.001–0.027
    for the unnormalized one. So on varied rows a softmax-readout tree may be unable to both hold exact messages
    and reach the Bayes loss. (On prefix rows every message outside the query's path is a point mass or all-ones, and
    the conflict does not arise.)
  - The tree must have learned the varied task for the test to bear on composition. That is why "not supported" needs
    the varied-loss condition above, and a failed H4 without it is recorded as inconclusive.
- **Follow-up whose reading is fixed now** (not run here; it needs only the saved checkpoints): the M1 projective
  probe from a bond to its normalized message λ̂_u on bonds collected from varied-pattern inputs, at nodes to the right
  of the query, scored on held-out patterns. If it reaches R² ≥ 0.8 in the `mixed` arm where the `prefix` arm does
  not, the mixed tree carries uncertain messages even where M4 fails.
- **Honest expectation.** The tree needs more than λ_u in a partially observed bond (the extended message and the
  query's place), and the varied task adds the observed leaves to the right of the query. A negative on H4 is the
  more likely outcome.
- **Descriptive:** the `varied` arm (a varied row is a pure prefix row with probability 1/(T − q), 21% on average at
  T = 16, so it sees fewer prefix rows than `mixed`, which has about 61%; it may not reach the prefix gate and is then
  censored); H1–H3 and every other measure of `msg_summary.py` per arm; the measures at update 3,000.

## Method

- **Grammars:** `v3m2s2L4`, `v8m2s2L4`, `v8m4s2L4`, `v16m4s2L4` (T = 16), the four T = 16 grammars of
  `rhm-ttn-messages`, all learned there at r = 32 in all three seeds. `v4m2s2L5` (T = 32) is left out: its runs take
  67 minutes each against 18.
- **Width and seeds:** r = 32; seeds 0, 1, 2 set the rules and the init. 4 grammars × 3 arms × 3 seeds = 36 runs.
- **Task and arms** (`mask_run.py`): the tree sees a sequence with some leaves replaced by the mask token and predicts
  the token under the leftmost mask. A row is (sequence, query position q); positions < q are observed, q is masked.
  - `prefix`: every position > q is masked. These are `ttn_run.py`'s rows in the same order (asserted in the check).
  - `varied`: each position > q is masked independently with probability u, with u ~ U(0, 1) per row.
  - `mixed`: each row is a prefix row or a varied row with probability 1/2.
  - The prefix rows are the u = 1 case of the varied rows, so the input format, model and read-out are the same in
    all arms, and every arm is scored on the prefix task.
- **Model and training:** as `rhm-ttn-messages`: the aligned untied TreeTN with identity init (ε = 0.1), every query
  position of 512 fresh sequences per update, Adam at peak LR 1e-3 with 100 warm-up updates, 30,000 updates,
  checkpoints at 0, 3000 and 30000 (the criteria use the first and the last; `rhm-ttn-messages` saved six, and its
  analysis time grows with their number). The sequence stream (generator 1000 + seed) and the init are the
  same in the three arms; masks come from a separate generator (2000 + seed).
- **Losses:** the prefix test loss against the exact Bayes loss on `ttn_run.py`'s test set (8,192 sequences), and the
  varied-task loss against its exact Bayes loss on the first 2,048 of them with fixed masks (seed 3000 + seed). The
  exact posterior under an arbitrary mask is `posterior_masked` in `mask_run.py`.
- **Analysis:** `rhm-ttn-messages/msg_analyze.py`, unchanged, on every run (20,000 analysis sequences, rows collected
  on prefix-pattern inputs, which are training inputs in every arm), then `msg_summary.py`, unchanged, per arm, and
  `mask_summary.py` for the comparison and the verdicts on H_A and H_B.
- **Figures** (`mask_plot.py`, PDFs from saved results only): the both-informative R² and the invalid read-out ratio
  per grammar, arm and seed (`summary.pdf`); prefix and varied excess loss over training per arm (`training.pdf`);
  and `msg_plot.py`'s production tables for seed 0 of each arm.

## Inputs

None. All data is generated from the seeded grammars. The code calls `ttn_run.py`
(`studies/rhm/rhm-ttn-belief-geometry/`), `msg_analyze.py`, `msg_summary.py`, `msg_plot.py` and `msg_check.py`
(`studies/rhm/rhm-ttn-messages/`), all read-only.

## Commands

From the project root on the worker. Checks:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
mkdir -p "$EXP_DIR/results"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_check.py | tee "$EXP_DIR/results/msg_check.txt"
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-masks/mask_run.py --check | tee "$EXP_DIR/results/mask_check.txt"
```

Training, four runs at a time, seed by seed so that a time cut leaves complete (grammar, seed) triples:

```bash
set -euo pipefail
mkdir -p runs/rhm-ttn-masks/logs
for s in 0 1 2; do for c in v16m4s2L4 v8m4s2L4 v8m2s2L4 v3m2s2L4; do for a in prefix mixed varied; do echo "$c $a $s"; done; done; done \
  | xargs -P 4 -n 3 sh -c 'OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-masks/mask_run.py "$0" --arm "$1" --r 32 --seed "$2" --out runs/rhm-ttn-masks > runs/rhm-ttn-masks/logs/train_$0_$1_s$2.log 2>&1 || echo "FAILED $0 $1 s$2"'
ls runs/rhm-ttn-masks/*/*/result.json | wc -l
```

Analysis of the `prefix` and `mixed` arms, six at a time (each analysis used 3.6 GB of GPU memory in
`rhm-ttn-messages`), then the verdicts and figures, as one `orch exec` step:

```bash
set -euo pipefail
analyze() { for a in "$@"; do ls -d runs/rhm-ttn-masks/$a/*_r32_s[012]; done \
  | xargs -P 6 -I{} sh -c 'a=$(basename $(dirname {})); r=$(basename {}); OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_analyze.py {} "$EXP_DIR/results/$a/$r" > runs/rhm-ttn-masks/logs/analyze_${a}_$r.log 2>&1 || echo "FAILED analyze $a $r"'; }
summarize() { for a in "$@"; do PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_summary.py "$EXP_DIR/results/$a" 12 | tee "$EXP_DIR/results/$a/summary.txt"; done
  PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-masks/mask_summary.py "$EXP_DIR/results" runs/rhm-ttn-masks studies/rhm/rhm-ttn-messages/results/summary.json | tee "$EXP_DIR/results/summary.txt"
  PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-masks/mask_plot.py "$EXP_DIR/results" runs/rhm-ttn-masks "$EXP_DIR/figures"
  for a in "$@"; do PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_plot.py "$EXP_DIR/results/$a" "$EXP_DIR/figures/$a"; done; }
analyze prefix mixed
summarize prefix mixed
```

Then the `varied` arm and the summary again, as a second step (the same two function definitions first), started only
if at least 2.5 hours remain before the cap on the RTX PRO 4500, or 4 hours on either other GPU, whose speed is
unmeasured. The first step's summary prints the arms analysed so far; its `run failed` line counts only those arms.

```bash
analyze varied
summarize prefix mixed varied
```

## Expected outputs

- `results/checks.json`, `results/msg_check.txt`, `results/mask_check.txt`.
- `runs/rhm-ttn-masks/<arm>/<cfg>_r32_s<seed>/`: `result.json` and 3 checkpoints of about 2 MB each; logs in
  `runs/rhm-ttn-masks/logs/`. About 250 MB in total.
- `results/<arm>/<cfg>_r32_s<seed>/analysis.json` and two `arrays_ckpt*.npz`; `results/<arm>/summary.json` and
  `summary.txt`; `results/summary.json` and `summary.txt`. Under 150 MB.
- `figures/summary.pdf`, `figures/training.pdf`, `figures/<arm>/*.pdf`.
- `README.md`, the write-up, and `NOTE.md`, a research note with the key figures embedded as PNGs under
  `figures/png/`.

## Controls

- **The prefix arm in the same run:** same grammars, seeds, init, sequence stream and optimizer; only the masks
  differ. It also repeats 12 runs of `rhm-ttn-messages` (same script arguments and seeds), and `mask_summary.py`
  prints the two side by side: a check on the new training script.
- **Prediction gates:** a grammar is judged for an arm only if all three seeds reach prefix excess loss < 0.02
  nats/token, so no arm can pass by predicting worse. "Not supported" on H_A also needs the `mixed` arm to reach
  varied excess loss < 0.02 nats/token in all three seeds of at least 3 grammars; the varied excess is reported for
  every arm.
- **Exact reference tree** (`msg_check.py`): the hand-built message-passing tree scores the ideal value on every
  measure and a random-core tree fails M3 and M4, which validates the analysis code on this worker.
- **`mask_run.py --check`:** the exact posterior under masks equals the Bayes prefix loss on prefix rows and
  `grammar.posterior_token` when only the query is hidden; the prefix arm's rows equal `ttn_run.prefixes`; in the
  varied rows the leftmost mask is the query and the observed tokens are the sequence's.
- **Identity-init tree and additive ceiling:** inside H4, as in `rhm-ttn-messages`.

## Success and failure criteria

All at the final checkpoint, from `mask_summary.py`.

- **H_A supported:** the `mixed` arm's H4 verdict is "supported" (H4 in all three seeds of at least 3 of the 4
  grammars) and the `prefix` arm's is not.
- **H_A partly supported:** `mixed` passes H4 in 1 or 2 grammars, or closes the gap over `prefix` on the
  both-informative R², seed by seed, in all three seeds of at least 3 grammars.
- **H_A not supported:** neither, and the `mixed` arm learned the varied task (varied excess loss < 0.02 nats/token
  in all three seeds) in at least 3 grammars.
- **H_A inconclusive:** fewer than 3 grammars are learned (prefix gate) by the `mixed` arm, or H4 fails and the `mixed`
  arm learned the varied task in fewer than 3 grammars.
- **H_B supported:** over the pairs where both arms pass the prefix gate, `m3_invalid_q` is lower in `mixed` than in
  `prefix` in at least 10 of 12 pairs and the median paired ratio mixed / prefix is at most 0.5; not supported
  otherwise; inconclusive with fewer than 12 such pairs. The paired ratios are reported with their range.
- **Overall result recorded:** H_A's verdict.
- **The run failed** if a check fails, if fewer than 31 of the 36 runs are trained, if fewer than 86% of the runs of
  the arms analysed are analysed (a `varied` arm censored for time is not a failure), or if the prefix arm
  disagrees with `rhm-ttn-messages` on the prefix excess loss by more than 0.005 nats/token in any shared run.
  `mask_summary.py` computes the last two and prints them as `run failed`.

## Time and cost

- Checks and setup: about 20 minutes. Training: 36 runs at about 18 minutes each, four at a time, about 2.7 hours.
- Analysis: `rhm-ttn-messages` took 4 h 41 m for 30 runs with six checkpoints each, mostly six at a time (17 to 75
  minutes per run by grammar). Here 36 runs have three checkpoints each and run six at a time from the start: about
  3 to 4 hours, of which `prefix` and `mixed` are two thirds. Total about 6 to 7 hours.
- At $0.72/hr (RTX PRO 4500 Blackwell, in stock in EU-CZ-1 and EU-RO-1 when this was written) that is about $4.30 to
  $5.00, and the 10-hour cap costs $7.20. The other two listed GPUs are cheaper ($0.69 and $0.59/hr) but unmeasured:
  every timing above is from the RTX PRO 4500. The budget is $10.

## If something fails

- **A check fails:** stop; that is a failed run, not a result.
- **A training run fails or diverges:** the others continue; rerun it once. Runs that still fail are reported as
  censored.
- **Out of memory with four runs at a time:** drop to two (`-P 2`) and note it as an amendment.
- **Time running out:** training goes seed by seed. The `prefix` and `mixed` arms are analysed and summarized first,
  as their own step, so their results exist before `varied` starts. The `varied` step starts only if at least 2.5
  hours remain (4 hours on a GPU other than the RTX PRO 4500); otherwise `varied` is reported as censored. Any other change to the method or criteria waits for the
  user.
- **Out of GPU memory in the analysis at six at a time:** drop to four and note it as an amendment.
- **No GPU in stock:** try the next in the list; none of the listed types exceeds the budget at the cap.

## Questions asked before launch

None needed. The user approved the plan in conversation on 2026-11-15 ("go ahead with these"): three arms, the
grammars the tree learned in `rhm-ttn-messages`, r = 32, three seeds, the existing M1–M4 tests unchanged. Two choices
were made here without asking, both within that plan: the query is the leftmost mask (no new query symbol, so the
prefix arm is the earlier experiment's task exactly), and the T = 32 grammar is left out for cost.

## Review

Reviewed on 2026-11-15 by the `experiment-reviewer` agent, which also ran `mask_run.py --check` and a tiny dry run of
every command (v4m2s2L3, r = 8, 400 updates). Verdict: needs revision. Findings and what changed:

- **P0, the analysis would overrun the cap.** The earlier experiment's analysis was 2.6 times slower than this
  definition assumed, giving about 9.7 hours against an 8-hour cap, with the summary after the analysis in the same
  step. Changed: three checkpoints per run instead of six, six analyses at a time, `prefix` and `mixed` analysed and
  summarized as their own step before `varied`, a 10-hour cap, and the 24 GB RTX 4090 dropped from the GPU list.
- **P1, a negative on H_A could not carry the reading given to it.** Changed: "not supported" now needs the `mixed`
  arm to have learned the varied task; the definition states what a failed H4 does not show (M4 uses no varied-row
  bond; the softmax readout may not admit exact messages at the Bayes loss); the claim that M4a separates the cases
  is removed; and a follow-up on varied-pattern bonds has its reading fixed here. The follow-up is not run in this
  experiment, so a failed H4 stays weak evidence until it is.
- **P1, H_B passed on an effect of any size.** Changed: the median paired ratio must be at most 0.5, and only pairs
  where both arms pass the prefix gate count.
- **P2, run-failed conditions were not in code.** Changed: `mask_summary.py` computes and prints them.
- **P2, the `varied` arm was described wrongly.** Changed: a varied row is a prefix row with probability 1/(T − q).
- **P2, an early stop if the mixed arm misses the prefix gate at seed 0.** Left as is: the runs are cheap and the
  other seeds still give the descriptive measures.
- **P2, timings come from one GPU type.** Noted under "Time and cost".

Re-check of the revision by the same reviewer: ready to run; the P0 and both P1s are resolved and nothing new at P0
or P1. It ran the revised `mask_summary.py` on fabricated summaries and each branch gave the stated verdict. Its
three optional notes were applied: the `run failed` count now covers only the arms analysed so far, "in any arm" was
corrected to "on varied rows", and the margin before starting `varied` is larger on the unmeasured GPUs.

## Amendments

None.

## Outcome

Run on 2026-11-15 and 2026-11-16 on one NVIDIA A40. Write-up: [README.md](README.md); research note: [NOTE.md](NOTE.md); the supervising session's report: [SUPERVISOR.md](SUPERVISOR.md).

- **H_A: inconclusive.** The `mixed` arm passes H4 in 0 of 4 grammars (0 of 12 runs), as does `prefix`; it closes the gap over `prefix` in 0 grammars (1 run of 12); and it learned the varied task in 2 grammars (`v3m2s2L4`, `v8m2s2L4`), where "not supported" needs 3.
- **H_B: not supported.** `m3_invalid_q` is lower in `mixed` than in `prefix` in 6 of 12 pairs (10 needed), and the median paired ratio is 1.10 (0.5 at most needed), range 0.53 to 1.74.
- **Overall result recorded:** inconclusive (H_A's verdict).
- **The run did not fail.** Every check passed; 36 of 36 runs trained and 36 of 36 were analysed, the `varied` arm included; the prefix arm's prefix excess loss agrees with `rhm-ttn-messages` to 1.1e-8 nats/token in the 12 shared runs.
- **Descriptive:** the `varied` arm misses the prefix gate in `v16m4s2L4` and passes H4 in 0 of its 3 judged grammars; H1 to H3 have the same verdicts in the `prefix` and `mixed` arms.
- **Cost and runtime:** 5 h 26 m and about $3.21, against a 10-hour cap, a $10 budget and an estimate of 6 to 7 hours.
