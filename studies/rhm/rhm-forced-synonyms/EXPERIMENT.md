---
name: rhm-forced-synonyms
title: Does a Bayes-optimal width-2 TreeTN on a 3-symbol RHM collapse synonyms, compose codes by the grammar, and respond to forbidden pairs?
status: approved
cpu: 2
count: 1
hours: 3
budget_usd: 1
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-17T22:33:34Z
---

# Does a Bayes-optimal width-2 TreeTN on a 3-symbol RHM collapse synonyms, compose codes by the grammar, and respond to forbidden pairs?

## Question

`rhm-ttn-messages` and `rhm-ttn-masks` asked whether a TreeTN trained on the RHM implements the grammar's message
passing and found that it does not: bonds are not linear in the message, and child pairs the grammar forbids keep as
much read-out mass as valid ones. Those experiments had no exact statement of what a Bayes-optimal tree *must* encode,
so a failed test could not be told apart from a wrong prediction.

This experiment uses a grammar small enough to prove that statement and then measures it. On the hand-built 3-symbol
grammar `sym` (below), at one prefix length, any width-2 tree at the Bayes loss must (a) give synonymous token pairs
the same bond, (b) compose the pair codes into the parent's code as the grammar does, and (c) give a **nonzero**
response to a forbidden pair, of a size fixed by two numbers that describe its codes. Does gradient descent find such
trees, do the trees it finds have these properties, and which of the allowed forbidden-pair responses does it pick? And
at widths 3 and 4, where none of this is forced, does it still happen?

**Where it runs.** On the coordinator's 2 vCPUs, in a tmux session, with no worker. This is a deviation from the
project's usual worker launch, forced on 2026-11-17 by two things: the toolkit creates and lists pods through
`rest.runpod.io`, and that host no longer resolves (NXDOMAIN from the zone's own name servers); and the Runpod key in
`/workspace/.env` was replaced that day by one for an account that does not hold the volumes named in `orch.toml`. So
`orch experiment launch` cannot start a worker. The experiment needs no GPU: every run is a full-batch problem with 24 inputs and a tree of at most 1,400
parameters. The `cpu`, `hours` and `budget_usd` lines above describe that local run; nothing is rented.

## Theory

Stated here so that the criteria below can be read without another document. `studies/rhm/toy.py` has the grammar and
`studies/rhm/toy_checks.py` verifies every numerical statement by enumeration.

- **Grammar `sym`.** Symbols 0, 1, 2 at every level; each parent produces the other two symbols in either order
  (0 → 12 | 21, 1 → 02 | 20, 2 → 01 | 10); depth 3, so 8 tokens; uniform root. The parent of a valid pair (b, c),
  b ≠ c, is the missing symbol. Pairs (b, b) are forbidden. Each pair symbol has two synonyms (its two orders).
- **Task.** The tree sees x_1..x_4, the last four leaves carry a mask token, and it predicts x_5. With A the symbol of
  the observed half, P(x_5 = y | x_1..4) = 1/4 if y = A and 3/8 otherwise. There are 24 legal prefixes, each of
  probability 1/24, and the population loss is computed exactly on them. A uniform predictor is 0.016417 nats above
  Bayes.
- **Model.** `ctn.tree.TreeTN(n=8, V=4, r, O=3)`, aligned: h_L, h_R are the bonds of the pairs (x_1 x_2), (x_3 x_4),
  h_P = W(h_L, h_R) the bond of the half, and the logits are bilinear in h_P and the constant bond m of the masked
  half. Softmax ignores a shift, so write read(h) for the logits of h_P = h centered over the outputs. The centered
  Bayes logits are F(A) = log(2/3)(e_A − 1/3): three vectors that sum to zero and span a plane.
- **T1, width 1 is not enough.** read is linear in h_P and must take three values spanning a plane.
- **T2, synonyms collapse (width 2).** read: R² → R³ hits three vectors spanning a plane, so it is injective, and
  prefixes with the same A have the same h_P. Fix the right pair to a realization of symbol c. The two other left
  symbols give two parents, so h_L ↦ read(W(h_L, h_R)) is a linear map of R² onto a plane, hence injective, and the two
  synonyms of a left symbol, which give the same prediction beside that right pair, have the same bond. So
  h_L = a_b and h_R = c_c depend only on the pair's symbol.
- **T3, composition (width 2).** With p_A the code of A on the half bond, W(a_b, c_c) = p_missing(b,c) for the six
  valid pairs, and p_0 + p_1 + p_2 = 0 (read is injective and the F(A) sum to zero).
- **T4, forbidden pairs respond (width 2).** W(a_b, ·) sends the two other right codes to two independent parent
  codes, so it is injective on R², and c_b ≠ 0, so W(a_b, c_b) ≠ 0.
- **T5, the response is fixed by two numbers (width 2; derived for this experiment).** Write a_0 = x a_1 + y a_2 and
  c_0 = u c_1 + w c_2 (a_1, a_2 are independent by T2). Expanding W(a_0, c_1) = p_2 and W(a_1, c_0) = p_2 gives
  x W(a_1, c_1) + y p_0 = p_2 = u W(a_1, c_1) + w p_0, so (u, w) = (x, y) and W(a_1, c_1) = (p_2 − y p_0)/x. Likewise
  W(a_2, c_2) = (p_1 − x p_0)/y and W(a_0, c_0) = x p_2 + y p_1. In units of a valid response, with
  ⟨F(A), F(A')⟩ = −|F|²/2:
  ρ_0 = √(x² + y² − xy), ρ_1 = √(1 + y + y²)/|x|, ρ_2 = √(1 + x + x²)/|y|.
  Any nonzero (x, y) is allowed by these equations; x = y = −1 (the three codes sum to zero) gives ρ = 1.
  **Corollary.** ρ_0 ≥ (√3/2)|x|, ρ_1 ≥ (√3/2)/|x| and ρ_2 ≥ (√3/2)/|y|, and likewise in y, so every two of the three
  responses have a product of at least 3/4: in any exact width-2 tree at most one forbidden pair responds below
  √3/2 ≈ 0.87 of a valid response. The smallest of the three is free: it ranges over (0, 1.43].
- **Two exact trees built by hand** (`toy.exact_complex`, `toy.exact_onehot`). Width 2: symbol a is the complex number
  e^(2πia/3), every core is (z, w) ↦ conj(zw); it has x = y = −1 and reads a forbidden pair (b, b) as b at full size.
  Width 3: one-hot symbol codes with the production table as the core, the grammar's own message passing; it sends
  forbidden pairs to zero. So at width 3 both a zero and a nonzero forbidden response are exact solutions.

T1–T5 are theorems about exact solutions. What the experiment adds is whether training reaches exact solutions, a
numerical test of the proofs and of the analysis code on trained trees, the distribution of (x, y) that training
selects, and what happens at widths 3 and 4.

## Hypothesis

"Exact" means an excess loss below 1e-10 nats after the polish. H2–H4 are judged on the exact width-2 runs and need
at least 10 of the 40; with fewer they are inconclusive.

- **H1 (T1).** No width-1 run ends below 1e-5 nats of excess loss. Against: any width-1 run below it, which would
  mean a bug in the target or the loss.
- **H2 (T2, T3): synonyms collapse and codes compose.** In every exact width-2 run: swapping a pair for its synonym
  changes the output by less than 1e-3 of the distance between two targets, under all six sibling bonds including the
  two that never occur beside it (`swap_L_all`, `swap_R_all`); the within-symbol share of each whitened bond
  (`within_L`, `within_R`, `within_P`) is below 1e-3; and all six valid compositions read out as the right parent
  with error below 1e-3 (`comp_acc` = 1, `comp_err`). Against: any exact run that violates one of these, which would
  mean an error in the proof or in the measures.
- **H3 (empirical): training does not silence any forbidden pair.** T4 and T5 force two of the three responses of an
  exact width-2 tree to be large and leave the smallest one free in (0, 1.43]. Take, per run, the smallest of its
  three responses ρ_b (`forb_rho`). Over the exact width-2 runs that Adam itself brought below 1e-6 nats (so that the
  solution was selected by Adam and only refined by the polish; at least 10 needed), the median of the per-run
  smallest response is at least 0.5 and no response is below 0.01. Against: a median below 0.5 (training tends to
  nearly silence one forbidden pair, moving toward the grammar's zero as far as width 2 allows) or any response below
  0.01. The same statistics over all exact runs are reported beside it.
- **H4 (T5): the response is the one (x, y) dictates.** In every exact width-2 run, (x, y) measured on the left bond
  equals (x, y) on the right bond to 1e-3, and the three measured ρ_b equal the formulas of T5 to a relative 1e-3.
  Against: a violation, which would mean an error in the derivation.
- **How a violation of H2 or H4 is reported.** These are theorems about exact trees, and a run at 1e-11 nats is only
  close to one: its swap effects scale as the square root of the excess loss times a factor that grows as (x, y)
  degenerates. A violating run is reported with its excess loss and its largest ρ (`summary.json` names the worst run
  per measure), and the verdict stays "not supported"; the write-up says which reading the numbers favor.
- **Descriptive, no criterion, with the reading fixed here.**
  - (i) The share of width-2 runs that become exact, per init, split into runs Adam found (excess below 1e-6 nats at
    update 20,000) and runs the polish found. In the pilot one identity-init run went from 2.4e-3 to 2.3e-17 in the
    polish; such a run's (x, y) was chosen by L-BFGS from a plateau and is not credited to gradient descent.
  - (ii) The distribution of (x, y) over exact width-2 runs, for the same two groups separately, the share within 0.1
    of (−1, −1) (does training pick the symmetric tree?), and how far the polish moved (x, y) (`xy_shift`).
  - (iii) Widths 3 and 4, where nothing is forced: the check builds an exact width-4 tree that keeps the synonym where
    only a forbidden context reads it, and at width 3 the one-hot tree has a zero forbidden response. Reading: a run
    **keeps the synonym** if its swap effect beside any sibling (`swap_*_all`) exceeds 1e-3 and is more than 10 times
    its swap effect beside siblings that occur (`swap_*_legal`, which is bounded by the prediction error); otherwise it
    has **collapsed**. A forbidden pair is **silenced** if its ρ is below 0.01. Counts are given per width and init,
    for exact runs and separately for runs between 1e-10 and 1e-5 nats. A width with fewer than 5 exact runs is
    reported as "not answered". `within_*` whitens every direction above 1e-3 of the largest, so at r ≥ 3 it is an
    indicator of whether any direction holds the synonym, read or not, and is labelled so.
  - (iv) The measures along training at width 2, against the excess loss.

## Method

- **Runs** (`run.py`): r ∈ {1, 2, 3, 4} × init ∈ {variance, identity (ε = 0.1)} × seeds 2000–2019 = 160 aligned trees on
  `sym`. Pilots and the smoke test used seeds below 1100 only.
- **Training** (`toy.train`, `toy.polish`): float64; full-batch Adam on the exact population loss (cross-entropy
  against the Bayes targets on the 24 prefixes), 20,000 updates, `train.lr_at`'s schedule (100 warm-up updates,
  cosine to 0.1 × peak), peak 1e-2 for the variance init and 3e-3 for the identity init (the best of 1e-3, 3e-3, 1e-2,
  3e-2 per init on pilot seeds 0–3 at r = 2). Then L-BFGS (strong Wolfe, at most 1,000 iterations) on the same loss:
  Adam stops near 1e-10 nats and the theorems are about exact solutions. Checkpoints at updates 0, 100, 300, 1000,
  3000, 10000, 20000 and after the polish. The number of L-BFGS iterations is saved, and runs that stop at the cap of
  1,000 (or 1,250 function evaluations) are counted as censored: they may not have converged.
- **Measures** (`analyze.py`, defined in its docstring): all taken through read, i.e. in output coordinates, so they
  do not depend on the basis of any bond; `within_*` whitens the bond instead. `forb_bond` is the one
  gauge-dependent number and is labelled so. Criteria use the polished trees; trajectories use every checkpoint.
- **Thresholds** were set after CPU pilots (seeds 0–3, 100–102 at 20,000 updates; seeds 1000–1001 at 2,000): exact
  runs ended at 1e-13 to 1e-17 nats and the others at 2e-3 or more; the H2 measures of exact runs were at 1e-16; the
  two exact width-2 smoke runs had ρ between 0.6 and 3.8 and matched T5 to 15 digits.
- **Figures** (`plot.py`, PDFs from saved results): `widths.pdf` (final excess loss per width and init),
  `collapse.pdf` (synonym-swap effect and composition error against the excess loss along training, r = 2),
  `forbidden.pdf` (measured against predicted ρ; the (x, y) of exact runs; ρ per width), `not_forced.pdf` (synonym
  information on the bonds of exact runs per width).

## Inputs

None. The grammar is written out in `studies/rhm/toy.py` and the 24 prefixes are enumerated.

## Commands

From the project root on the coordinator, as one script in tmux. `EXP_DIR=studies/rhm/rhm-forced-synonyms`.

```bash
set -euo pipefail
export EXP_DIR=studies/rhm/rhm-forced-synonyms PYTHONPATH=. OMP_NUM_THREADS=1
mkdir -p "$EXP_DIR/results" runs/rhm-forced-synonyms
uv run --frozen python -m studies.rhm.toy_checks "$EXP_DIR/results/toy_checks.json" | tee "$EXP_DIR/results/toy_checks.txt"
uv run --frozen python studies/rhm/rhm-forced-synonyms/analyze.py --check | tee "$EXP_DIR/results/measure_checks.txt"
uv run --frozen python studies/rhm/rhm-forced-synonyms/run.py --out runs/rhm-forced-synonyms --procs 2 | tee runs/rhm-forced-synonyms/train.log
uv run --frozen python studies/rhm/rhm-forced-synonyms/analyze.py runs/rhm-forced-synonyms "$EXP_DIR/results" | tee "$EXP_DIR/results/summary.txt"
uv run --frozen python studies/rhm/rhm-forced-synonyms/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

## Expected outputs

- `results/toy_checks.json`, `results/toy_checks.txt`, `results/measure_checks.txt`.
- `runs/rhm-forced-synonyms/<run>.pt`, 160 files of under 100 kB (config, loss log, 7 checkpoints, polished tree),
  and `train.log`.
- `results/runs.json` (every measure of every polished tree), `results/trajectories.json` (the measures at every
  checkpoint), `results/summary.json` and `results/summary.txt` (the verdicts), `results/diverged.json`,
  `results/provenance.json` (commit and library versions), and `results/polished_trees.pt` (every measured tree, under
  2 MB, so the numbers can be recomputed from git; `runs/` is git-ignored and its checkpoints can be regenerated from
  the seeds).
- `figures/widths.pdf`, `collapse.pdf`, `forbidden.pdf`, `not_forced.pdf`.
- `README.md` (the write-up, per the project's rules) and `NOTE.md` (a research note with the key figures as PNGs
  under `figures/png/`, as in the last two batches).

## Controls

- **Hand-built trees through the same measures** (`analyze.py --check`, run first): the width-2 complex tree must give
  zero swap effect, 6/6 compositions, ρ = 1, x = y = −1; the width-3 one-hot tree must give ρ = 0; a width-4 tree that
  hides the synonym where only a forbidden context reads it must give `swap_L_legal` = 0 and `swap_L_all` > 0; a
  random tree must fail; and every measure except `forb_bond` must be unchanged by a random change of basis of all
  bonds.
- **Width 1** is the floor for capacity (H1), and the uniform predictor (0.016417 nats) the floor for learning.
- **Two inits** with the same seeds, so no result rests on one starting point.
- **Runs that are not exact** are reported with their counts and final losses and never enter H2–H4.
- **`swap_*_legal` next to `swap_*_all`**: the legal version is bounded by the prediction error and must vanish in
  any exact tree; only the `all` version tests the collapse.

## Success and failure criteria

- H1–H4 as stated under Hypothesis, computed by `analyze.py` (`verdicts`), on the polished trees.
- **Overall:** supported if H1–H4 are all supported; not supported if any of H2–H4 fails on at least 10 exact
  width-2 runs; inconclusive if fewer than 10 width-2 runs are exact.
- **The run failed** if a check fails, or fewer than 150 of the 160 runs produce a finite tree (`analyze.py` prints
  `RUN FAILED` and sets `run_complete` false). A run that raises writes a `.err` file and does not stop the grid.
- Runs with an excess loss between 1e-10 and 1e-5 are neither exact nor floor: they are counted and reported.

## Time and cost

13 s per run (20,000 updates and the polish, one thread, measured on the coordinator at r = 4), so 160 runs take about
18 minutes with 2 processes; checks, analysis and figures add under 5 minutes. No rented hardware: $0. Each process
uses about 400 MB of the coordinator's 8 GB. If the timing comes out 1.5 times over, nothing is dropped.

## If something fails

- **A check fails:** stop; it is not a result. Report which.
- **The session is interrupted:** rerun the same script. `run.py` skips finished runs (each is written to a temporary
  file and renamed when complete), so at most the two runs in progress are lost.
- **A run ends with a non-finite loss:** `toy.train` stops it, `toy.polish` keeps Adam's endpoint if the line search
  fails; the run is saved, listed in `results/diverged.json`, counted as not exact and never measured.
- **A run raises:** its traceback goes to `runs/rhm-forced-synonyms/<run>.err`, the grid continues, and the count is
  in the summary. Read the traceback; rerunning the command retries only the missing runs.
- **Out of memory, or a hang:** `multiprocessing.Pool` does not notice a killed worker. If `train.log` gains no line
  for 5 minutes, stop the command and rerun it with `--procs 1`.
- Nothing here needs the user. A change to a threshold or to the grid would, and is not to be made.

## Fixed rules

- The thresholds (1e-10 exact, 1e-5 floor, 1e-6 for "found by Adam", 1e-3 for H2 and H4, 0.01 and 0.5 for H3, the
  "keeps the synonym" rule), the gates of 10 exact width-2 runs and 5 exact runs per wider width, the grid and the
  seeds are fixed.
- H2–H4 are judged on polished trees of exact width-2 runs only; widths 3 and 4 carry no criterion.
- No run is dropped; non-exact runs are counted.

## Questions asked before launch

None asked. The user supplied the two grammars and the predictions (their message of 2026-11-17, a worked answer to
"what theory would you want solved") and said to develop the experiments, run them, and batch them in tmux, which
covers approval. Choices made without asking: the L-BFGS polish; two inits; widths 3 and 4 as a descriptive arm; T5,
its corollary, H3's statistic and H4, which are derived here and were not in the user's text; running on the coordinator because workers cannot be
launched today.

## Review

Reviewed once by the `experiment-reviewer` agent before launch (2026-11-17). No P0. It ran both check scripts and
verified the T5 algebra by hand. Findings and what changed:

- **P1, H3's median could not fail.** T5 implies every two responses have a product of at least 3/4, so the pooled
  median of ρ is at least 0.87 in any exact width-2 tree. Changed: the bound is now stated as a corollary of T5 and
  checked on a grid in `analyze.py --check`; H3 is now about the per-run smallest response, which T5 leaves free.
- **P1, the polish sometimes does the optimization.** Changed: a cut of 1e-6 nats on Adam's own endpoint is fixed;
  H3 is judged on runs Adam found; counts and (x, y) are reported separately for the two groups, with the shift of
  (x, y) under the polish.
- **P1, widths 3 and 4 had no stated reading and `within_*` is binary there.** Changed: the "keeps the synonym" rule,
  the per-width minimum of 5 exact runs and the counts for runs between the thresholds are fixed above; the whitening
  cutoff went from 1e-8 to 1e-3 and `within_*` is labelled an indicator; `not_forced.pdf` now plots the swap effect
  beside any sibling against the one beside siblings that occur.
- **P2, a non-finite run crashed the analysis.** Changed: such runs are listed and skipped.
- **P2, the 150-run rule had no code behind it.** Changed: per-run exceptions are caught, and `analyze.py` checks the
  count.
- **P2, a borderline exact run could flip H2 or H4.** Changed: how a violation is reported is stated above, and the
  summary names the worst run per measure with its excess loss and largest ρ.
- **P2, provenance.** Changed: `provenance.json` and `polished_trees.pt` are written into `results/`.
- **P2, L-BFGS cap not recorded.** Changed: iterations are saved and runs at the cap are counted.

The revised definition was not sent back for a second pass: every change follows the reviewer's stated smallest fix,
and the revised code was rerun on the checks and the smoke grid.

## Amendments

None.

<!-- Each change made during the run, with the time and the reason, recorded and committed before acting on it. -->

## Outcome

Not run yet. Afterwards: one line per criterion with what was observed, the cost and runtime against the budget, and a link to the write-up. The write-up holds the detail; don't repeat it here.
