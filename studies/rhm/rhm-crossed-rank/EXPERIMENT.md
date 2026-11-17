---
name: rhm-crossed-rank
title: Does aligning the tree with the grammar save bond dimension? Aligned vs crossed TreeTN on two hand-built 3-symbol RHMs
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

# Does aligning the tree with the grammar save bond dimension? Aligned vs crossed TreeTN on two hand-built 3-symbol RHMs

## Question

`rhm-controls` and `rhm-belief-architectures` found that a tree aligned with the grammar learns the RHM where a tree
with permuted leaves does not, but could not separate two readings: the permuted tree lacks capacity at that width, or
it has the capacity and training does not find it (the alignment comparison fell one win short of its bar, at one
learning rate). This experiment uses two hand-built grammars where the capacity question has an exact answer. For a
fixed context the logits are linear in every bond, so the matrix of centered Bayes logits with rows indexed by a
subtree's tokens and columns by (the other tokens, the output) has rank at most r. On grammar `mod` that gives
r ≥ 3 for the aligned tree, which a hand-built tree attains, and r ≥ 4 for a tree that pairs x_1 with x_3 and x_2 with
x_4. On grammar `sym` both groupings need only r = 2. Does training reach the exact predictor exactly where the bounds
allow it, is the crossed tree's bound of 4 attained, and where a grouping has the capacity, is it as easy to train?

**Where it runs.** On the coordinator's 2 vCPUs, in a tmux session, with no worker. This is a deviation from the
project's usual worker launch, forced on 2026-11-17 by two things: the toolkit creates and lists pods through
`rest.runpod.io`, and that host no longer resolves (NXDOMAIN from the zone's own name servers); and the Runpod key in
`/workspace/.env` was replaced that day by one for an account that does not hold the volumes named in `orch.toml`. So
`orch experiment launch` cannot start a worker. The experiment needs no GPU: every run is a full-batch problem with 24 inputs and a tree of at most 1,400
parameters. The `cpu`, `hours` and `budget_usd` lines above describe that local run; nothing is rented.

## Theory

`studies/rhm/toy.py` has the grammars and `studies/rhm/toy_checks.py` verifies every numerical statement by
enumeration.

- **Grammars.** Three symbols at every level, two productions each, depth 3 (8 tokens), uniform root. `sym`: each
  parent produces the other two symbols in either order at every level (0 → 12 | 21, 1 → 02 | 20, 2 → 01 | 10).
  `mod`: the same at the two upper levels, and at the bottom level 0 → 12 | 21, 1 → 02 | 22, 2 → 01 | 11. A parent's
  first child has the same distribution in both, so the target is the same function of the half's symbol A.
- **Task.** The tree sees x_1..x_4, the last four leaves carry a mask token, and it predicts x_5:
  P(x_5 = y | x_1..4) = 1/4 if y = A, else 3/8. 24 legal prefixes of probability 1/24 each; the population loss is
  exact. A uniform predictor is 0.016417 nats above Bayes.
- **Groupings.** `aligned`: level-1 bonds see (x_1, x_2) and (x_3, x_4). `crossed`: `TreeTN(perm=[0, 2, 1, 3, ...])`,
  level-1 bonds see (x_1, x_3) and (x_2, x_4). Same parameter count.
- **Width lower bounds** (`toy.cut_bound`, `toy.tree_bound`). Only legal prefixes have a target, so the bound for a
  bond is the largest rank of a submatrix all of whose entries are legal, found exhaustively; the tree's bound is the
  maximum over its bonds. Results: `sym` aligned 2, `sym` crossed 2, `mod` aligned 3 (from the leaves x_1 and x_3),
  `mod` crossed 4 (from the bond (x_1, x_3): rows 00, 01, 02, 10 against columns (x_2 x_4, y) = (12, 0), (12, 1),
  (21, 0), (21, 1) form a legal 4 × 4 block with determinant −27 in units of log(2/3)/3).
- **Attained by hand:** `sym` at width 2 in both groupings (symbols as cube roots of unity, cores (z, w) ↦ conj(zw);
  on `sym` A = x_1 + x_2 + x_3 + x_4 mod 3, which any grouping can add up), and `mod` aligned at width 3 (one-hot
  symbols and the production table). No exact crossed tree for `mod` is known in advance; width 9 suffices trivially.
- A cell below its bound cannot be exact, and since the set of matrices of rank at most r is closed its excess loss is
  bounded away from zero. The size of that gap is not known in advance.
- The pairing the grid does not train, (x_1 x_4)(x_2 x_3), has the same bounds (4 on `mod`, 2 on `sym`; the review
  computed them), so the bound is not special to the crossed pairing chosen.
- The reverse does not follow from a small loss alone: the target is fixed only on the 24 legal prefixes, so a loss
  near zero could in principle be approached with diverging weights. An exact cell is therefore reported with the
  largest weight and the largest centered-logit error of its best tree.

## Hypothesis

A cell is one (grammar, grouping, width) with 20 runs. "Exact" means an excess loss below 1e-10 nats after the polish.
Reaching the exact predictor is an existence statement, so a cell is exact if any of its runs is.

- **H1, the bounds hold (a theorem; a test of the code).** In the cells below their bound (`sym` r = 1 in both
  groupings; `mod` aligned r = 1, 2; `mod` crossed r = 1, 2, 3), no run ends below 1e-5 nats. Against: any such run,
  which would mean an error in the bound or the target.
- **H2, alignment saves width on `mod`.** At r = 3 the aligned cell has at least one exact run and the crossed cell
  has none, with every crossed run above 1e-5 nats. Against: no exact aligned run at r = 3 among 20 (training cannot
  find the solution that exists, so the comparison says nothing about capacity), which is reported as "not
  supported by training" and kept apart from the bound, which stands regardless.
- **H3, existence control: on `sym` the grouping costs no width.** Both groupings have at least one exact run at
  r = 2. This shows only that the crossed tree's training can reach an exact solution where one exists; it says
  nothing about how often (that is the trainability reading below). Against: the crossed cell has none of 20, which
  would leave H2's crossed cell open to a training explanation.
- **H4, the crossed bound is attained on `mod`.** The crossed cell at r = 4 has at least one exact run, and that run's
  largest centered-logit error is below 1e-4 (a finite tree that attains the target, with its largest weight
  reported). If none of 20 is exact, tightness stays open; it is not refuted. With H1 and H2 this gives the exact
  cost of crossing: one bond dimension, 3 against 4.
- **Trainability at the bound, a pre-stated reading with no predicted direction.** Each grouping is compared at its own
  bound, where both have the capacity: on `sym`, aligned r = 2 against crossed r = 2; on `mod`, aligned r = 3 against
  crossed r = 4. The statistic is the number of exact runs of 20, compared by a two-sided Fisher exact test. A
  grouping is "harder to train at its bound" on a grammar if its count is lower with p < 0.05; otherwise "no
  detectable difference". With 20 runs per cell only large gaps are detectable (about 14 against 7), and the counts
  are also given per init, where the pilot shows the init matters (`mod` crossed r = 4: 3 of 3 with the variance init,
  0 of 3 with the identity init). The learning-rate pilot under Method says whether the rates favor one grouping.
- **Descriptive, no criterion.** (i) The smallest exact width per (grammar, grouping). (ii) The share of exact runs in
  every cell, per init, and how many of them Adam itself brought below 1e-6 nats. (iii) The floor of each cell below
  its bound, as a share of the uniform predictor's excess loss. (iv) Training curves on `mod` at r = 2, 3, 4.

## Method

- **Main grid** (`run.py`): grammar ∈ {mod, sym} × grouping ∈ {aligned, crossed} × r ∈ {1, …, 6} × init ∈ {variance,
  identity (ε = 0.1)} × seeds 2000–2009 = 480 trees, 20 per cell, at each init's default rate. The two groupings of a
  cell share seeds, inits and learning rates, and differ only in `perm`. Pilots and the smoke test used seeds below
  1100 only. Order: widths 3, 4, 2, 5, 1, 6, so the cells the criteria rest on finish first. H1–H4 use this grid only.
- **Rate arm** (`run.py`, after the main grid): the four cells of the trainability comparison (`sym` r = 2 in both
  groupings, `mod` aligned r = 3, `mod` crossed r = 4) at the three other rates of {1e-3, 3e-3, 1e-2, 3e-2}, same
  inits and seeds = 240 more trees, 720 in all. The trainability reading compares the groupings with each at its own
  best rate per init (most exact runs of 10); picking the best of four rates favors both groupings alike. The
  comparison at the default rates is given beside it, and carries the reading alone if the rate arm is incomplete.
- **Learning-rate pilot per grouping** (`lr_pilot.py`, run before launch on seeds 0–3, in `results/lr_pilot.txt`):
  exact runs of 4 at the default rate against the best rate, per cell and init. `sym` aligned 4 and 4 against 4 and 4;
  `sym` crossed 3 against 3 (variance) and 3 against 4 (identity); `mod` aligned r = 3: 4 against 4 for both inits;
  `mod` crossed r = 4: 4 against 4 (variance) and 0 against 1 (identity: 0, 0, 0 and 1 of 4 at the four rates, so that
  cell's failures come from the init, not the rate). The default rates cost a crossed cell at most one run of four,
  which is why the rate arm exists and why the main grid's rates were left alone.
- **Training** (`toy.train`, `toy.polish`): float64; full-batch Adam on the exact population loss on the 24 prefixes,
  20,000 updates, `train.lr_at`'s schedule (100 warm-up updates, cosine to 0.1 × peak), default peak 1e-2 for the
  variance init and 3e-3 for the identity init (chosen on pilot seeds 0–3 on `sym` aligned at r = 2, and not retuned
  per cell: both groupings get the same rates). Then L-BFGS (strong Wolfe, at most
  1,000 iterations) on the same loss.
- **Analysis** (`analyze.py`): per cell the number of exact runs (per init, and how many Adam found), the runs between
  the thresholds, the runs whose polish stopped on its iteration or evaluation limit (1,000 and 1,250; such a run may
  not have converged and is counted as censored), the smallest, median and largest final excess loss, and for the
  cell's best run its largest weight and largest centered-logit error; the bounds recomputed by `toy.tree_bound`; the
  verdicts and the Fisher tests.
- **Thresholds** were set after a CPU pilot (seeds 100–102, both inits, 20,000 updates): in cells at or above their
  bound the exact runs ended at 1e-11 to 1e-16 nats; in cells below it every run ended at 1e-3 or more
  (`mod` aligned r = 2: 2.0e-3; `mod` crossed r = 3: 1.1e-3 to 4.2e-3; `sym` r = 1: 9.7e-3). In that pilot `mod`
  crossed was exact at r = 4 with the variance init (3 of 3) and not with the identity init (0 of 3).
- **Figures** (`plot.py`, PDFs from saved results): `capacity.pdf` (final excess loss of every run against width,
  per grammar and grouping, with the bounds), `curves.pdf` (training curves on `mod` at r = 2, 3, 4).

## Inputs

None. Both grammars are written out in `studies/rhm/toy.py` and the 24 prefixes are enumerated.

## Commands

From the project root on the coordinator, as one script in tmux. `EXP_DIR=studies/rhm/rhm-crossed-rank`. The first
line waits, for at most an hour, until `rhm-forced-synonyms` has written its summary, so that the two trainings do not
share the 2 vCPUs; after the hour it starts regardless.

```bash
set -euo pipefail
export EXP_DIR=studies/rhm/rhm-crossed-rank PYTHONPATH=. OMP_NUM_THREADS=1
timeout 3600 sh -c 'until [ -f studies/rhm/rhm-forced-synonyms/results/summary.txt ]; do sleep 30; done' || true
mkdir -p "$EXP_DIR/results" runs/rhm-crossed-rank
uv run --frozen python -m studies.rhm.toy_checks "$EXP_DIR/results/toy_checks.json" | tee "$EXP_DIR/results/toy_checks.txt"
uv run --frozen python studies/rhm/rhm-crossed-rank/run.py --out runs/rhm-crossed-rank --procs 2 | tee runs/rhm-crossed-rank/train.log
uv run --frozen python studies/rhm/rhm-crossed-rank/analyze.py runs/rhm-crossed-rank "$EXP_DIR/results" | tee "$EXP_DIR/results/summary.txt"
uv run --frozen python studies/rhm/rhm-crossed-rank/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

## Expected outputs

- `results/toy_checks.json`, `results/toy_checks.txt`.
- `runs/rhm-crossed-rank/<run>.pt`, 720 files of under 200 kB, and `train.log`.
- `results/runs.json` (final excess loss of every run), `results/curves.json` (loss logs), `results/summary.json` and
  `results/summary.txt` (per-cell table and verdicts), `results/provenance.json` (commit and library versions),
  `results/best_trees.pt` (the best tree of every cell, the witnesses of "exact", under 1 MB; `runs/` is git-ignored
  and its files can be regenerated from the seeds), and `results/lr_pilot.json`, `results/lr_pilot.txt` (made before
  launch).
- `figures/capacity.pdf`, `figures/curves.pdf`.
- `README.md` (the write-up, per the project's rules) and `NOTE.md` (a research note with the key figures as PNGs
  under `figures/png/`, as in the last two batches).

## Controls

- **`sym` is the control grammar** for `mod`: same symbols, same target function of A, same trees, seeds and
  optimizer, and a bound that does not depend on the grouping (H3).
- **The hand-built exact trees** (`toy_checks`): they show the aligned capacity exists at the bound independently of
  training, so a failure to train to it is an optimization result.
- **Widths above the bound** (up to 6) show whether a crossed tree that has the capacity is found by training.
- **Two inits, same seeds in both groupings.**
- **Width 1 and the uniform predictor** are the floors.

## Success and failure criteria

- H1–H4 and the trainability reading as stated under Hypothesis, computed by `analyze.py`.
- **Overall:** supported if H1, H2 and H3 are all supported; H4 is reported beside it as "attained" or "open", and
  the trainability reading as stated. Not supported if H2 fails with H1 holding. If H1 fails the run is treated as
  failed pending a look at the bound and the code. If H3 fails and H2 holds, inconclusive: the crossed tree may be
  limited by training as well as capacity.
- **The run failed** if a check fails, or a width-3 `mod` cell has fewer than 20 runs (`analyze.py` prints
  `RUN FAILED`).
- Runs between 1e-10 and 1e-5 nats are counted per cell and reported; they make a cell neither exact nor floor.

## Time and cost

13 s per run (20,000 updates and the polish, one thread, measured on the coordinator at r = 6; the review measured 11
to 17 s), so 720 runs take about 80 minutes with 2 processes, after a wait of about 25 minutes for the other
experiment; checks, analysis and figures add under 5 minutes. No rented hardware: $0. Each process uses about 400 MB
of the coordinator's 8 GB. If the timing comes out 1.5 times over, nothing is dropped: the cap of 3 hours counts from
the start of training and leaves room. The rate arm runs last, so if time did run out it is what would be missing,
and the trainability reading then falls back to the default rates.

## If something fails

- **A check fails:** stop; it is not a result. Report which.
- **The session is interrupted:** rerun the same script. `run.py` skips finished runs, so at most the two runs in
  progress are lost. Widths 3, 4 and 2 run first, so an early stop still leaves the cells of H1–H3 on `mod`.
- **A run ends with a non-finite loss:** it is saved, counted as not exact (its loss is taken as infinite) and
  counted per cell.
- **A run raises:** its traceback goes to `runs/rhm-crossed-rank/<run>.err`, the grid continues, and the count is in
  the summary. Rerunning the command retries only the missing runs.
- **Out of memory, or a hang:** `multiprocessing.Pool` does not notice a killed worker. If `train.log` gains no line
  for 5 minutes, stop the command and rerun it with `--procs 1`.
- Nothing here needs the user. A change to a threshold or to the grid would, and is not to be made.

## Fixed rules

- The thresholds (1e-10 exact, 1e-5 floor, 1e-4 logit error for H4, p < 0.05 for the trainability reading), the grid,
  the seeds and the rule "a cell is exact if any run is" are fixed.
- The main grid's learning rates are not retuned per cell or per grouping; the rate arm is the four fixed rates.
- No run is dropped.

## Questions asked before launch

None asked. The user supplied the two grammars and the predictions (their message of 2026-11-17) and said to develop
the experiments, run them, and batch them in tmux, which covers approval. Choices made without asking: the L-BFGS
polish; two inits; widths up to 6; `sym` crossed as the control; H4, the trainability reading and the rate arm, which
came out of the review; running on the coordinator because workers cannot be
launched today.

## Review

Reviewed once by the `experiment-reviewer` agent before launch (2026-11-17). No P0. It ran the checks, the smoke
pipeline and 13 full runs on pilot seeds, and computed the bounds of the pairing the grid does not train. Findings and
what changed:

- **P1, the criteria could only confirm what the theorems and the pilot settle, and the open question had none.**
  Changed: H4 (the crossed bound of 4 is attained, "open" and not "refuted" if no run is exact); a trainability
  reading with its rule fixed in advance, each grouping at its own bound; H3 reworded as an existence control.
- **P1, the learning rates were tuned on the aligned grouping only.** Changed: `lr_pilot.py` repeated the four-rate
  pilot per grouping at its own bound (result under Method), and because the best rate differed in two cells by one
  run of four, a rate arm now runs the four trainability cells at all four rates, and the reading compares the
  groupings each at its own best rate.
- **P2, "attained" rested on the 1e-10 threshold alone.** Changed: each cell's best run is reported with its largest
  weight and largest centered-logit error, H4 requires the latter below 1e-4, and the best tree of every cell is saved.
- **P2, the polish's stopping reason was not saved.** Changed: iterations, function evaluations and a `capped` flag
  are saved (the evaluation limit of 1,250 is now explicit), and capped runs are counted per cell.
- **P2, a non-finite run broke the verdicts.** Changed: its loss is taken as infinite and it is counted per cell.
- **P2, no source snapshot.** Changed: the code is committed with this definition before the run, and
  `provenance.json` records the commit. The entry in `EXPERIMENTS.md` is added by the planning session.
- **P2, an out-of-memory kill would hang.** Changed: the 5-minute rule under "If something fails".

The revised definition was not sent back for a second pass: every change follows the reviewer's stated smallest fix,
and the revised code was rerun on the checks and the smoke grid.

## Amendments

None.

<!-- Each change made during the run, with the time and the reason, recorded and committed before acting on it. -->

## Outcome

Not run yet. Afterwards: one line per criterion with what was observed, the cost and runtime against the budget, and a link to the write-up. The write-up holds the detail; don't repeat it here.
