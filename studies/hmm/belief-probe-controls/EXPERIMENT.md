---
name: belief-probe-controls
title: Probe controls on the saved belief-geometry checkpoints: factor codes and low-dimensional probes
status: done
gpu: NVIDIA RTX PRO 4500 Blackwell,NVIDIA RTX A6000,NVIDIA A40
count: 1
hours: 7
budget_usd: 10
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-15T19:37:45Z
result: inconclusive
finished_utc: 2026-11-15T22:01:48Z
writeup: studies/hmm/belief-probe-controls/README.md
---

# Probe controls on the saved belief-geometry checkpoints: factor codes and low-dimensional probes

## Question

The belief-geometry batch left three readings open, each of which a reanalysis of the saved checkpoints can settle
without training anything:

1. **Factor codes.** On the product processes the tensor networks give the two factor marginals at R² ≈ 1 but the
   9-state joint belief only at 0.75–0.85. Is the joint simply carried as its two marginals, which an affine or
   projective probe cannot multiply?
2. **Is the belief the dominant geometry of the state?** The probes use every coordinate of the state against at most
   a few hundred distinct beliefs per position. A no-network check on the coordinator
   (`studies/rhm/rhm-ttn-belief-geometry/followups/scramble_oracle.py`) showed that the scrambled-target control could
   not have tested geometry: a state that is exactly the belief fits the scrambled target at R² 0.72–0.96 in the first
   half of every RHM cell, and in the second half the bilinear probe fits it far better than a belief code would
   (0.65–1.00 against 0.26–0.93). The Simplex-style claim needs a probe that can only use the state's directions of
   largest variance.
3. **Two RRXOR tree cells.** The r = 8 tree has one seed at R² −4.55, and the r = 32 tree had no eligible position.

## Hypothesis

- **H1, factor code (primary).** A measurement with its reading fixed, more than an open prediction: on `factored`
  the joint is exactly the product of the marginals, so the saved marginal scores (0.999 chain, 0.997–0.998 tree at
  r = 32, 0.985 tree at r = 8, three seeds each) already imply the product's score, as the reviewer pointed out. What
  the saved numbers do not contain is the floor: the direct joint probe on a state that is exactly [b^A, b^B], on the
  same split and positions (0.755 affine in the reviewer's computation on `factored`). Per arm, with affine probes,
  in all three seeds:
  - **marginals exact:** the product b̂^A ⊗ b̂^B built from the experiment's own marginal probes, with no further
    fitted parameter, reaches R² ≥ 0.99 and halves the unexplained variance left by the same product from
    token-readout marginals and from the initial network (1 − R² ≤ 0.5 × (1 − control)). Otherwise "marginals not
    exact".
  - Then, with gap = (the model's direct joint probe) − (the floor): **pure factor code** if gap ≤ 0.03, **marginals
    plus part of the joint** if gap > 0.03 and the direct probe is below 0.95, **joint readable** if it is at least
    0.95.
  - Expected from the saved numbers: the chain (direct 0.834–0.836) and the r = 32 tree (0.804–0.819) come out as
    "marginals plus part of the joint", 0.05–0.08 above the floor; the transformer (0.976–0.982) as "joint readable";
    the r = 8 tree sits at the floor (0.747–0.749) but its marginals are not exact. A CPU pilot of one run
    (`factored_chain_s0`, `results/factor_product_cpu_pilot.json`) gave 0.999 for the product against 0.845 for the
    direct projective probe, 0.967 for the token-marginal product and 0.784 for the initial network's.
  - What it shows: how the prediction is laid out. On these processes the belief is a function of the exact
    next-token distribution, so nothing here is information beyond the prediction.
  - `coupled` is descriptive. A smoke test found that its residual C = b^AB − b^A ⊗ b^B carries about 0.1% of the
    joint's variance, so it cannot test whether correlations beyond the marginals are represented. It is read only if
    that share is at least 1% on the full sample.
- **H2, is the belief the dominant geometry of the state?** Two labels per cell, each needing all three seeds:
  - **dominant** (primary): the reverse probe, from the belief to the state's top principal scores holding 90% of its
    variance, reaches R² ≥ 0.8 on held-out prefixes and halves the unexplained variance of the same on the initial
    network, and the experiment's forward probe is at least 0.8 on the same positions. So the belief is linearly
    readable and most of the state's variance is a function of it.
  - **compact:** the forward probe restricted to the state's top k90(t) principal directions, where k90(t) is the
    number of directions the belief itself needs to give the belief at R² ≥ 0.9, reaches a pooled R² ≥ 0.8 and halves
    the unexplained variance of the initial network and of the one-hot tokens restricted the same way.
  - Why two: the review showed that "compact" alone fails a state that is a stretched or duplicated image of the
    belief with nothing else in it (in a smoke test the Mess3 tree's second half scored 0.64 at k90, 0.98 at 2·k90
    and 0.99 in full), and it never measures how much of the state the belief accounts for.
  - Expected, from smoke tests on 3,000 sequences and one seed: Mess3 chain dominant and compact (reverse 0.92
    against 0.32 for the initial network; forward 0.99 from its top 2 directions); Mess3 tree, second half, neither
    (reverse 0.56); RRXOR chain dominant and compact (0.91, 0.95); the RHM transformer and tree not compact (0.43–0.49
    at k90 against 0.79–0.91 in full), with the reverse direction unmeasured.
  - Caveats that go with any verdict. On Mess3 the belief is a function of the exact next-token distribution, so a
    pass says the dominant directions are the model's own prediction; RRXOR is the process where the belief is more
    than that. The one-hot token control scored 0.00–0.02 at k90 on RRXOR and the RHM in the smoke tests, so there
    only the initial network is a real control. Where a position's distinct beliefs are close to affinely
    independent, the reverse probe measures whether the state's variance is a function of the belief state, not of
    its geometry. The tree's second half cannot be `dominant` by construction: its left top bond holds the left
    half's summary, which the belief at t does not determine, so a "no" there is architectural. The reverse score
    pools positions by the state's raw variance; the summary prints the equal-weight mean over positions next to it. For the tensor networks the principal directions depend on the bond's gauge, so the result
    describes the gauge that training left.
- **H3, RRXOR trees: a recorded reanalysis.** This part has already run on the coordinator's CPU
  (`results/rrxor_tree_cpu_pilot.json`), so its rules are no longer predictions. They are kept as written before that
  run, and the worker repeats it so every number in the write-up comes from one machine.
  - (a) Rule: the −4.55 is numerical if, without the test rows whose prediction leaves the range a belief can take,
    the seed scores within 0.05 of the other two seeds at the same position. CPU result at t = 13, seed 2: 146 of
    31,129 rows carry 99% of the squared error, and without them R² is 0.839, against 0.673 and 0.715 for the other
    seeds. So the blow-up is confined to 0.5% of rows, but the rule as written is not met: the trimmed score is
    higher than the other seeds', which dropping a probe's worst rows tends to produce. The affine probe on the same
    features, which drops no rows, gives 0.751 there and is the number to compare.
  - (b) Rule: with the r = 32 tree's product features reduced to the largest number the eligibility rule admits, the
    cell passes the experiment's H3b clauses in all three seeds (R² ≥ 0.8, halving the unexplained variance of the
    additive ceiling and of the next-token ceiling). CPU result: 0.884, 0.883 and 0.895 against an additive ceiling of
    0.772 (so 0.886 is needed) and a next-token ceiling of 0.466; the initial network scores 0.02–0.05. The cell fails
    by 0.002–0.003 in two seeds. No interval was computed, so this is a miss at the threshold, not a clear failure.
- **Overall result recorded:** H1's reading for the chain and the r = 32 tree: "supported" if both are "pure factor
  code", "partly supported" if both have exact marginals and at least one is "marginals plus part of the joint"
  (the expected outcome), "not supported" if either has marginals that are not exact or a readable joint. H2 and H3
  are reported with their own verdicts.

## Method

All three parts use the experiments' own sequences (60,000, same seeds), held-out prefix split, eligible position
groups and score (R² = 1 − SSE/SST about the fit mean, pooled over positions by summed SSE and SST). Each script
asserts that its split reproduces the saved one, and the summary reports how closely the full probes reproduce the
saved numbers.

- **Part 1, `factor_product.py`:** `factored` and `coupled` × transformer, chain, tree (r = 32), tree8 × seeds 0–2 =
  24 runs, final checkpoint and checkpoint 0. Affine and projective probes on the primary set (L4, chain state, tree
  `top`), on the experiment's group (`e`; `h1` for the tree); the tree's second half with `top_bil` is computed too.
  Per run: the direct joint probe, the marginal probes, their product, the product from token-readout marginals, the
  product of the exact marginals, the floor (the direct probe on the exact marginals), and for C: its variance share, affine probes from the state and the tokens, and an
  affine probe on the state's coordinate products where eligible.
- **Part 2, `lowdim.py`:** per run, final checkpoint and checkpoint 0, at the experiment's eligible positions: the
  affine probe from the top-k principal scores of the activation set (k on a grid from 1 to the number of
  directions), to the belief and to the experiment's scrambled target. The same for the one-hot tokens and for the
  exact belief itself (`oracle`), whose curve is the rank-k ceiling. For the model and the initial network also the
  reverse probe, from the belief to the state's top principal scores that hold 90% of its variance.
  - `hmm`: `mess3` and `rrxor` × four arms × three seeds, 24 runs (L4; chain state; tree `top`, groups h1 and h2).
  - `rhm-tree`: the 30 runs at r = 32 (`top`, groups h1 and h2).
  - `rhm-transformer`: all 51 runs (L4, group eL4).
  - `lowdim_summary.py` applies H2's two rules (`compact` on the judged positions, where k90 is at most half of the
    state's varying directions; `dominant` on all eligible positions), and `plot.py` draws
    `figures/lowdim_<family>.pdf` from the same saved files.
- **Part 3, `rrxor_tree.py`:** as its docstring: (a) the three r = 8 seeds at t = 13–15, (b) the three r = 32 seeds
  with standardized product features reduced to k = ⌊n_fit_prefixes/10⌋ − 1 principal scores.

## Inputs

The saved checkpoints of three finished experiments, read from their published archives on the global volume and
unpacked to the worker's local disk outside the project (`/root/ckpt`), so they are not published again:
`results/20261114-234443-hmm-belief-geometry-e1dlkqg8emkuoj`, `…-194742-rhm-ttn-belief-geometry-npv0zc4c9pk5pz` and
`…-183036-rhm-belief-geometry-dl1dad9eotnowo` (`pub-001/runs.tar` of each; about 2 GB without the projection files).
The saved analyses under `studies/*/results/` are in git. No model is trained.

## Commands

From the project root on the worker. Setup and checks:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
mkdir -p /root/ckpt "$EXP_DIR/results"
for j in 20261114-234443-hmm-belief-geometry-e1dlkqg8emkuoj 20261114-194742-rhm-ttn-belief-geometry-npv0zc4c9pk5pz 20261114-183036-rhm-belief-geometry-dl1dad9eotnowo; do
  tar -xf /workspace-global/results/$j/pub-001/runs.tar -C /root/ckpt --exclude='*.npz' --exclude='*.log'
done
du -sh /root/ckpt/runs/*
OMP_NUM_THREADS=8 uv run --frozen python -m studies.hmm.checks "$EXP_DIR/results/checks_hmm.json"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks_rhm.json"
```

Parts 3 and 1, then the HMM runs of part 2, as one step. Every part-2 run is its own process, so one failure does not
end a queue:

```bash
set -euo pipefail
export RUNS_ROOT=/root/ckpt/runs PYTHONPATH=.
queue() { fam=$1; shift; for r in "$@"; do OMP_NUM_THREADS=8 uv run --frozen python studies/hmm/belief-probe-controls/lowdim.py $fam $r || echo "FAILED $fam $r"; done; }
uv run --frozen python studies/hmm/belief-probe-controls/rrxor_tree.py | tee "$EXP_DIR/results/rrxor_tree.txt"
uv run --frozen python studies/hmm/belief-probe-controls/factor_product.py | tee "$EXP_DIR/results/factor_product.txt"
queue hmm $(for p in mess3 rrxor; do for a in transformer chain tree tree8; do for s in 0 1 2; do echo ${p}_${a}_s$s; done; done; done) 2>&1 | tee "$EXP_DIR/results/lowdim_hmm.log"
uv run --frozen python studies/hmm/belief-probe-controls/lowdim_summary.py | tee "$EXP_DIR/results/lowdim_summary.txt"
ls "$EXP_DIR"/results/lowdim/hmm/*.json | wc -l    # 24
```

The RHM runs of part 2 as a second step, the tree first. Three queues per family: every run whose beliefs are 4,096
wide (the v = 16 grammars and `v8m2s2L5`) in one queue, since two of those at once ran out of memory on a 32 GB card
in `rhm-belief-geometry`, and the rest split in two:

```bash
set -euo pipefail
export RUNS_ROOT=/root/ckpt/runs PYTHONPATH=.
queue() { fam=$1; shift; for r in "$@"; do OMP_NUM_THREADS=8 uv run --frozen python studies/hmm/belief-probe-controls/lowdim.py $fam $r || echo "FAILED $fam $r"; done; }
names() { (cd /root/ckpt/runs/$1 && ls -d $2); }
run_family() {  # family, run folder, glob
  queue $1 $(names $2 "$3" | grep -E '^(v16|v8m2s2L5)') > "$EXP_DIR/results/lowdim_$1_wide.log" 2>&1 & p1=$!
  queue $1 $(names $2 "$3" | grep -vE '^(v16|v8m2s2L5)' | awk 'NR % 2 == 0') > "$EXP_DIR/results/lowdim_$1_a.log" 2>&1 & p2=$!
  queue $1 $(names $2 "$3" | grep -vE '^(v16|v8m2s2L5)' | awk 'NR % 2 == 1') > "$EXP_DIR/results/lowdim_$1_b.log" 2>&1 & p3=$!
  wait $p1; wait $p2; wait $p3
  grep -h FAILED "$EXP_DIR"/results/lowdim_$1_*.log || true
  uv run --frozen python studies/hmm/belief-probe-controls/lowdim_summary.py | tee "$EXP_DIR/results/lowdim_summary.txt"
}
run_family rhm-tree rhm-ttn-belief-geometry 'v*_r32_s[012]'
ls "$EXP_DIR"/results/lowdim/rhm-tree/*.json | wc -l           # 30
run_family rhm-transformer rhm-belief-geometry 'v*_s[012]'
ls "$EXP_DIR"/results/lowdim/rhm-transformer/*.json | wc -l    # 51
uv run --frozen python studies/hmm/belief-probe-controls/plot.py
```

The run globs start with `v` on purpose: the tree archive also holds a folder `cpucheck_v8m4s2L4_r32_s0` with no
checkpoint in it.

## Expected outputs

- `results/checks_hmm.json`, `results/checks_rhm.json`.
- `results/rrxor_tree.json` and `.txt`; `results/factor_product.json` and `.txt`.
- `results/lowdim/<family>/<run>.json` (24 + 30 + 51 = 105 files, under 1 MB each), `results/lowdim_summary.json`
  and `.txt`, and the queue logs `results/lowdim_*.log`.
- `figures/lowdim_hmm.pdf`, `figures/lowdim_rhm-tree.pdf`, `figures/lowdim_rhm-transformer.pdf`.
- `README.md`, the write-up, and `NOTE.md`, a research note with the key figures embedded as PNGs under
  `figures/png/`.

## Controls

- **Reproduction of the saved numbers:** part 1's direct joint probe and part 2's full probe are the experiments'
  own probes; the summaries print the largest difference from the saved values (expected below 0.01; the CPU pilot
  reproduced 0.8358 and 0.8448 to four decimals). Each script asserts that its prefix split has the saved row counts.
- **Initial network and one-hot tokens,** probed identically, in parts 1 and 2. In part 1 the token-marginal product
  is the control that matters, since the tokens alone give the marginals at 0.96–0.97.
- **The exact belief as the activations** (`oracle`, part 2): what a state whose principal directions are the
  belief's own would score, for the belief and for the scrambled target, at every k.
- **Checks inside the scripts:** a state that holds the exact marginals gives product R² 1 and a direct affine joint
  below 0.99 (index order of the product); the principal-score probe on the target itself equals the Eckart–Young
  value, and is unchanged by a rotation that adds weaker directions.
- **Project checks** `studies.hmm.checks` and `studies.rhm.checks` first.

## Success and failure criteria

- **H1:** per arm, the class from the rule above; the overall result as defined under "Overall result recorded".
- **H2:** per cell and position group, "dominant" yes or no and "compact" yes or no by the rules above; a cell with
  fewer than three seeds that have an eligible (for compact: a judged) position is not judged. Reported per family as
  counts among the judged, learned cells; no single verdict is forced across families.
- **H3:** the two rules as written; the CPU results above are expected to repeat.
- **The run failed** if a project check fails, if a split assertion fails, if a reproduction difference exceeds
  0.01 (the summaries then exit with an error), or if fewer runs are produced than 22 of the 24 in part 1, or in
  part 2 than 22 of 24 (`hmm`), 27 of 30 (`rhm-tree`) or 46 of 51 (`rhm-transformer`).

## Time and cost

- Setup, unpacking about 7 GB of archives from the global volume and the checks: about 25 minutes.
- Part 3: minutes. Part 1: one chain run took 20 minutes on one CPU core, dominated by L-BFGS; a few minutes per run
  on a GPU, about 1 hour for 24 runs (the tree's second half has 1,088 features).
- Part 2: no projective fit. The cost per RHM run is the exact beliefs on the CPU, as in the original analyses, which
  took 2 h 44 m for the transformer's 51 runs with six checkpoints and seven activation sets each. Here two
  checkpoints and one set, three processes: about 2 to 3 hours for 81 runs, 20 minutes for the HMM runs.
- Total about 4 to 5 hours. At $0.72/hr (RTX PRO 4500 Blackwell) that is about $3.50; the 7-hour cap costs $5.04.
  The budget is $10.

## If something fails

- **A project check or a split assertion fails:** stop; that is a failed run.
- **A reproduction difference above 0.01:** stop that part and report it; do not interpret its numbers.
- **A part-2 run fails** (the queue prints `FAILED <family> <run>` and goes on): rerun the failed runs once, one
  process at a time, after the family's queues end; `lowdim.py` skips runs whose output exists. For out of memory on
  a v = 16 grammar (belief matrices of 60,000 × 4,096), that single-process rerun is the fix. Runs that still fail
  are reported as missing.
- **Time running out:** the order is H3, H1, then H2 for the HMM, the RHM tree and the RHM transformer, with a
  summary after each family, so a cut loses the transformer runs first. Runs not produced are reported as missing.
- Any change to a rule waits for the user.

## Questions asked before launch

None needed. The user approved the plan in conversation on 2026-11-15 ("go ahead with these"), including that these
reanalyses run on one small worker with their criteria written down first. Two things differ from the plan as it was
presented, both decided from results obtained since and both reported to the user in the same conversation before
launch: the matched permutation control was dropped, because the
no-network oracle showed that it behaves like the existing scrambled control (pooled 0.22–0.96 against 0.26–0.96),
and the low-dimensional probe became the geometry test; and the capacity ceiling and the parity measurement were done
on the coordinator's CPU and are not part of this experiment.

## Review

Reviewed on 2026-11-15 by the `experiment-reviewer` agent, which ran smoke tests of `lowdim.py` and
`lowdim_summary.py`, a dry run of the run globs against the archives, and a no-network computation of the
factor-code floor. Verdict: needs revision. Findings and what changed:

- **P0, the tree glob matched a folder that is not a run** (`cpucheck_v8m4s2L4_r32_s0`), and the process that got it
  died on its first name, which would have silently dropped 15 of the 30 tree runs and left every RHM tree cell
  unjudged. Changed: globs start with `v`, every run is its own process with a `FAILED` line, the queues are waited on
  by PID, files are counted per family, and the failure floor is per family.
- **P1, H1's rule was already decided by saved numbers and the informative comparison was missing.** Changed: the
  floor (direct joint probe on the exact marginals) is computed, the rule is on the gap to it with a stated
  tolerance, the expected outcome is stated from the saved numbers, and H1 is described as a measurement with its
  reading fixed.
- **P1, H2's "not dominant" could not tell other content from a belief laid out in more than k90 directions.**
  Changed: the reverse probe (belief to the state's top-variance scores) is added and carries the label "dominant";
  the top-k90 rule is renamed "compact"; the Mess3 next-token caveat and the vacuous token control on RRXOR and the
  RHM are stated.
- **P1, H3's outcome was already in the folder.** Changed: H3 is labelled a recorded reanalysis, the CPU numbers are
  in the definition, the rules are kept as written, the affine probe is named as the statistic that drops no rows,
  and (b) is described as a miss at the threshold with no interval. The CPU file is renamed
  `rrxor_tree_cpu_pilot.json`. No bootstrap was added.
- **P2, both RHM queues started with v = 16 runs.** Changed: all v = 16 runs go in one queue.

Re-check of the revision by the same reviewer: ready to run; the P0 and all three P1s are resolved and nothing new at
P0 or P1. It dry-ran the queue functions against `runs/` (30 tree and 51 transformer names, no `cpucheck` folder) and
smoke-ran every script. Its four optional notes were applied: `v8m2s2L5`, which also has 4,096-wide beliefs, joins
the serial queue; the definition says the tree's second half cannot be `dominant` by construction; the summary
prints the equal-weight mean of the reverse score; and the family counts cover learned cells only.
- **P2, smaller items.** The summaries now exit with an error on a reproduction difference above 0.01; the title no
  longer says "matched scrambles"; figures come from `plot.py`.

## Amendments

- **2026-11-15 21:36 UTC, supervisor, no rule changed.** Three RHM transformer runs (`v3m2s2L6_s1`, `v4m2s2L5_s0`,
  `v4m2s2L5_s2`) failed with CUDA out of memory while the three queues shared the 32 GB card; 48 of 51 were produced.
  As "If something fails" prescribes, they are rerun once, one process at a time, as an extra step `rerun`. The one
  addition to the written commands: `lowdim_summary.py` and `plot.py` are run again after the rerun, so the summary
  and the figures include the rerun runs.

## Outcome

Recorded as `inconclusive`, which stands here for the definition's "partly supported". Ran 2026-11-15, 19:38 to 21:40 UTC. The detail is in [`README.md`](README.md) and the research note in [`NOTE.md`](NOTE.md).

- **Run not failed:** met. Project checks 29 of 29 and 28 of 28; no split assertion failed; reproduction differences at most 8.2e-08 (part 1) and 2.3e-05 (part 2) against a limit of 0.01; 24 of 24 runs in part 1, and in part 2 24 of 24 (`hmm`), 30 of 30 (`rhm-tree`) and 51 of 51 (`rhm-transformer`), three of the last after the prescribed single-process rerun.
- **H1 (primary): partly supported, the expected outcome.** On `factored` the chain and the r = 32 tree are both "marginals plus part of the joint" in all three seeds (product 0.999 and 0.996 against 0.967 from token marginals; direct probe 0.078–0.080 and 0.049–0.064 above the floor of 0.755–0.756). Transformer "joint readable", r = 8 tree "marginals not exact". `coupled` not read: its residual carries 0.10–0.11% of the joint's variance.
- **H2:** among learned cells with three judged seeds, dominant in 5 of 10 and compact in the same 5 (`hmm`), dominant in 7 of 12 and compact in 0 of 12 (`rhm-transformer`), dominant in 2 of 15 and compact in 4 of 15 with none both (`rhm-tree`).
- **H3:** the CPU results repeat. (a) Rule not met: without its 146 out-of-range rows the seed scores 0.839 against 0.673 and 0.715; the affine probe gives 0.751. (b) Fail in two of three seeds: 0.884, 0.883 and 0.895 against a required 0.886, no interval.
- **Cost and time:** 2 h 02 min on one RTX PRO 4500 Blackwell, about $1.46, against a 7-hour cap and a $10 budget.
