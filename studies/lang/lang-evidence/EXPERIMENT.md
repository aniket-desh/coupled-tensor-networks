---
name: lang-evidence
title: "Evidence aggregation and event detection in a TreeTN: masked bigram at three meeting levels and a reserved-trigger OR"
status: approved
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 10
budget_usd: 12
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-13T23:50:56Z
---

# Evidence aggregation and event detection in a TreeTN: masked bigram at three meeting levels and a reserved-trigger OR

## Question

Two computations that real language processes need, each with a small predicted bond summary:

1. **Combining evidence from two directions (masked bigram).** For a Markov chain with leaf p masked, the Bayes logits are the sum of forward evidence log P(y | x_{p−1}) and backward evidence log P(x_{p+1} | y), so the task needs no conjunction code of the neighbour pair.
   - Does the TreeTN carry each evidence vector up from its neighbour and add them where the two meet?
   - Does the level of that meeting point matter for learning?
   - Mask positions:
     - p = 5: neighbours 4 and 6, meeting at a level-2 node.
     - p = 4: neighbours 3 and 5, meeting at a level-3 node.
     - p = 8: neighbours 7 and 9, which meet only in the root core.
2. **Detecting an event (reserved trigger).** The target depends on whether a reserved trigger symbol occurs anywhere in leaves 0–14, and on the current token in leaf 15.
   - The predicted summary is one flag per subtree, merged by OR.
   - Does each bond carry that flag, discard where the trigger sat, and does the output use it?

## Hypothesis

- **H1 (masked bigram is learned; capacity).** The untied r = 16 tree reaches final test KL ≤ 0.02 nats at every mask position, in every seed. r = 32 is not better than r = 16 by more than the tie tolerance (9 pairs). For context only: the sampled centered-logit rank for the instance with seed 0 is ≤ 16 at every cut for p = 5 and 8 (`studies/lang/results/rank.json`). That is a lower bound on sampled contexts, not part of the criterion.
- **H2 (meeting level; two-sided, no predicted direction).** Steps to threshold and final KL differ between p = 5 (level-2 meeting) and p = 8 (root meeting). Either consistent direction is a finding. With n = 3 and 3 of 3 required, an outcome without a consistent direction is reported as "unresolved at n = 3", with medians and ranges, and not as evidence that meeting level doesn't matter.
- **H3 (tree vs leaf permutation; descriptive).** The permutation moves the neighbours' meeting level in both directions. The perm arm's meeting levels, computed from the seeded permutations:

  | Seed | p = 5 | p = 4 | p = 8 |
  |---|---|---|---|
  | 0 | level 3 | root | root |
  | 1 | root | root | level 1 |
  | 2 | level 3 | level 2 | root |

  So tree vs perm can't isolate alignment as such. It is reported descriptively (9 pairs per measure), and every tree and perm run is also pooled by its meeting level (each run's level is recorded in `analysis.json`).
- **H4 (evidence mechanism, decided on the r = 32 tree, final checkpoint).**
  - Why r = 32: at r = 16 the combined-evidence probe subspace is 15 of 16 dimensions, so a random 15-dimensional subspace carries almost the whole bond and isn't a control. At r = 32 it is about half. Two 15-dimensional evidence vectors can't sit side by side in 16 dimensions either, so only r = 32 can tell "added" from "carried separately".
  - Evidence targets are centered log-evidence clipped at log 10⁻³. The unclipped tail (down to about −13 nats) dominates the variance, and a near-Bayes model (KL ≈ 0.02) reaches only R² ≈ 0.85 on it (reviewer's check).
  - (a) One-sided evidence is decodable (ridge R² ≥ 0.9) on every bond whose span holds exactly one neighbour and at least one other non-mask leaf. Leaves, and the level-1 node whose other leaf is the mask, pass at initialization and are excluded. The step-0 median is reported alongside.
  - (b) The combined evidence is decodable on every bond holding both neighbours, up to what the model's own output expresses: bond R² ≥ (R² of the same ridge fit from the model's centered output logits) − 0.05. With the clip, the attainable R² depends on how sharp the model's tail is (exact Bayes reaches only 0.81–0.83 on the clipped target, by the reviewer's check). Because the tree's logits are linear in each bond, the output's own fit is the right per-run ceiling.
  - (c) At the meeting node (p = 5: level 2; p = 4: level 3; none for p = 8), patching only the comb subspace from a donor brings the output close to the counterfactual Bayes posterior: median KL < half the unpatched KL and below the random-subspace KL.
  - Descriptive: fwd, bwd and comb R² at the meeting node (added vs side by side). The r = 16 tree is reported the same way.
- **H5 (event flag, trigger, main arm r = 32).**
  - (c), deciding: patching the flag subspace at internal nodes flips the prediction to the counterfactual map in ≥ 80% of affected pairs (median), while random subspaces do so in ≤ 40%.
  - (b): on examples with a trigger in the span, moving it within the span leaves the flag subspace nearly unchanged. The criterion is median relative change ≤ 0.1 over level ≥ 2 nodes, and below its step-0 value; per-level values are reported. Level-1 spans hold at most two background leaves, so moving the trigger there barely changes anything, even at initialization.
  - (a), descriptive: flag probe accuracy, with its step-0 value and the majority-class chance. The flag is linear in the one-hot tokens (at most one trigger occurs), so it is often decodable at initialization.
  - Descriptive: decodability of the trigger's location within level ≥ 2 spans, from step 0 to the end.
- **H6 (descriptive).** Trigger at r = 16 vs r = 32, the permuted tree and the chain; final KL by trigger stratum (no trigger, or the level of the trigger's common ancestor with leaf 15).

## Method

- **Processes** (`studies/lang/processes.py`), V = 16:
  - **`mb_p{p}`** (`MaskedBigram`), p ∈ {5, 4, 8}:
    - x_0 ~ π (stationary), then x_{t+1} ~ P(· | x_t). Each row of P puts Dirichlet(1) weight on 3 random successors, mixed with floor 0.02/V.
    - Leaf p receives a mask token (input vocabulary 17), and y = x_p.
    - The Bayes posterior is ∝ P(y | x_{p−1}) P(x_{p+1} | y). `checks.py` verifies it by brute force.
    - No extra target noise is needed, since every transition is positive.
  - **`trigger`**:
    - x_0..x_14 are iid uniform over 16 symbols. With probability 1/2, one uniform position in 0..14 holds the reserved trigger token 16.
    - x_15 is the current token b.
    - y = c(b) if a trigger is present, else d(b), where c and d are seeded random maps with c(b) ≠ d(b) for every b. Target noise is ε = 0.05.
  - The process instance is seeded by the run's seed.
- **Arms**:
  - **Masked bigram:**
    - `tree`: untied, r = 16; 65,792 parameters.
    - `perm`: the same tree on a fixed leaf permutation, seeded per seed (500 + seed) and redrawn until no adjacent pair (2k, 2k+1) are model siblings, as in rhm-controls.
    - `chain`: width 40, the parameter match to the r = 16 tree.
    - `tree32`: r = 32, at every p (H4 is decided on it).
  - **Trigger:**
    - `tree16`.
    - `tree32`, the main arm.
    - `perm32`.
    - `chain`: width 78, matched to the r = 32 tree.
  - The trigger's main width is 32 because some subtrees must carry the flag together with the current token's information.
- **Init**: identity init, ε_init = 0.1, for every arm.
- **Training**:
  - Online batches of 256 from a generator seeded per (condition, seed), the same for every arm.
  - Adam, 30,000 updates, warmup 100, then a cosine schedule to 0.1 × peak.
- **LR**: piloted per (process, arm) over {1e-4, 3e-4, 1e-3, 3e-3} on seed 9 (masked bigram on p = 4). The pilot runs 4,000 of the 30,000 updates and picks the lowest diagnostic loss, excluding LRs whose pilot learned and then collapsed.
- **Grid**:
  - masked bigram: 3 positions × 4 arms × 3 seeds = 36 runs;
  - trigger: 4 arms × 3 seeds = 12 runs;
  - 48 runs in all, seed 0 first.
- **Measures**:
  - The same as lang-source: a 20,000-example diagnostic set logged through training, and 50,000 test examples.
  - t = the first logged update with diagnostic accuracy ≥ 0.9 × Bayes accuracy.
  - The trigger's test KL is also broken down by stratum.
- **Interpretation** (`studies/lang/interpret.py`, at every checkpoint, for every tree-type run: tree, perm, tree32 and the trigger trees):
  - Summaries per span:
    - `fwd` and `bwd`: centered log-evidence vectors, ridge R²;
    - `comb`: the centered Bayes logits, where both neighbours are in the span;
    - `flag`: trigger present in the span;
    - `where`: the trigger's position in the span, or none;
    - `cur`: x_15.
  - Baselines, invariance and patching are as in lang-source. Invariance moves:
    - masked bigram: resample the span's non-neighbour tokens;
    - trigger: move the trigger within the span and redraw the background.

### Comparison rules

These are the same as lang-source (`studies/lang/analysis.py`): faster(·) with the logging-interval margin and the factor 1.25, and better(·) on test KL with a tie within max(0.002, 10%). A direction over 9 pairs needs ≥ 8 wins and no losses. Over 3 pairs it needs 3 of 3.

## Inputs

None. All data is generated from seeded processes.

## Commands

```bash
set -euo pipefail
mkdir -p "$EXP_DIR/results" "$EXP_DIR/figures"
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python -m studies.lang.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python -m studies.lang.rank "$EXP_DIR/results/rank.json" --configs lang-evidence
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-evidence/run.py --out runs/lang-evidence --hours 5
PYTHONPATH=. uv run --frozen python -m studies.lang.interpret_all runs/lang-evidence --procs 4 --hours 2
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-evidence/analyze.py runs/lang-evidence "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/lang/lang-evidence/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

Run the training and interpretation steps from a background command. The checks exit non-zero on failure. The training deadline (5 h) counts from the end of setup and the pilots.

## Expected outputs

- `results/checks.json` and `results/rank.json`.
- `runs/lang-evidence/<cond>_<arm>_s<seed>/`: `result.json`, checkpoints and `interpret.json`, plus `lr.json`, `meta.json` (parameters, permutations), `planned.json`, `status.json`, `pilot/` and `source/`. About 1–2 GB.
- `results/analysis.json`, `results/curves.json` and `results/interp.json`.
- `figures/curves.pdf`, `figures/probes.pdf` and `figures/trigger_strata.pdf`.
- `README.md`.

## Controls

- **Pairing.** Arms within a process share data streams and process instances, so they are paired.
- **Alignment.** The perm arm has the same width as the tree, so it isolates alignment.
- **Chain matching.** The chain is parameter-matched to the tree width named in each arm.
- **Capacity.** tree32 checks capacity for the masked bigram, and tree16 vs tree32 checks it for the trigger.
- **Probe baselines:** step 0, shuffled labels, an additive one-hot probe of the span (for the flag, the additive probe can also read the OR, because at most one trigger occurs, so it is reported but isn't a criterion), random-subspace patches, and whole-bond patching (equal to splicing).
- **Correctness checks** (`studies/lang/checks.py`), including the brute-force masked-bigram posterior, run first.

## Success and failure criteria

- **H1:** supported if all 9 tree runs have final KL ≤ 0.02 and tree32 beats tree in at most 1 of 9 pairs; inconclusive if fewer than 6 capacity pairs are usable; not supported otherwise.
- **H2:** "level-2 meeting faster/better" or "root meeting faster/better" if 3 of 3 seeds agree on a measure; "unresolved at n = 3" otherwise, with medians and ranges. Speed and quality are reported separately.
- **H3:** descriptive. Tree vs perm pair counts on both measures, and t and KL medians for the tree and perm runs pooled by meeting level. Tree vs chain is reported the same way.
- **H4:**
  - supported if (a), (b) and (c) hold over the tree32 arm's interpreted runs at the final checkpoint:
    - (a) ≥ 90% of the eligible one-neighbour (run, node) cases have R² ≥ 0.9;
    - (b) ≥ 90% of both-neighbour cases (9 cases on tree32, so all of them) have bond R² ≥ ceiling − 0.05;
    - (c) at the meeting node, median KL(sub-patch) < 0.5 × median KL(unpatched) and < median KL(random subspace);
  - partly supported if two hold;
  - not supported otherwise.
- **H5** (tree32, final checkpoint, internal nodes with background positions):
  - supported if (c) median sub-patch moved ≥ 0.8 with random ≤ 0.4, and (b) median invariance on triggered examples at level ≥ 2 is ≤ 0.1 and below its step-0 value;
  - partly supported if only (c) holds;
  - not supported otherwise.
  - tree16 and perm32 are reported the same way. For seed 2 the permutation keeps leaf 15 in place, and the background is symmetric under any permutation, so perm32 seed 2 is the same task as tree32 (noted when the two are compared).
- **Inconclusive:**
  - more than a quarter of the planned runs didn't finish (this overrides every verdict);
  - no interpretation files for H4 or H5.
- **Run failed:** the checks fail, CUDA is unavailable, or no tree run finished.

## Time and cost

- **Setup:** about 10 minutes, plus about 5 minutes for checks and rank.
- **Pilots:** 8 keys × 4 LRs × 4,000 updates.
- **Training:** 48 × 30,000 updates.
  - On the coordinator's CPU: tree16 about 16 ms/update, tree32 about 50 ms, the width-40 chain about 140 ms, and the width-78 chain about 450 ms.
  - On a GPU, a few ms for the trees and about 10–20 ms for the wide chain, so about 1–2 h with 6 runs sharing the GPU.
  - The deadline is 5 h.
- **Interpretation:** 36 runs, deadline 2 h.
- **Total:** at most about 7.5 h, under the 10 h cap.
- **Cost:** at most $10.90 at $1.09/h, inside $12. Expected about $1–3.

## If something fails

- **Checks fail or CUDA is unavailable:** stop, finish the worker, and mark the experiment failed.
- **A run crashes:** rerun the failed runs once into `runs/lang-evidence-rerun`, after copying `lr.json` there, with `--only <names> --hours <time left − 1>`. Record it as an amendment, and analyze those runs as descriptive.
- **The deadline skips runs:** seed 0 runs first. Rerun the skipped runs as above if the cap allows; otherwise report them as censored.
- **An arm collapses in more than a quarter of its runs:** rerun that arm once at the next lower grid LR into `runs/lang-evidence-rerun` as an amendment, and report both.
- **Interpretation times out:** analyze what finished, and report how many runs H4 and H5 rest on.
- **Out of GPU memory:** rerun with `--procs 3`.
- **No GPU in stock:** wait, then report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Batch and budget:** as for lang-source. The user chose 4 experiments with a 10 h cap and $12 per worker, then said "no budget, just work until finished", which is a standing go-ahead for the set.
- **Interpretation depth:** the user chose probes plus invariance plus subspace patching at every checkpoint.
- **Widths:** the user's guidance was r = 16 for the masked bigram with an r = 32 check at an aligned and a root-straddling mask, and for the trigger r = 32 as main against r = 16. The same width is used for tree and perm, and the chain is matched on parameter count.
- **Design from the user's review:**
  - the reserved trigger symbol;
  - the masked-bigram comparison read as evidence aggregation, with no predicted tree-vs-chain ordering;
  - tying isn't assumed to help.

## Review

An independent review returned "needs revision". Findings and changes:

- **The meeting node was never selected (P0, an off-by-one level in `analyze.py`).** It is now found as the lowest node holding both neighbours (`analysis.meeting_level`). Verified: p = 5 gives level 2, p = 4 level 3, p = 8 the root.
- **At r = 16 the random-subspace control is almost the whole bond (P0).** H4 is now decided on the r = 32 tree (comb subspace about 15 of 32 dimensions), with tree32 added at every mask position. fwd, bwd and comb R² at the meeting node are reported to separate "added" from "carried side by side".
- **R² ≥ 0.9 is set by the log-probability tail (P1).** Evidence targets are clipped at log 10⁻³ before centering. On re-check, the clip made the ceiling depend on the sharpness of the model's tail (exact Bayes reaches 0.81–0.83), so (b) is now relative to each run's own ceiling: the R² of the same ridge from the model's centered output logits, minus 0.05. (a) is unaffected (R² 0.999–1.000 from one-hot).
- **H5(a) and H5(b) pass at initialization (P1).** (c), the causal patch, is now deciding. Invariance is computed on examples with a trigger in the span, at level ≥ 2, against its step-0 value, and reported per level. Flag decodability is descriptive, with step-0 and chance. H4(a) excludes leaves and the mask-sibling nodes, and reports step 0.
- **The permutation moves the meeting level both ways (P1).** H3 is now descriptive, with tree and perm runs pooled by their recorded meeting level. The reviewer's table was verified against the code.
- **Re-check.** It confirmed that the P0s and the H5 fix hold, and found one remaining P1: the clipped-target ceiling (fixed as above with the per-run logit ceiling, the reviewer's proposed fix).
- **P2s.** The rank claim is reworded as context, not part of a criterion. The quarter-unfinished override and a minimum number of capacity pairs are implemented. H2 without a direction reads "unresolved at n = 3". The perm32 seed-2 equivalence to tree32 is noted. Shared code is committed before launch.

## Amendments

None.

## Outcome

Not run yet.
