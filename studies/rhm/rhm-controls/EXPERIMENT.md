---
name: rhm-controls
title: "RHM architecture controls: correct tree vs permuted leaves, tied cores and a parameter-matched chain"
status: done
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 6
budget_usd: 6.6
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-12T21:42:22Z
result: supported
finished_utc: 2026-11-12T22:14:47Z
writeup: studies/rhm/rhm-controls/README.md
---

# RHM architecture controls: correct tree vs permuted leaves, tied cores and a parameter-matched chain

## Question

When a TreeTN learns the Random Hierarchy Model, which parts of the architecture matter? Three questions, each answered by one paired comparison:

1. Does matching the data hierarchy help? The untied tree on the correct leaf order is compared with the same tree on a fixed random leaf permutation.
2. Does weight sharing help or hurt? The untied tree is compared with a level-tied tree: one leaf embedding and one fusion core per level, matching the grammar's own position-independent rules.
3. Does the hierarchy matter beyond being a tensor-network parameterization? The level-tied tree is compared with a shared-core chain.
   - Both share one leaf embedding across positions and, under the identity init, start position-blind, so they differ in structure rather than in positional freedom.
   - The untied tree vs the chain is also reported. It can't carry this question, because the untied tree additionally has position-specific parameters.

## Hypothesis

Same grammar and tasks as `rhm-dynamics` (v = n = 16, m = 4, s = 2, L = 4; 'root' and 'last').

- **H1, hierarchy match helps.** The correct tree reaches threshold sooner and ends with better final quality than the permuted tree.
  - Why: a random permutation splits sibling tokens across subtrees, so a bond must summarize a non-contiguous set of tokens whose latent structure doesn't factor, which r = 16 can't do exactly.
  - Against: the permuted tree is as fast and as good in a majority of pairs, or better.
- **H2, weight sharing (two-sided, no directional prediction).** The RHM's rules are the same at every position of a level. Tying could help by pooling gradient signal across positions, or hurt by removing positional freedom. Either consistent direction counts as a finding, and so does no consistent difference.
- **H3, the hierarchy beyond tensor parameterization.** The level-tied tree beats the parameter-heavier chain on both measures.
  - Why: a chain must carry every pending partial parse in one state, while each tree bond needs only one latent symbol.
  - Against: the chain is as fast and as good as the tied tree.
- Descriptive: untied tree vs chain (the parameter-matched pair).

All results are for these architectures with the identity init.

## Method

- **Initialization**: identity init (`ctn.tree.identity_init`, cfg `init='identity'`, eps = 0.1) for every arm. Bond channel 0 carries a constant and the fusion cores pass a signal times the other child's constant, so each network starts near its additive part.
  - Why: the TreeTN's default variance init collapses on the RHM. In CPU probes at r = 16, on both tasks and both LRs tried, Adam drove every core toward 0 within about 300 updates (the uniform-output saddle), and the diagnostic loss stayed at log 16 = 2.773.
  - With the identity init the probes learned: r = 16 root accuracy reached 0.94 by update 500 with the 100-update warmup. In probes under the earlier 1,000-update warmup, root accuracy at update 3,000 was 0.41 for the permuted tree, 0.64 for the chain and 1.00 for the tied tree. So every arm learns something, and no comparison is between collapsed models.
  - The tied tree and the chain start position-blind: one leaf embedding for all positions, so position enters only through the structure.
- **Arms** (all r = 16 except the chain width):
  - `tree`: untied TreeTN on the correct leaf order; 65,536 parameters ('root') / 65,792 ('last').
  - `perm`: the same untied TreeTN on `x[:, perm]`.
    - One seeded permutation per (grammar, init seed) (seed 500 + 10 g + s), redrawn until no sibling pair of the data tree are siblings in the model.
    - The permutations, and how many model nodes still cover a data node's exact leaf set at levels 1–3, are saved in `controls_meta.json`.
  - `tied`: `tie='level'`, with one leaf embedding A for all 16 leaves and one fusion core per level; 16,640 / 16,656 parameters.
  - `chain`: ChainTN (h_t = W(h_{t−1}, A onehot(x_t)), logits = U h_16, one W and one A for all positions).
    - Width 40, the width whose parameter count is closest to the untied tree's: 65,320 ('root') / 65,360 ('last').
    - The identity init carries over: h_0 = e_0 + noise, and W has the same pass-through structure, with h_{t−1} as the left child.
- **Grid**: 4 arms × 2 tasks × grammars 0–2 × init seeds 0, 1 = 48 runs. Seed 0 runs first (all arms, grammars, tasks), then seed 1. Every arm with the same (grammar, seed) sees the same online data stream.
- **Training**: as `rhm-dynamics`. Online data, cross-entropy, Adam, batch 256, float32, 40,000 updates. The LR warms up linearly over 100 updates (short, so steps to threshold don't measure the LR ramp), then follows a cosine to 0.1 × peak.
- **LR**: piloted per (task, arm) on grammar 0 over {1e-4, 3e-4, 1e-3, 3e-3}.
  - The pilot uses init seed 9, which no evaluation run uses, and runs 4,000 updates of the 40,000-update schedule; the lowest diagnostic loss wins.
  - Each arm gets its own best LR, so an arm doesn't lose only because the tree's LR suits it badly.
  - Choices and grid-edge flags are in `lr.json` and `results/analysis.json`.
  - `--lr` overrides the pilot; reruns go to a separate `--out`.
- **Measures**:
  - Steps to threshold: the first logged update where diagnostic accuracy (20,000 fixed trees per grammar) reaches 0.9 × Bayes accuracy, +inf if never. Logged at 60 log-spaced points, every 50 updates up to 3,000, and every 1,000. The cumulative LR to threshold is reported next to the steps.
  - Final test error rate and test loss on 50,000 separate trees.
  - ms/update per arm.
  - Checkpoints at the standard steps are saved for later analysis.

### Comparison rules (`studies/rhm/analysis.py`)

Each comparison A vs B has 12 pairs: (task, grammar, init seed). A is the reference: `tree` for H1, H2 and the descriptive tree vs chain comparison, `tied` for H3.

- **Which runs enter.** Only pairs where both runs finished enter. A pair where either run was truncated, diverged, skipped or failed, or has a non-finite metric, is dropped. Dropped pairs are counted and never count as ties.
- **Steps to threshold.** A is faster only if t_A < t_B − margin, with margin = max(the logging interval at the later of the two, the reference arm's seed range of t on that (task, grammar)). Otherwise the pair is a tie. Finite beats inf, and inf vs inf is a tie.
- **Final quality.** Test error rate first, with a tie within max(1e-3, the reference's seed range). Then test loss, with a tie within max(1e-3 nats, the reference's seed range). Float32 saturates 'root' losses near 0, so losses below 1e-3 nats are not distinguished.
- **Direction of a measure.** A measure goes to A (or B) if that arm wins at least 10 of the 12 pairs and the consistency rule holds (wins > losses in at least 2 of the 3 grammars and losses > wins in none). Ties count for neither side. 10 of 12 without ties is a one-sided sign test p = 0.019. Sign-test p-values and per-grammar counts are reported.
- **Reference must learn.** A comparison is inconclusive if its reference arm didn't learn on a task: median test accuracy < 2 × chance (0.125), or median final diagnostic loss within 0.05 nats of log 16. Whether the other arm learned is reported too, and if the other arm (the control) did not learn on some task by the same rule, the verdict is tagged "[but <arm> did not learn …: possibly an optimization failure]" and can't be reported as clean support: a control stuck at the uniform-output saddle or at a bad LR hands every pair to the reference.

## Inputs

None. All data is generated from the seeded grammars. No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-controls/run.py --out runs/rhm-controls --hours 4.5
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-controls/analyze.py runs/rhm-controls "$EXP_DIR/results"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-controls/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero on any failure, which stops the job. `--hours 4.5` is the training deadline, counted from the end of setup and the LR pilots. Run the training as its own `orch exec` step from a background command.

## Expected outputs

- `results/checks.json`.
- `runs/rhm-controls/<task>_<arm>_g<g>_s<seed>/`: `result.json` and 8 checkpoints (about 0.3 MB each).
- Also in `runs/rhm-controls/`: `controls_meta.json` (chain width, parameter counts, permutations), `lr.json`, `planned.json`, `status.json`, `clock.json`, `data/`, `pilot/` and `source/`. About 150 MB in total.
- `results/analysis.json`: health against the plan, which arms learned, every comparison with pair details, per-grammar counts, dropped pairs and per-task breakdowns, and the per-arm table with ms/update.
- `results/curves.json`.
- `figures/curves.pdf`, `figures/paired.pdf`.
- `README.md`: the write-up.

## Controls

- The untied correct tree is the reference for H1 and H2, rerun in the same job with the same data streams and seeds.
- The tied tree is the reference for H3. It and the chain both have one shared leaf embedding and start position-blind, so the comparison isolates the tree structure.
- The chain is matched on parameter count to the untied tree (within 0.7%), not on bond dimension. The tied tree is deliberately not matched: it has a quarter of the parameters, since sharing is the factor under test.
- Pairing by (task, grammar, init seed) removes grammar and data variation from each comparison, and the per-grammar counts show whether one grammar drives a result.
- The permutation removes every sibling pair, so `perm` cannot recover the level-1 structure by chance.
- The correctness checks run first (`studies/rhm/checks.py`).

## Success and failure criteria

Computed by `analyze.py` with the comparison rules above.

- **H1 (tree vs perm):**
  - supported if both measures go to the tree;
  - partly supported if one measure goes to the tree and none to `perm`;
  - not supported otherwise.
- **H2 (tree vs tied, two-sided):** "tied better" or "tree better" if both measures go one way. "Better on one measure" if only one measure has a direction. "No consistent difference" otherwise.
- **H3 (tied vs chain):** supported if both measures go to the tied tree, partly if one does and none goes to the chain, not supported otherwise.
- **Descriptive:** tree vs chain, reported the same way; per-task results (6 pairs each); per-arm medians.
- **Inconclusive:**
  - fewer than 12 usable pairs in a comparison (the number dropped is reported);
  - the reference arm didn't learn;
  - more than a quarter of the planned runs didn't finish (this overrides every verdict).
- **Run failed:** checks fail, CUDA is unavailable, or no `tree` run finished.

## Time and cost

- **Setup:** about 10 minutes.
- **LR pilots:** 32 runs × 4,000 updates (4 arms × 2 tasks × 4 LRs), about 15–20 minutes with 6 at a time.
- **Training:** 48 × 40,000 updates, about 1.9 M.
  - On the coordinator's CPU (one thread) the chain is the slowest arm at 36 ms/update (16 sequential steps at width 40), and the trees take 5–6 ms.
  - On the GPU the trees should be launch-bound at roughly 2–5 ms/update and the chain somewhat slower (not measured), so the grid takes about 1–2 h with 6 runs sharing the GPU.
  - The training deadline is 4.5 h after the pilots. If runs are skipped, seed 1 goes first.
- **Total:** at most about 0.4 h + 4.5 h + 0.1 h ≈ 5 h, under the 6 h cap.
- **Cost:** at up to $1.09/h (L40S, secure), the 6 h cap costs at most $6.54, inside the $6.6 budget.

## If something fails

- **Checks fail or CUDA is unavailable:** stop, finish the worker, mark failed.
- **A run crashes:** the others continue. Rerun the failed runs once into a separate `--out runs/rhm-controls-rerun` after copying the original LR choices there (`mkdir -p runs/rhm-controls-rerun && cp runs/rhm-controls/lr.json runs/rhm-controls-rerun/`), with `--only <names> --hours <left - 0.75>`, so the original results are kept, and note an amendment. Analyze the rerun separately, as descriptive.
- **An arm diverges in more than a quarter of its runs:** rerun that arm once with `--lr <the next lower grid LR>` into `runs/rhm-controls-rerun`, as an amendment, and report both.
- **The first wave's ms/update projects past the deadline:** let the grid run (seed-1 runs are skipped first and counted). If the worker cap leaves time, rerun the skipped runs with `--only` into a separate `--out`, after copying `lr.json` there. Otherwise the 6-pair seed-0 comparisons are reported as descriptive under the pre-set "fewer than 12 pairs" rule.
- **None of the GPU types in stock:** the launcher waits and retries; if none is found, report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Arms?** The user specified:
  - the untied correct tree (rerun);
  - a fixed random leaf permutation (seeded, recorded);
  - level-tied cores;
  - a shared-core chain parameter-matched to the r = 16 tree;
  - all at r = 16, both tasks, 3 grammars, init seeds 0 and 1.
- **Comparison?** Steps to threshold (first update with diagnostic accuracy ≥ 0.9 of Bayes) and final quality, paired by (grammar, seed), with never-reached as +inf, as the user specified. After review: margins, tie tolerances, consistency across grammars, and the tied tree as H3's reference.
- **Chain width?** Left to us: width 40 matches the untied tree's parameter count within 0.7%.
- **Initialization and warmup?** The user chose the identity init (eps 0.1) as the main init for all RHM experiments, after the CPU probes showed the variance init collapsing, and a 100-update warmup for every RHM run.
- **LR per arm?** A per-arm pilot over {1e-4, 3e-4, 1e-3, 3e-3}, so an arm isn't penalised by an LR chosen for another architecture.

## Review

An independent review returned "needs revision". Findings and changes:

- **H3 rested on the untied tree vs chain, which also differ by position-specific parameters.** H3 is now tied tree vs chain; both start position-blind under the identity init. Tree vs chain is still reported, as descriptive.
- **Unfinished runs and NaN metrics could enter comparisons, and health was counted from existing results.** Only pairs where both runs finished enter, dropped pairs are counted, health is counted against `planned.json`, and the "more than a quarter unfinished" flag overrides every verdict.
- **Final loss on 'root' saturates in float32.** Final quality is now error rate first, then loss, with tie tolerances max(1e-3, the reference's seed range).
- **Steps-to-threshold wins had no margin.** A win now needs more than max(the logging interval, the reference's seed range), and the cumulative LR to threshold is reported.
- **One grammar could carry a verdict.** Per-grammar counts are reported, and a direction needs the consistency rule.
- **No rule for a reference that didn't learn.** A comparison is now inconclusive if the reference's median test accuracy is < 2 × chance, or its final diagnostic loss is within 0.05 nats of log 16.
- **LR pilots.** 1e-4 was added for every arm, the pilot runs on seed 9, and grid-edge choices are recorded. `--lr` overrides exist and reruns go to a separate `--out`.
- **Budget.** The deadline starts after the pilots, ms/update is tabled per arm, and the cap and budget assume secure prices up to $1.09/h. A rerun rule for a deadline overrun was added.

## Amendments

None.

## Outcome

Result: **supported** (H1 and H3 supported, H2 no consistent difference). Write-up: [README.md](README.md).

- **Run health:** 48 of 48 planned runs finished, no pair dropped, checks 26/26; every arm learned on both tasks.
- **H1 (tree vs perm):** supported. The tree wins 12 of 12 pairs on steps to threshold and 12 of 12 on final quality, 4–0 in every grammar.
- **H2 (tree vs tied):** no consistent difference. Steps split 6–6 (the tied tree is faster on 'root', the tree on 'last'), and quality is tied in 12 of 12.
- **H3 (tied vs chain):** supported. The tied tree wins 12 of 12 pairs on both measures. On 'last' the steps win disappears in summed LR.
- **Descriptive (tree vs chain):** the tree is better on both measures, 12 of 12.
- **Caveat:** the losing controls' piloted LR sat at the grid top (3e-3) for the permuted tree on both tasks and the chain on 'root'.
- **Cost:** one RTX 4090 at $0.89/h for 25 minutes, about $0.36, against a $6.6 budget and a 6 h cap.
