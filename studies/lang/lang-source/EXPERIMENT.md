---
name: lang-source
title: "Selecting and combining source tokens in a TreeTN: fixed source vs two-source conjunctions whose meeting point moves through the tree"
status: done
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 10
budget_usd: 12
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-13T23:50:56Z
result: not-supported
finished_utc: 2026-11-14T02:48:29Z
writeup: studies/lang/lang-source/README.md
---

# Selecting and combining source tokens in a TreeTN: fixed source vs two-source conjunctions whose meeting point moves through the tree

## Question

The RHM results show that a TreeTN learns a grammar whose hierarchy matches its own. They can't separate two things a bond might be doing: preserving a useful token along its ancestor path, and combining two tokens once both are inside its span. This experiment isolates both with iid context, so the only thing that moves is where the useful inputs sit:

1. **Selection.** When the target depends on one token x_j (y = σ(x_j)), does the position of j matter? For the untied tree it can't, by symmetry, since all 16 leaves are equivalent under the tree's automorphisms and the context is iid. That makes this a control. The chain reads x_j 15 − j steps before its output, so for the chain position can matter.
2. **Composition.** When y = f(x_i, x_j) is a random conjunction, the lowest common ancestor (LCA) of leaves i and j is the first node where the network can combine them. Does learning speed depend on the LCA level, and does the network build f at the LCA and carry it upward?
3. **Mechanism.** Is the predicted summary (the source token below the LCA, f(x_i, x_j) at and above it) decodable from the bonds, not decodable additively, and causally used, as shown by subspace patching against random-subspace controls?

This is the first stage of the plan in `notes/2026-11-13-language-processes/plan.md`, revised after review. It uses the clean iid-context version of the fixed source, and a two-source process takes the place of the lag sweep, because with a masked last leaf the LCA with leaf 16 is not where any interaction happens.

## Hypothesis

- **H1 (symmetry control, tree).** The untied tree's steps to threshold do not depend on the source position j (fixed source), and they agree across the three root-LCA pairs (7,8), (0,8) and (0,15). This is a derived consequence of the architecture's automorphism symmetry plus iid inputs, so a violation would point to an implementation problem, not a scientific finding.
- **H2 (chain, sequence distance).** The chain learns the fixed source faster when it reads the source last (j = 15) than first (j = 0). Mechanism: the chain must carry x_j through 15 − j multiplications by its shared core. Against: no consistent difference, or the opposite.
- **H3 (tree, LCA level; two-sided, no predicted direction).** For the untied tree, steps to threshold for y = f(x_i, x_j) differ between an LCA at level 1 (pair (0,1)) and an LCA at level 3 (pair (3,4)).
  - Both conjunctions are formed in identity-initialized fusion cores, so only the level differs: a level-1 LCA forms f low and passes it through two more cores, while a level-3 LCA forms it just below the root.
  - Either consistent direction (3 of 3 seeds) is a finding. Otherwise the outcome is "unresolved at n = 3", not evidence of no effect.
  - Level 1 vs the root is reported separately and labelled as confounded. A root conjunction is formed in C, which keeps its variance init (interaction entries with std 1/r ≈ 0.06, against about 0.006 in the identity-initialized cores), so that comparison can't separate level from the core's init.
  - two_3_4 is also the LR-pilot condition, but on seed 9, which no evaluation run uses.
- **H4 (descriptive).** For the root-LCA pairs, the chain slows down with sequence distance ((0,15) vs (7,8)) and the tree does not (H1).
- **H5 (consistency check of the mechanism, untied tree, finished runs, final checkpoint).** The tree's logits are linear in each bond with the others held fixed. So once the task is learned, the cut-rank argument forces the LCA bond to determine f, and these criteria mostly confirm that the learned solution has the predicted form rather than testing a risky prediction:
  - (a) The source token stays decodable (probe accuracy ≥ 0.95) on every ancestor bond at levels 1 and above, below where it is combined. Leaf bonds are the token's own embedding and are excluded.
  - (b) For LCA levels 1–3, f(x_i, x_j) is decodable (≥ 0.95) on the LCA bond and every bond above it, and the LCA bond's f probe gains ≥ 0.2 over the same bond at step 0. The additive one-hot probe of the tokens is reported as a data floor (0.45–0.53 for these tables).
  - (c) Patching only the probe's f-subspace from a donor at the LCA moves the prediction to the counterfactual in ≥ 80% of the pairs where the argmax changes, and at least 0.4 more often than random subspaces of the same dimension.
- **H6 (selectivity; directional, can fail).** Distractor tokens become less decodable on the bonds that carry the sources: the median change of the distractor probe from step 0 to the final checkpoint, over (run, bond) cases at levels ≥ 1 whose span holds a source, is ≤ −0.2. Under the identity init every token passes into the bonds additively, and nothing in the loss explicitly rewards discarding a distractor, so the opposite (no change) is plausible. Disclosure: in a 300-update CPU smoke run of one tree, this fall was visible at one node (0.97 to 0.38), and that informed the direction.

## Method

- **Processes** (`studies/lang/processes.py`): V = 8 symbols, 16 iid uniform input tokens, target noise ε = 0.05 (with probability ε the target is replaced by a uniform symbol, so Bayes logits stay finite). Bayes accuracy is 1 − ε + ε/V = 0.956.
  - `fixed_j{j}`: y = σ(x_j), σ a seeded random permutation, j ∈ {0, 7, 8, 15} (0-based leaves).
  - `two_{i}_{j}`: y = f(x_i, x_j), f a seeded random 8 × 8 table with uniform values, so it is not additive and each source alone is only partly informative. The pairs and their LCA levels:
    - (0,1): level 1.
    - (1,2): level 2.
    - (3,4): level 3.
    - (7,8), (0,8), (0,15): the root (root core C), at sequence distances 1, 8 and 15.
  - The process instance (σ, f) is seeded by the run's seed.
- **Arms**:
  - `tree`: untied TreeTN at r = 16; 61,440 parameters for V_in = O = 8 (the r = 32 tree has 471,040).
  - `tied`: `tie='level'`, at r = 16.
  - `chain`: the shared-core MPS (`ctn.tree.ChainTN`), with its width chosen so its parameter count is closest to the r = 16 tree's: width 39.
  - `tree32`: untied tree at r = 32, run on `fixed_j0`, `two_0_1` and `two_7_8` only. This is a capacity check: the sampled centered-logit rank of every cut is ≤ 8 (`studies/lang/results/rank.json`), so r = 16 should not bind.
- **Init**: identity init (`ctn.tree.identity_init`, ε_init = 0.1) for every arm, as in the RHM experiments.
- **Training**:
  - Online data, a fresh batch of 256 every update, from a device generator seeded per (condition, seed), shared by every arm.
  - Adam, cross-entropy, 20,000 updates.
  - Linear warmup over 100 updates, then a cosine schedule to 0.1 × peak (`studies/rhm/train.py`, reused).
- **LR**: piloted per arm over {1e-4, 3e-4, 1e-3, 3e-3} on `two_3_4` with seed 9, which no evaluation run uses. The pilot runs 2,000 updates of the 20,000-update schedule (a longer pilot would choose by post-convergence loss) and picks the lowest diagnostic loss. An LR whose pilot learned and then collapsed to within 0.1 nats of log V is excluded. Choices are in `lr.json`.
- **Grid**: 10 conditions × 3 arms × seeds 0–2, plus `tree32` on 3 conditions × 3 seeds, for 99 runs. They are ordered by seed, so all of seed 0 runs first.
- **Measures**:
  - Diagnostic set: 20,000 fixed examples per process instance. Loss, accuracy and KL to Bayes are logged every 10 updates up to 2,000 (these processes are learned within a few hundred updates), plus about 60 log-spaced updates, every 50 up to 3,000 and every 1,000 after.
  - Final test set: 50,000 examples.
  - Steps to threshold t: the first logged update with diagnostic accuracy ≥ 0.9 × Bayes accuracy; +inf if never (censored).
- **Checkpoints**: for the tree-type arms, at updates 0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1,200, 2,000, 5,000, 10,000 and 20,000, plus the shared RHM checkpoint steps scaled to 20,000.
- **Interpretation** (`studies/lang/interpret.py`, on every tree, tied and tree32 run, at every checkpoint):
  - Bonds are whitened over 6,000 probe examples, which makes the probes gauge invariant.
  - Each node gets held-out probes of every summary the process defines over its span: the source token, f where both sources are inside, and one distractor token.
  - Baselines: the step-0 checkpoint, shuffled labels at the final checkpoint, and an additive one-hot probe on the span's tokens.
  - Invariance: distractors in the span are resampled.
  - Interventions: the probe subspace is patched from a donor, compared with 3 random subspaces of the same dimension and with whole-bond patching. The counterfactual is the Bayes posterior of the input with only that summary's tokens spliced in.

### Comparison rules (`studies/lang/analysis.py`, `analyze.py`)

- **faster(a, b)**: t_a < t_b, with t_b − t_a greater than the logging interval at either t, and t_a < t_b / 1.25. Finite beats +inf. A pair where both runs are censored is uninformative and excluded (counted as both_censored).
- **Cumulative LR.** Comparisons between arms with different piloted LRs (tree vs chain, tree vs tied) are also reported on cumulative LR to threshold, Σ η_s over the updates before t, the project's gradient-flow-time convention.
- **better(a, b)**: lower final test KL, outside max(0.002 nats, 10% of the larger). Otherwise a tie.
- Only pairs where both runs finished enter. Dropped pairs are counted.
- **Collapsed runs** (learned to below log V / 2, then returned within 0.1 nats of log V) stay in and are listed.

## Inputs

None. All data is generated from seeded processes. No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
mkdir -p "$EXP_DIR/results" "$EXP_DIR/figures"
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python -m studies.lang.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python -m studies.lang.rank "$EXP_DIR/results/rank.json" --configs lang-source
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-source/run.py --out runs/lang-source --hours 5
PYTHONPATH=. uv run --frozen python -m studies.lang.interpret_all runs/lang-source --procs 4 --hours 2
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-source/analyze.py runs/lang-source "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/lang/lang-source/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

Each line can be its own `orch exec` step. Run the training and interpretation from a background command. The checks exit non-zero on any failure, which stops the job. `--hours 5` is the training deadline, counted from the end of setup and the LR pilots.

## Expected outputs

- `results/checks.json` and `results/rank.json`.
- `runs/lang-source/<cond>_<arm>_s<seed>/`: `result.json`, checkpoints (tree-type arms, about 13 × 0.3–2 MB), and `interpret.json` (tree-type arms).
- Also in `runs/lang-source/`: `lr.json`, `meta.json`, `planned.json`, `status.json`, `clock.json`, `interpret_status.json`, `pilot/` and `source/`. About 1 GB in all.
- `results/analysis.json` (health, every verdict with its pairs), `results/curves.json` and `results/interp.json`.
- `figures/curves.pdf`, `figures/t.pdf` and `figures/probes.pdf`.
- `README.md`: the write-up.

## Controls

- **H1 as a control.** It's a symmetry check, so it tests the pipeline.
- **Pairing.** Every arm with the same (condition, seed) sees the same data stream and process instance, so comparisons are paired.
- **Chain matching.** The chain is matched on parameter count to the r = 16 tree. Matching on width would answer a different question.
- **Capacity.** The r = 32 tree checks that r = 16 isn't a bottleneck, and the sampled rank (≤ 8 at every cut) says it shouldn't be.
- **Probe baselines:**
  - the untrained step-0 checkpoint;
  - shuffled labels;
  - an additive one-hot probe on the span's tokens, which can't read a conjunction;
  - random-subspace patches of the same dimension;
  - whole-bond patching, which equals running on the spliced input (verified by `checks.py`).
- **Correctness checks** (`studies/lang/checks.py`) run first:
  - the Bayes predictors match generation frequencies;
  - closed-form cross-entropies;
  - summaries depend only on their span;
  - invariance moves keep the summary;
  - patching equals splicing;
  - rank on known cases.

## Success and failure criteria

Computed by `analyze.py`:

- **H1:**
  - "holds" if, for the fixed-source positions and for the three root pairs, the spread of per-condition median t is at most the larger of the seed range and the logging interval;
  - "violated" otherwise. A violation is investigated as a possible bug and reported, and it doesn't change the other verdicts.
- **H2:**
  - supported if the chain at j = 15 is faster than at j = 0 in 3 of 3 seeds;
  - partly supported at 2 of 3 with no seed the other way;
  - not supported otherwise.
- **H3:**
  - "level 1 faster" or "level 3 faster" if one direction holds in 3 of 3 seeds (two_0_1 vs two_3_4);
  - "unresolved at n = 3" otherwise;
  - inconclusive with fewer than 2 usable pairs.
  - Level 1 vs root is reported separately and labelled confounded with C's init. The root-level run is the root pair with the median t in that seed.
  - The Spearman correlation of t with LCA level over all tree runs is reported, as descriptive.
- **H4:** descriptive.
- **H5:**
  - "consistent" if (a), (b) and (c) all hold over the tree arm's finished, interpreted runs, at the final checkpoint:
    - (a) ≥ 90% of (run, ancestor bond at level ≥ 1) cases have probe ≥ 0.95;
    - (b) ≥ 90% of (run, node holding both sources) cases have an f probe ≥ 0.95, with a median gain at the LCA over step 0 of ≥ 0.2;
    - (c) mean sub-patch "moved" ≥ 0.8 and mean sub − random ≥ 0.4 at the LCA;
  - "partly consistent" if two hold;
  - "not consistent" otherwise.
  - Tied and tree32 are reported the same way, as descriptive.
- **H6:** supported if the median distractor change is ≤ −0.2; not supported otherwise.
- **Inconclusive** (implemented in `analyze.py` and `studies/lang/analysis.apply_health_override`):
  - more than a quarter of the planned runs didn't finish (this overrides every verdict);
  - a comparison has fewer than 2 usable pairs;
  - for H5, no interpretation files.
- **Run failed:** the checks fail, CUDA is unavailable, or no tree run finished.

## Time and cost

- **Setup:** about 10 minutes.
- **Checks and rank:** about 5 minutes.
- **Pilots:** 16 runs × 2,000 updates.
- **Training:** 99 × 20,000 updates.
  - On the coordinator's CPU, the r = 16 tree took about 7 ms/update, r = 32 about 26 ms and the chain about 75 ms.
  - On a GPU the trees should be launch-bound at a few ms/update and the chain slower. With 6 runs sharing the GPU, the grid should take about 1–2 h.
  - The training deadline is 5 h.
- **Interpretation:** 69 runs with about 20 checkpoints each (about 1.5 minutes per run on CPU, less on a GPU), so about 30–60 minutes with 4 at a time. Its deadline is 2 h, and seed 0 goes first.
- **Total:** at most about 0.3 + 5 + 2 + 0.2 ≈ 7.5 h, under the 10 h cap.
- **Cost:** at up to $1.09/h (L40S), the cap costs at most $10.90, inside the $12 budget. The expected cost is about $1–3.

## If something fails

- **Checks fail or CUDA is unavailable:** stop, finish the worker, and mark the experiment failed.
- **A run crashes:** the others continue. Rerun the failed runs once into `runs/lang-source-rerun`, after copying `runs/lang-source/lr.json` there, with `--only <names> --hours <time left − 1>`. Note it as an amendment and analyze those runs as descriptive.
- **The deadline skips runs:** seed 0 runs first, so later seeds are what gets skipped. If the cap leaves time, rerun the skipped runs as above. Otherwise report them as censored under the pre-set "fewer than 2 usable pairs" rule.
- **An arm's pilot picks a grid-edge LR:** proceed, and report it.
- **An arm collapses in more than a quarter of its runs:** rerun that arm once at the next lower grid LR into `runs/lang-source-rerun` as an amendment, and report both.
- **Interpretation times out:** analyze whatever finished. `interpret_status.json` lists the rest, and the write-up reports how many runs H5 rests on.
- **Out of GPU memory** (not expected at these sizes): rerun with `--procs 3`.
- **None of the GPU types are in stock:** the launcher waits and retries. If none is found, report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Which experiments, in what batch?** The user chose 4 experiments: lang-source, lang-evidence, lang-parity and lang-induction.
- **Budget?** The user first chose a 10 h cap and $12 per experiment, then said "no budget, just work until finished". The cap and budget stay as per-worker safety limits. The user's message is a standing go-ahead for this set of four.
- **Interpretation depth?** The user chose probes plus invariance plus subspace patching at every checkpoint.
- **Bond dimension?** The user's guidance was r = 16 for the simple processes, with r = 32 as a capacity check, and the same width for tree, perm and tied. The chain is matched on parameter count as a separate comparison.
- **Design revisions from the user's review of the plan:**
  - the iid-context fixed source;
  - the LCA-level claim is two-sided rather than predicted bands;
  - a two-source interaction whose meeting point moves through the tree;
  - centered logits for the rank diagnostic.

## Review

An independent review returned "needs revision". Findings and changes:

- **The pre-registered inconclusive rules weren't in the code (P1).** `analyze.py` now overrides every verdict when more than a quarter of the planned runs didn't finish (`analysis.apply_health_override`), and returns inconclusive when a comparison has fewer than 2 usable pairs. Both-censored pairs are excluded from counts.
- **H3 confounded LCA level with the root core's init (P1).** The primary comparison is now level 1 vs level 3 (both in identity-initialized cores). Level 1 vs root is reported separately and labelled confounded, and the pilot-condition overlap (two_3_4, seed 9) is noted.
- **H5 can hardly fail once the task is learned (P1).** It is reframed as a consistency check. The additive-probe clause, which reads the data rather than the bond, is replaced by the LCA bond's gain over step 0, and the causal clause needs sub − random ≥ 0.4 instead of random ≤ 0.4. A trajectory prediction that can fail is added as H6 (distractor decodability falls), with a disclosure that a smoke run informed its direction.
- **Timing resolution (P2).** Logging is now every 10 updates up to 2,000, checkpoints are dense below 2,000, and the pilot is shortened to 2,000 updates.
- **H5(a) padded by leaf bonds (P2).** It now starts at level 1.
- **Uncommitted shared code (P2).** `studies/lang/` is committed with the definitions before the first launch.
- **Minor.** H5 uses finished runs only, and tree-vs-chain and tree-vs-tied speed are also reported on cumulative LR (η × steps).

## Amendments

None. Every step ran as written. The one contingency that applied was a grid-edge learning rate: the tree32 pilot picked 1e-4, the bottom of the grid, and the run proceeded as "If something fails" says.

## Outcome

Result: **not supported**. The two directional predictions (H2, H6) failed, H3 is unresolved, and the control and the consistency check passed. Write-up: [README.md](README.md).

- **H1 (symmetry control):** holds. The tree's median t was 79, 79, 79 and 70 at j = 0, 7, 8 and 15 (spread 9, against a seed range of 13), and 130 at all three root pairs.
- **H2 (chain distance):** not supported. j = 15 against j = 0 tied in 3 of 3 seeds (median t 60 and 60). Middle positions took 810–910 (not predicted).
- **H3 (tree meeting level):** unresolved at n = 3. Level 1 against level 3 tied in 3 of 3 seeds (t 130 against 130–140). Level 1 against the root (confounded) also tied in 3 of 3. Spearman ρ = −0.14.
- **H4 (descriptive):** the tree took 130 at every root pair. The chain took 980 at (7,8), 2,550 at (0,8) and 140 at (0,15), so it was faster at (0,15) than at (7,8) in 3 of 3 seeds.
- **H5 (mechanism):** consistent. (a) 108/108 cases, (b) 18/18 cases with a median gain of 0.61, (c) sub-patch 1.00 against random 0.33.
- **H6 (distractor):** not supported. The median change was 0.00 over 123 cases.
- **Health:** 99/99 runs finished, 0 collapsed, 0 censored, and 69/69 were interpreted.
- **Cost:** A40, 2 h 47 m, about $1.64, against a 10 h cap and a $12 budget.
