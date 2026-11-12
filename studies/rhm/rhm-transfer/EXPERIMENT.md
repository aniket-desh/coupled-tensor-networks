---
name: rhm-transfer
title: "RHM feature reuse: do pretrained lower subtrees transfer to unseen root productions?"
status: approved
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 4
budget_usd: 4.5
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-12T21:43:41Z
---

# RHM feature reuse: do pretrained lower subtrees transfer to unseen root productions?

## Question

A TreeTN first learns the root task on part of the grammar: trees whose root production (the pair of level-1 symbols under the root) is one of 48 seen productions. Do its lower subtrees then hold codes of the level-1 symbols that a new readout can reuse on the 16 unseen productions?

The test trains only fresh upper cores on unseen-production trees, on top of frozen subtrees, and compares four sources for those subtrees:
- pretrained on this grammar;
- pretrained on a different grammar (trained and well scaled, but coding the wrong symbols);
- untrained;
- an oracle code of the true symbols.

**Scope.** Productions are distinct pairs, so an unseen pair can be told apart from the seen ones by its two side symbols. Phase 2 therefore tests whether per-side symbol codes can be reused by a new readout, not whether the network recombines conjunctions it has never seen.

## Hypothesis

Root task, main grammar (v = n = 16, m = 4, s = 2, L = 4), r = 16. All results are for the TreeTN with the identity init.

- **H1, the representation transfers.** With only the root core C fresh and trainable, the pretrained frozen subtrees (`frozen_learned_C`) reach 0.9 accuracy on unseen productions in fewer updates than training the whole network from scratch (`scratch`), and within 2× the median updates of the oracle (`oracle_C`: C on a fixed dense code, sqrt(r) Q e_a, of the true level-1 symbols). The same test is made with fresh level-3 cores and C (`_WC` variants), against the same oracle.

  Against: `frozen_learned` is no faster than scratch in a majority of pairs, or slower than 2× the oracle (frozen features that decode the symbols but are badly conditioned for a fresh readout).

  Subtrees pretrained on a different grammar (`frozen_other_C`) are reported as a descriptive floor; a re-check probe found it never reaches threshold, since its subtrees can't parse this grammar.
- Descriptive:
  - `frozen_learned` vs `frozen_random` (frozen at the init). Under the identity init an untrained subtree is a near-additive map of its tokens, so this comparison only shows that learned features beat a bag-of-tokens map.
  - `finetune` against `scratch` and against `frozen_learned_C`.
  - Final seen-production accuracy during phase 2 (forgetting in `finetune`).

## Method

- **Model**: untied TreeTN, r = 16, float32, identity init (`ctn.tree.identity_init`, cfg `init='identity'`, eps = 0.1). Bond channel 0 carries a constant, so the network starts near its additive part.
  - Why: the TreeTN's default variance init collapses on the RHM. In CPU probes at r = 16, on both tasks and both LRs tried, Adam drove every core toward 0 within about 300 updates, and the diagnostic loss stayed at log 16 = 2.773.
  - With the identity init the probes learned: r = 16 root accuracy reached 0.94 by update 500 with the 100-update warmup.
- **Split** (`train.split_root_productions`, seed 60,000 + grammar seed):
  - Exactly one of each class's 4 root productions is unseen (16 unseen, 48 seen), so every class keeps 3 seen productions and every unseen production's class is one the model was trained on.
  - The split is redrawn until every level-1 symbol that occurs on a side of an unseen production also occurs on that side in some seen production.
  - The split and the number of redraws are saved per unit.
- **Sampling**: the root class is uniform, the root production is uniform over the class's allowed productions (seen or unseen), and everything below follows the grammar (`train.sample_restricted`). Online, 256 fresh trees per update.
- **Units**: (grammar 0–2) × (init seed 0–2) = 9 units, seed 0 on every grammar first. Per unit, every model starts from the same init state.
  - **Pretraining** on seen productions, 5,000 updates (seen accuracy was 1.0 by about 1,500 updates in the CPU probes). Gate: seen-production diagnostic accuracy ≥ 0.95 at the end. Units that fail are excluded from every arm and counted.
  - **Other-grammar pretraining** on all productions of grammar 100 + g, 5,000 updates, with the same gate on its own root accuracy. If it fails, that unit's `frozen_other` arms are dropped and counted.
  - **Phase 2** on unseen productions, 10,000 updates per arm, run in this order:
    - `scratch`: every core trainable, from the init.
    - `finetune`: every core trainable, from the pretrained state, C included.
    - `frozen_learned_C`: leaves and fusion levels 1–3 frozen at the pretrained values, fresh C.
    - `frozen_other_C`: the same, frozen at the other-grammar values.
    - `frozen_random_C`: the same, frozen at the init.
    - `oracle_C`: fresh C on fixed top bonds sqrt(r) Q e_a of the true level-1 symbols (Q a fixed seeded random orthogonal matrix), from the exact parse: the ceiling. The code is dense because under Adam a one-hot code learns ~r times slower than dense unit-RMS bonds (a re-check probe found the one-hot oracle slower than the learned arm: 161 vs 87 updates).
    - `frozen_learned_WC`, `frozen_other_WC`, `frozen_random_WC`: leaves and levels 1–2 frozen; fresh level-3 cores and C.
  - **Fresh cores** are variance-init draws (TreeTN seed 1000 + init seed), the same draw in every arm that needs them. This includes the level-3 cores of the WC arms; under the identity init the level-3 cores would be near pass-through, so the fresh ones use the variance init as C does.
  - **Scale normalization** (`train.rescale_frozen`): the last frozen core is scaled so the fresh cores receive unit-RMS bonds on the unseen diagnostic set. W[−1] is scaled by 1/rho (rho = RMS of the level-3 bonds) for the C arms, and W[−2] by 1/rho (level-2 bonds) for the WC arms. With unit-RMS inputs, the variance-init level-3 cores give level-3 bonds of RMS about 1 (recorded).
    - Trainable cores are never rescaled, so every arm's fresh cores take the same relative steps.
    - Rescaling C instead had given the learned arms about 48 times the relative step.
- **Training**: cross-entropy, Adam, batch 256. Each phase and arm has its own Adam and its own schedule: linear warmup over 100 updates, then cosine to 0.1 × peak.
  - The short warmup matters most here, since training from scratch solves the root task in a few hundred updates.
  - Peak LR: the root-task online pilot (r = 16, grammar 0, init seed 9, which no evaluation unit uses; {1e-4, 3e-4, 1e-3, 3e-3}; 4,000 updates of 40,000; lowest diagnostic loss wins), used in every phase and arm. A grid-edge choice is recorded.
  - `--lr` overrides the pilot; reruns go to a separate `--out`.
- **Measures**: fixed diagnostic sets of 20,000 seen-production and 20,000 unseen-production trees per grammar (CPU-seeded), evaluated at 60 log-spaced updates, every 50 updates up to 3,000, and every 1,000.
  - Steps to threshold: the first logged phase-2 update with unseen accuracy ≥ 0.9, +inf if never. The cumulative LR is reported next to it.
  - Final unseen and seen accuracy, and ms/update per arm.

### Comparison rules (`studies/rhm/analysis.py`)

- **Which arms enter.** Only arms with status finished, in units that passed the pretraining gate, enter a pair. Other pairs are dropped and counted, never treated as ties.
- **Margin.** A is faster than B only if t_A < t_B − margin, with margin = max(the logging interval at the later of the two, the seed range of A's t over the units of that grammar). Otherwise the pair is a tie, excluded from sign tests. Finite beats inf, and inf vs inf is a tie.
- **Consistency.** A comparison counts only with wins > losses in at least 2 of the 3 grammars and losses > wins in none. Per-grammar counts are reported.
- **Scratch must learn.** The comparisons are inconclusive if `scratch` didn't learn: median final unseen accuracy < 2 × chance (0.125), or final unseen loss within 0.05 nats of log 16.

## Inputs

None. All data is generated from the seeded grammars. No data tars.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-transfer/run.py --out runs/rhm-transfer --hours 3
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-transfer/analyze.py runs/rhm-transfer "$EXP_DIR/results"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-transfer/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

The checks exit non-zero on any failure, which stops the job. They include:
- frozen parameters stay bitwise unchanged under the freezing used here;
- the scale normalization gives unit-RMS bonds;
- the oracle's parse-based code classifies every tree with a lookup C.

`--hours 3` is the training deadline, counted from the end of setup and the LR pilot. Run the training as its own `orch exec` step from a background command.

## Expected outputs

- `results/checks.json`.
- `runs/rhm-transfer/g<g>_s<s>/`:
  - `unit.json`: split, gates, arm statuses;
  - `pretrain/` and `pretrain_other/`: `result.json` and checkpoints;
  - one folder per arm, with `result.json` (log, rescale factors, trainable parameter count, ms/update) and its final checkpoint.
- Also in `runs/rhm-transfer/`: `lr.json`, `planned.json`, `status.json`, `clock.json`, `pilot/`, `source/`. About 60 MB in total.
- `results/analysis.json`: per-unit arm results; gate failures and missing units; criteria with pair details, per-grammar counts and dropped pairs; the per-arm table with ms/update.
- `results/curves.json`.
- `figures/phase2.pdf`, `figures/pretrain.pdf`.
- `README.md`: the write-up.

## Controls

- `frozen_other` is a trained floor: same frozen architecture, same fresh cores and normalization, same data, but its subtrees were trained on another grammar's symbols. It is descriptive: it is expected never to reach threshold.
- `oracle_C` is the ceiling: the true level-1 symbols, perfectly and densely coded (same norm and Adam geometry as learned bonds).
- `frozen_random` shows the untrained (near-additive) map.
- `scratch` is the no-transfer baseline with the whole model trainable from the same init, and it must learn for the comparisons to count.
- `finetune` shows what full fine-tuning adds and how much it forgets.
- All arms of a unit start from identical saved states and see the same phase-2 data stream.
- The split guarantees that every needed level-1 symbol was seen on its side during pretraining.
- The correctness checks run first (`studies/rhm/checks.py`).

## Success and failure criteria

Computed by `analyze.py` with the comparison rules above, over units that passed the pretraining gate.

- **H1 (`_C` variant) supported** if both hold:
  - `frozen_learned_C` is faster than `scratch`, with at least 5 untied pairs, one-sided sign-test p < 0.05, and the consistency rule;
  - its median steps to threshold are ≤ 2 × `oracle_C`'s (a dense code of the true level-1 symbols, so the ceiling has the same Adam geometry as learned bonds).
- **Not supported:** `frozen_learned_C` wins no more pairs against `scratch` than it loses, or its median steps exceed 2 × the oracle's.
- **Otherwise inconclusive:** faster than scratch in a majority, but not significant or not consistent across grammars.
- **Descriptive, expected censored:** `frozen_learned_C` vs the other-grammar floor `frozen_other_C`. A CPU re-check probe showed the floor never reaches threshold (unseen accuracy 0.12): subtrees trained on another grammar can't parse this one, so beating it only says the learned arm reaches 0.9.
- **H1 (`_WC` variant):** the same rule, reported separately, against the same oracle.
- **Inconclusive:**
  - fewer than 5 units passed the pretraining gate;
  - `scratch` didn't learn.

  The number of dropped pairs is reported with every comparison.
- **Descriptive:**
  - learned vs random per variant;
  - `finetune` vs `scratch` and vs `frozen_learned_C`;
  - per-arm medians and ranges of steps and cumulative LR to threshold, and of final unseen and seen accuracy.
- **Run failed:** checks fail, CUDA is unavailable, or no unit finished pretraining.

## Time and cost

- **Setup:** about 5 minutes; pilot: 4 runs × 4,000 updates (one per LR).
- **Training:** 9 units × (2 × 5,000 pretraining + 9 × 10,000 phase-2 updates) = 900,000 updates.
  - Units run 6 at a time. The longest unit is 100,000 sequential updates, about 5–10 minutes on the GPU at an assumed 2–5 ms/update (not measured; the coordinator's CPU does 5.6 ms at r = 16, one thread).
  - The oracle arm parses each batch on the CPU and may be slower.
  - The training deadline is 3 h after the pilot.
- **Total:** at most about 0.2 h + 3 h + 0.1 h ≈ 3.3 h, under the 4 h cap.
- **Cost:** at up to $1.09/h (L40S, secure), the 4 h cap costs at most $4.36, inside the $4.5 budget.

## If something fails

- **Checks fail or CUDA is unavailable:** stop, finish the worker, mark failed.
- **A unit crashes:** the others continue. Rerun it once with `--only g<g>_s<s> --hours <left - 0.75>` into a separate `--out runs/rhm-transfer-rerun` (first `mkdir -p runs/rhm-transfer-rerun && cp runs/rhm-transfer/lr.json runs/rhm-transfer-rerun/`, so the rerun uses the original LRs), so the original results are kept, and note an amendment.
- **Several units fail the gate:** report them, and don't lower the gate on the worker. If fewer than 5 pass, the result is inconclusive by the pre-set rule.
- **The first wave's ms/update projects past the deadline:** let it run. Arms are ordered so `scratch` and `finetune` finish first, and truncated arms are dropped from pairs and counted. If the worker cap leaves time, rerun the missing units with `--only` into a separate `--out` (copying `lr.json` there first). `analyze.py` reads one runs folder: analyze the rerun folder on its own and report it as an amendment next to the main verdict.
- **None of the GPU types in stock:** the launcher waits and retries; if none is found, report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Design?** The user specified:
  - the root task, main grammar, 3 grammars, r = 16;
  - a seeded 48/16 split of the 64 root productions, keeping every level-1 symbol in the seen productions and every unseen class defined;
  - pretraining until seen accuracy is high;
  - arms: (a) frozen learned subtrees with a fresh root core, and with fresh top fusion cores too; (b) the same with random frozen subtrees; (c) scratch; (d) fine-tune everything;
  - updates to threshold on unseen productions and final accuracy, seed-paired, 3 init seeds per grammar if time allows.
- **Choices made here:**
  - one unseen production per class;
  - the side-wise symbol coverage condition;
  - a 5,000-update pretraining with a 0.95 gate;
  - threshold 0.9.
- **After review:**
  - the other-grammar floor and the oracle ceiling;
  - normalization through the last frozen core;
  - variance-init fresh cores;
  - scratch and finetune first;
  - the scope statement about side-symbol reuse.
- **Initialization and warmup?** The user chose the identity init (eps 0.1) as the main init for all RHM experiments, after the CPU probes showed the variance init collapsing, and a 100-update warmup for every RHM run.

## Review

An independent review returned "needs revision". Findings and changes:

- **No arm could lose to `frozen_learned`; random frozen subtrees only test against a bag-of-tokens map.**
  - Added `frozen_other`, frozen subtrees pretrained on a different grammar, as a trained floor.
  - Added an oracle ceiling: C on a code of the true level-1 symbols (one-hot at first; made dense after the re-check, below).
  - H1 is now: beats the floor and scratch, and is within 2× the oracle. Learned vs random is reported as descriptive, with what it can and can't show.
- **The fresh-core rescaling gave the learned arms about 48 times the relative step.** The last frozen core is now scaled instead (W[−1] for C arms, W[−2] for WC arms), and trainable cores keep their init.
- **WC fresh cores used the identity init while C used the variance init.** Both fresh cores now use the variance init.
- **Late arms were the ones truncated, and truncated arms could enter pairs.** `scratch` and `finetune` now run first. Only finished arms enter pairs, and dropped pairs are counted.
- **Pretraining was longer than needed.** Seen accuracy is 1.0 by about 1,500 updates, so pretraining is now 5,000 updates with the 0.95 gate. Phase 2 is 10,000 updates.
- **The scope was overstated.** Unseen productions are distinct pairs of seen side symbols, so phase 2 tests reuse of side-symbol codes, not conjunctive recombination. Now stated.
- **Shared fixes.**
  - Margins on steps to threshold and per-grammar consistency.
  - The rule that scratch must learn.
  - The LR grid now includes 1e-4, the pilot runs on seed 9, and grid-edge choices are recorded.
  - `--lr` overrides exist and reruns go to a separate `--out`.
  - The deadline starts after the pilot, and ms/update is tabled per arm.
  - The cap and budget assume secure prices up to $1.09/h.

- **Re-check (second pass).** Three of four P1s were resolved; one remained: in a CPU probe all three H1 conditions held by construction. The other-grammar floor never learned (unseen accuracy 0.12), and the one-hot oracle learned slower than the learned arm (161 vs 87 updates to 0.9) because under Adam a one-hot code moves C ~r times slower than dense bonds. Changes: the oracle code is now dense (sqrt(r) Q e_a, Q a fixed seeded orthogonal matrix; same norm); H1 is decided by scratch and the oracle factor, and failing either is "not supported"; the floor is descriptive, stated in advance as expected to be censored. P2s: the pilot count says 4; reruns copy lr.json and are analyzed separately.

## Amendments

None.

## Outcome

Not run yet.
