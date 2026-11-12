---
name: lm-bytes
title: "Byte-level language modelling on TinyStories: TTN against n-grams, MLP, chain and transformer"
status: done
gpu: NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA L40S
count: 1
hours: 10
budget_usd: 10
disk_gb: 60
data: data__tinystories
secrets:
approved_utc: 2026-11-12T21:42:22Z
result: supported
finished_utc: 2026-11-12T23:46:01Z
writeup: studies/lm/lm-bytes/README.md
---

# Byte-level language modelling on TinyStories: TTN against n-grams, MLP, chain and transformer

## Question

Does a balanced tree tensor network learn useful features on real text? The task is to predict one byte of TinyStories from the 16 bytes before it (V = 256). The mentor proposed this as the first test of the TTN outside synthetic teachers.

"Useful" is measured against references that each rule out one explanation:

- byte n-grams up to order 8, to ask whether the TTN does more than local counting;
- an MLP with matched parameters, for a generic learned model of the same size;
- a shared-core chain (MPS) with matched parameters, to ask whether recurrent weight sharing explains the result;
- a level-tied TTN, for the cost of tying;
- a small causal transformer, for what a standard architecture reaches with the same targets.

## Hypothesis

Every model sees the same 10.24M training targets (40k updates of 256 windows). The headline number is test bits/byte.

- **H1, expected.** The untied r = 32 TTN beats every Kneser–Ney n-gram of order ≤ 4 (context ≤ 3 bytes), counted on the full 100 MB train set. Against: the TTN median is above the best order-≤4 value.
- **Open, no directional prediction.** Each of these is decided by the rule under "Success and failure criteria":
  - TTN vs KN order ≤ 8 (100 MB counts, and 10.24M bytes, the neural target count). TinyStories is repetitive, so order 8 is a strong baseline: it reaches 1.21 bits/byte from only the 10 MB pilot set.
  - TTN vs the MLP at matched parameters (852k vs 836k).
  - TTN vs the matched chain (r = 93, 852k). If the chain is as good, the tree's hierarchy is not what matters.
  - Untied vs level-tied TTN.
  - TTN vs the transformer (861k). My prior is that the transformer wins.
  - **TTN vs the frozen-fusion TTN** (identity init, fusion cores W frozen at init, only the leaves A and the root C trained; 3 seeds). Identity init starts the TTN near its additive part, a sum of per-position linear responses read out through C. Freezing W keeps the network close to that starting function class, so this asks what training the fusion cores adds. A CPU check (r = 16, LR 1e-3, 2k updates, pilot set) gave 2.72 bits/byte frozen against 2.22 fully trained. If the full TTN does not beat the frozen one, its features are mostly the init's additive structure.
- **Init, a finding from CPU pilots that changed the design.** The mentor proposed the variance-preserving init in `ctn/tree.py`, but under Adam it stalls. The TTN has no biases or nonlinearities, so every logit is a degree-16 multilinear function of the leaf embeddings, and even the byte marginal is a degree-16 feature.
  - After 20 Adam steps, the activation RMS falls from 1.0 to 0.49 at the top level and 0.28 at the logits: the product shrinks toward the zero, uniform-output saddle.
  - Neither a smaller root core nor a higher LR fixes it (table below).
  - Every TTN and chain run therefore uses `ctn.tree.identity_init(model, eps = 0.1)`, which the RHM study introduced for the same collapse. Bond channel 0 carries a constant, every fusion core starts near the pass-through tensor plus eps × its variance-init noise, and the root (chain: head) keeps the variance init. The network stays multilinear and bias-free, but it starts near its additive part instead of a product of 16 random factors.
  - CPU pilots on the 10 MB pilot set, with validation on 2^12 positions, clipping at 1.0 and 1k warmup (uniform is 8.0, the KN unigram 4.42):

  | model, init | peak LR, updates | validation bits/byte at 250 / 500 / 1,000 / 2,000 updates |
  |---|---|---|
  | TTN r = 16, variance | 1e-3, 2,000 | 7.99 / 7.93 / 7.84 / 7.65 |
  | TTN r = 16, variance, full 5k pilot schedule | 1e-3, 5,000 | 7.99 / 7.93 / 7.84 / 7.65; 7.21 at 5,000 |
  | TTN r = 16, variance, root core × 0.01 | 1e-3, 2,000 | 7.93 / 7.86 / 7.80 / 7.54 |
  | TTN r = 16, variance | 1e-2, 2,000 | 7.98 / 8.04 / 8.06 / 7.97 |
  | TTN r = 32, variance | 1e-3 and 3e-4, 800 | 7.64 and 7.69 at 800 |
  | **TTN r = 16, identity** | 1e-3, 2,000 | **4.14 / 3.03 / 2.53 / 2.22** |
  | **TTN r = 16, identity** | 3e-4, 2,000 | **7.00 / 3.91 / 2.94 / 2.62** |
  | chain r = 32, variance | 1e-3, 1,000 | 7.94 / 7.87 / 7.75 at 1,000 |
  | **chain r = 32, identity** | 1e-3, 1,000 | **3.77 / 3.12 / 2.62 at 1,000** |
  | TTN r = 16, identity, fusion cores frozen | 1e-3, 2,000 | 4.88 / 3.55 / 2.91 / 2.72 |

  A variance-init arm (untied r = 32, 2 seeds, 40k updates, the identity TTN's piloted LR) documents the behaviour on the GPU at full length. The expectation is "slow or stalled": the CPU pilots only cover 5k updates and the curve was still falling. `analyze.py` reports, for every run, the first curve step at which validation drops below the KN unigram, which tells a slow start (a late crossing) from a stall (none). It is the comparison "identity vs variance init"; everything else uses identity init.
- **Starting position biases two comparisons.** Under identity init, the level-tied TTN and the chain start position-blind. Their leaf embeddings are shared across positions, and the pass-through cores are symmetric in their two children, so to first order in eps the starting network responds to a byte the same way wherever it sits in the window. The untied TTN starts with a separate embedding per position. "Untied vs tied" and "TTN vs chain" therefore partly measure distance from the starting point, not only what each architecture can represent. The write-up must say so and give the early curve points (validation at 500, 1k, 2k, 5k and 10k updates, from `analyze.py`) next to the final numbers.
- **Remaining risk.** The identity-init TTN could still plateau above the n-grams. `analyze.py` reports `plateau.left_plateau` (the TTN median must beat the KN unigram) and qualifies the convergence verdict with it.

## Method

- **Data** (`studies/lm/data.py`, already prepared, see Inputs). UTF-8 bytes of TinyStories stories. Every split file is `0x00 s_1 0x00 s_2 ... 0x00`. 0x00 never occurs in the text, so story boundaries are visible in the context, and predicting the end of a story (0x00) is part of the task.
  - A window is the 16 bytes before a target position t, for every t ≥ 16 of a split. Windows may cross a story boundary.
  - Training windows are sampled uniformly with replacement.
- **Models** (`studies/lm/models.py`, `ctn/tree.py` used as it is):
  - untied TTN, `TreeTN(16, 256, r, 256)`, r ∈ {16, 32, 64, 128}: 188,416 / 851,968 / 4,980,736 / 34,078,720 parameters;
  - level-tied TTN, `tie='level'`, r = 32: 368,640;
  - chain `ChainTN(16, 256, r, 256)`, r = 93 (852,066, matched to the r = 32 TTN) and r = 32 (49,184);
  - MLP on the flattened 16 × 256 one-hot with 192 ReLU units: 836,032 (nn.Linear default init);
  - pre-LN causal transformer, d = 128, 4 layers, 4 heads, MLP width 512, learned positions, read out at the last position only, so it gets one target per window like every other model: 861,184 (GPT-2 init).
  - **Init.** Every TTN and chain uses `identity_init(model, eps = 0.1)` on top of the classes' variance-preserving init, except the variance-init arm. The spec fields `init` and `eps` are recorded in every run's config; see Hypothesis for why.
- **Training** (`studies/lm/train.py`), identical for every neural run:
  - batch of 256 windows; Adam (0.9, 0.999, eps 1e-8), no weight decay;
  - LR rises linearly from 0 to the peak over 1,000 updates, then follows a cosine down to 0.1× peak at the last update;
  - 40,000 updates (10.24M targets) for the main runs;
  - gradient-norm clipping at 1.0, applied to every family so the schedule is identical. The TTN needed it in the CPU pilots (run with the variance init): without clipping, the r = 32 TTN at LR 1e-3 reached gradient norms of 1e7 to 5e8 and 5,700 bits/byte by update 300, and at 3e-3 it went to 1e9 bits/byte. With clipping, both LRs trained stably for 800 updates. The clip value and the fraction of clipped steps are recorded per run;
  - fp32 with TF32 matmuls; the batch stream is seeded by seed + 10,000, so equal seeds see the same batches in every family.
- **LR pilots.** Each family runs at peak LR 3e-4 and 1e-3: 5,000 updates (1,000 warmup, cosine over the rest) on the 10 MB pilot set, seed 0. The LR with the lower final validation bits/byte on the 2^17 fixed curve positions is used for all of that family's runs, recorded in `lr_choice.json`.
  - The untied-TTN pilot (identity init) runs at r = 32, and its choice also serves r = 16, 64, 128, the long run and the variance-init arm.
  - The matched-chain pilot (r = 93) also serves the small chain.
- **Runs** (40k updates on the 100 MB train set unless stated), in this priority order:
  1. LR pilots (10 runs).
  2. Main comparison, seed-major (seed 0 of all five, then seed 1, then seed 2): untied TTN r = 32, MLP, matched chain, transformer, and the frozen-fusion TTN r = 32 (`ttn_frozenW_r32`: identity init, `requires_grad=False` on every W, only A and C trained, the TTN's piloted LR; 393,216 of 851,968 parameters train and the 458,752 fusion-core entries stay frozen). The n-grams run alongside on the CPU.
  3. The rest: untied TTN r = 32 with the variance init (seeds 0–1), level-tied TTN r = 32 (seeds 0–2), untied TTN r = 16 (0–2), r = 64 (0–1), chain r = 32 (0–2), untied TTN r = 128 (0), and one untied r = 32 run of 160,000 updates (seed 0, its own 160k cosine schedule).
- **Concurrency.** GPU runs go through a pool of 3 processes that take runs in the order above. A run is skipped, and recorded as `skipped` with its partial curve, if its projected finish, including the final evaluation, falls after the training deadline. A run with a non-finite loss stops and is recorded as `diverged`.
- **Checkpoints.** `state_dict` plus config of the identity-init untied r = 32 seed-0 main run at updates 0, 100, 300, 1,000, 3,000, 10,000, 20,000 and 40,000, for later interpretation.
- **n-grams** (`studies/lm/ngram.py`). Interpolated modified Kneser–Ney (Chen & Goodman), orders 1–8; order m conditions on m − 1 bytes.
  - Lower orders use continuation counts. The three discounts per order come from the closed-form count-of-counts estimate, not tuned on validation.
  - Counted on all of `train.bin` (100M targets) and, as a target-matched control, on its first 10.24M bytes.
  - Counting is vectorised over uint64-packed keys (100 MB in well under a minute on a CPU, a few GB of RAM).
- **Evaluation.** Bits/byte = cross-entropy / ln 2.
  - Final validation and test numbers use every target of `val.bin` (5,001,280) and `test.bin` (5,000,223), for neural models and n-grams alike.
  - Curves: every 500 updates, validation bits/byte on 2^17 fixed validation positions (seed 1) and the mean training loss since the last point.
  - Train bits/byte at the end on 2^17 fixed train positions, to show the generalisation gap.

## Inputs

- **Data tar `data__tinystories`** = `data/tinystories/`, prepared on the coordinator by `python -m studies.lm.data prepare`. It is unpacked at the repo root, so the run reads the same paths.
- **Source.** `TinyStoriesV2-GPT4-train.txt` from `roneneldan/TinyStories` on Hugging Face. Bytes 0–399,999,999 were fetched by HTTP range request (raw sha256 `a375a54b…5e8782`); the partial story after the last `<|endoftext|>` (118 bytes) and 23 empty stories were dropped. That leaves 488,132 stories.
- **Split.** Stories were shuffled with seed 0 and dealt whole to test, validation and train until each reached its size. The splits are disjoint by story; 351,395 stories of the prefix are unused.

| file | bytes | stories | sha256 |
|---|---|---|---|
| train.bin | 100,000,098 | 124,382 | 9a33f85e…c42a848 |
| val.bin | 5,001,296 | 6,195 | 31b22e4e…fb448ee |
| test.bin | 5,000,239 | 6,160 | f13c27b9…3c8e6 |
| pilot.bin (prefix of train.bin up to the last separator before 10^7) | 9,999,905 | 12,435 | 8015e019…393b546 |

Full hashes are in `data/tinystories/meta.json`. `studies.lm.checks` verifies all four files against it before training.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
mkdir -p "$EXP_DIR/results"
OMP_NUM_THREADS=4 uv run --frozen python -m studies.lm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/lm/lm-bytes/run.py --out "$EXP_DIR/results" --hours 8 --procs 3 --ckpt artifacts/lm-bytes/ttn_r32_s0
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lm/lm-bytes/analyze.py "$EXP_DIR/results"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lm/lm-bytes/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

- The checks exit non-zero on any failure and take about 15 s.
- Run the training as its own `orch exec` step from a background command.
- `OMP_NUM_THREADS` caps CPU threads: a sibling experiment stalled when torch started hundreds of threads under a 27-CPU cgroup quota. Each GPU run also calls `torch.set_num_threads(2)`.

## Expected outputs

- `results/checks.json`.
- `results/runs/<name>.json`, one per run (pilots `pilot_<family>_lr<lr>`, `ttn_r32_s0`, `ngram_100M`, ...), tens of kB each. Each holds the config (model, seed, LR, clip, schedule), status, params, targets seen, final validation, test and train bits/byte, the curve (every 500 updates: train and validation bits/byte, LR, clipped fraction, max gradient norm), wall time and ms/step.
- `results/lr_choice.json`, `results/plan.json`, `results/status.json`, `results/analysis.json`.
- `results/source/`, a copy of the code.
- `artifacts/lm-bytes/ttn_r32_s0/step_*.pt`: 8 checkpoints, about 3.4 MB each.
- `figures/curves.pdf` and `figures/params.pdf`.
- `README.md`: the write-up, with the table printed by `analyze.py`.

## Controls

- Every neural model gets the same data, batch stream (per seed), schedule, clipping and number of targets. Only the architecture and the piloted peak LR differ.
- The uniform model (8 bits/byte, checked exactly), the n-gram ladder (orders 1–8) and the target-matched n-gram (10.24M bytes) are the floors for "more than local counting".
- Matched-parameter controls for the r = 32 TTN: the MLP (836k), the chain (852k) and the transformer (861k).
- The tied TTN controls for weight sharing within the tree; the r sweep and the 160k run control for capacity and convergence.
- The variance-init arm (r = 32, 2 seeds) differs from the main TTN only in its init: same LR, data, batches and schedule.
- The frozen-fusion TTN (r = 32, 3 seeds) differs from the main TTN only in that W does not train. It is the floor for what the identity init's starting function class reaches with trained leaves and root.
- Pre-training checks (`studies.lm.checks`):
  - data hashes and window extraction;
  - parameter counts against their formulas;
  - the LR schedule's endpoints;
  - TTN per-level activation RMS and per-group gradient norms at init and after 20 Adam steps, for both inits, which must be finite and within loose bounds;
  - full-size identity-init networks (TTN r = 32 untied and tied, chain r = 93) give finite activations at every level, and channel 0 carries the constant;
  - every family overfits 128 random (context, byte) pairs, with the tree and chain checked under both inits;
  - the vectorised n-gram counts equal a naive count, and KN distributions sum to 1 at every order.

## Success and failure criteria

Computed by `analyze.py` on test bits/byte over the seeds that finished. MARGIN = 0.02 bits/byte. A vs B is:

- "A better" if every A seed is below every B seed and the medians differ by ≥ MARGIN;
- "equal" if the medians differ by < MARGIN and neither group's seed range is wider than MARGIN;
- "unresolved" otherwise.

MARGIN is a practical threshold fixed before the run, about 1.5% of the roughly 1.3 bits/byte the strong models should reach. Seed spread is unknown in advance, which is why "equal" also needs both seed ranges within MARGIN.

An n-gram is a single value.

- **H1 supported:** all finished untied r = 32 TTN seeds (at least 2 of 3) are below the best KN order ≤ 4 on 100 MB. **Not supported:** the TTN median is above it. **Partly supported:** in between.
- **Open comparisons:** a verdict each for TTN vs best KN ≤ 8 (100 MB and 10.24M), vs MLP, vs matched chain, untied vs tied, vs transformer, vs the frozen-fusion TTN, and identity vs variance init (expected: identity better). Also reported: the highest n-gram order the TTN median beats.
- **Convergence:** the 160k run (seed 0) against the range of the three 40k seeds. "Not converged at 40k" if it beats the best 40k seed by ≥ MARGIN; otherwise "converged at 40k". If the TTN has not left the plateau, the verdict carries that qualifier.
- **Inconclusive:** fewer than 2 untied r = 32 TTN seeds finished, or the 100 MB n-gram didn't finish.
- **Descriptive:**
  - TTN bits/byte against r;
  - train–validation gaps and the clipped fraction;
  - pilot curves;
  - whether the TTN left the initial plateau;
  - per run, the first curve step with validation below the KN unigram (slow start vs stall);
  - per group, validation at 500, 1k, 2k, 5k and 10k updates (the early points the position-blind caveat needs).
- **Run failed:** checks fail, CUDA is unavailable, or no main-comparison run finished.

Skipped, diverged and failed runs are counted per group in `analysis.json` and never imputed.

## Time and cost

- **Setup:** about 10–15 minutes (uv sync, unpacking the 120 MB data tar).
- **Training: 1.5–3 h expected, capped at 8 h by `--hours 8`.** That is about 1.37M updates in total:
  - 10 pilots × 5k;
  - 29 runs × 40k, including the 2 variance-init and 3 frozen-fusion runs;
  - one run × 160k.
  CPU ms/step at full size are in the table below, but GPU speed is not measured. These models are small, so a GPU step should cost 2–10 ms, dominated by kernel launches; the chain's 16 sequential contractions and the transformer's 4 layers are the slowest. Evaluation adds 1–3 min per run (10M final targets plus 80 curve evaluations of 2^17 positions). The n-grams take a few minutes on the CPU in parallel.
- **Cost.** The worst case is the full 10 h cap at the most expensive listed type (L40S, about $0.86/h, confirm at launch): about $8.60, inside the $10 budget. The expected cost is 2–3.5 h, about $1–3.

CPU ms/step at full size (1 thread on the shared 2-vCPU coordinator; batch 256, forward, backward, clip and Adam; the init does not change the cost). These bound nothing on the GPU, but they rank the per-step cost:

| model | params | CPU ms/step |
|---|---|---|
| TTN r = 16 | 188,416 | 40 |
| TTN r = 32 | 851,968 | 113 |
| TTN tied r = 32 | 368,640 | 111 |
| TTN r = 64 | 4,980,736 | 454 |
| TTN r = 128 | 34,078,720 | 2,888 |
| MLP | 836,032 | 20 |
| chain r = 93 | 852,066 | 1,218 |
| chain r = 32 | 49,184 | 76 |
| transformer | 861,184 | 745 |

## If something fails

- **Checks fail or CUDA unavailable:** stop, finish the worker, mark failed. A failing check is not a result.
Reruns go into `results/rerun`, use `--only` to name the runs, and take their LRs from `--lr` or from a copy of the original `lr_choice.json`. Set `--hours` to the time left minus 0.75 h, and record an amendment for each rerun. All of these may be decided without asking.

- **A TTN group diverges with clipping** (`diverged` in more than one seed): rerun that group once at the other pilot LR, for example for the r = 32 TTN at 3e-4:

  ```bash
  OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/lm/lm-bytes/run.py --out "$EXP_DIR/results/rerun" --hours <left - 0.75> --procs 3 --ckpt artifacts/lm-bytes/ttn_r32_s0_rerun --only ttn_r32_s0,ttn_r32_s1,ttn_r32_s2 --lr ttn=3e-4
  ```

- **A run fails with CUDA out of memory** (most likely r = 128, or concurrency on a 24 GB card): rerun only the failed runs, one at a time:

  ```bash
  mkdir -p "$EXP_DIR/results/rerun" && cp "$EXP_DIR/results/lr_choice.json" "$EXP_DIR/results/rerun/"
  OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/lm/lm-bytes/run.py --out "$EXP_DIR/results/rerun" --hours <left - 0.75> --procs 1 --only <failed names, comma-separated>
  ```

- **The n-gram process fails or runs out of RAM** (it needs a few GB): rerun it alone. No LR is needed:

  ```bash
  OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/lm/lm-bytes/run.py --out "$EXP_DIR/results/rerun" --hours <left - 0.75> --only ngram_10M,ngram_100M
  ```

- **Analysing reruns:** run `analyze.py` and `plot.py` on `results/rerun` as well, and report both folders. Don't mix their run files.
- **Time running out:** later runs are skipped automatically and recorded. Write up what finished, with the skipped list.
- **No listed GPU in stock:** the launcher waits and retries. If none is found, report that and stop.
- **The identity-init TTN stays on a plateau** (test bits/byte above the KN unigram at 40k): this is a result, not a failure. Report it with the curves; any further model or init change waits for the user. The variance-init arm is expected to be slow or stalled; either is the documented finding, not a failure.

## Questions asked before launch

- **Protocol?** The mentor's, scaled up as the user asked: 100 MB train, 40k updates of 256, the r sweep, the 160k run and checkpoints.
- **Clipping?** The brief said to document instability and clip rather than add nonlinearities. Clipping at 1.0 is applied to every family; see Method.
- **Init?** The mentor proposed the variance-preserving init. CPU pilots showed it stalls near 8 bits/byte under Adam without biases (see Hypothesis). The user chose `identity_init` with eps = 0.1 for every TTN and chain run, plus a 2-seed variance-init arm documenting the stall.
- **Hardware and budget?** The user fixed the GPU list, 10 h and $10.

## Review

An independent review of the first identity-init draft said "needs revision". Findings and changes:

- **P0, the checks crashed on a fresh worker.** `checks.py` wrote `checks.json` into `$EXP_DIR/results/`, which doesn't exist there, so it exited 1 after every check passed and nothing would train.
  - It now writes through `ctn.io.save_json`, which creates the folder, and the Commands run `mkdir -p "$EXP_DIR/results"` first.
  - Tested by pointing the checks at a path whose parents don't exist.
- **P1, per-run JSONs were git-ignored** by unanchored `runs/` and `artifacts/` patterns. The coordinator anchored them in `.gitignore`; `git check-ignore` no longer matches `results/runs/*.json`.
- **P1, no control for what training adds beyond the identity init's starting function class.** Added `ttn_frozenW_r32` (fusion cores frozen, only A and C trained, 3 seeds, main block) and the verdict "TTN r=32 vs frozen-fusion TTN".
  - A CPU check reproduced the reviewer's numbers: 2.72 bits/byte frozen against 2.22 fully trained, at 2k updates.
- **P2s:**
  - The variance arm's expectation is now "slow or stalled". Every run reports the first curve step below the KN unigram, which tells a slow start from a stall.
  - Convergence compares the 160k run against the range of the three 40k seeds.
  - "Equal" needs both seed ranges within MARGIN, and the unsupported claim that MARGIN exceeds seed spread is gone.
  - `run.py` gained `--only` and `--lr`. The reruns under "If something fails" are written out in full and were tested under `--smoke`: an LR override, a reused `lr_choice.json`, an n-gram-only run, and a clear error when no LR is available.
  - Under identity init the level-tied TTN and the chain start position-blind. The Hypothesis says so, and `analyze.py` reports early curve points next to the final numbers.

## Amendments

None.

## Outcome

**Supported** (H1). Test bits/byte. TTN means the untied r = 32 TTN: median of 3 seeds, 1.2860 [1.2857, 1.2867]. All 40 runs finished, with none skipped, diverged or failed.

- H1, TTN vs best KN order ≤ 4 (1.815): supported, with all 3 seeds below it.
- TTN vs best KN order ≤ 8: KN better, at 1.158 on 100 MB and 1.215 on 10.24M bytes. The highest order the TTN beats is 6.
- TTN vs MLP (1.619): TTN better.
- TTN vs matched chain (1.213): chain better.
- Untied vs level-tied (1.425): untied better. The tied TTN starts position-blind; early points are in the write-up.
- TTN vs transformer (1.301): equal (gap 0.015, under the 0.02 margin), though all 3 TTN seeds are lower.
- TTN vs frozen-fusion TTN (2.228): TTN better.
- Identity vs variance init (1.366, 2 seeds): identity better. The variance arm started slowly but didn't stall: it first went below the KN unigram at updates 7,000 and 7,500.
- Convergence: not converged at 40k. The 160k run reached 1.211.
- Plateau: left. Every run went below the KN unigram, the identity-init runs by update 500.
- Cost and time: 1.91 h of worker time on one A40 at $0.59/h, about $1.13, against a 10 h cap and a $10 budget. Training took 1.86 h.

Write-up: [README.md](README.md).
