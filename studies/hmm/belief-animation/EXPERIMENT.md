---
name: belief-animation
title: Belief geometry over training: dense checkpoints for an animation (Mess3, RRXOR, RHM)
status: approved
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX PRO 4500 Blackwell
count: 1
min_gpu_gb: 30
hours: 5
budget_usd: 6
disk_gb: 60
data:
secrets:
approved_utc: 2026-11-18T16:41:28Z
---

# Belief geometry over training: dense checkpoints for an animation (Mess3, RRXOR, RHM)

## Question

How does the decoded belief geometry form during training, from the initial network to the trained one, in every
architecture we have belief-geometry results for: on Mess3, on RRXOR and on a Random Hierarchy Model grammar? The
earlier experiments (`hmm-belief-geometry`, `rhm-belief-geometry`, `rhm-ttn-belief-geometry`,
`rhm-belief-architectures`) kept decoded beliefs only for the first and the last checkpoint, so the static figures
show two frames. This run repeats their seed-0 training with about 120 checkpoints each and saves the decoded beliefs
at every one, as the frames of an animation for presentations.

It is a visualization of existing results. It tests no new hypothesis and changes no verdict.

## Hypothesis

None is tested. The expectation, which the comparison below checks, is that each rerun reproduces its original run:
the same script, seed, data stream and settings, so the same trajectory up to GPU rounding. A rerun whose final loss
or final probe R² differs from the original beyond the tolerance under "Success and failure criteria" is reported as
not reproducing it, and its panel is described as a new run.

## Method

**Runs (13, seed 0 only).** Each uses the original experiment's training script and default settings, with one added
flag, `--ckpts`, that replaces the six saved checkpoints by a denser list (the default list is unchanged without it).

| video | run name | script | architecture | updates |
|---|---|---|---|---|
| Mess3, RRXOR | `<p>_transformer_s0` | `hmm_run.py <p> transformer` | 4 layers, d = 128, 4 heads | 30,000 |
| | `<p>_chain_s0` | `hmm_run.py <p> chain` | ChainTN, r = 32 | 30,000 |
| | `<p>_tree_s0` | `hmm_run.py <p> tree` | TreeTN, 16 leaves, r = 32 | 30,000 |
| | `<p>_tree8_s0` | `hmm_run.py <p> tree8` | TreeTN, 16 leaves, r = 8 | 30,000 |
| RHM v8m2s2L4 | `v8m2s2L4_s0` | `rhm-belief-geometry/run.py` | transformer, as above | 50,000 |
| | `v8m2s2L4_tree_s0` | `arch_run.py --arm tree` | aligned TreeTN, r = 32 | 30,000 |
| | `v8m2s2L4_perm_s0` | `arch_run.py --arm perm` | TreeTN on permuted leaves, r = 32 | 30,000 |
| | `v8m2s2L4_chain_s0` | `arch_run.py --arm chain` | ChainTN, parameter-matched width | 30,000 |
| | `v8m2s2L4_r16_s0` | `ttn_run.py --r 16` | aligned TreeTN, r = 16 | 30,000 |

Batch 512 fresh sequences per update, Adam, learning rate 1e-3 with each script's own warmup and schedule, T = 16
tokens, as in the originals.

**The RHM grammar is v8m2s2L4** (8 symbols, 2 productions, 4 levels, 16 tokens). It is the one grammar that all four
RHM belief experiments trained and that every architecture learned (final excess loss 0.0001 to 0.0016 nats/token in
the originals). The deck's standard grammar, v16m4s2L4, was not learned by the chain or the permuted tree and has
probe R² of 0.5 to 0.7, so it would show mostly failure.

**Checkpoints.** Update 0, then a log-spaced grid 10 · q^k with q = 3000^(1/119), rounded and deduplicated, up to the
last update, which is always included: 120 checkpoints for 30,000 updates and 128 for 50,000
(`anim_frames.py --steps N` prints them). The test loss is logged at every checkpoint. Logging and saving do not touch
the data generator, so the training trajectory is that of the original run.

**Frames (`anim_frames.py`).** At every checkpoint the probe that the static figures draw is refit from scratch and
its prediction on held-out sequences is saved. Analysis set, split and probes are the originals': 60,000 sequences
with the original analysis seed, the first half to fit and the second half to test (the sequence split of the
figures; the write-ups' criteria use the stricter prefix split, which this animation does not recompute).
- HMM (as `hmm_plot.stacked`): affine probe on the transformer's last layer; projective probe on the chain state;
  projective probe on the tree's top bonds [h_L, h_R] for t ≤ 8 and on [h_L, h_R, h_L ⊗ h_R] for t > 8.
- RHM: affine probe on the transformer's last layer, on the chain state, and on [h_A, h_B, h_A ⊗ h_B] for the trees
  (the "bilinear" probe of `ttn_analyze.py` and `arch_analyze.py`). The belief has up to 4,096 coordinates but at
  most a few dozen distinct values per position, so the probe is fit to its coordinates on an orthonormal basis of
  the span of the centered beliefs, which gives the same fit and R² (derivation in the script's docstring).
- Saved per run: the true beliefs and every checkpoint's predictions for the first 1,000 test sequences at every
  position (float16), and SSE and SST over the whole test half per checkpoint and position.

**Comparison and rendering, on the coordinator after the worker is finished.** `anim_compare.py` compares each rerun
with its original; `anim_render.py` writes one H.264 MP4 per process (1920 × 1080, 12 frames per second, one frame per
checkpoint) and their concatenation.

## Inputs

No datasets or weights. Sequences are sampled from the processes in `studies/hmm/processes.py` and from the RHM
grammar `v8m2s2L4` with grammar seed 0 (`studies/rhm/grammar.py`).

## Commands

From the project root on the worker.

Step 1, checks:

```bash
set -euo pipefail
nvidia-smi
free -g
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
mkdir -p "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python -m studies.hmm.checks "$EXP_DIR/results/checks_hmm.json"
PYTHONPATH=. OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks_rhm.json"
```

Step 2, timing: the slowest kind of run (the r = 32 tree on RRXOR) for 300 updates with its checkpoint list, then
the frames of 6 of its checkpoints at full analysis size:

```bash
set -euo pipefail
mkdir -p /tmp/belief-animation/timing
S=$(PYTHONPATH=. uv run --frozen python studies/hmm/belief-animation/anim_frames.py --steps 300)
PYTHONPATH=. uv run --frozen python studies/hmm/hmm-belief-geometry/hmm_run.py rrxor tree --steps 300 --ckpts "$S" --out /tmp/belief-animation/timing | tail -3
time PYTHONPATH=. uv run --frozen python studies/hmm/belief-animation/anim_frames.py /tmp/belief-animation/timing/rrxor_tree_s0 /tmp/belief-animation/timing/frames --every 12
```

Read off the seconds per checkpoint of the frame step: (wall time − about 20 s of setup) / 6. If it is above 45 s,
run step 3 with `EVERY_TREE32=2`; above 90 s, with `EVERY_TREE32=3` (see "Time and cost").

Step 3, all 13 runs, four at a time, each trained and then turned into frames (`anim_all.sh`, `anim_job.sh`):

```bash
set -euo pipefail
EVERY_TREE32=1 JOBS=4 sh studies/hmm/belief-animation/anim_all.sh
ls -la runs/belief-animation/frames | tee "$EXP_DIR/results/frames_ls.txt"
grep -h "ms_per_update" runs/belief-animation/frames/*.json | sort | uniq -c | head -20
```

Each line it prints is `DONE <run>`, `FAILED train <run>`, `FAILED frames <run>` or `SKIPPED <run>`. A second call
repeats only what is missing: training is skipped where `result.json` exists.

Then `orch finish`. On the coordinator:

```bash
PYTHONPATH=. uv run --frozen python studies/hmm/belief-animation/anim_compare.py runs/belief-animation/frames studies/hmm/belief-animation/results/compare.json | tee studies/hmm/belief-animation/results/compare.txt
PYTHONPATH=. uv run --frozen python studies/hmm/belief-animation/anim_render.py runs/belief-animation/frames studies/hmm/belief-animation/figures
```

## Expected outputs

- `runs/belief-animation/frames/<run>.npz` and `<run>.json` for the 13 runs: about 8 MB per Mess3 run, 13 MB per
  RRXOR run, 8 MB per RHM run, about 120 MB in all. Returned as data (not in git).
- `runs/belief-animation/logs/train_<run>.log`, `frames_<run>.log`.
- `studies/hmm/belief-animation/results/checks_hmm.json`, `checks_rhm.json`, `frames_ls.txt`, and after the worker
  `compare.json`, `compare.txt`.
- `studies/hmm/belief-animation/figures/mess3.mp4`, `rrxor.mp4`, `rhm.mp4`, `all.mp4` (H.264, a few MB each).
- `README.md`: what the videos show and do not show, and the comparison table.
- Model checkpoints stay on the worker's `/tmp` (about 4 GB) and are not published.

## Controls

- Each panel's first frame is the initial network under the same probe, which is the control the static figures
  draw. On Mess3 the initial networks already score R² 0.87 to 0.99, and the animation shows that.
- The comparison of every rerun with its original run (final excess loss; per-position R² of the drawn probe at the
  first and last checkpoint against the original `analysis.json`).
- No new scientific control is added. The originals' controls (token readout, next-token ceiling, scrambled targets,
  prefix split) are not recomputed per frame; the write-up points to them.

## Success and failure criteria

- **The run succeeded** if all 13 runs have a frames file covering every one of their checkpoints (every k-th for the
  two r = 32 HMM trees if the timing step set `EVERY_TREE32=k`), and the three videos render.
- **A rerun matches its original** if its final excess loss is within max(0.002 nats, 50 %) of the original's and
  its pooled R² at the last checkpoint is within 0.02 of the original's. A rerun that does not match keeps its panel,
  and the write-up says for which arm the animation is a new run and not the published one.
- **A video is incomplete** if one of its arms has no frames. It is then rendered without that arm only if the
  missing arm cannot be recovered within the budget, and the write-up says so.
- **The run failed** if the checks fail, or if fewer than all arms of two of the three videos have frames.

## Time and cost

- Measured in the originals (four runs sharing one RTX A6000 or similar): about 20 minutes per HMM transformer run,
  14 to 17 per chain and r = 8 tree, 28 to 30 per r = 32 tree; 17 minutes per r = 32 RHM tree (34 ms per update).
  The RHM transformer has 50,000 updates, estimated at 30 minutes. The 120 extra test-loss evaluations and saves add
  about 2 minutes per run. Total about 270 run-minutes, so about 70 minutes of training at four jobs.
- Frames: affine fits take a few seconds per checkpoint, so about 10 minutes per run. The projective fits on the
  r = 32 tree's 1,088 product features (7 positions per checkpoint, two L-BFGS starts of up to 300 iterations each)
  are not measured on a GPU; estimated 30 s per checkpoint, 60 minutes per run, two such runs in parallel. That is
  what the timing step measures. If it is above 45 s per checkpoint the two r = 32 HMM trees keep every second
  checkpoint (60 frames), above 90 s every third (40 frames); nothing else is dropped.
- Expected total about 2.5 hours including 15 minutes of setup; the cap of 5 hours leaves a factor of 2. At $0.59
  (A40) to $1.09 (L40S) per hour that is $1.5 to $2.7 expected and at most $5.45 at the cap.
- GPU memory: the largest process is the frame step of an r = 32 tree (30,000 × 1,088 float64 features and their
  Gram matrices, under 3 GB) or of an RHM run (one 60,000 × 4,096 float64 belief table at a time, 2 GB, 4 to 6 GB
  with temporaries); four processes fit in 30 GB. `min_gpu_gb: 30`.
- Threads: `OMP_NUM_THREADS=2` per job, four jobs. Sampling and the models run on the GPU.

## If something fails

- **A run fails or diverges:** rerun `anim_all.sh` once (it repeats only what is missing). If the same run fails
  again, leave it out and report it; do not change its settings.
- **Out of GPU memory:** rerun with `JOBS=2`.
- **Time:** at 4 hours, `touch runs/belief-animation/STOP` so that no new run starts, let running jobs end, and
  finish. Frames files are rewritten every 10 checkpoints, so a stopped frame step keeps all but the last 10 at most;
  a stopped training loses that run's frames (its checkpoints are on `/tmp`, which is not published).
- **Checks fail:** stop; that is not a result.
- All of the above may be decided without asking. Changing a model, a seed, a probe or the grammar must wait for the
  user.

## Fixed rules

- Seed 0, the original scripts and their default settings; no tuning.
- The probes are those the static figures draw, refit at every checkpoint; frames are never interpolated.
- The matching tolerance above.

## Questions asked before launch

None asked. The user's request fixed the scope: "create the animation for the belief geometry, moving from
initialization to learning the geometry ... for mess3, rrxor, rhm, across all the architectures we have data for. if
you need more data/frames, rerun! MP4 with H.264". Choices made without asking, to be reported with the result: the
RHM grammar (v8m2s2L4, reason under Method), seed 0, about 120 log-spaced checkpoints, and the budget ($6).

## Review

skipped: a rerun of four approved and completed designs at seed 0 with denser checkpoints, for visualization; no
hypothesis, criterion or comparison is new.

## Approval

The user's standing go-ahead of 2026-11-18 ("if you need more data/frames, rerun!") covers this run.

## Amendments

- **2026-11-18 16:52 UTC, before step 3: the two r = 32 HMM trees keep every checkpoint (`EVERY_TREE32=1`).** The
  timing step measured 299 s for 6 checkpoints of the r = 32 RRXOR tree's frames, about 47 s per checkpoint after
  setup, just above the 45 s at which "Time and cost" said to keep every second checkpoint. That rule was there to
  protect the time cap, and the same step showed the cap is not at risk: training took 9 s for 300 updates with 59
  test evaluations (alone on the A40), far below the estimate, and the frame step is bound by one CPU thread per
  process, so the two trees do not slow each other. At 47 s per checkpoint, 120 checkpoints take about 95 minutes per
  tree, both in parallel, so the whole run should end about 2 h 15 min after launch, against a cap of 5 h, at about
  $1.4 of the $6 budget. Halving the trees' frames would make two of the four HMM panels move at half the frame rate
  of the others for no saving that matters. The stop rule at 4 hours is unchanged.

<!-- Each change made during the run, with the time and the reason, recorded and committed before acting on it. -->

## Outcome

Not run yet. Afterwards: one line per criterion with what was observed, the cost and runtime against the budget, and a link to the write-up. The write-up holds the detail; don't repeat it here.
