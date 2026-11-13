---
name: lang-parity
title: "A tiny representation that is hard to find: parity of k bits in a TreeTN across init scales and input bias"
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

# A tiny representation that is hard to find: parity of k bits in a TreeTN across init scales and input bias

## Question

Parity separates representation from discovery.

- **Representation is trivial.** With s = ±1 bits, every subtree only needs h_A = ∏_{t ∈ A∩K} s_t, and parents multiply, so a scalar bond suffices. The sampled centered-logit rank is 1 at every cut (`studies/lang/results/rank.json`).
- **Discovery is hard.** With unbiased inputs, every proper subset of the k bits is independent of the target, so a network that starts near an additive function gets no low-order signal toward the answer.

The experiment asks:

1. How does the time to learn grow with the degree k (k = 2, 4, 8, 16, each an aligned subtree)?
2. Does the identity-init noise scale ε, which sets the size of the initial high-order coefficient, change that time, as the escape law from the TTN degree study suggests?
3. Does a small input bias, which gives low-order correlations of size (2·bias)^{k−1}, shorten the plateau?
4. When a run does learn, does the output causally use each subtree's partial parity? And does that causal use appear bottom-up, or all at once at the loss drop?

The motivating result is the degree experiment (`studies/ttn/degree/README.md`). There, the escape time grew with interaction degree under a constant-biased initialization.

## Hypothesis

- **H1 (degree).** At bias 0, for every ε, larger k is not faster: for no adjacent pair (k, 2k) is 2k faster in ≥ 2 of 3 seeds. And k = 2 is faster than k = 16 in ≥ 2 informative seeds, with none the other way. Against: a larger k learning faster, or k = 16 as fast as k = 2.
- **H2 (init scale, directional).** At bias 0 and k ∈ {4, 8}, ε = 0.3 learns faster than ε = 0.01.
  - Mechanism: as ε → 0 the identity init loses its input dependence (at ε = 0 every token embeds as e_0), and the k-th order term grows only through the ε-noise parts of the cores. With a larger initial high-order coefficient, the escape from the uniform-output saddle should come sooner.
  - ε = 1.0 is a separate regime. Its initial logits are in the hundreds (initial loss of order 10³ nats at k = 8, found by the reviewer), so its first phase shrinks huge logits rather than growing a small coefficient. It is reported against 0.01 and 0.1 but doesn't enter H2. The initial loss per ε is reported.
  - Against: no consistent difference, or the opposite.
- **H3 (input bias, directional).** Bias 0.2 learns the parity faster than bias 0 at k ∈ {4, 8}, ε ∈ {0.01, 0.1, 0.3}.
  - Runs with bias are timed on an unbiased diagnostic set. On biased inputs, a degree-k/2 threshold function of the count of ones already passes the 0.9 × Bayes threshold (k = 4: degree 2 gives 0.903; k = 8: degree 4 gives 0.930). So timing on biased data would not require the parity.
  - The t on the biased diagnostic set (`t_diag`) is reported next to it. The unbiased timing is conservative: a bias-trained run must also be right on configurations that are rare under bias (at k = 8, counts 0–3 are about 6% of the mass under bias but about 36% unbiased).
  - Against: no consistent difference, or the opposite.
- **H4 (mechanism, learned runs at bias 0, final checkpoint).** Deciding criterion (c): patching the probe's partial-parity subspace from a donor with the other partial parity flips the prediction to the counterfactual (median "moved" ≥ 0.8 over internal nodes whose span meets the k bits), at least 0.4 more often than random subspaces of the same dimension.
  - The probe accuracy is reported but not a criterion. After whitening, it is often already 1.0 at initialization, and for a multilinear tree that predicts the parity accurately it is implied by accuracy.
  - The additive one-hot probe is a property of the data: a linear classifier gets 3 of 4 XOR points, 0.75. It is reported as a floor, not a criterion.
  - The earlier claim that the probe distinguishes partial computations from a weak global coefficient is dropped, since whitening removes scale.
- **H5 (order of discovery; descriptive, two-sided).** For learned runs with k ≥ 4, the causal gap (sub-patch moved minus random moved, which is scale-sensitive and about 0 at initialization for a binary output) is computed at each checkpoint for each level's nodes holding ≥ 2 of the k bits. The first checkpoint at which the level median reaches 0.5 classifies the run as bottom-up (lower levels earlier), simultaneous (every level and t in one checkpoint interval), other, or not every level causal.
- **Descriptive.** The mod-4 sum (rank 3, so width 16 is ample), and parity with a narrow r = 2 bond.

## Method

- **Process** (`studies/lang/processes.py`, Parity):
  - x_t are iid bits for 16 leaves, with P(x = 1) = 1/2 + bias.
  - y = (Σ_{t<k} x_t) mod 2, with target noise ε_y = 0.05. Bayes accuracy is 0.975.
  - K = the first k leaves, which is an aligned subtree for k = 2, 4, 8 and 16.
  - mod-4 arm: x_t uniform in {0..3}, y = Σ_{t<k} x_t mod 4.
- **Model:** untied TreeTN, r = 16, V_in = O = 2; 58,368 parameters. The r2 arm uses r = 2 (184 parameters).
- **Init:** identity init (`ctn.tree.identity_init`) with noise ε ∈ {0.01, 0.1, 0.3, 1.0}. Each core is ε × (variance-init noise) + the identity pass-through T, and each leaf is e_0 + ε × noise, so at ε = 0 the network would be constant.
- **Grid:**
  - main: k ∈ {2, 4, 8, 16} × ε ∈ {0.01, 0.1, 0.3, 1.0} × bias ∈ {0, 0.2} × seeds 0–2 = 96 runs;
  - mod4: k ∈ {2, 4, 8}, ε = 0.1, bias 0, seeds 0–2 = 9 runs;
  - r2: k ∈ {4, 8}, ε = 0.1, bias 0, seeds 0–2 = 6 runs;
  - 111 runs in all, seed 0 first.
  - Every run with the same (k, bias, seed) sees the same online data stream.
- **Training:** online batches of 256, Adam, 60,000 updates, warmup 100, then a cosine schedule to 0.1 × peak.
- **LR:** 1e-3 peak, fixed in advance for every run, so ε, k and bias are the only factors that differ. There is no pilot. On easy cells every LR converges within the pilot window, so a pilot would choose by post-convergence noise, and every hypothesis here is about escape time. In the reviewer's CPU runs, k = 4 learned at 1e-3 for ε = 0.01, 0.1 and 1.0.
- **Measures:**
  - Diagnostic set of 20,000 examples (for runs with bias, also an unbiased diagnostic set of 20,000), test set of 50,000.
  - Logging every 10 updates up to 2,000, plus log-spaced updates.
  - t = the first logged update with diagnostic accuracy (unbiased set for runs with bias) ≥ 0.9 × Bayes accuracy; +inf if never in 60,000 updates (censored, never dropped).
  - The initial diagnostic loss is recorded.
- **Checkpoints:** about 21 per run: step 0, 14 log-spaced steps, and the shared RHM checkpoint steps scaled to 60,000.
- **Interpretation** (`studies/lang/interpret.py`, every run, every checkpoint):
  - The summary `par` is the partial parity, or partial sum mod q, of S ∩ K, for every span that meets K.
  - Baselines: step 0, shuffled labels, and an additive one-hot probe.
  - Invariance: flip two bits of S ∩ K together (which keeps the partial parity) and redraw S \ K.
  - Patching: subspace vs random vs whole bond.

### Comparison rules

- faster(·) as in `studies/lang/analysis.py`: the logging-interval margin and the factor 1.25, with finite beating inf.
- A pair where both runs are censored is uninformative and excluded from the counts (reported as both_censored). Verdicts need a stated minimum number of informative pairs.
- Medians are taken with censored runs counted as larger than any finite t.

## Inputs

None.

## Commands

```bash
set -euo pipefail
mkdir -p "$EXP_DIR/results" "$EXP_DIR/figures"
nvidia-smi
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python -m studies.lang.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python -m studies.lang.rank "$EXP_DIR/results/rank.json" --configs lang-parity
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-parity/run.py --out runs/lang-parity --hours 6 --procs 10
PYTHONPATH=. uv run --frozen python -m studies.lang.interpret_all runs/lang-parity --procs 6 --hours 1.5 --only _b0_
PYTHONPATH=. uv run --frozen python -m studies.lang.interpret_all runs/lang-parity --procs 6 --hours 0.75
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/lang/lang-parity/analyze.py runs/lang-parity "$EXP_DIR/results"
PYTHONPATH=. uv run --frozen python studies/lang/lang-parity/plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

Run the training and interpretation steps from a background command. The checks exit non-zero on failure. The training deadline (6 h) counts from the end of setup. Interpretation runs the bias-0 runs first, since they feed H4 and H5, then the rest (finished runs are skipped).

## Expected outputs

- `results/checks.json` and `results/rank.json`.
- `runs/lang-parity/<arm>_k<k>_e<eps>_b<bias>_s<seed>/`: `result.json`, about 21 checkpoints of about 0.25 MB each, and `interpret.json`. Plus `lr.json`, `meta.json`, `planned.json`, `status.json` and `source/`. About 0.7 GB.
- `results/analysis.json` (with the full (k, ε, bias) table of t and censoring counts), `results/curves.json` and `results/interp.json`.
- `figures/t_grid.pdf`, `figures/curves.pdf` and `figures/probes.pdf`.
- `README.md`.

## Controls

- **One LR, paired data.** One LR and one data stream per (k, bias, seed) across all ε, so ε is the only factor that differs in H2. Similarly bias in H3 (each bias has its own stream) and k in H1.
- **No capacity confound.** Representation is not the bottleneck: the sampled rank is 1 at every cut, and the r2 arm tests a near-minimal width.
- **Bias arm.** It adds a low-order signal of known size, (2·bias)^{k−1} per bit. It is timed on unbiased inputs, so a faster time means the parity itself was found sooner.
- **Probe baselines:** step 0, shuffled labels, the additive one-hot probe (a data floor: 0.75 for 2-bit XOR), random-subspace patches (the H4 margin) and whole-bond patching.
- **Censoring.** Censored runs are counted and shown, never dropped.
- **Correctness checks** run first. They include a test that no proper subset of K correlates with y at bias 0.

## Success and failure criteria

- **H1:**
  - supported if, for all 4 ε, no adjacent (k, 2k) pair has 2k faster in ≥ 2 seeds, and k = 2 beats k = 16 in ≥ 2 informative seeds with none the other way;
  - partly supported if this holds for 2–3 ε;
  - not supported otherwise.
- **H2** (ε = 0.3 vs 0.01, k ∈ {4, 8}, 6 pairs):
  - supported if there are ≥ 3 informative pairs, ε = 0.3 wins ≥ 80% of them and loses none;
  - "opposite direction" if the mirror holds;
  - inconclusive with fewer than 3 informative pairs;
  - not supported otherwise.
  - Per-k counts are reported, along with the ε = 1.0 comparisons and the initial losses.
- **H3** (bias 0.2 vs 0, k ∈ {4, 8}, ε ∈ {0.01, 0.1, 0.3}, 18 pairs, timed on unbiased data):
  - supported if there are ≥ 6 informative pairs, bias wins ≥ 75% of them and loses ≤ 1;
  - "opposite direction" if the mirror holds;
  - inconclusive with fewer than 6 informative pairs;
  - not supported otherwise.
- **H4:** supported if (c) holds over the finished, interpreted, learned runs at bias 0; not supported otherwise; inconclusive if none.
- **H5:** descriptive, with counts of bottom-up, simultaneous, other, and not every level causal.
- **Inconclusive:** more than a quarter of the planned runs didn't finish (this overrides every verdict). Censoring is not unfinished: a run that completes 60,000 updates without learning is finished and censored.
- **Run failed:** the checks fail, CUDA is unavailable, or no main run finished.

## Time and cost

- **Setup:** about 10 minutes, plus about 5 minutes for checks and rank.
- **Pilots:** none (LR fixed in advance).
- **Training:** 111 × 60,000 = 6.7 M updates of a tiny model.
  - On the coordinator's CPU it took about 13 ms/update (r = 16) and 2 ms (r = 2).
  - On a GPU these should be launch-bound at about 2–4 ms/update, so with 10 runs sharing the GPU the grid takes about 1–1.5 h.
  - The deadline is 6 h.
- **Interpretation:** 111 runs; deadlines 1.5 h (bias-0 runs first) + 0.75 h (the rest).
- **Total:** at most about 0.3 + 6 + 2.25 + 0.2 ≈ 8.75 h, under the 10 h cap.
- **Cost:** at most $10.90 at $1.09/h, inside $12. Expected about $1–3.

## If something fails

- **Checks fail or CUDA is unavailable:** stop and mark the experiment failed.
- **A run crashes:** rerun the failed runs once into `runs/lang-parity-rerun`, after copying `lr.json` there, with `--only <names>`. Record it as an amendment, and analyze those runs as descriptive.
- **The deadline skips runs:** seed 0 runs first. Rerun the skipped runs if the cap allows; otherwise report them as missing (not censored).
- **Most hard cells are censored:** that's a result. Report the censoring counts. The verdicts already exclude both-censored pairs.
- **GPU memory or CPU contention with 10 processes:** rerun with `--procs 6`.
- **Interpretation times out:** analyze what finished, and report how many runs H4 rests on.
- **No GPU in stock:** wait, then report and stop.
- Nothing here needs to wait for the user.

## Questions asked before launch

- **Batch, budget and interpretation depth:** as for lang-source. The user chose 4 experiments with a 10 h cap and $12 per worker, then said "no budget, just work until finished", a standing go-ahead for the set. They chose probes plus invariance plus patching at every checkpoint.
- **Width:** the user's guidance was r = 16 for the main comparison, so capacity doesn't bind, with narrow runs separately (r = 2 here).
- **Design from the user's review:** parity is the representation-versus-discovery contrast. A plateau is an initialization-dependent prediction rather than a certainty, so init scale is a factor. The partial-parity probe was meant to distinguish partial computations from a weak global coefficient (superseded after review: whitening removes scale, so H4 and H5 now rest on the causal patch).

## Review

An independent review returned "needs revision". Findings and changes:

- **`checks.py` and `rank.py` wrote into a missing `results/` folder (P0).** Both now create it, and the commands run `mkdir -p` first.
- **The probes measure data and gauge (P0 for H4/H5).** H4 is now decided by the causal patch alone, with a margin over random subspaces. The probe is descriptive (implied by accuracy, and often 1.0 at init after whitening). The additive probe is reported as a data floor (0.75 for 2-bit XOR). H5 now times the scale-sensitive causal gap per level instead of probe decodability. The "weak global coefficient" claim is dropped.
- **Bias makes the threshold reachable without parity (P1).** Runs with bias also log an unbiased diagnostic set, and their t uses it.
- **ε = 1.0 starts saturated (P1).** H2 is now ε = 0.3 vs 0.01. ε = 1.0 is reported as a separate regime with initial losses, and excluded from H3's main pairs.
- **Both-censored pairs counted as ties (P1).** They are now uninformative and excluded, and verdicts need a minimum number of informative pairs.
- **The pilot would choose by noise (P1).** The LR is fixed at 1e-3 in advance, with no pilot.
- **Re-check.** It found no remaining P0 or P1. Its P2s are fixed: the interpretation time adds up to 2.25 h, the stale sentence is marked superseded, and the biased-set t is reported next to the unbiased one.
- **P2s.** The checkpoint count is corrected to about 21. Interpretation runs the bias-0 runs first. H1 monotonicity uses the faster() margin on adjacent k. The init wording is corrected (at ε = 0 the network is constant).

## Amendments

None.

## Outcome

Not run yet.
