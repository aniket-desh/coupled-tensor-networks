---
name: rhm-belief-architectures
title: "RHM belief states across tensor-network architectures: aligned tree, permuted tree, chain"
status: done
gpu: NVIDIA RTX A6000,NVIDIA RTX PRO 4500 Blackwell,NVIDIA RTX PRO 4500 Blackwell Server Edition,NVIDIA RTX 6000 Ada Generation
count: 1
hours: 8
budget_usd: 10
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-14T21:42:31Z
result: inconclusive
finished_utc: 2026-11-15T02:15:08Z
writeup: studies/rhm/rhm-belief-architectures/README.md
---

# RHM belief states across tensor-network architectures: aligned tree, permuted tree, chain

## Question

`rhm-belief-geometry` (transformer) and `rhm-ttn-belief-geometry` (aligned tree) ask whether a network trained on
next-token prediction for the Random Hierarchy Model carries the exact belief state b_t = P(pending subtree symbols |
x_<t) in its hidden state. This experiment asks what the architecture contributes to the answer. Three tensor networks
are trained on the same task, grammars, seeds and data stream, and probed the same way:

- **Aligned tree:** the TreeTN whose leaf order is the grammar's. Its two top bonds summarize the left and right half
  of the sequence. This is the model of `rhm-ttn-belief-geometry`, rerun here as the reference.
- **Permuted tree:** the same TreeTN on a fixed random leaf permutation. Its top bonds summarize two scattered sets of
  positions, so no bond's span is a subtree of the grammar.
- **Chain:** a ChainTN read causally, h_t = W(h_{t−1}, token), logits = U h_t. Its state has seen the whole prefix,
  like a residual stream at position t, and its update has the form of filtering, η_t ∝ η_{t−1} T^(x). If it filtered
  that way, its state would be an unnormalized belief, whose scale varies from prefix to prefix. It is trained through
  a softmax of U h_t, though, so its state may instead be a code whose linear readout is the log of a distribution.

Two questions follow. Can the belief be read from the chain's recurrent state by a simple probe: affinely, up to
normalization, or through a softmax? And does
matching the tree to the grammar's hierarchy change how well the task is learned and how well the belief can be read
from the top bonds?

## Hypothesis

Both are judged on held-out prefixes at eligible positions (defined under Method). Every comparison is conditional on
one learning rate, 1e-3, for all arms.

- **H_A, chain: the belief can be read from the recurrent state by a simple probe.** In each grammar the chain learns
  (excess test loss < 0.02 nats/token in all three seeds), at least one of three probe families holds in all three
  seeds. Each family is judged against its own controls, the identity-init chain and the token readout under the same
  probe, so the three routes do not share a baseline.
  - **A1, affine:** on the positions eligible for the state probe, the affine probe on h_t reaches pooled R² ≥ 0.8 and
    beats both controls by at least 0.1.
  - **A2, up to normalization:** on the eligible positions with belief dimension D_t ≤ 64, the projective probe
    b̂ = q / Σq, q affine in h_t, reaches pooled R² ≥ 0.8 and beats both controls by at least 0.1.
  - **A3, readout-matched:** on the same positions, the probe b̂ = softmax(W h_t + c) reaches pooled R² ≥ 0.8 and beats
    both controls by at least 0.1.
  - Why: h_t is the only thing the chain passes forward, so a chain that predicts near Bayes has in h_t everything the
    prediction needs. The form is open. A multilinear update that filters, η_t ∝ η_{t−1} T^(x), would make h_t an
    unnormalized belief, which A2 reads and A1 does not. But the chain is trained through a softmax of U h_t, so
    nothing forces that form: h_t may instead be a code whose linear readout is the log of a distribution, which A3
    reads.
  - Which families hold is reported per grammar.
  - Against: none of the three holds in a majority of the learned grammars. That would mean the belief is not readable
    from the state by an affine, projective or log-linear map. It would not mean the belief is absent from the state.
- **H_B, alignment: the aligned tree learns the task better and holds a more readable belief than the permuted
  tree.** Runs are paired by (grammar, seed), 12 pairs, each pair sharing rules, data stream and held-out prefixes.
  - **Learning:** the permuted tree's final excess test loss is higher than the aligned tree's by more than the
    tolerance (the larger of 0.001 nats and the aligned tree's seed range on that grammar). Pairs where the aligned
    tree itself did not learn are not used.
  - **Representation:** the bilinear top-bond probe [h_A, h_B, h_A ⊗ h_B], on the common positions past the midpoint,
    in pairs where both trees learned.
    - The quantity is the gap R²(aligned) − R²(permuted). A pair goes to the aligned tree only if the gap at the end
      of training exceeds the same gap at identity init by at least 0.1. Two untrained trees already differ, because
      the leaf order alone decides what each top bond sees; subtracting that leaves what training added.
    - Only pairs where both trees learned count. A lower probe score in a tree that did not learn the task says
      nothing about alignment beyond the learning measure.
    - Only positions with t > T/2 count. Up to the midpoint the aligned tree's second top bond is constant, so its
      bilinear probe has far fewer effective features than the permuted tree's, and the two probes are not comparable.
      The same quantity for t ≤ T/2 is reported apart, with no verdict.
    - The bilinear probe is used because it can express the belief in both trees: the two top bonds together have
      seen the whole prefix, and an affine probe is bounded by an additive ceiling in each.
  - Why: `rhm-controls` found the permuted tree slower and worse on two single-target tasks (12 of 12 pairs on both
    measures), with each arm at its own piloted learning rate; its pilot chose the top of its grid for the permuted
    tree, so a higher rate was not ruled out there either. A bond over scattered positions has to summarize tokens
    whose latent structure does not factor.
  - The comparison is two-sided: a measure can go to either tree, and "the permuted tree is better" is a reportable
    outcome.
  - A measure needs at least 9 usable pairs (three grammars' worth); below that it is untestable. If the permuted tree
    fails to learn in most pairs, as `rhm-controls` suggests it may, the representation measure is untestable and H_B
    can be at most partly supported, on learning alone.
  - Against: on a testable measure the aligned tree does not win at least 80% of the usable pairs consistently across
    grammars.
- **What a supported H_A or the representation half of H_B does and does not show.** As in
  `rhm-ttn-belief-geometry`, a position has few distinct beliefs compared with a probe's features, so a probe that
  linearly identifies the belief state can fit any labelling of the states. These hypotheses test whether the hidden
  state determines the belief on unseen prefixes through the stated probe. Whether it holds the belief's geometry is
  descriptive, read from the margin over a scrambled target and from the belief split.
- **Descriptive:**
  - the chain against the aligned tree, paired the same way;
  - each tree's affine top-bond probe against its additive bound;
  - probe R² over training;
  - per-position overlays of the three arms, and against the transformer once `rhm-belief-geometry`'s results exist
    (made by the planning session afterwards: `arch_plot.py` takes that results folder as a third argument).

## Method

- **Grammars:** `v8m4s2L4`, `v8m2s2L4`, `v16m4s2L4` (T = 16) and `v4m2s2L5` (T = 32), all binary, with seeds 0, 1, 2.
  The seed sets the grammar's rules and the model's init, as in the other two belief experiments, so run
  (config, seed) here has the same rules, test set, Bayes loss, analysis sequences and held-out prefixes as theirs.
- **Arms** (`arch_run.py`), 4 × 3 × 3 = 36 runs, all with the identity init (eps = 0.1), no biases or nonlinearities:
  - `tree`: `ctn.tree.TreeTN`, untied, r = 32, leaf vocabulary v + 1 (mask token), built by `ttn_run.build`.
  - `perm`: the same TreeTN with `perm`: model leaf i reads position perm[i].
    - One seeded permutation per (config, seed), redrawn until no sibling pair of the data tree are siblings in the
      model (the rule of `rhm-controls/run.py`, generalized to any length).
    - The permutation, and how many model nodes still cover a data node's exact leaf set at each level, are saved in
      each run's `result.json`. For the 12 planned runs that count is 0 at every level.
  - `chain`: `ctn.tree.ChainTN` read causally (`belief_common.chain_states`): logits[:, t] = U h_t predicts x_t from
    x_<t, with no masks.
    - Its width is the one whose parameter count is closest to the aligned tree's: 78 for T = 16 (477,126 and 475,878
      parameters against the tree's 483,840 and 471,552) and 100 for T = 32 (1,000,900 against 992,256).
    - Matching parameters also matches the work per update: the chain's T steps of a width-w core cost w³ per
      sequence and step, and w³ ≈ (T − 2) r³, the tree's cores.
- **Task and training:**
  - Every arm's update uses the same 512 fresh sequences, from the same seeded stream per (config, seed), and the
    same loss: mean cross-entropy over sequences and positions.
  - The trees see every prefix of each sequence with the suffix masked (512 × T rows). The chain sees each sequence
    once and predicts at every position.
  - Adam, peak LR 1e-3, linear warmup over 100 updates, cosine to 0.1 × peak (`studies.rhm.train.lr_at`), 30,000
    updates, float32.
  - **LR:** one rate for all arms, 1e-3, and every result is conditional on it.
    - CPU pilot (v8m4s2L3, 1,500 updates of batch 128, with the warmup-cosine schedule compressed to those 1,500
      updates): the chain's excess loss ended at 0.158 (3e-4), 0.064 (1e-3) and 0.042 (3e-3). The aligned and
      permuted trees at 1e-3 ended at 0.043 and 0.087.
    - So the chain trains at 1e-3, and 3e-3 was ahead in the short pilot. 1e-3 is kept because `rhm-dynamics` saw wide
      trees collapse at 3e-3 only after several thousand updates, which a short pilot cannot rule out.
    - Whether a higher rate would help the permuted tree or the chain is not tested here: the GPU time measured on
      the sibling run leaves no room for a second rate. The write-up must say so next to every comparison.
  - Checkpoints at updates 0, 300, 1000, 3000, 10000 and 30000. Every log point records per-position test loss on
    8,192 fixed sequences next to the exact Bayes loss.
- **Hidden states and probes** (`arch_analyze.py`), per prefix length t:
  - Trees: the bonds of the tree run on its training input. `top` = [h_A, h_B], the two bonds under the root (affine,
    64 features); `top_bil` = [h_A, h_B, h_A ⊗ h_B] (bilinear, 1,088 features, first and last checkpoint); `all` =
    every bond of tree levels 1 and above (affine).
    - In the permuted tree both top bonds vary from the first token on, so it has no first half in which one bond
      carries the whole prefix.
  - Chain: `state` = h_t (affine, 78 or 100 features).
    - Projective probe (`belief_common.projective_fit_predict`): b̂ = q / Σq with q affine in h_t.
    - Softmax probe (`arch_analyze.softmax_fit_predict`): b̂ = softmax(W h_t + c), least squares on the belief by
      L-BFGS, started from the affine fit to the centered log-belief.
    - Both are fit at the first and last checkpoint, on the prefix split, where D_t ≤ 64.
  - The trees get no projective or softmax fits. They enter no verdict, and the 1,088-feature fits are slow on a CPU.
  - The fits, the 60,000 analysis sequences, the sequence split (figures) and the prefix split (criteria) are those of
    the other two experiments, with the same seeds.
- **Eligible positions:** at least 10 × (features + 1) distinct fit prefixes.
  - `ea`: eligible for the arm's affine probe (650 for `top`; 790 or 1,010 for `state`).
  - `eb`: eligible for the bilinear probe (10,890). This set is the same in every arm, the chain included, and is the
    common set on which the arms are compared. `eb_lo` and `eb_hi` are its positions with t ≤ T/2 and t > T/2.
  - `pa`: the `ea` positions with D_t ≤ 64, where the chain's projective and softmax probes exist.
- **Gauge:** affine, bilinear, projective and softmax probe scores are invariant to an invertible linear change of a
  bond or of the chain state. The raw PCA panels are not, and are labelled so.
- **Figures** (`arch_plot.py`, PDFs from saved results only), for seed 0:
  - `geometry_<config>_<arm>.pdf`: ground truth, the arm's probes, the shuffled control, the identity-init network,
    and the raw PCA of the probed state.
  - `positions.pdf`: per-position R² of the three arms' probes in one panel per grammar, with ceilings.
  - `summary.pdf`: pooled R² on the common set past the midpoint and final excess loss, one point per seed.
  - `training.pdf`.
- **Verdict:** `arch_summary.py` applies the criteria and writes `results/summary.json`.

## Inputs

None. All data is generated from the seeded grammars. The code imports, read-only, `studies/belief_common.py`,
`studies/rhm/rhm-belief-geometry/` (`analyze.py`, `beliefs.py`, `run.py`, `plot.py`) and
`studies/rhm/rhm-ttn-belief-geometry/` (`ttn_run.py`, `ttn_analyze.py`).

## Commands

From the project root on the worker. Three `orch exec` steps: the checks; then training and analysis, which run at
the same time; then the verdict and figures.

```bash
set -euo pipefail
nvidia-smi
nproc
free -g
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-geometry/check_probe.py
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-belief-geometry/ttn_check.py
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-architectures/arch_check.py
```

**Training**, four runs at a time on the GPU, grammar by grammar: all arms and seeds of one grammar before the next,
the T = 16 grammars first, v4m2s2L5 last with its chain runs at the very end. A time cut then loses whole grammars.

- No run starts after `STOP_AT`, the worker's 5.5 h mark. The command sets it to 5 h 10 min after the training step
  starts, which assumes that step starts 20 minutes after launch; if it starts later, the supervisor lowers 18600
  accordingly.
- A run is also skipped if a file `runs/rhm-belief-architectures/SKIP_<config>_<arm>` exists (see "Time and cost").

```bash
set -euo pipefail
mkdir -p runs/rhm-belief-architectures
export STOP_AT=$(( $(date +%s) + 18600 ))
{ for c in v8m4s2L4 v8m2s2L4 v16m4s2L4; do for s in 0 1 2; do for arm in chain perm tree; do echo "$c $arm $s"; done; done; done
  for s in 0 1 2; do for arm in perm tree; do echo "v4m2s2L5 $arm $s"; done; done
  for s in 0 1 2; do echo "v4m2s2L5 chain $s"; done
} | xargs -P 4 -n 3 sh -c 'if [ "$(date +%s)" -ge "$STOP_AT" ] || [ -e "runs/rhm-belief-architectures/SKIP_$0_$1" ]; then echo "SKIPPED $0 $1 s$2"; exit 0; fi; OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-architectures/arch_run.py "$0" --arm "$1" --seed "$2" --out runs/rhm-belief-architectures > runs/rhm-belief-architectures/train_$0_$1_s$2.log 2>&1 || echo "FAILED $0 $1 s$2"'
touch runs/rhm-belief-architectures/TRAINING_DONE
```

**Analysis**, started right after the training step and running beside it. It analyzes every run that has a
`result.json` and no `analysis.json` yet, and ends once training has ended and nothing is left.

- While training runs, the analyses use the CPU (`CUDA_VISIBLE_DEVICES=`), two at a time. On the sibling run, GPU
  analyses beside four training runs died with CUDA out of memory.
- Once `TRAINING_DONE` exists, the remaining analyses use the GPU, one at a time.
- The fits are float64 least squares on either device; only the network's float32 forward pass and the BLAS differ.
- A run whose analysis fails gets a `FAILED` file in its results folder and is not retried until that file is removed.

```bash
set -euo pipefail
mkdir -p "$EXP_DIR/results"
R=runs/rhm-belief-architectures
while :; do
  todo=$(for d in $R/*_s[012]; do r=$(basename "$d"); if [ -f "$d/result.json" ] && [ ! -f "$EXP_DIR/results/$r/analysis.json" ] && [ ! -f "$EXP_DIR/results/$r/FAILED" ]; then echo "$d"; fi; done)
  if [ -n "$todo" ]; then
    if [ -f $R/TRAINING_DONE ]; then export DEV=ANALYSIS_ON_GPU=1 PAR=1; else export DEV=CUDA_VISIBLE_DEVICES= PAR=2; fi
    echo "$todo" | xargs -P $PAR -I{} sh -c 'r=$(basename {}); mkdir -p "$EXP_DIR/results/$r"; env $DEV OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-architectures/arch_analyze.py {} "$EXP_DIR/results/$r" > runs/rhm-belief-architectures/analyze_$r.log 2>&1 || { touch "$EXP_DIR/results/$r/FAILED"; echo "FAILED $r"; }'
  elif [ -f $R/TRAINING_DONE ]; then break
  else sleep 60; fi
done
```

**Verdict and figures:**

```bash
set -euo pipefail
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-architectures/arch_summary.py "$EXP_DIR/results" 36 | tee "$EXP_DIR/results/summary.txt"
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-belief-architectures/arch_plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

## Expected outputs

- `results/checks.json`.
- `runs/rhm-belief-architectures/<config>_<arm>_s<seed>/` (36 folders): `result.json` plus 6 checkpoints of 2 to 4 MB
  each, with the training and analysis logs next to them. About 700 MB in total.
- `results/<config>_<arm>_s<seed>/`: `analysis.json`, `proj_ckpt0.npz`, `proj_ckpt30000.npz`. Under 200 MB in total.
- `results/summary.json` and `results/summary.txt`.
- `figures/*.pdf`: 12 geometry figures, `positions.pdf`, `summary.pdf`, `training.pdf`.
- `README.md`, the write-up.

## Controls

- **Aligned tree in the same run:** the reference of H_B, with the same seeds, data stream and hyperparameters as the
  permuted tree, differing only in the leaf order.
- **Parameter matching:** the chain's width is chosen to match the aligned tree's parameter count (within 1.4%); the
  two trees have identical counts. The chain is not matched on bond dimension.
- **Identity-init network:** checkpoint 0 of every arm, under every probe and both splits. For H_B's representation
  measure it is also the baseline of the gap itself: the aligned-minus-permuted gap of two untrained trees is
  subtracted from the gap after training.
- **Token readouts that see what the probe sees:**
  - affine probes: the affine probe on the one-hot prefix;
  - the chain's projective and softmax probes: the same probe on the one-hot prefix;
  - bilinear probes: least squares on [onehot(prefix tokens in A), onehot(prefix tokens in B), their products], with
    A and B the two position sets the top bonds see (the halves for the aligned tree). It is computed where it meets
    the eligibility rule, and the positions it could not cover are listed in `analysis.json`.
- **Additive bound:** the bound on any affine probe on two bonds that saw disjoint position sets, the best fit of the
  form f(what A saw) + g(what B saw), computed without the network. It is descriptive; no verdict uses it.
  - Aligned tree, t > T/2: f(left-half symbol) + g(right-half parse state), as in `rhm-ttn-belief-geometry`.
  - Permuted tree: f(id of the prefix tokens in A) + g(id of the prefix tokens in B), fit by backfitting, where the
    tables fit in memory.
  - Two numbers are stored. The fit's R² on its own fit rows is an upper bound on what any additive function reaches
    (`addup`). The same fit scored on held-out prefixes is a lower estimate (`addceil`).
  - For the aligned tree the two agree, because a few symbols parametrize the fit. For the permuted tree they do not:
    the sub-prefix ids are many and each co-occurs with few others. In the pilot at t = 5 the held-out estimate was
    0.19, below the affine token readout's 0.44, although that readout is itself additive; the upper bound was 0.60.
    So for the permuted tree the bound to read is `addup`, and the held-out estimate is stored only where both ids of
    at least 95% of the test rows occur in the fit rows.
  - The upper bound becomes uninformative (near 1) once the ids are almost as many as the prefixes; the id counts are
    stored next to it.
- **Prefix split and eligibility**, **shuffled labels**, **next-token ceiling**, **scrambled target** (with the number
  of distinct beliefs per pooled position) and **belief split** (with its span ceiling): as in
  `rhm-ttn-belief-geometry`. The belief split fits on few distinct beliefs and is a weak test either way; it is
  reported with that caveat.
- **Learning check:** excess test loss over the exact Bayes loss, per position and overall, for every run.
- **Correctness checks before training:**
  - `studies.rhm.checks`, `check_probe.py` and `ttn_check.py`, as in the other two experiments.
  - `arch_check.py`, which passes on the coordinator (17 checks):
    - the permutations are valid, deterministic and keep no sibling pair;
    - in the permuted tree each top bond depends only on its own position set, and both vary at t = 2, where the
      aligned tree's right bond is constant;
    - the chain's state and logits at t do not change when later tokens change, its loss rows are ordered like the
      trees', and its width is the closest in parameter count;
    - the backfitted additive fit equals dense least squares on the one-hot design, fits an additive target exactly
      and not a pure interaction;
    - the token-product features have the stated width and equal `ttn_analyze.token_product` for the aligned halves;
    - the softmax probe recovers a log-linear code of the belief (R² 1.000 where the affine probe scores 0.57), its
      rows sum to 1, and it is unchanged by a gauge change;
    - the pairing rules of `arch_summary.py` give the stated outcomes on hand-made runs: the 80% rule with grammar
      consistency and the minimum of 9 pairs, the identity-init gap subtracted, pairs with an unlearned permuted tree
      left out of the representation measure, and a grammar with an unfinished run censored whole.
  - `belief_common.py`'s projective probe was checked by the planning session on synthetic activations: on an
    unnormalized code it scored R² 1.000 where the affine probe scored 0.36, and it is unchanged by a gauge change.
- **Pilot of the whole pipeline** (CPU, v8m4s2L3, 1,500 updates of batch 128, 30,000 analysis sequences, seed 0; the
  only common position is t = 7): train, analyze, summary and plot ran end to end for the three
  arms. No arm had learned yet (excess loss 0.043 tree, 0.087 perm, 0.064 chain), so no verdict applies; the numbers
  only show the pipeline and the size of the controls.
  - Trees, bilinear top-bond probe at t = 7: aligned 0.760 (identity init 0.221), permuted 0.584 (init 0.345). The gap
    is 0.176 after training and −0.124 at init, so the adjusted gap is 0.300. The token-product readout scored 0.245
    and 0.360.
  - Chain, pooled over t = 5, 6, 7: affine 0.596 (init 0.126, tokens 0.266), projective 0.612 (init 0.140, tokens
    0.271), softmax of affine 0.755 (init 0.114, tokens 0.293).
  - Each analysis took 64 to 95 seconds on the coordinator's two shared cores.

## Success and failure criteria

Computed by `arch_summary.py` from `results/*/analysis.json` at the final checkpoint, on the prefix split.

- **Censoring is by whole grammar.** A run counts only with status `finished` and an analysis.
  - A grammar enters H_B only if all six of its tree and perm runs count.
  - A grammar enters H_A only if all three of its chain runs count; the descriptive tree-against-chain comparison
    needs those and the three tree runs.
  - Censored grammars are listed apart from cells that did not learn.
- **Learned:** all three seeds of a (grammar, arm) cell have excess test loss < 0.02 nats/token. A cell failing this is
  "not learned", which is a finding, not a failure.
- **H_A per grammar** (the chain cell must be learned). With "controls" meaning the identity-init chain and the token
  readout, both under the same probe:
  - A1 holds if, in all three seeds, R²(`state`, affine, `ea`) ≥ 0.8 and it exceeds both controls by at least 0.1.
  - A2 holds if, in all three seeds, R²(`state`, projective, `pa`) ≥ 0.8 and it exceeds both controls by at least 0.1.
  - A3 holds if, in all three seeds, R²(`state`, softmax of affine, `pa`) ≥ 0.8 and it exceeds both controls by at
    least 0.1.
  - The grammar supports H_A if A1, A2 or A3 holds. A grammar with no `ea` position and no `pa` position is
    untestable.
- **H_A overall, over learned and testable grammars:** supported if at least 75% support it; not supported if at most
  25% do; partly supported in between; inconclusive if fewer than 3 can be judged.
- **H_B, over the (grammar, seed) pairs of uncensored grammars**, at most 12.
  - **Learning.** Usable pairs: those where the aligned tree learned (excess < 0.02); the others are listed as
    "reference not learned".
    - gap = permuted excess − aligned excess. The aligned tree wins a pair if gap > max(0.001 nats, the aligned tree's
      seed range of excess loss on that grammar), the permuted tree wins if gap is below minus that, and otherwise
      the pair is a tie.
    - The median gap over the usable pairs is reported next to the counts.
  - **Representation.** Usable pairs: those where both trees learned and `eb_hi` is not empty.
    - gap = R²(aligned `top_bil`) − R²(permuted `top_bil`) on `eb_hi`, at the final checkpoint and at identity init.
    - The aligned tree wins a pair if gap(final) − gap(init) ≥ 0.1 and gap(final) > 0. The permuted tree wins if
      gap(final) − gap(init) ≤ −0.1 and gap(final) < 0. Otherwise the pair is a tie, so two trees that end with equal
      R² never produce a win.
    - Both gaps are printed for every pair, with the median adjusted gap.
    - The same counts on `eb_lo` and on all of `eb` are reported, with no verdict.
  - **A measure goes to an arm** if that arm wins at least 80% of the usable pairs (10 of 12, one-sided sign test
    p = 0.019 without ties; 8 of 9, p = 0.020), has more wins than losses in all but at most one of the grammars with
    usable pairs, and fewer in none. This is the consistency rule of `studies/rhm/analysis.py`.
  - **A measure with fewer than 9 usable pairs is untestable.**
  - **Verdict:**
    - supported if both measures go to the aligned tree;
    - partly supported if exactly one does and none goes to the permuted tree (the other is then named as "to
      neither" or "untestable");
    - "reversed on a measure" if a measure goes to the permuted tree;
    - not supported if no measure goes to either tree and at least one is testable;
    - inconclusive if both measures are untestable.
- **Descriptive, with no verdict:**
  - the chain against the aligned tree by the same two measures (the chain's `state` probe against the tree's
    `top_bil`);
  - each probe's margin over the scrambled target; the belief split;
  - each tree's affine probe against its additive bound; the bilinear probes against the token-product readout;
  - the chain's three probe families on the same positions.
- **Run failed:**
  - the checks fail;
  - no CUDA;
  - fewer than 31 of the 36 runs finish both training and analysis;
  - or the shuffled-label pooled R² exceeds 0.05 in any run, which would mean a probe bug.

## Time and cost

- **Setup and checks:** about 20 minutes.
- **Training:** 36 runs × 30,000 updates, four at a time.
  - Measured on the sibling run's worker (`rhm-ttn-belief-geometry`, a 31 GB GPU at $0.72/h, four runs sharing it at
    100% utilization): a T = 16, r = 32 tree run takes 17 minutes (34–36 ms per update), and a T = 32, r = 32 run about 65
    minutes (130 ms per update).
  - The tree and perm runs here are those models: 18 at T = 16 and 6 at T = 32, about 700 slot-minutes, or 2.9 h at
    four slots.
  - The chain's GPU speed is not measured. Its update does the same number of multiplications as the tree's (see
    Method), and on the coordinator's CPU it took 41 ms per update against the tree's 34 ms (v8m4s2L3, batch 128). If
    it costs the same on the GPU, the 9 + 3 chain runs add about 350 slot-minutes, and training ends about 4.4 h after
    it starts.
  - **Chain timing rule.** The first chain run (v8m4s2L4, seed 0) is among the first four runs. At its update 1,000
    the log shows the seconds elapsed; 30 times that is its projected length.
    - Up to 40 minutes projected: carry on. At 40 minutes the last v4m2s2L5 chain run would start at about the 5.3 h
      mark and end before the 7 h mark.
    - More than 40 minutes: the three v4m2s2L5 chain runs would take over 2.5 h each. The supervisor creates
      `runs/rhm-belief-architectures/SKIP_v4m2s2L5_chain`, so they are not started. v4m2s2L5 is then censored for
      H_A and for the tree-against-chain comparison and stays in H_B. Recorded under Amendments.
- **Analysis:** runs beside training on the CPU, two at a time, as runs finish.
  - The sibling's CPU analyses on that worker (112 cores) took about 10 minutes for a v = 8, T = 16 run and about 20
    for a v = 16 or a T = 32 run. This analysis has fewer activation sets and adds the chain's small projective and
    softmax fits, so about the same per run.
  - The 27 T = 16 runs then need about 3 h of analysis, spread over the first 4 h. The v4m2s2L5 runs finish last, and
    those still unanalyzed when training ends go to the GPU, one at a time, at an estimated 5–10 minutes each.
- **Total:** about 5.5 h expected if the chain costs what the tree does, and about 7 h at the 40-minute limit of the
  chain rule, with an 8 h cap.
- **Cost:**
  - Worst case: $1.09/h (RTX 6000 Ada, the most expensive GPU listed) × 8 h = $8.72, inside the $10 budget.
  - Expected: about $4–4.7 at $0.69–0.72/h.

## If something fails

- **A training or analysis run fails** (FAILED in a step's output, or a `FAILED` file in a results folder): read its
  log. May be decided without asking:
  - A bug in this folder's code: fix it and rerun only the failed runs, noted under Amendments. Files outside this
    folder belong to other experiments and must not be changed; copy what needs changing into this folder. After a
    fix to the analysis, remove the `FAILED` files so the loop retries those runs.
  - Out of GPU memory in training: rerun those runs alone (`-P 1`), noted under Amendments.
  - Out of memory in analysis: rerun that run alone, noted under Amendments. A smaller analysis set is not allowed
    without asking: it changes which positions are eligible and breaks the shared held-out prefixes.
  - A CPU analysis that is much slower than estimated on this worker (fewer cores than the sibling's): let it run;
    the analyses left when training ends go to the GPU anyway.
  - A run that diverges or collapses to the uniform output: report it as such, don't retune. Its grammar is censored
    (diverged) or its cell not learned (collapsed).
  - An arm that does not learn a grammar: report it, don't retune. The LR is fixed by the definition.
- **Time running short:**
  - No training run starts after the 5.5 h mark (`STOP_AT`). Runs already started finish.
  - Training goes grammar by grammar, so a cut loses the last grammar first. A grammar with an unfinished tree or perm
    run is censored for H_B as a whole, and one with an unfinished chain run is censored for H_A as a whole; its
    finished runs are still analyzed and reported descriptively.
  - At the 7 h mark, stop whatever training or analysis is still running, and run the verdict and figures on what
    exists.
  - Unfinished runs are reported as censored, never dropped silently.
- **Checks fail or there is no CUDA:** stop, delete the worker, report.
- **Must wait for the user:** any change to the grammars, the arms, the widths, the LR or the success criteria; any
  spend above the budget.

## Questions asked before launch

None needed. The user's go-ahead of 2026-11-14 for this batch was "might as well add in the rest of these", after the
theory agent's note proposing architecture comparisons on the same prediction task, and the same day the user
confirmed launching it within $10 once it passed its pilot and review. Choices made without asking, all recorded
above: the four grammars (a subset of the two running belief experiments, with enough distinct prefixes for the
bilinear probe; v3m2s2L5 and a second learning rate were dropped to fit the 8 h cap at the GPU speed measured on the
sibling run), tree width 32, the parameter-matched chain, one LR for all arms, and the bilinear probe as the common
measure for the two trees.

## Review

experiment-reviewer, first pass: "needs revision", one P0, two P1s and four P2s. The findings as the planning session
passed them on, and what changed:

- **P0. The plan did not fit the 8 h cap.** The time estimate assumed 5–12 minutes per run; the sibling run measures
  17 minutes at T = 16 and about 65 at T = 32 with four runs sharing the GPU, which put the 60 planned training runs at
  over 9 h.
  - Fixed, by the planning session's decisions:
    - the 15 LR-sensitivity runs are dropped, and every comparison is stated as conditional on LR 1e-3;
    - grammar v3m2s2L5 is dropped, leaving 4 grammars × 3 arms × 3 seeds = 36 runs;
    - training goes grammar by grammar with v4m2s2L5 last, no run starts after the 5.5 h mark, and censoring is by
      whole grammar;
    - the time estimate uses the measured tree rates, with a rule for timing the unmeasured chain at update 1,000 of
      its first run;
    - the analysis runs on the worker's CPU beside training, as the sibling's amendments do, and the trees'
      projective fits are dropped.
  - One refinement of "a grammar with any unfinished run is censored as a whole": it is applied per hypothesis (tree
    and perm runs for H_B, chain runs for H_A). Otherwise the chain rule, which drops the v4m2s2L5 chain runs first,
    would also discard that grammar's six finished tree runs.
- **P1. H_B's representation measure could be won without anything learned.** A permuted tree that did not learn, the
  leaf order's effect on two untrained trees, and the aligned tree's smaller effective probe before the midpoint
  could each produce the 0.1 gap.
  - Fixed: the measure uses only pairs where both trees learned, counts a pair only if the final gap exceeds the
    identity-init gap by 0.1, uses positions with t > T/2, prints the init gap beside every pair, and reports t ≤ T/2
    apart. A measure needs 9 usable pairs, three grammars' worth.
- **P1. H_A read a failure of the affine and projective probes as "the belief is not in the state".** The chain is
  trained through a softmax of U h_t, so its state need not be an unnormalized belief.
  - Fixed: a readout-matched probe, softmax of an affine map, is added as a third family with the same controls
    (`softmax_fit_predict`, checked on a synthetic log-linear code), and H_A and its "Against" are reworded.
- **P2s.**
  - The `PILOT_SPEED` placeholder is replaced by the measured rates.
  - The LR tag is gone with the LR-sensitivity runs.
  - The median paired gap is reported next to the win counts.

experiment-reviewer, second pass: the P0 and both P1s judged addressed, no new P0; "ready to run".

- **One note, not blocking: a pair could be scored as an aligned win with equal final R².** The win rule was
  gap(final) − gap(init) ≥ 0.1 alone, and the pilot's init gap is negative (−0.124), so a final gap near zero would
  still count.
  - Fixed: a win also needs the final gap on the winner's side (`arch_summary.py`), and the criterion says so.

## Amendments

- **2026-11-15 01:50Z, where the projection files are kept (supervisor, before finishing the worker).** The 72
  `proj_ckpt*.npz` files came to 1.2 GB, not the "under 200 MB" estimated under Expected outputs, and the repository's
  whole `.git` is 1.1 GB. Published inside the experiment folder they would have been committed by the merge, or split
  by the toolkit's 20 MB limit for repository files.
  - Change: after the verdict and figures step, the files are moved on the worker from `results/<run>/` to
    `runs/rhm-belief-architectures/proj/<run>/`, a data folder that comes back to the coordinator outside git. On the
    coordinator they are copied back into `results/<run>/`, where `arch_plot.py` expects them, and a `.gitignore` in
    this folder keeps them out of commits.
  - No effect on the method, the criteria or any number: `summary.json` and the figures were written before the move,
    and `arch_summary.py` reads only `analysis.json`.

## Outcome

Recorded as **inconclusive**. The detail is in [README.md](README.md), with the mathematics and figures in
[NOTE.md](NOTE.md). Every line is conditional on learning rate 1e-3.

- **Run failed: no.** The checks passed and CUDA was present; 36 of 36 runs finished training and analysis (31
  needed); the shuffled-label pooled R² is at most 0.0003 in every run (limit 0.05). No grammar is censored.
- **Learned:** the aligned tree learned all four grammars. The permuted tree and the chain learned v8m2s2L4 and
  v4m2s2L5 and did not learn v8m4s2L4 and v16m4s2L4.
- **H_A: inconclusive.** Two learned, testable grammars, and the rule needs three. Both support it: A1 and A2 hold
  in all three seeds of both, A3 in v4m2s2L5 only.
- **H_B: not supported.**
  - Learning: 9 wins, 0 losses, 3 ties of 12 usable pairs, to neither tree (10 wins needed); median gap 0.020 nats.
  - Representation: 6 wins, 0 losses of 6 usable pairs, untestable (9 pairs needed); median adjusted gap 0.153.
- **Descriptive:** aligned tree against chain, learning 9 wins, 0 losses, 3 ties of 12, and 6 ties of 6 on
  representation. On the two learned grammars the scrambled target scores within 0.06 of the real belief in every
  arm, and the belief split is negative.
- **Cost and runtime:** 4 h 13 min of the 8 h cap, about $3.04 of the $10 budget, on one RTX PRO 4500 Blackwell at
  $0.72/h. The user's relaxation of 2026-11-15 01:50 UTC (ceiling $15, movable time cutoffs) arrived after the last
  run had been analyzed and was not used.
