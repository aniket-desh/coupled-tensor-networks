# RHM belief states across tensor-network architectures: aligned tree, permuted tree, chain

Neither hypothesis reached a verdict of "supported", and for different reasons. Three tensor networks with matched
parameter counts were trained on next-token prediction for four Random Hierarchy Model (RHM) grammars, 3 seeds each,
30,000 updates, all at learning rate 1e-3. The tree whose leaf order is the grammar's (the aligned tree) learned all
four grammars, ending 0.0007 to 0.0059 nats/token above the exact Bayes loss. The same tree on a fixed random leaf
permutation and the causal chain learned only the two grammars with m = 2 synonyms per symbol, and stayed above the
0.02 threshold on the two with m = 4 (chain 0.024 to 0.068, permuted tree 0.039 to 0.157). That one fact limits both
hypotheses:

- **H_A, the belief can be read from the chain's state by a simple probe: inconclusive.** The rule needs three
  learned grammars and the chain learned two. Both support it: the affine probe on the chain state reaches R² 0.96 to
  0.98 on prefixes it never saw, against 0.45 to 0.55 for the same probe on the raw tokens and 0.19 to 0.31 for the
  untrained chain (2 grammars × 3 seeds).
- **H_B, the aligned tree learns better and holds a more readable belief than the permuted tree: not supported by
  the pre-set rule.** On learning, the aligned tree won 9 of 12 (grammar, seed) pairs by more than the tolerance, with
  3 ties and no losses, and the rule needs 10 of 12. On representation, it won all 6 usable pairs, and the rule needs
  9 usable pairs, so that measure is untestable. No pair on either measure went to the permuted tree.

Every comparison here is conditional on the single learning rate. Whether a higher rate or a longer run would let the
permuted tree or the chain learn the m = 4 grammars was not tested, and their loss was still falling over the last
10,000 updates.

Grammar names read vVmMsSLL: V symbols per level, M synonymous productions per symbol, branching S, depth L, so a
sequence has T = S^L tokens.

## The evidence

**Learning.** Final excess test loss over the exact Bayes loss, in nats/token, on 8,192 fixed test sequences: median
over 3 seeds with [min, max]. "Learned" means all three seeds below 0.02.

| grammar | T | aligned tree | permuted tree | chain |
|---|---|---|---|---|
| v8m2s2L4 | 16 | 0.0010 [0.0010, 0.0013] | 0.0014 [0.0013, 0.0029] | 0.0015 [0.0013, 0.0016] |
| v4m2s2L5 | 32 | 0.0008 [0.0007, 0.0008] | 0.0027 [0.0013, 0.0040] | 0.0026 [0.0020, 0.0028] |
| v8m4s2L4 | 16 | 0.0021 [0.0021, 0.0024] | 0.0502 [0.0390, 0.0696] **not learned** | 0.0262 [0.0235, 0.0293] **not learned** |
| v16m4s2L4 | 16 | 0.0052 [0.0048, 0.0059] | 0.1308 [0.0956, 0.1569] **not learned** | 0.0631 [0.0618, 0.0683] **not learned** |

[`figures/training.pdf`](figures/training.pdf) shows the loss of every run against updates. In the six chain runs and
six permuted-tree runs that did not learn, the excess loss was still decreasing over the last 10,000 updates: between
updates 20,000 and 30,000 it fell by 0.004 to 0.008 nats for the chain and by 0.006 to 0.012 for the permuted tree.
Over the last 1,000 updates it fell by 0.0001 to 0.0007 in eleven of the runs and did not move in one (v8m4s2L4,
chain, seed 2). "Not learned" therefore means not within 30,000 updates at this rate.

**H_A: three probes on the chain's state.** The chain's state h_t after t tokens is probed for the exact belief
b_t = P(pending subtree symbols | x_<t) in three ways: affine (b̂ = W h + c), projective (b̂ = q / Σq with q affine
in h, which reads a belief stored up to a per-prefix scale), and softmax of an affine map (which reads a belief stored
as log-probabilities). Each is judged against the untrained (identity-init) chain and against the token readout, the
same probe applied to the one-hot prefix. A family holds in a seed if R² ≥ 0.8 and it beats both controls by 0.1. R²
is on held-out prefixes, pooled over the eligible positions (the count is in the table); cells are the median over 3
seeds with [min, max].

| grammar | chain learned | probe | positions | trained | identity init | token readout | holds in seeds |
|---|---|---|---|---|---|---|---|
| v8m2s2L4 | yes | affine | 7 | 0.97 [0.96, 0.97] | 0.25 [0.24, 0.31] | 0.50 [0.45, 0.52] | 3 of 3 |
| v8m2s2L4 | yes | projective | 6 | 0.97 [0.97, 0.98] | 0.27 [0.25, 0.33] | 0.57 [0.51, 0.60] | 3 of 3 |
| v8m2s2L4 | yes | softmax | 6 | 1.00 [1.00, 1.00] | 0.40 [0.38, 0.52] | 0.92 [0.86, 0.98] | 1 of 3 |
| v4m2s2L5 | yes | affine | 22 | 0.98 [0.97, 0.98] | 0.20 [0.19, 0.21] | 0.49 [0.46, 0.55] | 3 of 3 |
| v4m2s2L5 | yes | projective | 21 | 0.99 [0.99, 0.99] | 0.24 [0.23, 0.26] | 0.57 [0.55, 0.63] | 3 of 3 |
| v4m2s2L5 | yes | softmax | 21 | 1.00 [1.00, 1.00] | 0.28 [0.27, 0.30] | 0.80 [0.77, 0.83] | 3 of 3 |
| v8m4s2L4 | no | affine | 11 | 0.73 [0.69, 0.77] | 0.08 [0.06, 0.09] | 0.19 [0.16, 0.19] | 0 of 3 |
| v8m4s2L4 | no | projective | 9 | 0.74 [0.70, 0.78] | 0.08 [0.03, 0.08] | 0.13 [0.07, 0.20] | 0 of 3 |
| v8m4s2L4 | no | softmax | 9 | 0.89 [0.85, 0.92] | 0.08 [0.05, 0.09] | 0.24 [0.19, 0.25] | 3 of 3 |
| v16m4s2L4 | no | affine | 11 | 0.53 [0.52, 0.53] | 0.04 [0.03, 0.05] | 0.15 [0.14, 0.16] | 0 of 3 |
| v16m4s2L4 | no | projective | 4 | 0.67 [0.66, 0.68] | 0.04 [0.04, 0.06] | −1.32 [−14.40, 0.12] | 0 of 3 |
| v16m4s2L4 | no | softmax | 4 | 0.87 [0.84, 0.87] | 0.04 [0.04, 0.05] | 0.28 [0.27, 0.30] | 3 of 3 |

- On the two learned grammars the affine and projective probes hold in all three seeds, so both grammars support H_A.
  The softmax probe reaches 1.00 on v8m2s2L4 but fails its margin in two seeds there, because the softmax probe on
  the raw tokens already scores 0.92 and 0.98.
- The two m = 4 grammars are not judged, because the chain did not learn them. Descriptively, the softmax probe reads
  the belief there at 0.84 to 0.92 with both controls at about 0.30 or below, while the affine probe reaches 0.52 to 0.77
  (0.65 to 0.77 on the softmax probe's positions) and the projective probe adds 0.01 to the affine probe scored on
  the same positions (0.74 against 0.73, and 0.67 against 0.66). The softmax probe scores 0.15 to 0.21 above the affine probe on the same positions, but it is also
  the strongest family on raw tokens there, and its margin over its own token control (0.57 to 0.68) is only 0.02 to
  0.10 above the projective probe's in the four seeds where that control is stable. That is a weak hint of a
  log-probability code over an unnormalized belief, on grammars the chain did not learn. The projective and softmax
  rows of v16m4s2L4 rest on 4 positions.
- The projective probe on raw tokens is unstable on v16m4s2L4 (R² of −1.3 and −14.4 in two seeds). It is a control,
  and it did not affect any verdict: in those two seeds the projective probe on the chain is below the 0.8 floor
  (0.66 and 0.67), and the grammar is not judged. A control that fails low makes the margin easier to pass, so on
  these positions only the identity-init control is informative for the projective family.

**H_B: aligned against permuted tree, pair by pair.** Each pair shares the grammar's rules, the stream of training
sequences and the held-out prefixes, and differs only in the leaf order. The learning gap is the permuted tree's
excess loss minus the aligned tree's, and the aligned tree wins if the gap exceeds the tolerance (the larger of 0.001
nats and the aligned tree's range over seeds). The representation columns give the R² of the bilinear probe on the
two bonds under the root, [h_A, h_B, h_A ⊗ h_B], on held-out prefixes at the common positions past the midpoint. The
adjusted gap is the gap at the end of training minus the same gap between the two untrained trees, and a pair goes to
the aligned tree if it is at least 0.1 with the final gap positive.

| pair | excess aligned | excess permuted | gap | tolerance | learning | R² aligned | R² permuted | final gap | init gap | adjusted | representation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v8m2s2L4, seed 0 | 0.0010 | 0.0014 | 0.0004 | 0.0010 | tie | 0.997 | 0.972 | 0.026 | −0.093 | 0.119 | aligned |
| v8m2s2L4, seed 1 | 0.0010 | 0.0013 | 0.0003 | 0.0010 | tie | 0.998 | 0.971 | 0.027 | −0.181 | 0.208 | aligned |
| v8m2s2L4, seed 2 | 0.0013 | 0.0029 | 0.0016 | 0.0010 | aligned | 0.997 | 0.915 | 0.082 | −0.091 | 0.174 | aligned |
| v4m2s2L5, seed 0 | 0.0008 | 0.0013 | 0.0006 | 0.0010 | tie | 0.997 | 0.976 | 0.021 | −0.122 | 0.143 | aligned |
| v4m2s2L5, seed 1 | 0.0007 | 0.0027 | 0.0021 | 0.0010 | aligned | 0.995 | 0.896 | 0.099 | −0.064 | 0.162 | aligned |
| v4m2s2L5, seed 2 | 0.0008 | 0.0040 | 0.0032 | 0.0010 | aligned | 0.999 | 0.923 | 0.077 | −0.047 | 0.124 | aligned |
| v8m4s2L4, seed 0 | 0.0024 | 0.0696 | 0.0672 | 0.0010 | aligned | 0.850 | 0.473 | 0.377 | −0.049 | 0.425 | not used |
| v8m4s2L4, seed 1 | 0.0021 | 0.0502 | 0.0481 | 0.0010 | aligned | 0.872 | 0.557 | 0.314 | −0.053 | 0.367 | not used |
| v8m4s2L4, seed 2 | 0.0021 | 0.0390 | 0.0368 | 0.0010 | aligned | 0.901 | 0.667 | 0.234 | −0.021 | 0.256 | not used |
| v16m4s2L4, seed 0 | 0.0048 | 0.0956 | 0.0908 | 0.0011 | aligned | 0.687 | 0.455 | 0.232 | −0.007 | 0.239 | not used |
| v16m4s2L4, seed 1 | 0.0059 | 0.1569 | 0.1510 | 0.0011 | aligned | 0.667 | 0.364 | 0.303 | −0.028 | 0.331 | not used |
| v16m4s2L4, seed 2 | 0.0052 | 0.1308 | 0.1257 | 0.0011 | aligned | 0.684 | 0.426 | 0.258 | −0.009 | 0.268 | not used |

"Not used" marks pairs where the permuted tree did not learn; the definition leaves them out of the representation
measure because a lower probe score in a tree that did not learn the task says nothing beyond the learning measure.

- **Learning: 9 wins, 0 losses, 3 ties of 12 usable pairs (one-sided sign test on the 9 untied pairs, p = 0.002),
  median gap 0.020 nats.** A measure goes to an arm at 80% of usable pairs, here 10 of 12, so it goes to neither. The
  three ties are pairs in which both trees end within 0.001 nats of each other, all on the m = 2 grammars. On the
  m = 4 grammars the gap is 0.037 to 0.151 nats in all six pairs.
- **Representation: 6 wins, 0 losses, 0 ties of 6 usable pairs (p = 0.016), median adjusted gap 0.153, range 0.119
  to 0.208.** Fewer than the 9 usable pairs the rule requires, so untestable.
- **How to read the six wins.** In all six the final gap is below 0.1 (0.021 to 0.099): where both trees learned,
  the permuted tree's top bonds give R² 0.90 to 0.98 against 0.995 to 0.999 for the aligned tree. The pairs pass
  because the untrained permuted tree starts ahead (init gap −0.047 to −0.181), which the rule subtracts by design.
- Past the midpoint is the only regime the rule uses. The same count on the common positions up to the midpoint,
  reported with no verdict, is 1 win and 2 ties of 3 pairs (v4m2s2L5 only).

**The three arms side by side.** R² on held-out prefixes over the common positions (those with at least 10,890
distinct fit prefixes, the same set in every arm). The trees are read with the bilinear top-bond probe (1,088
features) and the chain with the affine probe on its state (78 features at T = 16, 100 at T = 32), so the probes are
not matched in size. The scrambled target is the same probe fit to a random 8-vector per distinct belief: it measures
how well the probe can fit an arbitrary labelling of the belief states. The next-token ceiling is the best any
function of the exact next-token distribution can do. The token readout column is the affine probe on the one-hot
prefix for every arm, which is why it repeats within a grammar. The size-matched control for the bilinear probe, the
token readout with cross-set products, exists among the learned grammars only on v4m2s2L5, where it scores 0.55 to
0.57 against the aligned tree's 1.00 (11 positions) and 0.66 to 0.77 against the permuted tree's 0.89 to 0.98 (13 to
15 positions). On v8m2s2L4 it could not be computed at any common position.

| grammar | arm | positions | trained | identity init | scrambled target | token readout | next-token ceiling |
|---|---|---|---|---|---|---|---|
| v8m2s2L4 | aligned tree | 3–5 | 1.00 [1.00, 1.00] | 0.62 [0.53, 0.62] | 1.00 [0.99, 1.00] | 0.55 [0.49, 0.60] | 0.89 [0.73, 0.90] |
| v8m2s2L4 | permuted tree | 3–5 | 0.97 [0.91, 0.97] | 0.71 [0.71, 0.72] | 0.97 [0.89, 0.97] | 0.55 [0.49, 0.60] | 0.89 [0.73, 0.90] |
| v8m2s2L4 | chain | 3–5 | 0.97 [0.96, 0.97] | 0.29 [0.24, 0.31] | 0.94 [0.91, 0.94] | 0.55 [0.49, 0.60] | 0.89 [0.73, 0.90] |
| v4m2s2L5 | aligned tree | 18–19 | 0.99 [0.99, 0.99] | 0.48 [0.46, 0.57] | 0.98 [0.97, 0.99] | 0.51 [0.47, 0.55] | 0.63 [0.62, 0.64] |
| v4m2s2L5 | permuted tree | 18–19 | 0.93 [0.91, 0.98] | 0.58 [0.57, 0.72] | 0.90 [0.85, 0.96] | 0.51 [0.47, 0.55] | 0.63 [0.62, 0.64] |
| v4m2s2L5 | chain | 18–19 | 0.98 [0.97, 0.98] | 0.20 [0.19, 0.20] | 0.95 [0.94, 0.95] | 0.51 [0.47, 0.55] | 0.63 [0.62, 0.64] |
| v8m4s2L4 | aligned tree | 9 | 0.85 [0.83, 0.88] | 0.13 [0.10, 0.16] | 0.76 [0.73, 0.78] | 0.19 [0.16, 0.20] | 0.92 [0.92, 0.92] |
| v8m4s2L4 | permuted tree | 9 | 0.55 [0.44, 0.65] | 0.19 [0.16, 0.20] | 0.29 [0.18, 0.40] | 0.19 [0.16, 0.20] | 0.92 [0.92, 0.92] |
| v8m4s2L4 | chain | 9 | 0.72 [0.68, 0.77] | 0.07 [0.06, 0.09] | 0.39 [0.35, 0.45] | 0.19 [0.16, 0.20] | 0.92 [0.92, 0.92] |
| v16m4s2L4 | aligned tree | 10 | 0.66 [0.64, 0.66] | 0.03 [0.03, 0.04] | 0.61 [0.61, 0.62] | 0.15 [0.14, 0.16] | 0.79 [0.79, 0.80] |
| v16m4s2L4 | permuted tree | 10 | 0.40 [0.35, 0.43] | 0.05 [0.05, 0.07] | 0.23 [0.20, 0.28] | 0.15 [0.14, 0.16] | 0.79 [0.79, 0.80] |
| v16m4s2L4 | chain | 10 | 0.53 [0.52, 0.54] | 0.04 [0.03, 0.05] | 0.28 [0.28, 0.30] | 0.15 [0.14, 0.16] | 0.79 [0.79, 0.80] |

- **The supported probes identify the belief state; they do not show its geometry.** On the two learned grammars the
  scrambled target scores within 0.06 of the real belief in every run of every arm (aligned tree 1.00 against 1.00
  on v8m2s2L4 and 0.99 against 0.98 on v4m2s2L5, by median). A position has few distinct beliefs there (7 to 50 on v8m2s2L4 and 3 to 30 on v4m2s2L5, over the
  positions eligible for the chain's affine probe), so a probe that tells the belief states apart can fit any
  labelling of them. The definition anticipated this. What the high R² shows is that
  the hidden state of an unseen prefix determines its belief through the stated probe.
- **The belief split, which holds out distinct beliefs, is negative on the learned grammars for every arm** (chain
  affine probe −0.74 and −0.37 by median against span ceilings of 0.31 and 0.58; full table in
  [`results/tables.md`](results/tables.md)). It rests on 4 to 7 positions with few beliefs each and is a weak test
  either way, as the definition says, but it gives no evidence that the code is affine in the belief.
- On the m = 4 grammars the margin over the scrambled target is larger for the chain (0.72 against 0.39, 0.53 against
  0.28) than for the aligned tree (0.85 against 0.76, 0.66 against 0.61). For the aligned tree on v16m4s2L4 the margin
  comes from the positions up to the midpoint: past it the scrambled target scores 0.70 against 0.67 to 0.69 for the
  belief.
- The aligned tree learned v16m4s2L4 to within 0.006 nats, yet its bilinear top-bond probe reads the belief at only 0.66,
  below the next-token ceiling of 0.79. Predicting near Bayes did not make the belief readable from the top bonds by
  this probe on that grammar.

**Aligned tree against chain (descriptive, no verdict).** By the same learning measure the aligned tree wins 9 of 12
pairs by more than the tolerance, with 3 ties (all three v8m2s2L4 pairs) and no losses, median gap 0.012 nats. On the six
pairs where both learned, the tree's bilinear probe is 0.012 to 0.037 above the chain's affine probe past the
midpoint (0.995 to 0.999 against 0.961 to 0.983), and all six count as ties, because the untrained tree's bilinear
probe starts 0.21 to 0.38 above the untrained chain's.

**Each tree's affine probe against its additive bound (descriptive).** An affine probe on two bonds that saw
disjoint sets of positions can reach at most the best fit of the form f(what one bond saw) + g(what the other saw).
Past the midpoint the aligned tree's affine top-bond probe sits close to that bound on the m = 2 grammars (0.83
against 0.87 on v8m2s2L4, 0.85 against 0.86 on v4m2s2L5, medians over 3 seeds) and further below it on the m = 4
grammars (0.58 against 0.69, 0.39 against 0.64). For the permuted tree the bound is 0.94 to 0.99 by median where it
could be computed (0.88 to 0.99 per seed). On v4m2s2L5 and v8m4s2L4 (0.97 to 0.99) that is the uninformative regime
the definition warned of; on v8m2s2L4 it is 0.88 to 0.97 against an affine probe at 0.65 to 0.68. It could not be
computed on v16m4s2L4.
The table is in [`results/tables.md`](results/tables.md).

**Figures.** All are for seed 0 except `summary.pdf` and `training.pdf`, which show every seed.

- [`figures/summary.pdf`](figures/summary.pdf): per grammar, one point per seed, the pooled R² on the common
  positions past the midpoint of each arm's probe with its identity-init, token and scrambled-target controls (top),
  and the final excess loss with the 0.02 line (bottom).
- [`figures/positions.pdf`](figures/positions.pdf): R² per prefix length for each arm's probes, with the token
  readout, the next-token ceiling and each tree's additive upper bound. Positions outside the common set are shaded
  and the dashed line marks T/2.
- [`figures/training.pdf`](figures/training.pdf): excess loss and the pooled R² of each arm's affine probe over
  training.
- `figures/geometry_<grammar>_<arm>.pdf` (12 files): the true beliefs on their top two principal components at four
  positions, next to the arm's probes, the shuffled-label control, the identity-init network and the raw principal
  components of the probed state. These use the sequence split, which lets test prefixes repeat fit prefixes, so
  their R² is not the held-out number of the tables.
- `figures/png/`: PNG copies of the figures embedded in [`NOTE.md`](NOTE.md), made by `arch_figures_png.py` with the
  same plotting functions.

## How it was done

- **Arms.** The aligned tree is a binary tree tensor network with bond dimension 32, one core per node, whose leaf i
  reads token i. The permuted tree is the same network with leaf i reading token perm[i], one seeded permutation per
  (grammar, seed), drawn so that no sibling pair of the grammar's tree are siblings in the model. In all 12 permuted
  runs no internal model node below the root covers a grammar node's exact leaf set. The chain applies one shared core to its
  state and the next token, h_t = W(h_{t−1}, token), and predicts x_t from U h_t. Its width (78 at T = 16, 100 at
  T = 32) is the one whose parameter count is closest to the tree's: 475,878 or 477,126 against 471,552 or 483,840 at
  T = 16, and 1,000,900 against 992,256 at T = 32. No arm has biases or nonlinearities, and all start from the
  identity init.
- **Task and training.** Every update of every arm uses the same 512 fresh sequences from a seeded stream and the
  mean cross-entropy over sequences and positions. The trees see each prefix with the rest masked; the chain reads
  the sequence once. Adam, peak rate 1e-3 with warmup and cosine decay to 0.1×, 30,000 updates.
- **Probes and scoring.** 60,000 analysis sequences per run. At each prefix length the distinct prefixes are split at
  random into fit and test, so a probe is scored only on prefixes it never saw. R² is 1 − ΣSSE/ΣSST over belief
  coordinates, pooled over positions. A position counts only if it has at least 10 × (features + 1) distinct fit
  prefixes. The rules, test sets, analysis sequences and splits are those of `rhm-belief-geometry` and
  `rhm-ttn-belief-geometry` for the same (grammar, seed).
- **Verdicts** are computed by `arch_summary.py` from the criteria fixed in [`EXPERIMENT.md`](EXPERIMENT.md) before
  the run. The tables above are printed by `arch_table.py`.

## How it was checked

- **Correctness checks before training, all passed on the worker:** the 28 RHM checks, the probe check, the tree
  check and the 17 checks of `arch_check.py` (permutations, causality of the chain, the additive fit, the softmax
  probe on a synthetic log-linear code, and the pairing rules on hand-made runs). CUDA was present.
- **Completeness:** 36 of 36 runs reached 30,000 updates with status `finished`, and all 36 were analyzed. Nothing
  was skipped, censored or rerun.
- **Shuffled labels:** the pooled R² of the shuffled-label probe is between −0.015 and 0.000 in all 36 runs, against
  the 0.05 limit that would indicate a probe bug.
- **The verdict reproduces:** rerunning `arch_summary.py` on the coordinator from the returned `analysis.json` files
  gives output identical to the worker's `results/summary.txt`.
- **Device:** analyses ran on the worker's CPU while training used the GPU, and the loop moves analyses to the GPU
  once training ends, which by its rule applies to the last two (one chain and one aligned-tree run of v4m2s2L5).
  The fits are float64 least squares on either device.
- **Fact-check:** an independent fact-checker compared this write-up with the result files and logs. What it changed
  is listed at the end of this section.

The pass confirmed the four tables and the verdicts and changed eight statements, none of them a verdict. The main
ones: the reading of the softmax probe on the unlearned grammars is now "a weak hint", because that probe is also the
strongest on raw tokens; "the loss was still falling at the end" became "over the last 10,000 updates", because one
run was flat over its last 1,000; a wrong argument about the unstable token control was replaced; the permuted
tree's additive bound is now given per seed; and the size-matched token control of the bilinear probe, which the
draft left out, is described under the comparison table.

## Not yet checked

- **A higher learning rate or a longer run for the chain and the permuted tree.** This decides whether "did not
  learn the m = 4 grammars" is a property of the architecture or of the budget, and with it whether H_A and H_B's
  representation measure can be tested on four grammars. The short CPU pilot in the definition had the chain ahead at
  3e-3, and the loss here was still falling over the last 10,000 updates. The 12 unlearned runs took 8 minutes (chain) and 16 minutes
  (permuted tree) each with four sharing the GPU, so one more rate or a doubled run for those 12, with its analysis,
  is about 1.5 to 2.5 hours on one worker, under $2.
- **More grammars with m = 2.** H_A needs three learned grammars and H_B's representation measure nine usable pairs;
  this run supplied two and six. One more grammar that all three arms learn would make both testable. A v = 8,
  T = 16 grammar costs under an hour of worker time for its 9 runs and their analysis, and a v = 16 one about 1.5 to
  2 hours, because its analyses took 9 to 29 minutes each here.
- **A probe of the same size in every arm.** The trees are read with 1,088 bilinear features and the chain with 78
  or 100 affine ones, so "the tree's belief is slightly more readable than the chain's" (0.995 to 0.999 against 0.961
  to 0.983) compares probes as much as networks.
- **The transformer overlay.** `arch_plot.py` accepts the `rhm-belief-geometry` results folder as a third argument
  and adds the transformer's last-layer probe to `positions.pdf`. The planning session makes that figure.

## Judgement calls

- **The overall result is recorded as "inconclusive".** H_A is inconclusive by its rule. H_B is "not supported" by
  its rule, but through a win count one pair short of the bar with no losses, and with its second measure untestable,
  which is not a clean negative. The alternative label, "not-supported", would describe H_B's formal verdict and
  misdescribe H_A. Both formal verdicts are stated above and in the definition's Outcome.
- **The 1.2 GB of projection files are kept out of git** (see Amendments in the definition). They are the inputs of
  the geometry figures only. The job that moved them on the worker reported exit code 1: all 72 files were moved, and
  the code came from a `git status` at the end of the command, which fails on the worker because its copy of the
  project is not a git repository.

## What this means and what to do next

At this learning rate and budget, matching the tree to the grammar's hierarchy decided whether the two harder
grammars were learned at all: the aligned tree reached the Bayes loss to within 0.006 nats on every grammar, and the
permuted tree and the chain did not get below 0.02 on the m = 4 grammars. Where all three learned, the belief of an
unseen prefix can be read from each of them at R² of about 0.90 or above, so the data give no sign that alignment is
needed for a readable belief once the task is learned; the aligned tree's advantage over the permuted tree there is
0.02 to 0.10 in R². The chain
result is the most favourable to its hypothesis and the least tested: two grammars of two, with probes that identify
belief states without showing their geometry.

The cheapest next step is the learning-rate or longer-run check on the 12 unlearned runs, because it settles whether
the m = 4 grammars can enter H_A and H_B at all.

## Where the files are

- `results/<grammar>_<arm>_s<seed>/analysis.json`: every metric of a run. `results/summary.json` and
  `results/summary.txt`: the verdicts. `results/tables.md`: the tables of this write-up. `results/checks.json`: the
  RHM checks.
- `results/<run>/proj_ckpt*.npz` (1.2 GB, not in git): the probe projections behind the geometry figures. The copy
  of record is `runs/rhm-belief-architectures/proj/<run>/` on the coordinator; `arch_plot.py` and
  `arch_figures_png.py` expect them under `results/<run>/`, where they have been copied.
- `runs/rhm-belief-architectures/` (not in git): checkpoints at updates 0, 300, 1000, 3000, 10000 and 30000,
  `result.json` with the loss history of each run, and the training and analysis logs.
