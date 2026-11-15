---
name: rhm-ttn-messages
title: Do a tree tensor network's bonds and merges implement the RHM's message passing?
status: done
gpu: NVIDIA RTX A6000,NVIDIA RTX PRO 4500 Blackwell,NVIDIA RTX PRO 4500 Blackwell Server Edition,NVIDIA RTX 6000 Ada Generation
count: 1
hours: 8
budget_usd: 10
disk_gb: 40
data:
secrets:
approved_utc: 2026-11-14T21:40:06Z
result: not-supported
finished_utc: 2026-11-15T05:18:53Z
writeup: studies/rhm/rhm-ttn-messages/README.md
---

# Do a tree tensor network's bonds and merges implement the RHM's message passing?

## Question

`rhm-ttn-belief-geometry` asks whether the beliefs about the *future* can be read from a TreeTN's top bonds. This
experiment asks the stronger, local question: does a TreeTN trained on RHM next-token prediction learn the hierarchical
inference algorithm of the grammar itself?

Exact inference on the RHM passes an upward message at every node u,

    λ_u(a) = P(observed leaves under u | Z_u = a),        λ_u(a) = Σ_bc P(Z_L = b, Z_R = c | Z_u = a) λ_L(b) λ_R(c),

with P(b, c | a) = 1/m when (b, c) is a production of a. The TreeTN merge z_a = Σ_bc W_abc zL_b zR_c has the same
bilinear form, with learned coefficients. So the grammar makes a concrete prediction about each bond (it holds its
node's message) and about each core (it is the production tensor), up to a change of basis and a scale per node.

That prediction needs one correction for the task the tree is trained on. When a node is partially observed, the token
to predict lies inside its span, and everything the output learns about the span passes through that node's bond. The
message λ_u is then not enough: the tree needs the extended message

    M_u(a, c) = P(observed leaves under u, next token = c | Z_u = a),        Σ_c M_u(a, c) = λ_u(a),

and the number of observed leaves, which tells it which token to predict. Two prefixes with the same λ_u and different
M_u must get different bonds in a tree at the Bayes loss. So "the bond holds λ_u" can only be true up to this extra
content, and the tests below are built around that.

Under the masked prefix task a node is fully observed (its message is a point mass on its symbol), fully masked
(all-ones), or partially observed (a genuine distribution over symbols). The partially observed case is what makes
this discriminating: the target then varies over the simplex, where symbol decoding (`studies/rhm/interpret.py`) only
has point masses.

## Hypothesis

All four are judged per (grammar, width), in grammars the tree learns (excess test loss < 0.02 nats/token in all
three seeds). Messages are defined up to scale, so the target is the normalized message λ̂_u = λ_u / Σ_a λ_u(a) and
the probe is projective (the message up to normalization; the model and fit of `studies/belief_common.py`).

A measure "closes the gap" over a control when it leaves at most half of the shortfall from 1 that the control
leaves: 1 − R² ≤ 0.5 × (1 − max(R²_control, 0)), and the same for ve and accuracy. A fixed margin could be out of reach
when a control is already high, which the grammar decides (in the pilots the identity-init tree's production-table
accuracy at level 1 was 1.00 and an additive ceiling reached 0.99); this form can always be met.

- **H1, bonds hold the partially observed messages.** At eligible nodes, the projective probe from a node's bond
  recovers λ̂_u of partially observed nodes on local prefixes it never saw, with pooled R² ≥ 0.8, and closes the gap
  over the best of: the same probe on the identity-init tree, and the one-hot token readout of the same local tokens.
  Where the token readout with all pairwise products is eligible, the bond probe must also close the gap over it on
  those same nodes.
  - Against: R² < 0.8, or the gap over a control is not closed.
  - Where a node has no more distinct partially observed messages than the probe has parameters per coordinate
    (r + 1), H1 tests whether the bond identifies the message, not how it codes it. At 20,000 sequences the reviewer
    counted 6 distinct messages per eligible node for v3m2s2L4, 12–23 for v4m2s2L5 and 31 for v8m2s2L4, against 51–174
    for v8m4s2L4 and 94–301 for v16m4s2L4. The verdict names the cells where at least half of the eligible nodes are of
    this kind.
- **H2, a bond keeps what the task needs and little else.** Among rows that have a synonymous partner, the grouping
  accounts for at least 0.8 of the whitened bond variance, closing the gap over the identity-init tree. Synonymous here
  means a different local prefix with the same extended message M_u and the same number of observed leaves (for a
  fully observed or masked node, M_u is λ_u).
  - Against: the bond varies as much between synonymous realizations as between different extended messages.
  - Grouping by λ_u alone is reported next to it, with no verdict. The first review measured that 12–33% of the
    same-λ_u pairs at one prefix length have different Bayes next-token predictions, and 4–32% differ in the number of
    observed leaves, so a tree that solves the task cannot reach ve = 1 on that grouping. A low ve for λ_u alone
    therefore does not separate "no hierarchical inference" from "the bond passes the task's larger message"; the
    extended grouping and the M1x probe below do.
  - This is a strict test. Whitening gives every direction of the bond the same weight, so a bond with spare
    dimensions that keeps token detail in low-variance directions fails it even if a probe reads the message well.
- **H3, the cores reproduce the production rules.** With each child's class-mean bonds as inputs, the learned core
  followed by the parent's decoder returns the right parent symbol for at least 95% of the grammar's valid child
  pairs, over parents of tree level ≥ 2, closing the gap over the identity-init tree.
  - Level-1 parents are reported separately and carry no verdict. Their merged "class-mean" bond is an actual training
    row that the decoder was fit on, and a level-1 node has only 1 + v + mv distinct bonds against r + 1 probe
    directions, so the probe interpolates: in the pilot the identity-init tree scored 1.000 there.
  - This is a consistency test: it must hold if bonds cluster by symbol and fully observed parents decode. The part
    that is not implied is the invalid pairs, where exact message passing gives a zero message. Two numbers are
    reported for them with no threshold, because training never shows the tree an invalid pair: the norm of the merged
    bond, and the probe's unnormalized read-out q of it, each relative to the valid pairs.
- **H4, the merge is the grammar's update on uncertain messages, where the product matters.** Child bonds are built
  as mixtures of class means, Σ_b λ_L(b) μ_b and Σ_c λ_R(c) μ_c, merged with the learned core and decoded. Over parents
  of tree level ≥ 2, the decoded message matches the exact normalized update with R² ≥ 0.8 over independently paired
  empirical messages, and with R² ≥ 0.8 on the both-informative pairs (left child partially observed, right child
  partially or fully observed), closing the gap there over both the identity-init tree and the additive ceiling.
  - In training the right child sees tokens only once the left child is fully observed, so a both-informative
    combination of observation cases never occurs. The claim about them is limited, though. The core is bilinear, so
    the merge of a mixture is the same mixture of merges of fully observed class means,
    W(Σ_b λ_L(b) μ_b, z_R) = Σ_b λ_L(b) W(μ_b, z_R). Training did constrain those terms for the consistent pairings.
    What it never constrained is the jointly inconsistent pairings (child symbols that no production joins), which the
    exact update sets to zero. So H4 turns on what the core does with invalid pairs and on the relative scale of the
    valid ones, which M3 reports directly.
  - The unseen combinations are split in two, because the first review found that about 70% of them have a masked left
    child. With a uniform left message the exact update is linear in the right message alone, so those rows involve no
    product (additive ceiling 0.95 in the reviewer's count, against 0.43–0.56 for the both-informative rows). The
    left-masked rows are reported, per level, with no verdict.
  - The additive ceiling is the least-squares fit of the exact update on [λ_L, λ_R], fit and scored on the same subset
    of pairs: the best a merge without the product could do there.
  - Level-1 parents are excluded, as in H3: their children are leaves, which are never partially observed, and their
    pairs are training inputs.
  - Against: R² < 0.8 on the both-informative pairs, or the gap over their additive ceiling or the identity-init tree is
    not closed.
- **Honest expectation.** The tree outputs logits through a softmax, and a multilinear network that passed exact
  messages would output probabilities, not logits. So exact messages are not the only solution, and a learned code
  need not be linear in the message. The task also asks a partially observed bond for more than its message (the
  extended message above), so the hand-built exact tree, which holds λ_u only, is a reference for the analysis code
  and not a solution of the training task. H1 and H2 can hold for any code that identifies the message (the
  scrambled-target control quantifies this: few distinct messages against r features). H3's valid pairs follow from
  clustering. H4 is the test that only the message algebra passes, and it is the one most likely to fail; M4a (below)
  says whether a failure comes from a code that is not linear in the message.
- **Pilot** (CPU, seed 0, 3,000 analysis sequences, on short-trained checkpoints from the sibling experiment's pilots;
  the full runs train about 15 times longer with 4 times the batch). The numbers are from the revised analysis:

  | measure | v3m2s2L4, r = 16, 2,000 updates | v8m4s2L3, r = 32, 3,000 updates |
  |---|---|---|
  | excess loss (nats/token) | 0.0037 | 0.0097 |
  | M1 partial messages, projective R² | 0.49 (init 0.26, token readout 0.66, scrambled 0.27) | 0.09 (init 0.06, token readout 0.13) |
  | M1 distinct partial messages per eligible node | 6 and 6 (a test of message identity) | 51 |
  | M1x extended message, projective R² | 0.50 (init 0.11, token readout 0.30) | 0.21 (init −0.05, token readout −0.13) |
  | M2 ve, extended grouping (criterion) | 0.59 (init 0.34) | 0.74 (init 0.02) |
  | M2 ve, message alone | 0.27 (init 0.22) | 0.58 (init −0.01) |
  | M3 accuracy, level ≥ 2 | 0.92 of 24 pairs (init 0.71) | 0.94 of 32 pairs (init 0.44) |
  | M3 accuracy, level 1 | 1.00 (init 1.00) | 0.99 (init 0.91) |
  | M3 invalid / valid, bond norm | 2.0 | 1.1 |
  | M3 invalid / valid, read-out mass q | 4.2 | 1.2 |
  | M4 all empirical pairs, level ≥ 2, R² | 0.10 (init −0.33, additive ceiling 0.61) | 0.61 (init 0.18, ceiling 0.51) |
  | M4 both-informative pairs, R² (criterion) | −0.04 (init −0.38, ceiling 0.57) | 0.51 (init 0.16, ceiling 0.45) |
  | M4 left-masked pairs, R² | −0.65 (ceiling 0.97) | 0.47 (ceiling 0.99) |
  | M4a code linearity, R² | 0.79 (init 0.80) | 0.68 (init 0.59) |

  - No hypothesis passes on these short pilots. H3 was the only one passing before the review, on a pooled accuracy
    that level 1 carried; at level ≥ 2 it is 0.92 and 0.94, under the 0.95 bar.
  - The extended grouping raises M2 as the review predicted (0.27 → 0.59 and 0.58 → 0.74), and in the v8m4s2L3 pilot
    the bond reads the extended message better than the message alone (0.21 against 0.09), with both far from 0.8.
  - The invalid pairs are not suppressed: their merged bonds and their read-out mass are larger than the valid pairs',
    where exact message passing gives zero. That is what holds M4 down.
  - The merge looks like the grammar's update only at the leaves: in the v3m2s2L4 pilot the mixture update over all
    empirical pairs scored R² 0.92 at level 1 (identity init 0.83), −0.00 at level 2 and 0.40 at level 3.
- **Descriptive:** the affine probe next to the projective one; the scrambled-target selectivity with the number of
  distinct messages; the probe to the extended message (M1x); invariance by observation case, and grouped by λ_u alone
  and by the normalized message; the invalid pairs' norm and read-out; the left-masked and seen subsets and Dirichlet
  pairs in M4; level-1 parents in M3 and M4; code linearity (M4a); all measures per tree level and over training;
  r = 16 against r = 32.

## Method

- **Grammars:** `v3m2s2L4` (3 symbols, so messages live on a triangle), `v8m2s2L4`, `v8m4s2L4`, `v16m4s2L4` (T = 16) and
  `v4m2s2L5` (T = 32). Seeds 0, 1, 2 set the rules and the init. All are in the `rhm-ttn-belief-geometry` grid with
  the same rules per (config, seed).
- **Widths:** r = 16 and r = 32, giving 5 × 2 × 3 = 30 runs.
- **Model and training:** exactly `rhm-ttn-belief-geometry/ttn_run.py`, called unchanged: the aligned untied TreeTN
  with identity init, trained on every masked prefix of 512 fresh sequences per update, Adam at peak LR 1e-3, 30,000
  updates, checkpoints at 0, 300, 1000, 3000, 10000 and 30000, test loss against the exact Bayes loss.
  - These 30 runs repeat the sibling's script, arguments and seeds, so they have the same init and data stream as 30
    of its 60 runs. They are retrained here on purpose: this experiment then depends on nothing the sibling has yet to
    produce. The cost is about 2–2.5 h of GPU time.
- **Rows** (`msg_analyze.py`): 20,000 analysis sequences (seed 50,000 + grammar seed). A bond depends only on its span's
  tokens and on how many are observed, its local prefix. Node (l, k), with span 2^l, is collected at the prefix lengths
  t = k 2^l + j for j = 0 .. 2^l (t ≤ T − 1, the training range): j = 0 is the masked case, j = 2^l the fully observed
  case, the rest the partially observed case. The exact messages are `grammar.likelihoods` with the suffix unknown.
  - The extended message M_u of a partially observed node takes one more upward pass per candidate next token (v
    passes per prefix length). Its rows sum over the next token to λ_u, which `msg_check.py` verifies.
  - The last node of each level contains the leaf that is always masked in training, so it has no fully observed case
    and cannot serve as a child in M3 and M4; the last leaf's token embeddings are never trained.
- **M1, decodable.** Per internal node (tree levels 1 .. L−1; cores are untied, so each node has its own basis):
  - Affine and projective probes from the bond to λ̂_u, fit on half of the node's distinct local prefixes and scored on
    the other half, by observation case. A case's R² is against the fit mean of that case.
  - Decoding: the projective probe fits q = Wz + c so that q / Σq matches λ̂. Messages are nonnegative, so a bond is
    decoded as max(q, 0) / Σ max(q, 0). For a true message code that equals q / Σq, and it always lies on the simplex.
    The bare ratio is unbounded when Σq comes near zero outside the fit set: in the first pilot it gave R² −5.9 on the
    identity-init tree. The bare ratio is still recorded for M1.
  - Eligible nodes: at least 10 × (r + 1) distinct fit local prefixes, and partially observed rows on both sides.
    Level-1 nodes (span 2) never qualify; they are covered by M3 and M4.
  - Controls at the final checkpoint: the same two probes on the one-hot local tokens with the mask; the affine probe
    on those plus all pairwise products, where that is eligible; the two probes on a scrambled target (an independent
    Dirichlet(1) vector per distinct message); the affine probe on shuffled fit targets. Checkpoint 0 is the
    identity-init tree.
  - The pairwise token readout is eligible at few nodes (in the reviewer's count, one node of v8m4s2L4, which could
    flip by seed, and level 4 of v4m2s2L5). It is compared with the bond probe pooled over those same nodes.
  - **M1x, the extended message (descriptive):** the same two probes from the bond to the normalized extended message
    M̂_u = M_u / ΣM_u, a v × v table, on partially observed rows, at the first and last checkpoint, with the one-hot
    token readout as control. It says whether the bond carries the task's larger message when λ̂_u alone reads poorly.
- **M2, invariance.** On whitened bonds (`interpret.whitener`, which makes distances gauge invariant): the mean squared
  distance between rows with different local prefixes and the same exact message, divided by the same for random
  pairs with different local prefixes; ve = 1 − that ratio. It is pooled over nodes with at least 10 × (r + 1)
  distinct local prefixes: with fewer points than bond dimensions, whitening maps any network's bonds to the same
  configuration (in the pilot the level-1 nodes gave the same ve to three decimals before and after training). The
  share of rows that have such a partner is reported.
  - Three groupings: the same extended message and number of observed leaves (the criterion), the same unnormalized
    λ_u, and the same normalized λ̂_u (both descriptive).
- **M3, production table.** For every parent whose two children have a fully observed case: the class-mean bonds of
  the children's symbols (the leaf embeddings for leaves) are pushed through the learned core, pair by pair, and decoded
  with the parent's projective probe fit on all its rows.
  - Valid pairs: accuracy of the decoded symbol, and decoded mass on the true parent.
  - Invalid pairs: the norm of the merged bond in the metric of the parent's own bonds, zᵀS⁺z with S = E[zzᵀ], which
    no gauge changes, as the ratio of medians invalid / valid.
  - The probe's unnormalized read-out q_bc = W z_bc + c of every pair: its total nonnegative mass Σ_a max(q_bc(a), 0)
    as the ratio of medians invalid / valid, and its coefficient of variation over the valid pairs. Both are 0 for
    exact messages. The merge is bilinear and q is affine, so the read-out of any mixture bond in M4 is exactly
    Σ_bc λ_L(b) λ_R(c) q_bc: these two numbers are what M4 depends on.
  - Pooled over parents of level ≥ 2 for the criterion, and over level-1 parents separately.
- **M4, update.** For the same parents: 8,000 pairs (λ_L, λ_R) drawn independently from the two children's empirical
  messages, and 8,000 from Dirichlet(1). Pairs whose exact update is zero are dropped and counted.
  - The empirical pairs are split by the children's observation cases: seen (left full or right masked: combinations
    that occur in training), left-masked (left masked, right informative) and both-informative (left partially
    observed, right informative). The last two never occur in training.
  - Each subset is scored on the second half of its own pairs against the mean of its first half, next to its
    additive ceiling fit on that first half.
  - Pooled over parents of level ≥ 2 for the criterion, per level, and over level-1 parents separately.
  - **M4a, code linearity:** the mixture Σ_b λ(b) μ_b decoded with the node's own probe against λ itself.
- **Reference and check** (`msg_check.py`): a tree built by hand that passes the exact messages (bond dimension v, leaf
  embedding = one-hot token, mask = all-ones, core = production tensor). Its bonds equal the exact messages, and the
  same M1–M4 code must return the ideal value of every measure, on every grouping and subset, also after a random
  gauge change and rescaling of every bond. A tree with random cores must fail M3 and M4.
  - What it validates is the analysis code. The exact tree is not a solution of the training task: it holds λ_u only,
    so its M1x score is below 1, which the check also asserts.
- **Figures** (`msg_plot.py`, PDFs from saved results only), for seed 0 of each (grammar, width):
  - `prodtable_<cfg>_r<r>.pdf`: the grammar's production table next to the one read from the trained cores and from
    the identity-init tree, with invalid pairs shaded by their norm.
  - `m4_<cfg>_r<r>.pdf`: decoded against exact update for the mixture bonds.
  - `triangles_<cfg>_r<r>.pdf` (v = 3): exact and decoded messages on the 2-simplex.
  - `summary.pdf` and `training.pdf`.
- **Verdict:** `msg_summary.py` applies the criteria and writes `results/summary.json`.

## Inputs

None. All data is generated from the seeded grammars. The code calls `ttn_run.py` from
`studies/rhm/rhm-ttn-belief-geometry/` and imports `studies/belief_common.py`, `studies/rhm/interpret.py` (the
whitener) and `plot.py` from `studies/rhm/rhm-belief-geometry/`, all read-only.

## Commands

From the project root on the worker:

```bash
set -euo pipefail
nvidia-smi
free -g
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name())"
OMP_NUM_THREADS=8 uv run --frozen python -m studies.rhm.checks "$EXP_DIR/results/checks.json"
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_check.py | tee "$EXP_DIR/results/msg_check.txt"
```

Training, four runs at a time on the GPU, r = 32 and the largest grammars first, as its own `orch exec` step:

```bash
set -euo pipefail
mkdir -p runs/rhm-ttn-messages
for r in 32 16; do
  for c in v16m4s2L4 v4m2s2L5 v8m4s2L4 v8m2s2L4 v3m2s2L4; do
    for s in 0 1 2; do echo "$c $r $s"; done
  done
done | xargs -P 4 -n 3 sh -c 'OMP_NUM_THREADS=4 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-belief-geometry/ttn_run.py "$0" --r "$1" --seed "$2" --out runs/rhm-ttn-messages > runs/rhm-ttn-messages/train_$0_r$1_s$2.log 2>&1 || echo "FAILED $0 r$1 s$2"'
```

Analysis, two at a time and r = 32 first, then the verdict and the figures:

```bash
set -euo pipefail
{ ls -d runs/rhm-ttn-messages/*_r32_s[012] runs/rhm-ttn-messages/*_r16_s[012] 2>/dev/null | xargs -P 2 -I{} sh -c 'r=$(basename {}); OMP_NUM_THREADS=8 PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_analyze.py {} "$EXP_DIR/results/$r" > runs/rhm-ttn-messages/analyze_$r.log 2>&1 || echo "FAILED $r"'; } || true
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_summary.py "$EXP_DIR/results" 30 | tee "$EXP_DIR/results/summary.txt"
PYTHONPATH=. uv run --frozen python studies/rhm/rhm-ttn-messages/msg_plot.py "$EXP_DIR/results" "$EXP_DIR/figures"
```

## Expected outputs

- `results/checks.json` and `results/msg_check.txt`.
- `runs/rhm-ttn-messages/<cfg>_r<r>_s<seed>/`: `result.json` plus 6 checkpoints, each 0.2 to 4 MB, with the training and
  analysis logs next to them. About 250 MB in total.
- `results/<cfg>_r<r>_s<seed>/`: `analysis.json` (every measure per node and pooled, at every checkpoint),
  `arrays_ckpt0.npz` and `arrays_ckpt30000.npz`. Under 100 MB in total.
- `results/summary.json` and `results/summary.txt`.
- `figures/*.pdf`: about 30 files.
- `README.md`, the write-up.

## Controls

- **Exact reference tree:** the hand-built message-passing tree scores the ideal value on every measure, with and
  without a random gauge, and a random-core tree fails M3 and M4. This validates the analysis code.
- **Identity-init tree:** checkpoint 0 goes through the same M1–M4 code. Its bonds are close to an additive function
  of the tokens.
- **Token readouts (M1):** the probes on the one-hot local tokens with the mask, and the affine probe on those plus
  all pairwise products where it meets the same eligibility rule. They see exactly the inputs the bond sees. The
  pairwise readout is compared with the bond probe on the same nodes.
- **Held-out local prefixes and eligibility (M1):** the probe is scored on local prefixes it never saw, at nodes with
  at least 10 × (r + 1) distinct fit local prefixes.
- **Scrambled target (M1):** a random vector per distinct message, with the number of distinct messages per node. It
  shows how much of an R² any labelling of the messages would get.
- **Shuffled labels (M1):** expect R² ≈ 0.
- **Additive ceiling (M4):** the best a merge without the product could do on the same pairs.
- **Both-informative pairs (M4):** combinations of observation cases that never occur in training and where the
  update needs the product, judged against an additive ceiling fit on those same pairs. The left-masked pairs, where
  the update is linear in one message, are reported separately.
- **Levels (M3, M4):** verdicts use parents of level ≥ 2; level 1, where the test is an in-sample fit, is reported
  separately.
- **Extended message (M2, M1x):** the grouping and the probe target that include what the task requires of a partially
  observed bond, next to the λ-only versions.
- **Learning check:** excess test loss over the exact Bayes loss.
- **Correctness checks before training:** `studies.rhm.checks` and `msg_check.py`.

## Success and failure criteria

Computed by `msg_summary.py` from `results/*/analysis.json` at the final checkpoint, pooled over nodes. A
(grammar, width) cell is judged only if all three seeds finished with status `finished`; otherwise it is censored.
Censored cells are listed separately from cells that did not learn.

- **Learned:** all three seeds have excess test loss < 0.02 nats/token.
- **H1 per cell** (needs an eligible node): in all three seeds, pooled over eligible nodes on partially observed test
  rows,
  - R²(projective) ≥ 0.8,
  - the gap is closed over max(R²(identity init), R²(one-hot token readout)), and
  - where the pairwise token readout exists, the gap is closed over it by the bond probe pooled over the same nodes.
  - The verdict lists the cells where H1 is a test of message identity: in some seed, at least half of the eligible
    nodes have at most r + 1 distinct partially observed messages.
- **H2 per cell:** in all three seeds, ve ≥ 0.8 on observed rows, grouped by the extended message and the number of
  observed leaves, and the gap is closed over ve(identity init).
- **H3 per cell:** in all three seeds, production-table accuracy ≥ 0.95 over the valid pairs of parents of level ≥ 2,
  and the gap is closed over the identity-init tree's accuracy there.
- **H4 per cell:** in all three seeds, over parents of level ≥ 2,
  - R² ≥ 0.8 over all empirical pairs,
  - R² ≥ 0.8 over the both-informative pairs, and
  - the gap is closed over max(R²(identity init, both-informative), R²(additive ceiling, both-informative)).
- **Overall, per hypothesis and per width, over learned and testable cells:**
  - supported if at least 75% of the cells support it;
  - not supported if at most 25% do;
  - partly supported in between;
  - inconclusive if fewer than 3 cells can be judged.
- **Descriptive, with no verdict:** everything listed as descriptive under Hypothesis.
- **Run failed:**
  - `studies.rhm.checks` or `msg_check.py` fails;
  - no CUDA;
  - fewer than 26 of the 30 runs finish both training and analysis;
  - or the shuffled-label R² exceeds 0.05 at an eligible node of any run, which would mean a probe bug.

## Time and cost

- **Setup and checks:** about 15 minutes. `msg_check.py` runs about 200 small L-BFGS fits.
- **Training:** 30 runs × 30,000 updates, the same runs as half of `rhm-ttn-belief-geometry`'s grid.
  - That experiment measured, with four runs sharing the GPU, 17 minutes for a T = 16, r = 32 run and about an hour
    for a T = 32, r = 32 run (its amendment of 20:15 UTC, 2026-11-14).
  - The r = 32 arm here is 12 T = 16 runs and 3 T = 32 runs: 12 × 17 + 3 × 60 = 384 run-minutes, about 1.6 h at four
    at a time. The r = 16 arm is assumed three to four times faster, about 0.5 h. Training: roughly 2–2.5 h.
- **Analysis:** per run, 6 checkpoints × 14 nodes (T = 16) or 30 nodes (T = 32), each with two projective fits on at
  most 60,000 rows (about a second each on the GPU, estimated), plus the controls and the extended-message probes at
  the first and last checkpoints, and v extra upward passes per prefix length for the extended messages (CPU, seconds
  to a few minutes per run). About 5–10 minutes per run, so 1.5–2.5 h with 2 in parallel. This is the least certain
  number.
- **Total:** about 4–5.5 h expected, with an 8 h cap.
- **Cost:**
  - Worst case: $1.09/h (RTX 6000 Ada, the most expensive GPU listed) × 8 h = $8.72, inside the $10 budget.
  - Expected: about $3–4 at $0.69–0.72/h.

## If something fails

- **A training or analysis run fails** (FAILED in the step output): read its log. May be decided without asking:
  - A bug in this folder's code: fix it and rerun only the failed runs, noted under Amendments. Files outside this
    folder belong to other experiments and must not be changed; copy what needs changing into this folder.
  - Out of memory in analysis: rerun that run alone (`-P 1`), noted under Amendments. A smaller analysis set is not
    allowed without asking, because it changes which nodes are eligible.
  - A run that diverges or collapses to the uniform output: report it as such, don't retune.
- **Time running short:**
  - After 4.5 h, stop starting new training runs, then analyze and plot what exists: the analysis needs up to 2.5 h,
    and the cap is 8 h. Training order puts r = 32 and the largest grammars first, so the r = 16 arm of the smallest
    grammars is the first to go.
  - If the analysis is the slow part, analyze r = 32 first, and stop starting new analyses at 7.25 h so the summary
    and figures are made before the cap.
  - Unfinished runs are reported as censored, never dropped silently.
- **Checks fail or there is no CUDA:** stop, delete the worker, report.
- **Must wait for the user:** any change to the grammar list, the widths, the model or the success criteria; any spend
  above the budget.

## What is not tested

- **Output-level patching.** Replacing a bond with a donor's and comparing the tree's next-token distribution with the
  exact Bayes posterior of the spliced prefix (`interpret.py` has the last-token version). In a tree, whole-bond
  patching equals running the tree on the spliced input, and a spliced prefix is often impossible under the grammar,
  where the posterior is undefined. It is left for a follow-up.
- **Transfer of subtree representations** when upper-level productions change: `rhm-transfer` covers it.

## Questions asked before launch

None needed. The user asked for the rest of the theory agent's list to run alongside the two experiments in flight
(2026-11-14: "might as well add in the rest of these"). The budget is the project's default worker budget ($10,
`orch.toml`).

## Review

experiment-reviewer, first pass: "needs revision", no P0, two P1s and four P2s, all in the analysis and criteria.

- **P1. H2 counted as synonymous many prefixes that the task requires the tree to tell apart.** The next token lies
  inside a partially observed node, so λ_u is not enough to predict it. The reviewer measured that 12–33% of the
  same-λ_u pairs at one prefix length have different Bayes predictions and 4–32% differ in the number of observed
  leaves, so ve = 1 was out of reach for a tree that solves the task.
  - Fixed:
    - `targets()` computes the extended message M_u(a, c) = P(local prefix, next token = c | Z_u = a), with v extra
      upward passes per prefix length, and groups by it together with the number of observed leaves.
    - H2 is judged on that grouping. The λ-only and normalized groupings stay as descriptive numbers.
    - The Question and Hypothesis sections say what a negative on λ alone does not show.
    - A probe to the normalized extended message (M1x) was added as a descriptive measure.
    - The definition says the hand-built exact tree validates the code and is not a solution of the training task;
      the check asserts its M1x score is below 1.
- **P1. H4's "unseen combinations" were mostly rows where the update has no product.** About 70% had a masked left
  child, where the exact update is linear in the right message (additive ceiling 0.95, against 0.43–0.56 for the
  both-informative rows), and level-1 parents supplied 47% of the rows.
  - Fixed:
    - The unseen pairs are split into left-masked and both-informative, and reported per level.
    - H4's criterion uses the both-informative pairs at parents of level ≥ 2, with the additive ceiling fit and scored
      on those pairs and R² measured against the mean of that subset.
    - "Never trained" is replaced by what is true: bilinearity makes a mixture's merge a mixture of fully observed
      merges, and what training never constrained is the jointly inconsistent pairings.
  - Also changed, beyond the finding: H4's first criterion (all empirical pairs) is now over parents of level ≥ 2 as
    well, for the reason in the P2 on H3; and each node draws 8,000 pairs instead of 4,000, so the both-informative
    subset has enough rows.
- **P2. H3's pooled accuracy was mostly an in-sample fit at level 1.** Fixed: H3 is judged over parents of level ≥ 2,
  and level 1 is printed separately.
- **P2. H1 rests on few distinct targets in three of the five grammars, and the pairwise token control was pooled over
  different nodes than the probe.**
  - Fixed: the pairwise readout is compared with the bond probe pooled over the same nodes, and enters H1 as its own
    gap criterion there. The summary prints the distinct partially observed messages per node and names the cells
    where H1 is a test of message identity.
- **P2. M4 is a function of the v × v pair table.** Fixed: M3 saves and reports the probe's unnormalized read-out of
  every pair: the mass on invalid pairs relative to valid ones, and its spread over the valid ones.
- **P2. The training-time estimate was contradicted by the sibling's measurement.** Fixed: the estimate uses the
  measured 17 minutes (T = 16, r = 32) and about an hour (T = 32, r = 32), giving 2–2.5 h of training and 4–5.5 h in
  total. The cut-off for starting training runs moved from 5 h to 4.5 h, and the analysis now runs r = 32 first.
- **The reviewer's suggestion to reuse the sibling's checkpoints was not taken.** The trees are retrained here so the
  experiment depends on nothing the sibling has yet to produce; the definition says the runs repeat its seeds.
- After the revision `msg_check.py` passes 38 of 38, and the pilot numbers above are from the revised code.

experiment-reviewer, second pass (a fresh reviewer, since the first was no longer available): both P1s and all four
P2s judged addressed in the definition and in the code, no new P0; "ready to run". Three optional P2s:

- **The analysis step would stop before the summary if one width had no run folders** (`ls` fails under
  `set -euo pipefail`).
  - Fixed: the listing ignores a missing width and the pipeline cannot abort the step; failed analyses are still
    echoed as FAILED.
- **M2's random pairs are drawn across prefix lengths, so variance between prefix lengths counts as explained.** On the
  pilot, drawing the random partner at the same number of observed leaves lowers ve from 0.64 to 0.56 and from 0.55
  to 0.50, and at identity init from 0.32–0.33 to 0.19–0.21.
  - Left as is: the identity-init control carries the same inflation, so H2 stays interpretable. The write-up must
    say that ve includes between-prefix-length variance.
- **The both-informative additive ceiling is higher for small vocabularies** than the 0.43–0.56 quoted: 0.76–0.88 on
  the v = 3 and v = 4 grammars.
  - Left as is: the gap-closing criterion handles it. The write-up must give the ceiling per cell.

## Amendments

- **2026-11-15 00:42 UTC, by the supervising session (nobody was available to ask): more analyses in parallel.**
  - Reason: the analysis is three times slower than estimated. The first two runs (`v16m4s2L4`, r = 16) were still on
    their last checkpoint 25 minutes after the step started, against the estimated 5–10 minutes per run. At that rate
    30 analyses two at a time take about 6 h and end after the 8 h cap (05:40 UTC).
  - Change: the number of analyses running at once is raised from 2 to 6, by signalling the running `xargs`
    (`SIGUSR1`), so nothing is restarted. Each analysis used 3.6 GB of the GPU's 32 GB and about 1.3 of 256 cores. The
    analysis set, the code, the seeds and the criteria are unchanged; only wall time is affected. An out-of-memory
    failure would be rerun alone, as "If something fails" already says.
  - Also found: the analysis command does not run r = 32 first. Both globs go to one `ls`, which sorts all its
    arguments, so the order is by grammar name with r = 16 before r = 32. It matters only if analyses are left
    unfinished at the cut-off; if that looks likely, the remaining runs will be restarted in the intended order
    (r = 32 first) and recorded here.
- **2026-11-15 01:50 UTC, authorized by the user (relayed by the planning session): budget and cut-offs relaxed, and a
  research note added.**
  - The budget ceiling is $15 instead of $10.
  - The time-based cut-offs under "If something fails" (no new training runs after 4.5 h, no new analyses after
    7.25 h) may be moved so that every planned run is trained and analysed. Training had already finished (all 30
    runs, at 2.6 h). The 7.25 h cut-off for starting analyses is lifted: analyses keep starting until all 30 are done.
  - The worker's hard time cap was 8 h (05:40 UTC). The planning session extended it to 10:00 UTC on 2026-11-15 and
    confirmed that at about 02:20 UTC (worst case $8.88 at $0.72/h, inside the $15 ceiling). The cut-off for starting
    analyses therefore moves from 04:55 UTC to 09:00 UTC, which leaves an hour for the summary, the figures and
    `orch finish`.
  - Added outputs: `NOTE.md`, a research note with the mathematics in LaTeX and the key figures embedded as PNGs
    (`figures/png/<name>.png`, about 200 dpi, next to the unchanged PDFs), fact-checked like the README.

## Outcome

Result: **not supported** (recorded 2026-11-15). The tree does not implement the grammar's message passing; its bonds are close to a function of the extended message. Detail in [README.md](README.md); the mathematics in [NOTE.md](NOTE.md).

- **Run not failed:** checks passed (28 of 28, and all 38 lines of `msg_check.py`), CUDA present, 30 of 30 runs trained and analysed, shuffled-label R² at most 0.011.
- **Learned:** 9 of 10 cells; `v16m4s2L4` at r = 16 did not learn (excess loss 0.043–0.057) and is not judged. No cell is censored.
- **H1 (bonds hold the partially observed messages):** not supported at either width, 0 of 5 cells at r = 32 and 0 of 4 at r = 16; per-cell median R² 0.08–0.75 against 0.8.
- **H2 (a bond keeps what the task needs and little else):** supported at both widths, 4 of 5 cells at r = 32 and 3 of 4 at r = 16; ve 0.81–0.96 in the passing runs, identity init 0.08–0.41.
- **H3 (the cores reproduce the production rules):** partly supported at r = 32 (2 of 5 cells), not supported at r = 16 (1 of 4).
- **H4 (the merge is the grammar's update):** not supported at either width, 0 of 5 and 0 of 4 cells; per-cell median R² on the both-informative pairs 0.17–0.58 against 0.8.
- **Cost and runtime:** about $5.26 and 7 h 18 m on one RTX PRO 4500 Blackwell at $0.72/h, inside the original $10 budget and 8 h cap (raised to $15 and 10:00 UTC during the run, see Amendments).
