# Follow-up: how large are the partial-parity directions in the runs that stall?

In the 12 runs that never learn, the 4-bit partial parities grow by a factor of 3 × 10³ to 8 × 10⁹ during training
but end as a small part of the bond, a median of 1.5 × 10⁻⁴ of its variance (range 2.9 × 10⁻⁶ to 3.0 × 10⁻²). That
is about as large as the same share in the runs that do learn, measured at their last checkpoint before the loss
drops, so
the stalled runs are consistent with a first stage that is in place and a second merge that never starts. The output
does not use these directions at all. This was measured on 2026-11-15 on the saved checkpoints, outside the
experiment's criteria.

**Why.** The write-up found that in the stalled runs the 4-bit partial parities become linearly decodable from the
whitened level-2 bonds while the 8-bit ones stay at chance. Whitening removes scale, so that probe cannot tell a
large direction from a negligible one that whitening inflates.

**Measure.** For a node with span S, let $\chi(x) = (-1)^{\sum_{t \in S \cap K} x_t}$ be its partial parity as a sign,
with K the k parity bits. With unbiased bits $\chi$ has mean 0 and is orthogonal to every other parity function, so

$$
\text{share} = \frac{\lVert \mathbb{E}[z\,\chi] \rVert^2}{\mathbb{E}\lVert z - \mathbb{E} z \rVert^2}
$$

is the fraction of the bond's variance along the partial parity, 1 for a bond that is the parity and nothing else.
The expectation is exact: all 2¹⁶ inputs are enumerated, so there is no sampling floor. The share is taken in the
model's own coordinates and is not gauge invariant, which is the point: it is the size the next core receives.
Reported as the median over the nodes that hold 4 (or 8) of the k bits; bias-0 main runs, k ∈ {4, 8, 16}, 36 runs.

| init scale ε | learned runs: 4-bit share at the last checkpoint before t | stalled runs: 4-bit share at the end |
|---|---|---|
| 0.01 | 4.0 × 10⁻¹⁰ to 5.8 × 10⁻⁵, median 9.5 × 10⁻⁸ (n = 4) | 2.9 × 10⁻⁶ to 1.5 × 10⁻³, median 1.6 × 10⁻⁵ (n = 5) |
| 0.1 | 8.6 × 10⁻⁶ to 4.9 × 10⁻⁵, median 4.5 × 10⁻⁵ (n = 3) | 1.9 × 10⁻⁵ to 4.2 × 10⁻³, median 1.5 × 10⁻⁴ (n = 6) |
| 0.3 | 4.5 × 10⁻⁴ to 3.3 × 10⁻², median 3.8 × 10⁻³ (n = 8) | 3.0 × 10⁻² (n = 1) |
| 1.0 | 2.1 × 10⁻² to 1.7 × 10⁻¹, median 8.6 × 10⁻² (n = 9) | none stalled |

- **Stalled runs** (12: five at ε = 0.01, six at ε = 0.1, one at ε = 0.3). The 4-bit share starts at 1.6 × 10⁻¹³ to
  5.2 × 10⁻⁶ and ends at 2.9 × 10⁻⁶ to 3.0 × 10⁻², a growth of 3 × 10³ to 8 × 10⁹ (median 7 × 10⁴). So the
  decodability is not whitening acting on a static direction: training amplified these directions by orders of
  magnitude. The 8-bit share ends at 7 × 10⁻¹³ to 5.7 × 10⁻⁴ (median 3.6 × 10⁻⁹).
- **Learned runs** (24). Their 4-bit share is at or near 1 at the end (0.993–1.000), and at the last checkpoint
  before the discovery time t it is still small, in the ranges above. Learning is abrupt in this measure too.
- **Comparison at the same ε.** At each ε where both exist, the stalled runs' final share has a median at least as
  large as that of the learned runs just before they learn, and the two ranges overlap. The comparison is rough: consecutive checkpoints are a factor of
  1.1–1.8 apart, so "the last checkpoint before t" understates the share at the moment learning starts, and for
  k = 4 the 4-bit parity is the whole target, not an intermediate.
- **Causal effect at the output: none.** In the saved interpretation files, patching the 4-bit partial-parity
  subspace, or the whole level-2 bond, from a donor changes the KL to the counterfactual by at most 2.3 × 10⁻⁵ nats
  in every stalled run, against 3.35–3.58 nats in the learned runs. The output of a stalled run is still uniform, so
  nothing upstream can be causal for it yet.

**Reading.** The stalled runs have built the 4-bit partial parities as real, growing directions, small in absolute
size, and have not combined them. That fits staged learning that is stuck at the next merge, and it does not fit
"nothing was learned". It is not evidence of a bottom-up order in the runs that learn, where every level appears
within one checkpoint interval.

**Check.** `partial_scale.py` asserts that a bond made of a 4-bit parity and one independent bit has share exactly
1/2 and that the sign function is exactly balanced over the enumerated inputs. Numbers per run and checkpoint are in
[`partial_scale.json`](partial_scale.json).
