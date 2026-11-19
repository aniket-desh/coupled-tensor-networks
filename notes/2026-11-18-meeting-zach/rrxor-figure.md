# RRXOR belief panel, to send to Zach

Figure: [`studies/hmm/hmm-belief-geometry/figures/pca_rrxor.pdf`](../../studies/hmm/hmm-belief-geometry/figures/pca_rrxor.pdf)
(PNG copy: [`figures/png/pca_rrxor.png`](../../studies/hmm/hmm-belief-geometry/figures/png/pca_rrxor.png)).
All numbers below are from [`hmm-belief-geometry/README.md`](../../studies/hmm/hmm-belief-geometry/README.md) and
[`belief-probe-controls/README.md`](../../studies/hmm/belief-probe-controls/README.md).

## Caption

**RRXOR beliefs read out of a transformer, a chain and two tree tensor networks (seed 0).** Each point is one test
sequence at one position, placed by a probe's output on the top two principal components of the true beliefs and
colored by the true belief; outlined circles are the mean output per distinct belief. Columns: ground truth; transformer
(affine probe on the last layer); chain (projective probe on its state); tree at r = 32 and r = 8 (projective probe on
the two top bonds, with their product after the midpoint). Top row: trained. Bottom row: the same kind of probe, refit on the
initial networks. The R² in each panel (0.94, 0.998, 0.98, 0.81 trained; 0.60, 0.30, 0.61, 0.24 initial) is over the
points shown, on a split by sequence in which test sequences share prefixes with the sequences the probe was fit on.
It is a picture of the fit, not a held-out-prefix score.

## What the panel supports, in three sentences

On prefixes the probe never saw (three seeds), the belief is recovered at R² 0.865–0.872 from the transformer
(positions 14 and 15 only) and 0.994–0.997 from the chain (positions 11–15), against 0.47 for the best function of
the exact next-token distribution, so both states carry the belief beyond what the model's own prediction determines.
An affine probe also fits a random vector assigned to each belief state at 0.81 (transformer, against 0.865–0.872
for the belief) and 0.88–0.92 (chain, against 0.97–0.98 for the belief with the affine probe and 0.994–0.997 with the
projective one), because RRXOR has only 17 distinct beliefs per position; these scores therefore show that the state
identifies the belief state, and they are not evidence that it lays the beliefs out in the simplex geometry. The r = 8
tree is weaker: its held-out score is 0.733 and 0.767 in two seeds, with a numerical failure in the third, below the
0.772 that a sum of a left-half and a right-half function reaches. At r = 32 the probe drawn here has no position
with enough distinct prefixes for a held-out score; an affine probe on reduced features scores 0.88–0.90, which in
two of three seeds is 0.002 short of the pre-set bar of halving that additive ceiling's error.

## If one line is enough

RRXOR, seed 0: probe outputs on the top two principal components of the true beliefs (trained above, initial network
below); panel R² uses a sequence split with prefix overlap, the held-out-prefix scores are 0.87 (transformer) and
0.995 (chain, projective probe) against a next-token-only ceiling of 0.47, and an affine fit to scrambled targets
at 0.81–0.92 means this shows belief-state readout, not belief geometry.

## Geometry, if he asks

The separate test of geometry is in `belief-probe-controls`. For the chain, 0.89–0.92 of the variance in the state's
top principal directions is a function of the belief, and the top directions alone give the belief at 0.95–0.97, so
the belief is the main structure of the chain's state. For the transformer neither holds: only 0.10–0.43 of the
variance in the last layer's top one or two directions is a function of the belief.
