# Follow-up: does the transformer's probe read the belief's geometry, or only which belief state it is?

The experiment's main result is that an affine probe on the transformer's last layer recovers the exact belief state
on prefixes it never saw (R² ≥ 0.8 in all three seeds in 10 of the 12 grammars that were learned and testable, see
[../README.md](../README.md)). This follow-up asks what that R² reflects. On five grammars, one seed each, the same
probe fits a random relabelling of the belief states almost as well as the beliefs themselves (R² 0.70–0.96 against
0.84–0.97), and it does not extrapolate to beliefs it was not fit on (R² at most 0.35 where 1.0 was attainable). So in
these runs the high R² mostly shows that the last layer linearly identifies which belief state a prefix is in. It is
weak evidence that the layer holds the belief's geometry as an affine code. This is one seed per grammar on a third of
the experiment's analysis sample, all five grammars are ones that passed the main hypothesis, and none of it was part
of the pre-registered criteria.

## The evidence

All numbers are for the last block's output at the final checkpoint (50,000 updates), seed 0, on 20,000 analysis
sequences, at the positions eligible under the experiment's rule (at least 1,290 distinct prefixes to fit on). R² is
1 − ΣSSE/ΣSST pooled over those positions.

| grammar | eligible positions | belief, unseen prefixes | scrambled target, unseen prefixes | belief, unseen beliefs | scrambled target, unseen beliefs | ceiling | random init, unseen beliefs | positions in the belief split |
|---|---|---|---|---|---|---|---|---|
| v8m4s2L3 | 3 | 0.973 | 0.958 | 0.30 | −1.22 | 0.99 | −11.5 | 3 |
| v8m4s2L4 | 11 | 0.842 | 0.696 | −0.11 | −1.80 | 0.96 | −0.02 | 10 |
| v8m2s2L4 | 7 | 0.941 | 0.920 | −2.2 | −2.31 | 0.27 | −24 | 4 |
| v4m2s2L5 | 21 | 0.940 | 0.917 | −1.5 | −3.08 | 0.80 | −468 | 4 |
| v3m2s2L5 | 21 | 0.915 | 0.862 | −63 | −64.1 | 0.77 | −25,000 | 4 |

Grammar names read vVmMsSLL: v symbols per level, m synonymous productions per symbol, branching s, depth L.

What the columns are:

- **Belief, unseen prefixes.** The experiment's own measure: fit the probe on half of the distinct prefixes at a
  position, score it on the other half. This sample reproduces the experiment's 60,000-sequence numbers for the same
  five runs: 0.973, 0.842, 0.941, 0.940 and 0.915 here against 0.973, 0.844, 0.941, 0.941 and 0.916 there.
- **Scrambled target, unseen prefixes.** The same probe and split, but each distinct belief is replaced by an
  independent random 8-dimensional vector. A probe that can tell the belief states apart can fit this target, whatever
  the beliefs' geometry. The gap between the two columns is what the geometry adds: 0.015 to 0.15 pooled. Per position
  it runs from −0.04 to 0.32, and the scrambled target scores at or above the belief at 12 of the 63 positions.
- **Belief, unseen beliefs.** The distinct beliefs at a position are split at random into a fit half and a test half,
  so the probe must predict belief vectors it never saw. Only positions with at least 20 distinct beliefs are used.
  - The **ceiling** is the test beliefs projected onto the affine span of the fit beliefs, which is the best any probe
    fit on the fit beliefs could do.
  - An activation that is an exact linear image of the belief scores at or near the ceiling. A code that only
    identifies the state scores below zero. This was checked on synthetic activations (see "How it was checked").
  - The **scrambled target on this split** scores below zero in every grammar, as it must: a probe cannot predict a
    random vector for a state it never saw.

**In v8m4s2L4, the gap over the scrambled target is largest at the positions with the most distinct beliefs.** The
three positions with 193 to 356 distinct beliefs (t = 7, 11, 13), more than the probe's 129 parameters per output,
give belief R² 0.60–0.66 against 0.28–0.46 for the scrambled target. The seven positions with 37 to 64 beliefs give
0.84–0.99 against 0.58–0.99. So in this grammar the scrambled target is fit worst where the distinct beliefs
outnumber the probe's parameters (a gap of 0.16 to 0.32, against −0.004 to 0.26 at the other seven positions), and the
belief probe itself is then below 0.8. The other four grammars have no position with more than 59 distinct beliefs,
and their largest gaps (0.21 and 0.16, in v3m2s2L5) are at positions with 18 and 9 beliefs, so the number of beliefs
does not explain the gap on its own. This pattern was noticed after looking at the per-position numbers; it is
exploratory.

**The belief split gives little support for an affine code, and it is a noisy measure.**

- Six positions have a ceiling of 0.99 or more: t = 6 and 7 of v8m4s2L3, t = 12, 14 and 15 of v8m4s2L4, and t = 27
  of v3m2s2L5, with 21 to 44 distinct beliefs.
  - At the first five the trained network scores −0.48 to 0.35 and the random-init network −19 to −0.02. At three of
    them the trained network scores 0.33 to 0.35, above the random-init network there and above the below-zero score a
    state-identity code gets in the synthetic check.
  - At t = 27 of v3m2s2L5 the trained network scores −119 and the random-init network 0.00.
- Positions with a lower ceiling also swing to large negative values (−55 at t = 19 of v3m2s2L5, −6.7 at t = 9 of
  v8m2s2L4). A probe with 129 parameters is fit there on about 10 to 30 distinct targets and then asked to
  extrapolate, which is the likely reason; this was not tested.
- In v8m2s2L4, v4m2s2L5 and v3m2s2L5, and in the random-init column, the pooled numbers are dominated by such
  positions, so they show only that the probe does not extrapolate; they do not measure by how much.
- **The trained network does not extrapolate better than the random-init one overall.** The random-init column looks
  far worse, but its pooled numbers are set by single positions (−109,091 at t = 13 of v3m2s2L5, −1,424 at t = 23 of
  v4m2s2L5, −19 at t = 7 of v8m4s2L3). Per position the trained network beats random init at 9 of the 25 belief-split
  positions, and in v8m4s2L4 random init is better pooled (−0.02 against −0.11).
- In v8m2s2L4 and v3m2s2L5 the belief does no better on this split than the scrambled target (−2.2 against −2.31, and
  −63 against −64.1).

## How it was done

`controls.py` in this folder loads a run's first and last checkpoints from `runs/rhm-belief-geometry/<grammar>_s0/`,
samples 20,000 sequences with the experiment's analysis seed, computes the exact beliefs with the experiment's code
(`beliefs.py`), and fits the experiment's affine least-squares probe (`analyze.fit_predict`). The per-position numbers
are in `controls_<grammar>_s0.json`. The split seeds and the scrambled-target seeds are those of the tree tensor
network experiment (`rhm-ttn-belief-geometry`), which computes the same two controls, but that experiment uses 60,000
sequences, so the held-out sets are not identical.

## How it was checked

- **The belief split separates the two codes it is meant to separate** (`synth_check.py`). On exact beliefs of
  v8m4s2L3 with synthetic 128-dimensional activations, over six random seeds, a random linear image of the belief
  scored 0.80–0.82 at t = 5 (ceiling 0.854) and 1.000 at t = 7 (ceiling 1.000). A random code per distinct belief,
  which identifies the state and carries no geometry, scored −0.41 to −0.55 and −0.06 to −0.18.
- **The pipeline reproduces the experiment's numbers** on all five runs, to within 0.002 (listed above).
- **A fact-checking pass** compared 46 claims in the first draft of this note with the result files and the code. It
  found 12 that did not fully hold, and all were corrected. The main ones:
  - the "best posed" belief-split positions left out t = 27 of v3m2s2L5, the one where the trained network scores
    −119;
  - two explanations were stated as causes without a test;
  - the note said the full 60,000-sequence sample would make more positions eligible, which it would not;
  - the note said all three later experiments compute the same two controls, which two of them do not.
  The pass also asked for the trained-against-random-init comparison and the scrambled target on the belief split to
  be reported; both are above.

## Not yet checked

- **More seeds.** Every number is one seed. The experiment has seeds 1 and 2 for each grammar; a run here took a
  minute or two of CPU. The size of the scrambled-target gap could move.
- **The two grammars that failed the main hypothesis.** The v16m4s2L4 run was killed for memory on the coordinator
  (8 GB; some of its beliefs have 4,096 entries), with no log kept. v3m2s2L6 was not run. Both need a machine with
  more memory or a smaller sample.
- **The full 60,000-sequence sample,** which would score each position on three times as many sequences and make the
  held-out sets identical to the tree experiment's. It would not add positions in these five runs: the experiment's
  seed-0 analyses have the same eligible positions and the same distinct-belief counts.
- **A probe with fewer parameters.** The probe's 129 parameters per output exceed the number of distinct beliefs at 60
  of the 63 positions, which is the likely reason the scrambled target is fit well. Probing a low-dimensional
  projection of the layer would test the geometry more directly. Not designed yet.

## Judgement calls

- **20,000 sequences, not 60,000,** to fit the coordinator's memory and two CPU cores. The number of eligible
  positions still matches the experiment's table for these five grammars (3, 11, 7, 21 and 21, where the experiment
  has 3, 11, 7, 21–22 and 21), but each position is scored on fewer held-out prefixes.
- **Five grammars, chosen to cover depth 3 to 5 and 3 to 8 symbols,** not all 13 testable ones. All five passed the
  main hypothesis.

## What this means

The experiment's H1 stands as stated: the last layer linearly determines the belief on unseen prefixes, far better
than a random-init network or a token readout. What this follow-up adds is that, in these five runs, "linearly
determines the belief" is mostly "linearly separates a small set of belief states": 3 to 64 at 60 of the 63 positions,
with a median of 14.5 (16 over all 63). The Simplex claim that the residual stream holds the belief *geometry* needs evidence that a
state-identity code cannot produce, and on these runs that evidence is weak: a pooled gap of 0.015 to 0.15 over a
scrambled target, and little extrapolation to unseen beliefs.

The tree tensor network experiment and `rhm-belief-architectures` compute both controls with the same seeds on 60,000
sequences. `hmm-belief-geometry` computes the scrambled target only, and `rhm-ttn-messages` scrambles its messages
with a different construction and has no belief split. So the scrambled-target comparison can be made across all of
them, and the belief split only across the RHM belief experiments.
