# lang-evidence: evidence aggregation and event detection in a TreeTN

The questions were whether a tree tensor network (TreeTN) learns two computations that language processes need, and whether its bonds hold the small summaries those computations predict. Both are learned and both mechanisms show up causally: H1, H4 and H5 are supported. On the pre-registered measures, the neighbours' meeting level (H2) and leaf alignment (H3) made no difference. But the KL measure couldn't have shown one, and the speed measure was near its floor. Every run ends within 0.001–0.0044 nats of the Bayes-optimal predictor. For every masked-bigram tree run (r = 16, r = 32 and permuted) that is below the KL comparison's tie tolerance of 0.002 nats. Accuracy crosses the speed threshold within 90–300 of 30,000 updates (about 100 for the r = 16 trees). In the intermediate learning curves, the tree reaches diagnostic-KL levels later when the neighbours meet at the root than when they meet at level 2. That happens in 15 of 15 seed–level comparisons at r = 32 and in 14 of 15 at r = 16, where the other one is a tie. That lag was found after looking and isn't a pre-registered measure.

- **Masked bigram (H1, H4).** The task is a Markov chain over 16 symbols, with one leaf masked and the model predicting the masked symbol. Every untied r = 16 tree run reaches a final test KL of at most 0.0017 nats (maximum 0.00164; 9 runs; threshold 0.02), and the r = 32 tree ties it on final KL in all 9 pairs. In the r = 32 tree, patching only the subspace that carries the combined evidence brings the output from 8.8 nats away from the counterfactual Bayes posterior to 0.21 nats. A random subspace of the same size gets only to 4.0 (medians over 6 runs).
- **Reserved trigger (H5).** The target depends on whether a trigger symbol occurs anywhere in leaves 0–14, and on the current token in leaf 15. Patching a one-dimensional trigger-flag direction flips the prediction to the counterfactual in a median of 100% of affected pairs, against 0% for random directions. Moving the trigger within a span leaves that direction almost unchanged (median relative change 0.0008, against 0.47 at initialization). The flag patch fails, though, at the three nodes whose span also contains leaf 15. There the bond must hold the flag jointly with the current token, and only a whole-bond patch moves the output.
- **Meeting level and alignment (H2, H3).** Masking leaf 5 makes its neighbours meet at a level-2 node; masking leaf 8 makes them meet only at the root. All 3 seeds tie on both steps to threshold and final KL, so H2 is unresolved at n = 3. The tree and its leaf-permuted control tie on both measures in all 9 pairs. Post hoc, the root-meeting position (p = 8) lags p = 5 in the diagnostic-KL curves of both tree widths: in 15 of 15 seed–level comparisons at r = 32 and 14 of 15 at r = 16. The permuted and chain controls show no such lag (see Learning curves).

Cost: one A40 for 1.8 h, about $1.06 of the $12 budget. All 48 planned runs finished, none collapsed, and all 36 tree-type runs were interpreted.

## Evidence

### Learning curves

![curves](figures/curves.pdf)

[`figures/curves.pdf`](figures/curves.pdf): diagnostic KL to the Bayes posterior against update (log-log), one line per seed (n = 3 per mask position or arm). The trigger r = 32 arms (tree32, perm32) sit on a plateau about twice as long as the r = 16 tree and the chain. Against the r = 16 tree, the longer plateau reflects the LR: tree16 ran at 3e-3 against 3e-4, and in gradient-flow time it reaches threshold after tree32 in all 3 seeds. The chain ran at the same 3e-4 with no such plateau, so the r = 32 trees' lag behind the chain isn't an LR effect.

**A meeting-level lag in the curves (post hoc).** In the perm and chain arms the three mask positions overlap. In the tree and tree32 arms, p = 8 (root meeting) reaches each level of diagnostic KL later than p = 5 (level-2 meeting). Here are the first logged updates below each level, per seed (s0, s1, s2):

| Arm | KL level | p = 5 | p = 4 | p = 8 | p = 8 later than p = 5 |
|---|---|---|---|---|---|
| tree | 0.05 | 268, 225, 268 | 300, 230, 260 | 350, 300, 340 | 3 of 3 |
| tree | 0.02 | 990, 730, 790 | 1420, 660, 760 | 1710, 1060, 1270 | 3 of 3 |
| tree32 | 0.05 | 520, 440, 490 | 580, 450, 570 | 700, 660, 700 | 3 of 3 |
| tree32 | 0.01 | 4000, 2250, 2800 | 4626, 2650, 3686 | 6000, 4389, 5227 | 3 of 3 |

Over five levels (0.2, 0.05, 0.02, 0.01 and 0.005), p = 8 is later in 15 of 15 seed–level comparisons for tree32 and 14 of 15 for the tree; the exception is a tie at 0.01 in seed 0. In tree32, p = 4 (level-3 meeting) usually falls between the two. Neither perm (4 of 15) nor chain (3 of 15) shows the pattern. That's what you'd expect if meeting level is the cause, because the permutation reassigns meeting levels and the chain has none. I found this after looking at the curves and picked the KL levels then, so it's a lead for the next experiment, not a test of H2. The pre-registered measures miss it because accuracy crosses 0.9 × Bayes at about the point where the curves start to separate, and the final KLs are all below the tie tolerance.

### H1: the masked bigram is learned, and r = 32 doesn't lower the final KL

| Arm (masked bigram) | Peak LR | Final test KL, median (range), 9 runs | Updates to threshold t, median (range) |
|---|---|---|---|
| tree, r = 16 | 1e-3 | 0.0011 (0.0010–0.0016) | 94 (90–100) |
| perm, r = 16 | 1e-3 | 0.0014 (0.0010–0.0018) | 100 (94–110) |
| tree32, r = 32 | 1e-4 | 0.0015 (0.0010–0.0019) | 180 (170–210) |
| chain, width 40 | 3e-4 | 0.0035 (0.0023–0.0044) | 210 (189–290) |

Test KL is measured on 50,000 examples. t is the first logged update at which diagnostic accuracy reaches 0.9 × the Bayes accuracy (Bayes accuracy is 0.918–0.958 across instances). The criterion was that all 9 tree runs have KL ≤ 0.02 and that tree32 beats the tree in at most 1 of 9 pairs. All 9 tree runs pass, with a maximum of 0.00164, and tree32 against the tree is 0 wins, 0 losses and 9 ties. **H1 supported.** The capacity half of the criterion could only have come out this way, though. The tie tolerance is max(0.002, 10% of the larger KL), and every tree and tree32 final KL is below 0.002, so no pair could register a win. On speed, the tree reaches threshold first in updates, but tree32 does in gradient-flow time in all 9 pairs, because it ran at a tenfold lower LR. So read the t column next to the LR column. For context only, not part of the criterion: the sampled centered-logit rank across every cut is at most 16 for the seed-0 instances (`results/rank.json`).

### H2 and H3: meeting level and leaf alignment

| | p = 5 (meet at level 2) | p = 4 (level 3) | p = 8 (root) |
|---|---|---|---|
| t, median (range), 3 seeds | 94 (90–100) | 90 (90–94) | 100 (94–100) |
| final KL, median (range) | 0.00105 (0.00103–0.00139) | 0.00111 (0.00097–0.00136) | 0.00119 (0.00105–0.00164) |

The comparison between p = 5 and p = 8 ties on both measures in all 3 seeds, so **H2 is unresolved at n = 3**. Every t differs by at most 10 updates: two logged steps in seed 0 and one in seeds 1 and 2. That never exceeds the logging-interval margin, and it's far from the factor of 1.25 the rule also requires (against t = 100, a win needs t ≤ 80). The KL tie tolerance of 0.002 nats is larger than every one of these KLs, so the quality comparison was bound to tie. p = 5 has the lower raw KL in 2 of 3 seeds. In gradient-flow time (cumulative LR to threshold) the speed comparison is also 3 ties. This isn't evidence that meeting level doesn't matter: both pre-registered measures sit at their floor on this task, and the curves above show a consistent post hoc lag for p = 8.

H3 is descriptive. Tree against perm (the same r = 16 tree on a seeded leaf permutation) gives 9 ties on speed and 9 ties on KL. In gradient-flow time the speed count is 1 tree win and 8 ties. The KL ties are again forced by the tolerance, and the raw KL is lower for the tree in 8 of 9 pairs. Because the permutation moves the neighbours' meeting point both up and down, the tree and perm runs are also pooled by their recorded meeting level:

| Meeting level | Runs | t, median | Final KL, median |
|---|---|---|---|
| 1 | 1 | 94 | 0.00097 |
| 2 | 4 | 97 | 0.00113 |
| 3 | 5 | 94 | 0.00124 |
| root | 8 | 100 | 0.00151 |

Median KL rises with meeting level, but the largest gap (0.0005 nats) is a quarter of the tie tolerance, and this ordering was noticed after looking at the pooled table. It's a pattern to test with more seeds, not a finding.

Tree against the parameter-matched chain is descriptive. Counted in updates, the tree reaches threshold first in 9 of 9 pairs. In gradient-flow time the count is 3 tree wins and 6 ties, because the chain's piloted LR is 3e-4 against the tree's 1e-3. On KL the tree is lower in 5 pairs and tied in 4, which falls short of the 8-of-9 rule for a direction.

### H4: forward and backward evidence are carried up and combined (decided on the r = 32 tree)

![probes](figures/probes.pdf)

[`figures/probes.pdf`](figures/probes.pdf): probe R² (or accuracy) against update. Each panel has one line per tree level and seed, giving the median over that level's nodes (3 seeds per panel). The masked-bigram panels show the r = 16 tree, and the trigger panels show tree32. In the p = 5 forward-evidence panel, the probe at levels 2–3 (on the nodes that hold both neighbours) rises from 0.3–0.6 at initialization to 0.80–0.85 by update 1,330. The figure doesn't show the r = 32 tree's masked-bigram probes, which decide H4; they're in the table below.

The targets are centered log-evidence vectors, clipped at log 10⁻³: forward evidence log P(y | x_{p−1}), backward evidence log P(x_{p+1} | y), and their sum, the combined evidence (the centered Bayes logits). Probes are ridge regressions on the whitened bond, scored by held-out R².

| Criterion (tree32, final checkpoint) | Observed | Threshold | Holds |
|---|---|---|---|
| (a) one-sided evidence on bonds whose span holds one neighbour and at least one other non-mask leaf | 27 of 27 (run, node) cases have R² ≥ 0.9; the step-0 median is 0.87 | ≥ 90% of cases | yes |
| (b) combined evidence on every bond holding both neighbours | 9 of 9 cases have bond R² ≥ output ceiling − 0.05; bond R² 0.874–0.892 against output ceilings 0.845–0.863 | all 9 | yes |
| (c) patch only the combined-evidence subspace at the meeting node; KL to the counterfactual posterior, median over 6 runs | 0.21 (range 0.15–0.28), against 8.83 unpatched and 3.99 for a random subspace of the same size | < 0.5 × unpatched and < random | yes |

**H4 supported** (all three hold). In (b) the bond's R² exceeds the ceiling, the R² of the same ridge fit from the model's own centered output logits, in all 9 cases, by 0.02 to 0.04. The 32-dimensional bond holds slightly more of the clipped target than the 15 effective dimensions of the output express. In (c) the subspace patch moves the argmax to the counterfactual's in 86–92% of affected pairs, against 8–18% for random subspaces. A whole-bond patch reaches a KL of 0.0012–0.0019, so the subspace leaves about 0.2 nats of the counterfactual unexplained. The subspace is the fewest ridge directions holding 90% of the predicted variance: 9 to 11 of 32 dimensions here. p = 8 has no meeting node below the root, so (c) rests on the 6 runs with p = 5 and p = 4.

The r = 16 tree, reported the same way, also passes all three: (a) 27 of 27 cases, with a step-0 median of 0.53; (b) 9 of 9; (c) a median sub-patch KL of 0.41 against 8.83 unpatched and 2.10 for random subspaces.

**Added or carried side by side?** This was descriptive, with no decision rule. At the r = 32 meeting node, forward and backward evidence are each decodable at R² 0.90–0.97, and their sum at 0.87–0.89 (6 runs). At r = 16 the three are similar, at 0.84–0.91, and the forward evidence is slightly below the sum in 5 of 6 runs. So at r = 32 the bond doesn't compress the evidence to the sum alone: each side stays more decodable than the sum. We didn't test whether that's because the two vectors are stored separately or because, with only 3 likely successors per symbol, the sum often determines the neighbour pair.

### H5: the trigger flag is carried as one direction, except where it meets the current token

![strata](figures/trigger_strata.pdf)

[`figures/trigger_strata.pdf`](figures/trigger_strata.pdf): median final test KL (3 seeds per arm) by trigger stratum, meaning no trigger or the level at which the trigger's position first shares a subtree with leaf 15. Every arm and stratum sits between 0.0009 and 0.0031 nats. The L1 stratum is a trigger at leaf 14 only, about 3% of test examples, so it's the noisiest.

| Criterion (tree32, final checkpoint, internal nodes whose span holds background leaves; 14 nodes × 3 runs = 42 cases) | Observed | Threshold | Holds |
|---|---|---|---|
| (c) flag-subspace patch: fraction of affected pairs moved to the counterfactual's argmax | median 1.00 (random subspaces: median 0.00) | ≥ 0.8, random ≤ 0.4 | yes |
| (b) relative change of the flag subspace when the trigger moves within the span, triggered examples, level ≥ 2 (18 cases) | median 0.0008, against 0.47 at step 0; by level, L2 0.0006 and L3 0.051 | ≤ 0.1 and below step 0 | yes |

**H5 supported.** The median hides a consistent split. In every run, 11 of the 14 nodes move 100% of pairs, and the other 3 move at most 1% (0% in most runs). Those 3 are the internal nodes on the path from leaf 15 to the root (level 1, leaves 14–15; level 2, leaves 12–15; level 3, leaves 8–15), and the whole-bond patch moves 100% of pairs at all three. Away from leaf 15 a node only needs to pass on whether the trigger occurred, and one direction does that. Once the span holds the current token b, the downstream output is c(b) or d(b), so the bond has to hold the pair (flag, b), and the flag is no longer a separate direction. This split was found after looking. It holds in all 9 trigger tree runs (tree16, tree32 and perm32; in perm32 the three nodes are the ones whose permuted span holds leaf 15).

On the descriptive measures, the flag is decodable with accuracy 1.00 at both initialization and the end, against a majority-class chance of 0.93. This was expected: at most one trigger occurs, so the flag is linear in the one-hot tokens. The trigger's position within level-2 and level-3 spans also stays linearly decodable from the whole bond (median accuracy 0.999, against 0.98 at step 0 and a chance of 0.86). So the bond doesn't discard where the trigger sat; it keeps that information outside the flag subspace, which ignores it.

tree16 and perm32 meet both criteria the same way: (c) a median of 1.00, with random at 0.0015 and 0.00; (b) 0.0007 and 0.0008. perm32's seed-2 permutation keeps leaf 15 in place, and the background symbols are exchangeable, so that run solves the same task as tree32 seed 2.

### H6: trigger arms (descriptive)

| Trigger arm | Peak LR | t (3 seeds) | Final KL (3 seeds) |
|---|---|---|---|
| tree16 | 3e-3 | 110, 120, 120 | 0.0020, 0.0018, 0.0029 |
| tree32 | 3e-4 | 230, 280, 268 | 0.0013, 0.0012, 0.0014 |
| perm32 | 3e-4 | 240, 300, 268 | 0.0013, 0.0012, 0.0013 |
| chain, width 78 | 3e-4 | 100, 100, 100 | 0.0010, 0.0012, 0.0012 |

All KL comparisons are ties. tree16 reaches threshold before tree32 in updates in all 3 seeds, but after it in gradient-flow time in all 3, so the update-count gap comes from the tenfold LR difference. The chain is ahead of tree32 on both clocks.

## How it was done

- **Processes.** These are generated on the fly with V = 16 symbols, and each run's process instance is seeded by its seed (0, 1, 2); see `studies/lang/processes.py`.
  - In the masked bigram, each symbol has Dirichlet(1) weight on 3 random successors plus a floor of 0.02/V. Leaf p ∈ {5, 4, 8} shows a mask token, and the target is x_p.
  - In the trigger process, leaves 0–14 are uniform. With probability 1/2 one of them is the reserved trigger symbol. The target is c(x_15) if the trigger is present and d(x_15) otherwise, with 5% label noise.
- **Models.** The masked-bigram arms are an untied 16-leaf binary TreeTN at bond r = 16 and r = 32, the same r = 16 tree on a seeded leaf permutation with no adjacent pair kept as siblings, and a chain of width 40, parameter-matched to the r = 16 tree. The trigger arms are trees at r = 16 and 32, the permuted r = 32 tree, and a chain of width 78, matched to r = 32. All use identity initialization with ε = 0.1.
- **Training.** Each run takes 30,000 Adam updates on online batches of 256, with warmup and a cosine schedule. Arms share data streams within a process and seed, so comparisons are paired. The peak LR was piloted per (process, arm) on seed 9 over {1e-4, 3e-4, 1e-3, 3e-3}; the choices are in `runs/lang-evidence/lr.json`.
- **Interpretation.** At every checkpoint of every tree-type run, `studies/lang/interpret.py` runs three tests:
  - Ridge or logistic probes on the whitened bonds.
  - An invariance test: change the surface tokens inside a span while keeping its summary fixed, then measure the relative change in the probe subspace.
  - Interchange patching: put a donor example's subspace into the recipient's bond and compare the output with the Bayes posterior of the spliced input.
- **Rules.** The criteria and comparison rules were fixed in advance; see [`EXPERIMENT.md`](EXPERIMENT.md). Numbers come from [`results/analysis.json`](results/analysis.json). The per-node spreads and the gradient-flow comparisons quoted above were computed from `runs/lang-evidence/*/interpret.json` and `result.json` with the helpers in `studies/lang/analysis.py`.

## How it was checked

- `studies/lang/checks.py` passed 47 of 47 checks on the worker before training ([`results/checks.json`](results/checks.json)). They include the masked-bigram Bayes posterior against brute-force enumeration (max difference 7.8e-16), the generated data against the exact Bayes distribution for each process, and whole-bond patching equal to splicing in both the plain and permuted trees.
- The health check shows 48 of 48 runs finished, 0 collapsed and 36 of 36 interpreted, so the "more than a quarter unfinished" override did not apply.
- I computed the per-node values behind the H4(c) and H5(c) medians from the interpretation files. That is how the leaf-15 split in H5 came up. Without it, the median of 1.00 would read as uniform success.
- Because the arms were piloted to different LRs, I recomputed every speed comparison in gradient-flow time, the project's convention. This changed four descriptive readings. The tree's 9-of-9 speed advantage over the chain became 3 wins and 6 ties. Tree against perm went from 9 ties to 1 tree win and 8 ties. tree32 against the tree reversed, from 9 tree wins in updates to 9 tree32 wins. tree16's speed advantage over tree32 on the trigger also reversed. No criterion uses speed across LRs, so no verdict changed.
- The KL tie tolerance, max(0.002, 10%), is larger than every tree, perm and tree32 final KL (the maximum is 0.00194). So the H1 capacity comparison, H2 quality and H3 tree-against-perm quality were bound to tie, whatever the models did. The definition's review didn't catch this. I report the raw directions next to the ties, but they carry no pre-registered weight.
- An independent fact-checker compared the first draft against the result files and source code. It found the post hoc meeting-level lag in the curves, which the first draft had missed (it said the mask positions were indistinguishable). It also found the forced KL ties and the two extra speed readings that flip in gradient-flow time, a wrong description of the probes figure, and several small numerical overstatements. All of these are corrected above. A second pass on the revision found five more overstatements, also corrected.

## Not yet checked

- **Meeting level with a harder task or more seeds.** The p = 8 lag in the curves should be tested as a pre-registered measure: steps to a fixed diagnostic-KL level, chosen before the run, rather than steps to 0.9 × Bayes accuracy, with a KL tie tolerance scaled to the KLs actually reached. All 48 runs trained in about 1.3 h, an average of about 25 minutes per seed with consecutive seeds overlapping, so a rerun of the same grid costs about $1, and the cost scales with the number of seeds.
- **Separate or summed evidence.** Testing whether the r = 32 meeting node stores forward and backward evidence separately would take a patch of only the forward subspace, holding the backward one fixed. The saved checkpoints are enough for it, so no retraining is needed.
- **Two pilot LRs at the edge of the grid.** The r = 32 masked-bigram tree chose 1e-4 and the r = 16 trigger tree chose 3e-3. A wider grid could change their speeds, but not the KL verdicts, which are ties at the floor.

## Judgement calls

- **Pilot LRs at the grid edge were kept.** The definition doesn't act on a grid-edge pick, and no arm collapsed, so nothing was rerun. Setup and all 32 pilots took about 7 minutes, so a wider grid would have been cheap, but it could only affect descriptive speed comparisons.
- **Subspace sizes differ from the definition's estimate.** The definition expected the combined-evidence subspace to be about 15 dimensions (15 of 16 at r = 16 and about half of the bond at r = 32). The code takes the fewest ridge directions holding 90% of the predicted variance, capped at half the bond, which gives 8 of 16 at r = 16 and 9–11 of 32 at r = 32. So the r = 16 random-subspace control is a real control too. H4 is still decided on r = 32, as the definition says.
- **The H5(c) median was read as written.** The flag patch fails at 3 of 14 nodes per run. The definition's median criterion passes regardless, and the verdict is reported as supported, with the split stated next to it rather than as a reason to downgrade.

## What this means

A TreeTN learns both computations, and the bonds hold the predicted summaries in a form that causally drives the output. Evidence from the two neighbours travels up as one-sided log-evidence and is available, summed, at the node where they meet. The trigger travels up as a one-dimensional flag until it reaches a subtree containing the current token, where it joins the token in a joint code. On the pre-registered measures, neither task separated the tree from its permuted control or showed an effect of meeting level. Those measures were at their floor: accuracy crossed the threshold within a few hundred updates, and the KL tolerance exceeded the KLs. Meeting level showed up only as a post hoc lag in the intermediate learning curves: when the neighbours meet at the root, the tree takes longer to reach each KL level. That lag is the thing to pre-register next.

Files: [`EXPERIMENT.md`](EXPERIMENT.md) (definition, criteria and outcome), [`results/`](results/) (`analysis.json`, `curves.json`, `interp.json`, `checks.json`, `rank.json`), [`figures/`](figures/) (made by [`plot.py`](plot.py) from `results/` only). Run directories with checkpoints and interpretation files are in `runs/lang-evidence/` (0.73 GB, not in git).
