# Script for the RHM and tree tensor network deck, 2026-11-18

Notes on what to say over each slide of `slides.pdf`. This deck expands the first four slides of
`../2026-11-18-rhm`: the grammar, the architectures, sample complexity, the architecture controls, and the tests of
what the bonds hold. The numbers come from the experiment write-ups under `studies/rhm/`; each frame names its
experiment in a TeX comment.

## 1. The Random Hierarchy Model

- The RHM is a synthetic grammar. A root symbol expands into a pair of symbols, each of those expands again, and after
  L = 4 levels we observe 16 tokens.
- Every symbol has m = 4 productions, chosen uniformly. The four pairs that one symbol can produce are synonyms: they
  mean the same thing one level up. No pair belongs to two symbols, so a sequence has exactly one parse.
- Our standard grammar has 16 symbols per level and 16 root classes.
- Two tasks. "Root" classifies the root symbol from all 16 tokens, and its Bayes error is 0. "Last" predicts token 16
  from the first 15, and its Bayes error is 0.10 to 0.13 depending on the grammar.
- P* = n m^L = 4,096 is the number of training trees that deep CNNs need on this grammar (Cagnetta et al.). It's the
  reference for sample complexity on the next slides.

## 2. The tree tensor network

- The model is a binary tree with the same shape as the grammar: 16 leaves, three levels of fusion cores and a root
  core.
- Each leaf embeds its one-hot token with a matrix A_i, from dimension V = 16 to the bond dimension r. Each fusion
  core W is an r × r × r tensor that contracts its two children into the parent bond. The root core C maps the two
  top bonds to O = 16 logits.
- Every internal bond has dimension r. We use r = 16 by default and sweep 8 to 64. On the "last" task a mask token
  replaces x_16, so V = 17.
- There are no biases and no nonlinearities, so the logits are a multilinear function of the leaf embeddings.
- At r = 16 that's 65,536 parameters, and half of them sit in the eight lowest fusion cores.

## 3. Two controls change one thing in the tree

- Both controls keep the binary tree and r = 16, and each changes one thing.
- Tied tree: one embedding A for all 16 leaves and one fusion core per level, so the same fill means the same
  tensor. That's 256 + 3 × 4,096 + 4,096 = 16,640 parameters, a quarter of the untied tree. It asks whether
  position-specific cores buy anything.
- Permuted tree: the same untied cores, but leaf i reads token π(i). The red wires are the actual permutation of
  grammar 0, init seed 0. Each permutation was redrawn until no two grammar siblings are model siblings, and in all
  six of them no node of the model at levels 1 to 3 covers exactly the leaves of a grammar node. It asks whether
  the tree's advantage comes from matching the hierarchy.
- Every bond is r = 16, every physical leg V = 16, the output O = 16.

## 4. The chain matches the tree's parameter count

- The chain is a matrix product state read left to right: h_t = W(h_{t−1}, A e_{x_t}), with a learned start vector
  h_0, one W and one A shared by all 16 positions, and a linear read-out U of the last state.
- Width 40 is the width whose parameter count is closest to the untied tree: 40 + 640 + 64,000 + 640 = 65,320
  against 65,536.
- So the chain is the control for "a tensor network with as many parameters but no tree". It has four times the
  parameters of the tied tree.
- Not run: a chain reading the tokens in another order, or a chain with a separate core per position.

## 5. The tree learns the root between P* and 4P*

- Fixed training sets of N trees, nested within a grammar, three grammars per size, one init seed. Untied tree at
  r = 16.
- Left, the root task: median accuracy 0.084 at 2^10 and 0.152 at 2^12 = P*, against chance 1/16, then 0.99996 at
  2^14. So the threshold is somewhere between P* and 4P*, and this grid doesn't locate it more finely.
- Right, the last-token task: the KL to the Bayes predictor falls by a factor of 2 to 4 for each factor of 4 in N,
  from 2.45 nats at 2^10 to 0.012 at 2^20. There's no sharp step on this task.
- The dash-dot line is the same tree trained on fresh trees for 20,000 updates. On the last-token task the 2^20
  runs end below it, but that comparison isn't like for like: the online runs stopped at a fifth of the schedule.
- Why P* is the threshold, if asked. The grammar has 16 · 4^15 ≈ 1.7 × 10^10 distinct sequences, so a training set
  of 4,096 trees covers almost none of them and the network has to find the structure. What a bottom pair says about the
  class is averaged over about m^L paths spread across n classes, so that correlation scales like 1/√(n m^L), and
  the noise in estimating it from P samples scales like 1/√P. It becomes visible at P ≈ n m^L (Cagnetta et al.).
- If pushed: one (m, L) and three grammars show the same order as P*. They don't test the m^L scaling.

## 6. At N ≤ P* the tree memorizes its training set

- Root task, grammar 0, one curve per training-set size, light to dark with N. Dashed is the loss on the training
  set, solid is the loss on 20,000 held-out trees.
- At 2^10 and 2^12 the training loss goes to zero while the held-out loss climbs to 33 and 24 nats: the tree makes
  confident wrong predictions on new trees. Held-out accuracy ends at 0.09 and 0.28.
- From 2^14 up, the two losses fall together and accuracy reaches 1 within a few hundred updates.
- This is why the failures are read as too little data and not failed optimization: all 12 runs that fail (both
  tasks, N ≤ 2^12) fit their training set.
- Caveat: no weight decay or other regularization was tried, and both tasks ran at the top of the learning-rate
  grid.

## 7. On fresh data, only the matching trees reach Bayes

- Now fresh trees every update, so data isn't the limit. Six runs per architecture and task.
- Left, root: the tree and the tied tree pass accuracy 0.9 within about 180 to 540 updates. After 40,000 updates
  the permuted tree has levelled off at 0.60 to 0.72 and the chain is still climbing at 0.82 to 0.95.
- Right, last token: the four curves overlap for about 50 updates and have separated by 100. The tree and the
  tied tree end within 0.006 nats of the Bayes loss. The permuted tree and the chain level off above it.

## 8. Tying is free; permuting costs more than a chain

- The same 48 runs at their end. Left: updates until accuracy reaches 0.9 of the Bayes accuracy. Right: test error
  above the Bayes error on the same 50,000 test trees. Filled circles are the root task, open squares the last
  token.
- Tree against permuted tree, and tied tree against chain: the first wins 12 of 12 pairs on both measures. Those
  were the two pre-set comparisons.
- Tree against tied tree: the same final error in all 12 pairs. Speed flips with the task (tied is faster on root,
  untied on last token), and the root difference disappears once the different learning rates are accounted for.
- The permuted tree ends worse than the chain: 0.28 to 0.40 excess error on root against 0.05 to 0.18. That
  comparison wasn't pre-set.
- Caveat to say out loud: the permuted tree on both tasks and the chain on root ran at the top of their
  learning-rate grid, and their pilot losses were still falling there. A 10^-2 pilot hasn't been run.
- What these two slides can't tell us: whether the permuted tree lacks the capacity at r = 16 or has it and
  training doesn't find it.

## 9. What the bonds hold: six tests

- This is the map for the next five slides. A cell is one grammar at one width with three seeds, and a cell passes
  only if all three seeds pass.
- Three pass: lower subtrees transfer, the top bonds determine the Bayes belief, and a bond is close to a function
  of what the task needs from its subtree.
- Three fail: the bond isn't the grammar's upward message, the merge isn't the grammar's message update, and
  training on mixed masks doesn't change that.
- One line for the whole slide: the tree learns a sufficient code for the task, and that code is not the
  grammar's own message-passing algorithm.

## 10. Pretrained subtrees transfer to unseen productions

- Setup: hold out one of each class's 4 root productions, so 48 seen and 16 unseen. Pretrain on the seen ones,
  then train on the unseen ones only. Nine units, three grammars × three inits.
- Left: accuracy on the held-out productions during the second phase. Right: the first logged update at 0.9.
- With the subtrees frozen and only a fresh root core trained (blue), 0.9 comes after a median of 42 updates.
  Training everything from scratch (red) takes 148. An oracle handed the true level-1 symbols (teal) takes 36, so
  the learned bonds are almost as good as the true symbols.
- The two gray arms are the floors: frozen subtrees from another grammar, and untrained subtrees. Neither reaches
  0.9 in any unit.
- Two caveats. The unseen productions are new pairs of symbols that pretraining already saw on each side, so this
  shows per-side symbol codes are reusable, not that new conjunctions are recombined. And every crossing happens
  within the first 150 updates, where logged steps are 4 to 22 updates apart; scratch reads exactly 148 in every
  unit because the logged points near the threshold are 126 and 148.

## 11. The product of the top bonds gives the belief

- These trees are trained on next-token prediction at every prefix length. The target is the Bayes belief over the
  pending symbols given the prefix. The probe is linear in the two top bonds and their product.
- Rows are grammars, written v · m · L. Three seeds per row, r = 16 on the left and r = 32 on the right.
- Blue is the trained tree: 0.85 to 1.00 in 7 of 8 grammars at each width. The misses are 8 · 4 · 4 at r = 16 and
  16 · 4 · 4 at r = 32.
- Black ticks are the best fit by any function that is additive in the two halves, so the gap between tick and blue
  dot is what the product term buys. Gray is the same probe on the tree at initialization.
- The red crosses are the caveat. The same probe fits a scrambled target, a random vector per distinct belief,
  just as well. So this shows that the bonds linearly identify the belief state. It doesn't show that they lay the
  beliefs out in their own geometry.

## 12. Without the product, only the small alphabets pass

- First half of the sequence, where the right top bond sees only masks, so the probe is affine in the r
  coordinates of the left top bond.
- It passes in 2 of 6 grammars at each width: the two with v = 3 and v = 4, at 0.84 to 0.99. The grammars with
  v = 8 or 16 land at 0.43 to 0.77.
- Black ticks here are what any function of the exact next-token distribution could recover. In the two passing
  grammars the bond is 0.20 to 0.48 above it, so the bond carries more of the belief than next-token prediction
  alone requires.
- A likely reason for the failures, found after looking and on one seed: an affine map of r numbers can't fit
  beliefs that spread over more than r dimensions. Positions whose beliefs need far fewer components than r score
  high, and positions that need far more score low. The exact bound wasn't computed.

## 13. Bonds hold what the task needs, not messages

- Four pre-set measures, one per panel. Rows are the nine learned cells, r = 32 above and r = 16 below. Blue is
  the trained tree, gray the tree at initialization, the dashed line is the pass mark.
- Panel 1, does a probe read the grammar's upward message from the bond: 0 of 9 cells, median R² 0.08 to 0.75. In
  7 of 9 cells the bond scores below the same probe on the raw tokens (teal).
- Panel 2, is the bond a function of what the task needs from its subtree (the extended message and the number of
  observed leaves): 7 of 9 cells, with 0.81 to 0.96 of the bond variance explained in the passing runs, against
  0.08 to 0.41 at initialization.
- Panel 3, does merging the mean bonds of a valid child pair give the right parent: 3 of 9 cells.
- Panel 4, is the merge the grammar's update when both children are uncertain: 0 of 9 cells, median R² 0.17 to
  0.58, and it beats the best additive fit (black ticks) in 4 of 27 runs.
- Caveat on panel 2: the measure counts variance between prefix lengths as explained. A stricter version lowered
  it by 0.05 to 0.08 on pilot checkpoints, which would put one or two cells at the threshold.

## 14. Training on mixed masks does not repair the merge

- The idea: under prefix masks the tree never sees two uncertain children at once, so maybe the merge fails
  because it was never trained on that case. Three arms at r = 32: prefix masks, varied masks, and a half-and-half
  mix.
- Left: the merge test from the last slide. No run reaches 0.8 in any arm. Mixed is higher than prefix in 8 of 12
  runs, by a median of 0.12.
- Right: read-out mass of child pairs that the grammar forbids, relative to valid pairs. Exact message passing
  gives 0. Every arm sits around 1 or above, so forbidden pairs respond as strongly as valid ones.
- The experiment is recorded as inconclusive by its own rule: the mixed tree learned the varied task in only 2 of
  the 4 grammars, the two with m = 2. In the m = 4 grammars the varied loss was still falling when the schedule
  ended.
- Open markers are runs above the 0.02 nats/token gate on the prefix task.

## 15. Which check first?

- The learning-rate pilot is the cheapest and closes the main qualification on the architecture comparison. The
  write-up estimates 15 to 20 minutes on one GPU.
- The finer grid in N locates the threshold inside the factor of 4, and a sweep over m or L is what would test the
  m^L scaling.
- A longer schedule for mixed masks at m = 4 would separate "can't learn the varied task at r = 32" from "needed
  more updates".
