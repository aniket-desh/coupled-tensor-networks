# Script for the RHM update, 2026-11-18

Notes on what to say over each slide of `slides.pdf`. The numbers come from the experiment write-ups under
`studies/rhm/`; the slide names the experiment in a TeX comment above each frame.

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

## 3. The matching tree learns the RHM near P*

- Left: root accuracy against the size of a fixed training set, three grammars per size. The tree fails at 2^12 = P*
  (median accuracy 0.15, chance is 1/16) and is at 0.9999 by 2^14. So its sample complexity is within a factor of 4
  of P*. The runs that fail fit their training set exactly, so they're short of data, not badly optimized.
- Right: four architectures on fresh data. The tree and the tied tree (one shared core per level, a quarter of the
  parameters) reach accuracy 1 within a few hundred updates. The same tree with permuted leaves levels off at 0.60 to
  0.72, and a parameter-matched chain is still climbing at 0.82 to 0.95 after 40,000 updates.
- Caveat to say out loud: the permuted tree and the chain ran at the top of their learning-rate grid, so a higher
  rate isn't ruled out.
- What this slide could not tell us is why the permuted tree falls short: no capacity at this width, or capacity
  that training doesn't find. The second toy experiment answers that on a small grammar.

## 4. Bonds hold the belief, not the grammar's messages

- Lower subtrees transfer. Pretrain on 48 of the 64 root productions, freeze the subtrees, and a fresh root core
  reaches 0.9 accuracy on the held-out productions in 42 updates, against 148 from scratch. An oracle that is handed
  the true symbols takes 36.
- The product of the two top bonds determines the Bayes belief over the hidden symbols: a bilinear probe passes in 7
  of 8 grammars at both r = 16 and r = 32. This shows that the bonds linearly identify the belief state. It doesn't
  show that they hold an affine code of it.
- The tree does not implement the grammar's message passing. Bonds don't hold the upward messages (0 of 9
  grammar-width cells pass), and a merge is not the grammar's message update (0 of 9).
- A large part of that gap is that pairs the grammar forbids keep as much read-out mass as valid pairs.
- Training on mixed masks, so that both children are informative, didn't repair it: 0 of 12 runs pass. That
  experiment is recorded as inconclusive by its own rule, because the mixed tree only learned the task in 2 of 4
  grammars.
- The problem with all of this: we had no statement of what an optimal tree must encode, so a failed test couldn't
  be told apart from a wrong prediction. That's the motivation for last night's runs.

## 5. Mess3: every probe recovers the belief simplex

- These three slides are the belief-geometry attempts on hidden Markov processes, the setting of the Simplex work:
  train on next-token prediction, then ask whether a linear probe reads the Bayes belief off the hidden state.
- Columns: the true beliefs, then each model's probe output on held-out sequences. The transformer is probed on its
  last layer, the chain on its state, and the tree on its two top bonds, with their product in the second half of
  the sequence. Color is the true belief.
- Top row: the trained networks. All four reproduce the Mess3 fractal, with R² of 0.996 or more.
- Scope, to say plainly: Mess3 is the easy case. Its belief is an exact linear image of the next-token distribution,
  so a good probe here only shows that the model predicts well. And the bottom row shows that the untrained networks
  already score 0.87 to 0.99. This is one seed.

## 6. RRXOR: belief states, not yet belief geometry

- RRXOR is the process that separates belief from prediction: reading the belief from the next-token distribution
  tops out at 0.47.
- The chain is the cleanest (0.994 to 0.997 over three seeds) and passes its pre-registered test. The transformer
  passes too (0.865 to 0.872 on the test's own measure).
- The trees are weaker. The r = 8 tree fails its test: 0.73 and 0.77 in two seeds and a negative score in the third,
  against a ceiling of 0.77. The r = 32 tree in this picture looks good, but its pre-registered test was not defined
  at that width, and a follow-up with reduced product features misses its bar by 0.002 in two of three seeds.
- The main caveat: RRXOR has only 17 distinct beliefs per position, and a probe fits a random vector per belief
  state almost as well (0.88 to 0.92 for the chain, 0.81 for the transformer). So these plots show that the state
  identifies which belief state it's in. They don't show that it lays out the belief geometry.
- The experiment is logged as inconclusive, meaning mixed: transformer reproduction supported, chain and tree each
  partly supported.

## 7. The tensor networks keep marginals, not the joint

- `factored` is two independent 3-state processes, so the joint belief has 9 states and is the product of two
  marginals.
- Blue: multiply the model's own two marginal probes. That gives the joint at 0.999 for the chain and 0.996 for the
  r = 32 tree. Red: probe the joint directly. The chain and the tree drop to 0.80 to 0.84, a little above the 0.756
  that a state holding exactly the two marginals would score.
- So the tensor networks hold the two marginals exactly, plus a little of the joint. The transformer carries the
  joint directly (0.98). The r = 8 tree sits at the floor and its marginals aren't exact (0.98).
- Scope: this is a reanalysis of saved checkpoints, and most of it followed from numbers we already had. For the
  tree it's the first half of the sequence. The companion process with correlated factors can't test correlations,
  because the part beyond the product of marginals carries about 0.1% of the joint's variance.

## 8. Two 3-symbol grammars with provable optima

- Two hand-built grammars with three symbols, two productions per symbol and three levels, so 8 tokens.
- In `sym` every symbol produces the other two symbols in either order, at every level. `mod` is the same at the two
  upper levels and changes two productions at the bottom level (22 and 11 replace 20 and 10).
- The task is to predict x_5 from x_1 to x_4. The Bayes answer depends on the prefix only through A, the symbol
  above the observed half: probability 1/4 for A itself and 3/8 for each of the other two.
- There are only 24 legal prefixes, so the population loss is an exact sum with no sampling noise. A run counts as
  exact when its excess loss is below 10^-10 nats.
- The model is the same tree with 8 leaves. Legs have dimension 4 (three symbols and the mask), every bond has
  dimension r from 1 to 6, and there are 3 logits. The largest has 1,596 parameters.
- Two pairings. Aligned: the first merges see (x_1, x_2) and (x_3, x_4), the pairs the grammar produced. Crossed:
  swap leaves 2 and 3, so they see (x_1, x_3) and (x_2, x_4).

## 9. At r = 2 an exact tree has no room for the synonym

- The three centered Bayes logit vectors sum to zero and span a plane. The logits are linear in the bond of the
  observed half. So r = 1 can't be exact, and at r = 2 the read-out has to be injective.
- Injectivity forces three things on `sym`. Synonymous pairs get the same bond, so the bond depends only on the
  pair's symbol. The core composes those codes as the grammar does. And a forbidden pair (b, b) gets a nonzero
  response, where the grammar's own message passing would give zero.
- So the earlier finding that forbidden pairs keep their mass is, in this setting, a theorem.
- The size of the response is fixed by two numbers. Write the code of symbol 0 as x times the code of symbol 1 plus
  y times the code of symbol 2. Then the three responses are these closed forms, in units of a valid response.
- The product of any two is at least 3/4, so at most one forbidden pair can be quiet (below 0.87).
- All of this is proved for any exact width-2 tree, whatever the optimizer. The (x, y) formula was derived for this
  experiment.

## 10. Training follows the formulas and never finds 0

- 160 trees on `sym`: widths 1 to 4, two inits, 20 seeds, Adam and then an L-BFGS polish.
- Width 1 never gets below 8.4 × 10^-3 nats. At width 2, 38 of 40 runs are exact.
- Left: the measured response to each forbidden pair against the formula evaluated at that run's (x, y). 114 points
  over almost three decades, agreeing to a relative 9.1 × 10^-7. The synonym collapse holds too: swapping a pair for
  its synonym moves the output by at most 9 × 10^-5 of the distance between two targets.
- Training doesn't pick the symmetric tree and doesn't silence the one pair it could: the median smallest response
  is 0.83.
- Right: at widths 3 and 4 nothing is forced any more. Message passing, with response 0, is an exact solution there.
  All 80 runs are exact and none of the 240 responses is below 0.01. Training never finds the message-passing tree.
- With spare width the synonym is often kept in a direction that only a forbidden context reads: 8 of 40 runs at
  r = 3 and 36 of 40 at r = 4.
- Seen after the run and unexplained: with the identity init the responses cluster between 1.15 and 2.29.

## 11. Every bond bounds a rank

- For a fixed context the logits are linear in every bond. Cut the tree at a bond: the logits factor into r terms,
  a function of the tokens below the bond times a function of everything else.
- So the matrix of Bayes logits, with rows indexed by the tokens below the bond, has rank at most r on any block of
  legal prefixes. That gives a lower bound on r by exhaustive search over bonds and blocks.
- On `mod`, the crossed tree has a bond that holds (x_1, x_3). Four values of that pair against two contexts give a
  4 × 4 block of target logits with determinant −27, so the crossed tree needs r ≥ 4. Two tokens that the grammar
  never joined have no shared symbol to compress into.
- The aligned tree on `mod` needs r ≥ 3, and the grammar's message passing attains it.
- On `sym` the half symbol is the sum of the four tokens mod 3, and a sum can be accumulated in any order, so r = 2
  is enough for both pairings. That's the control.

## 12. Training is exact only where the bounds allow

- 480 trees: two grammars, two pairings, widths 1 to 6, 20 runs per cell.
- The dashed colored lines are the proved bounds. Left of its bound no run is exact: the best of 140 such runs ends
  at 1.15 × 10^-3 nats. At or right of it, runs become exact in every cell.
- On `mod` at r = 3 the aligned tree is exact in 20 of 20 runs and the crossed tree in none. So matching the
  grammar's pairing saves one bond dimension, 3 against 4.
- The crossed bound of 4 is attained by 10 of 20 runs. That's a numerical witness with finite weights. We don't have
  a closed-form tree yet.
- On `sym` both pairings are exact at r = 2, as the theory says.

## 13. The crossed tree is harder to find at its bound

- Each pairing at its own bound, 10 seeds, four learning rates.
- The clear pattern is in the table. On `sym` the crossed tree with the variance init is exact in only 4 or 5 of 10
  seeds at every rate. On `mod` the crossed tree with the identity init fails in every seed at three rates.
- The test fixed in advance gives p = 0.047 on `sym` and p = 0.23 on `mod`, and the two verdicts swap at the default
  rates. So I'd call the trainability result weak: the direction is consistent, but the test passes by one run.
- Seen after the run: the ten `mod` misses all end at 1.149171 × 10^-3 nats, the r = 3 floor to seven digits. They
  converged to the width-3 solution and left the fourth channel of the (x_1, x_3) bond unused.
- None of this shows that the permuted trees on the full RHM fell short for lack of capacity. It shows that a
  capacity account exists, is exact, and is visible in training when the grammar is small enough.

## 14. Which follow-up first?

- Housekeeping first: both grids were rerun from scratch on a Runpod CPU worker. Every verdict is the same and all
  880 run files are identical apart from wall-clock time. That shows the results reproduce. The arithmetic didn't
  differ, so it says nothing about sensitivity to numerical noise.
- Option 1, about 15 minutes: the 40 width-2 seeds under plain gradient descent. It says whether the median smallest
  response of 0.83 belongs to the loss surface or to Adam.
- Option 2: read a closed-form width-4 crossed tree for `mod` off the ten saved exact trees. It turns "attained
  numerically" into a construction and shows what the fourth channel encodes.
- Option 3: the same rank bounds on the larger RHM grammars, for the cuts whose legal blocks can be enumerated. It
  tests whether the permuted trees on slide 3 sat below a bound.
- Ask which one to do first.
