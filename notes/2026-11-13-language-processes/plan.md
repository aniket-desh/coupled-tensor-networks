**Language-generating processes as probes for tree tensor networks**

*A plan, not yet implemented: which synthetic token processes to add next to the RHM and belief-simplex processes, what each one tests about the TreeTN, and in what order to run them.*

The RHM experiments showed that the TreeTN learns a hierarchical grammar when its tree lines up with the grammar's tree. Scrambling the leaves loses 12 of 12 comparisons, while sharing cores across positions changes nothing. That result says alignment matters, but the RHM can't say *which* dependencies the tree handles well, because every dependency in the RHM is aligned by construction. Simple language processes let us place the dependency the model has to learn: next to the target, a fixed distance back, or at a position that changes from one sequence to the next. Each process below isolates one kind of dependency, has an exact Bayes predictor, and runs on the existing 16-leaf model with no architectural change.

**Shared setup**

Every process generates sequences $x_1, \dots, x_{16}$ over a vocabulary of size $V$. The default task is the RHM 'last' contract from `studies/rhm/common.py`. The model sees $x_1, \dots, x_{15}$, leaf 16 receives the mask token $V$, and the model predicts $x_{16}$ with cross-entropy against logits $\ell_o$. We'll also use one generalization, a *masked-position* task where leaf $p$ is masked and the target is $x_p$. This lets the same process put its dependency on either side of a tree cut, and it changes one line of `task_batch`.

We'd reuse the four `rhm-controls` arms, because each one now answers a sharper question:

- **tree:** the aligned, untied TreeTN at $r = 16$ with identity init, which is the reference.
- **perm:** the same network with scrambled leaves. On processes whose dependencies don't follow the tree, perm and tree should tie, and that tie is itself a prediction.
- **tied:** one leaf embedding and one core per level. The processes below are stationary, so the same rule applies at every position. Tying encodes exactly that, and it should help more here than on the RHM.
- **chain:** the shared-core MPS of width 40. It reads tokens left to right, so its notion of distance is sequence distance $|i - j|$ rather than tree distance.

Two quantities describe where a dependency sits. The *sequence lag* is $16 - j$ for a source token $x_j$. The *tree level* is the level of the lowest common ancestor of leaves $j$ and $16$. That level is 1 for $j = 15$, 2 for $j \in \{13, 14\}$, 3 for $j \in \{9, \dots, 12\}$, and 4 (the root) for $j \le 8$. The information about $x_j$ has to cross every bond from leaf $j$ up to that ancestor, so the tree level sets how many rank-$r$ bottlenecks the signal passes through.

**Logits versus probabilities**

A multilinear network naturally represents products of local factors. An HMM's joint probability is a matrix product state, for example, and a bigram process's joint probability is a product of transition matrices. The model is trained to output logits, though, and the logit of the Bayes predictor is $\log p(x_{16} \mid x_{<16})$, which is a log of a sum whenever the predictor mixes cases. So the rank a process needs at a given cut has to be computed on the Bayes *logit* table, not read off the generative model. For processes with a deterministic or near-deterministic target the two pictures agree, which is part of why the RHM 'root' task was clean. As the first calculation for each process, before any training, we'd estimate the numerical rank of the Bayes logit matrix across each tree cut. Rows are sampled configurations of the tokens below the cut, and columns are sampled configurations of the tokens above it together with the output index. This is cheap on CPU, and it tells us whether $r = 16$ is enough in principle, so that a training failure can be attributed to learning rather than capacity.

When a process restricts the support of $x_{16}$, the Bayes logits are $-\infty$ off-support. To keep the targets finite, each generator replaces $x_{16}$ with a uniform token with probability $\varepsilon = 0.05$, and the Bayes predictor includes that mixture. We'd report the gap to the Bayes cross-entropy, as in the RHM studies.

**1. Bigram (sanity baseline)**

The bigram process is a stationary Markov chain. The first token $x_1$ is drawn from the stationary distribution $\pi$, and each later token is drawn as $x_{t+1} \sim P(\cdot \mid x_t)$. The rows of the transition matrix $P$ are drawn from $\mathrm{Dirichlet}(\alpha \mathbf{1})$, and $\alpha$ sets how predictable the next token is. A sparse variant, where each token has $m$ random successors chosen uniformly, parallels the RHM's $m$ and gives Bayes accuracy $1/m$.

For the 'last' task, the Bayes predictor is $P(x_{16} \mid x_{15})$. The source is leaf 15, the sibling of leaf 16, so the dependency sits at tree level 1. The information still has to travel from that level-1 node up to the root, and that path needs rank $\operatorname{rank}(\log P) \le V$. Every arm should reach Bayes, and this run checks the samplers, the Bayes code, and the training loop.

The masked-position version is more interesting. The Bayes predictor is

$$
p(x_p \mid x_{\setminus p}) \propto P(x_p \mid x_{p-1})\, P(x_{p+1} \mid x_p),
$$

which depends on both neighbors. For $p = 9$, the two neighbors $x_8$ and $x_{10}$ sit on opposite sides of the root cut. For $p = 6$, both neighbors are inside the same level-2 subtree. The process and the rank requirement are the same in both cases, and only the alignment differs. If tree beats chain at $p = 6$ but not at $p = 9$, that means the tree's advantage comes from alignment rather than from capacity.

**2. Skip-bigram (lag sweep)**

The lag-$k$ process makes each token depend on the token $k$ positions back:

$$
x_t \sim \pi \ \ (t \le k), \qquad x_t \sim P(\cdot \mid x_{t-k}) \ \ (t > k).
$$

This is $k$ interleaved Markov chains, so the Bayes predictor for $x_{16}$ is $P(\cdot \mid x_{16-k})$ and every other token acts as a distractor. The Bayes cross-entropy is $H(X_t \mid X_{t-k})$, which doesn't depend on $k$, so the sweep $k = 1, \dots, 15$ keeps task difficulty fixed and moves only the dependency.

This is the cleanest test of tree distance against sequence distance. Under the tree hypothesis, difficulty should be constant within a level band and jump between bands. The bands are $k = 1$ (level 1), $k \in \{2, 3\}$ (level 2), $k \in \{4, \dots, 7\}$ (level 3), and $k \ge 8$ (the root). Under the sequence hypothesis, difficulty should grow smoothly with $k$, and the chain is the natural reference for that. The perm arm then has a definite prediction: its difficulty should track the tree level of the *permuted* position of $x_{16-k}$. For each $(k, \text{arm})$ we'd measure the number of steps until the gap to Bayes falls to 10% of its initial value, and the final gap. The plot would show steps against $k$, with the band boundaries marked.

A harder variant draws the lag per sequence, $k \sim \mathrm{Unif}(K)$ for a small set such as $K = \{1, 4, 8\}$, and doesn't reveal it. The Bayes predictor then has to infer $k$ from the whole sequence:

$$
p(x_{16} \mid x_{<16}) = \sum_{k \in K} p(k \mid x_{<16})\, P(x_{16} \mid x_{16-k}), \qquad p(k \mid x_{<16}) \propto \prod_{t \le 15} p_k(x_t \mid x_{<t}).
$$

Computing this exactly means enumerating $|K|$ cases. The posterior over $k$ is a product of local factors, which is the multilinear-friendly part, but it enters the logit through a log-sum. So this variant directly tests the logits-versus-probabilities issue above.

**3. Skip-trigram**

In a skip-trigram, the next token depends on the current token and on whether a trigger token appeared anywhere earlier. To generate one, draw the background from a bigram process. With probability $q$, insert a trigger $a$ at a uniform position $i \le 14$, and if $x_{15} = b$ then set $x_{16} = c(a, b)$ from a fixed random table $c$. The Bayes predictor needs a posterior over whether a trigger is present, which is computable exactly because the background likelihood is known. This is the transformer-circuits pattern $[a] \dots [b] \to [c]$. It sits between the skip-bigram, where the source position is fixed, and induction, where both the source position and the source content vary. For the tree it means an OR over positions in every subtree, with the result combined with $x_{15}$ at the root.

**4. Induction**

The induction process plants a repeated token whose successor is the answer. Draw $x_1, \dots, x_{14}$ uniformly, pick $i \sim \mathrm{Unif}\{1, \dots, 13\}$, and set $x_{15} = x_i$ and $x_{16} = x_{i+1}$. Accidental repeats of $x_{15}$ are common, about $13/V$ expected at uniform background, so the Bayes predictor is a uniform vote over all matches:

$$
p(x_{16} = y \mid x_{<16}) = \frac{1}{|M|} \sum_{j \in M} [x_{j+1} = y], \qquad M = \{\, j \le 13 : x_j = x_{15} \,\}.
$$

(The $\varepsilon$ mixture is added on top.) Neither the source position nor the source content is fixed. That removes the tree's alignment advantage entirely, so tree and perm should tie.

Induction is also a capacity probe with a concrete prediction. To score candidate $y$ for query $a = x_{15}$, the root needs the bigram counts $c_{a y} = \sum_{j} [x_j = a][x_{j+1} = y]$ for the pairs inside each half. Because the root is linear in each child's bond vector, the left child's bond must linearly determine all $V^2$ entries of the left half's count table, so $r \gtrsim V^2$ at the root even before taking the log. The chain has the same problem, since it must remember every bigram it has seen until the query arrives. The rank calculation would check this estimate in logit space. The experiment would then fix $V = 8$, so $V^2 = 64$, and sweep $r \in \{8, 16, 32, 64, 128\}$. The prediction is that the gap to Bayes closes only near $r \approx V^2$. Sweeping $V$ at fixed $r$ would be confounded with the match rate, which also changes with $V$.

**5. Modular sum (and parity)**

The modular-sum target is $x_{16} = \left(\sum_{t \le 15} x_t\right) \bmod V$, computed from uniform inputs. It's the opposite of induction: easy to represent but hard to learn. The indicator $[y = s \bmod V]$ factorizes through the discrete Fourier basis, $\frac{1}{V}\sum_{k} \omega^{k y} \prod_t \omega^{-k x_t}$, so its separation rank across every cut is $V$, and the TreeTN can represent it exactly at $r = V$ by passing the partial sum mod $V$ up the tree. With uniform inputs, though, every proper subset of tokens is independent of the target. There's no partial signal, and gradient descent starts at the uniform-output saddle with nothing pulling it away. $V = 2$ gives parity. This connects directly to the init-scale and degree results on saddles. The prediction is a long plateau at $\log V$ for every arm, regardless of alignment. Two ways to break it are worth comparing: bias the inputs so that low-order statistics carry signal, or train on a curriculum of shorter sums by masking leaves to a fixed token.

**Other candidates, lower priority**

- **Order-$k$ Markov** generalizes the bigram. Its dependency window of $k$ tokens covers a whole level-$\lceil \log_2 k \rceil$ subtree when aligned and straddles a cut when it isn't.
- **Copy or reversal** sets $x_{17 - t} = x_t$, so every dependency crosses the root at maximal tree distance. This is an anti-aligned extreme for the masked-position task.
- **Dyck languages** are bracket sequences with nesting. They are hierarchical like the RHM, but their tree changes from sequence to sequence, so they test whether a fixed tree can serve a variable parse.
- **Topic mixtures** draw a latent topic per sequence and then a topic-specific bigram. The posterior over topics is a product of per-token likelihoods, which aggregates evidence from all 16 leaves. This is the language-model version of the simplex processes, where the bonds should carry belief states.

**Order of work**

1. **Shared code.** Write `studies/lang/processes.py`. Each process gets a seeded sampler `sample(B, seed) -> (B, 16)` and an exact Bayes predictor `bayes(x, task) -> (B, V)` probabilities. A check script would compare each Bayes predictor with empirical conditional frequencies at small $V$ by Monte Carlo, and confirm the stated Bayes cross-entropies, such as $H(X_t \mid X_{t-k})$ for the skip-bigram. Training would reuse `studies/rhm/train.py` with a process in place of the grammar.
2. **Rank calculation.** On CPU, compute the sampled numerical rank of the Bayes logit table at each tree cut, for each process. This fixes $r$ for the induction sweep and flags any process that $r = 16$ can't represent.
3. **Bigram and skip-bigram lag sweep,** as one experiment. Bigram is the case $k = 1$. The run covers four arms, 15 lags, and the masked-position pair $p \in \{6, 9\}$. This is the most direct test of the tree-distance hypothesis, and it's cheap.
4. **Induction $r$ sweep.**
5. **Modular sum,** with and without the curriculum.
6. Skip-trigram, mixed-lag skip-bigram, and the lower-priority processes, depending on what steps 3 to 5 show.

Each training step goes through its own `EXPERIMENT.md` with success criteria fixed in advance. The predictions above are written to be falsifiable: band-shaped versus smooth lag curves, a tie between tree and perm on induction, a gap that closes at $r \approx V^2$, and a plateau on the modular sum. A negative result on any of them is still informative about what the tree does.
