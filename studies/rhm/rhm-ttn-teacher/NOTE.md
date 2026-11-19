**Is the RHM root task a tree-tensor-network student–teacher problem?**

*A research note on the exact aligned teacher, the population objective that our data induce, and which parts of the student–teacher results carry over.*

Zach Furman's note on Bayesian networks and tree tensor networks ([`furman-bayes-nets-ttn.pdf`](../../../notes/2026-11-18-meeting-zach/furman-bayes-nets-ttn.pdf)) shows that data from a tree-shaped Bayesian network, with independent uniform inputs and squared loss, give a tree tensor network (TTN) the same population loss as a TTN teacher would: the conditional $P(Y \mid X)$ is itself a TTN (Proposition 3.2), and the loss is $\lVert T(\theta) - T^\star \rVert_F^2 / (2\lvert\mathcal X\rvert)$ plus a constant (Proposition 4.1). The loss-landscape results of Furman, Wäldchen and Hodgkinson then apply whenever the student can realize $T^\star$. The Random Hierarchy Model (RHM) is a tree-shaped generative process, and we train a tree on it, so the natural question is whether our root-classification task is an instance.

The answer is yes for the teacher and no for the objective. **For the binary RHM there is an exact teacher in our `TreeTN` class with bond dimension $v$, tied per level, whose output is the one-hot root on every legal string and zero on every other string. The inputs are uniform on the set $S$ of legal strings, a fraction $2^{-30}$ of the product space in our main grammar, so squared loss gives the Frobenius error restricted to $S$ with prefactor $1/(2\lvert S\rvert)$. That restricted objective differs from the full Frobenius objective by a term that penalizes every output off $S$, and the two have different gradients and different minimizers. Our experiments also use cross-entropy, for which the teacher is a direction along which the loss decreases and no finite minimizer exists.** The first three statements are proved below and verified by enumeration on two tiny grammars. Whether any loss-landscape result holds for the restricted objective is open, and this note does not address it.

**The grammar and the tree count levels in opposite directions.** The grammar ([`grammar.py`](../grammar.py)) has levels $\ell = 0, \dots, L$ from the root down: level 0 holds one of $n$ root symbols, levels $1, \dots, L$ hold symbols from a vocabulary of $v$, and level $L$ is the string of $2^L$ observed tokens. Each level $\ell < L$ has one table `rules[l]` of $k_\ell m$ distinct ordered pairs ($k_0 = n$, $k_\ell = v$ otherwise), $m$ per parent symbol, used at every position of that level. Write $\mathrm{inv}_\ell(b, c)$ for the parent of the pair $(b, c)$ at level $\ell$, and $\mathrm{inv}_\ell(b, c) = \bot$ if no symbol produces it (`grammar.inverse` returns $-1$). The root and every production are drawn uniformly. The `TreeTN` of [`ctn/tree.py`](../../../ctn/tree.py) counts from the leaves up: its level $j$ holds the $2^{L-j}$ bonds that sit at grammar level $L - j$. Core `W[j]` fuses its level $j$ into level $j + 1$, so it sits where the grammar applies `rules[L-1-j]`, and the root core `C` sits at `rules[0]`.

**The teacher puts the inverse rule table of each level into the core at that level.** Take bond dimension $r = v$ and

| `TreeTN` parameter | shape with `tie='level'` | grammar table | entries |
|---|---|---|---|
| `A` | $(1, v, v)$ | none | $A_{ca} = \delta_{ca}$, token $c \mapsto e_c$ |
| `W[j]`, $j = 0, \dots, L-2$ | $(1, v, v, v)$ | `rules[L-1-j]` | $W^{(j)}_{abc} = \mathbf 1[\mathrm{inv}_{L-1-j}(b, c) = a]$ |
| `C` | $(n, v, v)$ | `rules[0]` | $C_{obc} = \mathbf 1[\mathrm{inv}_0(b, c) = o]$ |

A forbidden pair has an all-zero fibre: $W_{\cdot bc} = 0$ when $\mathrm{inv}(b, c) = \bot$. For $L = 4$ the correspondence is `W[0]`, `W[1]`, `W[2]`, `C` with `rules[3]`, `rules[2]`, `rules[1]`, `rules[0]`. The grammar reuses one table per level, which is exactly what `tie='level'` shares, so the teacher has 16,640 parameters in the main grammar ($v = n = 16$); the untied tree holds a copy of the level's core at every node. The code is `teacher()` in [`check.py`](check.py).

**The teacher computes the parse, and outputs zero as soon as any pair is forbidden.** Let $\sigma_{\ell, p}(x)$ be the symbol that `grammar.parse` assigns to node $p$ of level $\ell$, with $\sigma_{\ell,p}(x) = \bot$ if the tokens under the node have no derivation. We claim that the teacher's bond at that node is

```math
z_{\ell, p}(x) = \begin{cases} e_{\sigma_{\ell,p}(x)} & \sigma_{\ell,p}(x) \neq \bot , \\ 0 & \sigma_{\ell,p}(x) = \bot . \end{cases}
```

At the leaves, $z_{L,p} = A^\top e_{x_p} = e_{x_p}$. At an internal node the bond is bilinear in its children, $z_{\ell,p} = \sum_{b,c} W_{\cdot bc}\,(z_{\ell+1,2p})_b\,(z_{\ell+1,2p+1})_c$. If either child bond is zero the sum is zero, and $\sigma_{\ell,p} = \bot$ as well. If the children are $e_b$ and $e_c$ the sum is the single fibre $W_{\cdot bc}$, which is $e_{\mathrm{inv}_\ell(b,c)}$ for a legal pair and zero for a forbidden one. This is the recursion of `parse`. At the root, with $S$ the set of strings that have a derivation and $y(x) = \sigma_{0,0}(x)$,

```math
\boxed{\; T^\star(x) = \mathbf 1_S(x)\, e_{y(x)} \;}
```

**On legal strings this is the Bayes posterior, and on all strings it is the generative joint up to a constant.** The grammar is unambiguous, so $P(Y = y \mid X = x) = \mathbf 1[y = y(x)]$ for $x \in S$, which is $T^\star(x)$ and is what `posterior_root` returns. Each core is also $m$ times the generative conditional read backwards, $W_{abc} = m\,P_\ell(b, c \mid a)$, and a binary tree with $2^L$ leaves has $2^L - 1$ internal nodes, so contracting the cores gives $T^\star(x, y) = n\,m^{2^L - 1}\,P(X = x, Y = y)$ on the whole product space.

**This is the conclusion of Proposition 3.2 with a different route to it.** Zach's network has arrows pointing toward $Y$ and independent sources, and its cores are the conditionals $P(\text{parent} \mid \text{children})$ in tree terms. The RHM factorizes in the other direction, root to leaves. Its reversed conditionals are deterministic on legal pairs (his Remark 3.5) and undefined on forbidden pairs, which have probability zero. The zero fill above is one choice of completion, and any other choice gives the same function on $S$. Proposition 3.2 is stated for $x$ with $P(X = x) > 0$, and that is where it holds here.

**Bond dimension $v$ is sufficient for this construction, and we have not shown that it is necessary.** In `rhm-dynamics`, trees with $r = 8 < v = 16$ reached test error 0 and test KL below $10^{-7}$ nats on the root task in 3 of 3 grammars, under cross-entropy. Two freedoms could allow this, and this note does not separate them: the data constrain the tree only on $S$, and cross-entropy needs only the correct logit to be the largest.

**The input is uniform on the legal strings.** A derivation is a root symbol and one production choice at each of the $2^L - 1$ internal nodes, so its probability is

```math
\frac{1}{n} \left(\frac{1}{m}\right)^{2^L - 1} = \frac{1}{n\,m^{2^L - 1}} .
```

Two different derivations give different strings: the pairs of a level are distinct across all of its (symbol, production) entries, so the string fixes the symbols of level $L - 1$ and their production indices, those fix level $L - 2$, and so on up to the root. Hence derivations map one-to-one onto $S$,

```math
\lvert S \rvert = n\,m^{2^L - 1}, \qquad P(X = x) = \frac{\mathbf 1_S(x)}{\lvert S \rvert} .
```

The main grammar has $\lvert S \rvert = 16 \cdot 4^{15} = 2^{34}$ inside a product space of $\lvert \mathcal X \rvert = 16^{16} = 2^{64}$ strings.

**Squared loss on the raw output is the Frobenius error restricted to $S$.** Take the per-sample loss $\tfrac12 \lVert T(\theta)(x) - e_y \rVert^2$ of Zach's equation (7). The label is a function of the input, $e_{y(x)} = T^\star(x)$, so the conditional variance term of Proposition 4.1 is zero and averaging over the input law gives

```math
\boxed{\; L_S(\theta) = \frac{1}{2\lvert S \rvert} \sum_{x \in S} \lVert T(\theta)(x) - T^\star(x) \rVert^2 = \frac{1}{2\lvert S \rvert}\, \lVert P_S\,(T(\theta) - T^\star) \rVert_F^2 \;}
```

where $P_S$ sets the fibres off $S$ to zero. The first step of the proof of Proposition 4.1, the split at fixed $x$, is unchanged. The last step, the average over a uniform $X$ on $\mathcal X$, is where the two settings part.

**Uniform sampling over the product space adds a penalty on every off-support output.** If we label every string of $\mathcal X$ with the zero-filled teacher and sample uniformly, we are exactly in the setting of equation (1) of Zach's note, with a teacher that is realizable at bond dimension $v$. Splitting the sum over $\mathcal X$ into $S$ and its complement, and using $T^\star = 0$ off $S$,

```math
L_{\mathcal X}(\theta) = \frac{1}{2\lvert \mathcal X \rvert}\, \lVert T(\theta) - T^\star \rVert_F^2 = \frac{\lvert S \rvert}{\lvert \mathcal X \rvert}\, L_S(\theta) + \frac{1}{2\lvert \mathcal X \rvert} \sum_{x \notin S} \lVert T(\theta)(x) \rVert^2 .
```

The gradients differ by the gradient of the second term, which is not a multiple of $\nabla L_S$. Three consequences follow, and each one is checked numerically below.

- **The minimizers differ.** $L_{\mathcal X} = 0$ forces $T(\theta) = T^\star$ on all of $\mathcal X$. $L_S = 0$ forces it only on $S$, so every tree that agrees with the teacher on legal strings is a global minimizer, whatever it outputs elsewhere. Filling the forbidden-pair fibres of the teacher's cores with arbitrary numbers gives such trees: legal strings never reach those entries, so $L_S$ and $\nabla L_S$ stay exactly zero, while $L_{\mathcal X} > 0$ and $\nabla L_{\mathcal X} \neq 0$. In the main grammar these are $(v^2 - k_\ell m)\,k_\ell = 3{,}072$ free entries per tied core, directions along which $L_S$ is flat at the teacher and $L_{\mathcal X}$ is not.
- **The critical points differ.** The filled teachers above are critical points of $L_S$ and not of $L_{\mathcal X}$. In the other direction nothing was checked, but a critical point of $L_{\mathcal X}$ is one of $L_S$ only if the off-support gradient vanishes there, and it has no reason to.
- **The off-support term can dominate.** Across the four random students of the check below, the off-support term is 7 to 96 % of $L_{\mathcal X}$ (96 % for the untied student on the 8-leaf grammar, where $\lvert S \rvert / \lvert \mathcal X \rvert = 0.06$), and its gradient has 89 to 99 % of the norm of $\nabla L_{\mathcal X}$. In the main grammar the ratio $\lvert S \rvert / \lvert \mathcal X \rvert$ is $2^{-30}$.

So a result proved for $L_{\mathcal X}$ does not transfer to our data by Proposition 4.1 alone. It needs an argument for $L_S$, a tensor-completion objective whose mask $\mathbf 1_S(x) = \sum_y T^\star(x, y)$ is itself a TTN of bond dimension at most $v$. I have not read the Furman et al. manuscript, so I cannot say which of its results use the full norm and which might survive a mask of this form.

**The experiments use cross-entropy, and less transfers.** [`train.py`](../train.py) treats the raw output as logits $f_\theta(x) = T(\theta)(x)$. On the root task the Bayes posterior is the point mass $e_{y(x)}$, with zero entropy, so the population loss is

```math
L_{\mathrm{CE}}(\theta) = \frac{1}{\lvert S \rvert} \sum_{x \in S} -\log \mathrm{softmax}(f_\theta(x))_{y(x)} = \mathbb E_x\, \mathrm{KL}\!\left( p_{\mathrm{Bayes}}(\cdot \mid x) \,\Vert\, \mathrm{softmax}(f_\theta(x)) \right) ,
```

which is the `kl` that every run logs. Two things carry over from the squared-loss derivation: the input law, so the objective again sees only outputs on $S$, and the teacher as a direction, since the logits $\beta\,T^\star$ (one core multiplied by $\beta$) give accuracy 1 and loss $\log(1 + (n - 1)e^{-\beta})$, which decreases to 0 as $\beta \to \infty$. Three things depend on squared loss. The Frobenius form is gone, and the target is no longer one tensor on $S$: any logits whose correct entry exceeds the others on every legal string classify perfectly, a weaker condition than $f = e_y$, and one that the $r = 8$ trees met on all 50,000 test strings. The infimum is not attained at finite parameters, so there are no exact minimizers to compare with, and the logit scale must keep growing along any path to zero loss. And no statement about critical points of a Frobenius objective is a statement about this one.

**The masked last-token task is a separate extension.** There the posterior $P(x_{15} \mid x_{<15})$ is not a point mass on every input (the Bayes error is 0.097 to 0.128 in our three grammars), so squared loss would target the conditional tensor and cross-entropy its logarithm up to a shift per prefix. The generative contraction gives the joint $P(x_{<15}, x_{15})$, and the conditional divides it by $P(x_{<15})$, which is proportional to the number of legal completions of the prefix and varies from prefix to prefix. Normalizing therefore changes the tensor, and its TTN representation with it. The predicted variable also sits at a leaf while the tree reads out at the root. Neither the teacher nor the width $v$ above applies to that task without new work.

**The two toy notes already show both sides of the support restriction.** [`rhm-forced-synonyms`](../rhm-forced-synonyms/NOTE.md) proves that an exact tree of the smallest sufficient width must respond to forbidden pairs, and measures that at larger widths, where the zero response is an exact solution, none of 240 forbidden responses in 80 trained trees is below 0.01 of a valid one. Those trees were trained with cross-entropy on a next-token task, so they are not instances of the root construction. They attain the Bayes loss on all 24 legal prefixes and are far from the grammar's zero response off the support, which is the analogue of the freedom that $L_S$ leaves and $L_{\mathcal X}$ removes. [`rhm-crossed-rank`](../rhm-crossed-rank/NOTE.md) states the width bound in the form that respects this: the bond dimension must be at least the rank of every block of the target whose entries are all legal. A rank calculation for the larger grammars has to use the same accounting. Cutting a bond gives a matrix whose legal entries are fixed and whose other entries are free, and the necessary width is the smallest rank over all completions. The rank of a legal block bounds it from below, and the rank after filling the free entries with zeros, which is at most $v$, applies to that one completion only. Under cross-entropy the fixed entries become order constraints and the requirement is weaker again.

**What was checked.** [`check.py`](check.py) builds the teacher, enumerates two tiny grammars in float64 (8 leaves with $v = n = 3$, $m = 2$, where $\lvert S \rvert = 384$ and $\lvert \mathcal X \rvert = 6561$; and 4 leaves with $v = 4$, $n = 2$, $m = 2$), and reuses `parse`, `posterior_root`, `likelihoods` and the derivation enumeration of [`checks.py`](../checks.py). All 24 checks pass ([`results/check.txt`](results/check.txt)):

| claim | check | result |
|---|---|---|
| $T^\star = e_{y(x)}$ on $S$, $0$ off $S$, $= n m^{2^L-1} P(x, y)$ on $\mathcal X$ | teacher against `parse`, `posterior_root` and `likelihoods` on every string of $\mathcal X$ | equal to the last bit |
| uniform law on $S$, $\lvert S \rvert = n m^{2^L - 1}$ | enumerated derivations: count, probabilities, one-to-one onto the strings that parse | 384 and 16 derivations, as predicted |
| population squared loss $= L_S$ | fixed random students (untied $r = 3$, tied $r = 2$): loss and all parameter gradients, derivation average against the restricted Frobenius form | differences at most $1.8 \times 10^{-15}$ |
| $L_{\mathcal X} = \frac{\lvert S\rvert}{\lvert\mathcal X\rvert} L_S + {}$off-support term | same students, loss and gradients over all of $\mathcal X$ | identity holds within the $10^{-12}$ tolerance; $\cos(\nabla L_{\mathcal X}, \nabla L_S)$ between 0.82 and 0.99 |
| different minimizers | teacher with forbidden-pair entries filled with $N(0, 1)$ noise | $L_S = 0$ and $\nabla L_S = 0$ exactly; $L_{\mathcal X} = 3.49$ and $2.40$ |
| cross-entropy | derivation average against mean KL on $S$ for one untied student per grammar; loss and accuracy at $\beta T^\star$ for $\beta = 1, 5, 20$ | equal to $10^{-16}$; accuracy 1 at every $\beta$ |
| the teacher is in the experiments' model class | `common.build` with $r = 16$, tied and untied, main grammar seeds 0 to 2 | one-hot root on 20,000 diagnostic trees; 0 on about 19,700 one-token corruptions that do not parse |

```
PYTHONPATH=. python studies/rhm/rhm-ttn-teacher/check.py
```

A separate fact-check pass reran the script and compared the note with the code, the check output, Zach's note and the cited write-ups. It found nothing to correct in the derivation and nine statements that were worded more strongly than their sources, all revised here.

**What would come next.** The proved part stops at the objective. Three steps follow from it. The first is to read the landscape theorems of Furman et al. against the masked objective $L_S$ and find which hypotheses use the full norm. The second is a direct comparison on a grammar small enough to enumerate: the same tree trained on $L_S$, on $L_{\mathcal X}$ and on cross-entropy from the same initialization, measuring the off-support response at the end, which the formula above predicts is driven to zero only by $L_{\mathcal X}$. The third is the completion-rank bound per bond for the main grammar, which would say whether $r = 8$ is below $v$ because of the support or because of the weaker cross-entropy target. A further experiment uses the teacher as a reference for another architecture: its bonds are the one-hot latent symbols of every level, so one can ask of a transformer trained on the same labels whether those intermediate states are linearly readable from its residual stream at every level, or whether it reaches the root while skipping some of them.

**Sources.** Zach Furman, *Bayesian networks and tree tensor networks* (unpublished note, filed at [`notes/2026-11-18-meeting-zach/`](../../../notes/2026-11-18-meeting-zach/furman-bayes-nets-ttn.pdf)): Proposition 3.2 for the TTN form of $P(Y \mid X)$, Proposition 4.1 for the population loss, Remark 3.5 for the deterministic case. Zach Furman, Stephan Wäldchen and Liam Hodgkinson, *Benign loss landscapes can coexist with worst-case hardness* (manuscript, 2026), cited there for the Frobenius objective and its landscape; not read for this note. The grammar follows Petrini, Cagnetta, Tomasini, Favero and Wyart (2023), as adapted in [`grammar.py`](../grammar.py).
