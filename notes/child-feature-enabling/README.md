# When does a child feature enable a root computation?

In a two-level Random Hierarchy Model (RHM), one contrast of the root label can train the child feature that a second contrast needs. Under population squared-loss gradient flow, the easier contrast changes the hard interaction's learning time from order $1/\delta^2$ to order $\log(1/\delta)$. Here $\delta$ is the initial alignment with a hidden sign at fixed child norm. When the two children share a table, the gap from a fixed left-child alignment to fitting the interaction stays bounded. Releasing that tying constraint makes the gap grow logarithmically.

Read the [derivation as a PDF](theory.pdf), the [execution handoff](HANDOFF.md), or the [proposed experiments and figures](../../studies/rhm/child_feature_enabling/EXPERIMENT.md).

## The population-loss question

Zach Furman's *Bayesian networks and tree tensor networks* constructs a conditional teacher by sending conditional probability tables toward the output. In the RHM, a parent draws an ordered child pair. Put the joint conditional probability of that pair into a three-index core and contract upward with one-hot token vectors. Multiplying the root likelihood by its prior produces the **joint law** $P(x,y)$, summing the root gives $P(x)$, and normalizing gives the posterior on legal inputs. The existing repo already derived the one-hot parser and the support-restricted square loss. [Existing teacher construction](../../studies/rhm/rhm-ttn-teacher/NOTE.md) · [RHM paper](https://arxiv.org/abs/2307.02129)

This branch contracts the **loss of an arbitrary aligned student**. One student copy yields the mean logit for the true class. Two copies observing the same leaves give the expected squared logit norm. Both contributions propagate through conditional moments, including when two child symbols are marginally correlated. Differentiating the resulting scalar gives the population gradient of every core. For cross-entropy, conditioning on progressively more productions gives nested deterministic lower and upper bounds. Full refinement is exact and can be expensive. The depth-four study has $2^{34}$ legal strings. Revealing just the first two levels uses 1,024 cells. These are analytic bounds in real arithmetic; the implementation uses float64. [Joint law and loss](distribution.tex) · [Categorical root](categorical.tex) · [Inside and outside messages for interior gradients](interior.tex)

## What the time gap measures

The planted grammar has four root classes encoded as $(U,UV)$ for independent uniform signs $U,V$. Each sign is computed from two observed leaves. The student learns a child table and the root readouts from the root labels alone. The hard target is $UV$ in every arm. An easy $U$ target can teach the left table which sign matters. With normalized child alignments $r_L,r_R$, the best frozen bilinear root explains exactly $r_L^2r_R^2$ of the interaction variance. This bound follows the represented direction even when core norms change.

For a fixed hard-fit threshold strictly between zero and one, take positive alignment $\delta\downarrow0$, unit initial child norms, zero readout weights, and the Euclidean metric specified in the proofs. Write $L_\delta=\log(1/\delta)$, $\phi=(1+\sqrt5)/2$, $\lambda_1=1/\phi$, and $\lambda_2=(\sqrt{\phi^2+4\phi}-\phi)/2$.

| Child tables | Hard contrast alone | Both contrasts |
| --- | --- | --- |
| Tied | $1/(4\delta^2)+(5/2)L_\delta+O(1)$ | $L_\delta/\lambda_1+O(1)$ |
| Untied | $1/(2\delta^2)+3L_\delta+O(1)$ | $(1/\lambda_1+1/\lambda_2)L_\delta+O(1)$ |

With tying, every update that rotates the left table rotates its copy on the right. With separate tables the right alignment is still order $\delta$ when the left first reaches a small fixed alignment. Once the left branch has learned, it changes the right branch and root weight into a jointly unstable pair. Their rate $\lambda_2$ gives the additional wait. A separate proof recovers the Nov 18 logarithmic gap for aligned components sharing one **amplitude**, where both absolute times are order $1/\delta$. [Tied proof](toy.tex) · [Untied proof](untied.tex) · [November shared-factor proof](shared.tex)

## Which claims apply to the training grammars?

The categorical results apply to arbitrary finite widths at fixed weights for the aligned binary tree. At a uniform stationary predictor, permitting selected child parameters to move creates a negative Hessian eigenvalue precisely when they can change the root's class-dependent feature means to first order. The theorem determines a local direction of descent. The dynamical exponents in the table require the planted model and its optimizer. Matched initializations with two identical target modes provide an immediate counterexample to a universal positive lag from sharing. [Softmax Hessian and local optimal readout](escape.tex)

We enumerated the 64 top productions in each of the three standard grammars. In all three, every nonconstant root contrast has some marginal signal in at least one true child symbol. Every contrast also retains an interaction residual after the best additive fit. The least additive contrast has **6.35%, 10.84%, and 6.31%** of its variance explained by the best additive prediction in seeds 0, 1, and 2. These are properties of the fixed data generators. Whether the existing trained networks exploit these directions, and whether blocking one contrast's gradient slows another, remain experimental questions. [Rank calculation](rules.tex) · [Saved rule tables and spectra](../../studies/rhm/child_feature_enabling/results/grammar_audit.json)

The closest empirical comparisons in this repository are [frozen-subtree transfer to unseen root productions](../../studies/rhm/rhm-transfer/README.md) and [tests of grammar messages in next-token models](../../studies/rhm/rhm-ttn-messages/README.md). The first measures reuse after pretraining, and the second asks whether a network reproduces a specific message on uncertain inputs. The new calculation measures access to a specified root contrast during training, and can put an exact ceiling on what a frozen readout can learn.

The theoretical comparisons in the training dynamics channel include [Zach's local escape-time calculation](https://alignment-theory.zulipchat.com/#narrow/channel/641363/topic/Predicting.20saddle.20escape.20times/near/642640424), [Hamza's distinction between small overlap and small parameter norm](https://alignment-theory.zulipchat.com/#narrow/channel/641363/topic/Initialization.20and.20staircase.20behavior.20in.20TTNs/near/643138860), and [Guillaume's example where a cut rank decreases during learning](https://alignment-theory.zulipchat.com/#narrow/channel/641363/topic/Saddle.20to.20saddle.20TTN/near/643368691). The present derivations use the signed rule table to identify the label signal and a matched intervention to specify what the lower representation enables.

## Reproduce the calculation

From the repository root with the project Python dependencies installed:

```sh
python -m studies.rhm.child_feature_enabling.population --check
python -m studies.rhm.child_feature_enabling.losses --check
python -m studies.rhm.child_feature_enabling.grammar_audit
python -m studies.rhm.child_feature_enabling.toy_check
python -m studies.rhm.child_feature_enabling.plot
```

The first two commands check population quantities by enumeration. The next two reproduce the finite rule audit and small checks of the ODEs. The last draws two figures from equations and the saved spectra. The [handoff](HANDOFF.md) describes how to use a real checkpoint. To rebuild the manuscript, run `latexmk -pdf main.tex` in this directory. The GitHub copy of that build is named `theory.pdf`.
