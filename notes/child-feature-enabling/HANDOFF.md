# Execution handoff for the next Codex session

Work on `codex/child-feature-enabling` in `aniket-desh/coupled-tensor-networks`. The derivations and validation utilities are in this branch. The [experiment definition](../../studies/rhm/child_feature_enabling/EXPERIMENT.md) specifies the next work. Keep empirical results in the study directory beside the scripts. The theory note, grammar enumeration, and small correctness checks are complete. The paired depth-four intervention and the checkpoint analysis have not run.

## Start here

1. Read the [short theory guide](README.md) and the [full PDF](theory.pdf). In the planted model a child alignment means an uncentered normalized overlap with a known latent sign; hard fit means one minus population mean squared prediction error on a unit-variance target. Preserve these definitions when translating from parameter dynamics to plots. A correct sign prediction at arbitrarily small margin can already give perfect accuracy.
2. Inspect the checkout and available artifacts. The source baseline was `e802c31c90b8053e37f8c9ac387ee137576b90e1`. The package lives at `studies/rhm/child_feature_enabling/`. From the repo root use the five [reproduction commands](README.md#reproduce-the-calculation). Run the relevant existing checks in `EXPERIMENT_RULES.md` before expanding to training. Use the project's Python 3.11/3.12 dependencies. The finite results carry their grammar tables, versions or hashes and numerical tolerances.
3. For the planted time laws, start from `toy_check.py`, which has the tied reduced flow, threshold events, and an independent integration in untied raw coordinates. The reference for the older shared-amplitude comparison is `studies/shared_feature/model.py`, including `K_matrix` and `c0_balanced`. The five-core hierarchical example is in `studies/hierarchy/model.py`. It assumes aligned scalar factors. Changing its sharing pattern does not implement a rotated child table.
4. For the depth-four study, use the parameters and intervention in the experiment definition. Maintain separate columns for the frozen-root oracle, the actual predictor, and categorical accuracy. Report the comparison even if the root oracle is already saturated at initialization or the easy gradient has no detectable benefit.

## Exact quantities available now

This snippet runs as written from the repo root. It uses a fresh, untrained aligned tree so the functions can be exercised without a saved run.

```python
import torch
from studies.rhm import common, grammar
from studies.rhm.child_feature_enabling.population import root_moments, regression_oracle
from studies.rhm.child_feature_enabling.losses import one_hot_squared_loss, ce_bracket

cfg = dict(model='tree', tie='none', perm=None, r=8, seed=0, task='root',
           init='identity', grammar=dict(common.GRAMMAR, seed=0))
model = common.build(cfg).double()
rules = common.rules(cfg['grammar'])
x = grammar.sample_trees(1, rules, seed=123)[model.D]
joint = grammar.likelihoods(x, rules, model.V)[0][:, 0] / model.O
input_probability = joint.sum(-1)
posterior = joint / input_probability[:, None]  # x was sampled, so P(x)>0

mom = root_moments(model, rules)
contrast = torch.zeros(model.O, dtype=torch.float64)
contrast[:2] = torch.tensor([1., -1.])   # simple API example, not the study contrast
oracle = regression_oracle(mom, contrast)
fraction_explained = oracle.explained / contrast.square().mean().item()
risk = one_hot_squared_loss(model, rules, differentiable=True)
all_core_gradients = torch.autograd.grad(risk, tuple(model.parameters()))
intervals = [ce_bracket(model, rules, reveal=k) for k in (0, 1, 2)]
print(input_probability, posterior, fraction_explained, risk.item(), intervals)
```

As a worked check, this seed-0 example gives input probability $2^{-34}$ and a one-hot posterior. In the validation environment the squared risk was 0.739178 and the CE interval tightened from [2.7777, 2.8672] at the class partition to [2.7782, 2.8633] after two production levels. These are values for this fresh width-eight model.

For checkpoints replace `common.build` with `model, cfg, step = common.load(path, device='cpu')`, call `model.double().eval()`, then recover the grammar from the loaded config. Confirm `model='tree'`, `task='root'`, identity leaf order and matching vocabulary. Preserve the checkpoint's trained root for the CE interval. The regression oracle and zero-root gradient use only the lower tree. The new routines support any matching identity-permutation binary root tree, including tied and untied lower cores. The last-token mask and chain model require other formulas.

The float64 eigensolve in `regression_oracle` reports explained **mean-square units**, so divide by the target second moment as in the snippet. For centered contrasts this gives a variance fraction. Save the retained Gram rank and cutoff and check cutoff sensitivity before interpreting a jump. A width-$r$ root has $r^2$ input features, and the dense Gram requires $O(r^4)$ space. Start with widths 8 or 16. Use the class- or production-conditioned CE interval at a trained checkpoint. At 16 classes and four binary productions per symbol, `reveal=0,1,2,3,4` has 16, 64, 1,024, 262,144, and 17,179,869,184 cells respectively. A large interval is an unresolved estimate. Report its width, and use a held-out estimate with uncertainty if refinement costs exceed the agreed budget. Partially refined endpoints have their own gradients, which need not bracket the CE gradient.

## Inputs and missing data

`studies/rhm/rhm-dynamics/run.py` defines the original grammar, training arm, and checkpoint steps. `studies/rhm/common.py` owns checkpoint loading, architecture construction, and the diagnostic/test tree sets. `ctn/tree.py` has the actual root orientation and the `acts=True` and `edit=` interfaces. `studies/rhm/rhm-transfer/run.py` has existing patterns for root resets and matched frozen-subtree controls. Summary JSONs are tracked in Git. The original `runs/` directory is ignored, so first inventory `ckpt_*.pt` in the authorized artifact location and check config, step and file hash. Some historical result sets cover other tasks or only final snapshots. Record missing checkpoints explicitly; a curve in a result JSON cannot recover its child-feature Gram.

The grammar audit already rules out an exactly pure interaction of the original 16 root labels on the three saved seeds. Use the least-additive unit contrast per grammar from the eigenproblem in [the rank derivation](rules.tex). Recompute its vector from the saved rule table, store the vector, sign convention and spectral gap, and hold it fixed across arms and checkpoints. A residual after regressing a label on the true child symbols can differ between synonyms of one class. It is a different supervised task.

## Completion criteria for the next session

Deliver the raw per-run/per-checkpoint data, source commit, realized configs, seed pairs, estimator and precision, censored or missing rows, one regenerating plotting script, and a short results README. Show the analytic reference lines and the actual/optimal fits on common axes. Identify separately which result checks the planted theorem, which tests transfer to the three fixed grammars, and which is only observational at a CE checkpoint. The manually designed two-head loss is the bridge between those two regimes. The standard training code uses minibatch Adam and cross-entropy, so changes in its global accuracy alone do not establish the predicted Euclidean-flow clock.

The analytic checks and exact-grammar figures are CPU tasks. If the subsequent training proposal needs Runpod, use the current project setup in `orch.toml` and the user's Runpod rules. Confirm CBAI and the exact `aniket-` pod prefix, quote hardware, time limit and cost, and follow the experiment workflow before allocating. The present branch contains no new billable run or commitment.

## Status after execution (2026-11-25)

The pending work above has been run; the results, checks and limits are in the [study README](../../studies/rhm/child_feature_enabling/README.md). In short:

- **Planted sweep.** All four time laws hold at $\delta$ from 0.08 to 0.005 (untied wait slopes 1.441 to 1.432 against $1/\lambda_2=1.4317$; tied wait 1.0571 at every $\delta$).
- **Checkpoint readout.** All 135 checkpoints were found in the published archive and read; none is missing. Both contrasts are nearly absent from the root's features at initialization and the easy ceiling leads the hard one through the transition.
- **Intervention.** The protocol as written (rates up to $10^{-3}$, 4,000 updates) left every arm censored. Under the amended protocol (rate 0.3, horizon 40,000; amendments A3 to A5 in the experiment definition) detaching the easy gradient delayed the hard contrast's half fit by a median factor of 2.41 (2.09 to 2.51) in nine of nine pairs, with the trained fit on its frozen-child ceiling in both arms.

Not done, and listed in the README under "Not yet checked": a control with a second poorly additive contrast in place of the easy one, the same comparison under exact population gradient descent, other initialization scales, more grammars, and the original cross-entropy/Adam training. A draft post for the training dynamics channel is in [zulip-post.md](zulip-post.md) and an interactive page in [explorer.html](explorer.html).

**Interactive page (2026-11-25, later).** The page now has four connected parts (the grammar with synonym switches, the tree network beside it on the same example with the step-0 and trained intact/detached weights of grammars 0 to 2, the evidence for each hypothesis in all three protocols, and the planted theory with an exact 256-string example). `explorer/export.py` builds it from the saved results and weights into `explorer/site/` (published as a Claude Artifact) and `explorer.html` (one file); `explorer/check_site.py` checks it in a browser. The README numbers were rechecked against the per-pair files; the only change was the gradient cosine (0.014, previously rounded to 0.01). The trained checkpoints of the original cross-entropy runs past step 0 were not reachable when the page was built (the results volume was disconnected), so the page shows those runs only through the saved readout, not their weights.
