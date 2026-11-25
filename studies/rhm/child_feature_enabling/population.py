"""Population moments of the two bonds feeding an aligned RHM TreeTN root.

The frozen network is evaluated in CPU float64, without changing its parameters.
The grammar has a uniform root and uniform independent choices among the legal
productions of each symbol, as in studies.rhm.grammar. Positions follow the
data tree. A root feature has ordered index (left, right), flattened row-major.

Only the final Gram has shape (r², r²). Internal nodes contract their cores
before accumulating second moments, costing O(r⁴) operations per legal
production per node. The dense root Gram costs O(r⁴) memory and its full
eigendecomposition for the regression oracle costs O(r⁶) operations. The CE
descent direction concerns population cross-entropy with a bias-free
root at uniform logits. Nonzero descent means that training the
root alone can beat the uniform predictor. It does not directly give accuracy
or the gradient of a trained root.

Run python -m studies.rhm.child_feature_enabling.population --check to compare
the recursion with all valid strings in a tiny grammar.
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ctn.tree import TreeTN
from studies.rhm import grammar


@dataclass
class RootMoments:
    class_means: torch.Tensor  # (n, r²), E[Z | class]
    gram: torch.Tensor         # (r², r²), E[Z Z^T], uncentered

    @property
    def ce_descent_at_zero(self):
        """Negative CE gradient, shape (n, r²), matching the flattened root core."""
        n = self.class_means.shape[0]
        return (self.class_means - self.class_means.mean(dim=0)) / n


def root_moments(model, rules, *, differentiable=False):
    """Compute feature moments at fixed weights; the root core is unused.

    Set differentiable=True in a grad-enabled context to backpropagate through
    the CPU float64 contractions into the original lower cores. By default
    they are detached, so intermediate matrices can be released immediately.
    """
    def cpu_weight(weight):
        if not differentiable:
            weight = weight.detach()
        return weight.to(device="cpu", dtype=torch.float64)

    if not isinstance(model, TreeTN):
        raise TypeError("root_moments requires a TreeTN")
    if not torch.equal(model.perm.cpu(), torch.arange(model.n)):
        raise ValueError("the model must use the identity leaf permutation")
    if set(rules) != set(range(model.D)):
        raise ValueError("grammar depth must match the TTN")
    ordered_rules = [rules[level].detach().cpu() for level in range(model.D)]
    for level, table in enumerate(ordered_rules):
        n_parent = model.O if level == 0 else model.V
        if (table.ndim != 3 or table.shape[0] != n_parent or table.shape[1] == 0
                or table.shape[2] != 2):
            raise ValueError("expected binary rule tables matching the model vocabularies")
        if table.dtype != torch.long or table.min() < 0 or table.max() >= model.V:
            raise ValueError("each child must index the model vocabulary")

    # First axis = position; second axis = symbol conditioned on. At a leaf the
    # representation given its symbol is deterministic, so S_s = mu_s mu_s^T.
    leaf = cpu_weight(model.A)
    means = leaf.expand(model.n, -1, -1)
    seconds = (leaf.unsqueeze(-1) * leaf.unsqueeze(-2)).expand(model.n, -1, -1, -1)

    # The model lists cores bottom-up. The grammar numbers rules from the root.
    for height, weights in enumerate(model.W, start=1):
        table = ordered_rules[model.D - height]
        left_id, right_id = table.unbind(dim=-1)
        n_nodes = means.shape[0] // 2
        cores = cpu_weight(weights).expand(n_nodes, -1, -1, -1)
        next_means, next_seconds = [], []
        for node, core in enumerate(cores):
            left, right = 2 * node, 2 * node + 1
            # Average E[h_L | u] ⊗ E[h_R | v] across each parent's productions.
            product_mean = torch.einsum("ska,skb->sab", means[left][left_id],
                                        means[right][right_id]) / table.shape[1]
            flat_core = core.flatten(1)
            next_means.append(product_mean.flatten(1) @ flat_core.T)

            # W (S_L ⊗ S_R) W^T. For output channel i, reshape the contracted
            # row as S_L W_i S_R, then take its inner products with each W_j.
            node_seconds = []
            for productions in table:
                total = torch.zeros((model.r, model.r), dtype=torch.float64)
                for u, v in productions.tolist():
                    row = seconds[left, u] @ core @ seconds[right, v]
                    total += row.flatten(1) @ flat_core.T
                node_seconds.append(total / len(productions))
            next_seconds.append(torch.stack(node_seconds))
        means, seconds = torch.stack(next_means), torch.stack(next_seconds)

    root_table = ordered_rules[0]
    left_id, right_id = root_table.unbind(dim=-1)
    class_means = torch.einsum("jma,jmb->jab", means[0][left_id], means[1][right_id])
    class_means = class_means.flatten(1) / root_table.shape[1]

    # Mix the right conditional moments with joint child-symbol probabilities
    # before constructing the single unconditional r²-by-r² root Gram.
    pair_index = (left_id * model.V + right_id).flatten()
    pair_count = torch.bincount(pair_index, minlength=model.V ** 2).reshape(model.V, model.V)
    pair_prob = pair_count.to(torch.float64) / pair_index.numel()
    weighted_right = torch.einsum("uv,vbd->ubd", pair_prob, seconds[1])
    gram = torch.einsum("uac,ubd->abcd", seconds[0], weighted_right)
    gram = gram.reshape(model.r ** 2, model.r ** 2)
    return RootMoments(class_means, (gram + gram.T) / 2)


@dataclass
class RegressionOracle:
    weights: torch.Tensor
    min_mse: float       # E[(y - w.Z)²], without a factor of 1/2
    explained: float     # E[y²] - min_mse, in squared-target units
    rank: int            # positive Gram eigenvalues above the chosen cutoff


def regression_oracle(moments, targets, *, rtol=None):
    """Fit a scalar no-intercept squared-loss root, y = targets[root symbol].

    Uses the pseudoinverse cutoff lambda > rtol * lambda_max, defaulting to
    r² times float64 machine epsilon. Very small positive directions below
    the cutoff are treated as null. Tiny negative residual MSE can be roundoff.
    """
    targets = torch.as_tensor(targets, device="cpu", dtype=torch.float64)
    if targets.shape != (len(moments.class_means),):
        raise ValueError("provide one scalar target per root class")
    gram = moments.gram
    if rtol is None:
        rtol = len(gram) * torch.finfo(gram.dtype).eps
    if not 0 <= rtol < 1:
        raise ValueError("rtol must be in [0, 1)")
    rhs = moments.class_means.T @ targets / len(targets)
    values, vectors = torch.linalg.eigh(gram)
    keep = values > rtol * values[-1].clamp_min(0)
    basis = vectors[:, keep]
    weights = basis @ ((basis.T @ rhs) / values[keep])
    explained = float(rhs @ weights)
    return RegressionOracle(weights, float(targets.square().mean()) - explained,
                            explained, int(keep.sum()))


def check():
    """Independent reference from parsed Cartesian strings and actual TreeTN autograd."""
    rules = grammar.sample_rules(v=3, n=2, m=2, s=2, L=2, seed=7)
    strings = torch.cartesian_prod(*[torch.arange(3) for _ in range(4)])
    root = grammar.parse(strings, rules, 3)[0].flatten()
    legal = root >= 0
    x, y = strings[legal], root[legal]
    assert len(x) == 2 * 2 ** 3  # all valid strings have equal probability
    for tie in ("none", "level"):
        model = TreeTN(4, 3, 5, 2, tie=tie, seed=17).double()
        moments = root_moments(model, rules, differentiable=True)
        # All-core gradients must aggregate repeated uses of level-tied weights.
        # One-hot target squared norm is one for every tree.
        readout = model.C.flatten(1)
        quadratic = ((readout @ moments.gram) * readout).sum()
        cross = (readout * moments.class_means).sum() / model.O
        moment_loss = (quadratic + 1) / 2 - cross
        direct_loss = (model(x) - F.one_hot(y, model.O)).square().sum(dim=1).mean() / 2
        params = tuple(model.parameters())
        moment_grad = torch.autograd.grad(moment_loss, params)
        direct_grad = torch.autograd.grad(direct_loss, params)
        for calculated, expected in zip(moment_grad, direct_grad):
            torch.testing.assert_close(calculated, expected, rtol=1e-10, atol=1e-11)
        moments = root_moments(model, rules)
        with torch.no_grad():
            model.C.zero_()
        logits, levels = model(x, acts=True)
        left, right = levels[-1].detach().unbind(dim=1)
        design = (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(1)
        empirical_means = torch.stack([design[y == j].mean(0) for j in range(model.O)])
        torch.testing.assert_close(moments.class_means, empirical_means)
        torch.testing.assert_close(moments.gram, design.T @ design / len(x))
        loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, model.C)
        torch.testing.assert_close(moments.ce_descent_at_zero, -grad.flatten(1))
        targets = torch.tensor([-1., 1.], dtype=torch.float64)
        fit = regression_oracle(moments, targets)
        dense_weights = torch.linalg.lstsq(design, targets[y], driver="gelsd").solution
        torch.testing.assert_close(design @ fit.weights, design @ dense_weights)
        mse = float((targets[y] - design @ dense_weights).square().mean())
        assert abs(fit.min_mse - mse) < 1e-10
        assert fit.rank < moments.gram.shape[0]  # the check exercises a singular Gram

    # A zero representation makes uniform optimal and leaves regression at baseline.
    with torch.no_grad():
        model.A.zero_()
    empty = root_moments(model, rules)
    assert empty.ce_descent_at_zero.count_nonzero() == 0
    fit = regression_oracle(empty, targets)
    assert fit.rank == 0 and fit.min_mse == 1.
    model.perm.copy_(torch.tensor([1, 0, 2, 3]))
    try:
        root_moments(model, rules)
    except ValueError as exc:
        assert "permutation" in str(exc)
    else:
        raise AssertionError("a permuted tree was accepted")
    # Depth four traverses three different fusion levels. With one legal rule
    # per symbol there are exactly two strings, one for each root class.
    deep_rules = grammar.sample_rules(v=3, n=2, m=1, s=2, L=4, seed=11)
    classes = torch.arange(2)[:, None]
    deep_strings = grammar.expand({0: classes}, deep_rules, 0, torch.Generator())[4]
    deep_net = TreeTN(16, 3, 3, 2, seed=3).double()
    deep = root_moments(deep_net, deep_rules)
    with torch.no_grad():
        _, levels = deep_net(deep_strings, acts=True)
        left, right = levels[-1].unbind(dim=1)
        deep_design = (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(1)
    torch.testing.assert_close(deep.class_means, deep_design)
    torch.testing.assert_close(deep.gram, deep_design.T @ deep_design / len(classes))
    print(f"PASS: {len(x)} legal strings; tied/untied means, Gram, CE autograd, "
          "all-core squared-loss gradients, singular regression, zero features, "
          "permutation rejection; depth-four means and Gram")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    check()
