"""Deterministic population losses on the legal strings of an aligned binary RHM.

one_hot_squared_loss returns E[||logits - e_Y||²]/2 for every trainable core.
ce_bracket returns lower/upper bounds on population cross-entropy (natural logs).
Both use the CPU float64 root moments. Pass differentiable=True to retain the
autograd graph back to the original model parameters.

reveal=k fixes the root label and every production above depth k. With m choices
per production there are n_classes * m**(2**k - 1) equiprobable cells. The
count is checked against max_cells before tensor allocations. Each prefix costs
O(2**k * r³ + n_classes * r²) for dense fusion. Prefix batches bound forward
workspace; differentiable evaluation retains the graphs of all batches. The
unconditional root Gram additionally needs O(r⁴) storage (see population.py).

From the repository root:
    python3 -m studies.rhm.child_feature_enabling.losses --check
"""
import argparse
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ctn.tree import TreeTN
from studies.rhm import grammar
from studies.rhm.child_feature_enabling.population import root_moments


def _weight(tensor, differentiable):
    return (tensor if differentiable else tensor.detach()).to(device='cpu', dtype=torch.float64)


def one_hot_squared_loss(model, rules, *, differentiable=False):
    """Exact one-hot squared risk with one half and no division by class count."""
    moments = root_moments(model, rules, differentiable=differentiable)
    readout = _weight(model.C, differentiable).flatten(1)
    mean_target_logit = (readout * moments.class_means).sum() / model.O
    mean_square = ((readout @ moments.gram) * readout).sum()
    return (1 + mean_square) / 2 - mean_target_logit


class CEBounds(NamedTuple):
    lower: torch.Tensor
    upper: torch.Tensor
    cells: int


def _cell_count(model, rules, reveal, max_cells):
    if not isinstance(reveal, int) or not 0 <= reveal <= model.D:
        raise ValueError('reveal must be an integer between zero and grammar depth')
    if not isinstance(max_cells, int) or max_cells < model.O:
        raise ValueError('max_cells must permit at least one cell per root class')
    cells = model.O
    for level in range(reveal):
        m = rules[level].shape[1]
        if m < 1:
            raise ValueError('a grammar symbol must have at least one production')
        if m == 1:
            continue
        for _ in range(2 ** level):
            if cells > max_cells // m:
                raise ValueError(f'reveal={reveal} exceeds max_cells={max_cells}')
            cells *= m
    return cells


def _frontier_means(model, tables, cores, reveal, differentiable):
    """At each depth-k frontier node, condition its embedding on its symbol."""
    means = _weight(model.A, differentiable).expand(model.n, -1, -1)
    for level in range(model.D - 1, reveal - 1, -1):
        left_symbol, right_symbol = tables[level].unbind(-1)
        product = torch.einsum('numa,numb->nuab',
                               means[0::2, left_symbol], means[1::2, right_symbol])
        means = torch.einsum('nuab,noab->nuo', product, cores[level]) / left_symbol.shape[1]
    return means


def _prefix_logits(indices, cells_per_class, tables, cores, frontier, centered_root, reveal):
    """Decode mixed-radix production choices for a batch of equally weighted cells."""
    labels = indices // cells_per_class
    remaining = indices % cells_per_class
    symbols = labels[:, None]
    for level in range(reveal):
        table = tables[level]
        m = table.shape[1]
        choices = []
        for _ in range(2 ** level):
            choices.append(remaining % m)
            remaining = remaining // m
        symbols = table[symbols, torch.stack(choices, dim=1)].flatten(1)

    values = frontier[torch.arange(symbols.shape[1]), symbols]
    for level in range(reveal - 1, 0, -1):
        values = torch.einsum('noij,bni,bnj->bno', cores[level],
                              values[:, 0::2], values[:, 1::2])
    return torch.einsum('oij,bi,bj->bo', centered_root, values[:, 0], values[:, 1]), labels


def ce_bracket(model, rules, *, reveal=0, max_cells=65536, batch_size=256, differentiable=False):
    """CE lies in [lower, upper]. Full reveal gives equality on every legal tree."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    cells = _cell_count(model, rules, reveal, max_cells)
    # root_moments validates tree alignment, rule dimensions and vocabulary.
    moments = root_moments(model, rules, differentiable=differentiable)
    root = _weight(model.C, differentiable)
    centered = root - root.mean(dim=0)
    flat = centered.flatten(1)
    if reveal == 0:
        conditional = moments.class_means @ flat.T
        lower = F.cross_entropy(conditional, torch.arange(model.O))
        mean_norm_sq = conditional.square().sum() / model.O
    else:
        tables = [rules[level].detach().cpu() for level in range(model.D)]
        # The model keeps its cores bottom-up; index by depth from the root here.
        cores = {level: _weight(model.W[model.D - 1 - level], differentiable)
                 .expand(2 ** level, -1, -1, -1) for level in range(1, model.D)}
        frontier = _frontier_means(model, tables, cores, reveal, differentiable)
        total_ce = root.new_zeros(())
        total_norm_sq = root.new_zeros(())
        for begin in range(0, cells, batch_size):
            indices = torch.arange(begin, min(begin + batch_size, cells))
            conditional, labels = _prefix_logits(indices, cells // model.O, tables, cores,
                                                   frontier, centered, reveal)
            total_ce = total_ce + F.cross_entropy(conditional, labels, reduction='sum')
            total_norm_sq = total_norm_sq + conditional.square().sum()
        lower, mean_norm_sq = total_ce / cells, total_norm_sq / cells
    if reveal == model.D:
        return CEBounds(lower, lower, cells)
    total_norm = ((flat @ moments.gram) * flat).sum()
    # The exact difference is nonnegative. Truncate negative roundoff at zero.
    within = (total_norm - mean_norm_sq).clamp_min(0)
    return CEBounds(lower, lower + within / 4, cells)


def check():
    """Independent Cartesian enumeration, including all-core derivatives and partition means."""
    shallow = grammar.sample_rules(v=3, n=2, m=2, s=2, L=2, seed=14)
    deep = grammar.sample_rules(v=3, n=2, m=2, s=2, L=3, seed=5)
    deep[1] = deep[1][:, :1]  # nonconstant production counts across levels
    for rules in (shallow, deep):
        depth = len(rules)
        inputs = torch.cartesian_prod(*[torch.arange(3) for _ in range(2 ** depth)])
        parsed = grammar.parse(inputs, rules, 3)
        legal = parsed[0][:, 0] >= 0
        inputs, labels = inputs[legal], parsed[0][legal, 0]
        for tie in ('none', 'level'):
            model = TreeTN(2 ** depth, 3, 3, 2, tie=tie, seed=52).double()
            logits = model(inputs)
            direct_sq = (logits - F.one_hot(labels, model.O)).square().sum(1).mean() / 2
            exact_ce = F.cross_entropy(logits, labels)
            squared = one_hot_squared_loss(model, rules, differentiable=True)
            torch.testing.assert_close(squared, direct_sq)
            for k in range(model.D + 1):
                bounds = ce_bracket(model, rules, reveal=k, batch_size=3, differentiable=True)
                parent_states = parsed[k][legal]
                _, inverse = torch.unique(parent_states, dim=0, return_inverse=True)
                assert bounds.cells == int(inverse.max()) + 1
                centered = logits - logits.mean(dim=1, keepdim=True)
                predicted = torch.stack([centered[inverse == j].mean(0)
                                         for j in range(bounds.cells)])[inverse]
                low_ref = F.cross_entropy(predicted, labels)
                up_ref = low_ref + (predicted - centered).square().sum(1).mean() / 4
                torch.testing.assert_close(bounds.lower, low_ref)
                torch.testing.assert_close(bounds.upper, up_ref)
                assert bounds.lower.item() <= exact_ce.item() + 1e-12
                assert exact_ce.item() <= bounds.upper.item() + 1e-12
                if k:
                    assert prev.lower.item() <= bounds.lower.item() + 1e-12
                    assert bounds.upper.item() <= prev.upper.item() + 1e-12
                prev = bounds

            for calc, ref in ((squared, direct_sq), (bounds.lower, exact_ce)):
                params = tuple(model.parameters())
                for observed, expected in zip(torch.autograd.grad(calc, params),
                                              torch.autograd.grad(ref, params, retain_graph=True)):
                    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-11)
            with torch.no_grad():
                model.C.copy_(model.C[:1].expand_as(model.C).clone())
            constant = ce_bracket(model, rules)
            torch.testing.assert_close(constant.lower, constant.lower.new_tensor(2.).log())
            torch.testing.assert_close(constant.lower, constant.upper)

    try:
        ce_bracket(model, rules, reveal=3, max_cells=15)
    except ValueError as exc:
        assert 'exceeds max_cells' in str(exc)
    else:
        raise AssertionError('oversize prefix was accepted')
    print('Population risks, CE intervals at every reveal, gradients and cell cap passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    check()
