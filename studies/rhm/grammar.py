"""Random Hierarchy Model (Petrini, Cagnetta, Tomasini, Favero, Wyart 2023), binary productions.

Adapted from github.com/fracagnetta/random-hierarchy-model (datasets/random_hierarchy_model.py, MIT licence,
(c) 2023 Leonardo Petrini): `sample_rules`, `sample_trees`, `resample_rules` and `resample_symbols` follow it, with
explicit torch generators (the original seeds only Python's `random`, so its sample seeds don't affect torch.randint).
Added here: the inverse rule map (parse), and exact posteriors by an upward pass over the tree.

Grammar. Level 0 is the root, with n root symbols; levels 1 .. L-1 have v symbols; level L holds the s^L observed
tokens, also from a vocabulary of v. rules[l] has shape (v_l, m, s): symbol a at level l expands into rules[l][a, k]
with production k chosen uniformly from m. Each level's v_l * m productions are distinct s-tuples, sampled without
replacement from v^s, so every tuple is produced by at most one symbol and a valid tree is parsed uniquely.

A tree is a dict {l: (B, s^l) symbols} for l = 0 .. L, plus {('rule', l): (B, s^l) production indices} for l < L."""
from itertools import product

import torch


def sample_rules(v, n, m, s, L, seed):
    """rules[l]: (n if l == 0 else v, m, s) long tensor. Requires m * max(n, v) <= v^s."""
    g = torch.Generator().manual_seed(seed)
    tuples = torch.tensor(list(product(range(v), repeat=s)))
    rules = {}
    for l in range(L):
        k = n if l == 0 else v
        assert k * m <= len(tuples), 'not enough distinct productions: need m * v <= v^s'
        rules[l] = tuples[torch.randperm(len(tuples), generator=g)[:k * m]].reshape(k, m, s)
    return rules


def expand(trees, rules, level, g, device='cpu'):
    """Fill levels level+1 .. L of `trees` (in place) from trees[level], choosing productions uniformly."""
    L, m = len(rules), rules[0].shape[1]
    for l in range(level, L):
        sym = trees[l]
        k = torch.randint(0, m, sym.shape, generator=g, device=device)
        trees[('rule', l)] = k
        trees[l + 1] = rules[l].to(device)[sym, k].flatten(1)
    return trees


def sample_trees(B, rules, seed=None, g=None, device='cpu'):
    """B trees with uniform root symbols and uniform productions."""
    g = g or torch.Generator(device).manual_seed(seed)
    trees = {0: torch.randint(0, rules[0].shape[0], (B, 1), generator=g, device=device)}
    return expand(trees, rules, 0, g, device)


def _copy_above(trees, level):
    return {k: v.clone() for k, v in trees.items() if (k if isinstance(k, int) else k[1]) <= level}


def resample_rules(trees, rules, level, position, seed=None, g=None):
    """Keep the symbol at (level, position) and every symbol outside its subtree; resample all productions inside it
    (the observed span changes, the latent feature doesn't)."""
    g = g or torch.Generator(trees[0].device).manual_seed(seed)
    s, L = rules[0].shape[2], len(rules)
    new = {k: v.clone() for k, v in trees.items()}
    sym, span, pos = trees[level][:, position:position + 1], 1, position
    for l in range(level, L):
        k = torch.randint(0, rules[l].shape[1], sym.shape, generator=g, device=sym.device)
        new[('rule', l)][:, pos:pos + span] = k
        sym = rules[l].to(sym.device)[sym, k].flatten(1)
        pos, span = pos * s, span * s
        new[l + 1][:, pos:pos + span] = sym
    return new


def resample_symbols(trees, rules, level, position, seed=None, g=None):
    """Replace the symbol at (level, position) by a uniform one and regenerate its subtree. The tree above is kept,
    so the result can be ungrammatical above `level` (the parent's production may no longer exist)."""
    g = g or torch.Generator(trees[0].device).manual_seed(seed)
    new = {k: v.clone() for k, v in trees.items()}
    k = rules[level].shape[0]
    new[level][:, position] = torch.randint(0, k, (trees[level].shape[0],), generator=g, device=trees[level].device)
    return resample_rules(new, rules, level, position, g=g)


def inverse(rules, v):
    """inv[l]: (v^s,) long, the level-l symbol producing each s-tuple (flattened base v), or -1."""
    inv = {}
    for l, R in rules.items():
        k, m, s = R.shape
        code = (R * v ** torch.arange(s - 1, -1, -1)).sum(-1)  # (k, m)
        t = torch.full((v ** s,), -1, dtype=torch.long)
        t[code.flatten()] = torch.arange(k).repeat_interleave(m)
        inv[l] = t
    return inv


def parse(x, rules, v):
    """Latent symbols from observed tokens x (B, s^L): {l: (B, s^l)}, with -1 where the span is ungrammatical."""
    inv, s, L = inverse(rules, v), rules[0].shape[2], len(rules)
    out = {L: x}
    sym = x
    for l in range(L - 1, -1, -1):
        grp = sym.reshape(sym.shape[0], -1, s)
        bad = (grp < 0).any(-1)
        code = (grp.clamp_min(0) * v ** torch.arange(s - 1, -1, -1, device=x.device)).sum(-1)
        sym = inv[l].to(x.device)[code]
        sym[bad] = -1
        out[l] = sym
    return out


def likelihoods(x, rules, v, unknown=()):
    """Upward pass: lik[l] (B, s^l, v_l) = P(observed tokens under the node | node symbol), exact under the uniform
    production prior (probability 1/m each). Leaves at positions `unknown` are marginalized (all-ones likelihood).
    Float64; the root entry times the uniform root prior gives P(observed tokens)."""
    s, L = rules[0].shape[2], len(rules)
    lik = torch.nn.functional.one_hot(x.clamp_min(0), v).double()
    for p in unknown:
        lik[:, p] = 1.0
    out = {L: lik}
    for l in range(L - 1, -1, -1):
        R = rules[l].to(x.device)
        k, m, _ = R.shape
        child = lik.reshape(lik.shape[0], -1, s, v)  # (B, nodes, s, v)
        # P(children | a) = (1/m) sum_k prod_j lik(child_j | R[a, k, j]); s = 2 here, general s by a running product.
        prod_ = torch.ones(lik.shape[0], child.shape[1], k, m, dtype=lik.dtype, device=x.device)
        for j in range(s):
            prod_ = prod_ * child[:, :, j][:, :, R[:, :, j]]  # gather (B, nodes, k, m)
        lik = prod_.mean(-1)
        out[l] = lik
    return out


def posterior_root(x, rules, v, unknown=()):
    """P(root | observed tokens), (B, n)."""
    lr = likelihoods(x, rules, v, unknown)[0][:, 0]
    return lr / lr.sum(-1, keepdim=True)


def posterior_token(x, rules, v, position):
    """P(x_position | all other tokens), (B, v): one upward pass per candidate token."""
    B = x.shape[0]
    cand = x.repeat_interleave(v, 0).clone()
    cand[:, position] = torch.arange(v, device=x.device).repeat(B)
    p = likelihoods(cand, rules, v)[0][:, 0].sum(-1).reshape(B, v)  # uniform root prior cancels
    return p / p.sum(-1, keepdim=True)
