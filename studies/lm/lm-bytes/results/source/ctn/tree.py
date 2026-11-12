"""Discrete-input tensor networks trained by backprop, for sequence data (bytes, RHM tokens).

TreeTN: balanced binary tree over n = 2^D leaves. Each leaf embeds its token, z_i = A_i onehot(x_i) in R^r; each
non-root internal node fuses its two children multilinearly, z_a = sum_bc W_abc zL_b zR_c; the root core maps the two
top-level bonds to O logits, l_o = sum_ab C_oab zL_a zR_b. No biases or nonlinearities: the logits are a multilinear
function of the leaf embeddings, as in the theory. Untied by default (every node has its own core).
  tie='level': one A shared by all leaves and one W per level (equivalent local computations share weights).
  perm: a fixed permutation of the leaves (x[:, perm] enters the tree), to test whether matching the data hierarchy
  matters.
Parameter count (untied, no biases): n V r + (n - 2) r^3 + O r^2.

ChainTN: the shared-core chain (an MPS read left to right), h_t = W(h_{t-1}, A onehot(x_t)), logits = U h_n.

Levels: level 0 holds the n leaf bonds, level l the n / 2^l node bonds, and level D - 1 the two bonds the root fuses.
`forward(x, acts=True)` also returns the bond activations per level, [(B, n / 2^l, r) for l = 0 .. D-1];
`edit(level, z) -> z` replaces a level's bonds before they are fused (ablation and patching).

Init 'variance': leaf entries N(0, 1) and fusion entries N(0, 1/r^2), so every bond entry has second moment 1 at init
(given unit children, Var z_a = r^2 * (1/r^2) * 1 * 1); root N(0, 1/r^2) gives unit-variance logits. `scale` multiplies
every core (theory runs use a small uniform scale instead)."""
import math

import torch
from torch import nn


class TreeTN(nn.Module):
    def __init__(self, n, V, r, O, tie='none', perm=None, scale=1.0, seed=0):
        super().__init__()
        D = int(math.log2(n))
        assert 2 ** D == n and D >= 1, 'n must be a power of 2'
        self.n, self.V, self.r, self.O, self.D, self.tie = n, V, r, O, D, tie
        g = torch.Generator().manual_seed(seed)
        rnd = lambda *shape, std: nn.Parameter(torch.randn(*shape, generator=g) * std * scale)
        self.A = rnd(1 if tie == 'level' else n, V, r, std=1.0)
        # W[l - 1] fuses level l-1 into level l, for l = 1 .. D-1 (level D is the root core C).
        self.W = nn.ParameterList([rnd(1 if tie == 'level' else n // 2 ** l, r, r, r, std=1.0 / r) for l in range(1, D)])
        self.C = rnd(O, r, r, std=1.0 / r)
        self.register_buffer('perm', torch.as_tensor(perm if perm is not None else range(n), dtype=torch.long))

    def leaves(self, x):
        x = x[:, self.perm]
        if self.tie == 'level':
            return self.A[0][x]  # (B, n, r)
        return self.A[torch.arange(self.n, device=x.device), x]  # A[i, x_bi] -> (B, n, r)

    def forward(self, x, acts=False, edit=None):
        z = self.leaves(x)
        out = []
        for l in range(self.D):
            if edit is not None:
                z = edit(l, z)
            out.append(z)
            if l == self.D - 1:
                break
            zl, zr = z[:, 0::2], z[:, 1::2]  # children of node k at level l+1 are 2k and 2k+1
            W = self.W[l].expand(zl.shape[1], -1, -1, -1)
            z = torch.einsum('nabc,Bnb,Bnc->Bna', W, zl, zr)
        logits = torch.einsum('oab,Ba,Bb->Bo', self.C, z[:, 0], z[:, 1])
        return (logits, out) if acts else logits

    def groups(self):
        """Parameters by level, for restricted updates and gradient inner products: {'leaf': [A], 'L1': [W_1], ...,
        'root': [C]}. With untied cores, a level's tensor holds every node of that level (index the leading axis for one
        node)."""
        g = {'leaf': [self.A]}
        g.update({f'L{l}': [w] for l, w in enumerate(self.W, start=1)})
        g['root'] = [self.C]
        return g


class ChainTN(nn.Module):
    """h_0 learned; h_t = sum_bc W_abc h_{t-1,b} e_c with e = A onehot(x_t), one W and one A for all positions;
    logits = U h_n. Parameters: r + V r + r^3 + O r. Same unit-second-moment init as TreeTN."""

    def __init__(self, n, V, r, O, scale=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        rnd = lambda *shape, std: nn.Parameter(torch.randn(*shape, generator=g) * std * scale)
        self.n, self.V, self.r, self.O = n, V, r, O
        self.h0, self.A, self.W, self.U = rnd(r, std=1.0), rnd(V, r, std=1.0), rnd(r, r, r, std=1.0 / r), rnd(O, r, std=1.0 / math.sqrt(r))

    def forward(self, x, acts=False, edit=None):
        h = self.h0.expand(x.shape[0], -1)
        e = self.A[x]  # (B, n, r)
        out = []
        for t in range(self.n):
            h = torch.einsum('abc,Bb,Bc->Ba', self.W, h, e[:, t])
            if edit is not None:
                h = edit(t, h)
            out.append(h)
        logits = h @ self.U.T
        return (logits, out) if acts else logits

    def groups(self):
        return {'embed': [self.A, self.h0], 'core': [self.W], 'head': [self.U]}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def tree_params(n, V, r, O):
    """Untied TreeTN parameter count, n V r + (n - 2) r^3 + O r^2."""
    return n * V * r + (n - 2) * r ** 3 + O * r ** 2


def identity_init(model, eps):
    """Identity init, applied after construction (studies select it with init='identity'). Bond channel 0
    carries a constant: every leaf embedding is e_0 + eps * (variance-init noise), and every fusion core is
    eps * (its variance-init noise) + T with T[0,0,0] = 1 and T[a,a,0] = T[a,0,a] = 1 for a >= 1 (const x const ->
    const; a signal times the other child's constant passes through). The network then starts near its additive part,
    logits ~ C[:, 0, 0] + sum_i (linear response to leaf i), instead of a product of 16 random factors. The root core C
    (chain: U) keeps its variance init. The chain uses the same T with h_{t-1} as the left child and h_0 = e_0 + noise.
    With the variance init alone, Adam drives every core toward 0 (the uniform-output saddle of a multilinear net) on
    the RHM tasks and on bytes; this init keeps the network multilinear but starts it near its additive part."""
    with torch.no_grad():
        model.A.mul_(eps)
        model.A[..., 0] += 1.0
        Ws = [model.W] if isinstance(model, ChainTN) else list(model.W)
        for W in Ws:
            W.mul_(eps)
            W[..., 0, 0, 0] += 1.0
            idx = torch.arange(1, W.shape[-1])
            W[..., idx, idx, 0] += 1.0
            W[..., idx, 0, idx] += 1.0
        if isinstance(model, ChainTN):
            model.h0.mul_(eps)
            model.h0[0] += 1.0
