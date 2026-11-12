"""Models for next-byte prediction from CTX = 16 context bytes (V = 256 inputs and outputs).

A model spec is a flat dict; `build(spec, seed)` returns the module. Every model maps x (B, 16) long -> logits (B, 256).
  {'kind': 'tree', 'r': 32, 'tie': 'none' | 'level', 'init': 'identity' | 'variance', 'eps': 0.1}   ctn.tree.TreeTN
  {'kind': 'chain', 'r': 93, 'init': ..., 'eps': ...}  ctn.tree.ChainTN (shared-core MPS read left to right)
Tree and chain init: 'variance' is the classes' own variance-preserving init; 'identity' (the default) then applies
ctn.tree.identity_init(model, eps): bond channel 0 carries a constant and the fusion cores start near the pass-through
tensor, so the network starts near its additive part (still multilinear, no biases). Under the variance init alone,
Adam drives the cores toward the zero (uniform-output) saddle and the TTN stalls near 8 bits/byte.
  {'kind': 'mlp', 'hidden': 192}                      MLP below
  {'kind': 'transformer', 'd': 128, 'layers': 4, 'heads': 4}   Transformer below

Parameter counts: tree untied 16*256*r + 14 r^3 + 256 r^2 (851,968 at r = 32); chain r + 256 r + r^3 + 256 r
(852,066 at r = 93, the match to the r = 32 tree; 49,184 at r = 32); MLP 4096*192 + 192 + 192*256 + 256 = 836,032;
transformer (d = 128, 4 layers) 861,184."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ctn.tree import ChainTN, TreeTN, identity_init, n_params

CTX, V = 16, 256


class MLP(nn.Module):
    """logits = W2 relu(W1 onehot(x) + b1) + b2, where onehot(x) is the flattened (16 * 256) one-hot input. W1 onehot(x)
    is computed as the sum of the 16 active columns of W1, which is the dense product without the zeros. nn.Linear's
    default init (uniform with fan-in 4096 for W1)."""

    def __init__(self, n=CTX, V=V, hidden=192, O=V):
        super().__init__()
        self.n, self.V = n, V
        self.fc1, self.fc2 = nn.Linear(n * V, hidden), nn.Linear(hidden, O)
        self.register_buffer('offset', torch.arange(n) * V)

    def forward(self, x):
        h = self.fc1.weight.T[x + self.offset].sum(1) + self.fc1.bias  # (B, hidden)
        return self.fc2(F.relu(h))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        B, T, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, heads, T, d / heads)
        h = h + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return h + self.mlp(self.ln2(h))


class Transformer(nn.Module):
    """Pre-LN causal transformer with learned positions. Only the last position is read out (logits = head(LN(h_16))),
    so it is trained on one target per window, like every other model. GPT-2 init: N(0, 0.02) weights, zero biases,
    residual projections N(0, 0.02 / sqrt(2 layers))."""

    def __init__(self, n=CTX, V=V, d=128, layers=4, heads=4, O=V):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Parameter(torch.zeros(n, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, O)
        for name, p in self.named_parameters():
            if p.dim() >= 2 or name == 'pos':
                std = 0.02 / math.sqrt(2 * layers) if name.endswith('proj.weight') or name.endswith('mlp.2.weight') else 0.02
                nn.init.normal_(p, std=std)
            elif name.endswith('bias'):
                nn.init.zeros_(p)

    def forward(self, x):
        h = self.emb(x) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h[:, -1]))


def build(spec, seed):
    k = spec['kind']
    if k in ('tree', 'chain'):
        m = (TreeTN(n=CTX, V=V, r=spec['r'], O=V, tie=spec.get('tie', 'none'), seed=seed) if k == 'tree'
             else ChainTN(n=CTX, V=V, r=spec['r'], O=V, seed=seed))
        init = spec.get('init', 'identity')
        if init == 'identity':
            identity_init(m, spec.get('eps', 0.1))
        elif init != 'variance':
            raise ValueError(init)
        return m
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if k == 'mlp':
            return MLP(hidden=spec['hidden'])
        if k == 'transformer':
            return Transformer(d=spec['d'], layers=spec['layers'], heads=spec['heads'])
    raise ValueError(k)


def eval_batch(spec):
    """Evaluation batch size that keeps the largest intermediate near 2^27 floats (the tree's first fusion builds a
    (B, 8, r, r) tensor; the chain a (B, r, r) one)."""
    if spec['kind'] == 'tree':
        return int(min(16384, max(256, 2 ** 27 // (8 * spec['r'] ** 2))))
    if spec['kind'] == 'chain':
        return int(min(16384, max(256, 2 ** 27 // spec['r'] ** 2)))
    return 4096 if spec['kind'] == 'transformer' else 16384


__all__ = ['MLP', 'Transformer', 'build', 'eval_batch', 'n_params']
