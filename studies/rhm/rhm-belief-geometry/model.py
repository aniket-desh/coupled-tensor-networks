"""Causal transformer for next-token prediction on RHM sequences, with the residual stream exposed.

Input: [BOS, x_0, ..., x_{T-2}] (vocabulary v + 1, BOS = v); targets x_0 .. x_{T-1}, T = 2^L. The residual stream at
input position t has seen x_<t, so it is compared with the belief state b_t = grammar.stack_belief(x, rules, v, t).
Blocks are the pre-LN blocks of studies/lm/models.py (GPT-2 init)."""
import math

import torch
from torch import nn

from studies.lm.models import Block


class Transformer(nn.Module):
    def __init__(self, V, T, d=64, layers=4, heads=1):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V + 1, d), nn.Parameter(torch.zeros(T, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)
        for name, p in self.named_parameters():
            if p.dim() >= 2 or name == 'pos':
                res = name.endswith('proj.weight') or name.endswith('mlp.2.weight')
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers) if res else 0.02)
            elif name.endswith('bias'):
                nn.init.zeros_(p)

    def forward(self, x, resid=False):
        """x (B, T) long -> logits (B, T, V); with resid=True also the residual stream (layers + 1, B, T, d): the
        embedding, then the output of each block (before the final LayerNorm)."""
        h = self.emb(x) + self.pos
        hs = [h]
        for b in self.blocks:
            h = b(h)
            hs.append(h)
        logits = self.head(self.ln(h))
        return (logits, torch.stack(hs)) if resid else logits


def inputs(tokens, v):
    """[BOS, x_0 .. x_{T-2}] from tokens (B, T)."""
    bos = torch.full_like(tokens[:, :1], v)
    return torch.cat([bos, tokens[:, :-1]], 1)
