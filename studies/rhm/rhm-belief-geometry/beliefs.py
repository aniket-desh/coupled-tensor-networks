"""Exact left-to-right belief states for a batch of sampled RHM sequences.

For each position t = 0 .. T-1, the residual stream at input position t (which has seen BOS, x_0 .. x_{t-1}) is compared
with b_t = P(pending symbols | x_<t) from grammar.stack_belief. Positions whose joint has more than MAX_D entries are
skipped (reported, not hidden). Sequences are sampled from the RHM itself, so an unweighted fit over samples is the
prefix-probability-weighted fit."""
import torch

from studies.rhm import grammar as G

MAX_D = 4096


def belief_dim(t, gcfg):
    D = 1
    for l, _ in G.pending_nodes(t, gcfg['s'], gcfg['L']):
        D *= gcfg['n'] if l == 0 else gcfg['v']
    return D


def beliefs(x, rules, gcfg, budget=2e7):
    """x (B, T) long -> {t: (b_t (B, D_t) float64, p_t (B, v) Bayes P(x_t | x_<t))} for every t with D_t <= MAX_D,
    computed once per distinct prefix, in chunks of about `budget` float64 entries of the root's (B, n, m, D) product."""
    out = {}
    for t in range(x.shape[1]):
        D = belief_dim(t, gcfg)
        if D > MAX_D:
            continue
        chunk = max(1, int(budget // (max(gcfg['n'], gcfg['v']) * gcfg['m'] * D)))
        if t == 0:
            u, inv = x[:1], torch.zeros(len(x), dtype=torch.long)
        else:
            u, inv = torch.unique(x[:, :t], dim=0, return_inverse=True)
        b = torch.cat([G.stack_belief(u[i:i + chunk], rules, gcfg['v'], t) for i in range(0, len(u), chunk)])
        p = G.next_token_from_belief(b, rules, gcfg['v'], t)
        out[t] = (b[inv], p[inv])
    return out


def node_marginal(b, t, gcfg, k):
    """Marginal of the k-th pending node (pending_nodes order) from b_t (B, D_t): a linear map of b_t."""
    nodes = G.pending_nodes(t, gcfg['s'], gcfg['L'])
    sizes = [gcfg['n'] if l == 0 else gcfg['v'] for l, _ in nodes]
    return b.reshape(b.shape[0], *sizes).movedim(1 + k, 1).reshape(b.shape[0], sizes[k], -1).sum(-1)
