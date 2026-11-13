"""Sampled separation rank of the centered Bayes logits across each tree cut.

For a bond over positions S, any TreeTN realizing p*(y | x) satisfies
    r_S >= rank of the matrix F[(x_S), (x_rest, y)],   F_y(x) = log p*(y | x) - (1/O) sum_z log p*(z | x),
because softmax ignores a per-input shift and centering is linear in the logits. Here F is evaluated on R sampled
x_S (rows) times C sampled x_rest (columns), each combination spliced into one input. Columns with any invalid
combination (processes with support constraints: retrieval needs distinct keys, multimatch needs a match) are dropped,
and duplicate rows/columns removed. The numerical rank (singular values > TOL * largest) of a sampled submatrix is a
lower bound on the requirement on those contexts, not a certificate that width r suffices; 'cap' is the largest
rank the sampled matrix could show. rank_1pct / rank_10pct count singular values above 1% / 10% of the largest, a
rough measure of how many directions an approximate (not exact) representation would need.

  python -m studies.lang.rank <out.json> [--configs <experiment name>]
"""
import argparse
import json

import torch

from studies.lang import processes as P

TOL = 1e-7


def retrieval_sides(proc, S, R, C, g):
    """Rows and columns for the retrieval process whose every combination is valid: key slots inside S take a random
    arrangement of a fixed key set A (|A| = key slots in S), key slots outside take the complementary keys, and the
    query (wherever it sits) is drawn from all used keys; the blank stays blank."""
    V, Pn = proc.V, proc.P
    kin = [m for m in range(Pn) if 2 * m in S]
    keys = torch.randperm(V, generator=g)[:Pn]
    A, Ab = keys[:len(kin)], keys[len(kin):]

    def draw(B):
        x = torch.full((B, 16), V, dtype=torch.long)
        for b in range(B):
            ka = A[torch.randperm(len(A), generator=g)]
            kb = Ab[torch.randperm(len(Ab), generator=g)]
            ia = ib = 0
            for m in range(Pn):
                if m in kin:
                    x[b, 2 * m] = ka[ia]; ia += 1
                else:
                    x[b, 2 * m] = kb[ib]; ib += 1
                x[b, 2 * m + 1] = torch.randint(0, V, (1,), generator=g)
            x[b, 14] = keys[torch.randint(0, Pn, (1,), generator=g)]
        return x
    return draw(R), draw(C)


def cut_rank(proc, S, g, R=1200, C=300):
    if proc.name == 'retrieval':
        xr, xc = retrieval_sides(proc, S, R, C, g)
    else:
        xr, _ = proc.sample_xy(R, g)
        xc, _ = proc.sample_xy(C, g)
    out = [t for t in range(16) if t not in S]
    xr = torch.unique(xr[:, S], dim=0)
    xc = torch.unique(xc[:, out], dim=0)
    X = torch.empty(len(xr), len(xc), 16, dtype=torch.long)
    X[:, :, S] = xr[:, None, :]
    X[:, :, out] = xc[None, :, :]
    flat = X.reshape(-1, 16)
    ok = proc.valid(flat).reshape(len(xr), len(xc))
    cols = ok.all(0)
    if cols.sum() < 2:
        return dict(rank=None, rows=len(xr), cols=int(cols.sum()), note='too few valid columns')
    lp = torch.log(proc.bayes(flat).double().clamp_min(1e-300))
    F = (lp - lp.mean(1, keepdim=True)).reshape(len(xr), len(xc), -1)[:, cols]
    M = F.reshape(len(xr), -1)
    s = torch.linalg.svdvals(M)
    rank = int((s > TOL * s[0]).sum()) if s[0] > 0 else 0
    frac = lambda f: int((s > f * s[0]).sum()) if s[0] > 0 else 0
    return dict(rank=rank, rank_1pct=frac(1e-2), rank_10pct=frac(1e-1), rows=len(xr), cols=int(cols.sum()),
                cap=min(len(xr), int(cols.sum()) * F.shape[2]), sv=s[:min(len(s), 80)].tolist())


CONFIGS = {
    'lang-source': [dict(kind='fixed', V=8, j=0), dict(kind='two', V=8, i=0, j=1), dict(kind='two', V=8, i=1, j=2),
                    dict(kind='two', V=8, i=3, j=4), dict(kind='two', V=8, i=7, j=8)],
    'lang-evidence': [dict(kind='mbigram', V=16, p=5), dict(kind='mbigram', V=16, p=8), dict(kind='trigger', V=16)],
    'lang-parity': [dict(kind='parity', k=k) for k in (2, 4, 8, 16)] + [dict(kind='parity', k=4, q=4)],
    'lang-induction': [dict(kind='retrieval', V=8), dict(kind='multimatch', V=8)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--configs', default='', help='experiment name in CONFIGS (default: all)')
    a = ap.parse_args()
    names = [a.configs] if a.configs else list(CONFIGS)
    res = {}
    for nm in names:
        for spec in CONFIGS[nm]:
            proc = P.make(dict(spec, pseed=0))
            k = P.key(dict(spec, pseed=0))
            res[k] = {}
            for (l, kk), S in P.spans().items():
                r = cut_rank(proc, S, torch.Generator().manual_seed(100 + 7 * l + kk))
                res[k][f'{l}_{kk}'] = r
            print(k, {n: v['rank'] for n, v in res[k].items()}, flush=True)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
