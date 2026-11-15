"""Follow-up to the RHM belief experiments (not part of their criteria): what does the scrambled-target control score
for a state that IS the belief? No network is used.

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/followups/scramble_oracle.py [GRAMMAR ...]

The experiments read "the probe fits a scrambled target (a random vector per distinct belief) about as well as the
belief" as evidence against a code that lays the beliefs out in their geometry. That reading assumes a geometric code
would fit the scrambled target badly. This script measures it: the experiment's affine probe, split and score, with
the exact belief b_t itself as the "activations",
    oracle_scr(t)  = R2 of the affine fit  b_t -> scrambled target   (the experiment's target: N(0, 1) 8-vector per
                     distinct belief, seed 500 + t)
    oracle_perm(t) = R2 of the affine fit  b_t -> permuted beliefs   (the belief vectors reassigned among the distinct
                     beliefs by a random permutation; same set of target vectors as the truth; mean of N_PERM draws)
on the held-out prefix split. A code that is exactly affine in the belief scores oracle_* on these controls, so
    selectivity available at t = 1 - oracle,
and where the oracle is near 1 (the distinct beliefs are close to affinely independent: any labelling of them is an
affine function of the belief) the control cannot tell a geometric code from any other code that identifies the state.
Also saved per position: the number of distinct beliefs and their affine rank.

Beliefs are computed once per distinct prefix; the fit uses the distinct beliefs with their fit-side and test-side row
counts as weights, which is the row-level least squares of analyze.fit_predict (same relative ridge)."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'rhm-belief-geometry'))
import analyze as AZ  # noqa: E402
import beliefs as BL  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

RES, OUT, N_PERM = HERE.parent / 'results', HERE / 'scramble_oracle.json', 5


def distinct_beliefs(x, rules, g, t, budget=2e7):
    """(U (n_u, D) distinct beliefs rounded to 9 decimals as in the analyses, w_fit (n_u,), w_test (n_u,)): row counts
    on each side of the experiment's held-out prefix split."""
    u, _, counts = torch.unique(x[:, :t], dim=0, return_inverse=True, return_counts=True)
    side = torch.rand(len(u), generator=torch.Generator().manual_seed(100 + t)) < 0.5
    D = BL.belief_dim(t, g)
    chunk = max(1, int(budget // (max(g['n'], g['v']) * g['m'] * D)))
    U, wf, wt = torch.zeros(0, D, dtype=torch.float64), torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
    for i in range(0, len(u), chunk):
        b = G.stack_belief(u[i:i + chunk], rules, g['v'], t).round(decimals=9)
        c, s = counts[i:i + chunk].double(), side[i:i + chunk]
        U, inv = torch.unique(torch.cat([U, b]), dim=0, return_inverse=True)
        wf = torch.zeros(len(U), dtype=torch.float64).index_add_(0, inv, torch.cat([wf, c * s]))
        wt = torch.zeros(len(U), dtype=torch.float64).index_add_(0, inv, torch.cat([wt, c * ~s]))
    return U, wf, wt


def weighted_r2(X, Y, wf, wt, ridge=1e-8):
    """analyze.fit_predict + analyze.score for rows given once each with fit and test multiplicities."""
    mx, my = (X * wf[:, None]).sum(0) / wf.sum(), (Y * wf[:, None]).sum(0) / wf.sum()
    Xc, Yc = X - mx, Y - my
    if X.shape[1] > X.shape[0]:  # an affine function of n_u points lives in their span: fit in that basis
        Q = torch.linalg.svd(Xc, full_matrices=False)[2].T
        Xc = Xc @ Q
    Gm = (Xc * wf[:, None]).T @ Xc
    Gm += ridge * Gm.diagonal().mean().clamp_min(1e-30) * torch.eye(len(Gm), dtype=Gm.dtype)
    pred = Xc @ torch.linalg.solve(Gm, (Xc * wf[:, None]).T @ Yc)
    sse, sst = (wt[:, None] * (Yc - pred) ** 2).sum().item(), (wt[:, None] * Yc ** 2).sum().item()
    return sse, sst


def check():
    """weighted_r2 equals the row-level fit_predict / score on expanded rows."""
    gen = torch.Generator().manual_seed(0)
    X, Y = torch.rand(30, 5, generator=gen, dtype=torch.float64), torch.randn(30, 3, generator=gen, dtype=torch.float64)
    wf, wt = torch.randint(0, 4, (30,), generator=gen), torch.randint(0, 4, (30,), generator=gen)
    rf, rt = torch.repeat_interleave(torch.arange(30), wf), torch.repeat_interleave(torch.arange(30), wt)
    ref = AZ.score(Y[rt], AZ.fit_predict(X[rf], Y[rf], X[rt]), Y[rf].mean(0))
    sse, sst = weighted_r2(X, Y, wf.double(), wt.double())
    assert abs(sse - ref['sse']) < 1e-8 and abs(sst - ref['sst']) < 1e-8, (sse, sst, ref)


def run(name):
    ana = json.load(open(RES / f'{name}_r32_s0' / 'analysis.json'))
    g, n_ana = ana['cfg']['grammar'], ana['n_ana']
    rules = G.sample_rules(g['v'], g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(n_ana, rules, seed=AZ.ANA_SEED + g['seed'])[g['L']]
    out = {}
    for t in sorted(int(t) for t in ana['prefix_split']):
        U, wf, wt = distinct_beliefs(x, rules, g, t)
        sp = ana['prefix_split'][str(t)]
        assert (int(wf.sum()), int(wt.sum())) == (sp['n_fit'], sp['n_test']) and len(U) == ana['geom'][str(t)]['n_unique'], (name, t)
        scr = torch.randn(len(U), 8, generator=torch.Generator().manual_seed(500 + t), dtype=torch.float64)
        # the analyses index the scrambled vectors by torch.unique's order of the rounded beliefs, which is U's order
        sse, sst = weighted_r2(U, scr, wf, wt)
        perm = [weighted_r2(U, U[torch.randperm(len(U), generator=torch.Generator().manual_seed(700 + t + 1000 * k))], wf, wt)
                for k in range(N_PERM)]
        sv = torch.linalg.svdvals(U - U.mean(0))
        out[t] = dict(n_unique=len(U), rank=int((sv > 1e-9 * sv[0]).sum()) if len(U) > 1 and sv[0] > 0 else 0,
                      scr=dict(sse=sse, sst=sst, R2=1 - sse / sst if sst > 0 else None),
                      perm=dict(sse=sum(p[0] for p in perm), sst=sum(p[1] for p in perm),
                                R2=1 - sum(p[0] for p in perm) / sum(p[1] for p in perm) if sum(p[1] for p in perm) > 0 else None))
        f = lambda v: '  -  ' if v is None else f'{v:.3f}'
        print(f'{name} t={t:2d} beliefs={len(U):4d} rank={out[t]["rank"]:3d} oracle scr={f(out[t]["scr"]["R2"])} '
              f'perm={f(out[t]["perm"]["R2"])}', flush=True)
    return dict(grammar=g, per_pos=out, eligible={r: json.load(open(RES / f'{name}_r{r}_s0' / 'analysis.json'))['eligible']
                                                  for r in (16, 32)})


def main():
    check()
    done = json.load(open(OUT)) if OUT.exists() else {}
    for name in sys.argv[1:] or sorted({p.name.rsplit('_r', 1)[0] for p in RES.glob('*_r32_s0')}):
        done[name] = json.loads(json.dumps(run(name)))
        json.dump(done, open(OUT, 'w'), indent=1)
    print(f'\n{"cell":16s} group  positions  oracle scr  oracle perm   (pooled like the experiment: 1 - sum SSE / sum SST)')
    for name, d in sorted(done.items()):
        for r in ('16', '32'):
            for grp in ('h1', 'h2b'):
                ts = [str(t) for t in d['eligible'][r][grp]]
                if ts:
                    pool = lambda k: 1 - sum(d['per_pos'][t][k]['sse'] for t in ts) / sum(d['per_pos'][t][k]['sst'] for t in ts)
                    print(f'{name + "_r" + r:16s} {grp:5s}  {len(ts):9d}  {pool("scr"):10.3f}  {pool("perm"):11.3f}')


if __name__ == '__main__':
    main()
