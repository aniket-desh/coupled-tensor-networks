"""Correctness checks for the language-process experiments; exits non-zero on any failure.

  python -m studies.lang.checks [out.json]

1. Bayes vs generation: for every process, the empirical frequency of the generative target matches bayes(x) within
   groups of examples sharing the Bayes vector and one input token (every position in turn), to 5 binomial sigmas.
2. Closed-form Bayes cross-entropy for the processes with a one-hot clean target.
3. Masked bigram: bayes() equals the brute-force posterior from the joint chain probability.
4. Generated inputs are valid; parity with bias 0 has no signal in any proper subset of K.
5. summary(x, S) depends only on x_S; resample_same keeps x outside S, keeps the primary summary, and changes x_S.
6. Whole-bond patching of a node equals running the model on the spliced input (untied and permuted trees).
7. The cut-rank tool on cases with a known answer (fixed source: rank V - 1 across the source's cut, 1 elsewhere).
"""
import json
import math
import sys

import torch

from studies.lang import common as L
from studies.lang import processes as P
from studies.lang import rank as R

torch.set_num_threads(1)
results = {}


def check(name, ok, detail=''):
    results[name] = dict(ok=bool(ok), detail=str(detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


SMALL = dict(fixed=dict(kind='fixed', V=4, j=5, pseed=0), two=dict(kind='two', V=4, i=3, j=12, pseed=0),
             mbigram=dict(kind='mbigram', V=4, p=8, pseed=0), trigger=dict(kind='trigger', V=4, pseed=0),
             parity=dict(kind='parity', k=4, pseed=0), parity_bias=dict(kind='parity', k=4, bias=0.2, pseed=0),
             mod4=dict(kind='parity', k=3, q=4, pseed=0), retrieval=dict(kind='retrieval', V=7, pseed=0),
             multimatch=dict(kind='multimatch', V=3, pseed=0))


def calibration(proc, n=400_000, seed=1):
    g = torch.Generator().manual_seed(seed)
    x, y = proc.sample_xy(n, g)
    pb = proc.bayes(x)
    keyq = torch.round(pb * 1e6).long()
    worst = 0.0
    for t in range(16):
        key = torch.cat([keyq, x[:, t:t + 1]], 1)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        cnt = torch.bincount(inv).double()
        emp = torch.zeros(len(uniq), proc.O, dtype=torch.float64).index_add_(0, inv, torch.nn.functional.one_hot(y, proc.O).double())
        q = torch.zeros(len(uniq), proc.O, dtype=torch.float64).index_add_(0, inv, pb.double()) / cnt[:, None]
        # normal approximation only where both expected counts are >= 10
        ok = (cnt[:, None] * q >= 10) & (cnt[:, None] * (1 - q) >= 10)
        z = (emp / cnt[:, None] - q).abs() / torch.sqrt(q * (1 - q) / cnt[:, None] + 1e-12)
        worst = max(worst, z[ok].max().item() if ok.any() else 0.0)
    return worst


def main():
    for nm, spec in SMALL.items():
        proc = P.make(spec)
        z = calibration(proc)
        check(f'bayes_matches_generation_{nm}', z < 5.5, f'max |z| = {z:.2f} over cells with expected counts >= 10')
        g = torch.Generator().manual_seed(2)
        x, y = proc.sample_xy(20_000, g)
        check(f'valid_{nm}', bool(proc.valid(x).all()), f'{int((~proc.valid(x)).sum())} invalid')

    # 2. closed-form cross-entropy (one-hot clean target with eps noise)
    for nm in ('fixed', 'two', 'trigger', 'parity', 'mod4', 'retrieval'):
        proc = P.make(SMALL[nm])
        x, y = proc.sample_xy(200_000, torch.Generator().manual_seed(3))
        ce = -torch.log(proc.bayes(x).gather(1, y[:, None])).mean().item()
        O, e = proc.O, proc.eps
        hi, lo = 1 - e + e / O, e / O
        h = -(hi * math.log(hi) + (O - 1) * lo * math.log(lo))
        check(f'bayes_ce_closed_form_{nm}', abs(ce - h) < 0.01, f'{ce:.4f} vs {h:.4f}')

    # 3. masked bigram brute force
    proc = P.make(dict(kind='mbigram', V=4, p=6, pseed=1))
    x, _ = proc.sample_xy(64, torch.Generator().manual_seed(4))
    brute = torch.zeros(64, 4, dtype=torch.float64)
    for yv in range(4):
        xx = x.clone()
        xx[:, 6] = yv
        lp = torch.log(proc.pi[xx[:, 0]]) + sum(torch.log(proc.P[xx[:, t - 1], xx[:, t]]) for t in range(1, 16))
        brute[:, yv] = lp
    brute = torch.softmax(brute, 1)
    check('mbigram_bayes_brute_force', torch.allclose(brute, proc.bayes(x), atol=1e-10),
          f'max diff {(brute - proc.bayes(x)).abs().max().item():.2e}')

    # 4. parity: proper subsets of K carry no signal at bias 0 (correlation of y with every sub-parity)
    proc = P.make(dict(kind='parity', k=4, pseed=0, eps=0.0))
    x, y = proc.sample_xy(400_000, torch.Generator().manual_seed(5))
    s, ys = 1 - 2 * x.double(), 1 - 2 * y.double()
    worst = 0.0
    for mask in range(1, 16):
        sub = [t for t in range(4) if mask >> t & 1]
        if len(sub) < 4:
            worst = max(worst, abs((ys * s[:, sub].prod(1)).mean().item()))
    check('parity_no_lower_order_signal', worst < 5 / math.sqrt(400_000), f'max |corr| = {worst:.4f}')

    # 5. summaries depend only on x_S; resample_same contract
    g = torch.Generator().manual_seed(6)
    for nm, spec in SMALL.items():
        proc = P.make(spec)
        x, _ = proc.sample_xy(3000, g)
        ok_local, ok_res, n_res = True, True, 0
        for node, S in list(P.spans().items()) + list(P.spans(L.perm_split(0)).items()):
            out = [t for t in range(16) if t not in S]
            x2 = x.clone()
            x2[:, out] = torch.randint(0, proc.V_in, (len(x), len(out)), generator=g)
            a, b = proc.summary(x, S), proc.summary(x2, S)
            ok_local &= a.keys() == b.keys() and all(torch.equal(a[k][1], b[k][1]) for k in a)
            r = proc.resample_same(x, S, g)
            pk = proc.primary_of(a)
            if r is not None and pk is not None:
                n_res += 1
                same_out = torch.equal(r[:, out], x[:, out])
                same_sum = torch.equal(proc.summary(r, S)[pk][1], a[pk][1])
                changed = (r[:, S] != x[:, S]).any(1).double().mean().item() > 0.3
                ok_res &= same_out and same_sum and changed
                if not (same_out and same_sum and changed):
                    print('  resample_same problem', nm, node, same_out, same_sum, changed)
        check(f'summary_local_{nm}', ok_local)
        check(f'resample_same_{nm}', ok_res, f'{n_res} nodes tested')

    # 6. whole-bond patching equals the spliced input
    for perm in (None, L.perm_split(0)):
        cfg = dict(process=dict(kind='two', V=8, i=0, j=9, pseed=0), model='tree', r=8, tie='none', perm=perm, seed=0,
                   init='identity', init_eps=0.3, task='lang')
        model = L.build(cfg).double()
        proc = L.process(cfg)
        x, _ = proc.sample_xy(256, torch.Generator().manual_seed(7))
        d = torch.randperm(256, generator=torch.Generator().manual_seed(8))
        worst = 0.0
        with torch.no_grad():
            _, A = model(x, acts=True)
            for (l, k), S in P.spans(perm).items():
                spl = x.clone()
                spl[:, S] = x[d][:, S]

                def edit(level, z, l=l, k=k):
                    if level == l:
                        z = z.clone()
                        z[:, k] = A[l][d, k]
                    return z
                worst = max(worst, (model(x, edit=edit) - model(spl)).abs().max().item())
        check(f'patch_equals_splice_{"perm" if perm else "tree"}', worst < 1e-9, f'max diff {worst:.2e}')

    # 7. rank tool
    proc = P.make(dict(kind='fixed', V=6, j=3, pseed=0))
    r_in = R.cut_rank(proc, [2, 3], torch.Generator().manual_seed(9))['rank']
    r_out = R.cut_rank(proc, [8, 9], torch.Generator().manual_seed(9))['rank']
    check('rank_fixed_source', r_in == 5 and r_out == 1, f'across source cut {r_in} (expect 5), elsewhere {r_out} (expect 1)')

    ok = all(r['ok'] for r in results.values())
    if len(sys.argv) > 1:
        from pathlib import Path
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(sys.argv[1], 'w'), indent=1)
    print(f'{sum(r["ok"] for r in results.values())}/{len(results)} checks passed')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
