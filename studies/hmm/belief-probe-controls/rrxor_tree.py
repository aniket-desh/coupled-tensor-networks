"""Part 3 of belief-probe-controls: the two RRXOR tree cells that hmm-belief-geometry left open.

    PYTHONPATH=. python studies/hmm/belief-probe-controls/rrxor_tree.py

(a) r = 8, seed with R2 = -4.55. The experiment's probe at the judged positions t = 13 .. 15 is projective on
    top_bil = [h_L, h_R, h_L (x) h_R]: bhat = q / sum(q). A negative R2 can come from a handful of test rows whose
    denominator is near zero, which says nothing about the information in the bonds. Per seed and position this
    reports the experiment's R2, the number of test rows whose prediction leaves the range a belief can take
    (some |bhat_i| > 2), the share of the squared error those rows carry, the R2 without them, and the affine probe's
    R2 on the same rows and features (which has no denominator).
    Reading, fixed before running: if the R2 without the out-of-range rows is within 0.05 of the other two seeds'
    values, the -4.55 is a numerical failure of the ratio on those rows and the cell reads as three seeds below the
    0.8 bar; otherwise the seed really differs.
(b) r = 32. Its product features (1,088) have no eligible position on RRXOR: the rule asks for 10 x (features + 1)
    distinct fit prefixes and RRXOR has only 923 to 2,384 at t = 13 .. 15. Here the features are standardized and
    reduced to their top k principal scores, with k = floor(n_fit_prefixes / 10) - 1, the largest number the
    experiment's own rule admits, and probed affinely and projectively.
    Rule, fixed before running (the experiment's H3b clauses): the cell passes if in all three seeds the better of the
    two probes, pooled over t = 13 .. 15, reaches R2 >= 0.8 and halves the unexplained variance of the additive
    ceiling (the best f(left half) + g(right prefix), exact) and of the next-token ceiling
    (1 - R2 <= 0.5 (1 - control)). The same reduced probe on the initial network is reported.

Same sequences, held-out prefix split and score as the experiment."""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RHM = HERE.parents[1] / 'rhm'
sys.path.insert(0, str(RHM / 'rhm-ttn-belief-geometry'))
sys.path.insert(0, str(RHM / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent / 'hmm-belief-geometry'))
import hmm_analyze as HA  # noqa: E402
import hmm_run as HR  # noqa: E402
import ttn_analyze as TA  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.hmm import processes as P  # noqa: E402

RUNS, RES = Path(os.environ.get('RUNS_ROOT', 'runs')) / 'hmm-belief-geometry', HERE.parent / 'hmm-belief-geometry' / 'results'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TS, R2_MIN, RATIO = (13, 14, 15), 0.8, 0.5


def setup(name):
    res, ana = json.load(open(RUNS / name / 'result.json')), json.load(open(RES / name / 'analysis.json'))
    cfg = res['cfg']
    proc = P.get(cfg['process'])
    x = P.sample(proc, ana['n_ana'], cfg['T'], g=torch.Generator().manual_seed(HA.ANA_SEED))
    eta, pnext = (z.to(DEV) for z in P.beliefs(proc, x))
    split = {}
    for t in TS:
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        side = torch.rand(int(pid.max()) + 1, generator=torch.Generator().manual_seed(100 + t)) < 0.5
        split[t] = (torch.nonzero(side[pid])[:, 0].to(DEV), torch.nonzero(~side[pid])[:, 0].to(DEV), int(side.sum()))
        sp = ana['prefix_split'][str(t)]
        assert (len(split[t][0]), len(split[t][1])) == (sp['n_fit'], sp['n_test']), (name, t)

    def bil(ck, t):
        model = HR.build(cfg).to(DEV)
        model.load_state_dict(torch.load(RUNS / name / f'ckpt_{ck}.pt', map_location=DEV)['state'])
        return TA.bond_sets(model.eval(), x.to(DEV), t, cfg['V'], bilinear=True)['top_bil'].double()
    return res, ana, proc, eta, pnext, split, bil


def part_a(name):
    res, ana, proc, eta, _, split, bil = setup(name)
    out = {}
    for t in TS:
        fi, ti, _ = split[t]
        A, b = bil(res['ckpts'][-1], t), eta[:, t]
        mu = b[fi].mean(0)
        bh = HA.projective(A[fi], b[fi], A[ti], [(0, b.shape[1])])
        err = ((b[ti] - bh) ** 2).sum(1)
        bad = bh.abs().max(1).values > 2
        sst_ok = ((b[ti][~bad] - mu) ** 2).sum().item()
        saved = ana['metrics'][str(res['ckpts'][-1])]['per_pos']['top_bil^proj@prefix'][str(t)]['R2']
        out[t] = dict(R2=BC.score(b[ti], bh, mu)['R2'], R2_saved=saved, n_test=len(ti), n_out_of_range=int(bad.sum()),
                      sse_share_out_of_range=(err[bad].sum() / err.sum()).item(), max_abs=bh.abs().max().item(),
                      R2_without=1 - err[~bad].sum().item() / sst_ok,
                      R2_affine=BC.score(b[ti], BC.fit_predict(A[fi], b[fi], A[ti]), mu)['R2'])
        print(f'(a) {name} t={t}: R2 {out[t]["R2"]:.3f} (saved {saved:.3f}), {out[t]["n_out_of_range"]} of {len(ti)} rows out '
              f'of range carry {out[t]["sse_share_out_of_range"]:.3f} of the SSE (max |bhat| {out[t]["max_abs"]:.1f}); without '
              f'them {out[t]["R2_without"]:.3f}; affine {out[t]["R2_affine"]:.3f}', flush=True)
    return out


def part_b(name):
    res, ana, proc, eta, pnext, split, bil = setup(name)
    acc, per = {}, {}
    for label, ck in (('model', res['ckpts'][-1]), ('init', res['ckpts'][0])):
        for t in TS:
            fi, ti, n_fit_pre = split[t]
            A, b = bil(ck, t), eta[:, t]
            mu = b[fi].mean(0)
            k = n_fit_pre // 10 - 1
            X = (A - A[fi].mean(0)) / A[fi].std(0).clamp_min(1e-12)
            evals, evecs = torch.linalg.eigh(X[fi].T @ X[fi])
            Z = X @ evecs.flip(1)[:, :k]
            sc = {f'{label}_aff': BC.score(b[ti], BC.fit_predict(Z[fi], b[fi], Z[ti]), mu),
                  f'{label}_proj': BC.score(b[ti], HA.projective(Z[fi], b[fi], Z[ti], [(0, b.shape[1])]), mu)}
            if label == 'model':
                sc['nexttok'] = BC.score(b[ti], HA.nexttok_ceiling(pnext[:, t], b, pnext[ti, t], b[ti]), mu)
                ex = HA.exact_additive(proc, t, res['cfg']['T'] // 2)  # population (mse, var), pooled like the experiment
                sc['addceil'] = dict(sse=ex[0] * len(ti), sst=ex[1] * len(ti), R2=1 - ex[0] / ex[1])
            per.setdefault(t, {'k': k, 'n_fit_prefixes': n_fit_pre}).update({q: v['R2'] for q, v in sc.items()})
            for q, v in sc.items():
                a = acc.setdefault(q, [0.0, 0.0])
                a[0] += v['sse']
                a[1] += v['sst']
    pooled = {q: 1 - a[0] / a[1] for q, a in acc.items()}
    best = max(pooled['model_aff'], pooled['model_proj'])
    ok = bool(best >= R2_MIN and all((1 - best) <= RATIO * (1 - pooled[c]) for c in ('addceil', 'nexttok')))
    print(f'(b) {name}: k = {[per[t]["k"] for t in TS]}; pooled affine {pooled["model_aff"]:.3f} projective '
          f'{pooled["model_proj"]:.3f} | init {pooled["init_aff"]:.3f} / {pooled["init_proj"]:.3f} | additive ceiling '
          f'{pooled["addceil"]:.3f} next-token ceiling {pooled["nexttok"]:.3f} -> {"pass" if ok else "fail"}', flush=True)
    return dict(per_pos=per, pooled=pooled, passes=ok)


def main():
    out = dict(a={f'rrxor_tree8_s{s}': part_a(f'rrxor_tree8_s{s}') for s in range(3)},
               b={f'rrxor_tree_s{s}': part_b(f'rrxor_tree_s{s}') for s in range(3)})
    out['b_verdict'] = 'pass' if all(q['passes'] for q in out['b'].values()) else 'fail'
    print('(b) r = 32 tree on RRXOR, reduced product probe:', out['b_verdict'])
    (HERE / 'results').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'results' / 'rrxor_tree.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
