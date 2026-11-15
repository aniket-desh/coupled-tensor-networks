"""Follow-up to rhm-ttn-belief-geometry (not part of its pre-registered criteria): the rank ceiling on an affine probe
of a bond with r coordinates, computed from the exact beliefs alone. No network is used.

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/followups/capacity.py            # all grammars
    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/followups/capacity.py v8m4s2L4   # one grammar
    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/followups/capacity.py --table    # table from saved JSON

The bound. An affine probe bhat = W a + c on k features takes values in an affine subspace of dimension <= k. For the
test rows b_i of a position (Eckart-Young, with the mean free),
    min over affine maps of rank <= k of  sum_i |b_i - bhat_i|^2  =  sum_{j > k} lam_j,
    lam = eigenvalues of  sum_i (b_i - mean_test)(b_i - mean_test)^T, in decreasing order,
so the experiment's own score on those rows, R2 = 1 - SSE / SST with SST = sum_i |b_i - mean_fit|^2, obeys
    R2  <=  ceil_k(t) = 1 - sum_{j > k} lam_j(t) / SST(t),
pooled over positions like the experiment's R2_h1: 1 - sum_t tail_k(t) / sum_t SST(t). It holds for every code with k
coordinates, trained or not, and it is a bound on the score itself: same sequences (N_ANA, same seed), same held-out
prefix split, same denominator. k is the number of coordinates that vary:
    `top` at t <= T/2:  k = r      (the right top bond sees only masks and is constant)
    `top` at t >  T/2:  k = 2r
    `top_bil`:          k = 2r + r^2
Also saved: rho_k(t) = 1 - tail_k / total variance on all rows about their own mean (the population version).

Rule, fixed before running. H1 required R2(`top`) >= 0.8 pooled over the eligible first-half positions (h1). For each
cell (grammar, r) that failed H1:
    capacity-limited          the pooled ceiling at k = r is below 0.8: no r-coordinate code could have passed;
    not explained by capacity the pooled ceiling is >= 0.8.
Reported next to it: ceiling - probe score per seed. A gap of 0.1 or more (the experiment's margin) means the learned
code is also well short of what r coordinates allow.
Check on the results: no per-position probe score may exceed its ceiling (tolerance 1e-6).

Beliefs are computed once per distinct prefix in chunks and only their weighted first and second moments are kept
(D x D float64, D <= 4096), so a position never holds the N_ANA x D matrix."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent))
import analyze as AZ  # noqa: E402
import beliefs as BL  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

RES = HERE.parent / 'results'
OUT = HERE / 'capacity.json'
WIDTHS = (16, 32)
KS = sorted({r for r in WIDTHS} | {2 * r for r in WIDTHS} | {2 * r + r * r for r in WIDTHS} | {128})
THRESH, MARGIN = 0.8, 0.1


def moments(b, w):
    """Weighted moments of rows b (n, D) with weights w (n,): (sum w, sum w b, sum w b b^T)."""
    bw = b * w[:, None]
    return torch.stack([w.sum()]), bw.sum(0), bw.T @ b


def tails(S0, S1, S2, mu, ks):
    """From the moments of a set of rows: (SST about mu, {k: sum_{j > k} lam_j}) with lam the spectrum about the
    rows' own mean."""
    C = S2 - torch.outer(S1, S1) / S0
    lam = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)
    sst = (S2.diagonal().sum() - 2 * mu @ S1 + S0 * mu @ mu).item()
    return sst, {k: lam[k:].sum().item() for k in ks}


def position(x, rules, g, t, budget=2e7):
    """Ceiling data for prefix length t on the experiment's held-out prefix split."""
    u, pid, counts = torch.unique(x[:, :t], dim=0, return_inverse=True, return_counts=True)
    side = torch.rand(len(u), generator=torch.Generator().manual_seed(100 + t)) < 0.5  # as in analyze.py / ttn_analyze.py
    w = {'fit': counts * side, 'test': counts * ~side}
    D = BL.belief_dim(t, g)
    chunk = max(1, int(budget // (max(g['n'], g['v']) * g['m'] * D)))
    M = {k: [torch.zeros(1, dtype=torch.float64), torch.zeros(D, dtype=torch.float64),
             torch.zeros(D, D, dtype=torch.float64)] for k in w}
    for i in range(0, len(u), chunk):
        b = G.stack_belief(u[i:i + chunk], rules, g['v'], t)
        for k in w:
            for acc, new in zip(M[k], moments(b, w[k][i:i + chunk].double())):
                acc += new
    mu_fit = M['fit'][1] / M['fit'][0]
    sst, tail = tails(*M['test'], mu_fit, KS)
    all_ = [a + c for a, c in zip(M['fit'], M['test'])]
    var, tail_pop = tails(*all_, all_[1] / all_[0], KS)
    return dict(D=D, n_prefixes=len(u), n_fit=int(w['fit'].sum()), n_test=int(w['test'].sum()), sst=sst, tail=tail,
                rho={k: 1 - tail_pop[k] / var if var > 0 else float('nan') for k in KS})


def check():
    """The streamed moments give the bound that a direct rank-k PCA reconstruction of the test rows attains."""
    gen = torch.Generator().manual_seed(0)
    b = torch.rand(400, 6, generator=gen, dtype=torch.float64) @ torch.rand(6, 20, generator=gen, dtype=torch.float64)
    fit, test = b[:200], b[200:]
    mu = fit.mean(0)
    sst, tail = tails(*moments(test, torch.ones(200, dtype=torch.float64)), mu, [2, 6])
    c = test - test.mean(0)
    U, s, Vh = torch.linalg.svd(c, full_matrices=False)
    r2_pca = AZ.score(test, test.mean(0) + (U[:, :2] * s[:2]) @ Vh[:2], mu)['R2']
    assert abs(r2_pca - (1 - tail[2] / sst)) < 1e-10, (r2_pca, 1 - tail[2] / sst)
    assert tail[6] / sst < 1e-10  # the rows have affine rank <= 6
    # a least-squares probe on 2 arbitrary features cannot beat the k = 2 ceiling
    a = torch.rand(400, 2, generator=gen, dtype=torch.float64)
    assert AZ.score(test, AZ.fit_predict(a[:200], fit, a[200:]), mu)['R2'] <= 1 - tail[2] / sst


def run(name):
    ana = {r: json.load(open(RES / f'{name}_r{r}_s0' / 'analysis.json')) for r in WIDTHS}
    g, n_ana = ana[WIDTHS[0]]['cfg']['grammar'], ana[WIDTHS[0]]['n_ana']
    rules = G.sample_rules(g['v'], g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(n_ana, rules, seed=AZ.ANA_SEED + g['seed'])[g['L']]
    ts = sorted({t for a in ana.values() for t in a['eligible']['e_top'] + a['eligible']['e_bil']})
    per_pos = {}
    for t in ts:
        per_pos[t] = position(x, rules, g, t)
        sp = ana[WIDTHS[0]]['prefix_split'][str(t)]  # the split must be the experiment's
        assert (per_pos[t]['n_fit'], per_pos[t]['n_test']) == (sp['n_fit'], sp['n_test']), (name, t)
        print(f'{name} t={t} D={per_pos[t]["D"]} ' + ' '.join(
            f'k{k}={1 - per_pos[t]["tail"][k] / per_pos[t]["sst"]:.3f}' for k in KS[:4]), flush=True)
    return dict(grammar=g, n_ana=n_ana, half=ana[WIDTHS[0]]['half'], per_pos=per_pos)


def table(out):
    """Per cell: the pooled ceiling on h1, the probe's R2_h1 per seed, the label, and the per-position check."""
    summ = {q['run']: q for q in json.load(open(RES / 'summary.json'))['runs']}
    rows, worst = [], -float('inf')
    for name, d in sorted(out.items()):
        for r in WIDTHS:
            runs = [f'{name}_r{r}_s{s}' for s in range(3)]
            ana = [json.load(open(RES / q / 'analysis.json')) for q in runs]
            h1 = ana[0]['eligible']['h1']
            if not h1:
                continue
            pp = {int(t): q for t, q in d['per_pos'].items()}
            ceil = 1 - sum(pp[t]['tail'][str(r)] for t in h1) / sum(pp[t]['sst'] for t in h1)
            probe = [summ[q]['h1_top'] for q in runs]
            learned = [bool(summ[q]['learned']) for q in runs]
            for a in ana:  # the check: no per-position score above its ceiling
                per = a['metrics'][list(a['metrics'])[-1]]['per_pos']['top@prefix']  # final checkpoint
                for t in a['eligible']['e_top']:
                    k = r if t <= d['half'] else 2 * r
                    worst = max(worst, per[str(t)]['R2'] - (1 - pp[t]['tail'][str(k)] / pp[t]['sst']))
            failed = not all(p is not None and p >= THRESH for p in probe)
            label = ('(not learned: not a judged cell)' if not all(learned) else '-' if not failed else
                     'capacity-limited' if ceil < THRESH else 'not explained by capacity')
            rows.append(dict(cell=f'{name}_r{r}', h1=h1, ceiling=ceil, probe=probe, learned=learned, failed_08=failed,
                             label=label, gap=[ceil - p for p in probe],
                             per_pos={t: 1 - pp[t]['tail'][str(r)] / pp[t]['sst'] for t in h1}))
    print(f'{"cell":16s} {"h1 positions":14s} ceiling  probe (3 seeds)        gap (3 seeds)          label')
    for q in rows:
        print(f'{q["cell"]:16s} {str(q["h1"]):14s} {q["ceiling"]:.3f}    ' + ' '.join(f'{p:.3f}' for p in q['probe']) +
              '      ' + ' '.join(f'{p:+.3f}' for p in q['gap']) + f'      {q["label"]}')
    print(f'check: largest (probe R2 - ceiling) over all eligible positions and runs = {worst:+.2e}')
    assert worst <= 1e-6, 'a probe score exceeds its rank ceiling'
    return rows


def main():
    check()
    out = json.load(open(OUT))['grammars'] if OUT.exists() else {}
    args = sys.argv[1:]
    if args != ['--table']:
        names = args or sorted({p.name.rsplit('_r', 1)[0] for p in RES.glob('*_r16_s0')})
        for name in names:
            out[name] = json.loads(json.dumps(run(name)))  # string keys, as when reloaded
            json.dump(dict(ks=KS, grammars=out), open(OUT, 'w'), indent=1)
    rows = table(out)
    json.dump(dict(ks=KS, grammars=out, cells=rows), open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
