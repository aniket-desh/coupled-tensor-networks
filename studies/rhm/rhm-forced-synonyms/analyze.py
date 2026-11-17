"""What an aligned TreeTN on `sym` encodes on its bonds, how its cores combine the codes, and what it does with a
forbidden pair. Every measure is taken in read coordinates, so it does not depend on the basis of any bond.

    PYTHONPATH=. python studies/rhm/rhm-forced-synonyms/analyze.py runs/rhm-forced-synonyms studies/rhm/rhm-forced-synonyms/results
    PYTHONPATH=. python studies/rhm/rhm-forced-synonyms/analyze.py --check

Notation (aligned tree, studies/rhm/toy.py): h_L, h_R are the level-1 bonds of the pairs (x_1 x_2), (x_3 x_4); the
level-2 core gives the half's bond h_P = W(h_L, h_R); m is the constant bond of the masked half; and
    read(h) = center(C(h, m))  in R^3
is the centered logit vector the output computes from a half bond h. read is linear, and so is h_L -> read(W(h_L, e))
for a fixed right bond e. The Bayes target is F(A) = log(2/3)(e_A - 1/3), with |F| = F_NORM and |F(A) - F(A')| = F_DIST.
Each pair symbol b has two synonyms (two token pairs); mu_L[b], mu_R[c] are the means of their bonds.

  swap_L_all    max over symbols b and over all 6 right bonds e of |read(W(u - u', e))| / F_DIST, with u, u' the bonds
                of the two synonyms of b: the largest change of the output when a left pair is swapped for its
                synonym. It includes the 2 right bonds whose symbol equals b, which never occur beside b in data.
  swap_L_legal  the same over the 4 right bonds that do occur beside b (bounded by the prediction error).
  swap_R_*      the same with left and right exchanged.
  within_*      share of a bond's variance that lies within symbol classes, after whitening the bond over the 24
                prefixes (directions below 1e-3 of the largest singular value dropped): 0 when the bond is a function
                of the symbol. For the bonds L, R (pair symbol) and P (half symbol A). Whitening gives every kept
                direction equal weight, so at r >= 3 this is an indicator (it takes values near 0, 1/3, 1/2) of
                whether ANY direction holds the synonym, read by the output or not; swap_* measures what is read.
  keeps_synonym (r >= 3 reading, fixed before the run) max(swap_L_all, swap_R_all) > 1e-3 and more than 10 times
                max(swap_L_legal, swap_R_legal): the output would change on a synonym swap beside a sibling that never
                occurs, by more than the tree's own prediction error explains. Otherwise the tree has collapsed.
  comp_err      max over the 6 valid pairs (b != c) of |read(W(mu_L[b], mu_R[c])) - F(missing(b, c))| / F_DIST.
  comp_acc      share of the 6 valid pairs whose read-out has its smallest logit at the missing symbol.
  forb_rho[b]   |read(W(mu_L[b], mu_R[b]))| / F_NORM: the size of the response to the forbidden pair (b, b), in units
                of a valid response. The grammar's production tensor gives 0; the width-2 complex tree gives 1.
  forb_cos[b]   cosine between that response and F(b) (the complex tree gives 1: the pair reads out as b).
  forb_bond[b]  |W(mu_L[b], mu_R[b])| / mean over valid pairs of |W(mu_L[b], mu_R[c])|, in the model's own
                coordinates. NOT gauge invariant; reported because at r >= 3 a bond can respond in a direction the
                output does not read.
  tok_rho[b]    the same at the token level: max over the 6 right bonds e of
                |read(W(W1(leaf_1[b], leaf_2[b]), e))| / F_NORM for the forbidden token pair (b, b).
  xy_L, xy_R    (r = 2 only) the coefficients of mu[0] = x mu[1] + y mu[2]. Both are -1 when the three codes sum to
                zero, as in the complex tree.
  forb_pred[b]  (r = 2 only) the forbidden response an exact width-2 tree must have given its (x, y), from xy_L:
                rho_0 = sqrt(x^2 + y^2 - x y),  rho_1 = sqrt(1 + y + y^2) / |x|,  rho_2 = sqrt(1 + x + x^2) / |y|.
                Derivation (EXPERIMENT.md, "Theory"): write a_b, c_b, p_A for the three codes on the left, right and
                half bonds, with W(a_b, c_c) = p_missing(b, c) on the six valid pairs and p_0 + p_1 + p_2 = 0. Expanding
                W(a_0, c_1) = p_2 and W(a_1, c_0) = p_2 with a_0 = x a_1 + y a_2 and c_0 = u c_1 + w c_2 gives
                x W(a_1, c_1) + y p_0 = p_2 = u W(a_1, c_1) + w p_0, so (u, w) = (x, y) and
                W(a_1, c_1) = (p_2 - y p_0) / x; likewise W(a_2, c_2) = (p_1 - x p_0) / y and W(a_0, c_0) = x p_2 + y p_1.
                The sizes follow from <F(A), F(A')> = -|F|^2 / 2. Corollary: every two of the three responses have
                a product of at least 3/4, so at most one of them is below sqrt(3)/2 in any exact width-2 tree.
  xy_shift      (r = 2 only, per run) the largest change of (x, y) between Adam's last update and the polished tree.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from studies.rhm import toy as Y

torch.set_num_threads(1)
POP = Y.population('sym')
FB = torch.log(torch.tensor(2 / 3, dtype=torch.float64)) * (torch.eye(3, dtype=torch.float64) - 1 / 3)  # FB[A] = F(A)


def within(h, cls):
    """Within-class share of the whitened bond's variance over the 24 prefixes."""
    hc = h - h.mean(0)
    U, S, _ = torch.linalg.svd(hc, full_matrices=False)
    w = U[:, S > 1e-3 * S.max()]                       # whitened coordinates of each prefix
    res = sum((w[cls == a] - w[cls == a].mean(0)).pow(2).sum() for a in range(3))
    return (res / w.pow(2).sum()).item()


def measures(model):
    x, p = POP['x'], POP['p']
    with torch.no_grad():
        logits, (z0, z1, z2) = model(x, acts=True)
        hL, hR, hP, m = z1[:, 0], z1[:, 1], z2[:, 0], z2[0, 1]
        W1, W2, C = model.W[0][0], model.W[1][0], model.C
        read = lambda h: Y.center(torch.einsum('oab,...a,b->...o', C, h, m))
        merge = lambda a, b: torch.einsum('abc,...b,...c->...a', W2, a, b)
        out = dict(kl=Y.excess(logits, p).item(), max_err=(torch.softmax(logits, -1) - p).abs().max().item())
        out['gap_closed'] = 1 - out['kl'] / Y.UNIFORM_KL

        def by_symbol(h, cls, tok):
            """{symbol: (2, r) bonds of its two synonyms}, and each synonym's symbol-ordered list of all 6 bonds."""
            d = {}
            for a in range(3):
                keys = sorted({tuple(t.tolist()) for t in tok[cls == a]})
                assert len(keys) == 2
                d[a] = torch.stack([h[(tok == torch.tensor(k)).all(1)][0] for k in keys])
            return d

        sL, sR = by_symbol(hL, POP['cL'], x[:, 0:2]), by_symbol(hR, POP['cR'], x[:, 2:4])
        for side, mine, other, f in (('L', sL, sR, lambda d, e: read(merge(d, e))), ('R', sR, sL, lambda d, e: read(merge(e, d)))):
            eff = {(b, c): max(f(mine[b][0] - mine[b][1], e).norm().item() for e in other[c]) / Y.F_DIST
                   for b in range(3) for c in range(3)}
            out[f'swap_{side}_all'] = max(eff.values())
            out[f'swap_{side}_legal'] = max(v for (b, c), v in eff.items() if b != c)
        out['within_L'], out['within_R'], out['within_P'] = within(hL, POP['cL']), within(hR, POP['cR']), within(hP, POP['A'])
        s_all, s_legal = max(out['swap_L_all'], out['swap_R_all']), max(out['swap_L_legal'], out['swap_R_legal'])
        out['keeps_synonym'] = bool(s_all > 1e-3 and s_all > 10 * s_legal)

        muL, muR = torch.stack([sL[b].mean(0) for b in range(3)]), torch.stack([sR[c].mean(0) for c in range(3)])
        g = merge(muL[:, None], muR[None])               # (3, 3, r): W(mu_L[b], mu_R[c])
        rd = read(g)                                     # (3, 3, 3)
        valid = [(b, c) for b in range(3) for c in range(3) if b != c]
        miss = lambda b, c: (-(b + c)) % 3
        out['comp_err'] = max((rd[b, c] - FB[miss(b, c)]).norm().item() for b, c in valid) / Y.F_DIST
        out['comp_acc'] = float(np.mean([rd[b, c].argmin().item() == miss(b, c) for b, c in valid]))
        out['forb_rho'] = [rd[b, b].norm().item() / Y.F_NORM for b in range(3)]
        out['forb_cos'] = [(rd[b, b] @ FB[b] / (rd[b, b].norm() * FB[b].norm()).clamp_min(1e-300)).item() for b in range(3)]
        vnorm = np.mean([g[b, c].norm().item() for b, c in valid])
        out['forb_bond'] = [g[b, b].norm().item() / vnorm for b in range(3)]
        tok = torch.einsum('abc,vb,vc->va', W1, model.A[0, :3], model.A[1, :3])          # W1(leaf_1[b], leaf_2[b])
        allR = torch.cat([sR[c] for c in range(3)])
        out['tok_rho'] = [max(read(merge(tok[b], e)).norm().item() for e in allR) / Y.F_NORM for b in range(3)]
        if model.r == 2:
            for side, mu in (('L', muL), ('R', muR)):
                out[f'xy_{side}'] = torch.linalg.solve(mu[1:].T, mu[0]).tolist()
            x_, y_ = out['xy_L']
            out['forb_pred'] = [(x_ ** 2 + y_ ** 2 - x_ * y_) ** 0.5, (1 + y_ + y_ ** 2) ** 0.5 / abs(x_), (1 + x_ + x_ ** 2) ** 0.5 / abs(y_)]
    return out


# ------------------------------------------------------------------------------------------ verdicts

N_PLANNED, N_MIN = 160, 150
GATE = 10          # exact width-2 runs needed to judge H2-H4
N_WIDE = 5         # exact runs a width >= 3 needs for its descriptive counts to be read


def verdicts(rows, n_err=0):
    """The criteria of EXPERIMENT.md on the polished trees. rows: finite runs only, [dict(cfg..., polished measures)]."""
    r1 = [r for r in rows if r['r'] == 1]
    r2 = [r for r in rows if r['r'] == 2]
    ex = [r for r in r2 if r['kl'] < Y.EXACT]
    exa = [r for r in ex if r['kl_adam'] < Y.ADAM_EXACT]     # found by Adam; the polish only refined them
    by_init = lambda rs: {i: sum(r['init'] == i for r in rs) for i in ('variance', 'identity')}
    v = {'n_runs': len(rows), 'n_errored': n_err, 'run_complete': len(rows) >= N_MIN,
         'n_r2': len(r2), 'n_r2_exact': len(ex), 'n_r2_exact_by_init': by_init(ex),
         'n_r2_exact_found_by_adam': len(exa), 'n_r2_exact_found_by_adam_by_init': by_init(exa),
         'n_between_thresholds': sum(Y.EXACT <= r['kl'] <= Y.NOT_EXACT for r in rows),
         'n_polish_capped': sum(r['polish_capped'] for r in rows)}
    v['H1 width 1 cannot reach Bayes'] = ('supported' if r1 and min(r['kl'] for r in r1) > Y.NOT_EXACT else
                                           'not supported' if r1 else 'no runs')
    v['H1 detail'] = dict(n=len(r1), min_kl=min((r['kl'] for r in r1), default=None))
    names = ('H2 synonyms collapse and codes compose', 'H3 no forbidden pair is silenced by training',
             'H4 the forbidden response is fixed by (x, y)')
    if len(ex) < GATE:
        for k in names:
            v[k] = f'inconclusive (fewer than {GATE} exact width-2 runs)'
        return v
    keys = ('swap_L_all', 'swap_R_all', 'within_L', 'within_R', 'within_P', 'comp_err')
    worst = {k: max(ex, key=lambda r: r[k]) for k in keys}
    h2 = {k: dict(value=worst[k][k], run=worst[k]['run'], excess=worst[k]['kl'], largest_rho=max(worst[k]['forb_rho'])) for k in keys}
    h2['comp_acc'] = min(r['comp_acc'] for r in ex)
    v[names[0]] = 'supported' if all(h2[k]['value'] < 1e-3 for k in keys) and h2['comp_acc'] == 1.0 else 'not supported'
    v['H2 detail (worst exact width-2 run per measure)'] = h2

    # H3 is about what training selects, so it is judged on the runs Adam found; all exact runs are given beside it.
    def rho_stats(rs):
        rho = np.array([r['forb_rho'] for r in rs])
        if not len(rs):
            return dict(n_runs=0)
        lo = rho.min(1)
        return dict(n_runs=len(rs), smallest=float(rho.min()), median_of_per_run_smallest=float(np.median(lo)),
                    per_run_smallest_min_max=[float(lo.min()), float(lo.max())], share_of_runs_with_smallest_below_0p5=float((lo < 0.5).mean()),
                    pooled_median=float(np.median(rho)), pooled_max=float(rho.max()))
    sa, se = rho_stats(exa), rho_stats(ex)
    v[names[1]] = (f'inconclusive (fewer than {GATE} exact width-2 runs found by Adam)' if len(exa) < GATE else
                   'supported' if sa['smallest'] >= 0.01 and sa['median_of_per_run_smallest'] >= 0.5 else 'not supported')
    v['H3 detail'] = dict(found_by_adam=sa, all_exact=se)

    rho, pred = np.array([r['forb_rho'] for r in ex]), np.array([r['forb_pred'] for r in ex])
    xy = np.array([r['xy_L'] + r['xy_R'] for r in ex])
    rel = np.abs(pred - rho) / rho
    i = int(rel.max(1).argmax())
    h4 = dict(max_rel_err_rho=float(rel.max()), max_abs_diff_xy_L_R=float(np.abs(xy[:, :2] - xy[:, 2:]).max()),
              worst_run=ex[i]['run'], worst_run_excess=ex[i]['kl'], worst_run_largest_rho=float(rho[i].max()),
              smallest_pairwise_product_of_rho=float(min((rho[:, a] * rho[:, b]).min() for a, b in ((0, 1), (0, 2), (1, 2)))))
    v[names[2]] = 'supported' if h4['max_rel_err_rho'] < 1e-3 and h4['max_abs_diff_xy_L_R'] < 1e-3 else 'not supported'
    v['H4 detail'] = h4

    def xy_stats(rs):
        if not rs:
            return dict(n_runs=0)
        q = np.array([r['xy_L'] for r in rs])
        return dict(n_runs=len(rs), x_median_min_max=[float(np.median(q[:, 0])), float(q[:, 0].min()), float(q[:, 0].max())],
                    y_median_min_max=[float(np.median(q[:, 1])), float(q[:, 1].min()), float(q[:, 1].max())],
                    share_within_0p1_of_minus1=float((np.abs(q + 1).max(1) < 0.1).mean()),
                    xy_shift_by_polish_median_max=[float(np.median([r['xy_shift'] for r in rs])), float(max(r['xy_shift'] for r in rs))])
    v['descriptive: (x, y) of exact width-2 runs'] = dict(found_by_adam=xy_stats(exa), found_by_the_polish=xy_stats([r for r in ex if r not in exa]))

    wide = {}
    for r in sorted({q['r'] for q in rows if q['r'] >= 3}):
        for label, sel in (('exact', lambda q: q['kl'] < Y.EXACT), ('between_thresholds', lambda q: Y.EXACT <= q['kl'] <= Y.NOT_EXACT)):
            for init in ('variance', 'identity'):
                rs = [q for q in rows if q['r'] == r and q['init'] == init and sel(q)]
                rho_w = np.array([q['forb_rho'] for q in rs])
                wide[f'r{r}_{init}_{label}'] = dict(n=len(rs), keeps_synonym=sum(q['keeps_synonym'] for q in rs),
                                                    forb_rho_below_0p01=int((rho_w < 0.01).sum()) if len(rs) else 0,
                                                    forb_rho_median=float(np.median(rho_w)) if len(rs) else None)
        n_ex = sum(wide[f'r{r}_{i}_exact']['n'] for i in ('variance', 'identity'))
        wide[f'r{r}_answered'] = n_ex >= N_WIDE
    v['descriptive: widths 3 and 4'] = wide
    return v


def analyze(runs_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, traj, diverged, polished = [], [], [], {}
    for f in sorted(Path(runs_dir).glob('sym_aligned_*.pt')):
        run = torch.load(f)
        cfg = run['cfg']
        base = dict(run=f.stem, r=cfg['r'], init=cfg['init'], seed=cfg['seed'], kl_adam=run['kl_adam'], polish_capped=run['polish']['capped'])
        if not np.isfinite(run['kl_polished']):
            diverged.append(dict(base, kl=run['kl_polished']))     # counted as not exact, never measured
            continue
        polished[f.stem] = run['polished']
        row = {**base, **measures(Y.load_model(run))}
        last = None
        for step in sorted(run['ckpts']):
            if np.isfinite(Y.excess(Y.load_model(run, step)(POP['x']), POP['p']).item()):
                last = measures(Y.load_model(run, step))
                traj.append({**base, 'step': step, **last})
        if cfg['r'] == 2:
            row['xy_shift'] = float(np.abs(np.array(row['xy_L']) - np.array(last['xy_L'])).max())
        rows.append(row)
    n_err = len(list(Path(runs_dir).glob('*.err')))
    v = verdicts(rows, n_err)
    v['n_diverged'] = len(diverged)
    json.dump(rows, open(out_dir / 'runs.json', 'w'), indent=1)
    json.dump(diverged, open(out_dir / 'diverged.json', 'w'), indent=1)
    json.dump(traj, open(out_dir / 'trajectories.json', 'w'))
    json.dump(v, open(out_dir / 'summary.json', 'w'), indent=1)
    json.dump(Y.provenance(), open(out_dir / 'provenance.json', 'w'), indent=1)
    torch.save(polished, out_dir / 'polished_trees.pt')   # every measured tree, so the numbers can be recomputed from git

    med = lambda xs: (float(np.median(xs)), float(np.min(xs)), float(np.max(xs)))
    print(f'{len(rows)} finite runs, {len(diverged)} diverged, {n_err} errored. exact = polished excess loss below {Y.EXACT:g} nats; uniform is {Y.UNIFORM_KL:.6f}.')
    print('per width and init: n, n exact, polished excess [median (min, max)], and over the exact runs the median (min, max) of each measure')
    for r in sorted({q['r'] for q in rows}):
        for init in ('variance', 'identity'):
            cell = [q for q in rows if q['r'] == r and q['init'] == init]
            ex = [q for q in cell if q['kl'] < Y.EXACT]
            if not cell:
                continue
            print(f"r={r} {init:8s} n={len(cell)} exact={len(ex)} (Adam alone below {Y.ADAM_EXACT:g}: {sum(q['kl_adam'] < Y.ADAM_EXACT for q in ex)}) "
                  f"keeps_synonym={sum(q['keeps_synonym'] for q in ex)} excess %.2e (%.2e, %.2e) gap closed %.3f (%.3f, %.3f)"
                  % (med([q['kl'] for q in cell]) + med([q['gap_closed'] for q in cell])))
            for k in ('swap_L_all', 'swap_R_all', 'swap_L_legal', 'swap_R_legal', 'within_L', 'within_R', 'within_P',
                      'comp_err', 'comp_acc', 'forb_rho', 'forb_cos', 'forb_bond', 'tok_rho', 'xy_L', 'xy_R', 'forb_pred', 'xy_shift'):
                if ex and k in ex[0]:
                    print(f'    {k:13s} %.3g (%.3g, %.3g)' % med(np.array([q[k] for q in ex]).ravel()))
    print(json.dumps(v, indent=1))
    if not v['run_complete']:
        print(f'RUN FAILED: {len(rows)} of {N_PLANNED} runs produced a finite tree; {N_MIN} are needed')


# ------------------------------------------------------------------------------------------ check

def check():
    ok = True

    def c(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    z = measures(Y.exact_complex('aligned'))
    c('complex tree: exact, synonyms collapsed, 6/6 compositions', z['kl'] < 1e-14 and z['swap_L_all'] < 1e-12 and z['swap_R_all'] < 1e-12
      and max(z['within_L'], z['within_R'], z['within_P']) < 1e-12 and z['comp_err'] < 1e-12 and z['comp_acc'] == 1)
    c('complex tree: a forbidden pair (b, b) reads out as b at full size (rho = 1, cos = 1), also at the token level',
      np.allclose(z['forb_rho'], 1) and np.allclose(z['forb_cos'], 1) and np.allclose(z['tok_rho'], 1), (z['forb_rho'], z['tok_rho']))
    c('complex tree: the three codes sum to zero (x = y = -1), and forb_pred = 1', np.allclose(z['xy_L'] + z['xy_R'], -1)
      and np.allclose(z['forb_pred'], 1), z['xy_L'])
    g = measures(Y.regauge(Y.exact_complex('aligned'), seed=3))
    inv = [k for k in z if k != 'forb_bond']
    c('every measure except forb_bond is unchanged by a change of basis of all bonds',
      all(np.allclose(z[k], g[k], atol=1e-9) for k in inv) and not np.allclose(z['forb_bond'], g['forb_bond'], atol=1e-3),
      (z['forb_bond'], g['forb_bond']))
    o = measures(Y.exact_onehot('sym'))
    c('one-hot message-passing tree (width 3): exact, collapsed, 6/6, and forbidden pairs read out as zero',
      o['kl'] < 1e-14 and o['swap_L_all'] < 1e-12 and o['comp_acc'] == 1 and max(o['forb_rho']) < 1e-12 and max(o['tok_rho']) < 1e-12,
      o['forb_rho'])
    # A width-4 tree that is exact on data but keeps the synonym in a direction only a forbidden context reads: the
    # one-hot tree, with the token pair 21 (symbol 0) coded as e_0 + e_3 instead of e_0, and a level-2 core that reads
    # channel 3 only beside a right pair of symbol 0, which never occurs beside a left pair of symbol 0.
    m = Y.exact_onehot('sym', r=4)
    with torch.no_grad():
        m.W[0][0, 3, 2, 1] = 1.0
        m.W[1][0, 1, 3, 0] = 1.0
    s = measures(m)
    c('a tree that hides the synonym where only a forbidden context reads it: exact, swap_L_legal = 0, swap_L_all > 0, within_L > 0',
      s['kl'] < 1e-14 and s['swap_L_legal'] < 1e-12 and s['swap_L_all'] > 0.1 and s['within_L'] > 0.01 and s['swap_R_all'] < 1e-12,
      (s['swap_L_legal'], s['swap_L_all'], s['within_L']))
    rnd = measures(Y.build(2, 'aligned', seed=11))
    c('a random width-2 tree fails: synonyms differ, compositions wrong', rnd['swap_L_all'] > 0.05 and rnd['within_L'] > 0.05
      and rnd['comp_err'] > 0.05, (rnd['swap_L_all'], rnd['within_L'], rnd['comp_err']))
    xs = np.concatenate([-np.logspace(-4, 4, 161), np.logspace(-4, 4, 161)])
    X, Yg = np.meshgrid(xs, xs)
    R = np.stack([np.sqrt(X ** 2 + Yg ** 2 - X * Yg), np.sqrt(1 + Yg + Yg ** 2) / np.abs(X), np.sqrt(1 + X + X ** 2) / np.abs(Yg)])
    prod = min((R[a] * R[b]).min() for a, b in ((0, 1), (0, 2), (1, 2)))
    c('T5 corollary on a grid of (x, y): every two responses have product >= 3/4, so the second-smallest is >= sqrt(3)/2',
      prod >= 0.75 - 1e-9 and np.sort(R, 0)[1].min() >= 3 ** 0.5 / 2 - 1e-9, (prod, np.sort(R, 0)[1].min(), np.sort(R, 0)[0].max()))
    print('measure checks', 'passed' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        check()
    analyze(sys.argv[1], sys.argv[2])
