"""Criteria for init-scale, from saved results only. Usage: PYTHONPATH=. python studies/ttn/init-scale/analyze.py <results>
Writes <results>/analysis.json and prints the tables the write-up uses."""
import json
import sys
from pathlib import Path

import numpy as np

from studies.ttn.analysis import fmt, inf_nan, load, med_ci, ratio_ci, slope_ci

res = Path(sys.argv[1])
out = {}

# Part A, descriptive: second moments at alpha = 1.
rows = json.load(open(res / 'activations.json'))['rows']
out['part_a_alpha1'] = {r['r']: {k: r[f'{k}_m2_mean'] for k in ('leaf', 'quarter', 'half', 'out')} for r in rows if r['alpha'] == 1.0}

groups = sorted(p.stem for p in res.glob('alpha*_eta*.json'))
eta = max(float(g.split('_eta')[1]) for g in groups)
main = {float(g[5:].split('_')[0]): g for g in groups if float(g.split('_eta')[1]) == eta}
tab = {}
for al, g in sorted(main.items(), reverse=True):
    log, meta = load(res, g)
    h = np.array(meta['summary']['half'], float)
    tab[al] = dict(learned=[int(np.isfinite(h[:, k]).sum()) for k in range(3)], n=len(h),
                   **{f'h{k + 1}': med_ci(h[:, k]) for k in range(3)}, stopped_by=meta['stopped_by'])
    print(f"alpha {al}: learned {tab[al]['learned']}/{len(h)}  median h1 {fmt(*tab[al]['h1'])}  h2 {fmt(*tab[al]['h2'])}  h3 {fmt(*tab[al]['h3'])}")
out['by_alpha'] = tab

# Sanity: homogeneity forces median h1 ~ alpha^-13 as alpha -> 0 (not a test of depth).
ok = [al for al in tab if tab[al]['learned'][0] >= 0.75 * tab[al]['n']]
if len(ok) >= 3:
    p, ci, n = slope_ci(np.log(ok), [np.log(tab[al]['h1'][0]) for al in ok], n_boot=1)
    out['homogeneity_exponent'] = dict(p=-p, alphas=ok, expected_small_alpha=13)
    print(f'half-time exponent over alpha {ok}: p = {-p:.2f} (homogeneity predicts 13 as alpha -> 0)')

# Depth law, the real test: at alpha = 1, per-seed log h1 against log |c1(0)|. A depth-n product escaping from c0
# predicts slope -(1 - 2/n): -13/15 = -0.867 for n = 15, -0.6 for n = 5, -0.33 for n = 3.
_, meta = load(res, main[1.0])
h1 = np.array(meta['summary']['half'], float)[:, 0]
c0 = np.array(meta['summary']['c_start'], float)[:, 0]
sl, ci, n = slope_ci(np.log(c0), np.log(h1))
out['depth_law'] = dict(slope=sl, ci=ci, n=n, depth15=-13 / 15)
print(f'depth law: d log h1 / d log|c1(0)| = {fmt(sl, ci)} over {n} learned seeds (depth 15 predicts -0.867)')

# Discretization control: eta/2 against eta, paired by seed.
half_eta = [g for g in groups if float(g.split('_eta')[1]) < eta]
if half_eta:
    _, m2 = load(res, half_eta[0])
    r, rci = ratio_ci(np.array(m2['summary']['half'], float)[:, 0], h1)
    out['discretization'] = dict(median_ratio=r, ci=rci)
    print(f'eta/2 vs eta, paired h1 ratio: {fmt(r, rci)} (criterion: within 2% of 1)')

# Ordering at alpha = 1 (descriptive; sharing's role is tested in sharing-location).
h = inf_nan(np.array(meta['summary']['half'], float))
both = np.isfinite(h[:, 1]) | np.isfinite(h[:, 2])
out['c2_before_c3_alpha1'] = [int((h[both, 1] < h[both, 2]).sum()), int(both.sum())]
print(f"c2 before c3 at alpha = 1: {out['c2_before_c3_alpha1'][0]}/{out['c2_before_c3_alpha1'][1]} seeds with either learned")
json.dump(out, open(res / 'analysis.json', 'w'), indent=1, default=float)
