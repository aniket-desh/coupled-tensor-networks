"""Criteria for sharing-location, from saved results only.
Usage: PYTHONPATH=. python studies/ttn/sharing-location/analyze.py <results>  (writes <results>/analysis.json)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ctn.ttn import GROUPS
from studies.ttn.analysis import drive, fmt, load, med_ci, paired_sign, ratio_ci

res = Path(sys.argv[1])
fams = [f for f in ['none', 'leaf', 'split2', 'quarter', 'split4', 'half'] if (res / f'{f}.json').exists()]
out, H, K22s = {}, {}, {}
for f in fams:
    log, meta = load(res, f)
    h = np.array(meta['summary']['half'], float)
    H[f] = h
    # Drive decomposition of c2 over its rise (runs that reached the half-time), as fractions of the actual rise:
    # own K22 e2, cross from y1 K21 e1, cross from y3 K23 e3, and the pull of non-teacher coefficients ('rest').
    # 'resolved' = integral of the exact velocity / actual change (about 1 when the records resolve the rise).
    d = [drive(log, meta, r, 1) for r in np.nonzero(np.isfinite(h[:, 1]))[0]]
    frac = lambda key: float(np.median([x[key] / x['actual'] for x in d])) if d else float('nan')
    drv = dict(n=len(d), own=frac(1), cross_y1=frac(0), cross_y3=frac(2), rest=frac('rest'), resolved=frac('total'))
    # Own mobility K22 when c2 first reaches 2% of s2 (a fixed point on the rise, so families compare at equal c2),
    # total and the share contributed by each subtree group. Once y1 is learned, e1 ~ 0 and the cross term vanishes
    # whatever K21 is, so sharing can only speed y2 through a larger K22.
    K22 = np.full((len(h), len(GROUPS)), np.nan)
    s2 = meta['comps'][1]['s']
    for r in range(len(h)):
        up = np.nonzero(log['c'][:, r, 1] / s2 >= 0.02)[0]
        if len(up):
            K22[r] = log['K'][np.searchsorted(log['t_K'], log['t'][up[0]]).clip(max=len(log['t_K']) - 1), r, 1, 1]
    Kfrac = {g: float(np.nanmedian(K22[:, i] / K22.sum(1))) if np.isfinite(K22).any() else float('nan') for i, g in enumerate(GROUPS)}
    K22s[f] = K22.sum(1)
    out[f] = dict(shared_cores=meta['shared_cores'], learned=[int(np.isfinite(h[:, k]).sum()) for k in range(3)], n=len(h),
                  h1=med_ci(h[:, 0]), h2=med_ci(h[:, 1]), h3=med_ci(h[:, 2]), lag=med_ci(h[:, 1] - h[:, 0]),
                  drive_c2=drv, K22_at_rise=float(np.nanmedian(K22.sum(1))) if np.isfinite(K22).any() else float('nan'),
                  K22_share_by_subtree=Kfrac,
                  stopped_by=meta['stopped_by'])
    print(f"{f:8s} cores {meta['shared_cores']}  learned {out[f]['learned']}/{len(h)}  h2 {fmt(*out[f]['h2'])}  "
          f"lag h2-h1 {fmt(*out[f]['lag'])}  "
          f"c2 rise ({drv['n']} runs): own {drv['own']:.2f} cross-y1 {drv['cross_y1']:.2f} cross-y3 {drv['cross_y3']:.2f} "
          f"rest {drv['rest']:.2f} resolved {drv['resolved']:.2f}\n          K22 at c2 = 0.02 s2: {out[f]['K22_at_rise']:.3g}, "
          f"share by subtree {({g: round(v, 2) for g, v in Kfrac.items()})}")

# Primary: seed-paired h2 against the no-sharing family.
out['vs_none'] = {}
for f in [f for f in fams if f != 'none' and 'none' in fams]:
    w, l, p = paired_sign(H[f][:, 1], H['none'][:, 1])
    r, ci = ratio_ci(H[f][:, 1], H['none'][:, 1])
    kw, kl, kp = paired_sign(-K22s[f], -K22s['none'])  # larger K22 counts as a win
    out['vs_none'][f] = dict(faster=w, slower=l, p=p, median_ratio=r, ci=ci, K22_larger=kw, K22_smaller=kl, K22_p=kp)
    print(f'{f:8s} vs none: h2 faster in {w}, slower in {l} (sign p = {p:.3g}); paired ratio {fmt(r, ci)}; '
          f'K22 at c2 = 0.02 s2 larger in {kw}, smaller in {kl} (p = {kp:.3g})')

# Extent: does the median h2 fall as the number of shared cores grows (Spearman over the 6 families)?
cores = [out[f]['shared_cores'] for f in fams]
med = [out[f]['h2'][0] for f in fams]
rho = spearmanr(cores, med).statistic if len(fams) > 2 else float('nan')
out['spearman_cores_vs_h2'] = float(rho)
print(f'Spearman(shared cores, median h2) = {rho:.2f}')

# Location at equal extent (descriptive): subtree against scattered leaves.
for a, b in [(a, b) for a, b in [('quarter', 'split2'), ('half', 'split4')] if a in H and b in H]:
    w, l, p = paired_sign(H[a][:, 1], H[b][:, 1])
    out[f'{a}_vs_{b}'] = dict(faster=w, slower=l, p=p)
    print(f'{a} vs {b}: h2 faster in {w}, slower in {l} (sign p = {p:.3g})')

# Finite-data replicate: sampled quarter against population quarter, paired by seed (same init).
if (res / 'quarter_sampled.json').exists():
    _, ms = load(res, 'quarter_sampled')
    hs = np.array(ms['summary']['half'], float)
    out['sampled_vs_population'] = {f'h{k + 1}': ratio_ci(hs[:, k], H['quarter'][:, k]) for k in range(3)}
    print('sampled / population quarter, paired ratios:', {k: fmt(*v) for k, v in out['sampled_vs_population'].items()})
json.dump(out, open(res / 'analysis.json', 'w'), indent=1, default=float)
