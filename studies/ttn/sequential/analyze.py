"""Criteria for sequential, from saved results only. Usage: PYTHONPATH=. python studies/ttn/sequential/analyze.py <results>
Phase-2 half-times are measured from the start of phase 2. Seeds that failed the phase-1 gate are excluded from
every arm of that seed (and counted). The fresh-tree baseline is pooled per s2 across families: a fresh isotropic
tree on any of these y2 is the same problem (basis-permutation symmetry)."""
import json
import sys
from pathlib import Path

import numpy as np

from studies.ttn.analysis import fmt, load, med_ci, paired_sign, ratio_ci

res = Path(sys.argv[1])
out = {}
_, m1 = load(res, 'phase1')
gate = np.array(m1['gate_ok'], bool)
out['phase1'] = dict(gate_passed=int(gate.sum()), n=len(gate), h1=med_ci(np.array(m1['summary']['half'], float)[:, 0]))
print(f"phase 1: {gate.sum()}/{len(gate)} seeds passed the gate; median h1 {fmt(*out['phase1']['h1'])}")

H, interf = {}, {}
for p in sorted(res.glob('*_s2*.json')):
    log, meta = load(res, p.stem)
    S, s2, seeds, arms = meta['S'], meta['s2'], meta['seeds'], meta['arms']
    h = np.array(meta['summary']['half'], float)[:, 1]
    c1 = log['c'][:, :, 0]  # (T, R)
    for a, arm in enumerate(arms):
        idx = np.arange(a * seeds, (a + 1) * seeds)
        H[(S, s2, arm)] = np.where(gate, h[idx], np.nan) if arm != 'fresh' else h[idx]
        if arm != 'fresh':
            interf[(S, s2, arm)] = np.abs(c1[:, idx][:, gate] - 1.0).max(0)
for s2 in (0.5, 1.0):
    fresh = np.concatenate([H[(S, s2, 'fresh')] for S in ('none', 'quarter', 'half') if (S, s2, 'fresh') in H])
    out[f'fresh_s2{s2}'] = dict(h2=med_ci(fresh), n=len(fresh))
    print(f's2 {s2}: fresh tree pooled median h2 {fmt(*med_ci(fresh))} (n={len(fresh)})')
    for S in ('none', 'quarter', 'half'):
        for arm in ('all', 'freeze_q1', 'freeze_left'):
            if (S, s2, arm) not in H:
                continue
            h = H[(S, s2, arm)][gate]
            m, ci = med_ci(h)
            out[f'{S}_s2{s2}_{arm}'] = dict(h2=[m, ci], learned=int(np.isfinite(h).sum()), n=int(gate.sum()),
                                            transfer_vs_fresh=m / out[f'fresh_s2{s2}']['h2'][0],
                                            interference_median=float(np.median(interf[(S, s2, arm)])))
            print(f"  {S:8s} {arm:12s} learned {np.isfinite(h).sum()}/{gate.sum()}  h2 {fmt(m, ci)}  "
                  f"/fresh {out[f'{S}_s2{s2}_{arm}']['transfer_vs_fresh']:.3g}  peak|c1-1| {out[f'{S}_s2{s2}_{arm}']['interference_median']:.3g}")
    # Family contrast for the 'all' arm, paired by phase-1 seed.
    for S in ('quarter', 'half'):
        if (S, s2, 'all') in H and ('none', s2, 'all') in H:
            w, l, p = paired_sign(H[(S, s2, 'all')][gate], H[('none', s2, 'all')][gate])
            r, ci = ratio_ci(H[(S, s2, 'all')][gate], H[('none', s2, 'all')][gate])
            out[f'{S}_vs_none_all_s2{s2}'] = dict(faster=w, slower=l, p=p, median_ratio=r, ci=ci)
            print(f'  {S} vs none (all arm): faster {w}, slower {l}, sign p {p:.3g}; paired ratio {fmt(r, ci)}')
json.dump(out, open(res / 'analysis.json', 'w'), indent=1, default=float)
