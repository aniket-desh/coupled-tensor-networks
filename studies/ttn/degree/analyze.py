"""Criteria for degree, from saved results only. Usage: PYTHONPATH=. python studies/ttn/degree/analyze.py <results>
Prediction: with leaf variance fraction w on the constant, each active patch multiplies the initial coefficient by
sqrt((1 - w) / (8 w)); if escape time scales as c0^-(13/15) (the depth-15 law tested in init-scale), then
d log(median t_half) / dD = (13/15) * 0.5 * log(8 w / (1 - w)): 0 at w = 1/9, 0.425 at w = 0.25, 0.901 at w = 0.5."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from studies.ttn.analysis import fmt, inf_nan, load, med_ci, slope_ci

res = Path(sys.argv[1])
out = {}
groups = sorted(p.stem for p in res.glob('w*_D*.json'))
by_w = {}
for g in groups:
    _, meta = load(res, g)
    h = np.array(meta['summary']['half'], float)[:, 0]
    w, D = meta['w'], meta['D']
    m, ci = med_ci(h)
    by_w.setdefault(round(w, 3), {})[D] = h
    out[g] = dict(w=w, D=D, learned=int(np.isfinite(h).sum()), n=len(h), median=m, ci=ci, stopped_by=meta['stopped_by'])
    print(f'w {w:.3f} D {D}: learned {out[g]["learned"]}/{len(h)}  median t_half {fmt(m, ci)}')

# Symmetry control: at w = 1/9 degree must not matter (same distribution at D = 1 and D = 8).
iso = by_w.get(round(1 / 9, 3), {})
if 1 in iso and 8 in iso:
    u = mannwhitneyu(inf_nan(iso[1]), inf_nan(iso[8]))
    out['isotropic_D1_vs_D8_mannwhitney_p'] = float(u.pvalue)
    print(f'isotropic D=1 vs D=8: Mann-Whitney p = {u.pvalue:.3g} (symmetry predicts no difference)')

# Slope of log median t_half against D for each biased w, over D where at least 12 of 16 seeds learned.
for w in (0.25, 0.5):
    Ds = sorted(D for D, h in by_w.get(w, {}).items() if np.isfinite(h).sum() >= 12)
    x = [D for D in Ds for _ in by_w[w][D]]
    y = [np.log(t) for D in Ds for t in inf_nan(by_w[w][D])]
    # Per-run regression (censored runs excluded from the fit; their count is reported per group above).
    sl, ci, n = slope_ci(x, y)
    pred = (13 / 15) * 0.5 * np.log(8 * w / (1 - w))
    out[f'slope_w{w}'] = dict(slope=sl, ci=ci, n=n, Ds=Ds, predicted=pred)
    print(f'w {w}: d log t_half / dD = {fmt(sl, ci)} over D {Ds} ({n} runs); predicted {pred:.3f}')
json.dump(out, open(res / 'analysis.json', 'w'), indent=1, default=float)
