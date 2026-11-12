"""Figures for degree. Usage: PYTHONPATH=. python studies/ttn/degree/plot.py <results> <figures>"""
import sys
from pathlib import Path

import numpy as np

from ctn.style import C1, C2, INK, MUTED, plt
from studies.ttn.analysis import inf_nan
from studies.ttn.plotting import load, logy, save

res, fig_dir = Path(sys.argv[1]), sys.argv[2]
groups = sorted(p.stem for p in res.glob('w*_D*.json'))
COL = {0.111: MUTED, 0.25: C1, 0.5: C2}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
allh = []
for wi, w in enumerate(sorted({round(load(res, g)[1]['w'], 3) for g in groups})):
    col, pts = COL.get(w, INK), []
    for g in groups:
        log, m = load(res, g)
        if round(m['w'], 3) != w:
            continue
        h = inf_nan(np.array(m['summary']['half'], float)[:, 0])
        allh += list(h[np.isfinite(h)])
        x = np.full(len(h), m['D']) + 0.1 * (wi - 1)
        ax[0].scatter(x[np.isfinite(h)], h[np.isfinite(h)], s=10, color=col, alpha=0.6)
        ax[0].scatter(x[~np.isfinite(h)], np.full((~np.isfinite(h)).sum(), log['t'][-1]), s=12, marker='^', color=col, alpha=0.6)
        pts.append((m['D'], float(np.median(h))))
    fin = sorted((d, v) for d, v in pts if np.isfinite(v))
    if fin:
        ax[0].plot(*zip(*fin), color=col, label=f'w = {w:g} (median; triangles censored)')
        D0, v0 = fin[0]
        Ds = np.array(sorted(d for d, _ in pts))
        pred = (13 / 15) * 0.5 * np.log(8 * w / (1 - w))
        ax[1].plot(Ds, np.log(v0) + pred * (Ds - D0), color=col, ls='--', lw=1)
        ax[1].plot([d for d, _ in fin], [np.log(v) for _, v in fin], 'o-', color=col, label=f'w = {w:g}')
ax[0].set(xlabel='patches the product uses, $D$', ylabel=r'$t_{1/2}$', title='Learning time per seed')
logy(ax[0], allh)
ax[0].legend(fontsize=7)
ax[1].set(xlabel='$D$', ylabel=r'log median $t_{1/2}$', title='Measured (solid) vs depth-15 prediction (dashed)')
ax[1].legend(fontsize=7)
save(fig, fig_dir, 'degree')
