"""Figures for sharing-location. Usage: PYTHONPATH=. python studies/ttn/sharing-location/plot.py <results> <figures>"""
import sys
from pathlib import Path

import numpy as np

from ctn.style import C1, INK, MUTED, plt
from studies.ttn.analysis import inf_nan
from studies.ttn.plotting import COLORS, alignment_panel, coefficient_panel, load, logy, mobility_panel, save

res, fig_dir = Path(sys.argv[1]), sys.argv[2]
fams = [f for f in ['none', 'leaf', 'split2', 'quarter', 'split4', 'half', 'quarter_sampled'] if (res / f'{f}.json').exists()]

fig, ax = plt.subplots(3, len(fams), figsize=(3.4 * len(fams), 9), constrained_layout=True, squeeze=False)
for j, f in enumerate(fams):
    log, meta = load(res, f)
    coefficient_panel(ax[0, j], log, meta, runs=range(min(6, meta['R'])), title=f"{f} ({meta.get('shared_cores', '?')} shared cores)")
    ax[0, j].set_xscale('log')
    mobility_panel(ax[1, j], log, meta, 0, 1)
    alignment_panel(ax[2, j], log, meta, 1)
save(fig, fig_dir, 'families')

# Seed-paired c2 half-times against the no-sharing family, and medians against shared cores.
if 'none' not in fams:
    print('no none group: the paired figure is skipped')
    sys.exit(0)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
_, mn = load(res, 'none')
hn = inf_nan(np.array(mn['summary']['half'], float)[:, 1])
top = np.nanmax(np.where(np.isfinite(hn), hn, np.nan)) * 1.3 if np.isfinite(hn).any() else 1.0
for j, f in enumerate([f for f in fams if f not in ('none', 'quarter_sampled')]):
    _, m = load(res, f)
    h = inf_nan(np.array(m['summary']['half'], float)[:, 1])
    ax[0].scatter(np.where(np.isinf(hn), top, hn), np.where(np.isinf(h), top, h), s=14, color=COLORS[j % 4] if j < 4 else INK,
                  marker='os^vD'[j % 5], label=f)
ax[0].plot([1, top], [1, top], color=MUTED, lw=0.8)
ax[0].set(xscale='log', yscale='log', xlabel=r'$t_{1/2}(c_2)$ in none (censored at top)', ylabel=r'$t_{1/2}(c_2)$ in family',
          title='Seed-paired: below the diagonal = faster than none')
ax[0].legend(fontsize=7)
cores, med = [], []
for f in [f for f in fams if f != 'quarter_sampled']:
    _, m = load(res, f)
    cores.append(m['shared_cores'])
    med.append(np.median(inf_nan(np.array(m['summary']['half'], float)[:, 1])))
fin = np.isfinite(med)
ax[1].plot(np.array(cores)[fin], np.array(med)[fin], 'o', color=C1)
for c, mm, f in zip(cores, med, [f for f in fams if f != 'quarter_sampled']):
    if np.isfinite(mm):
        ax[1].annotate(f, (c, mm), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax[1].set(xlabel='cores y1 and y2 share at the teacher', ylabel=r'median $t_{1/2}(c_2)$ (finite only)', title='Amount of sharing')
logy(ax[1], med)
save(fig, fig_dir, 'paired_half_times')
