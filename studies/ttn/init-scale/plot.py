"""Figures for init-scale from its results folder. Usage: PYTHONPATH=. python studies/ttn/init-scale/plot.py <results> <figures>"""
import json
import sys
from pathlib import Path

import numpy as np

from ctn.style import C1, C2, INK, MUTED, plt
from studies.ttn.analysis import inf_nan
from studies.ttn.plotting import alignment_panel, coefficient_panel, load, logy, save

res, fig_dir = Path(sys.argv[1]), sys.argv[2]

# Activation second moments at init (descriptive).
rows = json.load(open(res / 'activations.json'))['rows']
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
for layer, col in zip(['leaf', 'quarter', 'half', 'out'], [C1, C2, MUTED, INK]):
    q = sorted((r['r'], r[f'{layer}_m2_q']) for r in rows if r['alpha'] == 1.0)
    rs, qq = [x for x, _ in q], np.array([y for _, y in q])
    ax[0].errorbar(rs, qq[:, 1], yerr=[qq[:, 1] - qq[:, 0], qq[:, 2] - qq[:, 1]], marker='o', capsize=2, color=col, label=layer)
ax[0].axhline(1, color=MUTED, lw=0.8, ls='--')
ax[0].set(xscale='log', xlabel='bond dimension $r$', ylabel='second moment (median, 5-95% over inits)', title=r'$\alpha=1$: activation scale by layer')
ax[0].legend()
for r_ in sorted({r['r'] for r in rows}):
    q = [r for r in rows if r['r'] == r_]
    ax[1].plot([r['alpha'] for r in q], [r['out_m2_mean'] for r in q], marker='o', label=f'r={r_}')
al = np.linspace(0.5, 1, 50)
ax[1].plot(al, al ** 30, color=INK, ls='--', label=r'$\alpha^{30}$')
ax[1].set(yscale='log', xlabel=r'init scale $\alpha$ (every core)', ylabel='output second moment', title='Output scale against $\\alpha$')
ax[1].legend(fontsize=7)
save(fig, fig_dir, 'activations')

# Coefficients per alpha group (first 8 seeds), log time.
groups = sorted((p.stem for p in res.glob('alpha*_eta*.json')), key=lambda g: (-float(g[5:].split('_')[0]), g))
if not groups:  # no training group finished: the activation figure is all there is
    sys.exit('no training groups saved; only figures/activations.pdf written')
eta = max(float(g.split('_eta')[1]) for g in groups)
fig, ax = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.2), constrained_layout=True, squeeze=False)
for i, g in enumerate(groups):
    log, meta = load(res, g)
    coefficient_panel(ax[0, i], log, meta, runs=range(min(8, meta['R'])), title=f"{g} ({meta['censored_runs']}/{meta['R']} censored)")
    ax[0, i].set_xscale('log')
save(fig, fig_dir, 'training')

# Depth law at alpha = 1: per-seed half-time against the seed's own initial coefficient.
main = f'alpha1.0_eta{eta}'
log, meta = load(res, main)
h1 = np.array(meta['summary']['half'], float)[:, 0]
c0 = np.array(meta['summary']['c_start'], float)[:, 0]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
ok = np.isfinite(h1)
ax[0].scatter(c0[ok], h1[ok], s=14, color=C1, label='learned')
ax[0].scatter(c0[~ok], np.full((~ok).sum(), log['t'][-1]), s=14, marker='^', color=MUTED, label='censored (at $t_{end}$)')
if ok.sum() >= 3:
    xs = np.geomspace(c0[ok].min(), c0[ok].max(), 20)
    ref = np.exp(np.mean(np.log(h1[ok]) + 13 / 15 * np.log(c0[ok])))
    ax[0].plot(xs, ref * xs ** (-13 / 15), color=INK, ls='--', label='slope $-13/15$ (depth 15)')
ax[0].set(xscale='log', xlabel=r'$|c_1(0)|$', ylabel=r'$t_{1/2}(c_1)$', title=r'Depth law at $\alpha=1$, one point per seed')
logy(ax[0], h1)
ax[0].legend(fontsize=7)
alignment_panel(ax[1], log, meta, 0)
save(fig, fig_dir, 'depth_law')

# Median half-time per alpha (censored counted as +inf; infinite medians are not drawn and are reported in analysis).
fig, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=True)
meds = []
for k, col in enumerate([C1, C2, MUTED]):
    pts = []
    for g in groups:
        _, m = load(res, g)
        if float(g.split('_eta')[1]) == eta:
            pts.append((m['alpha'], float(np.median(inf_nan(np.array(m['summary']['half'], float)[:, k])))))
    pts = sorted(p for p in pts if np.isfinite(p[1]))
    meds += [v for _, v in pts]
    if pts:
        ax.plot(*zip(*pts), marker='o', color=col, label=f'$c_{k + 1}$')
ax.set(xscale='log', xlabel=r'$\alpha$', ylabel='median half-time (finite only)', title='Plateau length against init scale')
logy(ax, meds)
ax.legend(fontsize=7)
save(fig, fig_dir, 'half_time_vs_alpha')
