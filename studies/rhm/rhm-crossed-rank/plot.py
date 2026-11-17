"""Figures for rhm-crossed-rank, from the saved results only.

    PYTHONPATH=. python studies/rhm/rhm-crossed-rank/plot.py studies/rhm/rhm-crossed-rank/results studies/rhm/rhm-crossed-rank/figures
"""
import json
import sys
from pathlib import Path

import numpy as np

from studies.rhm import toy as Y
from studies.rhm.plotting import CATEGORICAL, INK2, plt, save

FLOOR = 1e-18
COLOR = {'aligned': CATEGORICAL[0], 'crossed': CATEGORICAL[1]}
MARK = {'variance': 'o', 'identity': '^'}
clip = lambda v: np.maximum(np.asarray(v, dtype=float), FLOOR)


def main(res, figs):
    rows = [q for q in json.load(open(Path(res) / 'runs.json')) if q['default_rate']]   # the rate arm is in summary.json only
    summ = json.load(open(Path(res) / 'summary.json'))
    curves = json.load(open(Path(res) / 'curves.json'))
    bounds = summ['verdicts']['bounds']
    widths = sorted({q['r'] for q in rows})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, name in zip(axes, ('mod', 'sym')):
        for q in rows:
            if q['grammar'] == name:
                dx = (-0.18 if q['grouping'] == 'aligned' else 0.18) + 0.012 * (q['seed'] % 10 - 4.5) + (0.0 if q['init'] == 'variance' else 0.0)
                ax.plot(q['r'] + dx, clip(q['kl']), MARK[q['init']], ms=2.5, color=COLOR[q['grouping']], alpha=0.7, mew=0)
        for g in ('aligned', 'crossed'):
            b = bounds[f'{name}_{g}']
            ax.axvline(b - 0.5 + (-0.03 if g == 'aligned' else 0.03), color=COLOR[g], lw=0.8, ls='--')
        ax.axhline(Y.UNIFORM_KL, color=INK2, lw=0.6, ls=':')
        ax.axhline(Y.EXACT, color=INK2, lw=0.6, ls='--')
        ax.set_yscale('log')
        ax.set_xticks(widths)
        ax.set_xlabel('bond dimension r')
        ax.set_title(f"grammar `{name}`: exact needs r >= {bounds[name + '_aligned']} (aligned), {bounds[name + '_crossed']} (crossed)", fontsize=7)
    axes[0].set_ylabel('excess loss after the polish (nats)')
    for g, c in COLOR.items():
        axes[1].plot([], [], 'o', ms=3, color=c, label=g)
    for i, m in MARK.items():
        axes[1].plot([], [], m, ms=3, color=INK2, label=f'{i} init')
    axes[1].plot([], [], '--', color=INK2, lw=0.8, label='width lower bound (colored), exact threshold (gray)')
    axes[1].plot([], [], ':', color=INK2, lw=0.6, label='uniform predictor')
    axes[1].legend(fontsize=5.5, loc='center right')
    save(fig, Path(figs) / 'capacity.pdf')

    # training curves on mod at the widths around the two bounds
    show = [r for r in (2, 3, 4) if r in widths]
    fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 2.6), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], show):
        for q in rows:
            if q['grammar'] == 'mod' and q['r'] == r:
                st, kl = zip(*curves[q['run']])
                ax.plot(np.maximum(st, 50), clip(kl), color=COLOR[q['grouping']], lw=0.5, alpha=0.6,
                        ls='-' if q['init'] == 'variance' else '--')
                ax.plot(st[-1] * 1.6, clip(q['kl']), 'o', ms=2, color=COLOR[q['grouping']], alpha=0.7, mew=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Adam update (dots: after the polish)')
        ax.set_title(f'mod, r = {r}', fontsize=7)
    axes[0][0].set_ylabel('excess loss (nats)')
    for g, c in COLOR.items():
        axes[0][-1].plot([], [], '-', color=c, label=g)
    axes[0][-1].legend(fontsize=6)
    save(fig, Path(figs) / 'curves.pdf')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
