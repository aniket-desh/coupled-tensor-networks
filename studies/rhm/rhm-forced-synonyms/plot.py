"""Figures for rhm-forced-synonyms, from the saved results only.

    PYTHONPATH=. python studies/rhm/rhm-forced-synonyms/plot.py studies/rhm/rhm-forced-synonyms/results studies/rhm/rhm-forced-synonyms/figures
"""
import json
import sys
from pathlib import Path

import numpy as np

from studies.rhm import toy as Y
from studies.rhm.plotting import CATEGORICAL, INK2, plt, save

FLOOR = 1e-18                      # values below this (exact zeros in float64) are drawn here on log axes
INIT_COLOR = {'variance': CATEGORICAL[0], 'identity': CATEGORICAL[1]}
clip = lambda v: np.maximum(np.asarray(v, dtype=float), FLOOR)


def main(res, figs):
    rows = json.load(open(Path(res) / 'runs.json'))
    traj = json.load(open(Path(res) / 'trajectories.json'))
    exact = lambda q: q['kl'] < Y.EXACT
    widths = sorted({q['r'] for q in rows})

    # 1. widths: where training ends, per width and init
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for q in rows:
        dx = (-0.15 if q['init'] == 'variance' else 0.15) + 0.1 * ((q['seed'] % 20) / 20 - 0.5)
        ax.plot(q['r'] + dx, clip(q['kl']), 'o', ms=2.5, color=INIT_COLOR[q['init']], alpha=0.7)
    ax.axhline(Y.UNIFORM_KL, color=INK2, lw=0.6, ls=':')
    ax.text(widths[-1] + 0.45, Y.UNIFORM_KL, 'uniform', va='bottom', ha='right', fontsize=6, color=INK2)
    ax.axhline(Y.EXACT, color=INK2, lw=0.6, ls='--')
    ax.text(widths[-1] + 0.45, Y.EXACT, 'exact below', va='bottom', ha='right', fontsize=6, color=INK2)
    ax.set_yscale('log')
    ax.set_xticks(widths)
    ax.set_xlabel('bond dimension r')
    ax.set_ylabel('excess loss after the polish (nats)')
    for k, c in INIT_COLOR.items():
        ax.plot([], [], 'o', ms=3, color=c, label=f'{k} init')
    ax.legend(fontsize=6, loc='center left')
    save(fig, Path(figs) / 'widths.pdf')

    # 2. collapse: synonym-swap effect and composition error against the excess loss, along training, r = 2
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharex=True)
    runs2 = sorted({t['run'] for t in traj if t['r'] == 2})
    for name in runs2:
        pts = sorted([t for t in traj if t['run'] == name], key=lambda t: t['step'])
        final = next(q for q in rows if q['run'] == name)
        col = INIT_COLOR[final['init']]
        for ax, f in zip(axes, (lambda t: max(t['swap_L_all'], t['swap_R_all']), lambda t: t['comp_err'])):
            ax.plot(clip([t['kl'] for t in pts]), clip([f(t) for t in pts]), '-', lw=0.5, color=col, alpha=0.5)
            ax.plot(clip(final['kl']), clip(f(final)), 'o', ms=2.5, color=col)
    k = np.logspace(-18, -1, 50)
    for ax, lab in zip(axes, ('largest output change when a pair is swapped\nfor its synonym, any sibling (units of |F(A) - F(A\')|)',
                              'largest error of a composed valid pair\n(units of |F(A) - F(A\')|)')):
        ax.plot(k, np.sqrt(k), color=INK2, lw=0.6, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('excess loss (nats)')
        ax.set_ylabel(lab, fontsize=7)
    axes[0].text(1e-12, 3e-6, 'slope 1/2', fontsize=6, color=INK2)
    for kk, c in INIT_COLOR.items():
        axes[1].plot([], [], '-o', ms=3, lw=0.6, color=c, label=f'{kk} init (dot: after the polish)')
    axes[1].legend(fontsize=6, loc='upper left')
    fig.suptitle('r = 2: seven checkpoints along training, then the polished tree', fontsize=8)
    save(fig, Path(figs) / 'collapse.pdf')

    # 3. forbidden pairs
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    ex2 = [q for q in rows if q['r'] == 2 and exact(q)]
    ax = axes[0]
    if ex2:
        pred, rho = np.array([q['forb_pred'] for q in ex2]).ravel(), np.array([q['forb_rho'] for q in ex2]).ravel()
        lim = [min(pred.min(), rho.min()) / 1.5, max(pred.max(), rho.max()) * 1.5]
        ax.plot(lim, lim, color=INK2, lw=0.6, ls=':')
        ax.plot(pred, rho, 'o', ms=2.5, color=CATEGORICAL[0], alpha=0.7)
        ax.axhline(1, color=INK2, lw=0.4)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('forbidden response predicted from (x, y)')
    ax.set_ylabel('measured response to a forbidden pair (b, b)\n(units of a valid response)', fontsize=7)
    ax.set_title(f'r = 2, exact runs (n = {len(ex2)}, 3 pairs each)', fontsize=7)
    ax = axes[1]
    if ex2:
        xy = np.array([q['xy_L'] for q in ex2])
        ax.plot(xy[:, 0], xy[:, 1], 'o', ms=2.5, color=CATEGORICAL[0], alpha=0.7)
    ax.plot(-1, -1, '+', ms=9, color=CATEGORICAL[1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('r = 2: code of symbol 0 = x (code 1) + y (code 2), left bond;\ncross: the three codes sum to zero', fontsize=7)
    ax = axes[2]
    for q in rows:
        if exact(q) and q['r'] >= 2:
            for b, v in enumerate(q['forb_rho']):
                ax.plot(q['r'] + (-0.15 if q['init'] == 'variance' else 0.15) + 0.05 * (b - 1), clip(v), 'o', ms=2,
                        color=INIT_COLOR[q['init']], alpha=0.6)
    ax.axhline(1, color=INK2, lw=0.4)
    ax.set_yscale('log')
    ax.set_xticks([r for r in widths if r >= 2])
    ax.set_xlabel('bond dimension r')
    ax.set_ylabel('response to a forbidden pair (b, b)', fontsize=7)
    ax.set_title('exact runs at every width (the grammar\'s tensor gives 0)', fontsize=7)
    save(fig, Path(figs) / 'forbidden.pdf')

    # 4. what is not forced at r >= 3: does the output still depend on which synonym was shown?
    WCOL = {2: CATEGORICAL[0], 3: CATEGORICAL[2], 4: CATEGORICAL[3]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    for q in rows:
        if q['r'] >= 2 and q['kl'] <= Y.NOT_EXACT:
            ax.plot(clip(max(q['swap_L_legal'], q['swap_R_legal'])), clip(max(q['swap_L_all'], q['swap_R_all'])),
                    'o' if exact(q) else 'x', ms=3, color=WCOL.get(q['r'], INK2), alpha=0.7, mew=0.6)
    lim = np.array([FLOOR, 1e2])
    ax.plot(lim, lim, color=INK2, lw=0.6, ls=':')
    ax.plot(lim, 10 * lim, color=INK2, lw=0.6, ls='--')
    ax.axhline(1e-3, color=INK2, lw=0.6, ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('synonym-swap effect beside siblings that occur\n(bounded by the prediction error)', fontsize=7)
    ax.set_ylabel('synonym-swap effect beside any sibling', fontsize=7)
    for r, c in WCOL.items():
        ax.plot([], [], 'o', ms=3, color=c, label=f'r = {r}')
    ax.plot([], [], 'x', ms=3, color=INK2, label='excess between 1e-10 and 1e-5')
    ax.legend(fontsize=6, loc='lower right')
    ax.set_title('keeps the synonym: above both dashed lines', fontsize=7)
    ax = axes[1]
    for q in rows:
        if exact(q):
            dx = (-0.15 if q['init'] == 'variance' else 0.15) + 0.1 * ((q['seed'] % 20) / 20 - 0.5)
            ax.plot(q['r'] + dx, q['within_L'], 'o', ms=2.5, color=INIT_COLOR[q['init']], alpha=0.7)
    ax.set_xticks([r for r in widths if r >= 2])
    ax.set_xlabel('bond dimension r')
    ax.set_ylabel('within-symbol share of the whitened left pair bond\n(an indicator: any direction, read or not)', fontsize=7)
    ax.set_title('exact runs', fontsize=7)
    save(fig, Path(figs) / 'not_forced.pdf')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
