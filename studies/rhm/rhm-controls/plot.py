"""rhm-controls figures from <results>/curves.json and analysis.json.

  PYTHONPATH=. python studies/rhm/rhm-controls/plot.py studies/rhm/rhm-controls/results studies/rhm/rhm-controls/figures

curves.pdf: diagnostic loss and accuracy vs updates per arm (one line per grammar and seed), columns = task.
paired.pdf: per comparison (columns), steps to threshold and final test error rate, one point per (task, grammar,
seed); runs that never reached the threshold are drawn at the top/right edge, error 0 at the bottom/left."""
import json
import math
import sys
from pathlib import Path

from studies.rhm.plotting import CATEGORICAL, INK2, plt, save, steps_x

ARMS = ('tree', 'perm', 'tied', 'chain')
COL = dict(zip(ARMS, CATEGORICAL))


def main():
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    curves = json.load(open(res / 'curves.json'))
    an = json.load(open(res / 'analysis.json'))
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    for j, task in enumerate(('root', 'last')):
        seen = set()
        for n, c in sorted(curves.items()):
            if c['cfg']['task'] != task:
                continue
            arm = c['cfg']['arm']
            for i, key in enumerate(('diag_loss', 'diag_acc')):
                axes[i, j].plot(steps_x(c['step']), c[key], color=COL[arm], lw=0.9, alpha=0.8,
                                label=arm if (arm not in seen and i == 0) else None)
            seen.add(arm)
            if task == 'last':
                axes[0, j].axhline(c['bayes_diag']['loss'], color=INK2, ls='--', lw=0.8)
                axes[1, j].axhline(c['bayes_diag']['acc'], color=INK2, ls='--', lw=0.8)
        axes[0, j].set_title(f"task '{task}'")
        axes[0, j].set_yscale('log')
        axes[0, j].legend(fontsize=7)
        for i in range(2):
            axes[i, j].set_xscale('log')
        axes[1, j].set_xlabel('updates')
    axes[0, 0].set_ylabel('diagnostic loss (nats)')
    axes[1, 0].set_ylabel('diagnostic accuracy')
    save(fig, figs / 'curves.pdf')

    comps = [(k, v) for k, v in an['criteria'].items()]
    fig, axes = plt.subplots(2, len(comps), figsize=(3 * len(comps), 5.6))
    for j, (name, c) in enumerate(comps):
        a, b = c['a'], c['b']
        for i, (ka, kb, lab) in enumerate((('t_a', 't_b', 'updates to 0.9 x Bayes acc'), ('err_a', 'err_b', 'test error rate'))):
            ax = axes[i, j]
            vals = [v for pr in c['pairs'] for v in (pr[ka], pr[kb]) if v is not None and math.isfinite(v) and v > 0]
            top = max(vals) * 2 if vals else 1
            lo = min(vals) / 2 if vals else 1e-4
            for pr in c['pairs']:
                x = pr[ka] if pr[ka] is not None and math.isfinite(pr[ka]) else top
                y = pr[kb] if pr[kb] is not None and math.isfinite(pr[kb]) else top
                ax.scatter(max(x, lo), max(y, lo), s=18, color=CATEGORICAL[pr['g']], marker='o' if pr['task'] == 'root' else 's',
                           edgecolor='white', linewidth=0.5)
            ax.plot([lo, top], [lo, top], color=INK2, lw=0.8, ls='--')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f'{a}: {lab}')
            ax.set_ylabel(f'{b}: {lab}')
        axes[0, j].set_title(c['verdict'], fontsize=6)
    for g in range(3):
        axes[0, 0].scatter([], [], color=CATEGORICAL[g], label=f'grammar {g}')
    axes[0, 0].scatter([], [], color=INK2, marker='o', label="'root'")
    axes[0, 0].scatter([], [], color=INK2, marker='s', label="'last'")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('Above the diagonal: the first-named arm is better (never reached / error 0: drawn at the edges)')
    save(fig, figs / 'paired.pdf')


if __name__ == '__main__':
    main()
