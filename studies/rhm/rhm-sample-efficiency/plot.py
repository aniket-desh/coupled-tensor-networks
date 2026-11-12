"""rhm-sample-efficiency figures from <results>/analysis.json and curves.json.

  PYTHONPATH=. python studies/rhm/rhm-sample-efficiency/plot.py studies/rhm/rhm-sample-efficiency/results studies/rhm/rhm-sample-efficiency/figures

vs_N.pdf: test loss (loss-selected, filled; final, open) and the final KL to Bayes on the training set (x) vs N, and
the accuracy-selected test accuracy vs N (dotted horizontal: the online references); one color per grammar, columns =
task; dashed = Bayes; dotted vertical = P* = n m^L.
curves.pdf: training-set and diagnostic loss vs updates per N (grammar 0)."""
import json
import sys
from pathlib import Path

from studies.rhm.plotting import CATEGORICAL, INK2, plt, save, steps_x

TASKS = ('root', 'last')


def main():
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    an = json.load(open(res / 'analysis.json'))
    S = list(an['runs'].values())
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True)
    for j, t in enumerate(TASKS):
        ss = [s for s in S if s['task'] == t and s['N'] != 'online']
        for s in [s for s in S if s['task'] == t and s['N'] == 'online' and s['regime'] != 'unfinished']:
            axes[1, j].axhline(s['test_acc'], color=CATEGORICAL[s['g'] % len(CATEGORICAL)], ls=':', lw=0.8)
        for s in ss:
            c = CATEGORICAL[s['g'] % len(CATEGORICAL)]
            axes[0, j].scatter(s['N'], s['test_loss'], color=c, s=16, zorder=3)
            axes[0, j].scatter(s['N'], s['final_test_loss'], facecolor='none', edgecolor=c, s=16, zorder=3)
            if s['fit_kl_final'] is not None:
                axes[0, j].scatter(s['N'], max(s['fit_kl_final'], 1e-5), color=c, marker='x', s=16, zorder=3)
            axes[1, j].scatter(s['N'], s['test_acc'], color=c, s=16, zorder=3)
        if ss and t == 'last':
            axes[0, j].axhline(ss[0]['bayes_loss'], color=INK2, ls='--', lw=0.8)
            axes[1, j].axhline(ss[0]['bayes_acc'], color=INK2, ls='--', lw=0.8)
        for i in range(2):
            axes[i, j].set_xscale('log', base=2)
            axes[i, j].axvline(an['P_star'], color=INK2, ls=':', lw=0.8)
        axes[0, j].set_yscale('log')
        axes[0, j].set_title(f"task '{t}'")
        axes[1, j].set_xlabel('training set size N')
    axes[0, 0].set_ylabel('nats: test loss (dot: loss-selected, ring: final),\ntraining-set KL to Bayes (x)')
    axes[1, 0].set_ylabel('test accuracy (accuracy-selected; dotted: online)')
    for g in sorted({s['g'] for s in S}):
        axes[1, 0].scatter([], [], color=CATEGORICAL[g], s=16, label=f'grammar {g}')
    axes[1, 0].legend(fontsize=7)
    save(fig, figs / 'vs_N.pdf')

    curves = json.load(open(res / 'curves.json'))
    Ns = sorted({n.split('_N')[-1] for n, c in curves.items() if c['cfg']['grammar']['seed'] == 0 and '_N' in n}, key=int)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for j, t in enumerate(TASKS):
        for k, N in enumerate(Ns):
            c = curves.get(f'{t}_g0_N{N}')
            if c is None:
                continue
            col = CATEGORICAL[k % len(CATEGORICAL)]
            axes[j].plot(steps_x(c['step']), c['fit_loss'], color=col, lw=0.9, ls='--')
            axes[j].plot(steps_x(c['step']), c['diag_loss'], color=col, lw=0.9, label=f'N = 2^{int(N).bit_length() - 1}')
        axes[j].set_xscale('log')
        axes[j].set_yscale('log')
        axes[j].set_title(f"task '{t}', grammar 0 (dashed: training set)")
        axes[j].set_xlabel('updates')
    axes[0].set_ylabel('loss (nats)')
    axes[0].legend(fontsize=7)
    save(fig, figs / 'curves.pdf')


if __name__ == '__main__':
    main()
