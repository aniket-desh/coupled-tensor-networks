"""lang-source figures from saved results.

  PYTHONPATH=. python studies/lang/lang-source/plot.py <results dir> <figures dir>

curves.pdf   diagnostic KL to Bayes vs update; rows: fixed source / two-source, columns: tree / tied / chain; one line per
             (condition, seed), colour = condition.
t.pdf        steps to threshold per condition and arm, one point per seed (open = never reached).
probes.pdf   tree arm, seed-median over runs: probe accuracy of f(x_i, x_j) at each level containing both sources, the
             source token, and a same-span distractor, vs update, for the two-source pairs.
"""
import sys

from studies.lang import plotting as Pl

FIXED = ['fixed_j0', 'fixed_j7', 'fixed_j8', 'fixed_j15']
TWO = ['two_0_1', 'two_1_2', 'two_3_4', 'two_7_8', 'two_0_8', 'two_0_15']


def main():
    res, figs = sys.argv[1], sys.argv[2]
    an, cv, ip = Pl.load(res)
    fig, ax = Pl.plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for i, conds in enumerate((FIXED, TWO)):
        for j, arm in enumerate(('tree', 'tied', 'chain')):
            a = ax[i, j]
            for ci, c in enumerate(conds):
                for s in range(3):
                    n = f'{c}_{arm}_s{s}'
                    if n in cv:
                        Pl.curve(a, cv[n], color=Pl.COLORS[ci], lw=0.8, alpha=0.8, label=c if s == 0 else None)
            a.set_xscale('log'), a.set_yscale('log'), a.set_title(f'{arm}'), a.set_xlabel('update')
            if j == 0:
                a.set_ylabel('diagnostic KL(Bayes || model) [nats]')
            if j == 2:
                Pl.legend(a, fontsize=7)
    Pl.save(fig, figs, 'curves')

    fig, ax = Pl.plt.subplots(1, 1, figsize=(11, 3.6))
    conds = FIXED + TWO
    cap = 25_000
    for k, arm in enumerate(('tree', 'tied', 'chain', 'tree32')):
        xs, ts = [], []
        for ci, c in enumerate(conds):
            for s in range(3):
                n = f'{c}_{arm}_s{s}'
                if n in an.get('runs', {}):
                    xs.append(ci + 0.18 * (k - 1.5)), ts.append(an['runs'][n]['t'])
        Pl.t_points(ax, xs, ts, cap, color=Pl.COLORS[k], label=arm)
    ax.set_xticks(range(len(conds)), conds, rotation=30, ha='right')
    ax.set_yscale('log'), ax.set_ylabel('updates to 0.9 x Bayes accuracy')
    handles = [Pl.plt.Line2D([], [], marker='o', ls='', color=Pl.COLORS[k], label=arm) for k, arm in enumerate(('tree', 'tied', 'chain', 'tree32'))]
    ax.legend(handles=handles, fontsize=7)
    Pl.save(fig, figs, 't')

    tr = ip.get('trajectories', {})
    fig, ax = Pl.plt.subplots(3, len(TWO), figsize=(15, 7.5), sharex=True, sharey=True)
    for j, c in enumerate(TWO):
        for i, summ in enumerate(('f', 'src_i', 'distractor')):
            a = ax[i, j]
            for s in range(3):
                Pl.traj(a, tr, f'{c}_tree_s{s}', summ, (0, 1, 2, 3), lw=0.9, alpha=0.8)
            a.set_xscale('log'), a.set_ylim(0, 1.02)
            if i == 0:
                a.set_title(c)
            if j == 0:
                a.set_ylabel(f'{summ} probe accuracy')
            if i == 2:
                a.set_xlabel('update')
    Pl.legend(ax[1, 0], fontsize=7)
    Pl.save(fig, figs, 'probes')


if __name__ == '__main__':
    main()
