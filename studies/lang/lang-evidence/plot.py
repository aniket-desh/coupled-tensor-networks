"""lang-evidence figures from saved results.

  PYTHONPATH=. python studies/lang/lang-evidence/plot.py <results dir> <figures dir>

curves.pdf   diagnostic KL vs update. Left: masked bigram by mask position (colour) for tree / perm / chain / tree32.
             Right: trigger arms.
probes.pdf   tree arm (masked bigram) and tree32 (trigger): probe score per level vs update, one line per (level, seed):
             forward evidence R^2, combined evidence R^2, trigger flag accuracy, trigger location accuracy.
trigger_strata.pdf  final KL by stratum (0 = no trigger; 1-4 = level of the trigger's common ancestor with leaf 15).
"""
import sys

from studies.lang import plotting as Pl


def main():
    res, figs = sys.argv[1], sys.argv[2]
    an, cv, ip = Pl.load(res)
    fig, ax = Pl.plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for j, arm in enumerate(('tree', 'perm', 'chain', 'tree32')):
        for ci, p in enumerate((5, 4, 8)):
            for s in range(3):
                n = f'mb_p{p}_{arm}_s{s}'
                if n in cv:
                    Pl.curve(ax[j], cv[n], color=Pl.COLORS[ci], lw=0.8, label=f'mask {p}')
        ax[j].set_title(f'masked bigram, {arm}')
        Pl.legend(ax[j], fontsize=7)
    for k, arm in enumerate(('tree16', 'tree32', 'perm32', 'chain')):
        for s in range(3):
            n = f'trigger_{arm}_s{s}'
            if n in cv:
                Pl.curve(ax[4], cv[n], color=Pl.COLORS[k], lw=0.8, label=arm)
    ax[4].set_title('trigger')
    Pl.legend(ax[4], fontsize=7)
    for a in ax:
        a.set_xscale('log'), a.set_yscale('log'), a.set_xlabel('update')
    ax[0].set_ylabel('diagnostic KL [nats]')
    Pl.save(fig, figs, 'curves')

    tr = ip.get('trajectories', {})
    panels = [('mb_p5', 'tree', 'fwd'), ('mb_p5', 'tree', 'comb'), ('mb_p4', 'tree', 'comb'), ('mb_p8', 'tree', 'fwd'),
              ('trigger', 'tree32', 'flag'), ('trigger', 'tree32', 'where')]
    fig, ax = Pl.plt.subplots(1, len(panels), figsize=(17, 3.2), sharex=True)
    for a, (c, arm, summ) in zip(ax, panels):
        for s in range(3):
            Pl.traj(a, tr, f'{c}_{arm}_s{s}', summ, (0, 1, 2, 3), lw=0.9, alpha=0.8)
        a.set_xscale('log'), a.set_title(f'{c} {arm}: {summ}'), a.set_xlabel('update')
        a.set_ylim(-0.05 if summ in ('fwd', 'comb') else 0, 1.02)
    ax[0].set_ylabel('probe (R^2 or accuracy)')
    Pl.legend(ax[0], fontsize=7)
    Pl.save(fig, figs, 'probes')

    fig, ax = Pl.plt.subplots(1, 1, figsize=(5, 3.2))
    arms = an.get('H6_trigger_arms', {})
    for k, arm in enumerate(('tree16', 'tree32', 'perm32', 'chain')):
        st = (arms.get(arm) or {}).get('strata') or {}
        xs = [int(x) for x in sorted(st)]
        ax.plot([x + 0.08 * (k - 1.5) for x in xs], [Pl.num(st[str(x)]) for x in xs], 'o-', color=Pl.COLORS[k], label=arm, ms=4)
    ax.set_xticks(range(5), ['none', 'L1', 'L2', 'L3', 'root'])
    ax.set_xlabel('trigger stratum (common ancestor with leaf 15)'), ax.set_ylabel('median test KL [nats]'), ax.set_yscale('log')
    Pl.legend(ax, fontsize=7)
    Pl.save(fig, figs, 'trigger_strata')


if __name__ == '__main__':
    main()
