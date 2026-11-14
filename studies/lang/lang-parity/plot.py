"""lang-parity figures from saved results.

  PYTHONPATH=. python studies/lang/lang-parity/plot.py <results dir> <figures dir>

t_grid.pdf   steps to threshold vs init noise eps, one panel per degree k, bias 0 (blue) and 0.2 (orange); one point
             per seed, open circles at the top = never reached within 60,000 updates (censored).
curves.pdf   diagnostic loss vs update for every main run at bias 0, panels by k, colour by eps. Log loss axis: eps = 1
             starts at hundreds to thousands of nats, which would flatten the other curves on a linear axis.
probes.pdf   for the seed-0 run at k = 8 and k = 16 (bias 0) at each eps: partial-parity probe accuracy per level vs update,
             with the run's threshold step marked.
"""
import math
import sys

from studies.lang import plotting as Pl

KS, EPS = (2, 4, 8, 16), (0.01, 0.1, 0.3, 1.0)


def main():
    res, figs = sys.argv[1], sys.argv[2]
    an, cv, ip = Pl.load(res)
    runs = an.get('runs', {})
    cap = 70_000
    fig, ax = Pl.plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for a, k in zip(ax, KS):
        for bi, b in enumerate((0.0, 0.2)):
            for ei, e in enumerate(EPS):
                ts = [runs.get(f'main_k{k}_e{e:g}_b{b:g}_s{s}', {}).get('t') for s in range(3)]
                ts = [t for t in ts if t is not None]
                Pl.t_points(a, [math.log10(e) + 0.06 * (bi - 0.5)] * len(ts), ts, cap, color=Pl.COLORS[bi])
        a.set_title(f'k = {k}'), a.set_yscale('log'), a.set_xlabel('log10 init noise eps')
        a.axhline(60_000, color='gray', lw=0.6, ls=':')
    ax[0].set_ylabel('updates to 0.9 x Bayes accuracy')
    ax[0].legend(handles=[Pl.plt.Line2D([], [], marker='o', ls='', color=Pl.COLORS[i], label=f'bias {b:g}') for i, b in enumerate((0, 0.2))], fontsize=7)
    Pl.save(fig, figs, 't_grid')

    fig, ax = Pl.plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for a, k in zip(ax, KS):
        for ei, e in enumerate(EPS):
            for s in range(3):
                n = f'main_k{k}_e{e:g}_b0_s{s}'
                if n in cv:
                    Pl.curve(a, cv[n], key='diag_loss', color=Pl.COLORS[ei], lw=0.8, label=f'eps {e:g}')
        a.set_xscale('log'), a.set_yscale('log'), a.set_title(f'k = {k}, bias 0'), a.set_xlabel('update')
        a.axhline(math.log(2), color='gray', lw=0.6, ls=':')
    ax[0].set_ylabel('diagnostic loss [nats]')
    Pl.legend(ax[0], fontsize=7)
    Pl.save(fig, figs, 'curves')

    tr = ip.get('trajectories', {})
    fig, ax = Pl.plt.subplots(2, 4, figsize=(14, 5.6), sharex=True, sharey=True)
    for i, k in enumerate((8, 16)):
        for j, e in enumerate(EPS):
            n = f'main_k{k}_e{e:g}_b0_s0'
            a = ax[i, j]
            Pl.traj(a, tr, n, 'par', (0, 1, 2, 3))
            t = Pl.num(runs.get(n, {}).get('t'))
            if math.isfinite(t):
                a.axvline(t, color='k', lw=0.7, ls='--')
            a.set_xscale('log'), a.set_title(f'k = {k}, eps {e:g}, seed 0'), a.set_ylim(0.4, 1.02)
            if j == 0:
                a.set_ylabel('partial-parity probe accuracy')
            if i == 1:
                a.set_xlabel('update')
    Pl.legend(ax[0, 0], fontsize=7)
    Pl.save(fig, figs, 'probes')


if __name__ == '__main__':
    main()
