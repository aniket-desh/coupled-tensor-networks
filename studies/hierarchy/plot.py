"""Hierarchy figures from results/ only. Group {1,2} is blue, group {3,4} orange; solid is each group's strong
component and dashed its weak one, so identity survives without color.
Usage: python -m studies.hierarchy.plot [fig7 fig8 fig9]"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from ctn.style import C1, C2, C3, INK, MUTED, plt

from .model import G, load, merged, tau

FIGURES = Path(__file__).parent / 'figures'
RESULTS = Path(__file__).parent / 'results'
COL, LS = [C1, C1, C2, C2], ['-', '--', '-', '--']
S = [1.0, .3, .8, .25]
LAB = [f'$c_{k + 1}$, $s={v:g}$' for k, v in enumerate(S)]
ARMS = [('hierarchy', 'hierarchy'), ('group_only', 'group sharing only'), ('u_only', '$u$ sharing only'), ('independent', 'independent')]
ORDER, RACE = merged('order'), merged('race')


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f'{name}.pdf')
    plt.close(fig)


def fig7():
    Z = np.load(RESULTS / 'loss.npz')
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    t, c = Z['hierarchy_main_t'], Z['hierarchy_main_c']
    for k in range(4):
        ax[0].plot(t, c[k] / S[k], color=COL[k], ls=LS[k], label=LAB[k])
    ax[0].set(xlim=(117500, 118700), xlabel='time', ylabel='$c_k/s_k$', title=r'$\delta=0.01$: siblings arrive together, ahead of stronger cousins')
    ax[0].legend(loc='center right')
    L = load('loss')['results']
    L['independent_main'] = load('rerun')['independent_main']
    for row, (a, name) in enumerate(ARMS):
        half = L[f'{a}_main']['half']
        ax[1].plot([min(half), max(half)], [row, row], color='#c9c8c3', lw=1, zorder=1)
        for k in range(4):
            ax[1].scatter(half[k], row, s=46, color=COL[k] if k in (0, 2) else 'white', edgecolor=COL[k], lw=1.4, zorder=3,
                          label=LAB[k] if row == 0 else None, marker='o' if k < 2 else 's')
    ax[1].set(xscale='log', xlim=(9e4, 2.4e6), ylim=(4.4, -.5), yticks=range(4), yticklabels=[n for _, n in ARMS], xlabel='half-time', title=r'Half-times by sharing pattern, $\delta=0.01$')
    ax[1].set_xticks([1e5, 2e5, 5e5, 1e6, 2e6], ['$10^5$', r'$2\times10^5$', r'$5\times10^5$', '$10^6$', r'$2\times10^6$'])
    ax[1].minorticks_off()
    ax[1].legend(loc='lower center', ncol=4, columnspacing=.8, handletextpad=.2, fontsize=7.5)
    xs = sorted({r['x'] for r in ORDER})
    ys = sorted({r['y'] for r in ORDER})
    M = np.full((len(ys), len(xs)), np.nan)
    for r in ORDER:
        if r['arm'] == 'hierarchy' and r['delta'] == .001:
            M[ys.index(r['y']), xs.index(r['x'])] = r['half'][1] < r['half'][2]
    ax[2].imshow(M, origin='lower', aspect='auto', cmap=matplotlib.colors.ListedColormap(['#f6d3c5', '#cde2fb']), vmin=0, vmax=1)
    A_first = np.array([[tau(1, x) < tau(y, y / 3) for x in xs] for y in ys], float)
    ax[2].contour(A_first, levels=[.5], colors=INK, linewidths=1.4)
    ax[2].set(xticks=range(len(xs)), xticklabels=xs, yticks=range(len(ys)), yticklabels=ys, xlabel='sibling strength $x$ ($s=1,x,y,y/3$)', ylabel='cousin strength $y$',
              title=r'Blue: sibling beats cousin, $\delta=10^{-3}$. Line: $\tau_A=\tau_B$')
    ax[2].grid(False)
    save(fig, 'fig7_hierarchy_order')


def fig8():
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.6), constrained_layout=True)
    for d, col in zip([.01, .003, .001], ['#9ec5f4', '#3987e5', '#104281']):
        R = [r for r in RACE if r['delta'] == d]
        ax[0].scatter([r['gap_pred'] * d ** 2 for r in R], [r['gap'] * d ** 2 for r in R], s=9, color=col, label=rf'$\delta={d:g}$', alpha=.8, linewidths=0)
    lim = np.array([2e-4, 2])
    ax[0].plot(lim, lim, color=INK, lw=1, label='prediction')
    ax[0].set(xscale='log', yscale='log', xlabel=r'predicted cousin delay $\times\,\delta^2$', ylabel=r'measured cousin delay $\times\,\delta^2$', title='P2: cousin delay, 500 random teachers')
    ax[0].legend()
    Sc = load('scaling')
    de = np.array([r['delta'] for r in Sc['rows']])
    H = np.array([r['half'] for r in Sc['rows']])
    ax[1].plot(np.log(1 / de), H[:, 1] - H[:, 0], 'o-', ms=4, color=C1, label=r'$c_2$ after $c_1$')
    ax[1].plot(np.log(1 / de), H[:, 3] - H[:, 2], 's-', ms=4, color=C2, label=r'$c_4$ after $c_3$')
    x = np.log(1 / de[de <= .01])
    for y0, sl, col in [((H[:, 1] - H[:, 0])[de <= .01][0], 7 / 3, C1), ((H[:, 3] - H[:, 2])[de <= .01][0], 2.2, C2)]:
        ax[1].plot(x, y0 + sl * (x - x[0]), color=col, ls=':', lw=1.2)
    ax[1].set(xlabel=r'$\log(1/\delta)$', ylabel='sibling delay', title=r'P3 as written: dotted slope $(1-x)/x$ misses $c_4$')
    ax[1].legend()
    seeds = {}
    for r in RACE:
        seeds.setdefault(r['seed'], {})[r['delta']] = r
    pts = {True: [], False: []}
    for v in seeds.values():
        r = v[.01]
        s_ = np.array(r['s'])
        win = int(np.argmin(r['taus']))
        for g in (0, 1):
            idx = np.where(G == g)[0]
            top, sib = idx[np.argmax(s_[idx])], idx[np.argmin(s_[idx])]
            X = s_[sib] / s_[top]
            if X > .8:
                continue
            a = (np.sqrt(0.0 if g == win else s_[G == win].sum()) + np.sqrt(s_[top])) ** (2 / 5)
            meas = ((v[.001]['half'][sib] - v[.001]['half'][top]) - (r['half'][sib] - r['half'][top])) / np.log(10)
            pts[g == win].append(((1 - X) / (s_[sib] * np.sqrt(s_[top] * a)), meas))
    for first, col, lab in [(True, C1, 'first group'), (False, C2, 'second group')]:
        p = np.array(pts[first])
        ax[2].scatter(p[:, 0], p[:, 1], s=9, color=col, label=f'{lab}, $n={len(p)}$', alpha=.7, linewidths=0)
    lim = np.array([.2, 12])
    ax[2].plot(lim, lim, color=INK, lw=1)
    ax[2].set(xscale='log', yscale='log', xlabel=r'$(1-x)\,/\,(s_{\rm sib}\sqrt{s_{\rm top}\,a})$', ylabel=r'measured slope in $\log(1/\delta)$', title=r'Refined after the miss, $x\leq0.8$ (not a prediction)')
    ax[2].legend()
    Z = np.load(RESULTS / 'scaling.npz')
    t, c, rate = Z['t'], Z['c'], Z['rate']
    w = (c[2] > 1e-10) & (c[2] < .3) & (t > Z['half'][1]) & (rate[2] > 0)
    ax[3].plot(c[2][w], rate[2][w], '.', ms=2, color=C2, label=r'$c_3$, cousin')
    cc = np.geomspace(1e-10, 1e-2, 50)
    ax[3].plot(cc, Sc['theory']['cous_coef'] * np.sqrt(cc), color=INK, lw=1, ls='--', label=r'$4\sqrt{a}\,s_3\sqrt{c_3}$')
    w = (c[1] > 1e-10) & (c[1] < .03) & (c[0] > .999)
    ax[3].plot(c[1][w], rate[1][w], 'o', ms=3, color=C1, label=r'$c_2$, sibling')
    ax[3].axhline(Sc['theory']['sib_rate'], color=MUTED, lw=1, ls=':', label=r'$2\,a\,q^2 s_2$')
    ax[3].set(xscale='log', yscale='log', xlabel='$c_k$', ylabel=r'$d\log c_k/dt$', title=r'P5: escape laws, $\delta=10^{-3}$')
    ax[3].legend()
    save(fig, 'fig8_hierarchy_laws')


def fig9():
    Z = np.load(RESULTS / 'loss.npz')
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.6), constrained_layout=True)
    t, c = Z['hierarchy_main_t'], Z['hierarchy_main_c']
    H0 = load('loss')['results']['hierarchy_main']['half']
    for k in range(4):
        y = 100 * (c[k] / S[k] - 1)
        ax[0].plot(t - H0[0], y, color=COL[k], ls=LS[k], label=LAB[k])
    ax[0].axhline(0, color=MUTED, lw=.8)
    ax[0].set(xlim=(-5, 40), ylim=(-3, 5), xlabel='time since $c_1$ half-time', ylabel=r'$(c_k-s_k)/s_k$ (%)', title='Sibling learns: only $c_1$ overshoots')
    ax[0].legend(loc='lower right', ncol=2)
    for k in range(4):
        y = 100 * (c[k] / S[k] - 1)
        ax[1].plot(t - H0[2], y, color=COL[k], ls=LS[k])
    ax[1].axhline(0, color=MUTED, lw=.8)
    ax[1].set(xlim=(-5, 40), ylim=(-3, 5), xlabel='time since $c_3$ half-time', title='Cousins learn: $c_1,c_2$ pushed up, then $c_3$')
    OV = load('overshoot')['results']
    D = [[100 * r['O'][r['order'][i]] for r in OV] for i in range(4)]
    q = np.percentile(D, [5, 25, 50, 75, 95], axis=1)
    for i in range(4):
        ax[2].plot([i, i], [q[0, i], q[4, i]], color=C1, lw=1)
        ax[2].plot([i, i], [q[1, i], q[3, i]], color=C1, lw=5, solid_capstyle='round')
        ax[2].scatter(i, q[2, i], color='white', edgecolor=C1, zorder=3, s=30, lw=1.3)
    ax[2].set(xticks=range(4), xticklabels=['1st', '2nd', '3rd', 'last'], xlabel='position in learning order', ylabel='peak overshoot (%)', title='1500 random teachers, 5/25/50/75/95%')
    Dn = load('dense')['results']
    labels, shared, ctrl = [], [], []
    for s_ in [(1.0, .3, .8, .25), (1.0, .6, .9, .3)]:
        for sg in (.2, .1, .05):
            frac = []
            for sh in (True, False):
                Hd = np.array([r['half'] for r in Dn if tuple(r['s']) == s_ and r['sigma'] == sg and r['shared'] == sh])
                o = np.argsort(Hd, 1)
                frac.append(np.mean(G[o[:, 0]] == G[o[:, 1]]))
            shared.append(frac[0])
            ctrl.append(frac[1])
            labels.append(f'$s_2={s_[1]:g}$, $\\sigma={sg:g}$')
    y = np.arange(6)
    ax[3].barh(y - .19, shared, .36, color=C1, label='shared tree')
    ax[3].barh(y + .19, ctrl, .36, color=C3, label='independent networks')
    ax[3].set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel='fraction of 24 seeds', title='Random init: first two to finish are siblings')
    ax[3].set_axisbelow(True)
    ax[3].invert_yaxis()
    ax[3].legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2)
    save(fig, 'fig9_hierarchy_overshoot')


if __name__ == '__main__':
    for f in sys.argv[1:] or ['fig7', 'fig8', 'fig9']:
        globals()[f]()
