"""Regenerate figures/ from results/ only; no training happens here."""
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'  # c1 / c2 / third series; identity is fixed across figures
INK, MUTED = '#0b0b0b', '#8a8984'
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#104281']  # sequential: larger log(1/delta)

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e8e7e3', 'grid.linewidth': 0.6, 'axes.edgecolor': MUTED, 'lines.linewidth': 1.6,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'legend.fontsize': 8,
                     'savefig.dpi': 170, 'figure.facecolor': 'white'})


def load(name):
    return json.load(open(f'results/{name}.json'))


def fig1():
    d, js = np.load('results/e1.npz'), load('e1')
    fig, ax = plt.subplots(3, 2, figsize=(9.5, 8.2))
    for col, rho in enumerate(['0.5', '0.2']):
        tag = f'rho{rho}'
        info = js['summary'][tag]
        s = np.array([1.0, float(rho)])[:, None]
        t, c, e = d[f'{tag}_t'], d[f'{tag}_c'], d[f'{tag}_e']
        ti, ci = d[f'{tag}_ti'], d[f'{tag}_ci']
        a = ax[0, col]
        a.plot(t, c[0] / s[0], color=C1, label=r'$c_1/s_1$ shared')
        a.plot(t, c[1] / s[1], color=C2, label=r'$c_2/s_2$ shared')
        a.plot(ti, ci[0] / s[0], color=C1, ls='--', lw=1, label=r'$c_1/s_1$ independent')
        a.plot(ti, ci[1] / s[1], color=C2, ls='--', lw=1, label=r'$c_2/s_2$ independent')
        a.set(xlim=(0, info['t_half_2_independent'] * 1.15), xlabel='time', ylabel='coefficient / target',
              title=fr'$\rho={rho}$, $\delta={js["delta"]}$: sharing pulls $c_2$ up with $c_1$')
        a.legend(loc='lower right')

        a = ax[1, col]
        tc = info['t_cross']
        a.axhline(0, color=MUTED, lw=0.8)
        a.plot(t - tc, 100 * (c[0] - 1), color=C1, label='reduced 2D ODE')
        tg, cg = d[f'{tag}_tg'], d[f'{tag}_cg']
        tgc = tg[np.argmax(cg[0] >= 1)]
        st = 2 if rho == '0.5' else 8
        a.plot((tg - tgc)[::st], 100 * (cg[0, ::st] - 1), 'o', ms=3, mfc='none', color=INK, mew=0.6,
               label=fr'GD on 10 params, $\eta={js["eta"]}$')
        tr, cr = d[f'{tag}_tr'], d[f'{tag}_cr']
        a.plot(tr - tr[np.argmax(cr[0] >= 1)], 100 * (cr[0] - 1), color=C3, lw=1.2, label=r'random dense init, $\sigma=0.1$')
        span = 3 * (info['t_peak'] - tc)
        a.set(xlim=(-0.35 * span, span * 1.5), ylim=(-info['overshoot_ode'] * 60, info['overshoot_ode'] * 125),
              xlabel=r'time since $c_1$ first reaches $s_1$', ylabel=r'$(c_1-s_1)/s_1$ (%)',
              title=f'overshoot peaks at {100 * info["overshoot_ode"]:.2f}%, when $c_2$ is about half learned')
        a.legend(loc='upper right')

        a = ax[2, col]
        a.semilogy(t - tc, 0.5 * (e ** 2).sum(0), color=INK, label=r'$\mathcal{L}$')
        a.semilogy(t - tc, 0.5 * e[0] ** 2, color=C1, label=r'$\epsilon_1^2/2$')
        a.semilogy(t - tc, 0.5 * e[1] ** 2, color=C2, label=r'$\epsilon_2^2/2$')
        a.axvline(0, color=MUTED, lw=0.8)
        a.set(xlim=(-0.6 * span, span * 1.7), ylim=(1e-10, 1), xlabel=r'time since $c_1$ first reaches $s_1$', ylabel='loss',
              title=fr'$c_1$ loss regrows while total loss falls; at most {100 * info["max_L1_frac"]:.1f}% of $\mathcal{{L}}$')
        a.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('figures/fig1_anatomy.png')


def fig1a():
    """Top row of fig1 with the shared entry a added, plus a close-up of the transition, where the wide axis hides
    the shape of a. a and the shared c come from one 10-parameter flow; dashed control curves are from e1."""
    d, js, A = np.load('results/e1.npz'), load('e1'), np.load('results/e1a.npz')
    info_a = load('e1a')['summary']
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for col, rho in enumerate(['0.5', '0.2']):
        tag = f'rho{rho}'
        info, ia = js['summary'][tag], info_a[tag]
        s = np.array([1.0, float(rho)])[:, None]
        t, c, y = A[f'{tag}_t'], A[f'{tag}_c'], A[f'{tag}_a'] / ia['a_inf_predicted']
        ti, ci = d[f'{tag}_ti'], d[f'{tag}_ci']
        p = ax[0, col]
        p.plot(t, y, color=C3, label=r'$a/a_\infty$ shared')
        p.plot(t, c[0] / s[0], color=C1, label=r'$c_1/s_1$ shared')
        p.plot(t, c[1] / s[1], color=C2, label=r'$c_2/s_2$ shared')
        p.plot(ti, ci[0] / s[0], color=C1, ls='--', lw=1, label=r'$c_1/s_1$ independent')
        p.plot(ti, ci[1] / s[1], color=C2, ls='--', lw=1, label=r'$c_2/s_2$ independent')
        p.set(xlim=(0, info['t_half_2_independent'] * 1.15), xlabel='time', ylabel='value / final value',
              title=fr'$\rho={rho}$, $\delta={js["delta"]}$, $a_\infty=(1+\rho)^{{1/3}}={ia["a_inf_predicted"]:.3f}$')
        p = ax[1, col]
        p.axvline(ia['t_cross'], color=MUTED, lw=0.8)
        p.plot(t, y, color=C3)
        p.plot(t, c[0] / s[0], color=C1)
        p.plot(t, c[1] / s[1], color=C2)
        p.set(xlim=(ia['t_half_c'][0] - 12, ia['t_half_c'][1] + 12), ylim=(0, 1.08), xlabel='time', ylabel='value / final value',
              title=fr'close-up: at $c_1=s_1$ (grey line), $a$ is at {100 * ia["a_over_final_at_cross"]:.1f}% of final')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=5)
    fig.savefig('figures/fig1a_shared_entry.png')


def fig2():
    d = np.load('results/e2.npz')
    r, de, O, het = d['rhos'], d['deltas'], d['O1'], d['het']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    a = ax[0]
    for i in range(1, len(de)):
        a.plot(r, 100 * O[i], color=RAMP[i], lw=1.3, marker='o', ms=2.5, label=fr'$\delta={de[i]:g}$')
    a.plot(r, 100 * het, color=INK, label=r'$\delta\to0$ (saddle connection)')
    a.plot(r, 100 * d['adiabatic_full'], color=MUTED, ls=':', label=r'adiabatic $\max_x\, x(\rho-x)/(3+2x)$')
    a.plot(r, 100 * r ** 2 / 12, color=MUTED, ls='--', lw=1, label=r'$\rho^2/12$')
    a.set(xlabel=r'$\rho=s_2/s_1$', ylabel=r'peak overshoot $(c_1-s_1)/s_1$ (%)', ylim=(0, 7), xlim=(0, 1),
          title=r'Overshoot grows with $\rho$, then vanishes at $\rho\to1$ for finite $\delta$')
    a.legend(loc='upper left', ncol=2)
    a = ax[1]
    # Near-degenerate targets: is O/O_inf a function of (1-rho) log(1/delta) alone?
    sel = r >= 0.7
    shades = np.linspace(0.25, 1, sel.sum())
    for k, j in enumerate(np.where(sel)[0]):
        x = (1 - r[j]) * np.log(1 / de[1:])
        a.loglog(x, O[1:, j] / het[j], 'o-', lw=0.8, ms=3.5, color=plt.cm.Blues(shades[k]), label=fr'$\rho={r[j]:g}$')
    a.set(xlabel=r'$(1-\rho)\,\log(1/\delta)$', ylabel=r'overshoot / $\delta\to0$ limit', ylim=(1e-3, 1.5),
          title=r'Near degeneracy, one scaling variable')
    a.legend(loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig('figures/fig2_scaling.png')


def fig3():
    d, js = np.load('results/e3.npz'), load('e3')
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for a, rho in zip(ax, ['0.2', '0.5', '0.8']):
        tag, x = f'rho{rho}', float(rho)
        s = np.array([1.0, x])[:, None]
        c0, obs, sep = d[f'{tag}_c0'], d[f'{tag}_observed'], d[f'{tag}_sep'] / s
        a.scatter(*(c0[obs == 1] / s.T).T, s=9, color=C1, label=r'$c_1$ overshoots')
        a.scatter(*(c0[obs == 2] / s.T).T, s=9, color=C2, marker='s', label=r'$c_2$ overshoots')
        a.plot(sep[0], sep[1], color=INK, label='separatrix (strong stable manifold)')
        g = np.geomspace(1e-5, 1, 50)
        a.plot(g, g / x, color=MUTED, ls='--', lw=1, label=r'equal amplitudes $c_1=c_2$')
        info = js[tag]
        a.set(xscale='log', yscale='log', xlim=(1e-4, 1.3), ylim=(1e-4, 1.3), xlabel=r'$c_1(0)/s_1$',
              title=fr'$\rho={rho}$: winner predicted in {info["predicted_equals_observed"]}/{info["n"]}')
    ax[0].set_ylabel(r'$c_2(0)/s_2$')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.savefig('figures/fig3_genericity.png')


def fig4():
    d, js = np.load('results/e4.npz'), load('e4')['results']
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    style = {'full': (INK, '-', 'full training'), 'frozen_u': (C2, '-', 'freeze shared $u$ at crossing'),
             'clamp_c1': (C1, '--', 'project out motion of $c_1$')}
    s = np.array([1.0, 0.5])[:, None]
    for name, (col, ls, lab) in style.items():
        t, c = d[f'rho0.5_{name}_t'], d[f'rho0.5_{name}_c']
        ax[0].plot(t, 100 * (c[0] - 1), color=col, ls=ls, label=lab)
        ax[1].semilogy(t, 0.5 * ((s - c) ** 2).sum(0), color=col, ls=ls, label=lab)
    ax[0].set(xlim=(0, 15), xlabel=r'time since $c_1$ reaches $s_1$', ylabel=r'$(c_1-s_1)/s_1$ (%)',
              title=r'$\rho=0.5$: interventions remove the overshoot')
    ax[1].set(xlim=(0, 15), ylim=(1e-12, 1), xlabel=r'time since $c_1$ reaches $s_1$', ylabel=r'$\mathcal{L}$',
              title=r'Pinning $c_1$ barely moves the loss')
    ax[0].legend(loc='upper right')
    rhos = [0.1, 0.2, 0.5, 0.8]
    for name, col, mk in [('frozen_u', C2, 'o'), ('clamp_c1', C1, 's')]:
        y = [100 * (js[f'rho{r}']['arms'][name]['int_L'] / js[f'rho{r}']['arms']['full']['int_L'] - 1) for r in rhos]
        ax[2].plot(rhos, y, color=col, marker=mk, ms=5, label=style[name][2])
    ax[2].axhline(0, color=MUTED, lw=0.8)
    ax[2].set(xlabel=r'$\rho$', ylabel=r'extra $\int\mathcal{L}\,dt$ after crossing (%)',
              title='Freezing $u$ costs loss; pinning $c_1$ does not')
    ax[2].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('figures/fig4_loss.png')


def quartiles(rows, key, idx=0):
    return 100 * np.percentile([r[key][idx] for r in rows], [25, 50, 75])


def band(a, x, q, col, lab, marker='o', ls='-'):
    q = np.array(q)
    a.errorbar(x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], color=col, marker=marker, ms=3.5, capsize=2, ls=ls, label=lab)


def fig5():
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    # (a) step size, aligned sector
    res = load('e5a')['results']
    a = ax[0, 0]
    for shared, col, lab in [(True, C1, 'shared $u$'), (False, C3, 'unshared control')]:
        rows = [r for r in res if r['rho'] == 0.5 and r['shared'] == shared]
        a.plot([r['eta'] for r in rows], [100 * r['O'][0] for r in rows], color=col, marker='o', ms=3, label=lab)
    a.axvline(1 / 3, color=MUTED, ls='--', lw=1)
    a.text(1 / 3 + .006, 0.92, r'$\eta K(s_1)=1$', transform=a.get_xaxis_transform(), fontsize=7.5, color=INK)
    a.set(xlabel=r'GD step size $\eta$', ylabel=r'$c_1$ overshoot (%)', title=r'Step size, $\rho=0.5$: control crosses only past $\eta\approx1/3$')
    a.legend(loc='center left')
    # (b) random dense init, exact population flow
    res = load('e5b')['results']
    a = ax[0, 1]
    sigmas = [1.0, .3, .1, .03, .01]
    for rho, shade in [(.2, 0.45), (.5, 0.7), (.8, 0.95)]:
        q = [100 * np.percentile([r['O1'] for r in res if r['rho'] == rho and r['sigma'] == sg and r['shared']], [25, 50, 75]) for sg in sigmas]
        band(a, sigmas, q, plt.cm.Blues(shade), fr'shared, $\rho={rho}$')
    q = [100 * np.percentile([r['O1'] for r in res if r['sigma'] == sg and not r['shared']], [25, 50, 75]) for sg in sigmas]
    band(a, sigmas, q, C3, r'unshared, all $\rho$', marker='s')
    a.set(xscale='log', yscale='symlog', ylim=(-1e-4, 20), xlabel=r'random dense init scale $\sigma$', ylabel=r'$c_1$ overshoot (%), median and IQR',
          title='Random init, exact flow: signal is sharing-specific at small $\\sigma$')
    a.set_yscale('symlog', linthresh=1e-3)
    a.legend(loc='lower left', ncol=2)
    # (c) finite sample, (d) minibatch; both noiseless teacher, sigma=0.1
    res = load('e5c')['results']
    a = ax[1, 0]
    Ns = [16, 64, 256, 1024, 4096, 16384]
    for shared, col, lab in [(True, C1, 'shared $u$'), (False, C3, 'unshared control')]:
        band(a, Ns, [quartiles([r for r in res if r['rho'] == 0.5 and r['N'] == N and not r['batch'] and r['shared'] == shared], 'O_raw') for N in Ns], col, lab)
    a.axhline(100 * np.median([r['O_raw'][0] for r in res if r['rho'] == 0.5 and r['N'] == 0 and not r['batch'] and r['shared']]), color=INK, ls=':', lw=1, label='population, same init')
    a.set(xscale='log', xlabel=r'sample size $N$, full-batch GD', ylabel=r'$c_1$ overshoot (%), median and IQR', yscale='symlog',
          title=r'Finite data, $\rho=0.5$: scatters the signal without faking it')
    a.set_yscale('symlog', linthresh=1e-3)
    a.set_ylim(-1e-4, 10)
    a.legend(loc='lower right')
    a = ax[1, 1]
    bs = [8, 32, 128, 512]
    for shared, col, lab in [(True, C1, 'shared $u$'), (False, C3, 'unshared control')]:
        rows = [[r for r in res if r['rho'] == 0.5 and r['batch'] == b and r['shared'] == shared] for b in bs]
        band(a, bs, [quartiles(q, 'O_raw') for q in rows], col, lab + ', raw')
        band(a, bs, [quartiles(q, 'O_smooth') for q in rows], col, lab + ', 1-time-unit mean', marker='s', ls='--')
    a.set(xscale='log', xlabel='minibatch size, fresh samples', ylabel=r'$c_1$ overshoot (%), median and IQR',
          title=r'SGD, $\rho=0.5$, $\eta=0.005$: jitter is small against the signal')
    a.set_yscale('symlog', linthresh=1e-3)
    a.set_ylim(-1e-4, 300)
    a.legend(loc='upper center', ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/fig5_controls.png')


def fig6():
    J, tail = load('e6')['results'], load('e6b')['results']
    d = np.load('results/e6.npz')
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    # (a) Adam reverses who overshoots. s = (1, 0.5).
    a = ax[0]
    for arm, ls, name in [('shared', '-', 'shared'), ('control', '--', 'control')]:
        i, c1, c2, _ = d[f'adam_0.001_0.9_0.999_{arm}']
        a.plot(i, 100 * (c1 - 1), color=C1, ls=ls, lw=1.5 if arm == 'shared' else 1, label=fr'$c_1$, {name}')
        a.plot(i, 100 * (c2 / 0.5 - 1), color=C2, ls=ls, lw=1.5 if arm == 'shared' else 1, label=fr'$c_2$, {name}')
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(xlim=(350, 1000), ylim=(-12, 6), xlabel='Adam step', ylabel=r'$(c_k-s_k)/s_k$ (%)',
          title=r'Adam, lr $10^{-3}$, $\rho=0.5$: the weak mode overshoots')
    a.legend(loc='lower right', ncol=2, fontsize=7)
    # (b) Heavy ball: control threshold, plotted against the linear prediction.
    a = ax[1]
    rows = [r for r in J if r['init'] == 'aligned' and r['rho'] == 0.5 and r['opt'][0] == 'hb' and r['opt'][2] > 0 and r['opt'][1] / (1 - r['opt'][2]) <= 0.4]
    for beta, mk in [(.5, 'o'), (.9, 's'), (.99, '^')]:
        for shared, col in [(True, C1), (False, C3)]:
            q = sorted([r for r in rows if r['opt'][2] == beta and r['shared'] == shared], key=lambda r: r['opt'][1])
            x = [3 * r['opt'][1] / (1 - np.sqrt(beta)) ** 2 for r in q]
            a.plot(x, [100 * r['O'][0] for r in q], color=col, marker=mk, ms=3.5, lw=1,
                   label=fr'{"shared" if shared else "control"}, $\beta={beta}$')
    a.axvline(1, color=MUTED, ls='--', lw=1)
    a.set(xscale='log', ylim=(-1e-4, 3000), xlim=(5e-3, 2e2), xlabel=r'$\eta K(s_1)/(1-\sqrt{\beta})^2$', ylabel=r'$c_1$ overshoot (%)',
          title=r'Momentum, $\rho=0.5$: ringing swamps the signal past 1')
    a.set_yscale('symlog', linthresh=1e-3)
    a.legend(loc='upper left', ncol=2, fontsize=7)
    # (c) Adam without beta1: overshoot of c2 collapses on lr / (1 - beta2) and vanishes with it.
    a = ax[2]
    allr = J + tail
    for rho, shade in [(.5, 0.7), (.2, 0.95)]:
        for b2, mk in [(.9, 's'), (.999, 'o')]:
            q = sorted([r for r in allr if r['init'] == 'aligned' and r['rho'] == rho and r['opt'][0] == 'adam' and r['opt'][2] == 0.0 and r['opt'][3] == b2
                        and r['shared'] and r['opt'][1] <= 3e-2], key=lambda r: r['opt'][1])
            a.loglog([r['opt'][1] / (1 - b2) for r in q], [100 * r['O'][1] for r in q], color=plt.cm.Blues(shade), marker=mk, ms=4, lw=1,
                     label=fr'$\rho={rho}$, $\beta_2={b2}$')
    a.set(xlabel=r'lr$/(1-\beta_2)$', ylabel=r'$c_2$ overshoot (%), shared, $\beta_1=0$',
          title=r'Adam: at small lr, size tracks lr$/(1-\beta_2)$')
    a.legend(loc='upper left', fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/fig6_optimizers.png')


if __name__ == '__main__':
    import os
    import sys
    want = sys.argv[1:] or ['fig1', 'fig1a', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']
    os.makedirs('figures', exist_ok=True)
    for f in want:
        globals()[f]()
