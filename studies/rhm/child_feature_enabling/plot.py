"""Regenerate every figure of the study from equations and the saved results (no computation beyond that).

  python -m studies.rhm.child_feature_enabling.plot

  frozen-root.pdf, grammar-spectra.pdf   closed forms and the saved top-rule spectra
  planted-time-laws.pdf                  results/planted/planted.json
  checkpoint-readout.pdf                 results/checkpoints/checkpoints.json
  intervention*.pdf                      results/intervention*/pairs/*.json (one figure per protocol)
A figure whose results file is absent is skipped with a message.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from studies.rhm.plotting import CATEGORICAL, INK, INK2, plt, save


HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
BLUE, ORANGE, GREEN = CATEGORICAL[:3]
SEED_STYLES = ('-', '--', ':')


def load(path):
    path = HERE / path
    if not path.exists():
        print('skipped (no results):', path)
        return None
    return json.loads(path.read_text())


def panel_title(ax, letter, title):
    """Left-aligned title led by a bold panel letter."""
    ax.set_title(rf'$\mathbf{{{letter}}}$  {title}', loc='left', fontsize=8)


# ------------------------------------------------------------------------------------- closed forms and spectra

def theory():
    r = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), layout='constrained')
    ax, heat = axes
    ax.plot(r, r**4, color=CATEGORICAL[0], linewidth=2)
    threshold = 0.5**0.25
    ax.plot([0, threshold, threshold], [.5, .5, 0], '--', color='#626262')
    ax.scatter([threshold], [.5], color=CATEGORICAL[0], zorder=3)
    ax.annotate('0.841', (threshold, .02), xytext=(-5, 6), textcoords='offset points', ha='right')
    ax.set(title='Tied child tables', xlabel='Child alignment $r$',
           ylabel='Maximum fraction explained', xlim=(0, 1), ylim=(0, 1))
    upper = (r[:, None] * r[None, :])**2
    levels = [.0625, .25, .5, .75]
    heat.contourf(r, r, upper, levels=np.linspace(0, 1, 21), cmap='Blues', vmin=0, vmax=1)
    contours = heat.contour(r, r, upper, levels=levels, colors='#16354d', linewidths=.8)
    heat.clabel(contours, fmt={.0625: '1/16', .25: '1/4', .5: '1/2', .75: '3/4'}, fontsize=7)
    heat.plot(r, r, '--', color='#585858', linewidth=.8)
    heat.set(title='Separate child tables', xlabel='Left alignment', ylabel='Right alignment',
             xlim=(0, 1), ylim=(0, 1), aspect='equal')
    heat.grid(False)
    save(fig, FIG / 'frozen-root.pdf')

    tables = json.loads((HERE / 'results' / 'grammar_audit.json').read_text())['tables']
    fig, ax = plt.subplots(figsize=(5.7, 2.65), layout='constrained')
    markers = ['o', 's', '^']
    for i, table in enumerate(tables):
        fractions = table['additive_fractions']
        ax.plot(np.arange(1, len(fractions) + 1), fractions, marker=markers[i], ms=3.3,
                color=CATEGORICAL[i], label=f"Grammar {table['seed']}")
    ax.set(xlabel='Root contrast, ordered within each grammar',
           ylabel='Fraction explained by additive fit', ylim=(0, 1), xlim=(.6, 15.4),
           xticks=[1, 3, 5, 7, 9, 11, 13, 15])
    ax.legend()
    save(fig, FIG / 'grammar-spectra.pdf')


# ------------------------------------------------------------------------------------------------ planted sweep

def planted(overlay_delta=0.02):
    data = load('results/planted/planted.json')
    if data is None:
        return
    inv_lambda_2 = 1 / data['meta']['lambda_2']
    arms = data['arms']
    pick = lambda tied, s: sorted((a for a in arms if a['tied'] == tied and a['s'] == s), key=lambda a: -a['delta'])
    styles = {True: dict(color=BLUE, marker='o', label='tied'), False: dict(color=ORANGE, marker='s', label='untied')}
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.7), layout='constrained')
    gap_ax, slope_ax, pole_ax, time_ax = axes.flat

    for tied in (True, False):
        both = pick(tied, 1)
        logs = np.log(1 / np.array([a['delta'] for a in both]))
        gap = np.array([a['events']['hard_half'] - a['events']['left_half'] for a in both])
        st = styles[tied]
        points = dict(ls='none', marker=st['marker'], ms=4.5, color=st['color'], mfc='white', mew=1.2)
        # Theory fixes the slope only (0 tied, 1/lambda_2 untied); the O(1) offset is anchored at the smallest delta.
        slope = 0.0 if tied else inv_lambda_2
        gap_ax.plot(logs, gap[-1] + slope * (logs - logs[-1]), color=st['color'], lw=1)
        gap_ax.plot(logs, gap, **points)
        gap_ax.annotate(st['label'] + ': slope ' + ('0' if tied else r'$1/\lambda_2$'), (logs[2], gap[2]),
                        xytext=(0, 7) if tied else (-6, 5), textcoords='offset points', color=st['color'],
                        ha='center' if tied else 'right')
        mid = (logs[1:] + logs[:-1]) / 2
        slope_ax.axhline(slope, color=st['color'], lw=1)
        slope_ax.plot(mid, np.diff(gap) / np.diff(logs), **points)

        alone = pick(tied, 0)
        deltas = np.array([a['delta'] for a in alone])
        limit = 0.25 if tied else 0.5
        pole_ax.axhline(limit, color=st['color'], lw=1)
        pole_ax.plot(deltas, deltas ** 2 * np.array([a['events']['hard_half'] for a in alone]), **points)
        pole_ax.annotate(st['label'] + ': limit ' + ('1/4' if tied else '1/2'), (deltas[-1], limit), xytext=(0, -10),
                         textcoords='offset points', color=st['color'])

        arm = next(a for a in both if a['delta'] == overlay_delta)
        t = np.array(arm['t'])
        time_ax.plot(t, arm['A_h'], '--', color=st['color'], lw=1)
        time_ax.plot(t, arm['D_h'], '-', color=st['color'], lw=1.6)
        time_ax.plot([arm['events']['left_half']], [-0.03], marker='^', ms=5, color=st['color'], clip_on=False,
                     zorder=5)

    gap_ax.set(xlabel=r'$\log(1/\delta)$', ylabel='Time from $r_L=1/2$ to $D_h=1/2$', ylim=(0, 9.5))
    slope_ax.set(xlabel=r'$\log(1/\delta)$, midpoint of adjacent seeds', ylabel='Adjacent-point slope of the wait',
                 ylim=(-0.15, 1.75))
    slope_ax.annotate(rf'$1/\lambda_2={inv_lambda_2:.3f}$', (0.03, inv_lambda_2), xycoords=('axes fraction', 'data'),
                      xytext=(0, -10), textcoords='offset points', color=ORANGE)
    pole_ax.set(xscale='log', xlabel=r'Initial alignment $\delta$', ylabel=r'$\delta^2\,T_h$', ylim=(0.2, 0.6))
    pole_ax.set_xticks(deltas, [f'{d:g}' for d in deltas])
    pole_ax.minorticks_off()
    time_ax.set(xlabel='Gradient-flow time $t$', ylabel='Fraction of $UV$ variance', ylim=(-0.03, 1.02),
                xlim=(0, max(a['t_final'] for a in arms if a['s'] == 1 and a['delta'] == overlay_delta)))
    time_ax.legend(handles=[Line2D([], [], color=INK2, ls='--', lw=1, label='ceiling $A_h=r_L^2r_R^2$'),
                            Line2D([], [], color=INK2, lw=1.6, label='trained fit $D_h$'),
                            Line2D([], [], color=INK2, ls='none', marker='^', ms=5, label='time of $r_L=1/2$'),
                            Line2D([], [], color=BLUE, lw=1.6, label='tied'),
                            Line2D([], [], color=ORANGE, lw=1.6, label='untied')], loc='upper left', fontsize=7)
    titles = ('Both contrasts ($s=1$): wait for the hard fit', 'Measured slopes against the predicted ones',
              'Hard contrast alone ($s=0$)', rf'$s=1$, $\delta={overlay_delta}$: ceiling and trained fit')
    for ax, letter, title in zip(axes.flat, 'abcd', titles):
        panel_title(ax, letter, title)
    save(fig, FIG / 'planted-time-laws.pdf')


# ------------------------------------------------------------------------------------------- checkpoint readout

def step_axis(ax, last, zero_at):
    """Log update axis with update 0 drawn at `zero_at` (left of the first positive update)."""
    ax.set_xscale('log')
    ticks = [s for s in (100, 1000, 10000, 100000) if zero_at < s <= last]
    ax.set_xticks([zero_at] + ticks, ['0'] + [f'{s:,}' for s in ticks])
    ax.set_xlim(zero_at * 0.8, last * 1.25)
    ax.minorticks_off()


def checkpoints():
    data = load('results/checkpoints/checkpoints.json')
    if data is None:
        return
    rows = [r for r in data['rows'] if r['status'] == 'ok']
    grammars = sorted({r['grammar'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    zero_at = 25
    fig, axes = plt.subplots(3, len(grammars), figsize=(6.8, 5.3), layout='constrained', sharex=True, sharey='row')
    for col, g in enumerate(grammars):
        for s, ls in zip(seeds, SEED_STYLES):
            run = sorted((r for r in rows if r['grammar'] == g and r['seed'] == s), key=lambda r: r['step'])
            x = [max(r['step'], zero_at) for r in run]
            for key, color in (('e', BLUE), ('h', ORANGE)):
                axes[0, col].plot(x, [r[f'A_{key}'] for r in run], ls, color=color, marker='o', ms=2)
                axes[1, col].plot(x, [r[f'D_{key}_ce'] for r in run], ls, color=color, marker='o', ms=2)
            axes[2, col].plot(x, [r['ce'] for r in run], ls, color=INK, marker='o', ms=2)
        axes[0, col].set_title(f'Grammar {g}', fontsize=8)
        axes[2, col].axhline(np.log(16), color=INK2, lw=0.6, ls=(0, (1, 2)))
        axes[2, col].set_xlabel('Adam update of the original run')
        for ax in axes[:, col]:
            step_axis(ax, max(r['step'] for r in rows), zero_at)
    axes[2, 0].annotate('uniform, log 16', (1000, np.log(16)), xytext=(0, -9), textcoords='offset points',
                        color=INK2, fontsize=7)
    axes[0, 0].set(ylabel='Frozen-child ceiling $A_c$\n(best linear root, exact)', ylim=(-0.03, 1.03))
    axes[1, 0].set(ylabel="Classifier's own fit $D_c^{CE}$\n(posterior mean, 20,000 trees)", ylim=(-0.03, 1.03))
    axes[2, 0].set(ylabel='Cross-entropy (nats)\non the diagnostic trees')
    axes[1, -1].legend(handles=[Line2D([], [], color=BLUE, label='easy contrast $c_e$'),
                                Line2D([], [], color=ORANGE, label='hard contrast $c_h$')]
                       + [Line2D([], [], color=INK2, ls=ls, label=f'init seed {s}') for s, ls in zip(seeds, SEED_STYLES)],
                       loc='lower right', fontsize=7)
    save(fig, FIG / 'checkpoint-readout.pdf')


# ------------------------------------------------------------------------------------------------- intervention

ARM_COLORS = {'intact': BLUE, 'detached': ORANGE}
GRAMMAR_MARKERS = ('o', 's', '^')


def load_pairs(folder):
    """{(grammar, seed): pair record} from <folder>/pairs/*.json; folder is relative to this study unless absolute."""
    folder = Path(folder) if Path(folder).is_absolute() else HERE / folder
    files = sorted((folder / 'pairs').glob('g*_s*.json'))
    if not files:
        print('skipped (no results):', folder / 'pairs')
    pairs = [json.loads(f.read_text()) for f in files]
    return {(p['grammar'], p['seed']): p for p in pairs}


def intervention(folder, out, fit_ylim=(-0.03, 1.03)):
    """Per pair: trained hard fit D_h and frozen-child ceiling A_h of both arms against the update; then the two
    half-fit times of each pair joined by a line, with an arrow where an arm is censored at the horizon."""
    pairs = load_pairs(folder)
    if not pairs:
        return
    grammars = sorted({g for g, _ in pairs})
    seeds = sorted({s for _, s in pairs})
    any_pair = next(iter(pairs.values()))
    horizon, every, lr = any_pair['steps'], any_pair['eval_every'], any_pair['lr']
    shown = max(p['arms'][arm]['columns']['step'][-1] for p in pairs.values() for arm in ARM_COLORS)
    fig = plt.figure(figsize=(7.2, 1.45 * len(grammars) + 0.5), layout='constrained')
    grid = fig.add_gridspec(len(grammars), len(seeds) + 1, width_ratios=[1] * len(seeds) + [1.3])
    first = None
    for row, g in enumerate(grammars):
        for col, s in enumerate(seeds):
            ax = fig.add_subplot(grid[row, col], sharex=first, sharey=first)
            first = first or ax
            pair = pairs.get((g, s))
            if pair is None:
                ax.text(0.5, 0.5, 'not run', transform=ax.transAxes, ha='center', color=INK2)
                continue
            for arm, color in ARM_COLORS.items():
                cols = pair['arms'][arm]['columns']
                x = np.maximum(cols['step'], every)  # update 0 is drawn at the first evaluation interval
                ax.plot(x, cols['A_h'], '--', color=color, lw=0.9)
                ax.plot(x, cols['D_h'], '-', color=color, lw=1.4)
            if fit_ylim is not None:
                ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (1, 2)))
            ax.text(0.04, 0.93, f'grammar {g}, init {s}' + ('' if pair['status'] == 'finished' else ', ' + pair['status']),
                    transform=ax.transAxes, va='top', fontsize=7, color=INK2)
            ax.tick_params(labelbottom=row == len(grammars) - 1, labelleft=col == 0)
            if row == len(grammars) - 1:
                ax.set_xlabel('SGD update')
            if col == 0:
                ax.set_ylabel('Fraction of $c_h$ variance')
    first.set_xscale('log')
    first.set_xlim(every, shown * 1.15)
    if fit_ylim is not None:
        first.set_ylim(*fit_ylim)
    else:
        first.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.axes[0].set_title(r'$\mathbf{a}$  Hard contrast, per pair: trained fit $D_h$ (solid), ceiling $A_h$ (dashed)',
                          loc='left', fontsize=8)

    ax = fig.add_subplot(grid[:, -1])
    censored_any = False
    for (g, s), pair in sorted(pairs.items()):
        cross = {arm: pair['arms'][arm]['crossing'] for arm in ARM_COLORS}
        y = [horizon if cross[arm]['censored'] else cross[arm]['T'] for arm in ARM_COLORS]
        jitter = (seeds.index(s) - (len(seeds) - 1) / 2) * 0.05
        ax.plot([jitter, 1 + jitter], y, color=INK2, lw=0.7, alpha=0.7, zorder=1)
        for k, arm in enumerate(ARM_COLORS):
            hollow = cross[arm]['censored']
            censored_any |= hollow
            ax.plot([k + jitter], [y[k]], marker=GRAMMAR_MARKERS[grammars.index(g) % 3], ms=5, color=ARM_COLORS[arm],
                    mfc='white' if hollow else ARM_COLORS[arm], mew=1.1, zorder=3)
            if hollow:  # censored: the true time lies above the horizon
                ax.annotate('', (k + jitter, horizon * 1.45), (k + jitter, horizon * 1.05),
                            arrowprops=dict(arrowstyle='->', color=ARM_COLORS[arm], lw=0.9))
    ax.set_yscale('log')
    ax.set_xticks([0, 1], ['intact', 'easy gradient\ndetached'])
    for label, color in zip(ax.get_xticklabels(), ARM_COLORS.values()):  # the tick labels are the colour key
        label.set_color(color)
        label.set_fontweight('bold')
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel('Updates until $D_h\\geq 1/2$')
    low = min([c['T'] for p in pairs.values() for c in (p['arms'][a]['crossing'] for a in ARM_COLORS)
               if not c['censored']] + [horizon])
    ax.set_ylim(low / 1.8, horizon * (1.7 if censored_any else 1.15))
    ax.axhline(horizon, color=INK2, lw=0.6, ls=(0, (1, 2)))
    ax.annotate(f'horizon {horizon:,}', (1.4, horizon), xytext=(-2, -9), textcoords='offset points', ha='right',
                fontsize=7, color=INK2)
    ax.set_title(r'$\mathbf{b}$  Half-fit time, joined within a pair', loc='left', fontsize=8)
    handles = [Line2D([], [], color=INK2, ls='none', marker=m, ms=4, label=f'grammar {g}')
               for g, m in zip(grammars, GRAMMAR_MARKERS)]
    if censored_any:
        handles.append(Line2D([], [], color=INK2, ls='none', marker='o', mfc='white', ms=4, label='censored (open)'))
    ax.legend(handles=handles, loc='lower right', fontsize=7)
    stop = any_pair.get('stop_fit')
    fig.suptitle(f'Plain SGD, rate {lr:g}, batch 256, horizon {horizon:,} updates per arm'
                 + (f' (a pair ends once both arms have $D_h\\geq{stop:g}$)' if stop else ''),
                 fontsize=8, x=0.01, ha='left')
    save(fig, FIG / out)


def ceilings(folder, out):
    """Two parametric views of the same runs. (a) Trained hard fit against its frozen-child ceiling: points on the
    diagonal mean the head has already fitted what the current children allow. (b) The path through the two ceilings
    (A_e, A_h): which contrast becomes linearly available first in each arm, with the CE/Adam checkpoints of the
    original classifiers on the same axes."""
    pairs = load_pairs(folder)
    if not pairs:
        return
    fig, (fit_ax, path_ax) = plt.subplots(1, 2, figsize=(6.6, 3.2), layout='constrained')
    for ax in (fit_ax, path_ax):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (1, 2)))
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect='equal')
    for pair in pairs.values():
        for arm, color in ARM_COLORS.items():
            cols = pair['arms'][arm]['columns']
            fit_ax.plot(cols['A_h'], cols['D_h'], color=color, lw=0.7, alpha=0.8)
            path_ax.plot(cols['A_e'], cols['A_h'], color=color, lw=0.7, alpha=0.8)
    handles = [Line2D([], [], color=c, lw=1.2, label=f'{arm} ({len(pairs)} pairs)') for arm, c in ARM_COLORS.items()]
    saved = load('results/checkpoints/checkpoints.json')
    if saved is not None:
        rows = [r for r in saved['rows'] if r['status'] == 'ok' and r['step'] <= 300]
        path_ax.plot([r['A_e'] for r in rows], [r['A_h'] for r in rows], ls='none', marker='x', ms=4, mew=0.9,
                     color=INK, zorder=4)
        handles.append(Line2D([], [], color=INK, ls='none', marker='x', ms=4,
                              label='CE/Adam checkpoints,\nupdates 0 to 300'))
    fit_ax.set(xlabel='Frozen-child ceiling $A_h$', ylabel='Trained hard fit $D_h$')
    fit_ax.annotate('head has fitted all\nthe children allow', (0.55, 0.55), xytext=(0.14, 0.8), fontsize=7,
                    color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
    path_ax.set(xlabel='Easy ceiling $A_e$', ylabel='Hard ceiling $A_h$')
    path_ax.legend(handles=handles, loc='lower right', fontsize=7)
    fit_ax.set_title(r'$\mathbf{a}$  Trained fit against its ceiling', loc='left', fontsize=8)
    path_ax.set_title(r'$\mathbf{b}$  Which contrast the children expose first', loc='left', fontsize=8)
    save(fig, FIG / out)


def main():
    theory()
    planted()
    checkpoints()
    intervention('results/intervention', 'intervention-as-written.pdf', fit_ylim=None)
    intervention('results/intervention-extended', 'intervention-extended.pdf')
    ceilings('results/intervention-extended', 'intervention-ceilings.pdf')
    intervention('results/intervention-extended-lr0.1', 'intervention-extended-lr0.1.pdf')


if __name__ == '__main__':
    main()
