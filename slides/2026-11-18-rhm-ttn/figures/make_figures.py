"""Slide-sized figures for the RHM / tree tensor network deck of 2026-11-18, redrawn from saved results only.

    PYTHONPATH=. uv run --frozen python slides/2026-11-18-rhm-ttn/figures/make_figures.py

Style as in slides/2026-11-18-rhm/figures/make_figures.py (github.com/ChenLiu-1996/figures4papers): Helvetica-like
sans, no top/right spines, no grid, frameless legends in the right margin, its palette. Sizes are final inches; the
slides scale each figure by about 1.1 to 1.2, so 9 pt labels print at 10 to 11 pt on a 160 mm x 90 mm slide.

Sources (all in git, under studies/rhm/):
  samples.pdf, memorize.pdf     rhm-sample-efficiency/results/{analysis,curves}.json
  arch_curves.pdf, arch_final.pdf   rhm-controls/results/curves.json
  transfer.pdf                  rhm-transfer/results/{analysis,curves}.json
  belief_second.pdf, belief_first.pdf   rhm-ttn-belief-geometry/results/summary.json
  messages.pdf                  rhm-ttn-messages/results/summary.json
  masks.pdf                     rhm-ttn-masks/results/summary.json
Every point is one run. Nothing is averaged except the median lines named in a legend.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
RHM = Path('studies/rhm')
PALETTE = {'blue_main': '#0F4D92', 'blue_secondary': '#3775BA', 'red_strong': '#B64342', 'teal': '#42949E',
           'violet': '#9A4D8E', 'neutral': '#CFCECE'}                      # figures4papers palette
INK, INK2, GRAY = '#171717', '#545454', '#9A9A9A'
ARMS = {'tree': ('tree', PALETTE['blue_main']), 'tied': ('tied tree', PALETTE['teal']),
        'chain': ('chain,\nwidth 40', PALETTE['violet']), 'perm': ('permuted\ntree', PALETTE['red_strong'])}

for f in Path('/usr/share/texmf/fonts/opentype/public/tex-gyre').glob('texgyreheros-*.otf'):
    font_manager.fontManager.addfont(str(f))                               # the deck's sans; falls back if absent
plt.rcParams.update({
    'font.family': ['TeX Gyre Heros', 'DejaVu Sans'], 'font.size': 9, 'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 1.0,
    'ytick.major.width': 1.0, 'xtick.minor.visible': False, 'ytick.minor.visible': False, 'legend.frameon': False,
    'axes.titlesize': 9, 'pdf.fonttype': 42, 'axes.grid': False, 'lines.linewidth': 1.5})
load = lambda p: json.load(open(RHM / p))


def save(fig, name):
    fig.savefig(HERE / name, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', HERE / name)


def side_legend(ax, handles, **kw):
    """Legend in the margin to the right of `ax`, so that no label sits on the data."""
    return ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), borderaxespad=0, handlelength=1.5,
                     handletextpad=0.6, labelspacing=0.55, **kw)


def dot(color, label, marker='o', **kw):
    return Line2D([], [], color=color, marker=marker, ms=4.5, ls='', label=label, **kw)


def gname(cell):
    """'v8m2s2L4_r32' -> '$v$=8  $m$=2  $L$=4' (s = 2 in every grammar)."""
    v, m, L = re.match(r'v(\d+)m(\d+)s2L(\d+)', cell).groups()
    return f'{v} · {m} · {L}'


def gkey(cell):
    return tuple(int(z) for z in re.match(r'v(\d+)m(\d+)s2L(\d+)', cell).groups())


# ------------------------------------------------------------------------------------------------------------------
def samples():
    """Fixed training sets of N trees, r = 16 tree, 3 grammars per size. Left: root test accuracy. Right: KL of the
    last-token prediction to the Bayes predictor on test trees (loss-selected checkpoint)."""
    a = load('rhm-sample-efficiency/results/analysis.json')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.5})
    for axis, task, key in ((ax, 'root', 'test_acc'), (bx, 'last', 'test_kl')):
        runs = [q for q in a['runs'].values() if q['task'] == task and q['N'] != 'online']
        online = [q[key] for q in a['runs'].values() if q['task'] == task and q['N'] == 'online']
        Ns = sorted({q['N'] for q in runs})
        axis.axvline(a['P_star'], color=INK2, lw=0.8, ls=':')
        axis.axhline(np.median(online), color=PALETTE['teal'], lw=1.3, ls='-.')
        axis.plot(Ns, [np.median([q[key] for q in runs if q['N'] == N]) for N in Ns], '-', color=PALETTE['blue_main'], lw=1.6, zorder=2)
        for q in runs:
            axis.plot(q['N'], q[key], 'o', ms=4.5, color=PALETTE['blue_main'], mec='white', mew=0.6, zorder=3, clip_on=False)
        axis.set_xscale('log', base=2)
        axis.set_xticks(Ns)
        axis.text(a['P_star'], 1.02, '$P^* = n\\,m^L$', color=INK2, ha='center', va='bottom', transform=axis.get_xaxis_transform())
        axis.set_xlabel('training trees $N$ (fixed set)')
    ax.axhline(1 / 16, color=INK2, lw=0.8, ls='--')
    ax.set_ylim(0, 1.03)
    ax.set_ylabel('root test accuracy')
    bx.set_yscale('log')
    bx.set_ylim(0.006, 5)
    bx.set_ylabel('last token: KL to Bayes (nats)')
    side_legend(bx, [Line2D([], [], color=PALETTE['blue_main'], marker='o', ms=4.5, mec='white', lw=1.6, label='one grammar,\nmedian'),
                     Line2D([], [], color=PALETTE['teal'], lw=1.3, ls='-.', label='fresh trees\nevery update'),
                     Line2D([], [], color=INK2, lw=0.8, ls='--', label='chance, 1/16')])
    save(fig, 'samples.pdf')


def memorize():
    """Root task, grammar 0, one run per training-set size. Left: loss on the training set (dashed) and on held-out
    trees (solid). Right: held-out accuracy."""
    c = load('rhm-sample-efficiency/results/curves.json')
    Ns = sorted(int(k.split('_N')[1]) for k in c if k.startswith('root_g0_N'))
    ramp = ['#B9D2EC', '#8DB5DF', '#5F97CF', '#3775BA', '#0F4D92', '#082B55']        # light to dark with N
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.5})
    for col, N in zip(ramp, Ns):
        q = c[f'root_g0_N{N}']
        x = np.maximum(q['step'], 1)
        ax.plot(x, np.maximum(np.array(q['fit_loss'], dtype=float), 1e-6), color=col, lw=1.0, ls='--')
        ax.plot(x, np.maximum(q['diag_loss'], 1e-6), color=col, lw=1.3)
        bx.plot(x, q['diag_acc'], color=col, lw=1.3)
    bx.axhline(1 / 16, color=INK2, lw=0.8, ls='--')
    for axis in (ax, bx):
        axis.set_xscale('log')
        axis.set_xlim(10, 1.2e5)
        axis.set_xticks([10, 100, 1000, 10_000, 100_000])
        axis.set_xlabel('update')
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 100)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1, 100])
    ax.set_ylabel('root loss (nats)')
    bx.set_ylim(0, 1.03)
    bx.set_ylabel('held-out root accuracy')
    side_legend(bx, [Line2D([], [], color=col, lw=1.6, label=f'$N = 2^{{{N.bit_length() - 1}}}$') for col, N in zip(ramp, Ns)]
                + [Line2D([], [], color=INK2, lw=1.3, label='held-out'), Line2D([], [], color=INK2, lw=1.0, ls='--', label='training set')])
    save(fig, 'memorize.pdf')


# ------------------------------------------------------------------------------------------------------------------
def arch_curves():
    """Fresh trees every update, 6 runs per arm (3 grammars x 2 inits). Left: root accuracy. Right: last-token KL to Bayes."""
    c = load('rhm-controls/results/curves.json')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.5})
    for arm, (_, col) in ARMS.items():
        for k, q in c.items():
            x = np.maximum(q['step'], 1)
            if k.startswith(f'root_{arm}_'):
                ax.plot(x, q['diag_acc'], color=col, lw=0.9, alpha=0.75)
            if k.startswith(f'last_{arm}_'):
                bx.plot(x, q['diag_kl'], color=col, lw=0.9, alpha=0.75)
    ax.axhline(1 / 16, color=INK2, lw=0.8, ls='--')
    for axis in (ax, bx):
        axis.set_xscale('log')
        axis.set_xlim(10, 4e4)
        axis.set_xticks([10, 100, 1000, 10000])
        axis.set_xlabel('update (fresh trees each)')
    ax.set_ylim(0, 1.03)
    ax.set_ylabel('root accuracy')
    bx.set_yscale('log')
    bx.set_ylim(1e-3, 4)
    bx.set_ylabel('last token: KL to Bayes (nats)')
    side_legend(bx, [Line2D([], [], color=col, lw=1.6, label=label) for label, col in ARMS.values()]
                + [Line2D([], [], color=INK2, lw=0.8, ls='--', label='chance, 1/16')])
    save(fig, 'arch_curves.pdf')


def arch_final():
    """The same 48 runs at their end. Left: first logged update at which diagnostic accuracy reaches 0.9 x the Bayes
    accuracy (runs that never do are drawn on the 'never' line). Right: test error minus the Bayes error on the same
    50,000 test trees. Filled circles: root task; open squares: last-token task."""
    c = load('rhm-controls/results/curves.json')
    order = ('tree', 'tied', 'chain', 'perm')
    never = 1.2e5
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.55})
    for i, arm in enumerate(order):
        col = ARMS[arm][1]
        for task, dx, style in (('root', -0.17, dict(marker='o', mfc=col)), ('last', 0.17, dict(marker='s', mfc='white'))):
            runs = sorted(k for k in c if k.startswith(f'{task}_{arm}_'))
            for j, k in enumerate(runs):
                q = c[k]
                hit = np.nonzero(np.array(q['diag_acc']) >= 0.9 * q['bayes_diag']['acc'])[0]
                x = i + dx + 0.045 * (j - 2.5)
                ax.plot(x, q['step'][hit[0]] if len(hit) else never, ls='', ms=4.2, mec=col, mew=0.9, clip_on=False, **style)
                bx.plot(x, q['bayes_test']['acc'] - q['test']['acc'], ls='', ms=4.2, mec=col, mew=0.9, clip_on=False, **style)
    ax.set_yscale('log')
    ax.set_ylim(100, never)
    ax.set_yticks([100, 1000, 10_000, never])
    ax.set_yticklabels(['100', '1,000', '10,000', 'never'])
    ax.axhline(4e4, color=INK2, lw=0.8, ls=':')
    ax.text(-0.45, 4e4, 'end of training', color=INK2, ha='left', va='bottom', fontsize=7.5)
    ax.set_ylabel('updates to 0.9 × Bayes accuracy')
    bx.axhline(0, color=INK2, lw=0.8, ls='--')
    bx.set_ylim(-0.02, 0.42)
    bx.set_ylabel('test error above Bayes')
    for axis in (ax, bx):
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(['tree', 'tied', 'chain', 'perm.'])
        axis.set_xlim(-0.5, len(order) - 0.5)
    side_legend(bx, [Line2D([], [], color=INK2, marker='o', ms=4.5, ls='', label='root'),
                     Line2D([], [], color=INK2, marker='s', mfc='white', ms=4.5, ls='', label='last token')])
    save(fig, 'arch_final.pdf')


# ------------------------------------------------------------------------------------------------------------------
def transfer():
    """Phase 2 on the 16 held-out root productions, 9 units (3 grammars x 3 inits) per arm. Left: accuracy on them
    during phase 2. Right: first logged update at 0.9 accuracy. Frozen arms train only a fresh root core C."""
    c = load('rhm-transfer/results/curves.json')
    a = load('rhm-transfer/results/analysis.json')
    arms = {'frozen_other_C': ('frozen, other\ngrammar', GRAY), 'frozen_random_C': ('frozen,\nuntrained', PALETTE['neutral']),
            'scratch': ('from scratch', PALETTE['red_strong']), 'finetune': ('fine-tune all', PALETTE['violet']),
            'frozen_learned_C': ('frozen,\npretrained', PALETTE['blue_main']), 'oracle_C': ('true symbols\n(oracle)', PALETTE['teal'])}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.5, 'width_ratios': [1, 1.1]})
    for arm, (_, col) in arms.items():
        for k, q in c.items():
            if k.endswith('/' + arm):
                ax.plot(np.maximum(q['step'], 1), q['unseen_acc'], color=col, lw=0.9, alpha=0.8)
    ax.axhline(a['threshold'], color=INK2, lw=0.8, ls=':')
    ax.axhline(1 / 16, color=INK2, lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlim(1, 1e4)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('phase-2 update')
    ax.set_ylabel('accuracy, held-out productions')
    reach = ('oracle_C', 'frozen_learned_C', 'finetune', 'scratch')
    for i, arm in enumerate(reach):
        ts = [u['arms'][arm]['t_thresh'] for u in a['units'].values()]
        for j, t in enumerate(sorted(ts)):
            bx.plot(i + 0.06 * (j - 4), t, 'o', ms=4.2, color=arms[arm][1], mec='white', mew=0.5, clip_on=False)
    bx.set_xticks(range(len(reach)))
    bx.set_xticklabels(['oracle', 'frozen', 'fine-\ntune', 'from\nscratch'], fontsize=7.5)
    bx.set_xlim(-0.5, len(reach) - 0.5)
    bx.set_ylim(0, 160)
    bx.set_ylabel('updates to 0.9')
    side_legend(bx, [Line2D([], [], color=col, lw=1.6, label=label) for label, col in reversed(list(arms.values()))]
                + [Line2D([], [], color=INK2, lw=0.8, ls=':', label='0.9'), Line2D([], [], color=INK2, lw=0.8, ls='--', label='chance')])
    save(fig, 'transfer.pdf')


# ------------------------------------------------------------------------------------------------------------------
def rows_by_width(ax_pair, cells_by_r, draw, note=None):
    """Two panels (r = 16, r = 32) sharing a list of grammars on the y axis. `draw(ax, y, cell)` plots one cell;
    a grammar that is not in a panel's list gets the gray text `note(cell)`."""
    grams = sorted({c.split('_')[0] for cells in cells_by_r.values() for c in cells}, key=gkey)
    for ax, r in zip(ax_pair, (16, 32)):
        for y, g in enumerate(grams):
            cell = f'{g}_r{r}'
            if cell in cells_by_r[r]:
                draw(ax, y, cell)
            elif note:
                ax.text(0.5, y, note(cell), color=GRAY, ha='center', va='center', fontsize=7.5)
        ax.axvline(0.8, color=INK2, lw=0.8, ls='--')
        ax.set_xlim(-0.04, 1.03)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(len(grams) - 0.5, -0.5)
        ax.set_title(f'$r = {r}$')
        ax.tick_params(axis='y', length=0)
    ax_pair[0].set_yticks(range(len(grams)))
    ax_pair[0].set_yticklabels([gname(g) for g in grams])
    ax_pair[0].text(-0.03, 1.03, '$v$ · $m$ · $L$', color=INK2, ha='right', va='bottom', transform=ax_pair[0].transAxes)
    ax_pair[1].set_yticks([])
    ax_pair[1].spines['left'].set_visible(False)


def belief(name, hyp, series, ceil, xlabel):
    """Pooled R^2 of a probe from the two top bonds to the Bayes belief over pending symbols, on held-out prefixes,
    3 seeds per cell. `series`: (run key, marker, color, size) drawn in that order; `ceil`: run key drawn as a tick."""
    s = load('rhm-ttn-belief-geometry/results/summary.json')
    runs = {q['run']: q for q in s['runs']}
    cells = {r: set(s['verdicts'][f'{hyp}_r{r}']['judged']) for r in (16, 32)}

    def draw(ax, y, cell):
        for sd in range(3):
            q = runs[f'{cell}_s{sd}']
            yy = y + 0.2 * (sd - 1)
            ax.plot(max(q[ceil], -0.03), yy, '|', ms=7, mew=1.3, color=INK, zorder=4)
            for key, mk, col, ms in series:
                ax.plot(max(q[key], -0.03), yy, mk, ms=ms, color=col, mew=1.0 if mk == 'x' else 0, zorder=3, clip_on=False)

    def note(cell):
        return 'not learned' if not s['cells'][cell]['learned'] else 'too few distinct prefixes'

    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.15), gridspec_kw={'wspace': 0.08})
    rows_by_width(axes, cells, draw, note)
    fig.supxlabel(xlabel, fontsize=9, y=-0.1)
    return fig, axes


def belief_second():
    fig, axes = belief('belief_second.pdf', 'h2', (('h2_init_bil', 'o', GRAY, 4), ('h2_bil', 'o', PALETTE['blue_main'], 4.5),
                                                  ('h2_scr', 'x', PALETTE['red_strong'], 4)), 'h2_addceil',
                       '$R^2$ of the belief, second half of the sequence')
    side_legend(axes[1], [dot(PALETTE['blue_main'], 'trained tree'), dot(GRAY, 'identity-init\ntree'),
                          Line2D([], [], color=INK, marker='|', ms=7, mew=1.3, ls='', label='best fit with\nno product term'),
                          Line2D([], [], color=PALETTE['red_strong'], marker='x', ms=4.5, mew=1.0, ls='', label='scrambled\ntarget'),
                          Line2D([], [], color=INK2, lw=0.8, ls='--', label='pass, 0.8')])
    save(fig, 'belief_second.pdf')


def belief_first():
    fig, axes = belief('belief_first.pdf', 'h1', (('h1_init', 'o', GRAY, 4), ('h1_top', 'o', PALETTE['blue_main'], 4.5),
                                                 ('h1_scr', 'x', PALETTE['red_strong'], 4)), 'h1_nexttok',
                       '$R^2$ of the belief, first half of the sequence')
    side_legend(axes[1], [dot(PALETTE['blue_main'], 'trained tree'), dot(GRAY, 'identity-init\ntree'),
                          Line2D([], [], color=INK, marker='|', ms=7, mew=1.3, ls='', label='from the exact\nnext-token\ndistribution'),
                          Line2D([], [], color=PALETTE['red_strong'], marker='x', ms=4.5, mew=1.0, ls='', label='scrambled\ntarget'),
                          Line2D([], [], color=INK2, lw=0.8, ls='--', label='pass, 0.8')])
    save(fig, 'belief_first.pdf')


# ------------------------------------------------------------------------------------------------------------------
def messages():
    """The four pre-set measures of rhm-ttn-messages, one panel each, 3 seeds per learned cell (9 cells). Blue: trained
    tree; gray: identity-init tree; teal triangle: a probe on the one-hot tokens of the same span; black tick: best
    additive fit of the same pairs. Dashed: the pass threshold of that measure."""
    s = load('rhm-ttn-messages/results/summary.json')
    runs = {q['run']: q for q in s['runs']}
    cells = sorted((c for c, v in s['cells'].items() if v['learned']), key=lambda c: (-s['cells'][c]['r'], gkey(c)))
    panels = (('bond gives\nthe message', 'm1_proj', 'm1_init', ('m1_tok', '^', PALETTE['teal']), 0.8),
              ('bond keeps what\nthe task needs', 'm2_ve', 'm2_init', None, 0.8),
              ('valid pair\ngives its parent', 'm3_acc', 'm3_init', None, 0.95),
              ('merge is the\nmessage update', 'm4_both', 'm4_both_init', ('m4_both_ceil', '|', INK), 0.8))
    fig, axes = plt.subplots(1, 4, figsize=(5.1, 2.2), sharey=True, gridspec_kw={'wspace': 0.16})
    ys = {c: i + (0.6 if s['cells'][c]['r'] == 16 else 0) for i, c in enumerate(cells)}
    for ax, (title, key, init, extra, bar) in zip(axes, panels):
        for c in cells:
            for sd in range(3):
                q = runs[f'{c}_s{sd}']
                yy = ys[c] + 0.2 * (sd - 1)
                if extra:
                    ax.plot(max(q[extra[0]], -0.03), yy, extra[1], ms=6 if extra[1] == '|' else 3.6, mew=1.3 if extra[1] == '|' else 0,
                            color=extra[2], zorder=2)
                ax.plot(max(q[init], -0.03), yy, 'o', ms=3.6, color=GRAY, mew=0, zorder=3)
                ax.plot(max(q[key], -0.03), yy, 'o', ms=4.2, color=PALETTE['blue_main'], mew=0, zorder=4, clip_on=False)
        ax.axvline(bar, color=INK2, lw=0.8, ls='--')
        ax.set_xlim(-0.05, 1.04)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_title(title, loc='left', fontsize=8)
        ax.tick_params(axis='y', length=0)
        if ax is not axes[0]:
            ax.spines['left'].set_visible(False)
    axes[0].set_ylim(max(ys.values()) + 0.55, -0.55)
    axes[0].set_yticks([ys[c] for c in cells])
    axes[0].set_yticklabels([gname(c) for c in cells])
    for r in (32, 16):
        y = np.mean([ys[c] for c in cells if s['cells'][c]['r'] == r])
        axes[0].text(-0.62, y, f'$r = {r}$', rotation=90, ha='center', va='center', transform=axes[0].get_yaxis_transform())
    axes[0].text(-0.06, 1.03, '$v$ · $m$ · $L$', color=INK2, ha='right', va='bottom', transform=axes[0].transAxes, fontsize=8)
    side_legend(axes[-1], [dot(PALETTE['blue_main'], 'trained'), dot(GRAY, 'identity\ninit'), dot(PALETTE['teal'], 'probe on\nthe tokens', marker='^'),
                           Line2D([], [], color=INK, marker='|', ms=7, mew=1.3, ls='', label='additive\nfit'),
                           Line2D([], [], color=INK2, lw=0.8, ls='--', label='pass')])
    save(fig, 'messages.pdf')


def masks():
    """rhm-ttn-masks, r = 32, 3 seeds per grammar and training-mask arm. Left: R^2 of the decoded merge against the
    exact message update on pairs where both children are informative (tick: additive fit of that run's pairs).
    Right: read-out mass of child pairs the grammar forbids, relative to valid pairs. Open markers: runs above the
    0.02 nats/token gate on the prefix task."""
    s = load('rhm-ttn-masks/results/summary.json')
    arms = {'prefix': ('prefix masks', PALETTE['blue_main']), 'mixed': ('mixed', PALETTE['red_strong']), 'varied': ('varied masks', PALETTE['teal'])}
    grams = sorted(s['arms']['prefix']['cells'], key=gkey)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.45})
    for i, g in enumerate(grams):
        for a, (arm, (_, col)) in enumerate(arms.items()):
            for sd in range(3):
                q = s['arms'][arm]['runs'][f'{g}_s{sd}']
                x = i + 0.25 * (a - 1) + 0.06 * (sd - 1)
                kw = dict(ls='', ms=4.2, mec=col, mew=0.9, mfc=col if q['learned'] else 'white', clip_on=False)
                ax.plot([x - 0.11, x + 0.11], [q['m4_both_ceil']] * 2, color=INK, lw=0.9, zorder=2)
                ax.plot(x, q['m4_both'], 'o', zorder=3, **kw)
                bx.plot(x, q['m3_invalid_q'], 'o', zorder=3, **kw)
    ax.axhline(0.8, color=INK2, lw=0.8, ls='--')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('merge vs. exact update, $R^2$')
    bx.axhline(1, color=INK2, lw=0.8, ls=':')
    bx.axhline(0, color=INK2, lw=0.8, ls='--')
    bx.set_ylim(-0.08, 2.5)
    bx.set_ylabel('forbidden / valid pair mass')
    for axis in (ax, bx):
        axis.set_xticks(range(len(grams)))
        axis.set_xticklabels([gname(g).replace(' · ', '·') for g in grams], fontsize=8)
        axis.set_xlim(-0.5, len(grams) - 0.5)
        axis.set_xlabel('$v$ · $m$ · $L$')
    side_legend(bx, [dot(col, label) for label, col in arms.values()]
                + [Line2D([], [], color=INK, lw=1.2, label='additive fit'), Line2D([], [], color=INK2, lw=0.8, ls='--', label='pass (left),\nexact (right)'),
                   Line2D([], [], color=INK2, lw=0.8, ls=':', label='same as a\nvalid pair')])
    save(fig, 'masks.pdf')


if __name__ == '__main__':
    samples()
    memorize()
    arch_curves()
    arch_final()
    transfer()
    belief_second()
    belief_first()
    messages()
    masks()
