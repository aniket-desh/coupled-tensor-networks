"""Slide-sized figures for the 2026-11-18 RHM update, redrawn from saved results only (nothing is trained).

    PYTHONPATH=. uv run --frozen python slides/2026-11-18-rhm/figures/make_figures.py

Style follows github.com/ChenLiu-1996/figures4papers (scientific-figure-making): Helvetica-like sans, no top/right
spines, no grid, frameless legends, its palette (blue = the reference arm, red = the contrast). Legends sit in the right margin so
that no label covers data. Sizes are final inches: the slides include each figure at about 1.2 times its natural
width, so the 9 pt labels print near 11 pt on a 160 mm x 90 mm slide.

Sources (all in git):
  learning.pdf   studies/rhm/rhm-sample-efficiency/results/analysis.json, studies/rhm/rhm-controls/results/curves.json
  forbidden.pdf  studies/rhm/rhm-forced-synonyms/results/runs.json
  capacity.pdf   studies/rhm/rhm-crossed-rank/results/{runs,summary}.json
  belief_mess3.pdf, belief_rrxor.pdf   studies/hmm/hmm-belief-geometry/results/<process>_<model>_s0/ (seed 0), drawn
                 as that experiment's hmm_plot.geometry draws them, with its `stacked` and `r2`
  factored.pdf   studies/hmm/belief-probe-controls/results/factor_product.json
"""
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from studies.rhm import toy as Y  # noqa: E402  (UNIFORM_KL, EXACT: the same constants the analyses use)

HERE = Path(__file__).parent
RHM = Path('studies/rhm')
HMM = Path('studies/hmm')
# The belief experiment's own plotting module (its folder name has a hyphen, so it is loaded by path). Importing it
# sets that study's rcParams, so it is loaded before the slide style below.
_spec = importlib.util.spec_from_file_location('hmm_plot', HMM / 'hmm-belief-geometry/hmm_plot.py')
HP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HP)
PALETTE = {'blue_main': '#0F4D92', 'blue_secondary': '#3775BA', 'red_strong': '#B64342', 'teal': '#42949E',
           'violet': '#9A4D8E', 'neutral': '#CFCECE'}                      # figures4papers palette
INK, INK2 = '#171717', '#545454'                                           # the deck's ink and secondary gray
FLOOR = 1e-18                                                              # float64 zeros are drawn here on log axes

for f in Path('/usr/share/texmf/fonts/opentype/public/tex-gyre').glob('texgyreheros-*.otf'):
    font_manager.fontManager.addfont(str(f))                               # the deck's sans; falls back if absent
plt.rcParams.update({
    'font.family': ['TeX Gyre Heros', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 9, 'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 1.0,
    'ytick.major.width': 1.0, 'xtick.minor.visible': False, 'ytick.minor.visible': False, 'legend.frameon': False,
    'axes.titlesize': 9, 'pdf.fonttype': 42, 'axes.grid': False, 'lines.linewidth': 1.5})
clip = lambda v: np.maximum(np.asarray(v, dtype=float), FLOOR)


def save(fig, name):
    fig.savefig(HERE / name, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', HERE / name)


def side_legend(ax, handles, **kw):
    """Legend in the margin to the right of `ax`, so that no label sits on the data."""
    return ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), borderaxespad=0, handlelength=1.5,
                     handletextpad=0.6, labelspacing=0.55, **kw)


def learning():
    """Left: root test accuracy against the training-set size. Right: root accuracy during training, four architectures."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.5})
    a = json.load(open(RHM / 'rhm-sample-efficiency/results/analysis.json'))
    runs = [q for q in a['runs'].values() if q['task'] == 'root' and q['N'] != 'online']
    Ns = sorted({q['N'] for q in runs})
    med = [np.median([q['test_acc'] for q in runs if q['N'] == N]) for N in Ns]
    ax.axvline(a['P_star'], color=INK2, lw=0.8, ls=':')
    ax.axhline(1 / 16, color=INK2, lw=0.8, ls='--')
    ax.plot(Ns, med, '-', color=PALETTE['blue_main'], lw=1.6, zorder=2)
    for q in runs:
        ax.plot(q['N'], q['test_acc'], 'o', ms=4.5, color=PALETTE['blue_main'], mec='white', mew=0.6, zorder=3, clip_on=False)
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.text(a['P_star'], 1.02, '$P^* = n\\,m^L$', color=INK2, ha='center', va='bottom', transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('training trees $N$ (fixed set)')
    ax.set_ylabel('root test accuracy')

    c = json.load(open(RHM / 'rhm-controls/results/curves.json'))
    arms = {'tree': ('tree', PALETTE['blue_main']), 'tied': ('tied tree', PALETTE['teal']),
            'chain': ('chain,\nwidth 40', PALETTE['violet']), 'perm': ('permuted\ntree', PALETTE['red_strong'])}
    for arm, (label, col) in arms.items():
        for k, q in c.items():
            if k.startswith(f'root_{arm}_'):
                bx.plot(np.maximum(q['step'], 1), q['diag_acc'], color=col, lw=0.9, alpha=0.75)
    bx.axhline(1 / 16, color=INK2, lw=0.8, ls='--')
    bx.set_xscale('log')
    bx.set_xlim(10, 4e4)
    bx.set_xticks([10, 100, 1000, 10000])
    bx.set_ylim(0, 1.03)
    bx.set_xlabel('update (fresh trees each)')
    bx.set_ylabel('root accuracy')
    side_legend(bx, [Line2D([], [], color=col, lw=1.6, label=label) for label, col in arms.values()]
                + [Line2D([], [], color=INK2, lw=0.8, ls='--', label='chance, 1/16')])
    save(fig, 'learning.pdf')


def forbidden():
    """Left: measured against derived response to a forbidden pair, exact width-2 trees. Right: the response at every width."""
    rows = json.load(open(RHM / 'rhm-forced-synonyms/results/runs.json'))
    exact = lambda q: q['kl'] < Y.EXACT
    col = {'variance': PALETTE['blue_main'], 'identity': PALETTE['red_strong']}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.55, 1.95), gridspec_kw={'wspace': 0.5})
    ex2 = [q for q in rows if q['r'] == 2 and exact(q)]
    lim = [0.04, 120]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls=':')
    for q in ex2:
        ax.plot(q['forb_pred'], q['forb_rho'], 'o', ms=4, color=col[q['init']], mec='white', mew=0.5, alpha=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_xlabel(r'$\rho_b$ from the formulas')
    ax.set_ylabel(r'measured $\rho_b$')
    ax.set_title(f'$r = 2$, {len(ex2)} exact runs', loc='left')

    for q in rows:
        if exact(q) and q['r'] >= 2:
            for b, v in enumerate(q['forb_rho']):
                dx = (-0.17 if q['init'] == 'variance' else 0.17) + 0.012 * (q['seed'] % 20 - 9.5)
                bx.plot(q['r'] + dx, v, 'o', ms=2.6, color=col[q['init']], mew=0, alpha=0.6)
    bx.axhline(1, color=INK2, lw=0.8, ls='--')
    bx.set_yscale('log')
    bx.set_ylim(0.01, 120)
    bx.set_xlim(1.5, 4.5)
    bx.set_xticks([2, 3, 4])
    bx.set_xlabel('bond dimension $r$')
    bx.set_ylabel(r'measured $\rho_b$')
    bx.set_title('every exact run', loc='left')
    side_legend(bx, [Line2D([], [], color=col[k], marker='o', ms=4, ls='', label=f'{k}\ninit') for k in col]
                + [Line2D([], [], color=INK2, lw=0.8, ls='--', label='a valid\npair'),
                   Line2D([], [], color=INK2, lw=0.8, ls=':', label='measured =\nformula')])
    save(fig, 'forbidden.pdf')


GROUP = {'aligned': PALETTE['blue_main'], 'crossed': PALETTE['red_strong']}
MARK = {'variance': 'o', 'identity': '^'}


def capacity():
    """Excess loss after training against bond dimension, 20 runs per grammar, pairing and width."""
    rows = [q for q in json.load(open(RHM / 'rhm-crossed-rank/results/runs.json')) if q['default_rate']]
    bounds = json.load(open(RHM / 'rhm-crossed-rank/results/summary.json'))['verdicts']['bounds']
    fig, axes = plt.subplots(1, 2, figsize=(4.55, 2.0), sharey=True, gridspec_kw={'wspace': 0.1})
    for ax, name in zip(axes, ('mod', 'sym')):
        for q in rows:
            if q['grammar'] == name:
                dx = (-0.19 if q['grouping'] == 'aligned' else 0.19) + 0.014 * (q['seed'] % 10 - 4.5) + (0.07 if q['init'] == 'identity' else -0.07)
                ax.plot(q['r'] + dx, clip(q['kl']), MARK[q['init']], ms=3.2, color=GROUP[q['grouping']], alpha=0.75, mew=0)
        same = bounds[f'{name}_aligned'] == bounds[f'{name}_crossed']
        for g in ('aligned', 'crossed'):
            b = bounds[f'{name}_{g}']                                      # exact needs r >= b: the line sits left of b
            ax.axvline(b - 0.5 + (0 if not same else -0.04 if g == 'aligned' else 0.04), color=GROUP[g], lw=1.0, ls='--')
            if not same:                                                   # the bound, written above its line
                ax.text(b - 0.5 + (-0.06 if g == 'aligned' else 0.06), 1.02, f'$r \\geq {b}$', color=GROUP[g],
                        ha='right' if g == 'aligned' else 'left', va='bottom', transform=ax.get_xaxis_transform())
        if same:
            ax.text(bounds[f'{name}_aligned'] - 0.5 + 0.1, 1.02, f"both $r \\geq {bounds[name + '_aligned']}$", color=INK, ha='left', va='bottom',
                    transform=ax.get_xaxis_transform())
        ax.axhline(Y.UNIFORM_KL, color=INK2, lw=0.8, ls=':')
        ax.axhline(Y.EXACT, color=INK2, lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_ylim(3e-19, 0.5)
        ax.set_yticks([1e-18, 1e-14, 1e-10, 1e-6, 1e-2])
        ax.set_xticks(range(1, 7))
        ax.set_xlabel('bond dimension $r$')
        ax.text(0.97, 0.56, name, fontweight='bold', ha='right', va='center', transform=ax.transAxes)   # no run ends near 1e-8 at r = 5, 6
    axes[0].set_ylabel('excess loss after training (nats)')
    side_legend(axes[1], [Line2D([], [], color=GROUP[g], lw=1.6, label=g) for g in GROUP]
                + [Line2D([], [], color=INK2, marker=MARK[k], ms=4, ls='', label=f'{k}\ninit') for k in MARK]
                + [Line2D([], [], color=INK2, lw=1.0, ls='--', label='exact\nbelow'),
                   Line2D([], [], color=INK2, lw=0.8, ls=':', label='uniform\npredictor')])
    save(fig, 'capacity.pdf')


def belief(proc, name):
    """Beliefs decoded by each model's probe, held-out sequences of seed 0, all positions overlaid. Top row: the trained
    network; bottom row: the same probe on the network at initialization. Probes as in hmm_plot.geometry: affine on the
    transformer's last layer; projective on the chain state and on the tree's top bonds (with their product after T/2).
    Coordinates: the 2-simplex for a 3-state belief (Mess3), otherwise the top two principal components of the true
    beliefs (RRXOR). Color: the true belief, the same in every panel."""
    runs = HP.PL.load(HMM / 'hmm-belief-geometry/results')
    head = {'transformer': 'transformer', 'chain': 'chain', 'tree': 'tree, $r = 32$', 'tree8': 'tree, $r = 8$'}
    sel = {m: runs[f'{proc}_{m}_s0'] for m in HP.MODELS}
    S = next(iter(sel.values()))['cfg']['S']
    fig, axes = plt.subplots(2, 5, figsize=(5.6, 2.42), gridspec_kw={'wspace': 0.04, 'hspace': 0.22})
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    basis = None
    for j, (m, a) in enumerate(sel.items()):
        cks = sorted(int(c) for c in a['metrics'])
        for i, ck in enumerate((cks[-1], cks[0])):                         # last checkpoint, then initialization
            true, pred = HP.stacked(a, ck)
            if basis is None:                                              # one projection and coloring, from the truth
                uniq = np.unique(true.round(5), axis=0)
                few = len(uniq) <= 200
                if S == 3:
                    proj, col_of = (lambda z: np.stack(HP.PL.tri_xy(z), 1)), (lambda tr, u: np.clip(tr, 0, 1))
                else:
                    mu = (uniq if few else true).mean(0)
                    Vh = np.linalg.svd((uniq if few else true) - mu, full_matrices=False)[2][:3]
                    proj, col_of = (lambda z: (z - mu) @ Vh[:2].T), (lambda tr, u: HP.PL.rgb((tr - mu) @ Vh.T, u))
                basis = (proj, col_of, few)
            proj, col_of, few = basis
            uid = np.unique(true.round(5), axis=0, return_inverse=True)[1].ravel()
            c = col_of(true, uid)
            if j == 0:
                z = proj(true)
                axes[i, 0].scatter(z[:, 0], z[:, 1], c=c, s=9 if few else 0.3, alpha=1 if few else 0.5, edgecolors='none', rasterized=True)
                lims = (z.min(0), z.max(0))
            ax = axes[i, j + 1]
            z = proj(pred)
            ax.scatter(z[:, 0], z[:, 1], c=c, s=1.0 if few else 0.3, alpha=0.15 if few else 0.5, edgecolors='none', rasterized=True)
            if few:                                                        # mean probe output per distinct true belief
                ids = np.unique(uid)
                ax.scatter(*np.stack([z[uid == k].mean(0) for k in ids]).T, c=np.stack([c[uid == k][0] for k in ids]),
                           s=9, edgecolors='k', linewidths=0.3)
            ax.set_xlabel(f'$R^2 = {HP.r2(true, pred):.3f}$', labelpad=1, color=INK2)
            if i == 0:
                ax.set_title(head[m])
    axes[0, 0].set_title('Bayes belief')
    axes[0, 0].set_ylabel('trained')
    axes[1, 0].set_ylabel('untrained')
    for ax in axes.flat:
        if S == 3:
            ax.plot(outline[:, 0], outline[:, 1], color='0.6', lw=0.6, zorder=0)
            ax.set_xlim(-0.12, 1.12)
            ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.12)
        else:
            pad = 0.25 * (lims[1] - lims[0]).max()
            ax.set_xlim(lims[0][0] - pad, lims[1][0] + pad)
            ax.set_ylim(lims[0][1] - pad, lims[1][1] + pad)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(HERE / name, bbox_inches='tight', pad_inches=0.02, dpi=400)
    plt.close(fig)
    print('wrote', HERE / name)


def factored():
    """`factored` (two independent 3-state processes): R^2 of the 9-state joint belief on held-out prefixes, three seeds
    per model, read two ways: the product of the model's two marginal probes, and one direct joint probe."""
    fp = json.load(open(HMM / 'belief-probe-controls/results/factor_product.json'))['runs']
    arms = (('transformer', 'transformer'), ('chain', 'chain'), ('tree', 'tree,\n$r = 32$'), ('tree8', 'tree,\n$r = 8$'))
    fig, ax = plt.subplots(figsize=(4.3, 2.35))
    series = (('product_aff', 'o', PALETTE['blue_main'], 'product of the model\'s\nmarginal probes'),
              ('direct_aff', 's', PALETTE['red_strong'], 'direct joint probe'))
    for k, (arm, _) in enumerate(arms):
        for sd in range(3):
            q = fp[f'factored_{arm}_s{sd}']
            for key, mk, col, _ in series:
                ax.plot(k + 0.13 * (sd - 1), q[key], mk, ms=5, color=col, mec='white', mew=0.5, clip_on=False)
    q = fp['factored_chain_s0']
    ax.axhline(q['floor_aff'], color=INK2, lw=0.8, ls='--')
    ax.axhline(q['tok_product'], color=INK2, lw=0.8, ls=':')
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([lab for _, lab in arms])
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(0.7, 1.005)
    ax.set_ylabel('$R^2$ of the joint belief')
    side_legend(ax, [Line2D([], [], color=col, marker=mk, ms=5, ls='', label=lab) for _, mk, col, lab in series]
                + [Line2D([], [], color=INK2, lw=0.8, ls=':', label=f"marginals read from\nthe tokens ({q['tok_product']:.3f})"),
                   Line2D([], [], color=INK2, lw=0.8, ls='--', label=f"a state holding only\nthe marginals ({q['floor_aff']:.3f})")])
    save(fig, 'factored.pdf')


if __name__ == '__main__':
    learning()
    forbidden()
    capacity()
    belief('mess3', 'belief_mess3.pdf')
    belief('rrxor', 'belief_rrxor.pdf')
    factored()
