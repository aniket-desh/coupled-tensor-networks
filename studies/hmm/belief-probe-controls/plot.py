"""Figures for belief-probe-controls, from saved results only.

    PYTHONPATH=. python studies/hmm/belief-probe-controls/plot.py [RESULTS_SUBDIR [EXT]]   # default: lowdim pdf

EXT = png writes 200-dpi copies to figures/png/ for embedding in NOTE.md (GitHub does not render PDFs inline); the PDFs
are not touched.

figures/<sub>_<family>.pdf: one panel per (cell, position group) that has a judged position. Pooled R2 on held-out
prefixes over the judged positions, against the number k of top principal directions the probe may use. Solid: the
belief; dotted: the scrambled target. Black: the exact belief as the activations (the rank-k ceiling); blue: the trained
model; gray: the initial network; orange: the one-hot tokens. Seed 0 is drawn heavier than seeds 1 and 2.

figures/summary.pdf (added after the run, for the write-up; same saved files): left, part 1 on `factored`, the pooled
affine R2 of the joint belief per arm and seed: the product of the model's marginal probes, the direct joint probe, and
the three references (product of token-readout marginals, the floor, the initial network's product). Right, part 2,
one point per run with a judged position: the reverse score `rev` against the forward score G from the top k90(t)
directions (lowdim_summary.py), with the 0.8 bars of the two labels."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lowdim_summary as LS  # noqa: E402

COL = (('oracle', 'k'), ('model', '#1f77b4'), ('init', '#7f7f7f'), ('tokens', '#d95f02'))


def summary(fam_summary, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=(1, 1.15)))
    # left: part 1, `factored`, affine probes
    fp = json.load(open(HERE / 'results' / 'factor_product.json'))['runs']
    arms = (('transformer', 'transformer'), ('chain', 'chain'), ('tree', 'tree, r = 32'), ('tree8', 'tree, r = 8'))
    series = (('product_aff', 'product of the model\'s marginal probes', 'o', '#1f77b4'),
              ('direct_aff', 'direct joint probe', 's', '#d62728'),
              ('init_product_aff', 'product, initial network', 'x', '#7f7f7f'))
    for i, (arm, _) in enumerate(arms):
        for s in (0, 1, 2):
            q = fp[f'factored_{arm}_s{s}']
            for key, label, mk, col in series:
                a.plot(i + 0.12 * (s - 1), q[key], mk, color=col, ms=5, mfc='none' if mk == 's' else col,
                       label=label if (i, s) == (0, 0) else None)
    q = fp['factored_chain_s0']
    a.axhline(q['tok_product'], color='#d95f02', lw=0.9, label=f'product of token-readout marginals ({q["tok_product"]:.3f})')
    a.axhline(q['floor_aff'], color='k', lw=0.9, ls='--', label=f'floor: direct probe on the exact marginals ({q["floor_aff"]:.3f})')
    a.set_xticks(range(len(arms)), [n for _, n in arms], fontsize=8)
    a.set_ylim(0.15, 1.03)
    a.set_ylabel('R² of the 9-state joint belief, held-out prefixes', fontsize=8)
    a.set_title('`factored`: the joint belief from the marginals, three seeds', fontsize=9)
    a.legend(frameon=False, fontsize=6.5, loc='lower left')
    # right: part 2, reverse score against the forward score from the top k90 directions
    kinds = {}
    for fam, cells in fam_summary.items():
        for c, v in cells.items():
            cell, grp = c.split('|')
            model = 'transformer' if fam == 'rhm-transformer' or '_transformer' in cell else 'chain' if '_chain' in cell else 'tree'
            kind = (model if model != 'tree' else f'tree, {"first" if grp == "h1" else "second"} half',
                    'RHM' if fam.startswith('rhm') else 'HMM')
            for q in v['runs'].values():
                if q and q.get('judged'):
                    kinds.setdefault(kind if v['learned'] else ('not learned', ''), []).append((q['rev'], q['G'], q['full_all'] >= LS.G_MIN))
    col = {'transformer': '#1f77b4', 'chain': '#2ca02c', 'tree, first half': '#d95f02', 'tree, second half': '#9467bd',
           'not learned': '#7f7f7f'}
    for (kind, task), pts in sorted(kinds.items(), key=lambda kv: (kv[0][0] == 'not learned', kv[0])):
        mk = 'x' if kind == 'not learned' else 'o' if task == 'HMM' else 's'
        for full_ok in (True, False):
            sel = [(x, y) for x, y, ok in pts if ok == full_ok]
            if sel:
                b.plot(*zip(*sel), mk, color=col[kind], ms=5, mfc=col[kind] if full_ok else 'none', alpha=0.85,
                       label=f'{task} {kind}'.strip() if full_ok or not any(ok for *_, ok in pts) else None)
    b.axvline(LS.G_MIN, color='k', lw=0.6, ls=':')
    b.axhline(LS.G_MIN, color='k', lw=0.6, ls=':')
    b.set_xlim(0, 1.02)
    b.set_ylim(0, 1.02)
    b.set_xlabel('reverse R²: share of the state\'s top-90% variance that is a function of the belief', fontsize=7.5)
    b.set_ylabel('forward R² from the top k90(t) directions', fontsize=8)
    b.set_title('one point per run; hollow: full probe below 0.8 on its eligible positions', fontsize=9)
    b.legend(frameon=False, fontsize=6.5, loc='upper left')
    for ax in (a, b):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else 'lowdim'
    ext = sys.argv[2] if len(sys.argv) > 2 else 'pdf'
    out = HERE / 'figures' / ('png' if ext == 'png' else '')
    out.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['savefig.dpi'] = 200
    fam_summary, _ = LS.summarize(sub, quiet=True)
    data = LS.load(sub)
    if sub == 'lowdim':
        summary(fam_summary, out / f'summary.{ext}')
    if ext == 'pdf' and len(sys.argv) > 3:  # `plot.py lowdim pdf summary`: only the summary figure
        return
    for fam, fam_out in fam_summary.items():
        runs = data[fam][0]
        panels = [(c, v) for c, v in fam_out.items() if any(q and q.get('judged') for q in v['runs'].values())]
        if not panels:
            continue
        ncol = min(4, len(panels))
        nrow = -(-len(panels) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow), squeeze=False)
        for ax in axes.flat[len(panels):]:
            ax.axis('off')
        for ax, (c, v) in zip(axes.flat, panels):
            for i, (n, q) in enumerate(sorted(v['runs'].items())):
                if not q or not q.get('judged'):
                    continue
                cur = runs[n]['curves']
                for label, col in COL:
                    ks = sorted(set.intersection(*[set(map(int, cur[t][label]['belief'])) for t in q['judged']]))
                    for target, ls in (('belief', '-'), ('scr', ':')):
                        ax.plot(ks, [LS.pooled(cur, q['judged'], label, target, lambda t, k=k: k) for k in ks], ls, color=col,
                                lw=1.2 if i == 0 else 0.6, alpha=1 if i == 0 else 0.5,
                                label=f'{label}, {"belief" if target == "belief" else "scrambled"}' if i == 0 else None)
            ax.set_xscale('log', base=2)
            ax.set_ylim(-0.1, 1.02)
            ax.set_title(c.replace('|', ', '), fontsize=8)
            ax.set_xlabel('top-k principal directions', fontsize=7)
            ax.tick_params(labelsize=7)
        axes[0, 0].set_ylabel('R² on held-out prefixes', fontsize=7)
        axes[0, 0].legend(frameon=False, fontsize=5.5)
        fig.tight_layout()
        fig.savefig(out / f'{sub}_{fam}.{ext}')
        plt.close(fig)


if __name__ == '__main__':
    main()
