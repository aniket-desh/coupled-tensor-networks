"""Figures for belief-probe-controls, from saved results only.

    PYTHONPATH=. python studies/hmm/belief-probe-controls/plot.py [RESULTS_SUBDIR]   # default: lowdim

figures/<sub>_<family>.pdf: one panel per (cell, position group) that has a judged position. Pooled R2 on held-out
prefixes over the judged positions, against the number k of top principal directions the probe may use. Solid: the
belief; dotted: the scrambled target. Black: the exact belief as the activations (the rank-k ceiling); blue: the trained
model; gray: the initial network; orange: the one-hot tokens. Seed 0 is drawn heavier than seeds 1 and 2."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lowdim_summary as LS  # noqa: E402

COL = (('oracle', 'k'), ('model', '#1f77b4'), ('init', '#7f7f7f'), ('tokens', '#d95f02'))


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else 'lowdim'
    (HERE / 'figures').mkdir(exist_ok=True)
    summary, _ = LS.summarize(sub, quiet=True)
    data = LS.load(sub)
    for fam, fam_out in summary.items():
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
        fig.savefig(HERE / 'figures' / f'{sub}_{fam}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
