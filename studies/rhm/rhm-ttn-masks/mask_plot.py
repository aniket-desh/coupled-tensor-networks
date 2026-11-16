"""Figures for rhm-ttn-masks, from saved results only.

    PYTHONPATH=. python studies/rhm/rhm-ttn-masks/mask_plot.py RESULTS_DIR RUNS_DIR FIG_DIR [PNG_DIR]

With PNG_DIR, both figures are also saved there as 200-dpi PNGs of the same name (for NOTE.md, where PDFs don't render).

summary.pdf   per grammar, one point per seed and arm: the both-informative R2 of the mixture update (M4, parents of
              tree level >= 2; the additive ceiling as a tick), and the invalid / valid read-out mass ratio (M3).
training.pdf  prefix-task and varied-task excess loss over training, per grammar and arm (one line per seed;
              dotted: the 0.02 nats/token gate)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ('prefix', 'mixed', 'varied')
COL = dict(prefix='#7f7f7f', mixed='#1f77b4', varied='#d95f02')


def main():
    res, runs_dir, fig_dir = (Path(p) for p in sys.argv[1:4])
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    S = json.load(open(res / 'summary.json'))['arms']
    arms = [a for a in ARMS if a in S]
    cells = sorted({n.rsplit('_s', 1)[0] for a in arms for n in S[a]['runs']})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, label in ((axes[0], 'm4_both', 'both-informative update R²'),
                           (axes[1], 'm3_invalid_q', 'invalid / valid read-out mass')):
        for i, c in enumerate(cells):
            for j, a in enumerate(arms):
                xs = i + (j - (len(arms) - 1) / 2) * 0.22
                for s in range(3):
                    q = S[a]['runs'].get(f'{c}_s{s}')
                    if q is None or q[key] is None:
                        continue
                    ax.plot(xs, q[key], 'o', color=COL[a], ms=5, mfc=COL[a] if q['learned'] else 'none',
                            label=a if (i, s) == (0, 0) else None)
                    if key == 'm4_both' and q['m4_both_ceil'] is not None:
                        ax.plot([xs - 0.07, xs + 0.07], [q['m4_both_ceil']] * 2, color=COL[a], lw=1, alpha=0.5)
        ax.set_xticks(range(len(cells)), [c.split('_r')[0] for c in cells], fontsize=8)
        ax.set_ylabel(label)
        ax.axhline(0.8 if key == 'm4_both' else 0.0, color='k', lw=0.6, ls=':')
    axes[0].legend(frameon=False, fontsize=8, title='arm (open: prefix gate not met)', title_fontsize=7)
    axes[0].set_title('M4: merge of uncertain messages (tick: additive ceiling; dotted: 0.8 bar)', fontsize=8)
    axes[1].set_title('M3: invalid child pairs (exact messages: 0)', fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'summary.pdf')
    if png_dir:
        fig.savefig(png_dir / 'summary.png', dpi=200, bbox_inches='tight')

    fig, axes = plt.subplots(2, len(cells), figsize=(3.2 * len(cells), 5), squeeze=False, sharex=True)
    for i, c in enumerate(cells):
        for a in arms:
            for s in range(3):
                f = runs_dir / a / f'{c}_s{s}' / 'result.json'
                if not f.exists():
                    continue
                h = json.load(open(f))['hist'][1:]
                for ax, k in ((axes[0, i], 'excess'), (axes[1, i], 'vexcess')):
                    ax.plot([q['step'] for q in h], [max(q[k], 1e-5) for q in h], color=COL[a], lw=1, alpha=0.8,
                            label=a if s == 0 else None)
        axes[0, i].set_title(c.split('_r')[0], fontsize=9)
        for ax in axes[:, i]:
            ax.axhline(0.02, color='k', lw=0.6, ls=':')  # the gate of each task
            ax.set_xscale('log')
            ax.set_yscale('log')
        axes[1, i].set_xlabel('update')
    axes[0, 0].set_ylabel('prefix task\nexcess loss (nats/token)')
    axes[1, 0].set_ylabel('varied task\nexcess loss (nats/token)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'training.pdf')
    if png_dir:
        fig.savefig(png_dir / 'training.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
