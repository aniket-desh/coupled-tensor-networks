"""Figures for the 2026-11-12 experiment summary note.

1. figures/treetn.{pdf,png}: schematic of the 16-leaf TreeTN used in the RHM and byte experiments (drawn here).
2. figures/<study>__<name>.png: PNG copies of the result PDFs each study already produced, for inline display in the
   note. The PDFs under studies/*/figures/ stay the source of truth; nothing here recomputes data.

Run from the repository root:  python notes/2026-11-12-experiments/make_figures.py
PNG export needs PyMuPDF (pip install pymupdf); the diagram needs only matplotlib.
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')

# Result figures shown in the note: (source PDF, output name).
COPIES = [
    ('studies/rhm/rhm-dynamics/figures/widths.pdf', 'rhm-dynamics__widths'),
    ('studies/rhm/rhm-dynamics/figures/rerun_r64.pdf', 'rhm-dynamics__rerun_r64'),
    ('studies/rhm/rhm-sample-efficiency/figures/vs_N.pdf', 'rhm-sample-efficiency__vs_N'),
    ('studies/rhm/rhm-transfer/figures/phase2.pdf', 'rhm-transfer__phase2'),
    ('studies/rhm/rhm-controls/figures/curves.pdf', 'rhm-controls__curves'),
    ('studies/lm/lm-bytes/figures/params.pdf', 'lm-bytes__params'),
    ('studies/ttn/degree/figures/degree.pdf', 'degree__degree'),
    ('studies/ttn/init-scale/figures/activations.pdf', 'init-scale__activations'),
]


def treetn_diagram():
    """Balanced binary TreeTN over n = 16 leaves: leaf embeddings A_i, fusion cores W at levels 1-3, root core C."""
    n, D = 16, 4
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    ax.set_xlim(-0.2, 21.0), ax.set_ylim(-1.75, 6.0), ax.set_aspect('equal'), ax.axis('off')
    y = {l: 0.9 + 1.15 * l for l in range(D + 1)}   # level l of the tree; level D is the root core
    xs = {0: [i + 0.5 for i in range(n)]}
    for l in range(1, D + 1):
        xs[l] = [(xs[l - 1][2 * k] + xs[l - 1][2 * k + 1]) / 2 for k in range(len(xs[l - 1]) // 2)]

    # Shade the part frozen in the rhm-transfer "frozen_C" arms (everything below the root core).
    ax.add_patch(plt.Rectangle((0.0, y[0] - 0.42), 16.0, y[3] - y[0] + 0.84, color='#2a78d6', alpha=0.07, lw=0))
    ax.text(0.05, y[3] + 0.5, 'shaded: frozen in the\nrhm-transfer frozen_learned_C\narm (only $C$ trains)', ha='left', va='bottom', fontsize=8.5,
            color='#2a78d6')

    def node(x, yy, label, r=0.3, fc='#f2f2f2'):
        ax.add_patch(plt.Circle((x, yy), r, fc=fc, ec='k', lw=1.1, zorder=3))
        ax.text(x, yy, label, ha='center', va='center', fontsize=9.5 if len(label) < 8 else 8, zorder=4)

    # Edges, drawn first so the nodes cover their ends.
    for l in range(1, D + 1):
        for k, x in enumerate(xs[l]):
            for c in (2 * k, 2 * k + 1):
                ax.plot([x, xs[l - 1][c]], [y[l], y[l - 1]], 'k-', lw=1.0, zorder=1)
    for x in xs[0]:
        ax.plot([x, x], [y[0], y[0] - 0.75], 'k-', lw=1.0, zorder=1)
        ax.plot(x, y[0] - 0.75, 'ko', ms=4, zorder=2)
    ax.plot([xs[D][0]] * 2, [y[D], y[D] + 0.7], 'k-', lw=1.0)

    for i, x in enumerate(xs[0]):
        node(x, y[0], f'$A_{{{i + 1}}}$', r=0.27)
        ax.text(x, y[0] - 1.0, f'$x_{{{i + 1}}}$', ha='center', va='top', fontsize=9)
    for l in range(1, D):
        for k, x in enumerate(xs[l]):
            node(x, y[l], f'$W^{{({l})}}_{{{k + 1}}}$', r=0.36 if l < 3 else 0.4)
    node(xs[D][0], y[D], '$C$', r=0.36)
    ax.text(xs[D][0] + 0.15, y[D] + 0.72, r'logits $\ell_o$,  $o = 1,\dots,O$', ha='left', va='center', fontsize=10)

    # Bond labels on one edge per level.
    ax.text(xs[0][0] + 0.1, y[0] - 0.55, '$V$', ha='left', fontsize=8.5, color='#555')
    for l in range(D):
        xm, ym = (xs[l][0] + xs[l + 1][0]) / 2, (y[l] + y[l + 1]) / 2
        ax.text(xm - 0.18, ym, '$r$', ha='right', va='center', fontsize=9, color='#555')

    # Right-hand column: what each level's bond covers, and the matching RHM level.
    notes = {0: 'one token  $\\leftrightarrow$ RHM level 4 (tokens)',
             1: '2 tokens  $\\leftrightarrow$ RHM level-3 symbol',
             2: '4 tokens  $\\leftrightarrow$ RHM level-2 symbol',
             3: '8 tokens  $\\leftrightarrow$ RHM level-1 symbol',
             4: 'all 16  $\\leftrightarrow$ RHM root symbol'}
    for l, s in notes.items():
        ax.text(16.6, y[l], s, ha='left', va='center', fontsize=8.5, color='#333')
    ax.text(16.6, y[D] + 0.75, 'a bond at this level summarizes', ha='left', va='center', fontsize=8.5,
            color='#333', style='italic')
    ax.text(8.0, -1.7, r'$z_i = A_i\, e_{x_i}$,   $z^{(v)}_a = \sum_{bc} W^{(v)}_{abc}\, z^{(vL)}_b z^{(vR)}_c$,   '
            r'$\ell_o = \sum_{ab} C_{oab}\, z^{(L)}_a z^{(R)}_b$', ha='center', va='bottom', fontsize=10.5)
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'treetn.{ext}'), bbox_inches='tight', dpi=170)
    plt.close(fig)


def copy_result_figures():
    import pymupdf
    os.makedirs(OUT, exist_ok=True)
    for src, name in COPIES:
        pymupdf.open(src)[0].get_pixmap(dpi=150).save(os.path.join(OUT, name + '.png'))


if __name__ == '__main__':
    treetn_diagram()
    copy_result_figures()
    print('wrote', sorted(os.listdir(OUT)))
