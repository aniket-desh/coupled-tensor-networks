"""Figures from saved message-passing results only.

    PYTHONPATH=. python studies/rhm/rhm-ttn-messages/msg_plot.py RESULTS_DIR FIGURES_DIR [PNG_DIR]

RESULTS_DIR holds one folder per run (<config>_r<r>_s<seed>/ with analysis.json and arrays_ckpt*.npz).
With PNG_DIR, every figure is also saved there as a 200-dpi PNG of the same name (for NOTE.md, where PDFs don't render).

Per (grammar, width), seed 0, node 0 of each tree level:
  prodtable_<cell>.pdf  the production table: the grammar's (parent symbol of each child pair; gray = no production),
      the one read from the trained core on class-mean bonds, and the identity-init tree's. In the learned tables a
      valid pair is colored by its decoded parent, and an invalid pair is shaded by the norm of its merged bond relative
      to the median valid pair (white = 0, what exact message passing gives; black >= 1).
  m4_<cell>.pdf  decoded against exact update, every component, for mixtures of class-mean bonds: empirical message
      pairs and Dirichlet pairs, trained and identity-init. Among the empirical pairs, the both-informative ones (left
      child partially observed, right informative: the pairs H4 is judged on) are drawn darker.
  triangles_<cell>.pdf (v = 3)  exact and decoded messages of held-out local prefixes on the 2-simplex, colored by the
      exact message (RGB); large points = fully observed, small = partially observed.
Across runs:
  summary.pdf   the four pooled measures behind H1-H4 with their controls, one point per seed: M1 partial messages;
      M2 variance explained by the extended message; M3 accuracy and M4 on both-informative pairs, both over parents
      of tree level >= 2.
  training.pdf  the measures over updates, and the excess loss."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
import plot as PL  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from plotting import CATEGORICAL, plt, steps_x  # noqa: E402
from plotting import save as save_pdf  # noqa: E402

PNG_DIR = None  # set in main() from the optional third argument


def save(fig, out):
    if PNG_DIR is not None:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG_DIR / (Path(out).stem + '.png'), dpi=200, bbox_inches='tight')
    save_pdf(fig, out)

BLUE, ORANGE, GREEN, PINK, VIOLET = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[4], CATEGORICAL[6]


def cell(run):
    return run.rsplit('_s', 1)[0]


def label(a):
    return PL.label(a) + f" r={a['cfg']['r']}"


def arrays(a):
    cks = sorted(int(k) for k in a['metrics'])
    return np.load(a['dir'] / f'arrays_ckpt{cks[0]}.npz'), np.load(a['dir'] / f'arrays_ckpt{cks[-1]}.npz'), cks


def levels(npz, prefix):
    return sorted({int(k.split('_')[1][1:]) for k in npz.files if k.startswith(prefix)})


def prodtable(a, out):
    first, last, cks = arrays(a)
    ls = levels(last, 'prod_')
    v = last[f'prod_l{ls[0]}_true'].shape[0]
    cmap = ListedColormap(plt.get_cmap('tab20')(np.arange(20))[:max(v, 2)])
    fig, axes = plt.subplots(len(ls), 3, figsize=(6.6, 2.2 * len(ls)), squeeze=False)
    for i, l in enumerate(ls):
        true = last[f'prod_l{l}_true']
        for j, (name, src, ck) in enumerate((('grammar', None, None), ('trained core', last, cks[-1]), ('identity init', first, cks[0]))):
            ax = axes[i, j]
            ax.grid(False)
            if src is None:
                ax.imshow(np.where(true >= 0, 0.8, np.nan), cmap='gray', vmin=0, vmax=1)
                ax.imshow(np.ma.masked_less(true, 0), cmap=cmap, vmin=-0.5, vmax=max(v, 2) - 0.5)
                ax.set_facecolor('0.85')
            else:
                dec, nrm = src[f'prod_l{l}_decoded'], src[f'prod_l{l}_norm']
                rel = nrm / np.median(nrm[true >= 0])
                ax.imshow(np.where(true < 0, 1 - np.clip(rel, 0, 1), np.nan), cmap='gray', vmin=0, vmax=1)
                ax.imshow(np.ma.masked_where(true < 0, dec), cmap=cmap, vmin=-0.5, vmax=max(v, 2) - 0.5)
                m3 = a['metrics'][str(ck)]['nodes'][f'{l},0']['m3']
                ratio = m3['invalid_norm_ratio']
                ax.set_xlabel(f"accuracy {m3['acc_proj']:.2f}" + (f', invalid/valid norm {ratio:.2f}' if ratio is not None else ''), fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(f'tree level {l}\nleft child symbol', fontsize=7)
    fig.suptitle(f'{label(a)}: production table (color = parent symbol; columns = right child symbol)', fontsize=8)
    fig.tight_layout()
    save(fig, out)


def m4(a, out):
    first, last, cks = arrays(a)
    ls = levels(last, 'm4_')
    cols = [('emp', last, cks[-1], 'empirical pairs, trained'), ('emp', first, cks[0], 'empirical pairs, identity init'),
            ('dir', last, cks[-1], 'Dirichlet pairs, trained'), ('dir', first, cks[0], 'Dirichlet pairs, identity init')]
    fig, axes = plt.subplots(len(ls), 4, figsize=(8.4, 2.1 * len(ls)), squeeze=False)
    for i, l in enumerate(ls):
        for j, (name, src, ck, title) in enumerate(cols):
            ax = axes[i, j]
            if f'm4_l{l}_{name}_true' in src.files:
                ax.scatter(src[f'm4_l{l}_{name}_true'].ravel(), src[f'm4_l{l}_{name}_proj'].ravel(), s=1.5, alpha=0.15,
                           color=BLUE if src is last else '0.5', edgecolors='none', rasterized=True)
                q = a['metrics'][str(ck)]['nodes'][f'{l},0']['m4']
                txt = f"R²={q[f'{name}_proj']['R2']:.2f} (additive ceiling {q[f'{name}_addceil']['R2']:.2f})"
                both = src[f'm4_l{l}_emp_both'] if name == 'emp' and f'm4_l{l}_emp_both' in src.files else None
                if both is not None and both.any():
                    ax.scatter(src[f'm4_l{l}_emp_true'][both].ravel(), src[f'm4_l{l}_emp_proj'][both].ravel(), s=2.5, alpha=0.5,
                               color=VIOLET if src is last else '0.2', edgecolors='none', rasterized=True)
                    if 'emp_both_proj' in q:
                        txt += f"\nboth informative: R²={q['emp_both_proj']['R2']:.2f} (ceiling {q['emp_both_addceil']['R2']:.2f})"
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6, va='top')
            ax.plot([0, 1], [0, 1], color='0.4', lw=0.5)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.3, 1.3)
            if i == 0:
                ax.set_title(title, fontsize=7)
            if i == len(ls) - 1:
                ax.set_xlabel('exact update')
            if j == 0:
                ax.set_ylabel(f'tree level {l}\ndecoded')
    fig.suptitle(f'{label(a)}: merging mixtures of class-mean bonds against the exact message update (node 0 of each level)', fontsize=8)
    fig.tight_layout()
    save(fig, out)


def triangles(a, out):
    first, last, cks = arrays(a)
    ls = levels(last, 'tri_')
    if not ls:
        return
    rows = [('true', last, 'exact'), ('proj', last, 'decoded, trained'), ('proj', first, 'decoded, identity init')]
    fig, axes = plt.subplots(3, len(ls), figsize=(2.2 * len(ls), 6.0), squeeze=False)
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    for j, l in enumerate(ls):
        for i, (key, src, name) in enumerate(rows):
            ax = axes[i, j]
            true, case = src[f'tri_l{l}_true'], src[f'tri_l{l}_case']
            x, y = PL.tri_xy(src[f'tri_l{l}_{key}'])
            ax.plot(outline[:, 0], outline[:, 1], color='0.6', lw=0.6)
            for c, s, al in ((2, 14, 0.5), (1, 3, 0.4)):
                ax.scatter(x[case == c], y[case == c], c=np.clip(true[case == c], 0, 1), s=s, alpha=al, edgecolors='none', rasterized=True)
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.2, np.sqrt(3) / 2 + 0.2)
            ax.set_aspect('equal')
            ax.axis('off')
            if i == 0:
                n = a['metrics'][str(cks[-1])]['nodes'][f'{l},0']['info']
                ax.set_title(f"tree level {l} (span {2 ** l})\n{n['n_partial_messages']} partial messages", fontsize=7)
            if j == 0:
                ax.text(-0.25, 0.43, name, rotation=90, fontsize=7, va='center', transform=ax.transData)
    fig.suptitle(f'{label(a)}: upward messages of held-out local prefixes, node 0 of each level (projective probe)', fontsize=8)
    fig.tight_layout()
    save(fig, out)


def summary(runs, out):
    by = defaultdict(list)
    for r, a in runs.items():
        by[cell(r)].append(a)
    names = sorted(by, key=lambda n: (by[n][0]['cfg']['r'], n))
    fig, axes = plt.subplots(1, len(names), figsize=(1.9 * len(names), 2.8), sharey=True, squeeze=False)
    g = lambda P, key, f='R2': (P.get(key) or {}).get(f)
    spec = [  # x position: (label, scope, [(key, field, which checkpoint, marker, color, legend)])
        ('M1\npartial', 'all', [('m1_proj_partial', 'R2', 1, 'o', BLUE, 'trained'), ('m1_proj_partial', 'R2', 0, 'o', '0.6', 'identity init'),
                                ('m1_tok_aff_partial', 'R2', 1, '^', GREEN, 'token readout'),
                                ('m1_tok2_aff_partial@tok2', 'R2', 1, 'v', GREEN, 'tokens + pair products (its nodes)'),
                                ('m1_scr_proj_partial', 'R2', 1, 'x', PINK, 'scrambled target')]),
        ('M2 ve\nextended', 'all', [('m2_observed_xmsg', 've', 1, 'o', BLUE, None), ('m2_observed_xmsg', 've', 0, 'o', '0.6', None),
                                    ('m2_observed_msg', 've', 1, 'D', VIOLET, 'M2 grouped by the message alone')]),
        ('M3 acc.\nlevel ≥ 2', 'l2+', [('m3', 'acc_proj', 1, 'o', BLUE, None), ('m3', 'acc_proj', 0, 'o', '0.6', None)]),
        ('M4 both\nlevel ≥ 2', 'l2+', [('m4_emp_both_proj', 'R2', 1, 'o', BLUE, None), ('m4_emp_both_proj', 'R2', 0, 'o', '0.6', None),
                                       ('m4_emp_both_addceil', 'R2', 1, '_', 'k', 'additive ceiling')])]
    handles = {}
    for j, n in enumerate(names):
        ax = axes[0, j]
        for k, a in enumerate(by[n]):
            cks = sorted(int(c) for c in a['metrics'])
            for x, (_, scope, items) in enumerate(spec):
                P = [a['metrics'][str(cks[0])]['pooled'][scope], a['metrics'][str(cks[-1])]['pooled'][scope]]
                for key, f, which, mk, col, lab in items:
                    val = g(P[which], key, f)
                    if val is not None:
                        h = ax.plot([x + 0.12 * (k - 1)], [max(val, -0.5)], mk, color=col, ms=3.5)[0]
                        if lab:
                            handles[lab] = h
        ax.set_xticks(range(len(spec)), [s[0] for s in spec], fontsize=5.5)
        ax.set_xlim(-0.5, len(spec) - 0.5)
        ax.set_ylim(-0.52, 1.03)
        ax.set_title(label(by[n][0]).replace(' s=2', '').replace(' r=', '\nr='), fontsize=6.5)
    axes[0, 0].set_ylabel('pooled measure (clipped at −0.5)')
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=6)
    save(fig, out)


def training(runs, out):
    panels = [('m1_proj_partial', 'R2', 'M1: partial messages, projective R²', 'all'),
              ('m2_observed_xmsg', 've', 'M2: variance explained by the extended message', 'all'),
              ('m3', 'acc_proj', 'M3: production table accuracy, level ≥ 2', 'l2+'),
              ('m4_emp_both_proj', 'R2', 'M4: update on both-informative pairs, R²', 'l2+')]
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.7))
    names = sorted({cell(r).rsplit('_r', 1)[0] for r in runs})
    rmax = max(a['cfg']['r'] for a in runs.values())
    for r, a in runs.items():
        col = CATEGORICAL[names.index(cell(r).rsplit('_r', 1)[0]) % len(CATEGORICAL)]
        ls = '-' if a['cfg']['r'] == rmax else ':'
        cks = sorted(int(c) for c in a['metrics'])
        for ax, (key, f, _, scope) in zip(axes, panels):
            ys = [(a['metrics'][str(c)]['pooled'][scope].get(key) or {}).get(f) for c in cks]
            pts = [(c, max(y, -1)) for c, y in zip(cks, ys) if y is not None]
            if pts:
                ax.plot(steps_x([p[0] for p in pts]), [p[1] for p in pts], 'o', ls=ls, ms=2, lw=0.9, color=col,
                        label=PL.label(a) if r.endswith('_s0') and ls == '-' else None)
        h = a['hist']
        axes[4].plot(steps_x([x['step'] for x in h]), [max(x['excess'], 1e-5) for x in h], color=col, lw=0.8, ls=ls)
    for ax, (_, _, title, _) in zip(axes, panels):
        ax.set_xscale('log')
        ax.set_ylim(-1, 1.03)
        ax.set_xlabel('update')
        ax.set_title(title, fontsize=7)
    axes[4].set_xscale('log')
    axes[4].set_yscale('log')
    axes[4].set_xlabel('update')
    axes[4].set_title('excess test loss over Bayes (nats/token)', fontsize=7)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=5.5, loc='lower right')
    fig.tight_layout()
    save(fig, out)


def main():
    global PNG_DIR
    runs = PL.load(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    PNG_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    for r, a in runs.items():
        if r.endswith('_s0'):
            prodtable(a, fig_dir / f'prodtable_{cell(r)}.pdf')
            m4(a, fig_dir / f'm4_{cell(r)}.pdf')
            triangles(a, fig_dir / f'triangles_{cell(r)}.pdf')
    summary(runs, fig_dir / 'summary.pdf')
    training(runs, fig_dir / 'training.pdf')


if __name__ == '__main__':
    main()
