"""Figures from saved HMM analysis results only.

    PYTHONPATH=. python studies/hmm/hmm-belief-geometry/hmm_plot.py RESULTS_DIR FIGURES_DIR

RESULTS_DIR holds one folder per run (<process>_<model>_s<seed>/ with analysis.json and proj_ckpt*.npz).
All geometry panels show the sequence split's test rows of seed 0, every position t = 1 .. T-1 overlaid, each point
placed by its own position's probe and colored by the true belief. The probe drawn per model:
  transformer  affine on the last layer (L4);   chain  projective on the state h_t;
  tree         projective on the top bonds [h_L, h_R] for t <= T/2 and on [h_L, h_R, h_L (x) h_R] for t > T/2.

  triangle_<process>.pdf (3 states: mess3, coins3, z1r)  the belief simplex: ground truth | transformer | chain | tree,
      trained (top row) and the initial network (bottom row); color = belief as RGB (Shai et al. Fig. 5).
  pca_<process>.pdf (more than 3 states: rrxor, factored, coupled)  the same layout on the top two PCs of the true
      beliefs (of the distinct beliefs when there are at most 200, as Shai et al. Fig. 7; of all rows otherwise), with
      per-belief centroids when there are at most 200 distinct beliefs; color = truth on PCs 1-3, rank-normalized.
  positions.pdf  per-position R2 on unseen prefixes, one row per process and one column per model: affine and
      projective probes, the initial network, the token readout, the next-token ceiling, the scrambled target, and
      for the tree the bilinear set and the additive ceiling; gray = not eligible for the primary set; dashed = T/2.
  summary.pdf    pooled R2 behind each criterion, one point per seed.
  training.pdf   excess loss over Bayes, and the pooled affine R2 of the primary set, against update."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'rhm' / 'rhm-belief-geometry'))
import plot as PL  # noqa: E402
from plotting import CATEGORICAL, INK2, plt, save, steps_x  # noqa: E402

BLUE, ORANGE, GREEN, PINK, VIOLET = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[4], CATEGORICAL[6]
MODELS = ('transformer', 'chain', 'tree', 'tree8')  # arms; tree8 is the tree at r = 8, probed like the tree
PROCESSES = ('mess3', 'rrxor', 'z1r', 'coins3', 'factored', 'coupled')
DRAWN = dict(transformer='affine, last layer', chain='projective, state', tree='projective, top bonds (product after T/2)')


def stacked(a, ck):
    """(true (n, S), pred (n, S)) over all positions for the probe drawn for this model."""
    p = np.load(a['dir'] / f'proj_ckpt{ck}.npz')
    kind, H = a['cfg']['model'], a['half']  # kind: transformer | chain | tree
    tr, pd = [], []
    for t in a['positions']:
        key = dict(transformer='pred_L4', chain='pred_state^proj')[kind] if kind != 'tree' else \
            ('pred_top^proj' if t <= H else 'pred_top_bil^proj')
        tr.append(p[f't{t}_true'])
        pd.append(p[f't{t}_{key}'])
    return np.concatenate(tr), np.concatenate(pd)


def r2(true, pred):
    return 1 - ((true - pred) ** 2).sum() / ((true - true.mean(0)) ** 2).sum()


def geometry(runs, proc, out):
    sel = {m: runs.get(f'{proc}_{m}_s0') for m in MODELS}
    sel = {m: a for m, a in sel.items() if a is not None}
    if not sel:
        return
    S = next(iter(sel.values()))['cfg']['S']
    fig, axes = plt.subplots(2, 1 + len(sel), figsize=(2.6 * (1 + len(sel)), 5.0), squeeze=False)
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    basis = None
    for j, (m, a) in enumerate(sel.items()):
        cks = sorted(int(c) for c in a['metrics'])
        for i, ck in enumerate((cks[-1], cks[0])):
            true, pred = stacked(a, ck)
            if basis is None:  # one coordinate system and one coloring for the whole figure, from the truth
                uniq, uid = np.unique(true.round(5), axis=0, return_inverse=True)
                few = len(uniq) <= 200
                if S == 3:
                    proj, col_of = (lambda z: np.stack(PL.tri_xy(z), 1)), (lambda tr, u: np.clip(tr, 0, 1))
                else:
                    ref = uniq if few else true
                    mu = ref.mean(0)
                    Vh = np.linalg.svd(ref - mu, full_matrices=False)[2][:3]
                    proj, col_of = (lambda z: (z - mu) @ Vh[:2].T), (lambda tr, u: PL.rgb((tr - mu) @ Vh.T, u))
                basis = (proj, col_of, few)
            proj, col_of, few = basis
            uid = np.unique(true.round(5), axis=0, return_inverse=True)[1].ravel()
            c = col_of(true, uid)
            if j == 0:
                ax = axes[i, 0]
                z = proj(true)
                ax.scatter(z[:, 0], z[:, 1], c=c, s=20 if few else 0.3, alpha=1 if few else 0.5, edgecolors='none', rasterized=not few)
                if i == 0:
                    ax.set_title('ground truth', fontsize=8)
                lims = (z.min(0), z.max(0))
            ax = axes[i, j + 1]
            z = proj(pred)
            ax.scatter(z[:, 0], z[:, 1], c=c, s=0.3 if not few else 1.5, alpha=0.5 if not few else 0.15, edgecolors='none', rasterized=True)
            if few:
                ids = np.unique(uid)
                ax.scatter(*np.stack([z[uid == k].mean(0) for k in ids]).T, c=np.stack([c[uid == k][0] for k in ids]),
                           s=16, edgecolors='k', linewidths=0.2)
            ax.text(0.02, 0.02, f'R²={r2(true, pred):.3f}', transform=ax.transAxes, fontsize=6, color=INK2)
            if i == 0:
                ax.set_title(f"{m.replace('tree8', 'tree, r = 8')}\n{DRAWN[a['cfg']['model']]}", fontsize=6.5)
        axes[0, 0].set_ylabel('trained', fontsize=8)
        axes[1, 0].set_ylabel('initial network', fontsize=8)
    for ax in axes.flat:
        if S == 3:
            ax.plot(outline[:, 0], outline[:, 1], color='0.6', lw=0.6, zorder=0)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
        else:
            pad = 0.25 * (lims[1] - lims[0]).max()
            ax.set_xlim(lims[0][0] - pad, lims[1][0] + pad)
            ax.set_ylim(lims[0][1] - pad, lims[1][1] + pad)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f'{proc}: beliefs decoded from hidden states (sequence split, test rows, all positions overlaid; R² over the rows shown)',
                 fontsize=8)
    fig.tight_layout()
    save(fig, out)


LINES = dict(
    transformer=[('L4@prefix', 'fin', dict(marker='o', ls='-', ms=2.5, color=BLUE), 'affine, primary set'),
                 ('L4^proj@prefix', 'fin', dict(marker='D', ls='-', ms=2, lw=0.9, color=VIOLET), 'projective, primary set'),
                 ('concat@prefix', 'fin', dict(marker='o', ls=':', ms=1.5, lw=0.7, color=BLUE, alpha=0.6), 'affine, all layers / all bonds'),
                 ('L4@prefix', 'ini', dict(marker='o', ls='-', ms=2, color='0.6'), 'initial network, affine'),
                 ('L4~scr@prefix', 'fin', dict(marker='x', ls=':', ms=2.5, lw=0.7, color=PINK), 'scrambled target')],
    chain=[('state@prefix', 'fin', dict(marker='o', ls='-', ms=2.5, color=BLUE), 'affine, primary set'),
           ('state^proj@prefix', 'fin', dict(marker='D', ls='-', ms=2, lw=0.9, color=VIOLET), 'projective, primary set'),
           ('state^proj@prefix', 'ini', dict(marker='D', ls='-', ms=2, color='0.6'), 'initial network, projective'),
           ('state~scr@prefix', 'fin', dict(marker='x', ls=':', ms=2.5, lw=0.7, color=PINK), 'scrambled target')],
    tree=[('top@prefix', 'fin', dict(marker='o', ls='-', ms=2.5, color=BLUE), 'affine, primary set'),
          ('top^proj@prefix', 'fin', dict(marker='D', ls='-', ms=2, lw=0.9, color=VIOLET), 'projective, primary set'),
          ('top_bil^proj@prefix', 'fin', dict(marker='*', ls='-', ms=3.5, lw=0.9, color=CATEGORICAL[7]), 'projective, top bonds with product'),
          ('all@prefix', 'fin', dict(marker='o', ls=':', ms=1.5, lw=0.7, color=BLUE, alpha=0.6), 'affine, all layers / all bonds'),
          ('top_bil^proj@prefix', 'ini', dict(marker='*', ls='-', ms=3, color='0.6'), 'initial network, projective with product'),
          ('addceil@prefix', 'fin', dict(marker='_', ls='--', ms=4, lw=0.8, color='k'), 'additive ceiling (no network)'),
          ('top_bil~scr@prefix', 'fin', dict(marker='x', ls=':', ms=2.5, lw=0.7, color=PINK), 'scrambled target')])
COMMON = [('nexttok@prefix', 'fin', dict(marker='s', ls='--', ms=2, lw=0.8, color=ORANGE), 'next-token ceiling'),
          ('tokens@prefix', 'fin', dict(marker='^', ls=':', ms=2, lw=0.8, color=GREEN), 'linear in tokens')]


def positions(runs, out):
    procs = [p for p in PROCESSES if any(f'{p}_{m}_s0' in runs for m in MODELS)]
    fig, axes = plt.subplots(len(procs), len(MODELS), figsize=(2.9 * len(MODELS), 1.9 * len(procs)), sharey=True, squeeze=False)
    handles = {}
    for i, proc in enumerate(procs):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            a = runs.get(f'{proc}_{m}_s0')
            if a is None:
                ax.axis('off')
                continue
            cks = sorted(int(c) for c in a['metrics'])
            src = dict(fin=a['metrics'][str(cks[-1])]['per_pos'], ini=a['metrics'][str(cks[0])]['per_pos'])
            for t in a['positions']:
                if t not in a['eligible']['e']:
                    ax.axvspan(t - 0.5, t + 0.5, color='0.92', lw=0, zorder=0)
            if a['cfg']['model'] == 'tree':
                ax.axvline(a['half'] + 0.5, color='0.4', ls='--', lw=0.6)
            for key, which, style, lab in LINES[a['cfg']['model']] + COMMON:
                d = src[which].get(key, {})
                ts = [t for t in a['positions'] if str(t) in d]
                if ts:
                    handles[lab] = ax.plot(ts, [max(d[str(t)]['R2'], -0.6) for t in ts], **style)[0]
            ax.set_title(f'{proc}, {m}', fontsize=7)
            ax.set_ylim(-0.5, 1.02)
            ax.set_xlim(0.5, a['cfg']['T'] - 0.5)
            if i == len(procs) - 1:
                ax.set_xlabel('prefix length t')
        axes[i, 0].set_ylabel('R², unseen prefixes')
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out)


def summary(runs, out):
    """x slots per process: transformer (e), chain (e), tree first half (h1), tree second half (h2b)."""
    slots = [('transformer', 'e', 'L4', 'L4^proj'), ('chain', 'e', 'state', 'state^proj'), ('tree', 'h1', 'top', 'top^proj'),
             ('tree', 'h2b', 'top', 'top_bil^proj'), ('tree8', 'h1', 'top', 'top^proj'), ('tree8', 'h2b', 'top', 'top_bil^proj')]
    procs = [p for p in PROCESSES if any(r.startswith(p + '_') for r in runs)]
    fig, axes = plt.subplots(1, len(procs), figsize=(2.6 * len(procs), 2.8), sharey=True, squeeze=False)
    for ax, proc in zip(axes[0], procs):
        for xi, (m, grp, aff, prj) in enumerate(slots):
            rr = sorted(r for r in runs if r.startswith(f'{proc}_{m}_s'))
            for k, r in enumerate(rr):
                a = runs[r]
                cks = sorted(int(c) for c in a['metrics'])
                P, P0 = a['metrics'][str(cks[-1])]['pooled'], a['metrics'][str(cks[0])]['pooled']
                g = lambda pooled, key: pooled.get(key + '@prefix', {}).get('R2_' + grp)
                pts = [(g(P, aff), 'o', BLUE, 'trained, affine'), (g(P, prj), 'D', VIOLET, 'trained, projective'),
                       (g(P0, prj), 'D', '0.6', 'initial network, projective'), (g(P, 'tokens'), '^', GREEN, 'linear in tokens'),
                       (g(P, 'nexttok'), 's', ORANGE, 'next-token ceiling'),
                       (g(P, (prj if grp == 'h2b' else aff).split('^')[0] + '~scr'), 'x', PINK, 'scrambled target')]
                if grp == 'h2b':
                    pts.append((g(P, 'addceil'), '_', 'k', 'additive ceiling'))
                for val, mk, col, lab in pts:
                    if val is not None:
                        ax.plot([xi + 0.14 * (k - 1)], [max(val, -0.5)], mk, color=col, ms=3.2, label=lab)
        ax.set_xticks(range(6), ['transf.', 'chain', 'tree\n1st', 'tree\n2nd', 'tree8\n1st', 'tree8\n2nd'], fontsize=5)
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.52, 1.02)
        ax.set_title(proc, fontsize=8)
    axes[0, 0].set_ylabel('pooled R² on unseen prefixes\n(eligible positions; clipped at −0.5)')
    hl = {}
    for ax in axes[0]:
        hl.update(dict(zip(*ax.get_legend_handles_labels()[::-1])))
    fig.tight_layout()
    fig.legend(hl.values(), hl.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out)


def training(runs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.9))
    procs = [p for p in PROCESSES if any(r.startswith(p + '_') for r in runs)]
    ls = dict(transformer='-', chain='--', tree=':', tree8=':')
    for r, a in runs.items():
        col = CATEGORICAL[procs.index(a['cfg']['process']) % len(CATEGORICAL)]
        m = a['cfg'].get('arm', a['cfg']['model'])
        h = a['hist']
        a1.plot(steps_x([x['step'] for x in h]), [max(x['excess'], 1e-5) for x in h], color=col, ls=ls[m], lw=0.8,
                label=a['cfg']['process'] if r.endswith('_transformer_s0') else None)
        cks = sorted(int(c) for c in a['metrics'])
        grp = 'h1' if a['cfg']['model'] == 'tree' else 'e'
        ys = [a['metrics'][str(c)]['pooled'].get(a['primary'] + '@prefix', {}).get('R2_' + grp) for c in cks]
        if all(y is not None for y in ys):
            a2.plot(steps_x(cks), [max(y, -1) for y in ys], 'o', ls=ls[m], ms=2, lw=0.8, color=col)
    for ax in (a1, a2):
        ax.set_xscale('log')
        ax.set_xlabel('update')
    a1.set_yscale('log')
    a1.set_ylabel('excess test loss over Bayes (nats/token)')
    a2.set_ylim(-1, 1.02)
    a2.set_ylabel('pooled R², unseen prefixes\n(affine, primary set)')
    a1.legend(fontsize=6, title='solid transformer, dashed chain, dotted tree', title_fontsize=5.5)
    fig.tight_layout()
    save(fig, out)


def main():
    runs = PL.load(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    for proc in PROCESSES:
        if any(f'{proc}_{m}_s0' in runs for m in MODELS):
            S = next(a for r, a in runs.items() if r.startswith(proc + '_'))['cfg']['S']
            geometry(runs, proc, fig_dir / f"{'triangle' if S == 3 else 'pca'}_{proc}.pdf")
    positions(runs, fig_dir / 'positions.pdf')
    summary(runs, fig_dir / 'summary.pdf')
    training(runs, fig_dir / 'training.pdf')


if __name__ == '__main__':
    main()
