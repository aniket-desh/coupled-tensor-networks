"""Figures from saved TTN analysis results only.

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/ttn_plot.py RESULTS_DIR FIGURES_DIR [TRANSFORMER_RESULTS_DIR]

RESULTS_DIR holds one folder per run (<config>_r<r>_s<seed>/ with analysis.json and proj_ckpt*.npz).

Per (grammar, width), seed 0:
  geometry_<cfg>_r<r>.pdf  rows = two first-half and two second-half positions (preferring a next-token ceiling < 0.9,
      then the most distinct beliefs); columns = ground-truth beliefs on their top two PCs; the affine probe on the
      top bonds [h_L, h_R]; the bilinear probe [h_L, h_R, h_L (x) h_R]; the affine probe on all bonds; the
      shuffled-label control; the bilinear probe on the identity-init network; the raw top-2 PCs of the top bonds
      (gauge dependent). Sequence split, test sequences; colors as in rhm-belief-geometry/plot.py.
  triangles_<cfg>_r<r>.pdf (v = 3) one pending node's marginal on the 2-simplex: truth, affine, bilinear, init.
Across runs:
  positions.pdf  per-position R2 on unseen prefixes: affine top, bilinear top, affine all bonds, the additive ceiling
      (the bound on any affine probe for t > T/2), the bilinear probe on the scrambled target, init, the token readouts
      and the next-token ceiling; dashed line = T/2 (after it the right top bond is no longer constant); gray = not
      eligible for the affine top probe.
  beliefsplit.pdf  per-position R2 on distinct beliefs the probe never saw (the test of an affine code of the belief),
      with its span ceiling and the identity-init tree; gray = not in the belief split's pooled set.
  summary.pdf    pooled R2 for the two criteria, one point per seed: first half affine (H1), and second half affine
      against bilinear on the same positions (H2), each with its init, token-readout and scrambled-target controls,
      and the additive ceiling for the second half.
  training.pdf   pooled R2 of the affine top probe over updates (first and second half), and excess loss.
  compare.pdf    (with TRANSFORMER_RESULTS_DIR) per-position R2 on the same held-out prefixes: the transformer's
      last-layer and all-layer probes against the TTN's affine and bilinear top-bond probes."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
import plot as PL  # noqa: E402
from plotting import CATEGORICAL, INK2, plt, save, steps_x  # noqa: E402

BLUE, ORANGE, GREEN, PINK, VIOLET = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[4], CATEGORICAL[6]


def cell(run):
    return run.rsplit('_s', 1)[0]


def label(a):
    return PL.label(a) + f" r={a['cfg']['r']}"


def pick_rows(a, ceil):
    H = a['half']
    cand = [t for t in a['positions'] if a['geom'][str(t)]['n_unique'] > 1]
    key = lambda t: (ceil[str(t)]['R2'] >= 0.9, -a['geom'][str(t)]['n_unique'])
    return sorted(sorted([t for t in cand if t <= H], key=key)[:2] + sorted([t for t in cand if t > H], key=key)[:2])


def geometry(a, out):
    cks = sorted(int(k) for k in a['metrics'])
    first, last = (np.load(a['dir'] / f'proj_ckpt{c}.npz') for c in (cks[0], cks[-1]))
    m = a['metrics']
    ceil = m[str(cks[-1])]['per_pos']['nexttok']
    pos = pick_rows(a, ceil)
    cols = [('true', 'ground truth', last, None), ('pred_top', 'affine probe: top bonds', last, 'top'),
            ('pred_top_bil', 'bilinear probe: top bonds', last, 'top_bil'), ('pred_all', 'affine probe: all bonds', last, 'all'),
            ('pred_shuffle', 'shuffled labels', last, 'shuffle'), ('pred_top_bil', 'identity init, bilinear', first, 'top_bil'),
            ('raw_pca', 'raw PCA, top bonds', last, None)]
    fig, axes = plt.subplots(len(pos), len(cols), figsize=(2.0 * len(cols), 2.0 * len(pos)), squeeze=False)
    for i, t in enumerate(pos):
        uid, true = last[f't{t}_uid'], last[f't{t}_true']
        c = PL.rgb(true, uid)
        for j, (key, title, src, mk) in enumerate(cols):
            ax = axes[i, j]
            z = src[f't{t}_{key}']
            if key == 'true':
                u = np.unique(uid, return_index=True)[1]
                ax.scatter(z[u, 0], z[u, 1], c=c[u], s=30, edgecolors='none')
            else:
                ax.scatter(z[:, 0], z[:, 1], c=c, s=1.5, alpha=0.15, edgecolors='none', rasterized=True)
                if key != 'raw_pca':
                    ids = np.unique(uid)
                    ax.scatter(*np.stack([z[uid == k].mean(0) for k in ids])[:, :2].T,
                               c=np.stack([c[uid == k][0] for k in ids]), s=18, edgecolors='k', linewidths=0.2)
            if key != 'raw_pca':
                lim = np.abs(true[:, :2]).max() * 1.25 + 1e-9
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')
            if i == 0:
                ax.set_title(title, fontsize=8)
            if mk:
                r2 = m[str(cks[0] if src is first else cks[-1])]['per_pos'][mk][str(t)]['R2']
                ax.text(0.02, 0.02, f'R²={r2:.2f}', transform=ax.transAxes, fontsize=6, color=INK2)
            if j == 0:
                gm = a['geom'][str(t)]
                ax.set_ylabel(f"t={t} ({'first' if t <= a['half'] else 'second'} half)\n{gm['n_unique']} beliefs, D={gm['D']}\n"
                              f"next-token ceiling R²={ceil[str(t)]['R2']:.2f}", fontsize=7)
    fig.suptitle(f"{label(a)}: TTN bond states vs belief states (top PCs of the beliefs; sequence split, test sequences)"
                 + ('' if a['eligible']['e_top'] else '\nfew distinct prefixes: any injective code of the prefix also fits'),
                 fontsize=9)
    fig.tight_layout()
    save(fig, out)


def triangles(a, out):
    cks = sorted(int(k) for k in a['metrics'])
    p, p0 = (np.load(a['dir'] / f'proj_ckpt{c}.npz') for c in (cks[-1], cks[0]))
    keys = sorted({k.rsplit('_', 1)[0] for k in p.files if k.endswith('_true') and '_tri' in k},
                  key=lambda k: (int(k[1:].split('_')[0]), k))
    keys = [k for k in keys if len(np.unique(p[k + '_true'].round(6), axis=0)) > 2]
    if not keys:
        return
    H = a['half']
    tt = lambda k: int(k[1:].split('_')[0])
    keys = ([k for k in keys if tt(k) <= H][-5:] + [k for k in keys if tt(k) > H][:5])
    rows = [(p, '_true', 25, 1.0, 'ground truth'), (p, '_top', 1.5, 0.2, 'affine, top bonds'),
            (p, '_top_bil', 1.5, 0.2, 'bilinear, top bonds'), (p0, '_top_bil', 1.5, 0.2, 'identity init, bilinear')]
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(1.8 * len(keys), 1.8 * len(rows)), squeeze=False)
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    for j, k in enumerate(keys):
        c = np.clip(p[k + '_true'], 0, 1)
        for i, (src, suf, s, al, name) in enumerate(rows):
            ax = axes[i, j]
            x, y = PL.tri_xy(src[k + suf])
            ax.plot(outline[:, 0], outline[:, 1], color='0.6', lw=0.6)
            ax.scatter(x, y, c=c, s=s, alpha=al, edgecolors='none', rasterized=i > 0)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
            ax.set_aspect('equal')
            ax.axis('off')
            if j == 0:
                ax.text(-0.15, 0.43, name, rotation=90, fontsize=7, va='center', transform=ax.transData)
        t, node, lev = k.split('_')
        axes[0, j].set_title(f"t={t[1:]} ({'1st' if tt(k) <= H else '2nd'} half)\nnode {node[3:]}, level {lev[1:]}", fontsize=7)
    fig.suptitle(f'{label(a)}: marginal belief over one pending node (3 symbols), probes on the top bonds, sequence split'
                 + ('' if a['eligible']['e_top'] else '\n(few distinct prefixes: any injective code of the prefix also fits)'),
                 fontsize=8)
    fig.tight_layout()
    save(fig, out)


PREFIX_LINES = [('top@prefix', 'fin', dict(marker='o', ls='-', ms=2.5, color=BLUE), 'affine, top bonds'),
                ('top_bil@prefix', 'fin', dict(marker='D', ls='-', ms=2, lw=0.9, color=VIOLET), 'bilinear, top bonds'),
                ('all@prefix', 'fin', dict(marker='o', ls=':', ms=1.5, lw=0.7, color=BLUE, alpha=0.6), 'affine, all bonds'),
                ('addceil@prefix', 'fin', dict(marker='_', ls='--', ms=4, lw=0.8, color='k'), 'additive ceiling (no network)'),
                ('top_bil~scr@prefix', 'fin', dict(marker='x', ls=':', ms=2.5, lw=0.7, color=PINK), 'bilinear, scrambled target'),
                ('top_bil@prefix', 'ini', dict(marker='o', ls='-', ms=2, color='0.6'), 'identity init, bilinear'),
                ('nexttok@prefix', 'fin', dict(marker='s', ls='--', ms=2, lw=0.8, color=ORANGE), 'next-token ceiling'),
                ('tokens@prefix', 'fin', dict(marker='^', ls=':', ms=2, lw=0.8, color=GREEN), 'linear in tokens'),
                ('tokprod@prefix', 'fin', dict(marker='v', ls=':', ms=2, lw=0.8, color=GREEN, alpha=0.6), 'tokens with cross-half products')]
BELIEF_LINES = [('top@belief', 'fin', dict(marker='o', ls='-', ms=2.5, color=BLUE), 'affine, top bonds'),
                ('top_bil@belief', 'fin', dict(marker='D', ls='-', ms=2, lw=0.9, color=VIOLET), 'bilinear, top bonds'),
                ('all@belief', 'fin', dict(marker='o', ls=':', ms=1.5, lw=0.7, color=BLUE, alpha=0.6), 'affine, all bonds'),
                ('span@belief', 'fin', dict(marker='_', ls='--', ms=4, lw=0.8, color='k'), 'ceiling: span of the fit beliefs'),
                ('top_bil@belief', 'ini', dict(marker='o', ls='-', ms=2, color='0.6'), 'identity init, bilinear')]


def positions(runs, out, lines=PREFIX_LINES, ylabel='R² on unseen prefixes', elig='e_top'):
    seed0 = {cell(r): a for r, a in runs.items() if r.endswith('_s0')}
    rs = sorted({a['cfg']['r'] for a in seed0.values()})
    gs = sorted({n.rsplit('_r', 1)[0] for n in seed0})
    fig, axes = plt.subplots(len(rs), len(gs), figsize=(2.3 * len(gs), 2.2 * len(rs)), sharey=True, squeeze=False)
    handles = {}
    for i, r in enumerate(rs):
        for j, gname in enumerate(gs):
            ax = axes[i, j]
            a = seed0.get(f'{gname}_r{r}')
            if a is None:
                ax.axis('off')
                continue
            cks = sorted(int(c) for c in a['metrics'])
            src = dict(fin=a['metrics'][str(cks[-1])]['per_pos'], ini=a['metrics'][str(cks[0])]['per_pos'])
            for t in a['positions']:
                if t not in a['eligible'][elig]:
                    ax.axvspan(t - 0.5, t + 0.5, color='0.92', lw=0, zorder=0)
            ax.axvline(a['half'] + 0.5, color='0.4', ls='--', lw=0.6)
            for key, which, style, lab in lines:
                d = src[which].get(key, {})
                ts = [t for t in a['positions'] if str(t) in d]
                if ts:
                    handles[lab] = ax.plot(ts, [max(d[str(t)]['R2'], -0.6) for t in ts], **style)[0]
            ax.set_title(label(a).replace(' s=2', ''), fontsize=7)
            ax.set_ylim(-0.5, 1.02)
            ax.set_xlim(0.5, a['cfg']['T'] - 0.5)
            if i == len(rs) - 1:
                ax.set_xlabel('prefix length t')
        axes[i, 0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5)
    save(fig, out)


def summary(runs, out):
    by = defaultdict(list)
    for r, a in runs.items():
        by[cell(r)].append(a)
    names = sorted(by, key=lambda n: (by[n][0]['cfg']['r'], n))
    fig, axes = plt.subplots(1, len(names), figsize=(1.7 * len(names), 2.7), sharey=True, squeeze=False)
    cats = [('top@prefix', 'R2_h1', 'top@prefix', 'tokens@prefix'), ('top@prefix', 'R2_h2b', 'top@prefix', 'tokens@prefix'),
            ('top_bil@prefix', 'R2_h2b', 'top_bil@prefix', 'tokens@prefix')]
    for j, n in enumerate(names):
        ax = axes[0, j]
        for k, a in enumerate(by[n]):
            cks = sorted(int(c) for c in a['metrics'])
            P, P0 = a['metrics'][str(cks[-1])]['pooled'], a['metrics'][str(cks[0])]['pooled']
            for x, (key, f, key0, tok) in enumerate(cats):
                g = lambda pooled, kk: pooled.get(kk, {}).get(f)
                extra = [(g(P, key.replace('@', '~scr@')), 'x', PINK, 'scrambled target')]
                if x > 0:
                    extra.append((g(P, 'addceil@prefix'), '_', 'k', 'additive ceiling'))
                for val, mk, col, lab in [(g(P, key), 'o', BLUE, 'trained'), (g(P0, key0), 'o', '0.6', 'identity init'),
                                          (g(P, tok), '^', GREEN, 'linear in tokens')] + extra:
                    if val is not None:
                        ax.plot([x + 0.12 * (k - 1)], [max(val, -0.5)], mk, color=col, ms=3.5,
                                label=lab if k == 0 and x == (1 if lab == 'additive ceiling' else 0) else None)
        ax.set_xticks(range(3), ['1st half\naffine', '2nd half\naffine', '2nd half\nbilinear'], fontsize=5.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.52, 1.02)
        ax.set_title(label(by[n][0]).replace(' s=2', '').replace(' r=', '\nr='), fontsize=6.5)
    axes[0, 0].set_ylabel('pooled R² on unseen prefixes\n(eligible positions; clipped at −0.5)')
    hl = {}
    for ax in axes[0]:
        hl.update(dict(zip(*ax.get_legend_handles_labels()[::-1])))
    fig.tight_layout()
    fig.legend(hl.values(), hl.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5)
    save(fig, out)


def training(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    names = sorted({cell(r).rsplit('_r', 1)[0] for r in runs})
    for r, a in runs.items():
        col = CATEGORICAL[names.index(cell(r).rsplit('_r', 1)[0]) % len(CATEGORICAL)]
        ls = '-' if a['cfg']['r'] == max(x['cfg']['r'] for x in runs.values()) else ':'
        lab = PL.label(a) if r.endswith('_s0') and ls == '-' else None
        cks = sorted(int(c) for c in a['metrics'])
        for ax, f in zip(axes[:2], ('R2_h1', 'R2_h2')):
            ys = [a['metrics'][str(c)]['pooled'].get('top@prefix', {}).get(f) for c in cks]
            if all(y is not None for y in ys):
                ax.plot(steps_x(cks), [max(y, -1) for y in ys], 'o', ls=ls, ms=2, lw=0.9, color=col, label=lab)
        h = a['hist']
        axes[2].plot(steps_x([x['step'] for x in h]), [max(x['excess'], 1e-5) for x in h], color=col, lw=0.8, ls=ls)
    for ax, t in zip(axes[:2], ('first half (t ≤ T/2)', 'second half (t > T/2)')):
        ax.set_xscale('log')
        ax.set_ylim(-1, 1.02)
        ax.set_xlabel('update')
        ax.set_title(f'affine probe on the top bonds, {t}', fontsize=8)
    axes[0].set_ylabel('pooled R², unseen prefixes')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('update')
    axes[2].set_ylabel('excess test loss over Bayes (nats/token)')
    axes[2].set_title('solid: larger r, dotted: smaller r', fontsize=8)
    axes[0].legend(fontsize=5.5, loc='lower right')
    fig.tight_layout()
    save(fig, out)


def compare(runs, tf_dir, out):
    """Same sequences and the same held-out prefixes in both analyses (shared seeds)."""
    tf = PL.load(tf_dir)
    seed0 = {cell(r): a for r, a in runs.items() if r.endswith('_s0')}
    rmax = max(a['cfg']['r'] for a in seed0.values())
    pairs = [(n.rsplit('_r', 1)[0], a) for n, a in sorted(seed0.items()) if a['cfg']['r'] == rmax]
    pairs = [(gname, a, tf[gname + '_s0']) for gname, a in pairs if gname + '_s0' in tf]
    if not pairs:
        print('compare: no matching transformer runs in', tf_dir)
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.3 * len(pairs), 2.4), sharey=True, squeeze=False)
    for ax, (gname, a, b) in zip(axes[0], pairs):
        fa = a['metrics'][str(max(int(c) for c in a['metrics']))]['per_pos']
        fb = b['metrics'][str(max(int(c) for c in b['metrics']))]['per_pos']
        ts = [t for t in a['positions'] if str(t) in fa['top@prefix'] and str(t) in fb.get('L4@prefix', {})]
        for t in a['positions']:
            if t not in a['eligible']['e_top'] or t not in b['eligible']['eL4']:
                ax.axvspan(t - 0.5, t + 0.5, color='0.92', lw=0, zorder=0)
        ax.axvline(a['half'] + 0.5, color='0.4', ls='--', lw=0.6)
        ax.plot(ts, [fb['L4@prefix'][str(t)]['R2'] for t in ts], 's-', ms=2.5, color=ORANGE, label='transformer, last layer (affine)')
        ax.plot(ts, [fb['concat@prefix'][str(t)]['R2'] for t in ts], 's:', ms=1.5, lw=0.7, color=ORANGE, alpha=0.7,
                label='transformer, all layers (affine)')
        ax.plot(ts, [fa['top@prefix'][str(t)]['R2'] for t in ts], 'o-', ms=2.5, color=BLUE, label=f'TTN r={rmax}, top bonds (affine)')
        ax.plot(ts, [fa['top_bil@prefix'][str(t)]['R2'] for t in ts], 'D-', ms=2, lw=0.9, color=VIOLET,
                label=f'TTN r={rmax}, top bonds (bilinear)')
        ax.set_title(PL.label(a).replace(' s=2', ''), fontsize=7)
        ax.set_xlabel('prefix length t')
        ax.set_ylim(-0.5, 1.02)
    axes[0, 0].set_ylabel('R² on unseen prefixes')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out)


def main():
    runs = PL.load(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    for r, a in runs.items():
        if r.endswith('_s0'):
            geometry(a, fig_dir / f'geometry_{cell(r)}.pdf')
            triangles(a, fig_dir / f'triangles_{cell(r)}.pdf')
    positions(runs, fig_dir / 'positions.pdf')
    positions(runs, fig_dir / 'beliefsplit.pdf', BELIEF_LINES, 'R² on unseen beliefs', 'bel')
    summary(runs, fig_dir / 'summary.pdf')
    training(runs, fig_dir / 'training.pdf')
    if len(sys.argv) > 3 and Path(sys.argv[3]).is_dir():
        compare(runs, sys.argv[3], fig_dir / 'compare.pdf')


if __name__ == '__main__':
    main()
