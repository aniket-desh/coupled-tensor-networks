"""Figures from saved analysis results only.

    PYTHONPATH=. python studies/rhm/rhm-belief-geometry/plot.py RESULTS_DIR FIGURES_DIR

RESULTS_DIR holds one folder per run (<config>_s<seed>/ with analysis.json, proj_ckpt*.npz, dists.npz).

Per config (seed 0):
  geometry_<cfg>.pdf  rows = the positions with the most distinct belief states; columns = ground-truth beliefs on the
      top two PCs of the beliefs (probability-weighted); the concat probe (small points = test rows, large = mean per true belief,
      Shai et al. 2024 Fig. 7); the last-layer probe; the shuffled-label control; the same probe on the randomly
      initialized network; the raw top-2 PCs of the last-layer residual stream (no probe). Color = the true belief's
      coordinates on PCs 1-3 (rank-normalized) mapped to RGB, identical in every panel of a row, so a panel reproduces the geometry when
      neighbouring colours sit together.
  triangles_<cfg>.pdf (grammars with 3-symbol nodes) the marginal belief over one pending node on the 2-simplex,
      truth vs concat probe, colored by the true marginal as RGB (the Mess3 convention).
  distances_<cfg>.pdf pairwise distances of probe outputs against true-belief and next-token distances (Fig. 7D).
Across configs:
  summary.pdf   pooled R2 on unseen prefixes by activation set (dotted: sequence split), with random init, the token
      readout, the next-token ceiling and the shuffled control; one point per seed.
  positions.pdf per-position R2 (concat) on unseen prefixes against random init, the token readout and the
      next-token ceiling, with the sequence-split values dotted; gray = positions not eligible for H1 (too few
      distinct fit prefixes for the 129-feature probe).
  training.pdf  pooled R2 (concat) against update, and the excess loss over the Bayes loss against update."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plotting import CATEGORICAL, INK2, plt, save, steps_x  # noqa: E402

SETS = ['embed', 'L1', 'L2', 'L3', 'L4', 'lnf', 'concat']


def rgb(c, uid):
    """Each of PCs 1-3 rank-normalized over the distinct beliefs to [0.1, 0.9], so colors spread evenly."""
    u = np.unique(uid, return_index=True)[1]
    ranks = np.argsort(np.argsort(c[u], 0), 0) / max(1, len(u) - 1)
    lut = np.zeros((uid.max() + 1, 3))
    lut[uid[u]] = 0.1 + 0.8 * ranks
    return lut[uid]


def tri_xy(b):
    return b[:, 1] + 0.5 * b[:, 2], np.sqrt(3) / 2 * b[:, 2]


def load(res_dir):
    runs = {}
    for d in sorted(Path(res_dir).iterdir()):
        if (d / 'analysis.json').exists():
            a = json.load(open(d / 'analysis.json'))
            a['dir'] = d
            runs[d.name] = a
    return runs


def cfg_name(run):
    return run.rsplit('_s', 1)[0]


def label(a):
    g = a['cfg']['grammar']
    return f"v={g['v']} m={g['m']} s={g['s']} L={g['L']}" + (f" n={g['n']}" if g['n'] != g['v'] else '')


def geometry(a, out):
    cks = sorted(int(k) for k in a['metrics'])
    first, last = (np.load(a['dir'] / f'proj_ckpt{c}.npz') for c in (cks[0], cks[-1]))
    m = a['metrics']
    ceil = m[str(cks[-1])]['per_pos']['nexttok']
    # prefer positions where the belief is not a function of the next-token distribution, then the richest geometry
    cand = [t for t in a['positions'] if a['geom'][str(t)]['n_unique'] > 1]
    pos = sorted(sorted(cand, key=lambda t: (ceil[str(t)]['R2'] >= 0.9, -a['geom'][str(t)]['n_unique']))[:4])
    cols = [('true', 'ground truth', last), ('pred_concat', 'probe: all layers', last), ('pred_L4', 'probe: last layer', last),
            ('pred_shuffle', 'shuffled labels', last), ('pred_concat', 'random-init network', first),
            ('raw_pca', 'raw PCA, last layer', last)]
    fig, axes = plt.subplots(len(pos), len(cols), figsize=(2.0 * len(cols), 2.0 * len(pos)), squeeze=False)
    for i, t in enumerate(pos):
        uid, true = last[f't{t}_uid'], last[f't{t}_true']
        c = rgb(true, uid)
        for j, (key, title, src) in enumerate(cols):
            ax = axes[i, j]
            z = src[f't{t}_{key}']
            if key == 'true':
                u = np.unique(uid, return_index=True)[1]
                ax.scatter(z[u, 0], z[u, 1], c=c[u], s=30, edgecolors='none')
            else:
                ax.scatter(z[:, 0], z[:, 1], c=c, s=1.5, alpha=0.15, edgecolors='none', rasterized=True)
                if key != 'raw_pca':
                    ids = np.unique(uid)
                    cm = np.stack([z[uid == k].mean(0) for k in ids])
                    cc = np.stack([c[uid == k][0] for k in ids])
                    ax.scatter(cm[:, 0], cm[:, 1], c=cc, s=18, edgecolors='k', linewidths=0.2)
            if key != 'raw_pca':
                lim = np.abs(true[:, :2]).max() * 1.25 + 1e-9
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', adjustable='box')
            r2 = None
            if key == 'pred_concat':
                r2 = m[str(cks[0] if src is first else cks[-1])]['per_pos']['concat'][str(t)]['R2']
            elif key == 'pred_L4':
                r2 = m[str(cks[-1])]['per_pos']['L4'][str(t)]['R2']
            elif key == 'pred_shuffle':
                r2 = m[str(cks[-1])]['per_pos']['shuffle'][str(t)]['R2']
            if i == 0:
                ax.set_title(title, fontsize=8)
            if r2 is not None:
                ax.text(0.02, 0.02, f'R²={r2:.2f}', transform=ax.transAxes, fontsize=6, color=INK2)
            if j == 0:
                gm = a['geom'][str(t)]
                ax.set_ylabel(f"t={t}\n{gm['n_unique']} beliefs, D={gm['D']}\nnext-token ceiling R²={ceil[str(t)]['R2']:.2f}",
                              fontsize=7)
    fig.suptitle(f"{label(a)}: residual stream vs belief states (top PCs of the beliefs; sequence split, test sequences)"
                 + ('' if a['eligible']['eL4'] else '\nfew distinct prefixes: a random network also fits by memorizing'),
                 fontsize=9)
    fig.tight_layout()
    save(fig, out)


def triangles(a, out):
    cks = sorted(int(k) for k in a['metrics'])
    p, p0 = (np.load(a['dir'] / f'proj_ckpt{c}.npz') for c in (cks[-1], cks[0]))
    keys = sorted({k.rsplit('_', 1)[0] for k in p.files if '_tri' in k}, key=lambda k: (int(k[1:].split('_')[0]), k))
    keys = [k for k in keys if len(np.unique(p[k + '_true'].round(6), axis=0)) > 2]
    if not keys:
        return
    keys = keys[:10]
    fig, axes = plt.subplots(3, len(keys), figsize=(1.8 * len(keys), 5.4), squeeze=False)
    outline = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    for j, k in enumerate(keys):
        tb = p[k + '_true']
        c = np.clip(tb, 0, 1)
        for i, (src, suf, s, al) in enumerate(((p, '_true', 25, 1.0), (p, '_concat', 1.5, 0.2), (p0, '_concat', 1.5, 0.2))):
            ax = axes[i, j]
            x, y = tri_xy(src[k + suf])
            ax.plot(outline[:, 0], outline[:, 1], color='0.6', lw=0.6)
            ax.scatter(x, y, c=c, s=s, alpha=al, edgecolors='none', rasterized=i == 1)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
            ax.set_aspect('equal')
            ax.axis('off')
        t, node, lev = k.split('_')
        axes[0, j].set_title(f"{t.replace('t', 't=')}, node {node[3:]} (level {lev[1:]})", fontsize=7)
    axes[0, 0].text(-0.15, 0.4, 'ground truth', rotation=90, fontsize=8, transform=axes[0, 0].transData)
    axes[1, 0].text(-0.15, 0.4, 'probe, trained', rotation=90, fontsize=8, transform=axes[1, 0].transData)
    axes[2, 0].text(-0.15, 0.3, 'probe, random init', rotation=90, fontsize=8, transform=axes[2, 0].transData)
    fig.suptitle(f'{label(a)}: marginal belief over one pending node (3 symbols), probe from all layers, sequence split\n'
                 '(test prefixes also occur in the fit half, so a random network can match by memorizing; see positions.pdf)',
                 fontsize=8)
    fig.tight_layout()
    save(fig, out)


def distances(a, out):
    f = a['dir'] / 'dists.npz'
    if not f.exists():
        return
    d = np.load(f)
    ts = sorted({int(k.split('_')[0][1:]) for k in d.files})
    pick = sorted(ts, key=lambda t: -a['geom'][str(t)]['n_unique'])[:3]
    fig, axes = plt.subplots(2, len(pick), figsize=(2.4 * len(pick), 4.6), squeeze=False)
    last = str(max(int(k) for k in a['metrics']))
    for j, t in enumerate(sorted(pick)):
        dc = a['metrics'][last]['per_pos']['concat'][str(t)]['dist']
        for i, (k, name, r) in enumerate((('d_belief', 'true-belief distance', dc['R2_belief']),
                                          ('d_next', 'next-token distance', dc['R2_nexttok']))):
            ax = axes[i, j]
            ax.hist2d(d[f't{t}_{k}'], d[f't{t}_d_pred'], bins=25, cmap='binary', density=True, cmin=1e-9)
            ax.set_xlabel(name)
            ax.set_ylabel('probe-output distance')
            ax.set_title(f't={t}: R²={r:.2f}', fontsize=8)
            ax.grid(False)
    fig.suptitle(f'{label(a)}: pairwise distances (concat probe, test rows)', fontsize=9)
    fig.tight_layout()
    save(fig, out)


def summary(runs, out):
    by = defaultdict(list)
    for r, a in runs.items():
        by[cfg_name(r)].append(a)
    names = list(by)
    fig, axes = plt.subplots(1, len(names), figsize=(2.1 * len(names), 2.6), sharey=True, squeeze=False)
    for j, n in enumerate(names):
        ax = axes[0, j]
        for k, a in enumerate(by[n]):
            cks = sorted(int(c) for c in a['metrics'])
            fin, ini = a['metrics'][str(cks[-1])]['pooled'], a['metrics'][str(cks[0])]['pooled']
            xs = np.arange(len(SETS))
            first = k == 0
            ax.plot(xs, [fin[s + '@prefix']['R2'] for s in SETS], 'o-', color=CATEGORICAL[0], ms=3, label='trained' if first else None)
            if 'R2_eL4' in fin['L4@prefix']:
                ax.plot([4], [fin['L4@prefix']['R2_eL4']], '*', color=CATEGORICAL[6], ms=7,
                        label='L4, eligible positions (H1)' if first else None)
            ax.plot(xs, [fin[s]['R2'] for s in SETS], 'o:', color=CATEGORICAL[0], ms=2, lw=0.8, alpha=0.6,
                    label='trained, sequence split' if first else None)
            ax.plot(xs, [ini[s + '@prefix']['R2'] for s in SETS], 'o-', color='0.6', ms=3, label='random init' if first else None)
            ax.axhline(fin['nexttok@prefix']['R2'], color=CATEGORICAL[1], ls='--', lw=0.8, label='next-token ceiling' if first else None)
            ax.axhline(fin['tokens@prefix']['R2'], color=CATEGORICAL[2], ls=':', lw=0.8, label='linear in tokens' if first else None)
            ax.plot([len(SETS) - 1], [fin['shuffle']['R2']], 'x', color=CATEGORICAL[4], label='shuffled (seq. split)' if first else None)
        ax.set_xticks(np.arange(len(SETS)), SETS, rotation=60)
        ax.set_title(label(by[n][0]).replace(' s=2', ''), fontsize=7)
        ax.set_ylim(-0.5, 1.02)
    axes[0, 0].set_ylabel('pooled R² on unseen prefixes (belief probe)')
    axes[0, -1].legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    save(fig, out)


def positions(runs, out):
    seed0 = {cfg_name(r): a for r, a in runs.items() if r.endswith('_s0')}
    names = list(seed0)
    fig, axes = plt.subplots(1, len(names), figsize=(2.1 * len(names), 2.4), sharey=True, squeeze=False)
    for j, n in enumerate(names):
        a, ax = seed0[n], axes[0, j]
        cks = sorted(int(c) for c in a['metrics'])
        fin, ini = a['metrics'][str(cks[-1])]['per_pos'], a['metrics'][str(cks[0])]['per_pos']
        ts = a['positions']
        tp = [t for t in ts if str(t) in fin['concat@prefix']]
        for t in ts:
            if t not in a['eligible']['eL4']:
                ax.axvspan(t - 0.5, t + 0.5, color='0.92', lw=0, zorder=0)
        ax.plot(tp, [fin['L4@prefix'][str(t)]['R2'] for t in tp], 'D-', ms=2, lw=0.9, color=CATEGORICAL[6], label='trained, last layer')
        ax.plot(ts, [fin['concat'][str(t)]['R2'] for t in ts], 'o:', ms=2, lw=0.8, alpha=0.6, color=CATEGORICAL[0],
                label='trained, sequence split')
        ax.plot(ts, [ini['concat'][str(t)]['R2'] for t in ts], 'o:', ms=2, lw=0.8, alpha=0.6, color='0.6',
                label='random init, sequence split')
        ax.plot(tp, [fin['concat@prefix'][str(t)]['R2'] for t in tp], 'o-', ms=2.5, color=CATEGORICAL[0], label='trained')
        ax.plot(tp, [ini['concat@prefix'][str(t)]['R2'] for t in tp], 'o-', ms=2, color='0.6', label='random init')
        ax.plot(tp, [fin['nexttok@prefix'][str(t)]['R2'] for t in tp], 's--', ms=2, lw=0.8, color=CATEGORICAL[1], label='next-token ceiling')
        ax.plot(tp, [fin['tokens@prefix'][str(t)]['R2'] for t in tp], '^:', ms=2, lw=0.8, color=CATEGORICAL[2], label='linear in tokens')
        ax.set_title(label(a).replace(' s=2', ''), fontsize=7)
        ax.set_xlabel('position t')
        ax.set_ylim(-0.5, 1.02)
    axes[0, 0].set_ylabel('test R² (belief probe)\nsolid: unseen prefixes')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out)


def training(runs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    names = sorted({cfg_name(r) for r in runs})
    for r, a in runs.items():
        col = CATEGORICAL[names.index(cfg_name(r)) % len(CATEGORICAL)]
        lab = label(a) if r.endswith('_s0') else None
        cks = sorted(int(c) for c in a['metrics'])
        a1.plot(steps_x(cks), [a['metrics'][str(c)]['pooled']['concat@prefix']['R2'] for c in cks], 'o-', ms=2.5, color=col, label=lab)
        h = a['hist']
        a2.plot(steps_x([x['step'] for x in h]), [max(x['excess'], 1e-5) for x in h], color=col, lw=0.9)
    a1.set_xscale('log')
    a1.set_xlabel('update')
    a1.set_ylim(-1, 1.02)
    a1.set_ylabel('pooled R², unseen prefixes (all layers)')
    a2.set_xscale('log')
    a2.set_yscale('log')
    a2.set_xlabel('update')
    a2.set_ylabel('excess test loss over Bayes (nats/token)')
    a1.legend(fontsize=5.5, ncol=1, loc='lower right')
    fig.tight_layout()
    save(fig, out)


def main():
    runs = load(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    for r, a in runs.items():
        if r.endswith('_s0'):
            n = cfg_name(r)
            geometry(a, fig_dir / f'geometry_{n}.pdf')
            triangles(a, fig_dir / f'triangles_{n}.pdf')
            distances(a, fig_dir / f'distances_{n}.pdf')
    summary(runs, fig_dir / 'summary.pdf')
    positions(runs, fig_dir / 'positions.pdf')
    training(runs, fig_dir / 'training.pdf')


if __name__ == '__main__':
    main()
