"""Figures from saved analysis results only.

    PYTHONPATH=. python studies/rhm/rhm-belief-architectures/arch_plot.py RESULTS_DIR FIGURES_DIR [TRANSFORMER_RESULTS_DIR]

RESULTS_DIR holds one folder per run (<config>_<arm>_s<seed>/ with analysis.json and proj_ckpt*.npz).

Per (grammar, arm), seed 0:
  geometry_<config>_<arm>.pdf   as rhm-ttn-belief-geometry/ttn_plot.py: ground-truth beliefs on their top two PCs, the
      arm's affine probe (top bonds, or the chain state), the bilinear top-bond probe (trees), the shuffled-label
      control, the best probe on the identity-init network, and the raw top-2 PCs of the probed state (gauge
      dependent). Rows: two positions up to T/2 and two after it, preferring a next-token ceiling < 0.9 and then the
      most distinct beliefs. Sequence split, test sequences.
  triangles_<config>_<arm>.pdf  (v = 3) one pending node's marginal on the 2-simplex: truth, probes, identity init.
Across runs:
  positions.pdf  one panel per grammar (seed 0): per-position R2 on unseen prefixes of each arm's probes (trees:
      bilinear and affine top bonds; chain: affine, projective and softmax-of-affine state), with the token readout, the next-token
      ceiling and, where it could be computed, each tree's additive upper bound (no affine probe on the two top
      bonds can exceed it); gray = not in the common set eb;
      dashed line = T/2. With TRANSFORMER_RESULTS_DIR, the transformer's last-layer probe is added.
  summary.pdf    per grammar, one point per seed: pooled R2 on the common set past the midpoint (eb_hi, the set of
      H_B's representation measure) of each arm's probe with its
      identity-init and token controls (top), and the final excess loss with the 0.02 'learned' line (bottom).
  training.pdf   per grammar: excess test loss against updates for every run, and the pooled R2 (ea) of each arm's
      affine probe over checkpoints."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
import plot as PL  # noqa: E402
from plotting import CATEGORICAL, INK2, plt, save, steps_x  # noqa: E402

ARMS = ('tree', 'perm', 'chain')
COL = dict(tree=CATEGORICAL[6], perm=CATEGORICAL[1], chain=CATEGORICAL[2])
NAME = dict(tree='aligned tree', perm='permuted tree', chain='chain')


def label(a):
    return f"{PL.label(a)}, {NAME[a['arm']]}"


def last_first(a):
    cks = sorted(int(c) for c in a['metrics'])
    return a['metrics'][str(cks[-1])], a['metrics'][str(cks[0])], cks


def pick_rows(a, ceil):
    H = a['half']
    cand = [t for t in a['positions'] if a['geom'][str(t)]['n_unique'] > 1]
    key = lambda t: (ceil[str(t)]['R2'] >= 0.9, -a['geom'][str(t)]['n_unique'])
    return sorted(sorted([t for t in cand if t <= H], key=key)[:2] + sorted([t for t in cand if t > H], key=key)[:2])


def geometry(a, out):
    fin, ini, cks = last_first(a)
    first, last = (np.load(a['dir'] / f'proj_ckpt{c}.npz') for c in (cks[0], cks[-1]))
    aff, bil = a['probes']['affine'], a['probes']['bilinear']
    state = 'top bonds' if bil else 'chain state'
    ceil = fin['per_pos']['nexttok']
    pos = pick_rows(a, ceil)
    cols = [('true', 'ground truth', last, None, fin), ('pred_aff', f'affine probe: {state}', last, aff, fin)]
    if bil:
        cols.append(('pred_bil', 'bilinear probe: top bonds', last, bil, fin))
    cols += [('pred_shuffle', 'shuffled labels', last, 'shuffle', fin),
             ('pred_bil' if bil else 'pred_aff', 'identity init, ' + ('bilinear' if bil else 'affine'), first, bil or aff, ini),
             ('raw_pca', f'raw PCA, {state}', last, None, fin)]
    fig, axes = plt.subplots(len(pos), len(cols), figsize=(2.0 * len(cols), 2.0 * len(pos)), squeeze=False)
    for i, t in enumerate(pos):
        uid, true = last[f't{t}_uid'], last[f't{t}_true']
        c = PL.rgb(true, uid)
        for j, (key, title, src, mk, met) in enumerate(cols):
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
                ax.text(0.02, 0.02, f"R²={met['per_pos'][mk][str(t)]['R2']:.2f}", transform=ax.transAxes, fontsize=6, color=INK2)
            if j == 0:
                gm = a['geom'][str(t)]
                ax.set_ylabel(f"t={t}\n{gm['n_unique']} beliefs, D={gm['D']}\nnext-token ceiling R²={ceil[str(t)]['R2']:.2f}", fontsize=7)
    fig.suptitle(f'{label(a)}: hidden states vs belief states (top PCs of the beliefs; sequence split, test sequences)', fontsize=9)
    fig.tight_layout()
    save(fig, out)


def triangles(a, out):
    fin, ini, cks = last_first(a)
    p, p0 = (np.load(a['dir'] / f'proj_ckpt{c}.npz') for c in (cks[-1], cks[0]))
    keys = sorted({k.rsplit('_', 1)[0] for k in p.files if k.endswith('_true') and '_tri' in k},
                  key=lambda k: (int(k[1:].split('_')[0]), k))
    keys = [k for k in keys if len(np.unique(p[k + '_true'].round(6), axis=0)) > 2]
    if not keys:
        return
    H, tt = a['half'], (lambda k: int(k[1:].split('_')[0]))
    keys = [k for k in keys if tt(k) <= H][-5:] + [k for k in keys if tt(k) > H][:5]
    bil = a['probes']['bilinear']
    rows = [(p, '_true', 25, 1.0, 'ground truth'), (p, '_aff', 1.5, 0.2, 'affine probe')]
    if bil:
        rows.append((p, '_bil', 1.5, 0.2, 'bilinear probe'))
    rows.append((p0, '_bil' if bil else '_aff', 1.5, 0.2, 'identity init'))
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
        axes[0, j].set_title(f't={t[1:]}\nnode {node[3:]}, level {lev[1:]}', fontsize=7)
    fig.suptitle(f'{label(a)}: marginal belief over one pending node (3 symbols), sequence split', fontsize=8)
    fig.tight_layout()
    save(fig, out)


def by_config(runs, seed=None):
    out = defaultdict(dict)
    for r, a in runs.items():
        if seed is None or a['cfg']['grammar']['seed'] == seed:
            out[a['cfg']['config']].setdefault(a['arm'], []).append(a)
    return dict(sorted(out.items()))


def positions(runs, out, tf_dir=None):
    cfgs = by_config(runs, 0)
    tf = PL.load(tf_dir) if tf_dir else {}
    fig, axes = plt.subplots(1, len(cfgs), figsize=(2.9 * len(cfgs), 2.6), sharey=True, squeeze=False)
    handles = {}

    def line(ax, d, ts, lab, **style):
        ts = [t for t in ts if str(t) in d]
        if ts:
            handles[lab] = ax.plot(ts, [max(d[str(t)]['R2'], -0.6) for t in ts], **style)[0]
    for ax, (c, arms) in zip(axes[0], cfgs.items()):
        ref = next(iter(arms.values()))[0]
        for t in ref['positions']:
            if t not in ref['eligible']['eb']:
                ax.axvspan(t - 0.5, t + 0.5, color='0.92', lw=0, zorder=0)
        ax.axvline(ref['half'] + 0.5, color='0.4', ls='--', lw=0.6)
        fr = last_first(ref)[0]['per_pos']
        line(ax, fr['nexttok@prefix'], ref['positions'], 'next-token ceiling', marker='s', ls='--', ms=1.5, lw=0.7, color='0.35')
        line(ax, fr['tokens@prefix'], ref['positions'], 'linear in tokens', marker='^', ls=':', ms=1.5, lw=0.7, color='0.6')
        for arm in ARMS:
            if arm not in arms:
                continue
            a = arms[arm][0]
            f = last_first(a)[0]['per_pos']
            aff, bil = a['probes']['affine'], a['probes']['bilinear']
            if bil:
                line(ax, f.get(bil + '@prefix', {}), a['positions'], f'{NAME[arm]}: bilinear, top bonds', marker='D', ls='-', ms=2, lw=1.0, color=COL[arm])
                line(ax, f[aff + '@prefix'], a['positions'], f'{NAME[arm]}: affine, top bonds', marker='o', ls=':', ms=1.5, lw=0.7, color=COL[arm], alpha=0.7)
                line(ax, f.get('addup@prefix', {}), a['positions'], f'{NAME[arm]}: additive upper bound', marker='_', ls='none', ms=5, color=COL[arm])
            else:
                line(ax, f[aff + '@prefix'], a['positions'], f'{NAME[arm]}: affine, state', marker='o', ls='-', ms=2, lw=1.0, color=COL[arm])
                line(ax, f.get(aff + '~proj@prefix', {}), a['positions'], f'{NAME[arm]}: projective, state', marker='x', ls='none', ms=3, color=COL[arm])
                line(ax, f.get(aff + '~soft@prefix', {}), a['positions'], f'{NAME[arm]}: softmax of affine, state', marker='+', ls='none', ms=4, color=COL[arm])
        if c + '_s0' in tf:
            ft = last_first(tf[c + '_s0'])[0]['per_pos']
            line(ax, ft.get('L4@prefix', {}), ref['positions'], 'transformer: affine, last layer', marker='s', ls='-', ms=2, lw=0.9, color=CATEGORICAL[0])
        ax.set_title(PL.label(ref).replace(' s=2', ''), fontsize=7)
        ax.set_xlabel('prefix length t')
        ax.set_ylim(-0.5, 1.02)
    axes[0, 0].set_ylabel('R² on unseen prefixes')
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out)


def summary(runs, out):
    cfgs = by_config(runs)
    fig, axes = plt.subplots(2, len(cfgs), figsize=(2.1 * len(cfgs), 4.4), sharey='row', squeeze=False)
    handles = {}
    for j, (c, arms) in enumerate(cfgs.items()):
        for x, arm in enumerate(ARMS):
            for a in arms.get(arm, []):
                fin, ini, _ = last_first(a)
                best = a['probes']['bilinear'] or a['probes']['affine']
                dx = 0.14 * (a['cfg']['grammar']['seed'] - 1)
                for val, mk, col, lab in ((fin['pooled'].get(best + '@prefix', {}).get('R2_eb_hi'), 'o', COL[arm], 'trained'),
                                          (ini['pooled'].get(best + '@prefix', {}).get('R2_eb_hi'), 'o', '0.6', 'identity init'),
                                          (fin['pooled'].get('tokens@prefix', {}).get('R2_eb_hi'), '^', '0.3', 'linear in tokens'),
                                          (fin['pooled'].get(best + '~scr@prefix', {}).get('R2_eb_hi'), 'x', CATEGORICAL[4], 'scrambled target')):
                    if val is not None:
                        h = axes[0, j].plot([x + dx], [max(val, -0.5)], mk, color=col, ms=3.5)[0]
                        handles[lab] = h if lab != 'trained' else handles.get(lab, axes[0, j].plot([], [], 'o', color='k', ms=3.5)[0])
                axes[1, j].plot([x + dx], [max(a['final_excess'], 1e-5)], 'o', color=COL[arm], ms=3.5)
        ref = next(iter(arms.values()))[0]
        axes[0, j].set_title(PL.label(ref).replace(' s=2', ''), fontsize=7)
        axes[1, j].axhline(0.02, color='0.4', ls='--', lw=0.6)
        axes[1, j].set_yscale('log')
        for ax in axes[:, j]:
            ax.set_xticks(range(3), [NAME[a].replace(' ', '\n') for a in ARMS], fontsize=6)
            ax.set_xlim(-0.5, 2.5)
        axes[0, j].set_ylim(-0.52, 1.02)
    axes[0, 0].set_ylabel('pooled R², unseen prefixes, common set, t > T/2\n(trees: bilinear top; chain: state)')
    axes[1, 0].set_ylabel('final excess test loss (nats/token)')
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, out)


def training(runs, out):
    cfgs = by_config(runs)
    fig, axes = plt.subplots(2, len(cfgs), figsize=(2.3 * len(cfgs), 4.4), sharey='row', squeeze=False)
    for j, (c, arms) in enumerate(cfgs.items()):
        for arm in ARMS:
            for k, a in enumerate(arms.get(arm, [])):
                h = a['hist']
                axes[0, j].plot(steps_x([x['step'] for x in h]), [max(x['excess'], 1e-5) for x in h], color=COL[arm], lw=0.8,
                                label=NAME[arm] if k == 0 else None)
                cks = sorted(int(q) for q in a['metrics'])
                ys = [a['metrics'][str(q)]['pooled'].get(a['probes']['affine'] + '@prefix', {}).get('R2_ea') for q in cks]
                if all(y is not None for y in ys):
                    axes[1, j].plot(steps_x(cks), [max(y, -1) for y in ys], 'o-', ms=2, lw=0.8, color=COL[arm])
        ref = next(iter(arms.values()))[0]
        axes[0, j].set_title(PL.label(ref).replace(' s=2', ''), fontsize=7)
        axes[0, j].axhline(0.02, color='0.4', ls='--', lw=0.6)
        axes[0, j].set_xscale('log')
        axes[0, j].set_yscale('log')
        axes[1, j].set_xscale('log')
        axes[1, j].set_ylim(-1, 1.02)
        axes[1, j].set_xlabel('update')
    axes[0, 0].set_ylabel('excess test loss over Bayes\n(nats/token)')
    axes[1, 0].set_ylabel('pooled R², unseen prefixes\n(affine probe, own eligible positions)')
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, out)


def main():
    runs = PL.load(sys.argv[1])
    fig_dir = Path(sys.argv[2])
    tf_dir = sys.argv[3] if len(sys.argv) > 3 and Path(sys.argv[3]).is_dir() else None
    for r, a in runs.items():
        if a['cfg']['grammar']['seed'] == 0:
            geometry(a, fig_dir / f"geometry_{a['cfg']['config']}_{a['arm']}.pdf")
            triangles(a, fig_dir / f"triangles_{a['cfg']['config']}_{a['arm']}.pdf")
    positions(runs, fig_dir / 'positions.pdf', tf_dir)
    summary(runs, fig_dir / 'summary.pdf')
    training(runs, fig_dir / 'training.pdf')


if __name__ == '__main__':
    main()
