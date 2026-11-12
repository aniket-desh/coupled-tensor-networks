"""Figures from studies.rhm.interpret JSONs.

Run:  python -m studies.rhm.plot_interp <interpret.json> [more json ...] <figure dir>
With several JSONs (runs of the same setting) each curve is the median over runs with the min-max range shaded;
only checkpoints present in every run are used.

interp_training.pdf, three panels over training on the same log-scaled update axis (step 0, the untrained network,
appears as dotted horizontal references):
  1. diagnostic loss, with the Bayes loss for the 'last' task (0 for 'root');
  2. latent decoding: held-out probe accuracy relative to the Bayes ceiling, median over the nodes of each tree level,
     with the untrained network, the local-token probe and shuffled labels as baselines;
  3. causal importance: loss increase from mean-ablating the top-4 LDA directions of the whitened bond, against random
     4-dimensional whitened subspaces (the 'lda' subspace with k = min(classes - 1, d) is close to the whole bond for
     r = 16, so it is drawn thin for reference).
interp_invariance_coupling.pdf:
  1. invariance: squared whitened distance of same-latent (solid) and changed-latent (dashed) resampled pairs,
     relative to independent pairs (0 = invariant, 1 = no invariance);
  2. fraction of whitened variance explained by the latent;
  3. coupling: cosine between grad L_A and grad L_B over the subtree under node (l, 0), for B sharing A's latent at the
     node (solid) or not (dashed);
  4. coupling test: actual / predicted change of L_A against |predicted| for the restricted SGD steps (all
     checkpoints, subsets and nodes); 1 = first-order agreement, > 0 = same sign."""
import json
import os
import sys

import numpy as np

from ctn.style import C1, C2, C3, INK, MUTED, plt

LEVEL_C = {1: C1, 2: C2, 3: C3}


def load(paths):
    runs = [json.load(open(p)) for p in paths]
    steps = sorted(set.intersection(*[{c['step'] for c in r['checkpoints']} for r in runs]))
    return runs, steps


def series(runs, steps, get):
    """(len(steps), n_runs) array of get(checkpoint) (NaN when missing)."""
    out = np.full((len(steps), len(runs)), np.nan)
    for j, r in enumerate(runs):
        by = {c['step']: c for c in r['checkpoints']}
        for i, s in enumerate(steps):
            v = get(by[s])
            out[i, j] = np.nan if v is None else v
    return out


def lev(key, l):
    return lambda c: c.get('levels', {}).get(str(l), {}).get(key)


def band(ax, x, Y, color, ls='-', lw=1.6, label=None):
    med = np.nanmedian(Y, 1)
    ax.plot(x, med, ls=ls, color=color, lw=lw, label=label)
    if Y.shape[1] > 1:
        ax.fill_between(x, np.nanmin(Y, 1), np.nanmax(Y, 1), color=color, alpha=0.15, lw=0)


def hline(ax, Y0, color, ls=':', lw=1.2):
    v = np.nanmedian(Y0)
    if np.isfinite(v):
        ax.axhline(v, color=color, ls=ls, lw=lw)


def coupling_get(l, setting, subset, key):
    def f(c):
        for r in c.get('coupling', []):
            if r['l'] == l and r['setting'] == setting and r['subset'] == subset:
                return r[key]
        return None
    return f


def main():
    *paths, outdir = sys.argv[1:]
    assert paths, __doc__
    os.makedirs(outdir, exist_ok=True)
    runs, steps = load(paths)
    task = runs[0]['cfg']['task']
    pos = [s for s in steps if s > 0]
    x = np.array(pos)
    sel = [steps.index(s) for s in pos]
    i0 = [steps.index(0)] if 0 in steps else []
    S = lambda get: series(runs, steps, get)
    levels = [l for l in (1, 2, 3) if any(str(l) in c.get('levels', {}) for c in runs[0]['checkpoints'])]
    tag = f"{task} task, {runs[0]['cfg']['model']} r = {runs[0]['cfg']['r']}, {len(runs)} run(s)"

    # ---- figure 1: loss, decoding, causal importance
    fig, ax = plt.subplots(3, 1, figsize=(8.6, 8.2), sharex=True)
    out_legend = dict(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7)
    Y = S(lambda c: c['loss'])
    band(ax[0], x, Y[sel], INK, label='diagnostic loss')
    if i0:
        hline(ax[0], Y[i0], MUTED)
    bl = np.median([r['bayes']['loss'] for r in runs])
    if bl > 0:
        ax[0].axhline(bl, color=C2, ls='--', lw=1.2, label=f'Bayes ({bl:.3f})')
    ax[0].plot([], [], color=MUTED, ls=':', label='untrained (step 0)')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('cross-entropy')
    ax[0].set_title(f'Diagnostic loss ({tag})' + ('; Bayes loss 0' if bl == 0 else ''))
    ax[0].legend(**out_legend)

    for l in levels:
        c = LEVEL_C[l]
        Y = S(lev('probe_rel', l))
        band(ax[1], x, Y[sel], c, label=f'tree level {l} (grammar level {4 - l})')
        if i0:
            hline(ax[1], Y[i0], c, ':')
        hline(ax[1], S(lev('local', l))[sel], c, '-.')
    Ysh = np.nanmedian(np.stack([S(lev('probe_shuffled', l)) for l in levels]), 0)
    band(ax[1], x, Ysh[sel], MUTED, lw=1.0, label='shuffled labels (median over levels)')
    ax[1].plot([], [], color=MUTED, ls=':', label='untrained (step 0)')
    ax[1].plot([], [], color=MUTED, ls='-.', label='local-token probe')
    ax[1].set_ylabel('probe acc / Bayes ceiling' if task == 'last' else 'held-out probe accuracy')
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title('Latent decoding from whitened bonds (median over nodes of a level)')
    ax[1].legend(**out_legend)

    for l in levels:
        c = LEVEL_C[l]
        band(ax[2], x, S(lev('abl_lda4', l))[sel], c, label=f'level {l}: top-4 LDA')
        band(ax[2], x, S(lev('abl_rand4', l))[sel], c, '--', lw=1.2)
        band(ax[2], x, S(lev('abl_lda', l))[sel], c, '-', lw=0.6)
    ax[2].plot([], [], color=MUTED, ls='--', label='random 4-dim subspace')
    ax[2].plot([], [], color=MUTED, lw=0.6, label='LDA, k = min(classes-1, d)')
    ax[2].axhline(0, color=MUTED, lw=0.6)
    ax[2].set_yscale('symlog', linthresh=1e-2)
    ax[2].set_ylabel('loss increase (symlog)')
    ax[2].set_title('Causal importance: mean-ablating whitened subspaces')
    ax[2].legend(**out_legend)
    ax[2].set_xscale('log')
    ax[2].set_xlabel('update')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'interp_training.pdf'))
    plt.close(fig)

    # ---- figure 2: invariance and coupling
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.6))
    a = ax[0, 0]
    for l in levels:
        band(a, x, S(lev('inv_same', l))[sel], LEVEL_C[l], label=f'level {l}')
        band(a, x, S(lev('inv_sym', l))[sel], LEVEL_C[l], '--', lw=1.2)
    a.plot([], [], color=MUTED, ls='--', label='changed latent')
    a.set_xscale('log')
    a.set_ylim(bottom=0)
    a.set_title('Invariance (solid: same latent)')
    a.set_ylabel('E|du|^2 / E|du|^2 (independent pairs)')
    a.legend(fontsize=7)
    a = ax[0, 1]
    for l in levels:
        band(a, x, S(lev('var_explained', l))[sel], LEVEL_C[l], label=f'level {l}')
        if i0:
            hline(a, S(lev('var_explained', l))[i0], LEVEL_C[l], ':')
    a.set_xscale('log')
    a.set_ylim(0, 1.02)
    a.set_title('Variance explained by the latent (dotted: step 0)')
    a.legend(fontsize=7)
    a = ax[1, 0]
    for l in levels:
        band(a, x, S(coupling_get(l, 'same', 'subtree', 'cos'))[sel], LEVEL_C[l], label=f'node ({l}, 0)')
        band(a, x, S(coupling_get(l, 'diff', 'subtree', 'cos'))[sel], LEVEL_C[l], '--', lw=1.2)
    a.plot([], [], color=MUTED, ls='--', label='B: different latent')
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_xscale('log')
    a.set_xlabel('update')
    a.set_title('Subtree gradient coupling (solid: same latent)')
    a.set_ylabel('cos(grad L_A, grad L_B)')
    a.legend(fontsize=7)
    a = ax[1, 1]
    pts = [(r['predicted'], r['actual'], r['l'], r['setting']) for run in runs for c in run['checkpoints']
           for r in c.get('coupling', [])]
    pts = [p for p in pts if p[0] != 0]
    if pts:
        P = np.array([p[:2] for p in pts])
        for (p, q, l, st) in pts:
            a.scatter(abs(p), q / p, s=12, color=LEVEL_C[l], marker='o' if st == 'same' else 'x', lw=0.8)
        a.axhline(1, color=MUTED, lw=0.8)
        a.axhline(0, color=MUTED, lw=0.6, ls=':')
        a.set_xscale('log')
        agree = np.mean(np.sign(P[:, 0]) == np.sign(P[:, 1]))
        a.set_title(f'Coupling test (sign agreement {agree:.0%})')
    a.scatter([], [], color=MUTED, marker='o', s=12, label='B: same latent')
    a.scatter([], [], color=MUTED, marker='x', s=12, label='B: different latent')
    a.set_xlabel('|predicted dL_A| = eta |<grad L_A, grad L_B>|')
    a.set_ylabel('actual / predicted dL_A')
    a.legend(fontsize=7)
    fig.suptitle(tag, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'interp_invariance_coupling.pdf'))
    plt.close(fig)
    print('wrote', os.path.join(outdir, 'interp_training.pdf'), os.path.join(outdir, 'interp_invariance_coupling.pdf'))


if __name__ == '__main__':
    main()
