"""lang-induction figures from saved results.

  PYTHONPATH=. python studies/lang/lang-induction/plot.py <results dir> <figures dir>

width.pdf    final test KL vs width for retrieval and multi-match: tree (r = 16..128), perm (16, 64) and the chain at
             the parameter count of the r = 16 and r = 64 trees (plotted at that tree width); one point per seed.
strata.pdf   retrieval: test accuracy by target pair (0-3 meet the query only at the root, 4-5 at level 3, 6 at level 2),
             one line per tree width (seed median), Bayes accuracy dashed.
staging.pdf  retrieval tree runs: left-half dictionary R^2 (node 3_0) and diagnostic loss vs update, one panel per width.
curves.pdf   diagnostic KL vs update for every run, panels by process and arm family.
"""
import sys

from studies.lang import plotting as Pl

WIDTHS = (16, 32, 64, 128)


def main():
    res, figs = sys.argv[1], sys.argv[2]
    an, cv, ip = Pl.load(res)
    runs = an.get('runs', {})
    fig, ax = Pl.plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, pn in zip(ax, ('retrieval', 'multimatch')):
        for k, (fam, ws) in enumerate((('tree', WIDTHS), ('perm', (16, 64)), ('chain', (16, 64)))):
            for w in ws:
                for s in range(3):
                    r = runs.get(f'{pn}_{fam}{w}_s{s}')
                    if r:
                        a.plot(w * (1 + 0.06 * (k - 1)), Pl.num(r['kl']), 'o', ms=4, color=Pl.COLORS[k],
                               mfc='none' if r.get('collapsed') else Pl.COLORS[k], label=fam)
        a.set_xscale('log', base=2), a.set_yscale('log'), a.set_title(pn), a.set_xlabel('bond dimension r (chain: matched params)')
        a.axhline(0.01, color='gray', lw=0.6, ls=':')
        Pl.legend(a, fontsize=7)
    ax[0].set_ylabel('final test KL [nats] (open = collapsed)')
    Pl.save(fig, figs, 'width')

    fig, ax = Pl.plt.subplots(1, 1, figsize=(6, 3.6))
    psa = an.get('H2_retrieval_strata', {}).get('per_stratum_acc', {})
    for k, w in enumerate(WIDTHS):
        d = psa.get(str(w)) or {}
        ax.plot(range(7), [Pl.num(d.get(str(m))) for m in range(7)], 'o-', color=Pl.COLORS[k], label=f'tree r={w}', ms=4)
    bayes = [Pl.num(runs.get(f'retrieval_tree16_s0', {}).get('bayes_acc'))] * 7
    ax.plot(range(7), bayes, 'k--', lw=0.8, label='Bayes')
    ax.set_xlabel('target pair m (0-3: root, 4-5: level 3, 6: level 2)'), ax.set_ylabel('test accuracy (seed median)')
    Pl.legend(ax, fontsize=7)
    Pl.save(fig, figs, 'strata')

    fig, ax = Pl.plt.subplots(1, 4, figsize=(15, 3.3), sharey=True)
    st = (an.get('H5_staging') or {}).get('runs', {})
    for a, w in zip(ax, WIDTHS):
        for s in range(3):
            n = f'retrieval_tree{w}_s{s}'
            d = st.get(n) or {}
            if d.get('r2'):  # node 3_0 (the left half), from the staging analysis
                a.plot([max(int(x), 1) for x in d['steps']], [Pl.num(v) for v in d['r2']], color=Pl.COLORS[s], label=f'dict R^2 (3_0) s{s}')
            if n in cv:
                L = [Pl.num(v) for v in cv[n]['diag_loss']]
                a.plot([max(x, 1) for x in cv[n]['step']], [v / max(L[0], 1e-9) for v in L], color=Pl.COLORS[s], ls='--', lw=0.8,
                       label=f'loss / initial s{s}')
            if n in st and 'dict_half' in st[n]:
                a.axvline(Pl.num(st[n]['dict_half']), color=Pl.COLORS[s], lw=0.5, ls=':')
        a.set_xscale('log'), a.set_title(f'retrieval tree r={w}'), a.set_xlabel('update'), a.set_ylim(-0.05, 1.05)
    Pl.legend(ax[0], fontsize=6)
    Pl.save(fig, figs, 'staging')

    fig, ax = Pl.plt.subplots(2, 3, figsize=(13, 6), sharey=True)
    for i, pn in enumerate(('retrieval', 'multimatch')):
        for j, fam in enumerate(('tree', 'perm', 'chain')):
            for k, w in enumerate(WIDTHS):
                for s in range(3):
                    n = f'{pn}_{fam}{w}_s{s}'
                    if n in cv:
                        Pl.curve(ax[i, j], cv[n], color=Pl.COLORS[k], lw=0.8, label=f'r={w}')
            ax[i, j].set_xscale('log'), ax[i, j].set_yscale('log'), ax[i, j].set_title(f'{pn} {fam}')
            Pl.legend(ax[i, j], fontsize=7)
    Pl.save(fig, figs, 'curves')


if __name__ == '__main__':
    main()
