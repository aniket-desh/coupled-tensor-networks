"""Markdown tables for README.md and NOTE.md, from results/summary.json and results/*/analysis.json only.

    PYTHONPATH=. python studies/rhm/rhm-belief-architectures/arch_table.py RESULTS_DIR

Every cell is the median over the 3 seeds with [min, max]. All R2 are on held-out prefixes at the final checkpoint,
pooled over a position group of arch_analyze.py (ea, pa, eb, eb_hi; see its docstring)."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ('tree', 'perm', 'chain')
NAME = dict(tree='aligned tree', perm='permuted tree', chain='chain')
ORDER = ('v8m2s2L4', 'v4m2s2L5', 'v8m4s2L4', 'v16m4s2L4')  # learned by every arm first


def mr(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '–'
    return f'{statistics.median(vals):.{nd}f} [{min(vals):.{nd}f}, {max(vals):.{nd}f}]'


def cnt(vals):
    vals = sorted(set(vals))
    return str(vals[0]) if len(vals) == 1 else f'{vals[0]}–{vals[-1]}'


def main():
    res = Path(sys.argv[1])
    S = json.load(open(res / 'summary.json'))
    cell = defaultdict(list)
    for r in sorted(S['runs'], key=lambda r: r['seed']):
        cell[(r['config'], r['arm'])].append(r)
    col = lambda c, a, k: [r.get(k) for r in cell[(c, a)]]

    print('Table 1. Final excess test loss (nats/token), 3 seeds; learned = all three below 0.02\n')
    print('| grammar | T | ' + ' | '.join(NAME[a] for a in ARMS) + ' |\n|---|---|---|---|---|')
    for c in ORDER:
        T = json.load(open(res / f'{c}_tree_s0' / 'analysis.json'))['cfg']['T']
        cells = [mr(col(c, a, 'excess'), 4) + ('' if S['cells'][f'{c}_{a}']['learned'] else ' **not learned**') for a in ARMS]
        print(f'| {c} | {T} | ' + ' | '.join(cells) + ' |')

    print('\nTable 2. Chain state h_t, three probe families against their own controls (H_A)\n')
    print('| grammar | chain learned | probe | positions | trained | identity init | token readout | holds in seeds |\n|---|---|---|---|---|---|---|---|')
    fam = (('affine (A1)', 'ea', 'ea_aff', 'ea_init', 'ea_tokens', 'A1'),
           ('projective (A2)', 'pa', 'pa_proj', 'pa_proj_init', 'pa_proj_tokens', 'A2'),
           ('softmax (A3)', 'pa', 'pa_soft', 'pa_soft_init', 'pa_soft_tokens', 'A3'))
    for c in ORDER:
        for lab, grp, k, k0, kt, flag in fam:
            rr = cell[(c, 'chain')]
            print(f"| {c} | {'yes' if S['cells'][f'{c}_chain']['learned'] else 'no'} | {lab} | {cnt([r['n'][grp] for r in rr])} | "
                  f"{mr(col(c, 'chain', k))} | {mr(col(c, 'chain', k0))} | {mr(col(c, 'chain', kt))} | {sum(bool(r.get(flag)) for r in rr)} of 3 |")

    print('\nTable 3. Aligned against permuted tree, per (grammar, seed) pair (H_B)\n')
    print('| pair | excess aligned | excess permuted | gap | tolerance | learning | R2 aligned | R2 permuted | final gap | init gap | adjusted | representation |\n'
          '|---|---|---|---|---|---|---|---|---|---|---|---|')
    word = {1: 'aligned', -1: 'permuted', 0: 'tie'}
    for p in sorted(S['H_B']['pairs'], key=lambda p: (ORDER.index(p['pair'].rsplit('_s', 1)[0]), p['pair'])):
        h = p['eb_hi']
        used = word[h['sign']] if p['other_learned'] else 'not used (permuted not learned)'
        print(f"| {p['pair']} | {p['ref_excess']:.4f} | {p['other_excess']:.4f} | {p['learning_gap']:.4f} | {p['tol']:.4f} | "
              f"{word[p['learning']]} | {h['ref_R2']:.3f} | {h['other_R2']:.3f} | {h['gap']:.3f} | {h['init_gap']:.3f} | {h['adjusted']:.3f} | {used} |")

    print('\nTable 4. The three arms on the common positions eb (trees: bilinear top-bond probe; chain: affine state probe)\n')
    print('| grammar | arm | positions | trained | identity init | scrambled target | token readout | next-token ceiling |\n|---|---|---|---|---|---|---|---|')
    for c in ORDER:
        for a in ARMS:
            print(f"| {c} | {NAME[a]} | {cnt([r['n']['eb'] for r in cell[(c, a)]])} | {mr(col(c, a, 'eb_best'))} | {mr(col(c, a, 'eb_init'))} | "
                  f"{mr(col(c, a, 'eb_scr'))} | {mr(col(c, a, 'eb_tokens'))} | {mr(col(c, a, 'eb_nexttok'))} |")

    print('\nTable 5. Trees past the midpoint (eb_hi): affine and bilinear top-bond probes and the additive upper bound\n')
    print('| grammar | arm | positions | affine | additive upper bound | bilinear |\n|---|---|---|---|---|---|')
    for c in ORDER:
        for a in ('tree', 'perm'):
            aff, up = [], []
            for r in cell[(c, a)]:
                A = json.load(open(res / r['run'] / 'analysis.json'))
                P = A['metrics'][str(max(int(k) for k in A['metrics']))]['pooled']
                aff.append(P['top@prefix'].get('R2_eb_hi'))
                up.append(P.get('addup@prefix', {}).get('R2_eb_hi'))
            print(f"| {c} | {NAME[a]} | {cnt([r['n']['eb_hi'] for r in cell[(c, a)]])} | {mr(aff)} | {mr(up)} | {mr(col(c, a, 'eb_hi_best'))} |")
    print('\nTable 6. Belief split (distinct beliefs held out), pooled over its own positions\n')
    print('| grammar | arm | positions | affine probe | bilinear probe | span ceiling |\n|---|---|---|---|---|---|')
    for c in ORDER:
        for a in ARMS:
            print(f"| {c} | {NAME[a]} | {cnt([r['n']['bel'] for r in cell[(c, a)]])} | {mr(col(c, a, 'bel_aff'))} | {mr(col(c, a, 'bel_bil'))} | "
                  f"{mr(col(c, a, 'bel_span'))} |")
    rr = S['runs']
    print(f"\nshuffled-label pooled R2 over the {len(rr)} runs: min {min(r['shuffle'] for r in rr):.4f}, max {max(r['shuffle'] for r in rr):.4f}")
    print('parameters:', {f'{c}_{a}': cell[(c, a)][0]['n_params'] for c in ORDER for a in ARMS})


if __name__ == '__main__':
    main()
