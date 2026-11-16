"""Markdown tables for README.md, from saved results only.

    PYTHONPATH=. python studies/rhm/rhm-ttn-masks/mask_table.py RESULTS_DIR

Reads RESULTS_DIR/summary.json (mask_summary.py), RESULTS_DIR/<arm>/summary.json (msg_summary.py: every measure per
run at the final checkpoint) and RESULTS_DIR/<arm>/<run>/analysis.json (the same measures at update 3,000).
A cell is `median [min, max]` over the three seeds of one grammar in one arm. No number is recomputed here except the
gap rule 1 - R2_mixed <= 0.5 (1 - max(R2_prefix, 0)), which is mask_summary.py's, and the per-seed differences."""
import json
import statistics
import sys
from pathlib import Path

ARMS, CELLS = ('prefix', 'mixed', 'varied'), ('v3m2s2L4', 'v8m2s2L4', 'v8m4s2L4', 'v16m4s2L4')


def cell(vals, d=2):
    vals = [v for v in vals if v is not None]
    return '-' if not vals else f'{statistics.median(vals):.{d}f} [{min(vals):.{d}f}, {max(vals):.{d}f}]'


def table(head, rows):
    print('| ' + ' | '.join(head) + ' |\n|' + ' --- |' * len(head))
    for r in rows:
        print('| ' + ' | '.join(str(x) for x in r) + ' |')
    print()


def main():
    res = Path(sys.argv[1])
    S = json.load(open(res / 'summary.json'))
    R = {a: {q['run']: q for q in json.load(open(res / a / 'summary.json'))['runs']} for a in ARMS}
    V = {a: S['arms'][a]['runs'] for a in ARMS}                      # carries vexcess, which msg_summary does not
    seeds = lambda a, c, k, src=None: [(src or R)[a][f'{c}_r32_s{s}'][k] for s in range(3)]

    print('losses at update 30,000 (nats/token)\n')
    table(['grammar', 'arm', 'prefix excess', 'varied excess', 'prefix gate (< 0.02, 3 seeds)', 'varied gate (< 0.02, 3 seeds)'],
          [(f'`{c}`', a, cell(seeds(a, c, 'excess'), 4), cell(seeds(a, c, 'vexcess', V), 3),
            'yes' if max(seeds(a, c, 'excess')) < 0.02 else 'no', 'yes' if max(seeds(a, c, 'vexcess', V)) < 0.02 else 'no')
           for c in CELLS for a in ARMS])

    print('M4, parents of level >= 2: R2 of the decoded merge against the exact update\n')
    table(['grammar', 'arm', 'both-informative', 'identity init', 'additive ceiling', 'all empirical pairs', 'left-masked',
           'seen in prefix training', 'runs passing H4'],
          [(f'`{c}`', a, cell(seeds(a, c, 'm4_both')), cell(seeds(a, c, 'm4_both_init')), cell(seeds(a, c, 'm4_both_ceil')),
            cell(seeds(a, c, 'm4_emp')), cell(seeds(a, c, 'm4_left')), cell(seeds(a, c, 'm4_seen')),
            f"{sum(bool(x) for x in seeds(a, c, 'h4'))}/3") for c in CELLS for a in ARMS])

    print('both-informative R2 seed by seed: mixed against prefix (gap rule: 1 - mixed <= 0.5 (1 - max(prefix, 0)))\n')
    rows = []
    for c in CELLS:
        p, m, v = (seeds(a, c, 'm4_both') for a in ARMS)
        rows.append((f'`{c}`', ', '.join(f'{x:.2f}' for x in p), ', '.join(f'{x:.2f}' for x in m), ', '.join(f'{x:.2f}' for x in v),
                     ', '.join(f'{b - a:+.2f}' for a, b in zip(p, m)),
                     f'{sum(1 - b <= 0.5 * (1 - max(a, 0)) for a, b in zip(p, m))}/3',
                     ', '.join(f'{x:.2f}' for x in seeds('mixed', c, 'm4_both_ceil'))))
    table(['grammar', 'prefix (seeds 0, 1, 2)', 'mixed', 'varied', 'mixed − prefix', 'seeds closing the gap', 'additive ceiling'], rows)

    print('M3, parents of level >= 2: invalid child pairs\n')
    rows = []
    for c in CELLS:
        ratios = [S['h_b']['ratios'][f'{c}_r32_s{s}'] for s in range(3)]
        for a in ARMS:
            rows.append((f'`{c}`', a, cell(seeds(a, c, 'm3_invalid_q')), cell(seeds(a, c, 'm3_invalid_q_init')),
                         cell(seeds(a, c, 'm3_invalid')), cell(seeds(a, c, 'm3_acc')), cell(seeds(a, c, 'm3_valid_q_cv')),
                         ', '.join(f'{x:.2f}' for x in ratios) if a == 'mixed' else ''))
    table(['grammar', 'arm', 'invalid / valid read-out mass', 'identity init', 'invalid / valid bond norm', 'accuracy on valid pairs',
           'spread of valid read-out mass (cv)', 'paired ratio mixed / prefix (seeds 0, 1, 2)'], rows)

    print('other measures of msg_summary.py (descriptive here)\n')
    table(['grammar', 'arm', 'M1: partial message from bond', 'M1: fully observed rows', 'M1x: extended message',
           'M2: ve by extended message', 'M2: ve by message alone', 'M4a: code linearity', 'H1', 'H2', 'H3', 'H4'],
          [(f'`{c}`', a, cell(seeds(a, c, 'm1_proj')), cell(seeds(a, c, 'm1_full')), cell(seeds(a, c, 'm1x_proj')),
            cell(seeds(a, c, 'm2_ve')), cell(seeds(a, c, 'm2_msg')), cell(seeds(a, c, 'm4a_emp')),
            *(f"{sum(bool(x) for x in seeds(a, c, h))}/3" for h in ('h1', 'h2', 'h3', 'h4'))) for c in CELLS for a in ARMS])

    print('update 3,000 against update 30,000 (level >= 2)\n')
    rows = []
    for c in CELLS:
        for a in ARMS:
            both, inv, ex = [], [], []
            for s in range(3):
                A = json.load(open(res / a / f'{c}_r32_s{s}' / 'analysis.json'))
                H = A['metrics']['3000']['pooled']['l2+']
                both.append(H['m4_emp_both_proj']['R2'])
                inv.append(H['m3']['invalid_q_ratio'])
                ex.append(next(h['excess'] for h in A['hist'] if h['step'] == 3000))
            rows.append((f'`{c}`', a, cell(ex, 3), cell(both), cell(seeds(a, c, 'm4_both')), cell(inv), cell(seeds(a, c, 'm3_invalid_q'))))
    table(['grammar', 'arm', 'prefix excess at 3,000', 'both-informative R2 at 3,000', 'at 30,000', 'invalid / valid read-out mass at 3,000',
           'at 30,000'], rows)

    allr = lambda a, k, src=None: [(src or R)[a][f'{c}_r32_s{s}'][k] for c in CELLS for s in range(3)]
    print('ranges over the 12 runs of an arm')
    for a in ARMS:
        print(f"  {a}: both {min(allr(a, 'm4_both')):.2f}-{max(allr(a, 'm4_both')):.2f}, emp {min(allr(a, 'm4_emp')):.2f}-"
              f"{max(allr(a, 'm4_emp')):.2f}, inv_q {min(allr(a, 'm3_invalid_q')):.2f}-{max(allr(a, 'm3_invalid_q')):.2f}, "
              f"excess {min(allr(a, 'excess')):.4f}-{max(allr(a, 'excess')):.4f}, vexcess {min(allr(a, 'vexcess', V)):.3f}-"
              f"{max(allr(a, 'vexcess', V)):.3f}, both > ceiling in {sum(b > c for b, c in zip(allr(a, 'm4_both'), allr(a, 'm4_both_ceil')))}/12")
    d = [m - p for p, m in zip(allr('prefix', 'm4_both'), allr('mixed', 'm4_both'))]
    print(f'  mixed - prefix on both: median {statistics.median(d):+.2f}, range {min(d):+.2f} to {max(d):+.2f}, higher in {sum(x > 0 for x in d)}/12')
    d = [m - p for p, m in zip(allr('prefix', 'm4_both'), allr('varied', 'm4_both'))]
    print(f'  varied - prefix on both: median {statistics.median(d):+.2f}, range {min(d):+.2f} to {max(d):+.2f}, higher in {sum(x > 0 for x in d)}/12')
    rep = S['prefix_vs_rhm_ttn_messages']
    for k in ('excess', 'm4_both', 'm3_invalid_q'):
        print(f'  prefix arm against rhm-ttn-messages, largest |difference| in {k}: {max(abs(a - b) for a, b in (v[k] for v in rep.values())):.4f}')


if __name__ == '__main__':
    main()
