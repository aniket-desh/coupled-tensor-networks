"""Cell-level tables for the write-up, from results/summary.json (written by ttn_summary.py) and results/*/analysis.json.

    python studies/rhm/rhm-ttn-belief-geometry/ttn_table.py RESULTS_DIR

Each entry is the median over the three seeds with the range in brackets. Columns as in ttn_summary.py: h1 = first-half
positions eligible for the affine top probe, h2b = second-half positions eligible for the bilinear probe, h2c = the h2b
positions whose additive ceiling is below 0.8. The last table is exploratory (made after seeing the results): per
first-half position of seed 0, the affine probe's R2 next to the number of distinct beliefs and the number of principal
components holding 99% of the belief variance (k99), against the r coordinates of h_L."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

res = Path(sys.argv[1])
S = json.load(open(res / 'summary.json'))
by = defaultdict(list)
for x in S['runs']:
    by[x['run'].rsplit('_s', 1)[0]].append(x)
order = sorted(by, key=lambda c: (-by[c][0]['r'], c))


def mr(rows, k, fmt='.2f'):
    v = [x.get(k) for x in rows]
    if any(a is None for a in v):
        return '-'
    return f'{median(v):{fmt}} [{min(v):{fmt}}, {max(v):{fmt}}]'


def verdict(c, h):
    v = S['cells'][c]
    if not v['learned']:
        return 'not learned'
    return {True: 'yes', False: 'no', None: 'not testable'}[v[h]]


print('shuffled-label pooled R2 over all runs: min %.4f max %.4f (n = %d)' % (
    min(x['shuffle'] for x in S['runs']), max(x['shuffle'] for x in S['runs']), len(S['runs'])))
print('excess loss over all runs: max %.4f; runs with excess >= 0.02: %s' % (
    max(x['excess'] for x in S['runs']), [x['run'] for x in S['runs'] if x['excess'] >= 0.02]))
print('\nH1 | cell | excess | #pos | top | init | tokens | nexttok ceiling | scrambled | selectivity | H1')
for c in order:
    rr = by[c]
    print(f"| {c} | {mr(rr, 'excess', '.4f')} | {sorted({x['n_h1'] for x in rr})} | {mr(rr, 'h1_top')} | {mr(rr, 'h1_init')} | "
          f"{mr(rr, 'h1_tokens')} | {mr(rr, 'h1_nexttok')} | {mr(rr, 'h1_scr')} | {mr(rr, 'h1_selectivity')} | {verdict(c, 'h1')} |")
print('\nH2 | cell | #h2b | affine | bilinear | init bilinear | tokens | scrambled | selectivity | #h2c | bilinear h2c | addceil h2c | '
      'bilinear h2t | tokprod h2t | H2')
for c in order:
    rr = by[c]
    print(f"| {c} | {sorted({x['n_h2b'] for x in rr})} | {mr(rr, 'h2_top')} | {mr(rr, 'h2_bil')} | {mr(rr, 'h2_init_bil')} | "
          f"{mr(rr, 'h2_tokens')} | {mr(rr, 'h2_scr')} | {mr(rr, 'h2_selectivity')} | {sorted({x['n_h2c'] for x in rr})} | "
          f"{mr(rr, 'h2c_bil')} | {mr(rr, 'h2c_addceil')} | {mr(rr, 'h2t_bil')} | {mr(rr, 'h2t_tokprod')} | {verdict(c, 'h2')} |")
print('\nbelief split | cell | #pos | top | bilinear | span ceiling | init top | init bilinear')
for c in order:
    rr = by[c]
    print(f"| {c} | {sorted({x['n_bel'] for x in rr})} | {mr(rr, 'bel_top')} | {mr(rr, 'bel_bil')} | {mr(rr, 'bel_span')} | "
          f"{mr(rr, 'bel_init_top')} | {mr(rr, 'bel_init_bil')} |")
print('\nexploratory: first-half positions of seed 0 | run | t | distinct beliefs | belief rank | k99 | R2 top | R2 scrambled | nexttok')
for c in order:
    a = json.load(open(res / f'{c}_s0' / 'analysis.json'))
    P = a['metrics'][str(max(int(k) for k in a['metrics']))]['per_pos']
    for t in a['eligible']['h1']:
        g = a['geom'][str(t)]
        print(f"| {c}_s0 | {t} | {g['n_unique']} | {g['rank']} | {g['k99']} | {P['top@prefix'][str(t)]['R2']:.2f} | "
              f"{P['top~scr@prefix'][str(t)]['R2']:.2f} | {P['nexttok@prefix'][str(t)]['R2']:.2f} |")
