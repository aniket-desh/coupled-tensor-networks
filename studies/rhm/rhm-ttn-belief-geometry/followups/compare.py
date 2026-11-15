"""Follow-up figure: the transformer's and the tree tensor network's probes, position by position.

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/followups/compare.py

Reads the saved analyses of rhm-ttn-belief-geometry and rhm-belief-geometry (seed 0; the tree at r = 32) and draws,
for every grammar both experiments ran, R2 on held-out prefixes at each prefix length t: the transformer's last-layer
and all-layer affine probes, and the tree's affine and bilinear probes on its two top bonds. Both analyses use the
same 60,000 sequences and the same split seeds, so a position is scored on the same held-out prefixes in both.
Gray = positions not eligible for one of the two primary probes; dashed line = T/2. Writes compare.pdf and
compare.png here, and compare.json with the plotted numbers."""
import json
import sys
from pathlib import Path

import matplotlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ttn_plot as TP  # noqa: E402

matplotlib.rcParams['savefig.dpi'] = 150
ttn_res, tf_res = HERE.parent / 'results', HERE.parent.parent / 'rhm-belief-geometry' / 'results'
runs = TP.PL.load(ttn_res)
TP.compare(runs, tf_res, HERE / 'compare.pdf')
TP.compare(runs, tf_res, HERE / 'compare.png')

tf = TP.PL.load(tf_res)
out = {}
for name, a in sorted(runs.items()):
    if not name.endswith('_r32_s0'):
        continue
    g = name.rsplit('_r', 1)[0]
    if g + '_s0' not in tf:
        continue
    b = tf[g + '_s0']
    fa = a['metrics'][str(max(int(c) for c in a['metrics']))]['per_pos']
    fb = b['metrics'][str(max(int(c) for c in b['metrics']))]['per_pos']
    rows = []
    for t in a['positions']:
        t_ = str(t)
        if t_ in fa['top@prefix'] and t_ in fb.get('L4@prefix', {}):
            rows.append(dict(t=t, half='first' if t <= a['half'] else 'second',
                             eligible=bool(t in a['eligible']['e_top'] and t in b['eligible']['eL4']),
                             transformer_L4=fb['L4@prefix'][t_]['R2'], transformer_all=fb['concat@prefix'][t_]['R2'],
                             ttn_affine=fa['top@prefix'][t_]['R2'], ttn_bilinear=fa['top_bil@prefix'][t_]['R2']))
    out[g] = rows
json.dump(out, open(HERE / 'compare.json', 'w'), indent=1)
med = lambda v: sorted(v)[len(v) // 2] if v else None
print('grammar: eligible positions | medians over eligible positions (transformer L4, TTN affine, TTN bilinear), by half')
for g, rows in out.items():
    line = [g]
    for half in ('first', 'second'):
        e = [r for r in rows if r['eligible'] and r['half'] == half]
        f = lambda k: 'none' if not e else f"{med([r[k] for r in e]):.2f}"
        line.append(f"{half}: n={len(e)} L4 {f('transformer_L4')} affine {f('ttn_affine')} bilinear {f('ttn_bilinear')}")
    print(' | '.join(line))
