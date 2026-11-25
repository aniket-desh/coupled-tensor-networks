"""Build notes/child-feature-enabling/explorer.html from template.html, planted.js and the saved intervention results.

  python notes/child-feature-enabling/explorer/build.py [--fragment PATH]

explorer.html is a self-contained page (open it in a browser). --fragment also writes the same page without the
document wrapper, the form an Artifact publish expects. The depth-four curves are every second saved evaluation
(every 50 updates), to five significant digits; the half-fit times are the saved ones.
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / 'studies/rhm/child_feature_enabling/results'
PROTOCOLS = (('intervention-extended', 'rate 0.3'), ('intervention-extended-lr0.1', 'rate 0.1'),
             ('intervention', 'as written'))
KEEP = ('D_h', 'A_h', 'D_e', 'A_e')


def short(values):
    return [float(f'{v:.5g}') for v in values]


def protocol(folder, label):
    files = sorted((RESULTS / folder / 'pairs').glob('g*_s*.json'))
    if not files:
        return None
    pairs = []
    for f in files:
        p = json.loads(f.read_text())
        row = dict(g=p['grammar'], s=p['seed'])
        for arm, rec in p['arms'].items():
            cols = rec['columns']
            pick = sorted(set(range(0, len(cols['step']), 2)) | {len(cols['step']) - 1})
            row[arm] = dict(T=rec['crossing']['T'], last=cols['step'][-1], step=[cols['step'][i] for i in pick],
                            **{k: short(cols[k][i] for i in pick) for k in KEEP})
        pairs.append(row)
    lr, horizon = p['lr'], p['steps']
    ratios = [q['detached']['T'] / q['intact']['T'] for q in pairs if q['detached']['T'] and q['intact']['T']]
    censored = sum(q[a]['T'] is None for q in pairs for a in ('intact', 'detached'))
    note = f"{len(pairs)} pairs at SGD rate {lr:g}, horizon {horizon:,} updates."
    if ratios:
        note += (f" The detached arm takes {statistics.median(ratios):.2f} times as long at the median"
                 f" (range {min(ratios):.2f} to {max(ratios):.2f}, {len(ratios)} pairs).")
    if censored:
        note += f" {censored} of {2 * len(pairs)} arms never reach 1/2 within the horizon."
    return dict(key=folder, label=label, lr=lr, note=note, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fragment')
    a = parser.parse_args()
    data = dict(protocols=[p for p in (protocol(*x) for x in PROTOCOLS) if p is not None])
    core = (HERE / 'planted.js').read_text().split("if (typeof module")[0]
    page = (HERE / 'template.html').read_text()
    assert page.count('/*__PLANTED_JS__*/') == 1 and page.count('/*__DATA_JSON__*/') == 1
    page = page.replace('/*__PLANTED_JS__*/', core).replace('/*__DATA_JSON__*/', json.dumps(data, separators=(',', ':')))
    wrapper = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    out = HERE.parent / 'explorer.html'
    out.write_text(wrapper + page)
    print('wrote', out, f'{out.stat().st_size / 1e3:.0f} kB')
    if a.fragment:
        Path(a.fragment).write_text(page)
        print('wrote', a.fragment)


if __name__ == '__main__':
    main()
