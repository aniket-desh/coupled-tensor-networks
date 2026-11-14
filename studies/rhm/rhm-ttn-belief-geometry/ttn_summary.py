"""Apply the success criteria of EXPERIMENT.md to results/*/analysis.json; print tables, write summary.json.

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/ttn_summary.py RESULTS_DIR [N_PLANNED]

All numbers are at the final checkpoint. Position groups (ttn_analyze.py): h1 = first half, eligible for the affine top
probe; h2b = second half, eligible for the bilinear probe; h2c = the h2b positions whose additive ceiling is below 0.8;
h2t = the h2b positions where the token-product readout is eligible; bel = positions of the belief split.
A (grammar, width) cell is judged only if all N_SEEDS seeds finished training with status 'finished'; otherwise it is
censored, and censored cells are listed separately from cells that did not learn."""
import json
import sys
from collections import defaultdict
from pathlib import Path

R2_MIN, MARGIN, EXCESS_MAX, SHUFFLE_MAX, N_SEEDS = 0.8, 0.1, 0.02, 0.05, 3


def main():
    res_dir = Path(sys.argv[1])
    planned = int(sys.argv[2]) if len(sys.argv) > 2 else None
    runs = {d.name: json.load(open(d / 'analysis.json')) for d in sorted(res_dir.iterdir()) if (d / 'analysis.json').exists()}
    by, rows, shuffle_bad = defaultdict(list), [], []
    for name, a in runs.items():
        cks = sorted(int(c) for c in a['metrics'])
        P, P0 = a['metrics'][str(cks[-1])]['pooled'], a['metrics'][str(cks[0])]['pooled']
        get = lambda pooled, k, f: pooled.get(k, {}).get(f)
        if P['shuffle']['R2'] > SHUFFLE_MAX:
            shuffle_bad.append(name)
        E = a['eligible']
        x = dict(run=name, r=a['cfg']['r'], excess=a['final_excess'], status=a['status'],
                 learned=a['final_excess'] < EXCESS_MAX, shuffle=P['shuffle']['R2'],
                 n_h1=len(E['h1']), n_h2b=len(E['h2b']), n_h2c=len(E['h2c']), n_h2t=len(E['h2t']), n_bel=len(E['bel']),
                 n_beliefs_h1=[a['geom'][str(t)]['n_unique'] for t in E['h1']],
                 n_beliefs_h2b=[a['geom'][str(t)]['n_unique'] for t in E['h2b']],
                 # H1: first half, affine
                 h1_top=get(P, 'top@prefix', 'R2_h1'), h1_init=get(P0, 'top@prefix', 'R2_h1'),
                 h1_tokens=get(P, 'tokens@prefix', 'R2_h1'), h1_nexttok=get(P, 'nexttok@prefix', 'R2_h1'),
                 h1_scr=get(P, 'top~scr@prefix', 'R2_h1'),
                 # H2: second half, bilinear
                 h2_top=get(P, 'top@prefix', 'R2_h2b'), h2_bil=get(P, 'top_bil@prefix', 'R2_h2b'),
                 h2_all=get(P, 'all@prefix', 'R2_h2b'), h2_addceil=get(P, 'addceil@prefix', 'R2_h2b'),
                 h2_init_bil=get(P0, 'top_bil@prefix', 'R2_h2b'), h2_tokens=get(P, 'tokens@prefix', 'R2_h2b'),
                 h2_nexttok=get(P, 'nexttok@prefix', 'R2_h2b'), h2_scr=get(P, 'top_bil~scr@prefix', 'R2_h2b'),
                 h2c_bil=get(P, 'top_bil@prefix', 'R2_h2c'), h2c_addceil=get(P, 'addceil@prefix', 'R2_h2c'),
                 h2c_top=get(P, 'top@prefix', 'R2_h2c'),
                 h2t_bil=get(P, 'top_bil@prefix', 'R2_h2t'), h2t_tokprod=get(P, 'tokprod@prefix', 'R2_h2t'),
                 # belief split (descriptive)
                 bel_top=get(P, 'top@belief', 'R2_bel'), bel_bil=get(P, 'top_bil@belief', 'R2_bel'),
                 bel_all=get(P, 'all@belief', 'R2_bel'), bel_span=get(P, 'span@belief', 'R2_bel'),
                 bel_init_top=get(P0, 'top@belief', 'R2_bel'), bel_init_bil=get(P0, 'top_bil@belief', 'R2_bel'))
        if x['n_h1']:
            x['h1'] = bool(x['h1_top'] >= R2_MIN and x['h1_top'] - max(x['h1_init'], x['h1_tokens']) >= MARGIN)
            x['h1_selectivity'] = x['h1_top'] - x['h1_scr']
        if x['n_h2b'] and x['n_h2c']:
            base = max([x['h2_init_bil'], x['h2_tokens']])
            ok = x['h2_bil'] >= R2_MIN and x['h2c_bil'] - x['h2c_addceil'] >= MARGIN and x['h2_bil'] - base >= MARGIN
            if x['n_h2t']:
                ok = ok and x['h2t_bil'] - x['h2t_tokprod'] >= MARGIN
            x['h2'] = bool(ok)
        if x['n_h2b']:
            x['h2_selectivity'] = x['h2_bil'] - x['h2_scr']
        rows.append(x)
        by[name.rsplit('_s', 1)[0]].append(x)
    cells, verdicts = {}, {}
    for cell, rr in by.items():
        censored = len(rr) < N_SEEDS or any(x['status'] != 'finished' for x in rr)
        learned = (not censored) and all(x['learned'] for x in rr)
        cells[cell] = dict(r=rr[0]['r'], n_runs=len(rr), censored=censored, learned=learned,
                           h1=(all(x['h1'] for x in rr) if learned and all('h1' in x for x in rr) else None),
                           h2=(all(x['h2'] for x in rr) if learned and all('h2' in x for x in rr) else None))
    for h in ('h1', 'h2'):
        for r in sorted({c['r'] for c in cells.values()}):
            judged = [c for c, v in cells.items() if v['r'] == r and v[h] is not None]
            sup = sum(cells[c][h] for c in judged)
            if len(judged) < 3:
                v = 'inconclusive (fewer than 3 learned, testable grammars)'
            elif sup >= 0.75 * len(judged):
                v = 'supported'
            elif sup <= 0.25 * len(judged):
                v = 'not supported'
            else:
                v = 'partly supported'
            verdicts[f'{h}_r{r}'] = dict(verdict=v, supported=sup, judged=judged)
    failed = []
    if shuffle_bad:
        failed.append(f'shuffled-label R2 > {SHUFFLE_MAX} in {shuffle_bad}')
    if planned and len(runs) < 0.85 * planned:
        failed.append(f'only {len(runs)} of {planned} runs analyzed')
    json.dump(dict(verdicts=verdicts, cells=cells, runs=rows, run_failed=failed, n_runs=len(runs)),
              open(res_dir / 'summary.json', 'w'), indent=1)
    f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
    yn = lambda x, k: 'Y' if x.get(k) else ('n' if k in x else '-')
    print('H1, first half, affine probe on the top bonds (held-out prefixes; scr = the same probe on the scrambled target)')
    print(f"{'run':22s} excess  #pos  top   init  tokens ceil   scr  H1  distinct beliefs per position")
    for x in rows:
        print(f"{x['run']:22s} {x['excess']:.4f} {x['n_h1']:4d} {f(x['h1_top'])}{f(x['h1_init'])}{f(x['h1_tokens'])}"
              f"{f(x['h1_nexttok'])}{f(x['h1_scr'])}  {yn(x, 'h1')}  {x['n_beliefs_h1']}")
    print('\nH2, second half, bilinear probe on the top bonds (h2b positions; then the h2c positions with additive ceiling < 0.8;'
          ' then the h2t positions with a token-product readout)')
    print(f"{'run':22s} #h2b affine bilin  all  addceil initB tokens  scr | #h2c bilin addceil affine | #h2t bilin tokprod | H2")
    for x in rows:
        print(f"{x['run']:22s} {x['n_h2b']:4d} {f(x['h2_top'])}{f(x['h2_bil'])}{f(x['h2_all'])}{f(x['h2_addceil'])}"
              f"{f(x['h2_init_bil'])}{f(x['h2_tokens'])}{f(x['h2_scr'])} | {x['n_h2c']:4d} {f(x['h2c_bil'])}{f(x['h2c_addceil'])}"
              f"{f(x['h2c_top'])} | {x['n_h2t']:4d} {f(x['h2t_bil'])}{f(x['h2t_tokprod'])} | {yn(x, 'h2')}")
    print('\nBelief split (descriptive): probes scored on distinct beliefs they never saw; span = the ceiling')
    print(f"{'run':22s} #pos  top   bilin  all   span  init-top init-bil")
    for x in rows:
        print(f"{x['run']:22s} {x['n_bel']:4d} {f(x['bel_top'])}{f(x['bel_bil'])}{f(x['bel_all'])}{f(x['bel_span'])}"
              f"{f(x['bel_init_top'])}  {f(x['bel_init_bil'])}")
    print()
    for k, v in verdicts.items():
        print(f"{k}: {v['verdict']}: {v['supported']} of {len(v['judged'])} learned, testable grammars ({', '.join(v['judged'])})")
    print('censored (a seed missing or not finished): ' + (', '.join(c for c, v in cells.items() if v['censored']) or 'none'))
    print('not learned (excess loss >= 0.02 in some seed): '
          + (', '.join(c for c, v in cells.items() if not v['censored'] and not v['learned']) or 'none'))
    print('run failed: ' + ('; '.join(failed) if failed else 'no'))


if __name__ == '__main__':
    main()
