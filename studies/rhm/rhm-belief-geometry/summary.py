"""Apply the success criteria of EXPERIMENT.md to results/*/analysis.json; print a table, write summary.json.

    PYTHONPATH=. python studies/rhm/rhm-belief-geometry/summary.py RESULTS_DIR [N_PLANNED]

All numbers are at the final checkpoint, on held-out prefixes, pooled over the positions eligible for the
one-layer probe (>= 10 (d + 1) distinct fit prefixes; R2_eL4 in analyze.py)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

R2_MIN, MARGIN, EXCESS_MAX, SHUFFLE_MAX = 0.8, 0.1, 0.02, 0.05
N_SEEDS = 3  # a grammar is judged only if all its seeds finished; otherwise it is censored


def main():
    res_dir = Path(sys.argv[1])
    planned = int(sys.argv[2]) if len(sys.argv) > 2 else None
    runs = {d.name: json.load(open(d / 'analysis.json')) for d in sorted(res_dir.iterdir()) if (d / 'analysis.json').exists()}
    by = defaultdict(dict)
    rows, h2_cases, h2_beyond, shuffle_bad = [], 0, 0, []
    for r, a in runs.items():
        cks = sorted(int(c) for c in a['metrics'])
        fin, ini = a['metrics'][str(cks[-1])], a['metrics'][str(cks[0])]
        P, P0 = fin['pooled'], ini['pooled']
        if P['shuffle']['R2'] > SHUFFLE_MAX:
            shuffle_bad.append(r)
        row = dict(run=r, excess=a['final_excess'], learned=a['final_excess'] < EXCESS_MAX,
                   n_elig=len(a['eligible']['eL4']), seq_L4=P['L4']['R2'], seq_concat=P['concat']['R2'],
                   seq_random_concat=P0['concat']['R2'], shuffle=P['shuffle']['R2'])
        if row['n_elig']:
            row.update(L4=P['L4@prefix']['R2_eL4'], random=P0['L4@prefix']['R2_eL4'], tokens=P['tokens@prefix']['R2_eL4'],
                       nexttok=P['nexttok@prefix']['R2_eL4'], concat_eL4=P['concat@prefix']['R2_eL4'])
            row['margin'] = row['L4'] - max(row['random'], row['tokens'])
            row['h1'] = row['L4'] >= R2_MIN and row['margin'] >= MARGIN
            for t in a['eligible']['eL4']:
                c = fin['per_pos']['nexttok@prefix'][str(t)]['R2']
                if c < 0.9:
                    h2_cases += 1
                    h2_beyond += fin['per_pos']['L4@prefix'][str(t)]['R2'] > c + 0.05
        if a['eligible']['eC']:
            row['concat_eC'] = P['concat@prefix']['R2_eC']
            row['random_concat_eC'] = P0['concat@prefix']['R2_eC']
        rows.append(row)
        by[r.rsplit('_s', 1)[0]][r] = row
    grammars = {}
    for gname, rs in by.items():
        rr = list(rs.values())
        learned = len(rr) == N_SEEDS and all(x['learned'] for x in rr)
        testable = len(rr) == N_SEEDS and all(x['n_elig'] for x in rr)
        grammars[gname] = dict(learned=learned, testable=testable, n_runs=len(rr),
                               h1=(all(x['h1'] for x in rr) if learned and testable else None))
    tl = [g for g, v in grammars.items() if v['learned'] and v['testable']]
    sup = sum(grammars[g]['h1'] for g in tl)
    if len(tl) < 4:
        verdict = 'inconclusive (fewer than 4 learned, testable grammars)'
    elif sup >= 0.75 * len(tl):
        verdict = 'supported'
    elif sup <= 0.25 * len(tl):
        verdict = 'not supported'
    else:
        verdict = 'partly supported'
    failed = []
    if shuffle_bad:
        failed.append(f'shuffled-label R2 > {SHUFFLE_MAX} in {shuffle_bad}')
    if planned and len(runs) < 0.85 * planned:
        failed.append(f'only {len(runs)} of {planned} runs analyzed')
    out = dict(verdict=verdict, supported=sup, testable_learned=tl, grammars=grammars, runs=rows,
               h2=dict(cases=h2_cases, beyond=h2_beyond), run_failed=failed, n_runs=len(runs))
    json.dump(out, open(res_dir / 'summary.json', 'w'), indent=1)
    f = lambda x: '   -  ' if x is None else f'{x:6.3f}'
    print(f"{'run':16s} excess  #elig    L4  random tokens nexttok margin H1 | seqL4 seqC  seqRandC shuffle | concat_eC randC_eC")
    for x in rows:
        print(f"{x['run']:16s} {x['excess']:.4f} {x['n_elig']:5d} {f(x.get('L4'))} {f(x.get('random'))} {f(x.get('tokens'))} "
              f"{f(x.get('nexttok'))} {f(x.get('margin'))} {'Y' if x.get('h1') else ('n' if 'h1' in x else '-')}  | "
              f"{f(x['seq_L4'])} {f(x['seq_concat'])} {f(x['seq_random_concat'])} {f(x['shuffle'])} | "
              f"{f(x.get('concat_eC'))} {f(x.get('random_concat_eC'))}")
    print(f'\nH1: {verdict}: {sup} of {len(tl)} learned, testable grammars ({", ".join(tl)})')
    print(f'H2: {h2_beyond} of {h2_cases} eligible (grammar, seed, position) cases with next-token ceiling < 0.9 '
          f'have R2(L4) > ceiling + 0.05')
    print('run failed: ' + ('; '.join(failed) if failed else 'no'))


if __name__ == '__main__':
    main()
