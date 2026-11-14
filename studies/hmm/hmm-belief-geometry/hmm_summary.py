"""Apply the success criteria of EXPERIMENT.md to results/*/analysis.json; print tables, write summary.json.

    PYTHONPATH=. python studies/hmm/hmm-belief-geometry/hmm_summary.py RESULTS_DIR [N_PLANNED]

All numbers are at the final checkpoint, on held-out prefixes, pooled over a position group (hmm_analyze.py):
e = eligible for the model's primary set; nt = e with next-token ceiling < 0.9; tree: h1 = e in the first half,
h2b = second half eligible for the bilinear set, h2c = h2b with an exact additive ceiling below 0.9, h2t = h2b with a
token-product readout. A (process, model) cell is judged only if all N_SEEDS seeds finished with status 'finished';
otherwise it is censored. `scr` is the same probe's R2 on the scrambled target.

Comparisons with controls use an error ratio, not a fixed R2 margin: the probe beats a control if its unexplained
variance is at most RATIO times the control's, (1 - R2_probe) <= RATIO (1 - R2_control). On mess3 the belief is
nearly linear in the one-hot prefix (token readout R2 0.97 in the pilot) and nearly additive in the two halves, so a
margin of 0.1 would be unreachable there however good the probe is.

Learned: excess loss < min(EXCESS_MAX, half the excess of the optimal predictor from the last 2 tokens of that
process), in all seeds. A fixed 0.02 would pass a model that uses only the last two tokens on mess3, factored and
coupled (its excess is about 0.015), which has not learned to filter. The excess of the optimal predictor from the
last 3 tokens is printed next to it.
Saturated control: if the initial network or the token readout already scores >= SAT on a criterion's positions in
some seed, halving its error is asking for less than the probe's own noise, so the cell is "not testable (control
saturated)" for that hypothesis and leaves the count. SAT is above the 0.9 used for the ceilings because mess3's
controls sit at 0.96-0.975, where a trained probe can still be told apart (0.998 in the pilot).
Next-token clause (H1 on rrxor, H2, H3b): where the next-token ceiling is below 0.9, the probe must beat it too;
elsewhere the belief is a function of the model's own prediction target and the cell cannot show more than that.
Form of the chain's code (a label, not a verdict): "projective" if the projective probe beats the affine one on e,
"affine" otherwise."""
import json
import sys
from collections import defaultdict
from pathlib import Path

R2_MIN, RATIO, EXCESS_MAX, SHUFFLE_MAX, N_SEEDS, SAT = 0.8, 0.5, 0.02, 0.05, 3, 0.99


def beats(probe, *controls):
    return all((1 - probe) <= RATIO * (1 - c) for c in controls)
H1_PROCESSES = ('mess3', 'rrxor')


def main():
    res_dir = Path(sys.argv[1])
    planned = int(sys.argv[2]) if len(sys.argv) > 2 else None
    runs = {d.name: json.load(open(d / 'analysis.json')) for d in sorted(res_dir.iterdir()) if (d / 'analysis.json').exists()}
    by, rows, shuffle_bad = defaultdict(list), [], []
    for name, a in runs.items():
        cks = sorted(int(c) for c in a['metrics'])
        fin, ini = a['metrics'][str(cks[-1])], a['metrics'][str(cks[0])]
        P, P0 = fin['pooled'], ini['pooled']
        g = lambda pooled, k, grp: pooled.get(k + '@prefix', {}).get('R2_' + grp)
        if P['shuffle']['R2'] > SHUFFLE_MAX:
            shuffle_bad.append(name)
        E, kind, proc = a['eligible'], a['cfg']['model'], a['cfg']['process']
        arm = a['cfg'].get('arm', kind)
        g2, g3 = a['context_excess']['2'], a['context_excess']['3']
        gate = min(EXCESS_MAX, 0.5 * g2)
        x = dict(run=name, process=proc, model=kind, arm=arm, excess=a['final_excess'], status=a['status'],
                 ctx2=g2, ctx3=g3, gate=gate,
                 learned=a['final_excess'] < gate, shuffle=P['shuffle']['R2'], n=len(E['e']), n_nt=len(E['nt']),
                 n_beliefs=[a['geom'][str(t)]['n_unique'] for t in E['e']])
        prim = a['primary']
        grp = 'h1' if kind == 'tree' else 'e'
        x.update(grp=grp, n_grp=len(E[grp]), aff=g(P, prim, grp), proj=g(P, prim + '^proj', grp),
                 init_aff=g(P0, prim, grp), init_proj=g(P0, prim + '^proj', grp), tokens=g(P, 'tokens', grp),
                 nexttok=g(P, 'nexttok', grp), scr=g(P, prim + '~scr', grp),
                 nt_aff=g(P, prim, 'nt'), nt_proj=g(P, prim + '^proj', 'nt'), nt_ceiling=g(P, 'nexttok', 'nt'),
                 marg_aff=g(P, 'marg|' + prim, grp), marg_proj=g(P, 'marg|' + prim + '^proj', grp),
                 marg_tokens=g(P, 'marg|tokens', grp))
        pool = fin.get('pool', {})
        x.update(onemap=pool.get(prim + '@prefix', {}).get('R2'), onemap_proj=pool.get(prim + '^proj@prefix', {}).get('R2'),
                 onemap_seq=pool.get(prim, {}).get('R2'), onemap_init=ini.get('pool', {}).get(prim + '@prefix', {}).get('R2'))
        if kind == 'transformer' and x['n_grp']:
            ok = x['aff'] >= R2_MIN and beats(x['aff'], x['init_aff'], x['tokens'])
            if proc == 'rrxor':
                ok = ok and x['n_nt'] > 0 and beats(x['nt_aff'], x['nt_ceiling'])
            x['h1'], x['h1_sat'] = bool(ok), max(x['init_aff'], x['tokens']) >= SAT
        if kind == 'chain' and x['n_grp']:
            ok = x['proj'] >= R2_MIN and beats(x['proj'], x['init_proj'], x['tokens'])
            if x['n_nt']:
                ok = ok and beats(x['nt_proj'], x['nt_ceiling'])
            x['h2'], x['h2_sat'] = bool(ok), max(x['init_proj'], x['tokens']) >= SAT
            x['form'] = 'projective' if beats(x['proj'], x['aff']) else 'affine'
        if kind == 'tree':
            x.update(n_h2b=len(E['h2b']), n_h2c=len(E['h2c']), n_h2t=len(E['h2t']), n_h2n=len(E['h2n']),
                     n_bil_proj=g(P, 'top_bil^proj', 'h2n'), n_ceiling=g(P, 'nexttok', 'h2n'),
                     b_top=g(P, 'top', 'h2b'), b_top_proj=g(P, 'top^proj', 'h2b'), b_bil=g(P, 'top_bil', 'h2b'),
                     b_bil_proj=g(P, 'top_bil^proj', 'h2b'), b_init=g(P0, 'top_bil^proj', 'h2b'),
                     b_tokens=g(P, 'tokens', 'h2b'), b_scr=g(P, 'top_bil~scr', 'h2b'), b_addceil=g(P, 'addceil', 'h2b'),
                     c_bil_proj=g(P, 'top_bil^proj', 'h2c'), c_addceil=g(P, 'addceil', 'h2c'),
                     t_bil_proj=g(P, 'top_bil^proj', 'h2t'), t_tokprod=g(P, 'tokprod', 'h2t'),
                     b_marg=g(P, 'marg|top_bil^proj', 'h2b'), b_marg_tokens=g(P, 'marg|tokens', 'h2b'))
            if x['n_grp']:
                x['h3a'] = bool(x['proj'] >= R2_MIN and beats(x['proj'], x['init_proj'], x['tokens']))
                x['h3a_sat'] = max(x['init_proj'], x['tokens']) >= SAT
            if x['n_h2b'] and x['n_h2c']:
                ok = (x['b_bil_proj'] >= R2_MIN and beats(x['c_bil_proj'], x['c_addceil'])
                      and beats(x['b_bil_proj'], x['b_init'], x['b_tokens'])
                      and beats(x['b_bil_proj'], x['b_top_proj']))  # the product adds to the top bonds alone
                if x['n_h2t']:
                    ok = ok and beats(x['t_bil_proj'], x['t_tokprod'])
                if x['n_h2n']:
                    ok = ok and beats(x['n_bil_proj'], x['n_ceiling'])
                x['h3b'], x['h3b_sat'] = bool(ok), max(x['b_init'], x['b_tokens']) >= SAT
        rows.append(x)
        by[(proc, arm)].append(x)
    cells, verdicts = {}, {}
    for (proc, arm), rr in by.items():
        kind = rr[0]['model']
        censored = len(rr) < N_SEEDS or any(x['status'] != 'finished' for x in rr)
        learned = (not censored) and all(x['learned'] for x in rr)
        c = dict(process=proc, model=kind, arm=arm, n_runs=len(rr), censored=censored, learned=learned)
        for h in ('h1', 'h2', 'h3a', 'h3b'):
            c[h] = all(x[h] for x in rr) if learned and all(h in x for x in rr) else None
            if c[h] is not None and any(x[h + '_sat'] for x in rr):
                c[h], c[h + '_note'] = None, 'not testable (control saturated)'
        if kind == 'transformer' and proc not in H1_PROCESSES:
            c['h1'] = None  # the reproduction claim is about mess3 and rrxor; the other processes are descriptive
            c.pop('h1_note', None)
        if kind == 'chain':
            forms = {x.get('form') for x in rr}
            c['form'] = forms.pop() if len(forms) == 1 else 'mixed'
        cells[f'{proc}_{arm}'] = c
    for h in ('h1', 'h2', 'h3a', 'h3b'):
        judged = [k for k, v in cells.items() if v[h] is not None]
        sup = sum(cells[k][h] for k in judged)
        if not judged:
            v = 'inconclusive (no learned, testable cell)'
        elif sup >= 0.75 * len(judged):
            v = 'supported'
        elif sup <= 0.25 * len(judged):
            v = 'not supported'
        else:
            v = 'partly supported'
        verdicts[h] = dict(verdict=v, supported=sup, judged=judged)
    failed = []
    if shuffle_bad:
        failed.append(f'shuffled-label R2 > {SHUFFLE_MAX} in {shuffle_bad}')
    if planned and len(runs) < 0.85 * planned:
        failed.append(f'only {len(runs)} of {planned} runs analyzed')
    json.dump(dict(verdicts=verdicts, cells=cells, runs=rows, run_failed=failed, n_runs=len(runs)),
              open(res_dir / 'summary.json', 'w'), indent=1)
    f = lambda v: '   -  ' if v is None else f'{max(v, -9.99):6.3f}'
    yn = lambda x, k: 'Y' if x.get(k) else ('n' if k in x else '-')
    print('Primary probe on held-out prefixes (transformer: L4 on e; chain: state on e; tree: top on the first half h1)')
    print(f"{'run':26s} excess (gate; last-2, last-3 tokens)  #pos affine  proj  init-a init-p tokens ceil   scr  | nt: #  affine proj  ceil | one map: affine proj  init | verdict")
    for x in rows:
        v = ' '.join(f'{h.upper()}={yn(x, h)}' + ('(sat)' if x.get(h + '_sat') else '') for h in ('h1', 'h2', 'h3a') if h in x)
        v += f" form={x['form']}" if 'form' in x else ''
        print(f"{x['run']:26s} {x['excess']:.4f} ({x['gate']:.4f}; {x['ctx2']:.4f}, {x['ctx3']:.4f}){'' if x['learned'] else ' NOT LEARNED':12s} {x['n_grp']:3d} {f(x['aff'])}{f(x['proj'])}{f(x['init_aff'])}{f(x['init_proj'])}"
              f"{f(x['tokens'])}{f(x['nexttok'])}{f(x['scr'])} | {x['n_nt']:5d} {f(x['nt_aff'])}{f(x['nt_proj'])}{f(x['nt_ceiling'])} |"
              f"          {f(x['onemap'])}{f(x['onemap_proj'])}{f(x['onemap_init'])} | {v}")
    print('\nTree, second half (h2b; then h2c = exact additive ceiling available and below 0.9; then h2t = token-product readout eligible)')
    print(f"{'run':26s} #h2b top   top^p  bil   bil^p init  tokens  scr  addceil | #h2c bil^p addceil | #h2t bil^p tokprod | #h2n bil^p nexttok | H3b")
    for x in rows:
        if x['model'] == 'tree':
            print(f"{x['run']:26s} {x['n_h2b']:4d} {f(x['b_top'])}{f(x['b_top_proj'])}{f(x['b_bil'])}{f(x['b_bil_proj'])}{f(x['b_init'])}"
                  f"{f(x['b_tokens'])}{f(x['b_scr'])}{f(x['b_addceil'])} | {x['n_h2c']:4d} {f(x['c_bil_proj'])}{f(x['c_addceil'])} | "
                  f"{x['n_h2t']:4d} {f(x['t_bil_proj'])}{f(x['t_tokprod'])} | {x['n_h2n']:4d} {f(x['n_bil_proj'])}{f(x['n_ceiling'])} | "
                  f"{yn(x, 'h3b')}{'(sat)' if x.get('h3b_sat') else ''}")
    print('\nProduct processes (descriptive): joint belief against the two factor marginals, same probe and positions')
    print(f"{'run':26s} joint: affine proj  tokens | marginals: affine proj  tokens | tree 2nd half bil^p: joint  marg")
    for x in rows:
        if x['marg_aff'] is not None:
            print(f"{x['run']:26s}        {f(x['aff'])}{f(x['proj'])}{f(x['tokens'])} |            {f(x['marg_aff'])}{f(x['marg_proj'])}"
                  f"{f(x['marg_tokens'])} |                      {f(x.get('b_bil_proj'))}{f(x.get('b_marg'))}")
    print('\ndistinct beliefs per pooled position (for reading scr): ')
    for x in rows:
        if x['run'].endswith('_s0'):
            print(f"  {x['run']:26s} {x['n_beliefs']}")
    print()
    names = dict(h1='H1 transformer reproduction', h2='H2 chain: the state determines the belief up to normalization',
                 h3a='H3a tree, first half', h3b='H3b tree, second half: the product of the top bonds')
    for k, v in verdicts.items():
        print(f"{names[k]}: {v['verdict']}: {v['supported']} of {len(v['judged'])} judged cells ({', '.join(v['judged'])})")
    for h in names:
        sat = [k for k, v in cells.items() if v.get(h + '_note')]
        if sat:
            print(f"  {names[h]}: not testable, control saturated (>= {SAT}): {', '.join(sat)}")
    print("form of the chain's code (projective if the projective probe beats the affine one): "
          + (', '.join(f"{k}: {v['form']}" for k, v in cells.items() if 'form' in v) or 'no chain cell'))
    print('censored (a seed missing or not finished): ' + (', '.join(k for k, v in cells.items() if v['censored']) or 'none'))
    print('not learned (excess loss >= min(0.02, half the excess of the optimal last-2-token predictor) in some seed): '
          + (', '.join(k for k, v in cells.items() if not v['censored'] and not v['learned']) or 'none'))
    print('run failed: ' + ('; '.join(failed) if failed else 'no'))


if __name__ == '__main__':
    main()
