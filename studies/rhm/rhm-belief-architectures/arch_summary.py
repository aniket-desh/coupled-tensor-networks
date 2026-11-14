"""Apply the success criteria of EXPERIMENT.md to results/*/analysis.json; print tables, write summary.json.

    PYTHONPATH=. python studies/rhm/rhm-belief-architectures/arch_summary.py RESULTS_DIR [N_PLANNED]

All numbers are on held-out prefixes, pooled over the position groups of arch_analyze.py (ea: eligible for the arm's
affine probe; eb: eligible for the trees' bilinear probe, the common set; eb_lo / eb_hi: eb with t <= T/2 / t > T/2;
pa: ea with D_t <= 64, where the chain's projective and softmax probes exist).

Censoring. Training runs grammar by grammar and a time cut loses whole grammars, so completeness is judged per grammar
and per hypothesis: a grammar enters H_B only if all six of its tree and perm runs have status 'finished' and an
analysis, and enters H_A (and the descriptive tree-against-chain comparison) only if all three chain runs do (and, for
the comparison, the three tree runs). Censored grammars are listed apart from cells that did not learn.

H_A (chain). Three probe families on the state h_t, each against its own controls (the identity-init chain and the
token readout under the same probe): A1 affine on ea; A2 projective and A3 softmax-of-affine on pa. A grammar supports
H_A if one family holds in all three seeds.

H_B (aligned against permuted tree), paired by (grammar, seed).
  learning        pairs where the aligned tree learned. gap = permuted excess - aligned excess; the aligned tree wins if
                  gap > tol, the permuted tree if gap < -tol, tol = max(0.001, the aligned tree's seed range of excess
                  on that grammar).
  representation  pairs where both trees learned and eb_hi is not empty. gap(ckpt) = R2(aligned top_bil) - R2(permuted
                  top_bil) on eb_hi; the pair goes to the aligned tree if gap(final) - gap(identity init) >= 0.1, to the
                  permuted tree if <= -0.1. Subtracting the init gap removes what the leaf order alone gives two
                  untrained trees. The same quantity on eb_lo and on all of eb is reported, with no verdict.
A measure goes to an arm if that arm wins at least WIN_FRAC of the usable pairs, with at least MIN_PAIRS usable pairs
(three grammars' worth), has more wins than losses in all but at most one of the grammars with usable pairs, and fewer
in none (the consistency rule of studies/rhm/analysis.py)."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

R2_MIN, MARGIN, EXCESS_MAX, SHUFFLE_MAX, N_SEEDS, FLOOR = 0.8, 0.1, 0.02, 0.05, 3, 1e-3
WIN_FRAC, MIN_PAIRS = 0.8, 9


def direction(pairs):
    """pairs: [(grammar, +1 reference wins | -1 other wins | 0 tie)] -> counts, sign test and who takes the measure
    ('ref', 'other', None for neither, 'untestable' below MIN_PAIRS usable pairs)."""
    w, l, ties = sum(s > 0 for _, s in pairs), sum(s < 0 for _, s in pairs), sum(s == 0 for _, s in pairs)
    by = defaultdict(int)
    for gname, s in pairs:
        by[gname] += s
    out = dict(ref_wins=w, other_wins=l, ties=ties, usable=len(pairs), by_grammar=dict(by),
               sign_p=binomtest(max(w, l), w + l, 0.5, alternative='greater').pvalue if w + l else None, to=None)
    if len(pairs) < MIN_PAIRS:
        out['to'] = 'untestable'
        return out
    for name, wins, sgn in (('ref', w, 1), ('other', l, -1)):
        if (wins >= WIN_FRAC * len(pairs) and sum(sgn * d > 0 for d in by.values()) >= len(by) - 1
                and not any(sgn * d < 0 for d in by.values())):
            out['to'] = name
    return out


def verdict_of(learn_to, rep_to, other):
    tos = [learn_to, rep_to]
    if 'other' in tos:
        return f'reversed on a measure: the {other} arm better'
    if tos == ['ref', 'ref']:
        return 'supported'
    if tos == ['untestable', 'untestable']:
        return f'inconclusive (fewer than {MIN_PAIRS} usable pairs on both measures)'
    if 'ref' in tos:
        which = 'learning' if learn_to == 'ref' else 'representation'
        rest = rep_to if learn_to == 'ref' else learn_to
        return f'partly supported ({which} only; the other measure: {"untestable" if rest == "untestable" else "to neither"})'
    return 'not supported'


def compare(cell, configs, complete, ref, other):
    """Paired comparison of arm `ref` against arm `other`: learning and representation, with the identity-init gap."""
    learn, rep, extra, pairs_out, censored, ref_unlearned = [], [], {'eb_lo': [], 'eb': []}, [], [], []
    for c in configs:
        if not (complete[(c, ref)] and complete[(c, other)]):
            censored.append(c)
            continue
        a_runs = {x['seed']: x for x in cell[(c, ref)]}
        b_runs = {x['seed']: x for x in cell[(c, other)]}
        ex = [x['excess'] for x in a_runs.values()]
        tol = max(FLOOR, max(ex) - min(ex))
        for s in range(N_SEEDS):
            a, b = a_runs[s], b_runs[s]
            row = dict(pair=f'{c}_s{s}', ref_excess=a['excess'], other_excess=b['excess'], tol=tol,
                       ref_learned=a['learned'], other_learned=b['learned'])
            if not a['learned']:
                ref_unlearned.append(f'{c}_s{s}')
                pairs_out.append(row)
                continue
            d = b['excess'] - a['excess']
            row.update(learning_gap=d, learning=1 if d > tol else -1 if d < -tol else 0)
            learn.append((c, row['learning']))
            for grp in ('eb_hi', 'eb_lo', 'eb'):
                vals = [a[grp + '_best'], b[grp + '_best'], a[grp + '_init'], b[grp + '_init']]
                if any(v is None for v in vals):
                    continue
                gap, gap0 = vals[0] - vals[1], vals[2] - vals[3]
                # a win needs the final gap on the winner's side as well, so equal final R2 is never a win
                sgn = 1 if gap - gap0 >= MARGIN and gap > 0 else -1 if gap - gap0 <= -MARGIN and gap < 0 else 0
                row[grp] = dict(ref_R2=vals[0], other_R2=vals[1], gap=gap, init_gap=gap0, adjusted=gap - gap0, sign=sgn)
                if b['learned']:
                    (rep if grp == 'eb_hi' else extra[grp]).append((c, sgn))
            pairs_out.append(row)
    L_, R_ = direction(learn), direction(rep)
    med = lambda key, sub=None: (lambda v: statistics.median(v) if v else None)(
        [(p[key][sub] if sub else p[key]) for p in pairs_out if key in p and (sub is None or p['other_learned'])])
    return dict(verdict=verdict_of(L_['to'], R_['to'], other), learning=L_, representation=R_,
                representation_eb_lo=direction(extra['eb_lo']), representation_eb=direction(extra['eb']),
                median_learning_gap=med('learning_gap'), median_adjusted_gap=med('eb_hi', 'adjusted'),
                censored_grammars=censored, reference_not_learned=ref_unlearned, pairs=pairs_out)


def main():
    res_dir = Path(sys.argv[1])
    planned = int(sys.argv[2]) if len(sys.argv) > 2 else None
    runs = {d.name: json.load(open(d / 'analysis.json')) for d in sorted(res_dir.iterdir()) if (d / 'analysis.json').exists()}
    rows, shuffle_bad = {}, []
    for name, a in runs.items():
        cks = sorted(int(c) for c in a['metrics'])
        P, P0 = a['metrics'][str(cks[-1])]['pooled'], a['metrics'][str(cks[0])]['pooled']
        get = lambda pooled, k, f: pooled.get(k, {}).get(f) if k else None
        if P['shuffle']['R2'] > SHUFFLE_MAX:
            shuffle_bad.append(name)
        E, aff, bil = a['eligible'], a['probes']['affine'], a['probes']['bilinear']
        best = bil or aff  # the arm's probe on the common set: bilinear for the trees, the state for the chain
        x = dict(run=name, config=a['cfg']['config'], arm=a['arm'], seed=a['cfg']['grammar']['seed'], excess=a['final_excess'],
                 status=a['status'], finished=a['status'] == 'finished', learned=a['final_excess'] < EXCESS_MAX,
                 n_params=a['n_params'], shuffle=P['shuffle']['R2'], n={k: len(E[k]) for k in E},
                 n_beliefs_ea=[a['geom'][str(t)]['n_unique'] for t in E['ea']],
                 # the arm's affine probe on its own eligible positions
                 ea_aff=get(P, aff + '@prefix', 'R2_ea'), ea_init=get(P0, aff + '@prefix', 'R2_ea'),
                 ea_tokens=get(P, 'tokens@prefix', 'R2_ea'), ea_nexttok=get(P, 'nexttok@prefix', 'R2_ea'),
                 ea_scr=get(P, aff + '~scr@prefix', 'R2_ea'),
                 # the chain's projective and softmax probes (pa)
                 pa_aff=get(P, aff + '@prefix', 'R2_pa'),
                 pa_proj=get(P, aff + '~proj@prefix', 'R2_pa'), pa_proj_init=get(P0, aff + '~proj@prefix', 'R2_pa'),
                 pa_proj_tokens=get(P, 'tokens~proj@prefix', 'R2_pa'),
                 pa_soft=get(P, aff + '~soft@prefix', 'R2_pa'), pa_soft_init=get(P0, aff + '~soft@prefix', 'R2_pa'),
                 pa_soft_tokens=get(P, 'tokens~soft@prefix', 'R2_pa'),
                 # the common set and its halves, final and identity init
                 eb_aff=get(P, aff + '@prefix', 'R2_eb'), eb_tokens=get(P, 'tokens@prefix', 'R2_eb'),
                 eb_nexttok=get(P, 'nexttok@prefix', 'R2_eb'), eb_scr=get(P, best + '~scr@prefix', 'R2_eb'),
                 eb_addup=get(P, 'addup@prefix', 'R2_eb'),
                 ebt_best=get(P, best + '@prefix', 'R2_ebt'), ebt_tokprod=get(P, 'tokprod@prefix', 'R2_ebt'),
                 bel_aff=get(P, aff + '@belief', 'R2_bel'), bel_bil=get(P, bil + '@belief', 'R2_bel') if bil else None,
                 bel_span=get(P, 'span@belief', 'R2_bel'), bel_init=get(P0, aff + '@belief', 'R2_bel'),
                 addceil_uncomputable=len(a.get('addceil_uncomputable', {})), tokprod_skipped=len(a.get('tokprod_skipped', [])))
        for grp in ('eb', 'eb_lo', 'eb_hi'):
            x[grp + '_best'], x[grp + '_init'] = get(P, best + '@prefix', 'R2_' + grp), get(P0, best + '@prefix', 'R2_' + grp)
        if x['arm'] == 'chain':
            margin = lambda v, *ctrl: v is not None and v >= R2_MIN and v - max(ctrl) >= MARGIN
            if x['n']['ea']:
                x['A1'] = bool(margin(x['ea_aff'], x['ea_init'], x['ea_tokens']))
            if x['n']['pa']:
                x['A2'] = bool(margin(x['pa_proj'], x['pa_proj_init'], x['pa_proj_tokens']))
                x['A3'] = bool(margin(x['pa_soft'], x['pa_soft_init'], x['pa_soft_tokens']))
        rows[name] = x
    cell = defaultdict(list)
    for x in rows.values():
        cell[(x['config'], x['arm'])].append(x)
    configs = sorted({c for c, _ in cell})
    whole = lambda c, arm: len(cell.get((c, arm), [])) == N_SEEDS and all(x['finished'] for x in cell[(c, arm)])
    complete = {(c, arm): whole(c, arm) for c in configs for arm in ('tree', 'perm', 'chain')}
    cells = {f'{c}_{arm}': dict(config=c, arm=arm, n_runs=len(cell.get((c, arm), [])), complete=complete[(c, arm)],
                                learned=complete[(c, arm)] and all(x['learned'] for x in cell[(c, arm)]),
                                excess=[x['excess'] for x in sorted(cell.get((c, arm), []), key=lambda q: q['seed'])])
             for c in configs for arm in ('tree', 'perm', 'chain')}
    # H_A: the chain's state carries the belief
    ha, ha_censored, ha_unlearned = {}, [], []
    for c in configs:
        if not complete[(c, 'chain')]:
            ha_censored.append(c)
            continue
        if not cells[f'{c}_chain']['learned']:
            ha_unlearned.append(c)
            continue
        rr = cell[(c, 'chain')]
        fam = {k: (all(x[k] for x in rr) if all(k in x for x in rr) else None) for k in ('A1', 'A2', 'A3')}
        if all(v is None for v in fam.values()):
            continue
        ha[c] = dict(affine=fam['A1'], projective=fam['A2'], softmax=fam['A3'], supported=any(bool(v) for v in fam.values()))
    sup = sum(v['supported'] for v in ha.values())
    if len(ha) < 3:
        va = 'inconclusive (fewer than 3 learned, testable grammars)'
    else:
        va = 'supported' if sup >= 0.75 * len(ha) else 'not supported' if sup <= 0.25 * len(ha) else 'partly supported'
    # H_B needs all six tree and perm runs of a grammar; the descriptive comparison the tree and chain runs
    trees_ok = {(c, arm): complete[(c, 'tree')] and complete[(c, 'perm')] for c in configs for arm in ('tree', 'perm')}
    hb = compare(cell, configs, trees_ok, 'tree', 'perm')
    tc = compare(cell, configs, complete, 'tree', 'chain')
    failed = []
    if shuffle_bad:
        failed.append(f'shuffled-label R2 > {SHUFFLE_MAX} in {shuffle_bad}')
    if planned and len(runs) < 0.85 * planned:
        failed.append(f'only {len(runs)} of {planned} runs analyzed')
    json.dump(dict(H_A=dict(verdict=va, supported=sup, grammars=ha, censored_grammars=ha_censored, not_learned=ha_unlearned),
                   H_B=hb, descriptive_tree_vs_chain=tc, cells=cells, runs=list(rows.values()), run_failed=failed,
                   n_runs=len(runs)), open(res_dir / 'summary.json', 'w'), indent=1)

    f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
    yn = lambda x, k: 'Y' if x.get(k) else ('n' if k in x else '-')
    order = sorted(rows.values(), key=lambda x: (x['config'], x['arm'], x['seed']))
    print('H_A, chain state h_t. A1 affine on ea; A2 projective and A3 softmax-of-affine on pa (D_t <= 64); each with the')
    print('identity-init chain and the token readout under the same probe; scr = the affine probe on a scrambled target')
    print(f"{'run':22s} excess | #ea affine  init tokens  ceil   scr A1 | #pa  proj  init tokens A2 |  soft  init tokens A3 | affine(pa) | distinct beliefs")
    for x in order:
        if x['arm'] == 'chain':
            print(f"{x['run']:22s} {x['excess']:.4f} | {x['n']['ea']:3d} {f(x['ea_aff'])}{f(x['ea_init'])}{f(x['ea_tokens'])}{f(x['ea_nexttok'])}"
                  f"{f(x['ea_scr'])}  {yn(x, 'A1')} | {x['n']['pa']:3d} {f(x['pa_proj'])}{f(x['pa_proj_init'])}{f(x['pa_proj_tokens'])}  {yn(x, 'A2')} "
                  f"| {f(x['pa_soft'])}{f(x['pa_soft_init'])}{f(x['pa_soft_tokens'])}  {yn(x, 'A3')} | {f(x['pa_aff'])}     | {x['n_beliefs_ea']}")
    print("\nCommon positions (eb): the arm's probe (bilinear top bonds for the trees, the state for the chain), final and at")
    print('identity init, for t <= T/2 (lo) and t > T/2 (hi), with controls on all of eb (addup = upper bound on any affine')
    print('probe on two bonds with disjoint inputs, shown where it covers all eb positions)')
    print(f"{'run':22s} excess | #lo  probe  init | #hi  probe  init |  all   affine tokens  ceil   scr  addup | #ebt probe tokprod")
    for x in order:
        print(f"{x['run']:22s} {x['excess']:.4f} | {x['n']['eb_lo']:3d} {f(x['eb_lo_best'])}{f(x['eb_lo_init'])} | {x['n']['eb_hi']:3d} "
              f"{f(x['eb_hi_best'])}{f(x['eb_hi_init'])} | {f(x['eb_best'])}{f(x['eb_aff'])}{f(x['eb_tokens'])}{f(x['eb_nexttok'])}"
              f"{f(x['eb_scr'])}{f(x['eb_addup'])} | {x['n']['ebt']:3d} {f(x['ebt_best'])}{f(x['ebt_tokprod'])}")
    print('\nBelief split (descriptive; fits on few distinct beliefs, so a weak test either way)')
    print(f"{'run':22s} #pos affine bilin  span   init")
    for x in order:
        print(f"{x['run']:22s} {x['n']['bel']:3d} {f(x['bel_aff'])}{f(x['bel_bil'])}{f(x['bel_span'])}{f(x['bel_init'])}")
    mark = lambda v: 'Y' if v else 'n' if v is not None else '-'
    print(f"\nH_A (the chain's state carries the belief): {va}: {sup} of {len(ha)} learned, testable grammars "
          + ', '.join(f"{c} [affine {mark(v['affine'])}, projective {mark(v['projective'])}, softmax {mark(v['softmax'])}]"
                      for c, v in ha.items()))
    print(f"  censored grammars (a chain run missing or unfinished): {ha_censored or 'none'}; chain not learned: {ha_unlearned or 'none'}")
    for cmp, lab, other in ((hb, 'H_B (aligned against permuted tree)', 'permuted'),
                            (tc, 'descriptive (aligned tree against chain)', 'chain')):
        show = lambda d: (f"{d['ref_wins']}-{d['other_wins']}-{d['ties']} of {d['usable']} (aligned-{other}-tie), by grammar "
                          f"{d['by_grammar']}, sign p {'-' if d['sign_p'] is None else format(d['sign_p'], '.3f')}, "
                          f"to {dict(ref='the aligned tree', other='the ' + other).get(d['to'], d['to'] or 'neither')}")
        print(f"{lab}: {cmp['verdict']}")
        print(f"  learning (excess loss; pairs where the aligned tree learned): {show(cmp['learning'])}; median gap "
              f"{f(cmp['median_learning_gap']).strip()} nats")
        print(f"  representation (t > T/2, both learned, gap minus identity-init gap): {show(cmp['representation'])}; median "
              f"adjusted gap {f(cmp['median_adjusted_gap']).strip()}")
        print(f"  the same for t <= T/2: {show(cmp['representation_eb_lo'])}")
        print(f"  the same for all of eb: {show(cmp['representation_eb'])}")
        print(f"  {'pair':18s} excess: aligned {other:8s} gap    | t > T/2: aligned {other:8s} gap  init gap adjusted | {other} learned")
        for p in cmp['pairs']:
            h = p.get('eb_hi', {})
            print(f"  {p['pair']:18s}         {f(p['ref_excess'])} {f(p['other_excess'])}  {f(p.get('learning_gap'))} |          "
                  f"{f(h.get('ref_R2'))} {f(h.get('other_R2'))}  {f(h.get('gap'))} {f(h.get('init_gap'))}  {f(h.get('adjusted'))}  | "
                  f"{'yes' if p['other_learned'] else 'no'}{'' if p['ref_learned'] else '  (aligned tree not learned: pair unused)'}")
        print(f"  censored grammars (a run of either arm missing or unfinished): {cmp['censored_grammars'] or 'none'}")
    print('not learned (excess loss >= 0.02 in some seed), among complete cells: '
          + (', '.join(f"{k} {[round(e, 3) for e in v['excess']]}" for k, v in cells.items() if v['complete'] and not v['learned']) or 'none'))
    print('incomplete cells: ' + (', '.join(f"{k} ({v['n_runs']} of {N_SEEDS})" for k, v in cells.items() if not v['complete']) or 'none'))
    print('run failed: ' + ('; '.join(failed) if failed else 'no'))


if __name__ == '__main__':
    main()
