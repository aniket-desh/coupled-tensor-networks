"""Apply the success criteria of EXPERIMENT.md to results/*/analysis.json; print tables, write summary.json.

    PYTHONPATH=. python studies/rhm/rhm-ttn-messages/msg_summary.py RESULTS_DIR [N_PLANNED]

All numbers are at the final checkpoint, pooled over nodes (msg_analyze.pooled); `init` is checkpoint 0, the
identity-init tree, pooled over the same nodes. Scopes: M1 and M2 pool their eligible nodes at all levels; M3 and M4
are judged over parents of tree level >= 2 (`l2+`), because a level-1 parent's merged class-mean bonds are rows its
decoder was fit on, and its children (leaves) are never partially observed; level 1 is printed separately.
A (grammar, width) cell is judged only if all N_SEEDS seeds finished training with status 'finished'; otherwise it is
censored, listed separately from cells that did not learn."""
import json
import sys
from collections import defaultdict
from pathlib import Path

R2_MIN, ACC_MIN, EXCESS_MAX, SHUFFLE_MAX, N_SEEDS, GAP = 0.8, 0.95, 0.02, 0.05, 3, 0.5


def closes_gap(r2, *controls):
    """The measure (R2, ve or accuracy) leaves at most GAP of the shortfall from 1 that the best control leaves:
    1 - R2 <= GAP * (1 - max(control, 0)).
    Unlike a fixed margin, this can be met whatever the control's level, and a control below 0 counts as 0."""
    best = max([c for c in controls if c is not None] + [0.0])
    return 1 - r2 <= GAP * (1 - best)


def main():
    res_dir = Path(sys.argv[1])
    planned = int(sys.argv[2]) if len(sys.argv) > 2 else None
    runs = {d.name: json.load(open(d / 'analysis.json')) for d in sorted(res_dir.iterdir()) if (d / 'analysis.json').exists()}
    by, rows, shuffle_bad = defaultdict(list), [], []
    for name, a in runs.items():
        cks = sorted(int(c) for c in a['metrics'])
        fin, ini = a['metrics'][str(cks[-1])], a['metrics'][str(cks[0])]
        P, P0 = fin['pooled']['all'], ini['pooled']['all']
        H, H0 = fin['pooled']['l2+'], ini['pooled']['l2+']
        L1, L10 = fin['pooled'].get('l1', {}), ini['pooled'].get('l1', {})
        g = lambda pooled, key, f='R2': (pooled.get(key) or {}).get(f)
        best = lambda *vals: max([x for x in vals if x is not None], default=None)
        r = a['cfg']['r']
        elig = [n for n in fin['nodes'].values() if n['info']['eligible']]
        shuf = [n['m1']['shuffle_all']['R2'] for n in elig if 'shuffle_all' in n['m1']]
        if shuf and max(shuf) > SHUFFLE_MAX:
            shuffle_bad.append(name)
        n_msgs = [n['info']['n_partial_messages'] for n in elig]
        levels = sorted(int(k[1:]) for k in fin['pooled'] if k[0] == 'l' and k[1:].isdigit())
        x = dict(run=name, r=r, excess=a['final_excess'], status=a['status'], learned=a['final_excess'] < EXCESS_MAX,
                 # M1: partially observed messages, held-out local prefixes, eligible nodes
                 n_elig=P['m1_n_eligible'], n_partial_messages=n_msgs,
                 # a node where the distinct partially observed messages are no more than the probe's r + 1 parameters
                 # per coordinate: there M1 tests whether the bond identifies the message, not how it codes it
                 n_identity_nodes=sum(m <= r + 1 for m in n_msgs),
                 m1_proj=g(P, 'm1_proj_partial'), m1_aff=g(P, 'm1_aff_partial'), m1_init=g(P0, 'm1_proj_partial'),
                 m1_tok=best(g(P, 'm1_tok_aff_partial'), g(P, 'm1_tok_proj_partial')),
                 # the pairwise token readout, and the bond probe pooled over the same nodes
                 m1_tok2=g(P, 'm1_tok2_aff_partial@tok2'), m1_proj_at_tok2=g(P, 'm1_proj_partial@tok2'),
                 m1_tok2_nodes=g(P, 'm1_tok2_aff_partial@tok2', 'nodes') or 0,
                 m1_scr=g(P, 'm1_scr_proj_partial'), m1_full=g(P, 'm1_proj_full'), shuffle=max(shuf) if shuf else None,
                 m1x_proj=g(P, 'm1_xproj_partial'), m1x_aff=g(P, 'm1_xaff_partial'), m1x_init=g(P0, 'm1_xproj_partial'),
                 m1x_tok=g(P, 'm1_xtok_aff_partial'),
                 # M2: invariance to the realization; the criterion groups by the extended message
                 m2_ve=g(P, 'm2_observed_xmsg', 've'), m2_init=g(P0, 'm2_observed_xmsg', 've'), m2_cover=g(P, 'm2_observed_xmsg', 'cover'),
                 m2_partial=g(P, 'm2_partial_xmsg', 've'), m2_full=g(P, 'm2_full_xmsg', 've'),
                 m2_msg=g(P, 'm2_observed_msg', 've'), m2_msg_init=g(P0, 'm2_observed_msg', 've'),
                 m2_msg_partial=g(P, 'm2_partial_msg', 've'), m2_nmsg=g(P, 'm2_observed_nmsg', 've'),
                 # M3: production table, parents of level >= 2 (level 1 separately)
                 m3_acc=g(H, 'm3', 'acc_proj'), m3_mass=g(H, 'm3', 'mass_proj'), m3_init=g(H0, 'm3', 'acc_proj'),
                 m3_pairs=g(H, 'm3', 'n_valid'), m3_invalid=g(H, 'm3', 'invalid_norm_ratio'),
                 m3_invalid_init=g(H0, 'm3', 'invalid_norm_ratio'), m3_invalid_q=g(H, 'm3', 'invalid_q_ratio'),
                 m3_invalid_q_init=g(H0, 'm3', 'invalid_q_ratio'), m3_valid_q_cv=g(H, 'm3', 'valid_q_cv'),
                 m3_l1_acc=g(L1, 'm3', 'acc_proj'), m3_l1_init=g(L10, 'm3', 'acc_proj'), m3_l1_pairs=g(L1, 'm3', 'n_valid'),
                 # M4: mixture update, parents of level >= 2
                 m4_emp=g(H, 'm4_emp_proj'), m4_init=g(H0, 'm4_emp_proj'), m4_ceil=g(H, 'm4_emp_addceil'),
                 m4_seen=g(H, 'm4_emp_seen_proj'), m4_seen_ceil=g(H, 'm4_emp_seen_addceil'),
                 m4_left=g(H, 'm4_emp_leftmask_proj'), m4_left_init=g(H0, 'm4_emp_leftmask_proj'), m4_left_ceil=g(H, 'm4_emp_leftmask_addceil'),
                 m4_both=g(H, 'm4_emp_both_proj'), m4_both_init=g(H0, 'm4_emp_both_proj'), m4_both_ceil=g(H, 'm4_emp_both_addceil'),
                 m4_both_nodes=g(H, 'm4_emp_both_proj', 'nodes') or 0,
                 m4_dir=g(H, 'm4_dir_proj'), m4_dir_ceil=g(H, 'm4_dir_addceil'), m4_dir_init=g(H0, 'm4_dir_proj'),
                 m4_l1_emp=g(L1, 'm4_emp_proj'), m4_l1_init=g(L10, 'm4_emp_proj'),
                 m4_by_level={l: dict(both=g(fin['pooled'][f'l{l}'], 'm4_emp_both_proj'), both_ceil=g(fin['pooled'][f'l{l}'], 'm4_emp_both_addceil'),
                                      both_init=g(ini['pooled'][f'l{l}'], 'm4_emp_both_proj'),
                                      left=g(fin['pooled'][f'l{l}'], 'm4_emp_leftmask_proj'), left_ceil=g(fin['pooled'][f'l{l}'], 'm4_emp_leftmask_addceil'),
                                      emp=g(fin['pooled'][f'l{l}'], 'm4_emp_proj'), dir=g(fin['pooled'][f'l{l}'], 'm4_dir_proj'))
                              for l in levels},
                 m4a_emp=g(P, 'm4a_emp_proj'), m4a_dir=g(P, 'm4a_dir_proj'), m4a_init=g(P0, 'm4a_emp_proj'))
        if x['n_elig'] and x['m1_proj'] is not None:
            ok = x['m1_proj'] >= R2_MIN and closes_gap(x['m1_proj'], x['m1_init'], x['m1_tok'])
            if x['m1_tok2'] is not None and x['m1_proj_at_tok2'] is not None:  # like with like: the same nodes
                ok = ok and closes_gap(x['m1_proj_at_tok2'], x['m1_tok2'])
            x['h1'] = bool(ok)
            x['m1_selectivity'] = x['m1_proj'] - x['m1_scr'] if x['m1_scr'] is not None else None
        if x['m2_ve'] is not None and x['m2_init'] is not None:
            x['h2'] = bool(x['m2_ve'] >= R2_MIN and closes_gap(x['m2_ve'], x['m2_init']))
        if x['m3_acc'] is not None:
            x['h3'] = bool(x['m3_acc'] >= ACC_MIN and closes_gap(x['m3_acc'], x['m3_init']))
        if x['m4_both'] is not None and x['m4_emp'] is not None:
            x['h4'] = bool(x['m4_emp'] >= R2_MIN and x['m4_both'] >= R2_MIN
                           and closes_gap(x['m4_both'], x['m4_both_init'], x['m4_both_ceil']))
        rows.append(x)
        by[name.rsplit('_s', 1)[0]].append(x)
    cells, verdicts = {}, {}
    for cell, rr in by.items():
        censored = len(rr) < N_SEEDS or any(x['status'] != 'finished' for x in rr)
        learned = (not censored) and all(x['learned'] for x in rr)
        cells[cell] = dict(r=rr[0]['r'], n_runs=len(rr), censored=censored, learned=learned,
                           # H1 is a test of message identity in this cell if, in some seed, at least half of the eligible
                           # nodes have no more distinct partially observed messages than probe parameters per coordinate
                           h1_identity_test=any(x['n_elig'] and 2 * x['n_identity_nodes'] >= x['n_elig'] for x in rr),
                           **{h: (all(x[h] for x in rr) if learned and all(h in x for x in rr) else None) for h in ('h1', 'h2', 'h3', 'h4')})
    for h in ('h1', 'h2', 'h3', 'h4'):
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
            if h == 'h1':
                verdicts[f'{h}_r{r}']['identity_test_cells'] = [c for c in judged if cells[c]['h1_identity_test']]
    failed = []
    if shuffle_bad:
        failed.append(f'shuffled-label R2 > {SHUFFLE_MAX} at an eligible node in {shuffle_bad}')
    if planned and len(runs) < 0.85 * planned:
        failed.append(f'only {len(runs)} of {planned} runs analyzed')
    json.dump(dict(verdicts=verdicts, cells=cells, runs=rows, run_failed=failed, n_runs=len(runs)),
              open(res_dir / 'summary.json', 'w'), indent=1)
    f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
    yn = lambda x, k: 'Y' if x.get(k) else ('n' if k in x else '-')
    print('H1 / M1: partially observed messages from the bond, projective probe, held-out local prefixes, eligible nodes '
          '(tok = token readout; scr = scrambled target; @tok2 = the nodes where the pairwise token readout is eligible: bond '
          'probe, then that readout, on those nodes; id = eligible nodes with at most r + 1 distinct partial messages)')
    print(f"{'run':22s} excess #nodes  proj   aff   init   tok    scr  full  shuffle | @tok2 #n  proj   tok2 | H1  id   distinct partial messages per node")
    for x in rows:
        print(f"{x['run']:22s} {x['excess']:.4f} {x['n_elig']:4d}  {f(x['m1_proj'])}{f(x['m1_aff'])}{f(x['m1_init'])}{f(x['m1_tok'])}"
              f"{f(x['m1_scr'])}{f(x['m1_full'])}{f(x['shuffle'])}  |     {x['m1_tok2_nodes']:3d} {f(x['m1_proj_at_tok2'])}{f(x['m1_tok2'])} | "
              f"{yn(x, 'h1')}  {x['n_identity_nodes']}/{x['n_elig']}  {x['n_partial_messages']}")
    print('\nM1x (descriptive): the normalized extended message (with the next token) from the bond, partially observed rows')
    print(f"{'run':22s}  proj   aff   init   tok")
    for x in rows:
        print(f"{x['run']:22s} {f(x['m1x_proj'])}{f(x['m1x_aff'])}{f(x['m1x_init'])}{f(x['m1x_tok'])}")
    print('\nH2 / M2: share of whitened bond variance explained (ve) among rows with a synonymous partner (cover). Criterion: '
          'the same extended message and number of observed leaves (xmsg). Descriptive: the same message (msg), normalized (nmsg)')
    print(f"{'run':22s} xmsg:ve  init  cover partial full  H2 | msg:ve  init partial | nmsg:ve")
    for x in rows:
        print(f"{x['run']:22s}  {f(x['m2_ve'])}{f(x['m2_init'])}{f(x['m2_cover'])}{f(x['m2_partial'])}{f(x['m2_full'])}  {yn(x, 'h2')}  | "
              f"{f(x['m2_msg'])}{f(x['m2_msg_init'])}{f(x['m2_msg_partial'])} | {f(x['m2_nmsg'])}")
    print('\nH3 / M3: production table read from the cores on class-mean bonds, parents of level >= 2 (invalid: norm, then read-out '
          'mass q, of invalid pairs / valid pairs, 0 for exact messages; cv = spread of q over valid pairs); level 1 separately')
    print(f"{'run':22s} #pairs  acc   mass  init-acc H3 | invalid: norm  init    q    init  valid-cv | level 1: #pairs acc   init")
    for x in rows:
        print(f"{x['run']:22s} {x['m3_pairs'] or 0:6d} {f(x['m3_acc'])}{f(x['m3_mass'])}{f(x['m3_init'])}   {yn(x, 'h3')}  |        "
              f"{f(x['m3_invalid'])}{f(x['m3_invalid_init'])}{f(x['m3_invalid_q'])}{f(x['m3_invalid_q_init'])}{f(x['m3_valid_q_cv'])}   |"
              f"          {x['m3_l1_pairs'] or 0:5d} {f(x['m3_l1_acc'])}{f(x['m3_l1_init'])}")
    print('\nH4 / M4: exact update on mixtures of class means, parents of level >= 2 (emp = all empirical pairs; both = left partial, '
          'right informative; left = left masked, right informative; ceil = additive ceiling of the same subset; M4a = code linearity)')
    print(f"{'run':22s}  emp   init   ceil | both  b-init b-ceil #nodes | left  l-init l-ceil | seen  s-ceil |  dir  d-init d-ceil | "
          f"M4a-emp M4a-dir M4a-init | level 1 emp init | H4")
    for x in rows:
        print(f"{x['run']:22s} {f(x['m4_emp'])}{f(x['m4_init'])}{f(x['m4_ceil'])} |{f(x['m4_both'])}{f(x['m4_both_init'])}{f(x['m4_both_ceil'])}"
              f"  {x['m4_both_nodes']:3d}   |{f(x['m4_left'])}{f(x['m4_left_init'])}{f(x['m4_left_ceil'])} |{f(x['m4_seen'])}{f(x['m4_seen_ceil'])} |"
              f"{f(x['m4_dir'])}{f(x['m4_dir_init'])}{f(x['m4_dir_ceil'])} | {f(x['m4a_emp'])}  {f(x['m4a_dir'])}  {f(x['m4a_init'])}   | "
              f"{f(x['m4_l1_emp'])}{f(x['m4_l1_init'])}  | {yn(x, 'h4')}")
    print('\nM4 per tree level of the parent: both (init, ceiling) / left-masked (ceiling) / all empirical / Dirichlet')
    for x in rows:
        print(f"{x['run']:22s} " + '  '.join(
            f"l{l}: both{f(q['both'])} ({f(q['both_init']).strip()},{f(q['both_ceil']).strip()}) left{f(q['left'])} ({f(q['left_ceil']).strip()}) "
            f"emp{f(q['emp'])} dir{f(q['dir'])}" for l, q in x['m4_by_level'].items()))
    print()
    for k, v in verdicts.items():
        extra = ''
        if k.startswith('h1') and v.get('identity_test_cells'):
            extra = f" [in {', '.join(v['identity_test_cells'])} H1 is a test of message identity: few distinct messages]"
        print(f"{k}: {v['verdict']}: {v['supported']} of {len(v['judged'])} learned, testable grammars ({', '.join(v['judged'])}){extra}")
    print('censored (a seed missing or not finished): ' + (', '.join(c for c, v in cells.items() if v['censored']) or 'none'))
    print('not learned (excess loss >= 0.02 in some seed): '
          + (', '.join(c for c, v in cells.items() if not v['censored'] and not v['learned']) or 'none'))
    print('run failed: ' + ('; '.join(failed) if failed else 'no'))


if __name__ == '__main__':
    main()
