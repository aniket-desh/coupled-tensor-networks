"""lang-evidence analysis: masked bigram (evidence from two directions) and the reserved trigger (event flag).

  PYTHONPATH=. python studies/lang/lang-evidence/analyze.py runs/lang-evidence <results dir>

Rules are in EXPERIMENT.md; helpers and shared definitions in studies/lang/analysis.py."""
import sys

from studies.lang import analysis as A

SEEDS = (0, 1, 2)
MB_LEVEL = {5: 2, 4: 3, 8: 4}


def interp_final(out, runs, arm, kind):
    """Per interpreted run of this arm and process kind: final-checkpoint values per node and summary."""
    res = {}
    for name, r in runs.items():
        if r['cfg']['arm'] != arm or r['cfg']['process']['kind'] != kind:
            continue
        ip = A.load_interp(out, name)
        if ip is not None:
            st = A.steps_of(ip)
            res[name] = (r, ip, st[0], st[-1])
    return res


def mb_mechanism(out, runs, arm):
    """H4 for one arm. (a) one-sided evidence on bonds whose span holds exactly one neighbour and at least one other
    non-mask leaf (leaves and the level-1 node whose other leaf is the mask pass at init and are excluded); (b) combined
    evidence on every bond holding both neighbours; (c) causal patching of the comb subspace at the meeting node
    (the lowest node holding both neighbours), compared with no patch and with random subspaces; plus, descriptively,
    fwd / bwd / comb R^2 at the meeting node (added vs carried side by side)."""
    fwd_ok, fwd0, comb_ok, sub, rand, none, whole, side, n_ip = [], [], [], [], [], [], [], [], 0
    comb_detail = []
    for name, (r, ip, s0, sf) in interp_final(out, runs, arm, 'mbigram').items():
        n_ip += 1
        p = r['cfg']['process']['p']
        spans = A.node_spans(r['cfg'])
        lvl = A.meeting_level(r['cfg'], [p - 1, p + 1])
        meet = next((nd for nd, S in spans.items() if int(nd.split('_')[0]) == lvl and p - 1 in S and p + 1 in S), None)
        for node, S in spans.items():
            a, b = (p - 1) in S, (p + 1) in S
            others = [t for t in S if t != p]
            if (a != b) and len(others) >= 2:
                nm = 'fwd' if a else 'bwd'
                fwd_ok.append(A.get(ip, sf, node, nm, 'probe') >= 0.9)
                fwd0.append(A.get(ip, s0, node, nm, 'probe'))
            if a and b:
                pr, ce = A.get(ip, sf, node, 'comb', 'probe'), A.get(ip, sf, node, 'comb', 'ceiling')
                comb_ok.append(pr >= ce - 0.05)
                comb_detail.append(dict(run=name, node=node, probe=pr, ceiling=ce))
        if meet is not None:
            sub.append(A.get(ip, sf, meet, 'comb', 'patch', 'sub', 'kl_target'))
            rand.append(A.get(ip, sf, meet, 'comb', 'patch', 'rand', 'kl_target'))
            none.append(A.get(ip, sf, meet, 'comb', 'patch', 'none', 'kl_target'))
            whole.append(A.get(ip, sf, meet, 'comb', 'patch', 'whole', 'kl_target'))
            side.append(dict(run=name, node=meet, k=A.get(ip, sf, meet, 'comb', 'k'), d=A.get(ip, sf, meet, 'comb', 'd'),
                             fwd=A.get(ip, sf, meet, 'fwd', 'probe'), bwd=A.get(ip, sf, meet, 'bwd', 'probe'),
                             comb=A.get(ip, sf, meet, 'comb', 'probe')))
    f1, n1 = A.frac_true(fwd_ok)
    f2, n2 = A.frac_true(comb_ok)
    ms, mr, mn = A.med(sub), A.med(rand), A.med(none)
    crit = dict(one_sided_evidence=dict(frac=f1, n=n1, median_step0=A.med(fwd0), ok=f1 >= 0.9),
                combined_evidence=dict(frac=f2, n=n2, ok=f2 >= 0.9, detail=comb_detail,
                                       rule='bond R^2 >= (the model output logits\' R^2 on the same target) - 0.05'),
                causal=dict(median_kl_sub=ms, median_kl_rand=mr, median_kl_none=mn, median_kl_whole=A.med(whole),
                            n=len([x for x in sub if x is not None]),
                            ok=bool([x for x in sub if x is not None]) and ms < 0.5 * mn and ms < mr),
                meeting_node_descriptive=side, runs_interpreted=n_ip)
    return crit


def verdict3(crit, keys):
    k = sum(crit[x]['ok'] for x in keys)
    return 'supported' if k == 3 else 'partly supported' if k == 2 else 'not supported'


def trigger_mechanism(out, runs, arm):
    """H5 for one arm. (c) causal patching of the flag subspace (deciding); (b) invariance to moving the trigger,
    computed on examples with a trigger in the span, at level >= 2 nodes, against the step-0 value; (a) flag
    decodability, reported with its step-0 value and majority-class chance (descriptive: the flag is linear in the
    one-hot tokens and is often decodable at init)."""
    flag, flag0, chance, inv, inv0, pred, sub, rand, where, where0, n_ip = [], [], [], [], [], [], [], [], [], [], 0
    inv_by_level = {}
    for name, (r, ip, s0, sf) in interp_final(out, runs, arm, 'trigger').items():
        n_ip += 1
        for node, S in A.node_spans(r['cfg']).items():
            lvl = int(node.split('_')[0])
            bg = [t for t in S if t < 15]
            if lvl == 0 or not bg:
                continue
            flag.append(A.get(ip, sf, node, 'flag', 'probe'))
            flag0.append(A.get(ip, s0, node, 'flag', 'probe'))
            chance.append(A.get(ip, sf, node, 'flag', 'chance'))
            sub.append(A.get(ip, sf, node, 'flag', 'patch', 'sub', 'moved'))
            rand.append(A.get(ip, sf, node, 'flag', 'patch', 'rand', 'moved'))
            v, v0 = A.get(ip, sf, node, 'flag', 'inv', 'sub'), A.get(ip, s0, node, 'flag', 'inv', 'sub')
            if v is not None:
                inv_by_level.setdefault(lvl, []).append(v)
                if lvl >= 2:
                    inv.append(v), inv0.append(v0), pred.append(A.get(ip, sf, node, 'flag', 'inv', 'pred_same'))
            if lvl >= 2 and A.get(ip, sf, node, 'where', 'probe') is not None:
                where.append(A.get(ip, sf, node, 'where', 'probe'))
                where0.append(A.get(ip, s0, node, 'where', 'probe'))
    crit = dict(causal=dict(median_sub_moved=A.med(sub), median_rand_moved=A.med(rand), n=len([x for x in sub if x is not None]),
                            ok=bool([x for x in sub if x is not None]) and A.med(sub) >= 0.8 and A.med(rand) <= 0.4),
                invariant=dict(median_sub_level_ge2=A.med(inv), median_sub_step0=A.med(inv0), median_pred_same=A.med(pred),
                               by_level={l: A.med(v) for l, v in sorted(inv_by_level.items())}, n=len(inv),
                               ok=bool(inv) and A.med(inv) <= 0.1 and A.med(inv) < A.med(inv0)),
                flag_descriptive=dict(median_probe=A.med(flag), median_probe_step0=A.med(flag0), median_chance=A.med(chance)),
                location_descriptive=dict(median_where_probe_final=A.med(where), median_where_probe_step0=A.med(where0)),
                runs_interpreted=n_ip)
    c, i = crit['causal']['ok'], crit['invariant']['ok']
    crit['verdict'] = ('supported' if c and i else 'partly supported' if c else 'not supported') if n_ip else 'inconclusive (no interpretation files)'
    return crit


def main():
    out, res = sys.argv[1], sys.argv[2]
    runs, planned = A.load_runs(out)
    R = lambda n: runs.get(n)
    rows = {n: A.row(r) for n, r in runs.items()}
    fin = [n for n in planned if n in runs and A.finished(runs[n])]
    an = dict(health=dict(planned=len(planned), finished=len(fin), missing=[n for n in planned if n not in fin],
                          collapsed=[n for n in fin if rows[n]['collapsed']]), runs=rows)
    # H1: the tree learns the masked bigram at every mask position; r = 32 adds nothing (capacity check)
    kls = [rows[f'mb_p{p}_tree_s{s}']['kl'] for p in MB_LEVEL for s in SEEDS if f'mb_p{p}_tree_s{s}' in rows]
    cap = A.compare([(R(f'mb_p{p}_tree32_s{s}'), R(f'mb_p{p}_tree_s{s}')) for p in MB_LEVEL for s in SEEDS], A.better)
    an['H1_mb_learned'] = dict(max_kl=max(kls) if kls else None, n=len(kls), capacity_tree32_vs_tree=cap,
                               verdict=('inconclusive (fewer than 6 capacity pairs)' if cap['n'] < 6 else
                                        'supported' if len(kls) == 9 and max(kls) <= 0.02 and cap['wins'] <= 1 else 'not supported'))
    # H2: two-sided effect of where the neighbours meet: p = 5 (level 2) vs p = 8 (root), per seed
    c = A.compare([(R(f'mb_p5_tree_s{s}'), R(f'mb_p8_tree_s{s}')) for s in SEEDS], A.faster)
    q = A.compare([(R(f'mb_p5_tree_s{s}'), R(f'mb_p8_tree_s{s}')) for s in SEEDS], A.better)
    dirn = lambda c: 'level-2 meeting faster/better' if c['wins'] == 3 else ('root meeting faster/better' if c['losses'] == 3 else 'unresolved at n = 3 (see medians and ranges)')
    an['H2_mb_meeting_level'] = dict(speed=c, quality=q, speed_direction=dirn(c), quality_direction=dirn(q),
                                     medians={p: dict(t=A.med([rows.get(f'mb_p{p}_tree_s{s}', {}).get('t') for s in SEEDS]),
                                                      t_range=A.rng([rows.get(f'mb_p{p}_tree_s{s}', {}).get('t') for s in SEEDS]),
                                                      kl=A.med([rows.get(f'mb_p{p}_tree_s{s}', {}).get('kl') for s in SEEDS]),
                                                      kl_range=A.rng([rows.get(f'mb_p{p}_tree_s{s}', {}).get('kl') for s in SEEDS])) for p in MB_LEVEL})
    # H3 (descriptive): the permutation moves the neighbours' meeting level in both directions, so tree and perm runs
    # are also pooled by meeting level
    by_level = {}
    for n, r in runs.items():
        if r['cfg']['process']['kind'] == 'mbigram' and r['cfg']['arm'] in ('tree', 'perm'):
            p = r['cfg']['process']['p']
            lv = A.meeting_level(r['cfg'], [p - 1, p + 1])
            rows[n]['meeting_level'] = lv
            by_level.setdefault(lv, []).append(dict(name=n, t=rows[n]['t'], kl=rows[n]['kl']))
    an['H3_mb_by_meeting_level'] = {lv: dict(n=len(v), t=A.med([x['t'] for x in v]), kl=A.med([x['kl'] for x in v]), runs=v)
                                    for lv, v in sorted(by_level.items())}
    # tree vs perm (9 pairs) and tree vs chain, descriptive
    for b in ('perm', 'chain'):
        sp = A.compare([(R(f'mb_p{p}_tree_s{s}'), R(f'mb_p{p}_{b}_s{s}')) for p in MB_LEVEL for s in SEEDS], A.faster)
        ql = A.compare([(R(f'mb_p{p}_tree_s{s}'), R(f'mb_p{p}_{b}_s{s}')) for p in MB_LEVEL for s in SEEDS], A.better)
        d = lambda c: 'tree' if c['wins'] >= 8 and c['losses'] == 0 else (b if c['losses'] >= 8 and c['wins'] == 0 else 'no consistent difference')
        an[f'H3_mb_tree_vs_{b}' if b == 'perm' else 'desc_mb_tree_vs_chain'] = dict(speed=sp, quality=ql, speed_direction=d(sp), quality_direction=d(ql))
    # H4 masked-bigram mechanism
    # H4: the causal test needs a probe subspace well below the bond width, so it is decided on tree32 (comb: k <= 15
    # of d = 32); the r = 16 tree is reported the same way (there k = 15 of 16, so sub, random and whole coincide)
    m32, m16 = mb_mechanism(out, runs, 'tree32'), mb_mechanism(out, runs, 'tree')
    m32['verdict'] = (verdict3(m32, ('one_sided_evidence', 'combined_evidence', 'causal')) if m32['runs_interpreted']
                      else 'inconclusive (no interpretation files)')
    an['H4_mb_mechanism'] = dict(tree32=m32, tree=m16, verdict=m32['verdict'])
    # H5 trigger mechanism (main arm tree32; tree16 and perm32 reported the same way)
    an['H5_trigger_mechanism'] = {arm: trigger_mechanism(out, runs, arm) for arm in ('tree32', 'tree16', 'perm32')}
    an['H5_trigger_mechanism']['verdict'] = an['H5_trigger_mechanism']['tree32']['verdict']
    # H6 trigger arms, descriptive
    an['H6_trigger_arms'] = {arm: dict(t=A.med([rows.get(f'trigger_{arm}_s{s}', {}).get('t') for s in SEEDS]),
                                       kl=A.med([rows.get(f'trigger_{arm}_s{s}', {}).get('kl') for s in SEEDS]),
                                       strata={st: A.med([(rows.get(f'trigger_{arm}_s{s}', {}).get('strata') or {}).get(st, {}).get('kl') for s in SEEDS]) for st in '01234'})
                             for arm in ('tree16', 'tree32', 'perm32', 'chain')}
    an['H6_trigger_arms']['tree16_vs_tree32_quality'] = A.compare([(R(f'trigger_tree16_s{s}'), R(f'trigger_tree32_s{s}')) for s in SEEDS], A.better)
    A.apply_health_override(an)
    A.dump(f'{res}/analysis.json', an)
    A.dump(f'{res}/curves.json', A.curves(runs))
    A.dump(f'{res}/interp.json', dict(trajectories=A.trajectories(out, runs)))
    for k in ('H1_mb_learned', 'H2_mb_meeting_level', 'H3_mb_tree_vs_perm', 'H4_mb_mechanism', 'H5_trigger_mechanism'):
        if k not in an:
            continue
        print(k, {kk: v for kk, v in an[k].items() if 'direction' in kk or kk == 'verdict'})
    print('health', {k: (len(v) if isinstance(v, list) else v) for k, v in an['health'].items()})


if __name__ == '__main__':
    main()
