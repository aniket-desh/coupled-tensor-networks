"""lang-source analysis: run health, the four pre-registered comparisons (EXPERIMENT.md) and the mechanism criteria.

  PYTHONPATH=. python studies/lang/lang-source/analyze.py runs/lang-source <results dir>

Writes <results>/analysis.json (verdicts with every pair), <results>/curves.json (logs) and <results>/interp.json
(per-run probe, invariance and patching summaries at every checkpoint, used by plot.py)."""
import math
import sys

from studies.lang import analysis as A

SEEDS = (0, 1, 2)
FIXED = (0, 7, 8, 15)
LEVEL = {'two_0_1': 1, 'two_1_2': 2, 'two_3_4': 3, 'two_7_8': 4, 'two_0_8': 4, 'two_0_15': 4}
ROOT_PAIRS = ('two_7_8', 'two_0_8', 'two_0_15')


def verdict_count(c, n_needed):
    """Direction for a over b: 'a' if wins >= n_needed and no losses, 'b' if losses >= n_needed and no wins."""
    if c['wins'] >= n_needed and c['losses'] == 0:
        return 'a'
    if c['losses'] >= n_needed and c['wins'] == 0:
        return 'b'
    return 'none'


def mechanism(out, runs, arm):
    """Criteria (a)-(d) at the final checkpoint for one tree-type arm; returns the summary and per-run detail."""
    a_ok, b_ok, b_local, c_sub, c_rand, d_change, per_run = [], [], [], [], [], [], {}
    n_ip = 0
    for name, r in runs.items():
        if r['cfg']['arm'] != arm or not A.finished(r):
            continue
        ip = A.load_interp(out, name)
        if ip is None:
            continue
        n_ip += 1
        st = A.steps_of(ip)
        s0, sf = st[0], st[-1]
        spans = A.node_spans(r['cfg'])
        p = r['cfg']['process']
        src = [p['j']] if p['kind'] == 'fixed' else [p['i'], p['j']]
        cond = name.rsplit(f'_{arm}_', 1)[0]
        det = {}
        for node, S in spans.items():
            has = [t in S for t in src]
            if node.startswith('0_'):
                continue  # a leaf bond is the token's own embedding: true by construction
            if p['kind'] == 'fixed' and has[0]:
                a_ok.append(A.get(ip, sf, node, 'src', 'probe') >= 0.95)
            if p['kind'] == 'two' and sum(has) == 1:
                nm = 'src_i' if has[0] else 'src_j'
                a_ok.append(A.get(ip, sf, node, nm, 'probe') >= 0.95)
            if any(has) and A.get(ip, s0, node, 'distractor', 'probe') is not None:
                d_change.append(A.get(ip, sf, node, 'distractor', 'probe') - A.get(ip, s0, node, 'distractor', 'probe'))
            if p['kind'] == 'two' and all(has) and LEVEL[cond] <= 3:
                fp = A.get(ip, sf, node, 'f', 'probe')
                b_ok.append(fp >= 0.95)
                det[node] = dict(f_probe=fp, f_probe_step0=A.get(ip, s0, node, 'f', 'probe'),
                                 f_local=A.get(ip, sf, node, 'f', 'local'),
                                 sub_moved=A.get(ip, sf, node, 'f', 'patch', 'sub', 'moved'),
                                 rand_moved=A.get(ip, sf, node, 'f', 'patch', 'rand', 'moved'))
        if det:
            lca = min(det, key=lambda nd: int(nd.split('_')[0]))
            b_local.append(det[lca]['f_probe'] - det[lca]['f_probe_step0'])
            c_sub.append(det[lca]['sub_moved'])
            c_rand.append(det[lca]['rand_moved'])
        per_run[name] = det
    fa, na = A.frac_true(a_ok)
    fb, nb = A.frac_true(b_ok)
    crit = dict(
        a_source_retained=dict(frac=fa, n=na, ok=fa >= 0.9),
        b_conjunction=dict(frac_f_decodable=fb, n=nb, median_gain_over_step0_at_lca=A.med(b_local),
                           ok=fb >= 0.9 and A.med(b_local) >= 0.2),
        c_causal=dict(mean_sub_moved=sum(c_sub) / len(c_sub) if c_sub else float('nan'),
                      mean_rand_moved=sum(c_rand) / len(c_rand) if c_rand else float('nan'), n=len(c_sub)),
        d_distractor=dict(median_change=A.med(d_change), n=len(d_change)),
        runs_interpreted=n_ip)
    c = crit['c_causal']
    c['ok'] = c['n'] > 0 and c['mean_sub_moved'] >= 0.8 and c['mean_sub_moved'] - c['mean_rand_moved'] >= 0.4
    d = crit['d_distractor']
    d['ok'] = d['n'] > 0 and d['median_change'] <= -0.2
    k = sum(crit[x]['ok'] for x in ('a_source_retained', 'b_conjunction', 'c_causal'))
    crit['verdict'] = 'consistent' if k == 3 else ('partly consistent' if k == 2 else 'not consistent')
    crit['H6_distractor_verdict'] = 'supported' if d['ok'] else 'not supported'
    if n_ip == 0:
        crit['verdict'] = crit['H6_distractor_verdict'] = 'inconclusive (no interpretation files)'
    return crit, per_run


def main():
    out, res = sys.argv[1], sys.argv[2]
    runs, planned = A.load_runs(out)
    R = lambda n: runs.get(n)
    rows = {n: A.row(r) for n, r in runs.items()}
    fin = [n for n in planned if n in runs and A.finished(runs[n])]
    health = dict(planned=len(planned), finished=len(fin), missing=[n for n in planned if n not in fin],
                  collapsed=[n for n in fin if rows[n]['collapsed']])
    an = dict(health=health, runs=rows)

    # H1 symmetry control (tree): fixed-source positions and the three root-LCA pairs
    def spread(names_by_cond):
        meds = {c: A.med([rows[n]['t'] for n in ns if n in rows]) for c, ns in names_by_cond.items()}
        ranges = [A.rng([rows[n]['t'] for n in ns if n in rows]) for ns in names_by_cond.values()]
        sr = max((b - a) for a, b in ranges) if ranges else float('nan')
        gaps = [A.gap(runs[n], rows[n]['t']) for ns in names_by_cond.values() for n in ns if n in rows]
        fm = [m for m in meds.values() if math.isfinite(m)]
        sp = (max(fm) - min(fm)) if len(fm) == len(meds) else float('inf')
        return dict(medians=meds, spread=sp, seed_range=sr, max_gap=max(gaps) if gaps else 0,
                    ok=sp <= max(sr, max(gaps) if gaps else 0))
    an['H1_symmetry'] = dict(
        fixed=spread({f'j{j}': [f'fixed_j{j}_tree_s{s}' for s in SEEDS] for j in FIXED}),
        root_pairs=spread({c: [f'{c}_tree_s{s}' for s in SEEDS] for c in ROOT_PAIRS}))
    h1 = an['H1_symmetry']
    missing = any(not math.isfinite(m) for part in ('fixed', 'root_pairs') for m in h1[part]['medians'].values())
    h1['verdict'] = ('inconclusive (a condition has no finite median)' if missing else
                     'holds' if h1['fixed']['ok'] and h1['root_pairs']['ok'] else 'violated')

    # H2 chain distance: j = 15 (read last) faster than j = 0 (read first)
    c = A.compare([(R(f'fixed_j15_chain_s{s}'), R(f'fixed_j0_chain_s{s}')) for s in SEEDS], A.faster)
    v = verdict_count(c, 3)
    an['H2_chain_distance'] = dict(c, medians_eta={f'j{j}': A.med([rows.get(f'fixed_j{j}_chain_s{s}', {}).get('eta_t') for s in SEEDS]) for j in FIXED},
                                   medians={f'j{j}': A.med([rows.get(f'fixed_j{j}_chain_s{s}', {}).get('t') for s in SEEDS]) for j in FIXED},
                                   verdict='inconclusive (fewer than 2 usable pairs)' if c['n'] < 2 else
                                   'supported' if v == 'a' else ('partly supported' if c['wins'] == 2 and c['losses'] == 0 else 'not supported'))

    # H3 tree LCA level (two-sided). Primary: level 1 (two_0_1) vs level 3 (two_3_4), both formed in identity-initialized
    # fusion cores. Level 1 vs the root is reported separately: a root conjunction is formed in C, which keeps its
    # variance init, so that comparison confounds level with the core's init.
    c13 = A.compare([(R(f'two_0_1_tree_s{s}'), R(f'two_3_4_tree_s{s}')) for s in SEEDS], A.faster)
    v13 = verdict_count(c13, 3)
    pairs = []
    for s in SEEDS:
        root = [R(f'{p}_tree_s{s}') for p in ROOT_PAIRS]
        root = [x for x in root if x is not None]
        r4 = sorted(root, key=A.t_thresh)[len(root) // 2] if root else None
        pairs.append((R(f'two_0_1_tree_s{s}'), r4))
    c = A.compare(pairs, A.faster)
    v = verdict_count(c, 3)
    from scipy.stats import spearmanr
    lv, ts = zip(*[(LEVEL[n.split('_tree_')[0]], rows[n]['t']) for n in rows if n.startswith('two_') and '_tree_s' in n]) if rows else ((), ())
    rho = spearmanr(lv, [t if math.isfinite(t) else 1e12 for t in ts]) if len(lv) > 2 else None
    an['H3_tree_lca_level'] = dict(
        primary_level1_vs_level3=c13,
        verdict=('inconclusive (fewer than 2 usable pairs)' if c13['n'] < 2 else
                 {'a': 'level 1 faster than level 3', 'b': 'level 3 faster than level 1',
                  'none': 'unresolved at n = 3 (no direction in 3 of 3 seeds)'}[v13]),
        level1_vs_root_confounded=dict(c, direction={'a': 'level 1 faster than the root', 'b': 'root faster than level 1',
                                                     'none': 'no consistent difference'}[v]),
                                   spearman_level_vs_t=dict(rho=rho.correlation, p=rho.pvalue) if rho else None,
                                   medians_by_level={L: A.med([rows[n]['t'] for n in rows if n.startswith('two_') and '_tree_s' in n and LEVEL[n.split('_tree_')[0]] == L]) for L in (1, 2, 3, 4)},
                                   final_kl_by_level={L: A.med([rows[n]['kl'] for n in rows if n.startswith('two_') and '_tree_s' in n and LEVEL[n.split('_tree_')[0]] == L]) for L in (1, 2, 3, 4)})

    # H4 descriptive: tree vs chain at the root-LCA pairs, and the chain's dependence on sequence distance
    an['H4_root_pairs'] = {p: {arm: dict(t=A.med([rows.get(f'{p}_{arm}_s{s}', {}).get('t') for s in SEEDS]),
                                         kl=A.med([rows.get(f'{p}_{arm}_s{s}', {}).get('kl') for s in SEEDS]))
                               for arm in ('tree', 'tied', 'chain')} for p in ROOT_PAIRS}
    an['H4_root_pairs']['chain_0_15_vs_7_8'] = A.compare([(R(f'two_7_8_chain_s{s}'), R(f'two_0_15_chain_s{s}')) for s in SEEDS], A.faster)

    # descriptive comparisons over all conditions
    conds = [f'fixed_j{j}' for j in FIXED] + list(LEVEL)
    for a_, b_ in (('tree', 'chain'), ('tree', 'tied')):
        an[f'desc_{a_}_vs_{b_}'] = dict(
            speed=A.compare([(R(f'{c}_{a_}_s{s}'), R(f'{c}_{b_}_s{s}')) for c in conds for s in SEEDS], A.faster),
            speed_cumulative_lr=A.compare([(R(f'{c}_{a_}_s{s}'), R(f'{c}_{b_}_s{s}')) for c in conds for s in SEEDS], A.faster_eta),
            quality=A.compare([(R(f'{c}_{a_}_s{s}'), R(f'{c}_{b_}_s{s}')) for c in conds for s in SEEDS], A.better))
    an['capacity_tree32_vs_tree'] = dict(
        quality=A.compare([(R(f'{c}_tree32_s{s}'), R(f'{c}_tree_s{s}')) for c in ('fixed_j0', 'two_0_1', 'two_7_8') for s in SEEDS], A.better),
        kl={f'{c}_{arm}': A.med([rows.get(f'{c}_{arm}_s{s}', {}).get('kl') for s in SEEDS]) for c in ('fixed_j0', 'two_0_1', 'two_7_8') for arm in ('tree', 'tree32')})

    # H5 mechanism
    an['H5_mechanism'] = {}
    per_run = {}
    for arm in ('tree', 'tied', 'tree32'):
        an['H5_mechanism'][arm], per_run[arm] = mechanism(out, runs, arm)
    an['H5_mechanism']['verdict'] = an['H5_mechanism']['tree']['verdict']
    an['H6_distractor'] = dict(verdict=an['H5_mechanism']['tree']['H6_distractor_verdict'],
                               median_change=an['H5_mechanism']['tree']['d_distractor']['median_change'],
                               n=an['H5_mechanism']['tree']['d_distractor']['n'])
    A.apply_health_override(an)
    A.dump(f'{res}/analysis.json', an)
    A.dump(f'{res}/curves.json', A.curves(runs))
    A.dump(f'{res}/interp.json', dict(trajectories=A.trajectories(out, runs), lca_detail=per_run))
    for k in ('H1_symmetry', 'H2_chain_distance', 'H3_tree_lca_level', 'H5_mechanism', 'H6_distractor'):
        v = an[k].get('verdict', an[k].get('direction'))
        print(k, v)
    print('health', {k: (len(v) if isinstance(v, list) else v) for k, v in health.items()})


if __name__ == '__main__':
    main()
