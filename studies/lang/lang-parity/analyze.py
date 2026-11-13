"""lang-parity analysis: degree, init scale and input bias effects on discovery time, and the causal partial-parity
mechanism and its order of appearance.

  PYTHONPATH=. python studies/lang/lang-parity/analyze.py runs/lang-parity <results dir>

Rules are in EXPERIMENT.md; shared definitions (t, faster, censoring) in studies/lang/analysis.py. Pairs where both
runs are censored are uninformative and excluded from the counts (reported as both_censored)."""
import math
import sys

from studies.lang import analysis as A

SEEDS = (0, 1, 2)
KS, EPS, BIAS = (2, 4, 8, 16), (0.01, 0.1, 0.3, 1.0), (0.0, 0.2)
EPS_MAIN = (0.01, 0.1, 0.3)  # eps = 1.0 starts with logits in the hundreds (a separate regime, reported apart)


def nm(k, e, b, s, arm='main'):
    return f'{arm}_k{k}_e{e:g}_b{b:g}_s{s}'


def big(t):
    return t if math.isfinite(t) else 1e12


def causal_gap(ip, step, node):
    sub = A.get(ip, step, node, 'par', 'patch', 'sub', 'moved')
    rnd = A.get(ip, step, node, 'par', 'patch', 'rand', 'moved')
    return None if sub is None or rnd is None or (isinstance(sub, float) and math.isnan(sub)) else sub - rnd


def mechanism(out, runs, rows):
    """H4: causal patching of the partial-parity subspace (deciding), with the probe and the additive floor reported.
    H5: per level, the first checkpoint at which the median causal gap (sub moved - random moved) over the level's
    nodes with >= 2 of the k bits reaches 0.5, compared with the run's t."""
    sub, rnd, probe, local, order, n_ip = [], [], [], [], {}, 0
    for name, r in runs.items():
        if (r['cfg']['arm'] != 'main' or r['cfg']['process']['bias'] != 0 or not A.finished(r)
                or not math.isfinite(rows[name]['t'])):
            continue
        ip = A.load_interp(out, name)
        if ip is None:
            continue
        n_ip += 1
        st = A.steps_of(ip)
        sf = st[-1]
        k = r['cfg']['process']['k']
        spans = A.node_spans(r['cfg'])
        nodes_by_level = {}
        for node, S in spans.items():
            m = sum(t < k for t in S)
            lvl = int(node.split('_')[0])
            if m >= 1 and lvl >= 1:
                sub.append(A.get(ip, sf, node, 'par', 'patch', 'sub', 'moved'))
                rnd.append(A.get(ip, sf, node, 'par', 'patch', 'rand', 'moved'))
            if m >= 2:
                probe.append(A.get(ip, sf, node, 'par', 'probe'))
                local.append(dict(bits=m, local=A.get(ip, sf, node, 'par', 'local')))
                nodes_by_level.setdefault(lvl, []).append(node)
        if k >= 4 and nodes_by_level:
            first = {}
            for lvl, nodes in sorted(nodes_by_level.items()):
                first[lvl] = next((int(s) for s in st
                                   if A.med([causal_gap(ip, s, nd) for nd in nodes]) >= 0.5), None)
            ck = [int(s) for s in st]
            interval = lambda s: next((i for i, c in enumerate(ck) if c >= s), len(ck))
            vals = [first[l] for l in sorted(first)]
            t = rows[name]['t']
            if any(v is None for v in vals):
                cls = 'not every level causal'
            elif len({interval(v) for v in vals} | {interval(t)}) == 1:
                cls = 'simultaneous'
            elif all(a < b for a, b in zip(vals, vals[1:])):
                cls = 'bottom-up'
            else:
                cls = 'other'
            order[name] = dict(first_causal_by_level=first, t=t, cls=cls)
    ms, mr = A.med(sub), A.med(rnd)
    crit = dict(causal=dict(median_sub_moved=ms, median_rand_moved=mr, n=len([x for x in sub if x is not None]),
                            ok=bool([x for x in sub if x is not None]) and ms >= 0.8 and ms - mr >= 0.4),
                probe_descriptive=dict(median=A.med(probe), n=len(probe),
                                       note='implied by accuracy for a multilinear tree on an aligned parity, and '
                                            'often met at initialization after whitening'),
                additive_floor=dict(median_local_2bit=A.med([x['local'] for x in local if x['bits'] == 2]),
                                    median_local_more=A.med([x['local'] for x in local if x['bits'] > 2]),
                                    note='a property of the data: a linear classifier gets 3 of 4 XOR points (0.75)'),
                runs_interpreted=n_ip, order=order,
                order_counts={c: sum(o['cls'] == c for o in order.values())
                              for c in ('bottom-up', 'simultaneous', 'other', 'not every level causal')})
    crit['verdict'] = ('supported' if crit['causal']['ok'] else 'not supported') if n_ip else \
        'inconclusive (no interpreted learned runs)'
    return crit


def pair_verdict(c, n_min, frac_win, max_loss):
    if c['n'] < n_min:
        return f'inconclusive (fewer than {n_min} informative pairs)'
    if c['wins'] >= frac_win * c['n'] and c['losses'] <= max_loss:
        return 'supported'
    if c['losses'] >= frac_win * c['n'] and c['wins'] <= max_loss:
        return 'opposite direction'
    return 'not supported'


def main():
    out, res = sys.argv[1], sys.argv[2]
    runs, planned = A.load_runs(out)
    R = lambda n: runs.get(n)
    rows = {n: A.row(r) for n, r in runs.items()}
    for n, r in runs.items():
        rows[n]['initial_diag_loss'] = r.get('initial_diag_loss')
    fin = [n for n in planned if n in runs and A.finished(runs[n])]
    an = dict(health=dict(planned=len(planned), finished=len(fin), missing=[n for n in planned if n not in fin],
                          collapsed=[n for n in fin if rows[n]['collapsed']]), runs=rows)
    T = lambda k, e, b, arm='main': [rows.get(nm(k, e, b, s, arm), {}).get('t') for s in SEEDS]
    an['table'] = {f'k{k}_e{e:g}_b{b:g}': dict(t=T(k, e, b), median=A.med([big(t) for t in T(k, e, b) if t is not None]),
                                              censored=sum(t == A.INF for t in T(k, e, b) if t is not None))
                   for k in KS for e in EPS for b in BIAS}
    an['initial_loss_by_eps'] = {f'{e:g}': A.med([rows.get(nm(k, e, 0.0, s), {}).get('initial_diag_loss') for k in KS for s in SEEDS])
                                 for e in EPS}
    # H1 degree (bias 0), per eps: no adjacent (k, 2k) pair where 2k is faster in >= 2 seeds, and k = 2 faster than
    # k = 16 in >= 2 informative seeds with none the other way
    per_eps = {}
    for e in EPS:
        viol = {}
        for k1, k2 in ((2, 4), (4, 8), (8, 16)):
            c = A.compare([(R(nm(k2, e, 0.0, s)), R(nm(k1, e, 0.0, s))) for s in SEEDS], A.faster)
            viol[f'{k2}_faster_than_{k1}'] = c
        c216 = A.compare([(R(nm(2, e, 0.0, s)), R(nm(16, e, 0.0, s))) for s in SEEDS], A.faster)
        ok = all(v['wins'] < 2 for v in viol.values()) and c216['wins'] >= 2 and c216['losses'] == 0
        per_eps[f'{e:g}'] = dict(medians=[A.med([big(t) for t in T(k, e, 0.0) if t is not None]) for k in KS],
                                 adjacent=viol, k2_vs_k16=c216, ok=ok)
    n_ok = sum(v['ok'] for v in per_eps.values())
    an['H1_degree'] = dict(per_eps=per_eps, verdict='supported' if n_ok == 4 else ('partly supported' if n_ok >= 2 else 'not supported'))
    # H2 init scale: eps 0.3 vs eps 0.01 at k in {4, 8}, bias 0 (6 pairs; both-censored pairs uninformative)
    c = A.compare([(R(nm(k, 0.3, 0.0, s)), R(nm(k, 0.01, 0.0, s))) for k in (4, 8) for s in SEEDS], A.faster)
    per_k = {k: A.compare([(R(nm(k, 0.3, 0.0, s)), R(nm(k, 0.01, 0.0, s))) for s in SEEDS], A.faster) for k in (4, 8)}
    an['H2_init_scale'] = dict(c, per_k=per_k, verdict=pair_verdict(c, 3, 0.8, 0),
                               medians={f'k{k}': {f'{e:g}': A.med([big(t) for t in T(k, e, 0.0) if t is not None]) for e in EPS} for k in KS},
                               eps1_regime=dict(initial_loss=an['initial_loss_by_eps']['1'],
                                                vs_eps001=A.compare([(R(nm(k, 1.0, 0.0, s)), R(nm(k, 0.01, 0.0, s))) for k in (4, 8) for s in SEEDS], A.faster),
                                                vs_eps01=A.compare([(R(nm(k, 1.0, 0.0, s)), R(nm(k, 0.1, 0.0, s))) for k in (4, 8) for s in SEEDS], A.faster)))
    # H3 input bias: bias 0.2 vs 0 at k in {4, 8}, eps in {0.01, 0.1, 0.3} (18 pairs), t on the unbiased diagnostic set
    c = A.compare([(R(nm(k, e, 0.2, s)), R(nm(k, e, 0.0, s))) for k in (4, 8) for e in EPS_MAIN for s in SEEDS], A.faster)
    an['H3_bias'] = dict(c, verdict=pair_verdict(c, 6, 0.75, 1),
                         eps1=A.compare([(R(nm(k, 1.0, 0.2, s)), R(nm(k, 1.0, 0.0, s))) for k in (4, 8) for s in SEEDS], A.faster))
    an['H4_mechanism'] = mechanism(out, runs, rows)
    an['H5_order'] = dict(counts=an['H4_mechanism']['order_counts'])
    an['desc_mod4'] = {f'k{k}': dict(t=T(k, 0.1, 0.0, 'mod4'), kl=A.med([rows.get(nm(k, 0.1, 0.0, s, 'mod4'), {}).get('kl') for s in SEEDS])) for k in (2, 4, 8)}
    an['desc_r2'] = {f'k{k}': dict(t=T(k, 0.1, 0.0, 'r2'), t_r16=T(k, 0.1, 0.0), kl=A.med([rows.get(nm(k, 0.1, 0.0, s, 'r2'), {}).get('kl') for s in SEEDS])) for k in (4, 8)}
    A.apply_health_override(an)
    A.dump(f'{res}/analysis.json', an)
    A.dump(f'{res}/curves.json', A.curves(runs))
    A.dump(f'{res}/interp.json', dict(trajectories=A.trajectories(out, runs)))
    for k in ('H1_degree', 'H2_init_scale', 'H3_bias', 'H4_mechanism'):
        print(k, an[k]['verdict'])
    print('order', an['H5_order']['counts'], 'initial loss by eps', an['initial_loss_by_eps'])
    print('health', {k: (len(v) if isinstance(v, list) else v) for k, v in an['health'].items()})


if __name__ == '__main__':
    main()
