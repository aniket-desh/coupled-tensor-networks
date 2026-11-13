"""lang-induction analysis: width, alignment and stratified capacity on retrieval; width on multi-match induction;
staging of association storage vs the loss drop.

  PYTHONPATH=. python studies/lang/lang-induction/analyze.py runs/lang-induction <results dir>

Rules are in EXPERIMENT.md; shared definitions in studies/lang/analysis.py."""
import sys

from studies.lang import analysis as A

SEEDS = (0, 1, 2)
WIDTHS = (16, 32, 64, 128)


def kl_by_width(rows, pn):
    return {r: [rows.get(f'{pn}_tree{r}_s{s}', {}).get('kl') for s in SEEDS] for r in WIDTHS}


def converged(r, window=10_000):
    """Diagnostic KL fell by < 5% (or < 0.002 nats) over the last `window` updates of the run."""
    S, K = r['log']['step'], r['log']['diag_kl']
    end = S[-1]
    i = next(j for j, s in enumerate(S) if s >= end - window)
    return (K[i] - K[-1]) < max(0.05 * K[i], 0.002)


def select_width(rows, runs, pn):
    """Smallest stable width: every seed at r and 2r finished and not collapsed, median KL < 0.01 nats, and doubling
    improves the median KL by < 0.002 nats. 'unresolved (still improving)' if the candidate's runs haven't converged."""
    kls = kl_by_width(rows, pn)
    meds = {r: A.med(v) for r, v in kls.items()}
    stable = {r: all((n := f'{pn}_tree{r}_s{s}') in runs and A.finished(runs[n]) and not rows[n]['collapsed'] for s in SEEDS)
              for r in WIDTHS}
    conv = {r: all(converged(runs[f'{pn}_tree{r}_s{s}']) for s in SEEDS if f'{pn}_tree{r}_s{s}' in runs) for r in WIDTHS}
    for r in WIDTHS[:-1]:
        if stable[r] and stable[2 * r] and meds[r] < 0.01 and meds[r] - meds[2 * r] < 0.002:
            return r, meds, stable, conv
    last = WIDTHS[-1]
    if stable[last] and meds[last] < 0.01:
        return last, meds, stable, conv
    return (None if conv[last] else 'unresolved (still improving)'), meds, stable, conv


def strata_acc(row, keys):
    st = row.get('strata') or {}
    n = sum(st[k]['n'] for k in keys if k in st)
    return sum(st[k]['acc'] * st[k]['n'] for k in keys if k in st) / n if n else None


def staging(out, runs, rows):
    res = {}
    for name, r in runs.items():
        if (r['cfg']['process']['kind'] != 'retrieval' or r['cfg']['model'] != 'tree' or not r['cfg']['arm'].startswith('tree')
                or not A.finished(r)):
            continue
        ip = A.load_interp(out, name)
        if ip is None or rows[name]['t'] == A.INF:
            continue
        st = A.steps_of(ip)
        ck = [int(s) for s in st]
        r2 = [A.get(ip, s, '3_0', 'dict', 'probe') for s in st]
        if any(v is None for v in r2):
            continue
        if r2[-1] <= r2[0]:
            res[name] = dict(cls='no gain', r2_final=r2[-1], r2_step0=r2[0], steps=ck, r2=r2)
            continue
        d_half = next(c for c, v in zip(ck, r2) if v >= r2[0] + 0.5 * (r2[-1] - r2[0]))
        L, S = r['log']['diag_loss'], r['log']['step']
        l_half = next(s for s, v in zip(S, L) if v <= L[0] - 0.5 * (L[0] - L[-1]))
        idx = lambda s: next((i for i, c in enumerate(ck) if c >= s), len(ck))
        cls = 'together' if idx(d_half) == idx(l_half) else ('dict first' if d_half < l_half else 'loss first')
        res[name] = dict(dict_half=d_half, loss_half=l_half, cls=cls, r2_final=r2[-1], r2_step0=r2[0], steps=ck, r2=r2,
                         patch_sub=A.get(ip, st[-1], '3_0', 'dict', 'patch', 'sub'),
                         patch_rand=A.get(ip, st[-1], '3_0', 'dict', 'patch', 'rand'))
    return dict(runs=res, counts={c: sum(v['cls'] == c for v in res.values()) for c in ('dict first', 'together', 'loss first', 'no gain')})


def main():
    out, res = sys.argv[1], sys.argv[2]
    runs, planned = A.load_runs(out)
    R = lambda n: runs.get(n)
    rows = {n: A.row(r) for n, r in runs.items()}
    fin = [n for n in planned if n in runs and A.finished(runs[n])]
    an = dict(health=dict(planned=len(planned), finished=len(fin), missing=[n for n in planned if n not in fin],
                          collapsed=[n for n in fin if rows[n]['collapsed']]), runs=rows)
    # H1 retrieval width
    kls = kl_by_width(rows, 'retrieval')
    c = A.compare([(R(f'retrieval_tree128_s{s}'), R(f'retrieval_tree16_s{s}')) for s in SEEDS], A.better)
    sel, meds, stable, conv = select_width(rows, runs, 'retrieval')
    an['H1_retrieval_width'] = dict(c, kl=kls, median_kl=meds, selected_width=sel, stable=stable, converged=conv,
                                    collapsed_by_width={r: sum(rows.get(f'retrieval_tree{r}_s{s}', {}).get('collapsed', False) for s in SEEDS) for r in WIDTHS},
                                    verdict=('inconclusive (fewer than 2 usable pairs)' if c['n'] < 2 else
                                             'supported' if c['wins'] == 3 else 'not supported'))
    # H2 stratified capacity: the only bond that can't hold its part at r = 16 is the left half (node 3_0, rank >= 29;
    # 2_2: 15, 2_3: 13, 3_1: 11), so capacity predicts acc(mid: pairs 4-5) ~ acc(near: pair 6) > acc(far: pairs 0-3),
    # while distance alone predicts a graded near > mid > far. Seeds whose near accuracy at r = 16 is below
    # 0.8 x Bayes did not learn and are not scored.
    pat, learned, shrink, detail = 0, 0, 0, {}
    for s in SEEDS:
        r16, r128 = runs.get(f'retrieval_tree16_s{s}'), runs.get(f'retrieval_tree128_s{s}')
        if r16 is None or not A.finished(r16):
            detail[s] = 'missing'
            continue
        row = rows[r16['name']]
        far, mid, near = (strata_acc(row, k) for k in (['0', '1', '2', '3'], ['4', '5'], ['6']))
        if near is None or near < 0.8 * row['bayes_acc']:
            detail[s] = dict(far=far, mid=mid, near=near, cls='did not learn')
            continue
        learned += 1
        ok = abs(mid - near) <= 0.02 and min(mid, near) > far + 0.05
        pat += ok
        g16 = mid - far
        g128 = None
        if r128 is not None and A.finished(r128):
            row2 = rows[r128['name']]
            g128 = strata_acc(row2, ['4', '5']) - strata_acc(row2, ['0', '1', '2', '3'])
            shrink += g128 < 0.5 * g16
        detail[s] = dict(far=far, mid=mid, near=near, capacity_pattern=ok, gap_mid_far_r16=g16, gap_mid_far_r128=g128,
                         graded=near > mid > far)
    an['H2_retrieval_strata'] = dict(per_seed=detail, learned_seeds=learned, capacity_pattern_seeds=pat, shrink_seeds=shrink,
                                     per_stratum_acc={r: {m: A.med([strata_acc(rows.get(f'retrieval_tree{r}_s{s}', {}), [str(m)]) for s in SEEDS]) for m in range(7)} for r in WIDTHS},
                                     verdict=('inconclusive (fewer than 2 learned seeds at r = 16)' if learned < 2 else
                                              'supported' if pat == learned and shrink >= learned - 1 else
                                              'partly supported' if pat == learned else 'not supported'))
    # H3 tree vs perm at equal width (retrieval)
    c = A.compare([(R(f'retrieval_tree{r}_s{s}'), R(f'retrieval_perm{r}_s{s}')) for r in (16, 64) for s in SEEDS], A.better)
    sp = A.compare([(R(f'retrieval_tree{r}_s{s}'), R(f'retrieval_perm{r}_s{s}')) for r in (16, 64) for s in SEEDS], A.faster)
    an['H3_retrieval_tree_vs_perm'] = dict(quality=c, speed=sp,
                                           verdict=('inconclusive (fewer than 4 usable pairs)' if c['n'] < 4 else
                                                    'supported' if c['wins'] >= 5 and c['losses'] == 0 else
                                                    'opposite direction' if c['losses'] >= 5 and c['wins'] == 0 else 'not supported'))
    # H4 multimatch width
    kls_m = kl_by_width(rows, 'multimatch')
    c = A.compare([(R(f'multimatch_tree128_s{s}'), R(f'multimatch_tree16_s{s}')) for s in SEEDS], A.better)
    sel_m, meds_m, stable_m, conv_m = select_width(rows, runs, 'multimatch')
    gap = meds_m[128] > 0.01
    an['H4_multimatch_width'] = dict(c, kl=kls_m, median_kl=meds_m, selected_width=sel_m, stable=stable_m, converged=conv_m,
                                     gap_remains_at_128=gap if conv_m[128] else 'unresolved (r = 128 still improving)',
                                     strata={r: {m: A.med([((rows.get(f'multimatch_tree{r}_s{s}', {}).get('strata') or {}).get(str(m)) or {}).get('kl') for s in SEEDS]) for m in (1, 2, 3)} for r in WIDTHS},
                                     verdict=('inconclusive (fewer than 2 usable pairs)' if c['n'] < 2 else
                                              'not supported' if c['wins'] < 3 else
                                              'partly supported (r = 128 still improving)' if gap and not conv_m[128] else
                                              'supported' if gap else 'partly supported'))
    an['desc_multimatch_tree_vs_perm'] = dict(
        quality=A.compare([(R(f'multimatch_tree{r}_s{s}'), R(f'multimatch_perm{r}_s{s}')) for r in (16, 64) for s in SEEDS], A.better))
    for pn in ('retrieval', 'multimatch'):
        an[f'desc_{pn}_tree_vs_chain'] = {r: dict(
            quality=A.compare([(R(f'{pn}_tree{r}_s{s}'), R(f'{pn}_chain{r}_s{s}')) for s in SEEDS], A.better),
            speed=A.compare([(R(f'{pn}_tree{r}_s{s}'), R(f'{pn}_chain{r}_s{s}')) for s in SEEDS], A.faster)) for r in (16, 64)}
    an['H5_staging'] = staging(out, runs, rows)
    A.apply_health_override(an)
    A.dump(f'{res}/analysis.json', an)
    A.dump(f'{res}/curves.json', A.curves(runs))
    A.dump(f'{res}/interp.json', dict(trajectories=A.trajectories(out, runs)))
    for k in ('H1_retrieval_width', 'H2_retrieval_strata', 'H3_retrieval_tree_vs_perm', 'H4_multimatch_width'):
        print(k, an[k]['verdict'])
    print('staging', an['H5_staging']['counts'], 'selected widths', sel, sel_m)
    print('health', {k: (len(v) if isinstance(v, list) else v) for k, v in an['health'].items()})


if __name__ == '__main__':
    main()
