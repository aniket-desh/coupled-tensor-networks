"""rhm-controls analysis: paired comparisons between architectures.

  PYTHONPATH=. python studies/rhm/rhm-controls/analyze.py runs/rhm-controls studies/rhm/rhm-controls/results

Comparisons (A vs B; A is the reference whose seed range sets the tie margins and which must have learned):
  H1  tree vs perm     does matching the data hierarchy help
  H2  tree vs tied     does weight sharing help or hurt (two-sided)
  H3  tied vs chain    does the hierarchy matter beyond a tensor parameterization: both share one leaf embedding
                       across positions and start position-blind, so they differ in structure, not in positional freedom
  descriptive: tree vs chain (also differs by position-specific parameters, so it can't carry H3)
Measures per pair (task, grammar, init seed; 12 pairs): steps to threshold and final quality (error rate, then loss).
A measure goes to A (or B) if it wins at least 10 of the 12 pairs (ties count for neither; one-sided sign test
p = 0.019 at 10/12 without ties) and the per-grammar consistency rule holds. Pair, tie and margin rules:
studies/rhm/analysis.py. Writes <results>/analysis.json and <results>/curves.json."""
import argparse
import json
import math
from pathlib import Path

from studies.rhm.analysis import (FLOOR, cmp_quality, cmp_steps, cum_lr, curves, dump, health, learned, load_runs,
                                  log_gap, lr_info, med_range, ok, seed_range, t_thresh, tally)

ARMS, TASKS, G3, SEEDS = ('tree', 'perm', 'tied', 'chain'), ('root', 'last'), (0, 1, 2), (0, 1)
NEED = 10


def compare(by, a, b, tasks=TASKS):
    st, q, detail = [], [], []
    for t in tasks:
        for g in G3:
            refs = [by.get((t, a, g, s)) for s in SEEDS]
            refs = [r for r in refs if ok(r)]
            nz_t = seed_range([t_thresh(r) for r in refs])
            nz_e = seed_range([1 - r['test']['acc'] for r in refs])
            nz_l = seed_range([r['test']['loss'] for r in refs])
            for s in SEEDS:
                ra, rb = by.get((t, a, g, s)), by.get((t, b, g, s))
                if not (ok(ra) and ok(rb)):
                    st.append((g, None))
                    q.append((g, None))
                    continue
                ta, tb = t_thresh(ra), t_thresh(rb)
                later = max(ta, tb) if math.isfinite(max(ta, tb)) else min(ta, tb)
                margin = max(log_gap(ra['log']['step'], later), nz_t)
                st.append((g, cmp_steps(ta, tb, margin)))
                q.append((g, cmp_quality(ra, rb, max(FLOOR, nz_e), max(FLOOR, nz_l))))
                detail.append(dict(task=t, g=g, seed=s, t_a=ta, t_b=tb, margin=margin, cumlr_a=cum_lr(ra, ta),
                                   cumlr_b=cum_lr(rb, tb), err_a=1 - ra['test']['acc'], err_b=1 - rb['test']['acc'],
                                   loss_a=ra['test']['loss'], loss_b=rb['test']['loss']))
    return dict(steps=tally(st), quality=tally(q), pairs=detail)


def direction(tl):
    return 'a' if tl['a_better'] >= NEED and tl['consistent_a'] else 'b' if tl['b_better'] >= NEED and tl['consistent_b'] else None


def verdict(c, a, b, ref_ok, health_ok, two_sided=False):
    if not health_ok:
        return 'inconclusive (more than a quarter of planned runs did not finish)'
    if not ref_ok:
        return f'inconclusive (the reference arm {a} did not learn)'
    used = min(c['steps']['pairs_used'], c['quality']['pairs_used'])
    if used < 12:
        return f'inconclusive ({used} of 12 pairs usable)'
    d = [direction(c['steps']), direction(c['quality'])]
    if d == ['a', 'a']:
        return f'{a} better on both measures' + ('' if two_sided else ' (supported)')
    if d == ['b', 'b']:
        return f'{b} better on both measures' + ('' if two_sided else ' (not supported)')
    if 'a' in d and 'b' not in d:
        return f'{a} better on one measure' + (' (partly)' if two_sided else ' (partly supported)')
    if 'b' in d and 'a' not in d:
        return f'{b} better on one measure' + ('' if two_sided else ' (not supported)')
    return 'no consistent difference' if two_sided else 'not supported (no consistent difference)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs')
    p.add_argument('results')
    a = p.parse_args()
    runs = load_runs(a.runs)
    by = {(r['cfg']['task'], r['cfg']['arm'], r['cfg']['grammar']['seed'], r['cfg']['seed']): r for r in runs.values()}
    hl = health(a.runs, runs)
    health_ok = hl['frac_not_finished'] <= 0.25
    learn = {arm: {t: learned([by.get((t, arm, g, s)) for g in G3 for s in SEEDS], 16) for t in TASKS} for arm in ARMS}
    ref_ok = lambda arm: all(v['learned'] for v in learn[arm].values())
    crit = {}
    for name, x, y, two in (('H1_hierarchy_match (tree vs perm)', 'tree', 'perm', False),
                            ('H2_weight_sharing (tree vs tied; two-sided)', 'tree', 'tied', True),
                            ('H3_hierarchy_beyond_tensor_param (tied vs chain)', 'tied', 'chain', False),
                            ('descriptive_tree_vs_chain', 'tree', 'chain', True)):
        c = compare(by, x, y)
        c['per_task'] = {t: {k: v for k, v in compare(by, x, y, (t,)).items() if k != 'pairs'} for t in TASKS}
        c['a'], c['b'], c['verdict'] = x, y, verdict(c, x, y, ref_ok(x), health_ok, two)
        c['other_arm_learned'] = learn[y]
        if not ref_ok(y) and not c['verdict'].startswith('inconclusive'):
            # a control stuck at the uniform-output saddle (or at a bad LR) hands every pair to the reference: flag it
            c['verdict'] += f' [but {y} did not learn on some task: possibly an optimization failure, check its LR pilot]'
        crit[name] = c
    table = {}
    for t in TASKS:
        for arm in ARMS:
            rs = [by.get((t, arm, g, s)) for g in G3 for s in SEEDS]
            fin = [r for r in rs if ok(r)]
            table[f'{t}_{arm}'] = dict(planned=len(rs), finished=len(fin),
                                       t_thresh=med_range([t_thresh(r) for r in fin]),
                                       test_err=med_range([1 - r['test']['acc'] for r in fin]),
                                       test_loss=med_range([r['test']['loss'] for r in fin]),
                                       n_params=sorted({r['n_params'] for r in fin}), lr=sorted({r['lr'] for r in fin}),
                                       ms_per_step=med_range([r['ms_per_step'] for r in rs if r]))
    meta = Path(a.runs) / 'controls_meta.json'
    res = dict(health=hl, learned=learn, criteria=crit, table=table, lr=lr_info(a.runs),
               meta=json.load(open(meta)) if meta.exists() else None)
    dump(Path(a.results) / 'analysis.json', res)
    dump(Path(a.results) / 'curves.json', curves(runs))
    for k, v in crit.items():
        print(f"{k}: {v['verdict']}")


if __name__ == '__main__':
    main()
