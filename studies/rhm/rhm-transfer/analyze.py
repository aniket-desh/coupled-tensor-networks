"""rhm-transfer analysis: updates to threshold on unseen root productions per arm, paired by (grammar, init seed).

  PYTHONPATH=. python studies/rhm/rhm-transfer/analyze.py runs/rhm-transfer studies/rhm/rhm-transfer/results

Steps to threshold: first logged phase-2 update with unseen diagnostic accuracy >= 0.9 (+inf if the arm finished
without reaching it). Only finished arms of units that passed the pretraining gate enter a pair; other pairs are
dropped and counted. A is faster than B only if t_A < t_B - margin, margin = max(the logging interval at the later of
the two, the seed range of A's t over the units of that grammar); otherwise a tie (excluded from sign tests).
H1 (per variant V in C, WC): frozen_learned_V is faster than scratch, with at least 5 usable pairs, one-sided sign
test p < 0.05 and the per-grammar consistency rule, AND its median t_threshold is within 2x the oracle's (oracle_C, a
dense code of the true symbols, for both variants). Not supported if it is no faster than scratch (wins <= losses) or
slower than 2x the oracle. The other-grammar floor is descriptive (expected censored: its subtrees can't parse). Inconclusive if scratch did not learn (median
final unseen accuracy < 2 x chance or final unseen loss within 0.05 nats of log 16) or fewer than 5 units passed.
Writes <results>/analysis.json and <results>/curves.json."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from studies.rhm.analysis import cmp_steps, cum_lr, dump, log_gap, med_range, seed_range, steps_to, tally

ARMS = ('scratch', 'finetune', 'frozen_learned_C', 'frozen_other_C', 'frozen_random_C', 'oracle_C',
        'frozen_learned_WC', 'frozen_other_WC', 'frozen_random_WC')
THR, ALPHA, MIN_PAIRS, ORACLE_FACTOR = 0.9, 0.05, 5, 2.0
KEYS = ('step', 'seen_acc', 'unseen_acc', 'seen_loss', 'unseen_loss')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs')
    p.add_argument('results')
    a = p.parse_args()
    planned = json.load(open(Path(a.runs) / 'planned.json')) if (Path(a.runs) / 'planned.json').exists() else []
    units, curves, gate_failed, missing = {}, {}, [], []
    for name in planned or sorted(p.parent.name for p in Path(a.runs).glob('g*_s*/unit.json')):
        uf = Path(a.runs) / name / 'unit.json'
        if not uf.exists():
            missing.append(name)
            continue
        u = json.load(open(uf))
        for ph, key in (('pretrain', ('seen_acc', 'unseen_acc')), ('pretrain_other', ('own_acc',))):
            f = uf.parent / ph / 'result.json'
            if f.exists():
                r = json.load(open(f))
                curves[f'{name}/{ph}'] = {k: r['log'][k] for k in ('step', *key)}
        if not u['gate_passed']:
            gate_failed.append(name)
            continue
        rec = dict(g=u['cfg']['grammar']['seed'], seed=u['cfg']['seed'], other_gate_passed=u['other_gate_passed'], arms={})
        for arm in ARMS:
            f = uf.parent / arm / 'result.json'
            if not f.exists():
                rec['arms'][arm] = dict(status=u['arms'].get(arm, 'missing'))
                continue
            r = json.load(open(f))
            lg = r['log']
            t = steps_to(lg, 'unseen_acc', THR)
            rec['arms'][arm] = dict(status=r['status'], t_thresh=t, cumlr_thresh=cum_lr(r, t), log_steps=lg['step'],
                                    final_unseen_acc=lg['unseen_acc'][-1], final_unseen_loss=lg['unseen_loss'][-1],
                                    final_seen_acc=lg['seen_acc'][-1], unseen_acc_at_0=lg['unseen_acc'][0],
                                    n_trainable=r['n_trainable'], rescale=r['rescale'], ms_per_step=r['ms_per_step'])
            curves[f'{name}/{arm}'] = {k: lg[k] for k in KEYS}
        units[name] = rec
    fin = lambda u, arm: arm in u['arms'] and u['arms'][arm]['status'] == 'finished'

    def paired(x, y):
        oc, detail = [], []
        for n, u in units.items():
            if not (fin(u, x) and fin(u, y)):
                oc.append((u['g'], None))
                continue
            same_g = [v['arms'][x]['t_thresh'] for v in units.values() if v['g'] == u['g'] and fin(v, x)]
            tx, ty = u['arms'][x]['t_thresh'], u['arms'][y]['t_thresh']
            later = max(tx, ty) if math.isfinite(max(tx, ty)) else min(tx, ty)
            margin = max(log_gap(u['arms'][x]['log_steps'], later), seed_range(same_g))
            oc.append((u['g'], cmp_steps(tx, ty, margin)))
            detail.append(dict(unit=n, t_a=tx, t_b=ty, margin=margin))
        return dict(tally(oc), pairs=detail)

    scratch = [u['arms']['scratch'] for u in units.values() if fin(u, 'scratch')]
    scr_acc = float(np.median([s['final_unseen_acc'] for s in scratch])) if scratch else 0.0
    scr_loss = float(np.median([s['final_unseen_loss'] for s in scratch])) if scratch else math.inf
    scratch_learned = bool(scratch) and scr_acc >= 2 / 16 and scr_loss < math.log(16) - 0.05
    tmed = lambda arm: (float(np.median([u['arms'][arm]['t_thresh'] for u in units.values() if fin(u, arm)]))
                        if any(fin(u, arm) for u in units.values()) else None)
    crit = {}
    for v in ('C', 'WC'):
        L = f'frozen_learned_{v}'
        vs_floor, vs_scratch = paired(L, f'frozen_other_{v}'), paired(L, 'scratch')
        sig = [c['pairs_used'] - c['ties'] >= MIN_PAIRS and c['p_a_better'] < ALPHA and c['consistent_a']
               for c in (vs_floor, vs_scratch)]
        tl, to = tmed(L), tmed('oracle_C')
        near_oracle = tl is not None and to is not None and math.isfinite(tl) and tl <= ORACLE_FACTOR * max(to, 1.0)
        if len(units) < MIN_PAIRS:
            verdict = f'inconclusive ({len(units)} units passed the pretraining gate, need {MIN_PAIRS})'
        elif not scratch_learned:
            verdict = 'inconclusive (scratch did not learn the unseen productions)'
        elif sig[1] and near_oracle:
            verdict = 'supported'
        elif vs_scratch['a_better'] <= vs_scratch['b_better'] or (tl is not None and to is not None and not near_oracle):
            verdict = ('not supported (' + ('no faster than scratch' if vs_scratch['a_better'] <= vs_scratch['b_better']
                       else f'slower than {ORACLE_FACTOR:g}x the oracle') + ')')
        else:
            verdict = 'inconclusive (faster than scratch in a majority, not significant or not consistent across grammars)'
        # the other-grammar floor is descriptive: its subtrees can't parse this grammar, so it is expected to be censored
        crit[f'H1_transfer_{v}'] = dict(descriptive_learned_vs_other_grammar_floor=vs_floor, learned_vs_scratch=vs_scratch,
                                        median_t_learned=tl, median_t_oracle=to, within_oracle_factor=near_oracle,
                                        verdict=verdict)
        crit[f'descriptive_learned_vs_random_{v} (learned features beat a near-additive map of the tokens)'] = \
            paired(L, f'frozen_random_{v}')
    crit['descriptive_finetune_vs_scratch'] = paired('finetune', 'scratch')
    crit['descriptive_finetune_vs_frozen_learned_C'] = paired('finetune', 'frozen_learned_C')
    table = {arm: dict(finished=sum(fin(u, arm) for u in units.values()), of=len(units),
                       t_thresh=med_range([u['arms'][arm]['t_thresh'] for u in units.values() if fin(u, arm)]),
                       cumlr_thresh=med_range([u['arms'][arm]['cumlr_thresh'] for u in units.values() if fin(u, arm)]),
                       final_unseen_acc=med_range([u['arms'][arm]['final_unseen_acc'] for u in units.values() if fin(u, arm)]),
                       final_seen_acc=med_range([u['arms'][arm]['final_seen_acc'] for u in units.values() if fin(u, arm)]),
                       ms_per_step=med_range([u['arms'][arm]['ms_per_step'] for u in units.values() if fin(u, arm)]))
             for arm in ARMS}
    for u in units.values():
        for v in u['arms'].values():
            v.pop('log_steps', None)
    lr = json.load(open(Path(a.runs) / 'lr.json')) if (Path(a.runs) / 'lr.json').exists() else None
    res = dict(planned_units=len(planned), missing_units=missing, gate_failed=gate_failed, units=units, table=table,
               criteria=crit, threshold=THR, scratch_learned=dict(learned=scratch_learned, median_final_unseen_acc=scr_acc,
                                                                  median_final_unseen_loss=scr_loss), lr=lr)
    dump(Path(a.results) / 'analysis.json', res)
    dump(Path(a.results) / 'curves.json', curves)
    print(f'{len(units)} units passed the gate, {len(gate_failed)} failed {gate_failed}, {len(missing)} missing')
    for k, v in crit.items():
        if 'verdict' in v:
            print(f"{k}: {v['verdict']}")


if __name__ == '__main__':
    main()
