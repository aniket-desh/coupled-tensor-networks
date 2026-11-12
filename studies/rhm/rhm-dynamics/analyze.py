"""rhm-dynamics analysis: per-run metrics, the pre-registered criteria, compact curves for plot.py, and (as a separate
step with its own time limit) the interpretation CLI on the saved checkpoints.

  PYTHONPATH=. python studies/rhm/rhm-dynamics/analyze.py runs/rhm-dynamics studies/rhm/rhm-dynamics/results
  PYTHONPATH=. python studies/rhm/rhm-dynamics/analyze.py runs/rhm-dynamics studies/rhm/rhm-dynamics/results \\
      --interpret-only --interpret-hours 3 [--interpret-procs 4]

The first form writes <results>/analysis.json and <results>/curves.json. The second runs, identity-init r = 16 runs
first, then the others, `--interpret-procs` at a time, starting no new run after `--interpret-hours`:
  python -m studies.rhm.interpret <run dir> --out <run dir>/interpret.json --diag <out>/data/diag_<m>_<g>.pt --device D
(D = cuda when available, else cpu; output in <run dir>/interpret.log), records the status in analysis.json, and is
skipped with a note if studies.rhm.interpret is not importable.

Comparison rules (pairing, ties, margins, dropped pairs, the did-the-control-learn rule): studies/rhm/analysis.py."""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from studies.rhm.analysis import (FINISHED, FLOOR, cmp_quality, cmp_steps, cum_lr, curves, dump, health, learned,
                                  load_runs, log_gap, lr_info, med_range, ok, seed_range, t_thresh, tally)

G3, TASKS = (0, 1, 2), ('root', 'last')
KL_GOOD, KL_BAD, ACC_GOOD, ACC_BAD = 0.05, 0.2, 0.99, 0.9


def key(r):
    c = r['cfg']
    return (c['task'], c['grammar']['m'], c['grammar']['seed'], c['r'], c['seed'], c.get('init', 'variance'))


def summarize(r):
    t = t_thresh(r)
    return dict(task=r['cfg']['task'], m=r['cfg']['grammar']['m'], g=r['cfg']['grammar']['seed'], r=r['cfg']['r'],
                seed=r['cfg']['seed'], init=r['cfg'].get('init', 'variance'), status=r['status'], steps=r['steps_done'],
                lr=r['lr'], ms_per_step=r['ms_per_step'], test_loss=r['test']['loss'], test_err=1 - r['test']['acc'],
                test_kl=r['test']['kl'], bayes_loss=r['bayes_test']['loss'], bayes_acc=r['bayes_test']['acc'],
                t_thresh=t, cumlr_thresh=cum_lr(r, t), t_inside_warmup=t <= r['warmup'])


def noise(by, task, g):
    """r = 16 init/data-order seed ranges (seeds 0-2, identity init, m = 4) on (task, grammar): the tie margins."""
    rs = [by.get((task, 4, g, 16, s, 'identity')) for s in (0, 1, 2)]
    rs = [r for r in rs if ok(r)]
    return dict(t=seed_range([t_thresh(r) for r in rs]), err=seed_range([1 - r['test']['acc'] for r in rs]),
                loss=seed_range([r['test']['loss'] for r in rs]), n_seeds=len(rs))


def verdict(tl, need, against, base_ok, health_ok, cond=''):
    """need: wins (of 6) for 'supported'; against: losses for 'not supported'."""
    if not health_ok:
        return 'inconclusive (more than a third of planned runs did not finish)'
    if not base_ok:
        return 'inconclusive (the r = 16 reference did not learn)'
    if tl['pairs_used'] < 4:
        return f"inconclusive ({tl['pairs_used']} usable pairs, {tl['pairs_dropped']} dropped)"
    if tl['a_better'] >= need and tl['consistent_a']:
        v = 'supported'
    elif tl['b_better'] >= against:
        v = 'not supported'
    else:
        v = 'partly supported' if tl['a_better'] > tl['b_better'] else 'not supported'
    return v + cond


def criteria(runs, out):
    by = {key(r): r for r in runs.values()}
    hl = health(out, runs)
    health_ok = hl['frac_not_finished'] <= 1 / 3
    res = dict(health=hl)
    ref = {t: learned([by.get((t, 4, g, 16, 0, 'identity')) for g in G3], 16) for t in TASKS}
    base_ok = all(v['learned'] for v in ref.values())
    res['reference_r16_learned'] = ref
    # H1 learnability at r = 16 (m = 4, seed 0).
    root = [by.get(('root', 4, g, 16, 0, 'identity')) for g in G3]
    last = [by.get(('last', 4, g, 16, 0, 'identity')) for g in G3]
    root_acc = [r['test']['acc'] if ok(r) else None for r in root]
    last_kl = [r['test']['kl'] if ok(r) else None for r in last]
    n = sum(v is not None for v in root_acc + last_kl)
    good = sum(a is not None and a >= ACC_GOOD for a in root_acc) + sum(k is not None and k <= KL_GOOD for k in last_kl)
    bad = (sum(a is not None and a < ACC_BAD for a in root_acc) >= 2) or (sum(k is not None and k > KL_BAD for k in last_kl) >= 2)
    res['H1_learnable_r16'] = dict(root_test_acc=root_acc, last_test_kl=last_kl, verdict=(
        'inconclusive (more than a third of planned runs did not finish)' if not health_ok else
        f'inconclusive ({6 - n} of 6 runs missing or unfinished)' if n < 6 else
        'supported' if good == 6 else 'not supported' if bad else 'partly supported'))
    # LR preferences of the other widths (pilot on grammar 0): H2 is conditional on the shared LR if they differ.
    lr = lr_info(out)['choices']
    differ = [f'{t}_r{w}' for t in TASKS for w in (8, 64) if f'{t}_r{w}' in lr and t in lr
              and lr[f'{t}_r{w}']['lr'] != lr[t]['lr']]
    cond = f' (conditional on the shared LR: {", ".join(differ)} preferred another LR in the pilot)' if differ else ''
    # H2a: r = 16 better final quality than r = 8 (error rate, then loss), per (task, grammar), seed 0.
    oc = []
    for t in TASKS:
        for g in G3:
            a, b = by.get((t, 4, g, 16, 0, 'identity')), by.get((t, 4, g, 8, 0, 'identity'))
            nz = noise(by, t, g)
            oc.append((g, cmp_quality(a, b, max(FLOOR, nz['err']), max(FLOOR, nz['loss'])) if ok(a) and ok(b) else None))
    tl = tally(oc)
    res['H2a_r16_better_than_r8'] = dict(**tl, verdict=verdict(tl, 5, 3, base_ok, health_ok, cond))
    # H2b: r = 64 (and r = 32, descriptive) reach threshold earlier than r = 16, beyond the margin.
    for wide in (64, 32):
        oc, detail = [], []
        for t in TASKS:
            for g in G3:
                a, b = by.get((t, 4, g, wide, 0, 'identity')), by.get((t, 4, g, 16, 0, 'identity'))
                if not (ok(a) and ok(b)):
                    oc.append((g, None))
                    continue
                ta, tb = t_thresh(a), t_thresh(b)
                margin = max(log_gap(a['log']['step'], max(ta, tb) if math.isfinite(max(ta, tb)) else min(ta, tb)),
                             noise(by, t, g)['t'])
                oc.append((g, cmp_steps(ta, tb, margin)))
                detail.append(dict(task=t, g=g, t_wide=ta, t_r16=tb, margin=margin, cumlr_wide=cum_lr(a, ta),
                                   cumlr_r16=cum_lr(b, tb)))
        tl = tally(oc)
        v = verdict(tl, 5, 3, base_ok, health_ok, cond)
        res[f'H2b_r{wide}_faster_than_r16'] = dict(**tl, pairs=detail, verdict=v + (' (descriptive)' if wide == 32 else ''))
    # Descriptive.
    desc = {}
    for t in TASKS:
        for g in G3:
            ss = [summarize(r) for r in (by.get((t, 4, g, 16, s, 'identity')) for s in G3) if ok(r)]
            desc[f'init_and_data_order_spread_r16_{t}_g{g}'] = dict(
                t_thresh=med_range([s['t_thresh'] for s in ss]), test_kl=med_range([s['test_kl'] for s in ss]),
                test_err=med_range([s['test_err'] for s in ss]))
        for m in (4, 8):
            ss = [summarize(r) for r in (by.get((t, m, g, 16, 0, 'identity')) for g in G3) if ok(r)]
            desc[f'm{m}_r16_{t}'] = dict(t_thresh=med_range([s['t_thresh'] for s in ss]),
                                         test_kl=med_range([s['test_kl'] for s in ss]),
                                         bayes_loss=med_range([s['bayes_loss'] for s in ss]),
                                         bayes_acc=med_range([s['bayes_acc'] for s in ss]))
    res['descriptive'] = desc
    res['variance_init'] = {f"{r['cfg']['task']}_g{r['cfg']['grammar']['seed']}": dict(
        status=r['status'], steps=r['steps_done'], test_acc=r['test']['acc'], final_diag_loss=r['log']['diag_loss'][-1],
        stays_at_chance=r['test']['acc'] < 2 / 16) for r in runs.values() if r['cfg'].get('init') == 'variance'}
    res['lr'] = lr_info(out)
    res['lr']['other_widths_prefer_other_lr'] = differ
    return res


def table(runs):
    """Per (task, m, r, init): median and range over grammars and seeds of final test KL, error, t_thresh, ms/update."""
    groups = defaultdict(list)
    for r in runs.values():
        s = summarize(r)
        groups[(s['task'], s['m'], s['r'], s['init'])].append(s)
    out = {}
    for (t, m, r, i), ss in sorted(groups.items()):
        fin = [s for s in ss if s['status'] in FINISHED]
        out[f'{t}_m{m}_r{r}' + ('_varinit' if i == 'variance' else '')] = dict(
            runs=len(ss), finished=len(fin), test_kl=med_range([s['test_kl'] for s in fin]),
            test_err=med_range([s['test_err'] for s in fin]), t_thresh=med_range([s['t_thresh'] for s in fin]),
            cumlr_thresh=med_range([s['cumlr_thresh'] for s in fin]),
            ms_per_step=med_range([s['ms_per_step'] for s in ss]))
    return out


def interpret(runs_dir, runs, hours, procs):
    """Run the interpretation CLI, identity-init r = 16 runs first, `procs` at a time with
    OMP_NUM_THREADS = (OMP_NUM_THREADS or 8) // procs; no run starts after `hours`. Returns a status per run."""
    if importlib.util.find_spec('studies.rhm.interpret') is None:
        print('note: studies.rhm.interpret is not available; skipping the interpretation step', flush=True)
        return {'skipped': 'studies.rhm.interpret not importable'}
    first = lambda n: runs[n]['cfg']['r'] == 16 and runs[n]['cfg'].get('init') == 'identity'
    order = sorted((n for n in runs if ok(runs[n])), key=lambda n: (not first(n), runs[n]['cfg']['r'], n))
    deadline = time.time() + 3600 * hours
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, int(os.environ.get('OMP_NUM_THREADS', 8)) // procs)))

    def one(n):
        d = Path(runs_dir) / n
        if (d / 'interpret.json').exists():
            return n, 'exists'
        if time.time() > deadline:
            return n, 'skipped (time)'
        diag = Path(runs_dir) / runs[n]['diag_file']  # the run's own diagnostic trees (data/diag_<m>_<g>.pt)
        cmd = [sys.executable, '-m', 'studies.rhm.interpret', str(d), '--out', str(d / 'interpret.json'),
               '--diag', str(diag), '--device', device]
        print(' '.join(cmd), flush=True)
        t = time.time()
        with open(d / 'interpret.log', 'w') as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        st = 'ok' if rc == 0 else f'failed (exit {rc}, see interpret.log)'
        print(f'{n}: interpret {st} in {time.time() - t:.0f}s', flush=True)
        return n, st

    with ThreadPoolExecutor(procs) as ex:  # map submits in order, so the r = 16 runs start first
        return dict(ex.map(one, order))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs')
    p.add_argument('results')
    p.add_argument('--interpret-only', action='store_true', help='run only the interpretation step')
    p.add_argument('--interpret-hours', type=float, default=3.0, help='no interpretation run starts after this')
    p.add_argument('--interpret-procs', type=int, default=4, help='interpretation processes at a time')
    a = p.parse_args()
    runs = load_runs(a.runs)
    path = Path(a.results) / 'analysis.json'
    if a.interpret_only:
        res = json.load(open(path)) if path.exists() else {}
        res['interpret'] = interpret(a.runs, runs, a.interpret_hours, a.interpret_procs)
        dump(path, res)
        return
    res = dict(n_runs=len(runs), criteria=criteria(runs, a.runs), table=table(runs),
               runs={n: summarize(r) for n, r in runs.items()})
    dump(Path(a.results) / 'curves.json', curves(runs))
    dump(path, res)
    for k, v in res['criteria'].items():
        if isinstance(v, dict) and 'verdict' in v:
            print(f"{k}: {v['verdict']}")


if __name__ == '__main__':
    main()
