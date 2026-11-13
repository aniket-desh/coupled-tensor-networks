"""Shared analysis helpers for the language-process experiments (each experiment's analyze.py applies its own rules).

Definitions used by every experiment:
  t (steps to threshold): the first logged update at which diagnostic accuracy reaches FRAC x the Bayes diagnostic
      accuracy; +inf if never (censored at the run's length; reported, never dropped or imputed).
  gap(t): the logging interval around t (the distance to the next logged step), the time resolution of t.
  faster(a, b): t_a < t_b with t_b - t_a > max(gap(a), gap(b)) and t_a < t_b / REL; finite beats inf. A pair where
      both runs are censored (inf vs inf) is uninformative: compare() counts it as 'both_censored', not as a tie.
  t uses the 'unb' (unbiased) diagnostic set when the run logged one (parity with input bias), so that the threshold
      requires the parity itself rather than an easier function of the biased inputs.
  final quality: test KL(Bayes || model) in nats; a tie within max(KL_TOL, 10% of the larger KL).
  collapsed: the run learned (diagnostic loss < max(log(O)/2, log(O) - 0.5) at some log step) and later returned
      within 0.1 nats of log(O) (studies.lang.train.collapsed). Collapsed runs stay in every comparison and are counted.
"""
import json
import math
from pathlib import Path

import numpy as np

from studies.lang import processes as P
from studies.lang.train import collapsed as _collapsed

FRAC, REL, KL_TOL = 0.9, 1.25, 0.002
INF = float('inf')


def load_runs(out):
    runs = {}
    for f in Path(out).glob('*/result.json'):
        r = json.load(open(f))
        runs[r['name']] = r
    planned = json.load(open(Path(out) / 'planned.json')) if (Path(out) / 'planned.json').exists() else sorted(runs)
    return runs, planned


def t_thresh(r, frac=FRAC, key=None):
    key = key or ('unb' if 'unb_acc' in r['log'] else 'diag')
    acc, steps = r['log'][f'{key}_acc'], r['log']['step']
    target = frac * r[f'bayes_{key}']['acc']
    return next((s for s, a in zip(steps, acc) if a >= target), INF)


def gap(r, t):
    steps = r['log']['step']
    if not math.isfinite(t):
        return 0
    i = steps.index(t)
    return (steps[i + 1] - t) if i + 1 < len(steps) else (t - steps[i - 1] if i else 0)


def faster(ra, rb):
    """+1 if run a reaches threshold faster than run b, -1 if slower, 0 if tied (rules in the module docstring)."""
    ta, tb = t_thresh(ra), t_thresh(rb)
    if not math.isfinite(ta) and not math.isfinite(tb):
        return 0
    if math.isfinite(ta) != math.isfinite(tb):
        return 1 if math.isfinite(ta) else -1
    lo, hi, sgn = (ta, tb, 1) if ta < tb else (tb, ta, -1)
    m = max(gap(ra, ta), gap(rb, tb))
    return sgn if (hi - lo > m and lo < hi / REL) else 0


def better(ra, rb, tol=KL_TOL):
    """+1 if a has the lower final test KL beyond the tie tolerance, -1 if higher, 0 if tied."""
    ka, kb = ra['test']['kl'], rb['test']['kl']
    if abs(ka - kb) <= max(tol, 0.1 * max(ka, kb)):
        return 0
    return 1 if ka < kb else -1


def finished(r):
    return r['status'] in ('finished', 'early_stopped') and math.isfinite(r['test']['kl'])


def collapsed(r):
    return _collapsed(r['log']['diag_loss'], P.make(r['cfg']['process']).O)


def eta_t(r, t):
    """Cumulative LR to threshold, sum of the scheduled LR over updates 0..t-1 (gradient-flow time, the project's
    convention for comparing runs at different LRs); inf if t is."""
    if not math.isfinite(t):
        return INF
    from studies.rhm.train import lr_at
    return float(sum(lr_at(s, r['lr'], r['schedule_total']) for s in range(int(t))))


def faster_eta(ra, rb):
    """faster() on cumulative LR instead of updates (same margin rule, with the gap converted at the run's LR)."""
    ta, tb = eta_t(ra, t_thresh(ra)), eta_t(rb, t_thresh(rb))
    if not math.isfinite(ta) and not math.isfinite(tb):
        return 0
    if math.isfinite(ta) != math.isfinite(tb):
        return 1 if math.isfinite(ta) else -1
    lo, hi, sgn = (ta, tb, 1) if ta < tb else (tb, ta, -1)
    m = max(gap(ra, t_thresh(ra)) * ra['lr'], gap(rb, t_thresh(rb)) * rb['lr'])
    return sgn if (hi - lo > m and lo < hi / REL) else 0


def row(r):
    t = t_thresh(r)
    return dict(name=r['name'], status=r['status'], t=t, eta_t=eta_t(r, t), t_diag=t_thresh(r, key='diag'), kl=r['test']['kl'], acc=r['test']['acc'],
                bayes_acc=r['bayes_test']['acc'], loss=r['test']['loss'], bayes_loss=r['bayes_test']['loss'],
                collapsed=collapsed(r), lr=r['lr'], params=r['n_params'], ms=r['ms_per_step'], steps=r['steps_done'],
                strata=r.get('strata'))


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float('nan')


def rng(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return [float(min(xs)), float(max(xs))] if xs else [float('nan'), float('nan')]


def compare(pairs, fn):
    """pairs: [(run_a, run_b)]; fn: faster or better. Counts of wins, losses and ties for a, plus the pairs dropped
    because either run did not finish."""
    w = l = t = dropped = both_cens = 0
    detail = []
    for ra, rb in pairs:
        if ra is None or rb is None or not (finished(ra) and finished(rb)):
            dropped += 1
            continue
        if fn in (faster, faster_eta) and not math.isfinite(t_thresh(ra)) and not math.isfinite(t_thresh(rb)):
            both_cens += 1
            detail.append((ra['name'], rb['name'], 'both censored'))
            continue
        s = fn(ra, rb)
        w, l, t = w + (s > 0), l + (s < 0), t + (s == 0)
        detail.append((ra['name'], rb['name'], s))
    return dict(wins=w, losses=l, ties=t, dropped=dropped, both_censored=both_cens, n=w + l + t, detail=detail)


def curves(runs):
    return {n: dict(step=r['log']['step'], diag_loss=r['log']['diag_loss'], diag_acc=r['log']['diag_acc'],
                    diag_kl=r['log']['diag_kl'], bayes_diag=r['bayes_diag'], cfg=r['cfg'], lr=r['lr'])
            for n, r in runs.items()}


# ------------------------------------------------------------------------------------- interpretation

def load_interp(out, name):
    f = Path(out) / name / 'interpret.json'
    return json.load(open(f)) if f.exists() else None


def steps_of(ip):
    return sorted(ip['steps'], key=int)


def node_spans(cfg):
    return {f'{l}_{k}': S for (l, k), S in P.spans(cfg.get('perm')).items()}


def get(ip, step, node, name, *path):
    v = ip['steps'].get(str(step), {}).get(node, {}).get(name)
    for p in path:
        if not isinstance(v, dict) or p not in v:
            return None
        v = v[p]
    return v


def trajectories(out, runs):
    """Per interpreted run: {summary: {level: {step: median probe over the level's nodes}}}, plus per-node invariance
    and patching are left in the run's interpret.json."""
    tr = {}
    for name, r in runs.items():
        ip = load_interp(out, name)
        if ip is None:
            continue
        d = {}
        for st in steps_of(ip):
            for node, summ in ip['steps'][st].items():
                lvl = node.split('_')[0]
                for nm, v in summ.items():
                    d.setdefault(nm, {}).setdefault(lvl, {}).setdefault(st, []).append(v.get('probe'))
        tr[name] = {nm: {lvl: {st: med(v) for st, v in x.items()} for lvl, x in y.items()} for nm, y in d.items()}
    return tr


def frac_true(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else float('nan'), len(xs)


def apply_health_override(an):
    """If more than a quarter of the planned runs did not finish, every verdict becomes inconclusive (the original
    verdict is kept alongside)."""
    h = an['health']
    h['frac_unfinished'] = len(h['missing']) / max(h['planned'], 1)
    if h['frac_unfinished'] <= 0.25:
        return an
    for k, v in an.items():
        if isinstance(v, dict) and 'verdict' in v:
            v['verdict_before_override'] = v['verdict']
            v['verdict'] = 'inconclusive (more than a quarter of the planned runs did not finish)'
    return an


def meeting_level(cfg, positions):
    """Level of the lowest model node whose span holds all the given positions (4 = only the root core joins them)."""
    for l in range(4):
        for (ll, k), S in P.spans(cfg.get('perm')).items():
            if ll == l and all(t in S for t in positions):
                return l
    return 4


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return str(o)
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return clean(o.item())
        return o
    with open(path, 'w') as f:
        json.dump(clean(obj), f, indent=1)
