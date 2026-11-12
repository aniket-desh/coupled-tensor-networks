"""Shared analysis helpers for the RHM experiments: loading runs, run health, steps to threshold, and paired
comparisons with explicit tie rules.

Rules every analyze.py follows (stated in each EXPERIMENT.md):
- Only runs with status finished or early_stopped enter a comparison. A pair where either run is truncated,
  diverged, skipped, failed or missing, or has a non-finite metric, is dropped (counted, never a tie).
- Run health is counted against <out>/planned.json (the full planned list), not against the result files present.
- Steps to threshold: first logged update where diagnostic accuracy >= 0.9 x Bayes accuracy; +inf if never (the run
  finished without reaching it). A is better than B only if t_A < t_B - margin, with margin = max(the logging interval
  at the later of the two, the reference arm's init/data-seed range of t on that (task, grammar)); otherwise a tie.
  Finite beats inf; inf vs inf is a tie.
- Final quality: test error rate first; a difference within max(1e-3, the reference arm's seed range) is a tie, and
  then test loss decides with the same kind of tolerance, max(1e-3 nats, seed range). Float32 log_softmax saturates
  (loss exactly 0 beyond a ~17-nat margin), so losses below 1e-3 nats are not distinguished.
- Ties are excluded from sign tests and reported. A verdict also needs consistency across grammars: wins > losses in
  at least 2 of the 3 grammars and losses > wins in none, so one grammar can't carry it.
- A comparison is inconclusive if its reference (control) arm didn't learn: median test accuracy < 2 x chance, or
  median final diagnostic loss within 0.05 nats of log(classes)."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

THRESH = 0.9
FINISHED = ('finished', 'early_stopped')
FLOOR = 1e-3


def load_runs(out, subdir=''):
    """{name: result dict} for every <out>/<subdir>/<name>/result.json (pilot runs live in <out>/pilot)."""
    base = Path(out) / subdir
    return {p.parent.name: json.load(open(p)) for p in sorted(base.glob('*/result.json'))}


def health(out, runs):
    """Status of every planned run (missing = never wrote a result)."""
    p = Path(out) / 'planned.json'
    planned = json.load(open(p)) if p.exists() else sorted(runs)
    st = {n: (runs[n]['status'] if n in runs else 'missing') for n in planned}
    bad = {n: s for n, s in st.items() if s not in FINISHED}
    return dict(planned=len(planned), finished=len(planned) - len(bad), not_finished=bad,
                frac_not_finished=len(bad) / max(len(planned), 1))


def ok(r):
    return r is not None and r.get('status') in FINISHED


def steps_to(log, key, target, above=True):
    """First logged step where log[key] >= target (or <= when above=False); inf if never."""
    for s, v in zip(log['step'], log[key]):
        if v is not None and math.isfinite(v) and (v >= target if above else v <= target):
            return float(s)
    return math.inf


def t_thresh(res, frac=THRESH, prefix='diag'):
    """Steps until diagnostic accuracy >= frac * Bayes diagnostic accuracy."""
    return steps_to(res['log'], f'{prefix}_acc', frac * res['bayes_diag']['acc'])


def log_gap(steps, t):
    """The logging interval at step t: distance from t to the next logged step (0 for inf or the last step)."""
    if not math.isfinite(t):
        return 0.0
    later = [s for s in steps if s > t]
    return float(later[0] - t) if later else 0.0


def cum_lr(res, T):
    """Sum of the learning rate over the updates before T (the 'time' a threshold took, insensitive to the warmup)."""
    if not math.isfinite(T):
        return math.inf
    from studies.rhm.train import lr_at
    return float(sum(lr_at(t, res['lr'], res['schedule_total'], res['warmup']) for t in range(int(T))))


def seed_range(vals):
    """max - min over finite values (0 with fewer than two)."""
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(max(v) - min(v)) if len(v) >= 2 else 0.0


def cmp_steps(ta, tb, margin):
    """'a' if a reaches threshold earlier by more than margin, 'b' the reverse, else 'tie'."""
    if math.isinf(ta) and math.isinf(tb):
        return 'tie'
    if math.isinf(ta) != math.isinf(tb):
        return 'a' if math.isinf(tb) else 'b'
    return 'a' if ta < tb - margin else 'b' if tb < ta - margin else 'tie'


def cmp_quality(ra, rb, tol_err, tol_loss):
    """Final quality, smaller is better: error rate, then loss when error rates tie. None if a metric is missing."""
    ea, eb = 1 - ra['test']['acc'], 1 - rb['test']['acc']
    la, lb = ra['test']['loss'], rb['test']['loss']
    if not all(map(math.isfinite, (ea, eb, la, lb))):
        return None
    if abs(ea - eb) > tol_err:
        return 'a' if ea < eb else 'b'
    if abs(la - lb) > tol_loss:
        return 'a' if la < lb else 'b'
    return 'tie'


def tally(outcomes):
    """outcomes: list of (grammar, 'a' | 'b' | 'tie' | None). Sign test of a against b over non-tied pairs, plus
    per-grammar counts and the consistency rule."""
    kept = [(g, o) for g, o in outcomes if o is not None]
    w, l = sum(o == 'a' for _, o in kept), sum(o == 'b' for _, o in kept)
    per = defaultdict(lambda: dict(a=0, b=0, tie=0))
    for g, o in kept:
        per[g][o] += 1
    n = w + l
    cons = sum(v['a'] > v['b'] for v in per.values()) >= 2 and not any(v['b'] > v['a'] for v in per.values())
    cons_b = sum(v['b'] > v['a'] for v in per.values()) >= 2 and not any(v['a'] > v['b'] for v in per.values())
    return dict(a_better=w, b_better=l, ties=len(kept) - n, pairs_used=len(kept), pairs_dropped=len(outcomes) - len(kept),
                p_a_better=binomtest(w, n, 0.5, alternative='greater').pvalue if n else 1.0,
                p_b_better=binomtest(l, n, 0.5, alternative='greater').pvalue if n else 1.0,
                per_grammar={f'g{g}': v for g, v in sorted(per.items())}, consistent_a=cons, consistent_b=cons_b)


def learned(results, classes):
    """The 'did it learn' rule for a control arm: median test accuracy >= 2 x chance and median final diagnostic
    loss below log(classes) - 0.05. results: list of result dicts (finished runs only)."""
    rs = [r for r in results if ok(r)]
    if not rs:
        return dict(learned=False, n=0, reason='no finished runs')
    acc = float(np.median([r['test']['acc'] for r in rs]))
    loss = float(np.median([r['log']['diag_loss'][-1] for r in rs]))
    good = acc >= 2.0 / classes and loss < math.log(classes) - 0.05
    return dict(learned=bool(good), n=len(rs), median_test_acc=acc, median_final_diag_loss=loss,
                chance=1.0 / classes, log_classes=math.log(classes))


def med_range(x):
    """Median and range with inf kept (runs that never reached a threshold count as +inf); n included."""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if not len(x):
        return dict(n=0)
    return dict(n=len(x), median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                never=int(np.isinf(x).sum()))


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else
              (None if isinstance(o, float) and not math.isfinite(o) else float(o)))


def lr_info(out):
    """lr.json with the grid-edge flags (for analysis.json)."""
    p = Path(out) / 'lr.json'
    lr = json.load(open(p)) if p.exists() else {}
    return dict(choices=lr, at_grid_edge=sorted(k for k, v in lr.items() if v.get('at_grid_edge')))


def curves(runs, keys=('train_loss', 'diag_loss', 'diag_acc', 'diag_kl')):
    """Compact per-run curves and metadata for plot.py (saved next to the analysis, so figures need no run dirs)."""
    out = {}
    for name, r in runs.items():
        out[name] = dict(cfg=r['cfg'], status=r['status'], lr=r['lr'], bayes_diag=r['bayes_diag'],
                         bayes_test=r['bayes_test'], test=r['test'], step=r['log']['step'],
                         **{k: r['log'][k] for k in keys if k in r['log']},
                         **{k: r[k] for k in ('best', 'best_acc') if k in r})
    return out
