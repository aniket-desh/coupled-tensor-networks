"""Shared statistics for the TTN analyses. Censored half-times (nan: never reached before the run stopped) are slower
than every learned run, so they enter medians and comparisons as +inf, never dropped."""
import json
from math import comb
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


def load(folder, name):
    z = np.load(Path(folder) / f'{name}.npz')
    return {k: z[k] for k in z.files}, json.load(open(Path(folder) / f'{name}.json'))


def inf_nan(h):
    return np.where(np.isnan(np.asarray(h, float)), np.inf, np.asarray(h, float))


def med_ci(h, n_boot=4000):
    """Median of censored half-times (nan -> inf) and a bootstrap 95% interval over runs."""
    h = inf_nan(h)
    boots = np.median(RNG.choice(h, (n_boot, len(h))), axis=1)
    return float(np.median(h)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_sign(a, b):
    """Seed-paired comparison of half-times a vs b (nan -> inf). Counts seeds with a < b and a > b; ties (including
    both censored) are excluded. Returns (wins, losses, two-sided binomial sign-test p)."""
    a, b = inf_nan(a), inf_nan(b)
    w, l = int((a < b).sum()), int((a > b).sum())
    n = w + l
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(w, l) + 1)) / 2 ** n) if n else 1.0
    return w, l, p


def ratio_ci(a, b, n_boot=4000):
    """Median of seed-paired ratios a/b (inf where a is censored, 0 where only b is) and a bootstrap 95% interval."""
    a, b = inf_nan(a), inf_nan(b)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(np.isinf(a) & np.isinf(b), np.nan, a / b)
    r = r[~np.isnan(r)]
    if not len(r):
        return float('nan'), [float('nan'), float('nan')]
    boots = np.median(RNG.choice(r, (n_boot, len(r))), axis=1)
    return float(np.median(r)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def slope_ci(x, y, n_boot=4000):
    """Least-squares slope of y on x and a bootstrap 95% interval over points (finite pairs only)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), [float('nan'), float('nan')], int(len(x))
    fit = lambda i: np.polyfit(x[i], y[i], 1)[0]
    boots = [fit(RNG.integers(0, len(x), len(x))) for _ in range(n_boot)]
    return float(np.polyfit(x, y, 1)[0]), [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))], int(len(x))


def drive(log, meta, run, target_k, t_from=0.0, rise=0.02):
    """Decompose c_k's gradient-flow velocity v_k = dc_k/dt for one run, integrated over c_k's rise: from the first time
    after t_from that c_k/s_k reaches `rise` (K is recorded densely from then on) up to c_k's half-time:
    v_k = sum_j K_kj e_j  (pull of each teacher component's residual e_j = s_j - c_j)
          + rest          (pull of the student's coefficients on all non-teacher basis functions, = v_k - sum_j K_kj e_j).
    Uses the dense K and v records (t_K), with c interpolated onto them. Returns a dict: j -> integral of K_kj e_j, 'rest',
    'total' (integral of v_k) and 'actual' (c_k at the end minus at the start). total ~ actual checks that the records
    resolve the rise."""
    s = np.array([c['s'] for c in meta['comps']])
    t, c = log['t'], log['c'][:, run]
    tK, K, v = log['t_K'], log['K'][:, run].sum(-1), log['v'][:, run, target_k].sum(-1)  # sums over subtree groups
    half = meta['summary']['half'][run][target_k]
    stop = (t_from + half) if half is not None and np.isfinite(half) else t[-1]
    up = np.nonzero((t >= t_from) & (c[:, target_k] / s[target_k] >= rise))[0]
    start = t[up[0]] if len(up) else np.inf
    sel = (tK >= start) & (tK <= stop)
    if sel.sum() < 2:
        return {j: float('nan') for j in [*range(len(s)), 'rest', 'total', 'actual']}
    e = np.stack([s[j] - np.interp(tK[sel], t, c[:, j]) for j in range(len(s))], -1)
    terms = K[sel][:, target_k, :] * e  # (T, K)
    out = {j: float(np.trapezoid(terms[:, j], tK[sel])) for j in range(len(s))}
    out['rest'] = float(np.trapezoid(v[sel] - terms.sum(1), tK[sel]))
    out['total'] = float(np.trapezoid(v[sel], tK[sel]))
    out['actual'] = float(np.interp(tK[sel][-1], t, c[:, target_k]) - np.interp(tK[sel][0], t, c[:, target_k]))
    return out


def fmt(m, ci):
    return f'{m:.4g} [{ci[0]:.4g}, {ci[1]:.4g}]'
