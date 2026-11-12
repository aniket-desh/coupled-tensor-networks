"""Score the hierarchy sweeps against P1-P6 in hierarchy_experiments.py; numbers only, written to results/h_score.json."""
import json

import numpy as np
from scipy.integrate import quad

import hierarchy as h

J = lambda n: json.load(open(f'results/h_{n}.json'))
REDO = J('rerun')
key = lambda r: tuple(r.get(k) for k in ('arm', 'x', 'y', 'delta', 'seed'))
over = lambda rows, new: [{key(n): n for n in new}.get(key(r), r) for r in rows]
ORDER = over(J('order')['results'], REDO['order'])
RACE = over(J('race')['results'], REDO['race'])
OV = J('overshoot')
out = {'unfinished_after_rerun': sum(not r['done'] for r in ORDER + RACE)}

# P1, P2: race between groups
out['race'] = {}
for d in sorted({r['delta'] for r in RACE}, reverse=True):
    R = [r for r in RACE if r['delta'] == d]
    ratio = np.array([r['gap'] / r['gap_pred'] for r in R])
    gp = np.array([r['gap_pred'] for r in R])
    out['race'][d] = dict(n=len(R), tau_agree=np.mean([r['pred'] == r['obs'] for r in R]),
                          strength_agree=np.mean([r['strength_first'] == r['obs'] for r in R]),
                          # cases where the two rules disagree are the informative ones
                          disagree_n=sum(r['pred'] != r['strength_first'] for r in R),
                          disagree_tau_right=np.mean([r['pred'] == r['obs'] for r in R if r['pred'] != r['strength_first']]),
                          gap_ratio_q=np.percentile(ratio, [5, 25, 50, 75, 95]).tolist(),
                          gap_ratio_q_large_gap=np.percentile(ratio[gp > np.median(gp)], [5, 25, 50, 75, 95]).tolist(),
                          errors=[dict(s=r['s'], gap=r['gap'], gap_pred=r['gap_pred']) for r in R if r['pred'] != r['obs']][:5])

# P2, P3 on the main teacher: delays against delta
S = J('scaling')
rows = S['rows']
de = np.array([r['delta'] for r in rows])
H = np.array([r['half'] for r in rows])
sib = H[:, 1] - H[:, 0]
cous = H[:, 2] - H[:, 0]
m = de <= .01
out['scaling'] = dict(sib_slope=np.polyfit(np.log(1 / de[m]), sib[m], 1)[0], sib_pred=(1 - S['x']) / S['x'], cous_exponent=np.polyfit(np.log(de[m]), np.log(cous[m]), 1)[0],
                      cous_times_d2=(cous * de ** 2).tolist(), cous_pred=S['tau_gap'] / S['a'], overshoot_at_smallest=rows[-1]['O'])
sib2 = H[:, 3] - H[:, 2]
x2 = .25 / .8
out['scaling']['sib2_slope'] = np.polyfit(np.log(1 / de[m]), sib2[m], 1)[0]
out['scaling']['sib2_pred'] = (1 - x2) / x2

# P3 over the order map: slope of the sibling delay in log(1/delta), per x
out['sib_map'] = {}
for x in sorted({r['x'] for r in ORDER}):
    sl = []
    for y in sorted({r['y'] for r in ORDER}):
        q = sorted([r for r in ORDER if r['arm'] == 'hierarchy' and r['x'] == x and r['y'] == y and r['delta'] <= .01], key=lambda r: r['delta'])
        sl.append(np.polyfit([np.log(1 / r['delta']) for r in q], [r['half'][1] - r['half'][0] for r in q], 1)[0])
    out['sib_map'][x] = dict(pred=(1 - x) / x, median=np.median(sl), lo=min(sl), hi=max(sl))

# P4: does the weak sibling (s = x) finish before the stronger cousin (s = y), restricted to x < y?
out['p4'] = {}
for arm, d in [('hierarchy', .1), ('hierarchy', .03), ('hierarchy', .01), ('hierarchy', .003), ('hierarchy', .001), ('u_only', .01), ('group_only', .01), ('independent', .01)]:
    R = [r for r in ORDER if r['arm'] == arm and r['delta'] == d and r['x'] < r['y']]
    out['p4'][f'{arm}_{d}'] = dict(n=len(R), sib_before_stronger_cousin=np.mean([r['half'][1] < r['half'][2] for r in R]),
                                   strength_order=np.mean([np.argsort(r['half']).tolist() == np.argsort(-np.array([1, r['x'], r['y'], r['y'] / 3])).tolist() for r in R]))
    q = [r for r in R if r['y'] >= .95]
    out['p4'][f'{arm}_{d}']['near_degenerate_y'] = dict(n=len(q), frac=np.mean([r['half'][1] < r['half'][2] for r in q]), y=[r['y'] for r in q if r['half'][1] >= r['half'][2]][:12]) if q else None

# P5: escape rates, read from the delta = 1e-3 curves
Z = np.load('results/h_scaling.npz')
t, c, rate = Z['t'], Z['c'], Z['rate']
s = np.array(S['s'])
def window(k, lo, hi, after):
    w = (c[k] / s[k] > lo) & (c[k] / s[k] < hi) & (t > after)
    return w
t0 = Z['half'][0]
w = (c[1] > 1e-7) & (c[1] < 1e-3) & (c[0] > .999 * s[0])
out['p5'] = dict(sib_rate_pred=S['theory']['sib_rate'], sib_rate_q=np.percentile(rate[1][w], [5, 50, 95]).tolist(), sib_n=int(w.sum()))
w = (c[2] > 1e-7) & (c[2] < 1e-3) & (t > Z['half'][1])
out['p5'].update(cous_coef_pred=S['theory']['cous_coef'], cous_coef_q=np.percentile((rate[2] / np.sqrt(c[2]))[w], [5, 50, 95]).tolist(), cous_n=int(w.sum()),
                 cous_c_range=[float(c[2][w].min()), float(c[2][w].max())] if w.any() else None)

# P6: overshoot pattern
R = OV['results']
cnt = np.bincount([r['count'] for r in R], minlength=5)
gap = np.array([r['gap'] for r in R])
match = np.array([r['match'] for r in R])
out['p6'] = dict(n=len(R), count_hist=cnt.tolist(), last_overshoots=int(sum(r['last_overshoots'] for r in R)),
                 match=match.mean(), match_by_gap={f'{lo:g}-{hi:g}': (int(((gap >= lo) & (gap < hi)).sum()), float(match[(gap >= lo) & (gap < hi)].mean()) if ((gap >= lo) & (gap < hi)).any() else None) for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1), (1, 1e9)]},
                 who_not=[r['count'] for r in R if not r['last_overshoots']][:3],
                 # which position in learning order overshoots, and how big, aggregated
                 O_by_position=[np.percentile([100 * r['O'][r['order'][i]] for r in R], [25, 50, 75]).tolist() for i in range(4)],
                 O_pos_frac=[float(np.mean([r['O'][r['order'][i]] > OV['floor'] for r in R])) for i in range(4)],
                 not_three=[dict(s=r['s'], O=r['O'], order=r['order']) for r in R if r['count'] != 3][:5],
                 stages=OV['stages'], arms={k: v for k, v in OV['arms'].items()})
out['p6']['arms']['independent_main'] = REDO['independent_main']

# Loss
L = J('loss')['results']
L['independent_main'] = REDO['independent_main']
out['loss'] = L

# Dense random init: order and overshoot against independent networks
D = J('dense')['results']
out['dense'] = {}
for s_ in sorted({tuple(r['s']) for r in D}):
    for sg in sorted({r['sigma'] for r in D}, reverse=True):
        for sh in (True, False):
            q = [r for r in D if tuple(r['s']) == s_ and r['sigma'] == sg and r['shared'] == sh]
            Hd = np.array([r['half'] for r in q])
            Od = 100 * np.array([r['O'] for r in q])
            out['dense'][f'{s_}|{sg}|{"shared" if sh else "control"}'] = dict(n=len(q), done=int(sum(r['done'] for r in q)), nan=int(np.isnan(Hd).any(1).sum()),
                sib_before_cousin=float(np.mean(Hd[:, 1] < Hd[:, 2])), O_median=np.median(Od, 0).round(4).tolist(), O_q75=np.percentile(Od, 75, axis=0).round(4).tolist(),
                groups_first=float(np.mean(np.argsort(Hd, 1)[:, :2].max(1) <= 1)),
                first_two_siblings=float(np.mean(h.G[np.argsort(Hd, 1)[:, 0]] == h.G[np.argsort(Hd, 1)[:, 1]])), median_half=np.median(Hd, 0).round(1).tolist())

# P3 refined after P3 missed the second group: after the top of group G saturates, a q^2 = sqrt(s_top a), so
# slope = (1 - x) / (s_sib sqrt(s_top a)), with a = (sqrt(C_other) + sqrt(s_top))^(2/5), C_other = 0 if G learns first.
def sib_slopes(r_by_d):
    r = r_by_d[.01]
    s_ = np.array(r['s'])
    win = int(np.argmin(r['taus']))
    for g in (0, 1):
        idx = np.where(h.G == g)[0]
        top, sib = idx[np.argmax(s_[idx])], idx[np.argmin(s_[idx])]
        C_other = 0.0 if g == win else s_[h.G == win].sum()
        a = (np.sqrt(C_other) + np.sqrt(s_[top])) ** (2 / 5)
        x = s_[sib] / s_[top]
        meas = ((r_by_d[.001]['half'][sib] - r_by_d[.001]['half'][top]) - (r['half'][sib] - r['half'][top])) / np.log(10)
        yield dict(first=g == win, x=x, meas=meas, refined=(1 - x) / (s_[sib] * np.sqrt(s_[top] * a)), original=(1 - x) / x)


seeds = {}
for r in RACE:
    seeds.setdefault(r['seed'], {})[r['delta']] = r
SL = [q for v in seeds.values() for q in sib_slopes(v)]
rel = lambda q, k: abs(q[k] / q['meas'] - 1)
out['p3_refined'] = {}
for label, f in [('first group', lambda q: q['first']), ('second group', lambda q: not q['first'])]:
    for xlab, g in [('x<=.8', lambda q: q['x'] <= .8), ('x>.8', lambda q: q['x'] > .8)]:
        Q = [q for q in SL if f(q) and g(q)]
        out['p3_refined'][f'{label} {xlab}'] = dict(n=len(Q), refined_rel_err_q=np.percentile([rel(q, 'refined') for q in Q], [50, 95]).tolist(),
                                                    original_rel_err_q=np.percentile([rel(q, 'original') for q in Q], [50, 95]).tolist())

# P4: is every exception to "sibling before stronger cousin" explained by group B winning the race?
out['p4_explained'] = {}
tau_ = lambda top, sib: quad(lambda u: 1 / (u ** 2 + u ** (1 + sib / top)), 1, np.inf)[0] / (2 * top)
for d in (.01, .003, .001):
    R = [r for r in ORDER if r['arm'] == 'hierarchy' and r['delta'] == d and r['x'] < r['y']]
    A_first = [tau_(1, r['x']) < tau_(r['y'], r['y'] / 3) for r in R]
    sib_first = [r['half'][1] < r['half'][2] for r in R]
    out['p4_explained'][d] = dict(n=len(R), A_first=int(sum(A_first)), agree=int(sum(a == b for a, b in zip(A_first, sib_first))))

# P6: signs nearer convergence for the runs that missed; floor exceptions; relaxation against learning rates
SG = J('signs')['results']
sg_gap = np.array([r['gap'] for r in SG])
n_all = len(OV['results'])
out['p6_signs'] = dict(misses=len(SG), now_match=int(sum(r['match'] for r in SG)), done=int(sum(r['done'] for r in SG)),
                       still_missing_gaps=np.percentile(sg_gap[[not r['match'] for r in SG]], [0, 50, 100]).tolist() if not all(r['match'] for r in SG) else None,
                       total_match_frac=(sum(r['match'] for r in OV['results']) + sum(r['match'] for r in SG)) / n_all)
ex = [r for r in OV['results'] if r['count'] != 3]
out['p6_exceptions'] = [dict(s=np.round(r['s'], 4).tolist(), O=r['O'], order=r['order']) for r in ex]
out['p6_min_positive'] = [float(np.min([r['O'][r['order'][i]] for r in OV['results']])) for i in range(3)]
for name, s_ in [('main', [1.0, .3, .8, .25]), ('alt', [1.0, .6, .9, .3])]:
    out[f'K_diag_{name}'] = np.diag(h.K_matrix(np.array(s_))).tolist()

json.dump(out, open('results/h_score.json', 'w'), indent=1, default=float)
