"""The breakdowns quoted in README.md and NOTE.md that summary.json does not hold, from the saved results only.
Written after the run for the write-up; it changes no criterion (those are computed by analyze.py's `verdicts`).

    PYTHONPATH=. python studies/rhm/rhm-forced-synonyms/tables.py studies/rhm/rhm-forced-synonyms/results | tee studies/rhm/rhm-forced-synonyms/results/tables.txt

Names as in analyze.py's docstring. "exact" = polished excess loss below 1e-10 nats; "found by Adam" = Adam's own
endpoint below 1e-6 nats; "capped" = the L-BFGS polish stopped on its iteration or evaluation limit (censored)."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from studies.rhm import toy as Y

res = Path(sys.argv[1])
R = json.load(open(res / 'runs.json'))
T = json.load(open(res / 'trajectories.json'))
INITS = ('variance', 'identity')
q5 = lambda v: ' '.join(f'{a:.3g}' for a in np.percentile(v, [0, 25, 50, 75, 100]))   # min, quartiles, max
exact = lambda q: q['kl'] < Y.EXACT
swap_all = lambda q: max(q['swap_L_all'], q['swap_R_all'])
swap_legal = lambda q: max(q['swap_L_legal'], q['swap_R_legal'])

print('1. Runs per width and init: n, exact, found by Adam (among exact), capped, final excess loss [min q1 median q3 max]')
for r in (1, 2, 3, 4):
    for i in INITS:
        c = [q for q in R if q['r'] == r and q['init'] == i]
        print(f"   r={r} {i:8s} n={len(c)} exact={sum(map(exact, c))} adam={sum(exact(q) and q['kl_adam'] < Y.ADAM_EXACT for q in c)} "
              f"capped={sum(q['polish_capped'] for q in c)} (capped and exact: {sum(q['polish_capped'] and exact(q) for q in c)})  excess {q5([q['kl'] for q in c])}")
print('   width-1 loss levels (rounded to 1e-5 nats), as a share of the uniform predictor\'s excess:')
for i in INITS:
    lv = Counter(round(q['kl'], 5) for q in R if q['r'] == 1 and q['init'] == i)
    print(f"   r=1 {i:8s} " + ', '.join(f'{k:.5f} ({k / Y.UNIFORM_KL:.2f} of uniform) x{n}' for k, n in sorted(lv.items())))

print('\n2. Width-2 runs that are not exact, or that the polish (not Adam) made exact')
for q in R:
    if q['r'] == 2 and (not exact(q) or q['kl_adam'] >= Y.ADAM_EXACT):
        print(f"   {q['run']}: Adam {q['kl_adam']:.3e}, polished {q['kl']:.3e}, capped {q['polish_capped']}, (x, y) = ({q['xy_L'][0]:.3g}, {q['xy_L'][1]:.3g}), "
              f"forb_rho = {', '.join(f'{v:.3g}' for v in q['forb_rho'])}, xy_shift {q['xy_shift']:.3g}")
        t = next(t for t in T if t['run'] == q['run'] and t['step'] == Y.STEPS)
        print(f"      at Adam's last update: excess {t['kl']:.3e}, forb_rho = {', '.join(f'{v:.3g}' for v in t['forb_rho'])}")

ex2 = [q for q in R if q['r'] == 2 and exact(q)]
print(f'\n3. Exact width-2 runs (n = {len(ex2)}): the H2 measures [min q1 median q3 max]')
for k in ('swap_L_all', 'swap_R_all', 'within_L', 'within_R', 'within_P', 'comp_err', 'max_err'):
    print(f'   {k:11s} {q5([q[k] for q in ex2])}')
print('   three largest max(swap_L_all, swap_R_all):', ', '.join(f"{swap_all(q):.3g} ({q['init']} s{q['seed']}, capped {q['polish_capped']})" for q in sorted(ex2, key=swap_all)[-3:]))
print(f"   Adam's own endpoint over these runs: {q5([q['kl_adam'] for q in ex2])}")

print('\n4. Forbidden responses of exact width-2 runs: per run, the sorted triple (smallest, middle, largest)')
for label, sel in (('found by Adam, variance', lambda q: q['kl_adam'] < Y.ADAM_EXACT and q['init'] == 'variance'),
                   ('found by Adam, identity', lambda q: q['kl_adam'] < Y.ADAM_EXACT and q['init'] == 'identity'),
                   ('found by Adam, both', lambda q: q['kl_adam'] < Y.ADAM_EXACT), ('all exact', lambda q: True)):
    c = [q for q in ex2 if sel(q)]
    s = np.sort(np.array([q['forb_rho'] for q in c]), 1)
    xy = np.array([q['xy_L'] for q in c])
    print(f'   {label} (n = {len(c)} runs)')
    print(f'      smallest {q5(s[:, 0])};  runs with smallest < 0.5: {(s[:, 0] < 0.5).sum()};  < sqrt(3)/2: {(s[:, 0] < 3 ** 0.5 / 2).sum()}')
    print(f'      middle   {q5(s[:, 1])};  largest {q5(s[:, 2])}')
    print(f'      x {q5(xy[:, 0])};  y {q5(xy[:, 1])};  smallest max(|x + 1|, |y + 1|) = {np.abs(xy + 1).max(1).min():.3g}')
    print(f"      forb_cos (pooled over the 3 pairs) {q5(np.array([q['forb_cos'] for q in c]).ravel())}")

print('\n5. Widths 3 and 4 (every run is exact): the reading fixed in EXPERIMENT.md')
for r in (3, 4):
    for i in INITS:
        c = [q for q in R if q['r'] == r and q['init'] == i and exact(q)]
        keep = [q for q in c if q['keeps_synonym']]
        rho, cos = np.array([q['forb_rho'] for q in c]), np.array([q['forb_cos'] for q in c])
        side = Counter(('L' if q['swap_L_all'] > 1e-3 else '') + ('R' if q['swap_R_all'] > 1e-3 else '') for q in keep)
        holds = [q for q in c if max(q['within_L'], q['within_R']) > 1e-3]
        print(f'   r={r} {i:8s} exact n={len(c)}; keeps the synonym: {len(keep)} (left only {side["L"]}, right only {side["R"]}, both {side["LR"]})')
        if keep:
            print(f'      swap beside any sibling, keepers: {q5([swap_all(q) for q in keep])};  smallest ratio to the legal swap: {min(swap_all(q) / swap_legal(q) for q in keep):.3g}')
        if len(keep) < len(c):
            print(f'      swap beside any sibling, collapsed runs: max {max(swap_all(q) for q in c if not q["keeps_synonym"]):.3g}')
        print(f'      swap beside siblings that occur, all runs: max {max(map(swap_legal, c)):.3g}')
        print(f'      runs with within_L or within_R > 1e-3 (some direction holds the synonym): {len(holds)}, of which collapsed by the reading: {sum(not q["keeps_synonym"] for q in holds)}')
        print(f'      within_L values {dict(sorted(Counter(round(q["within_L"], 2) for q in c).items()))}, within_R {dict(sorted(Counter(round(q["within_R"], 2) for q in c).items()))}')
        print(f'      forb_rho over {rho.size} pairs: {q5(rho.ravel())};  below 0.01: {(rho < 0.01).sum()};  below 0.5: {(rho < 0.5).sum()} (in {(rho.min(1) < 0.5).sum()} runs)')
        print(f'      forb_cos over {cos.size} pairs: {q5(cos.ravel())}')

print(f'\n6. Along training, the {len(ex2)} exact width-2 runs: per Adam update, median [min, max] over runs')
names = {q['run'] for q in ex2}
for step in sorted({t['step'] for t in T}):
    tt = [t for t in T if t['run'] in names and t['step'] == step]
    kl, sw = np.array([t['kl'] for t in tt]), np.array([max(t['swap_L_all'], t['swap_R_all']) for t in tt])
    ce, lo = np.array([t['comp_err'] for t in tt]), np.array([min(t['forb_rho']) for t in tt])
    ok = kl > 1e-14                       # the ratio to sqrt(excess) is only meaningful above the loss's rounding noise
    ratio = sw[ok] / np.sqrt(kl[ok])
    print(f'   update {step:5d}: excess {np.median(kl):.2e} [{kl.min():.1e}, {kl.max():.1e}]  swap_all {np.median(sw):.2e} [{sw.min():.1e}, {sw.max():.1e}]  '
          f'comp_err {np.median(ce):.2e}  runs with 6/6 compositions {sum(t["comp_acc"] == 1 for t in tt)}  '
          f'swap_all / sqrt(excess) over the {ok.sum()} runs above 1e-14 nats: {np.median(ratio):.2f} [{ratio.min():.2g}, {ratio.max():.2g}]  smallest forb_rho {np.median(lo):.3f}')
ok = [q for q in ex2 if q['kl'] > 0]
ratio = np.array([swap_all(q) / np.sqrt(q['kl']) for q in ok])
print(f'   polished: excess {q5([q["kl"] for q in ex2])};  swap_all / sqrt(excess): {q5(ratio)}')
