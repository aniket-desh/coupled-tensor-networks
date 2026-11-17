"""Which (grammar, grouping, width) cells reach the exact predictor, against the width lower bounds of toy.tree_bound.

    PYTHONPATH=. python studies/rhm/rhm-crossed-rank/analyze.py runs/rhm-crossed-rank studies/rhm/rhm-crossed-rank/results

A cell is one grammar, grouping and width with 20 runs (2 inits x 10 seeds) at the default rates toy.LR; runs at other
rates (the rate arm of run.py) enter only the trainability comparison. Reaching the exact predictor is an
existence statement about the cell, so the cell's number is its best run: `exact` if any run ends below toy.EXACT
(1e-10 nats of excess loss after the polish), and the cell's floor is the smallest excess loss of its runs."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from studies.rhm import toy as Y


N_CELL = 20
TRAIN_PAIRS = {'sym': (('sym', 'aligned', 2), ('sym', 'crossed', 2)), 'mod': (('mod', 'aligned', 3), ('mod', 'crossed', 4))}


def analyze(runs_dir, out_dir):
    from scipy.stats import fisher_exact
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, states = [], {}
    for f in sorted(Path(runs_dir).glob('*.pt')):
        run = torch.load(f)
        cfg = run['cfg']
        kl = run['kl_polished'] if np.isfinite(run['kl_polished']) else float('inf')   # a diverged run is not exact
        rows.append(dict(run=f.stem, grammar=cfg['grammar'], grouping=cfg['grouping'], r=cfg['r'], init=cfg['init'],
                         seed=cfg['seed'], lr=cfg['lr'], default_rate=cfg['lr'] == Y.LR[cfg['init']], kl_adam=run['kl_adam'], kl=kl, diverged=not np.isfinite(run['kl_polished']),
                         polish_capped=run['polish']['capped'], log=run['log']))
        states[f.stem] = run
    n_err = len(list(Path(runs_dir).glob('*.err')))
    bounds = {f'{n}_{g}': Y.tree_bound(n, g) for n in Y.GRAMMARS for g in Y.PERMS}
    all_rows, rows = rows, [q for q in rows if q['default_rate']]
    cells, best = {}, {}
    for n in Y.GRAMMARS:
        pop = Y.population(n)
        for g in Y.PERMS:
            for r in sorted({q['r'] for q in rows}):
                cell = [q for q in rows if (q['grammar'], q['grouping'], q['r']) == (n, g, r)]
                if not cell:
                    continue
                kl = np.array([q['kl'] for q in cell])
                top = min(cell, key=lambda q: q['kl'])
                c = dict(grammar=n, grouping=g, r=r, n=len(cell), n_exact=int((kl < Y.EXACT).sum()),
                         n_exact_by_init={i: sum(q['kl'] < Y.EXACT for q in cell if q['init'] == i) for i in ('variance', 'identity')},
                         n_by_init={i: sum(q['init'] == i for q in cell) for i in ('variance', 'identity')},
                         n_exact_found_by_adam=sum(q['kl'] < Y.EXACT and q['kl_adam'] < Y.ADAM_EXACT for q in cell),
                         n_between=int(((kl >= Y.EXACT) & (kl <= Y.NOT_EXACT)).sum()), n_diverged=sum(q['diverged'] for q in cell),
                         n_polish_capped=sum(q['polish_capped'] for q in cell),
                         n_between_and_capped=sum(q['polish_capped'] and Y.EXACT <= q['kl'] <= Y.NOT_EXACT for q in cell),
                         min=float(kl.min()), median=float(np.median(kl)), max=float(kl.max()), below_bound=r < bounds[f'{n}_{g}'],
                         best_run=top['run'])
                if not top['diverged']:
                    # Is the best run a finite tree that attains the target, or a loss driven down by growing weights?
                    model = Y.load_model(states[top['run']])
                    with torch.no_grad():
                        err = (Y.center(model(pop['x'])) - Y.center(pop['p'].log())).abs().max().item()
                    c['best_run_largest_weight'] = max(p_.abs().max().item() for p_ in model.parameters())
                    c['best_run_largest_centered_logit_error'] = err
                    best[top['run']] = states[top['run']]['polished']
                cells[f'{n}_{g}_r{r}'] = c
    cell = lambda n, g, r: cells.get(f'{n}_{g}_r{r}')
    smallest = {k: min((c['r'] for c in cells.values() if f"{c['grammar']}_{c['grouping']}" == k and c['n_exact']), default=None)
                for k in bounds}

    v = dict(bounds=bounds, smallest_exact_width=smallest, n_runs=len(rows), n_runs_rate_arm=len(all_rows) - len(rows), n_errored=n_err, n_cells=len(cells),
             n_runs_between_thresholds=sum(c['n_between'] for c in cells.values()),
             n_diverged=sum(c['n_diverged'] for c in cells.values()))
    below = [c for c in cells.values() if c['below_bound']]
    v['H1 no run below its bound is exact'] = ('supported' if below and min(c['min'] for c in below) > Y.NOT_EXACT else
                                                'not supported' if below else 'no runs')
    v['H1 detail'] = dict(cells=len(below), runs=sum(c['n'] for c in below), smallest_excess=min((c['min'] for c in below), default=None))
    need = [cell('mod', 'aligned', 3), cell('mod', 'crossed', 3)]
    if None in need or min(c['n'] for c in need) < N_CELL:
        v['H2 alignment saves width on mod'] = f'inconclusive (a width-3 mod cell has fewer than {N_CELL} runs)'
    else:
        a, c = need
        v['H2 alignment saves width on mod'] = ('supported' if a['n_exact'] >= 1 and c['n_exact'] == 0 and c['min'] > Y.NOT_EXACT
                                                 else 'not supported')
        v['H2 detail'] = dict(aligned_r3_exact=a['n_exact'], crossed_r3_exact=c['n_exact'], crossed_r3_min=c['min'])
    ctl = [cell('sym', 'aligned', 2), cell('sym', 'crossed', 2)]
    v['H3 existence control: on sym both groupings have an exact run at width 2'] = (
        'no runs' if None in ctl else 'supported' if all(c['n_exact'] >= 1 for c in ctl) else 'not supported')
    t = cell('mod', 'crossed', 4)
    v['H4 the crossed bound of 4 is attained on mod'] = (
        'no runs' if t is None else 'supported' if t['n_exact'] >= 1 and t['best_run_largest_centered_logit_error'] < 1e-4
        else f"open (no exact run among {t['n']}; this does not refute it)")
    if t:
        v['H4 detail'] = {k: t[k] for k in ('n', 'n_exact', 'n_exact_by_init', 'min', 'best_run', 'best_run_largest_weight',
                                             'best_run_largest_centered_logit_error')}
    # Trainability, each grouping at its own bound: exact counts of 20 (10 per init), two-sided Fisher exact test,
    # (a) at the default rates and (b) with each grouping at its own best rate per init (most exact runs of 10 among
    # the four rates of toy.LRS). (b) carries the reading; picking the best of four favors both groupings alike.
    def counts(key):
        n, g, r = key
        per = {}
        for init in ('variance', 'identity'):
            by_lr = {}
            for lr in Y.LRS:
                kl = [q['kl'] for q in all_rows if (q['grammar'], q['grouping'], q['r'], q['init'], q['lr']) == (n, g, r, init, lr)]
                if kl:
                    by_lr[lr] = (sum(k < Y.EXACT for k in kl), len(kl))
            per[init] = by_lr
        return per

    def fisher(a, c):
        p = float(fisher_exact([[a[0], a[1] - a[0]], [c[0], c[1] - c[0]]])[1])
        return p, ('no detectable difference' if p >= 0.05 else 'the crossed tree is harder to train at its bound'
                   if c[0] / c[1] < a[0] / a[1] else 'the aligned tree is harder to train at its bound')

    tr = {}
    for name, (ka, kc) in TRAIN_PAIRS.items():
        pa, pc = counts(ka), counts(kc)
        if not all(Y.LR[i] in p_[i] for p_ in (pa, pc) for i in p_):
            continue
        tot = lambda per, pick: tuple(sum(pick(per[i], i)[j] for i in per) for j in (0, 1))
        default = lambda by_lr, i: by_lr[Y.LR[i]]
        tuned = lambda by_lr, i: max(by_lr.values(), key=lambda t: t[0] / t[1])
        d = {}
        for label, pick in (('default_rates', default), ('best_rate_per_init', tuned)):
            a, c = tot(pa, pick), tot(pc, pick)
            p, reading = fisher(a, c)
            d[label] = dict(aligned=f'{a[0]}/{a[1]} at r={ka[2]}', crossed=f'{c[0]}/{c[1]} at r={kc[2]}', fisher_p=p, reading=reading)
        d['rates_complete'] = all(len(p_[i]) == len(Y.LRS) for p_ in (pa, pc) for i in p_)
        d['exact_of_n_by_init_and_rate'] = {g: {i: {f'{lr:g}': f'{e}/{m}' for lr, (e, m) in per[i].items()} for i in per}
                                            for g, per in (('aligned', pa), ('crossed', pc))}
        d['reading'] = d['best_rate_per_init']['reading'] if d['rates_complete'] else 'rate arm incomplete: ' + d['default_rates']['reading'] + ' (default rates only)'
        tr[name] = d
    v['trainability at the bound (pre-stated reading)'] = tr
    json.dump([{k: q[k] for k in q if k != 'log'} for q in all_rows], open(out_dir / 'runs.json', 'w'), indent=1)
    json.dump({q['run']: q['log'] for q in rows}, open(out_dir / 'curves.json', 'w'))
    json.dump(dict(cells=cells, verdicts=v), open(out_dir / 'summary.json', 'w'), indent=1)
    json.dump(Y.provenance(), open(out_dir / 'provenance.json', 'w'), indent=1)
    torch.save(best, out_dir / 'best_trees.pt')   # the best tree of every cell: the witnesses of "exact", kept in git

    print(f'{len(rows)} runs at the default rates in {len(cells)} cells, {len(all_rows) - len(rows)} in the rate arm, {n_err} errored. exact: excess loss below {Y.EXACT:g} nats after the polish; uniform is {Y.UNIFORM_KL:.6f}.')
    print('cell                 bound  n  exact (var, id; by Adam)  between  capped  min excess  median     max        min/uniform  best run: max |weight|, max logit error')
    for k, c in cells.items():
        print(f"{k:20s} {bounds[c['grammar'] + '_' + c['grouping']]:5d} {c['n']:3d} {c['n_exact']:5d} ({c['n_exact_by_init']['variance']:2d}, {c['n_exact_by_init']['identity']:2d}; {c['n_exact_found_by_adam']:2d})"
              f"        {c['n_between']:3d}     {c['n_polish_capped']:3d}     {c['min']:.2e}   {c['median']:.2e}  {c['max']:.2e}   {c['min'] / Y.UNIFORM_KL:.3f}"
              f"        {c.get('best_run_largest_weight', float('nan')):.2f}, {c.get('best_run_largest_centered_logit_error', float('nan')):.1e}")
    print(json.dumps(v, indent=1))
    if None in need or min(c['n'] for c in need) < N_CELL:
        print(f'RUN FAILED: a width-3 mod cell has fewer than {N_CELL} runs')


if __name__ == '__main__':
    analyze(sys.argv[1], sys.argv[2])
