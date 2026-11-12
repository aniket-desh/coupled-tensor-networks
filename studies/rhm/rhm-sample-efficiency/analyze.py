"""rhm-sample-efficiency analysis: test accuracy and loss vs N for the r = 16 TreeTN with the identity init, next to
how well each run fit its training set.

  PYTHONPATH=. python studies/rhm/rhm-sample-efficiency/analyze.py runs/rhm-sample-efficiency studies/rhm/rhm-sample-efficiency/results

Selections (from the diagnostic set, never the test set): accuracy-based numbers ('learned', the regimes, H1) use the
checkpoint with the highest diagnostic accuracy; loss and KL use the one with the lowest diagnostic loss.
Regime per run (primary table, counts per (task, N)):
  learned               accuracy-selected test accuracy >= 0.9 x Bayes accuracy
  data-limited          not learned, and at the end the KL to Bayes on (the first 20,000 trees of) its own training
                        set is < 0.1 nats (it fit its data; for 'root' this KL is the training loss)
  optimization-limited  not learned, the run ended normally, and that training-set KL is >= 0.1 nats
  unfinished            truncated, diverged, skipped, failed or missing (never counted as a regime)
Medians per (task, N) need all 3 grammars finished; a size with fewer is missing for H1.
Writes <results>/analysis.json and <results>/curves.json."""
import argparse
import math
from collections import defaultdict
from pathlib import Path


from studies.rhm.analysis import THRESH, curves, dump, health, load_runs, lr_info, med_range, ok

P_STAR = 16 * 4 ** 4  # n m^L = 4,096: the RHM sample complexity of deep CNNs (Cagnetta et al.); N is given in units of it
FIT_KL, NS = 0.1, (2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20)


def summarize(r):
    c = r['cfg']
    sa, sl = r.get('best_acc', {}), r.get('best', {})
    acc = sa.get('test', r['test'])['acc']
    fit_kl = r['log']['fit_kl'][-1] if 'fit_kl' in r['log'] else None
    # fit = final training loss on the fit subset minus the Bayes loss on the same trees (0 for 'root'); < FIT_KL nats
    fit_excess = (r['log']['fit_loss'][-1] - (r.get('bayes_fit') or {}).get('loss', 0.0)) if 'fit_loss' in r['log'] else None
    learned = acc >= THRESH * r['bayes_test']['acc']
    regime = ('unfinished' if not ok(r) else 'learned' if learned else 'not learned (online)' if r['data'] == 'online'
              else 'data-limited' if fit_excess is not None and fit_excess < FIT_KL else 'optimization-limited')
    return dict(task=c['task'], g=c['grammar']['seed'], N=r['data'], status=r['status'], steps=r['steps_done'],
                acc_step=sa.get('step'), test_acc=acc, loss_step=sl.get('step'),
                test_loss=sl.get('test', r['test'])['loss'], test_kl=sl.get('test', r['test'])['kl'],
                final_test_loss=r['test']['loss'], final_test_acc=r['test']['acc'], fit_kl_final=fit_kl, fit_excess_final=fit_excess,
                fit_loss_final=r['log']['fit_loss'][-1] if 'fit_loss' in r['log'] else None,
                bayes_loss=r['bayes_test']['loss'], bayes_acc=r['bayes_test']['acc'], regime=regime,
                ms_per_step=r['ms_per_step'])


def h1(med):
    """Decide on the sizes present (median over 3 finished grammars); inconclusive only if no present size decides."""
    lo, mid, hi = med.get(2 ** 10), {N: med.get(N) for N in (2 ** 12, 2 ** 14, 2 ** 16)}, {N: med.get(N) for N in (2 ** 18, 2 ** 20)}
    reached = [N for N, v in mid.items() if v is not None and v >= 0.9]
    if reached:
        if lo is None:
            return f'supported at N = {min(reached)} (2^10 missing: the low end is untested)'
        return 'supported' if lo < 0.5 else 'partly supported (already >= 0.5 at 2^10 = P*/4)'
    if mid[2 ** 16] is not None:  # every present size up to 16 P* is below 0.9, and 16 P* itself is present
        big = [N for N, v in hi.items() if v is not None and v >= 0.9]
        return ('not supported (reached only at N = %d)' % min(big) if big else
                'not supported (not reached at any present size)')
    return 'inconclusive (2^16 missing and no present size up to it reaches 0.9)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs')
    p.add_argument('results')
    a = p.parse_args()
    runs = load_runs(a.runs)
    S = {n: summarize(r) for n, r in runs.items()}
    hl = health(a.runs, runs)
    groups = defaultdict(list)
    for s in S.values():
        groups[(s['task'], s['N'])].append(s)
    regimes = {}
    for t in ('root', 'last'):
        for N in NS:
            ss = groups.get((t, N), [])
            fin = [s for s in ss if s['regime'] != 'unfinished']
            regimes[f'{t}_N{N}'] = dict(N=N, N_over_Pstar=N / P_STAR, finished=len(fin), of=3,
                                        **{k: sum(s['regime'] == k for s in fin) for k in
                                           ('learned', 'data-limited', 'optimization-limited')})
    table = {}
    for (t, N), ss in sorted(groups.items(), key=lambda kv: (kv[0][0], math.inf if kv[0][1] == 'online' else kv[0][1])):
        fin = [s for s in ss if s['regime'] != 'unfinished']
        full = len(fin) == 3
        table[f'{t}_N{N}'] = dict(N=N, N_over_Pstar=(N / P_STAR) if N != 'online' else None, finished=len(fin),
                                  median_valid=full, test_acc=med_range([s['test_acc'] for s in fin]),
                                  test_loss=med_range([s['test_loss'] for s in fin]),
                                  test_kl=med_range([s['test_kl'] for s in fin]),
                                  fit_kl_final=med_range([s['fit_kl_final'] for s in fin]),
                                  bayes_loss=med_range([s['bayes_loss'] for s in fin]),
                                  ms_per_step=med_range([s['ms_per_step'] for s in ss]))
    med = lambda t: {N: table[f'{t}_N{N}']['test_acc']['median'] for N in NS
                     if f'{t}_N{N}' in table and table[f'{t}_N{N}']['median_valid']}
    crit = {'H1_root_sample_complexity': dict(median_test_acc_by_N=med('root'), verdict=h1(med('root')))}
    big = [s for s in S.values() if s['N'] != 'online' and s['N'] >= 2 ** 16]
    planned_big = 3 * 2 * 3  # N in {2^16, 2^18, 2^20} x 3 grammars x 2 tasks
    fin_big = [s for s in big if s['regime'] != 'unfinished']
    opt = [f"{s['task']}_g{s['g']}_N{s['N']}" for s in fin_big if s['regime'] == 'optimization-limited']
    crit['H2_failures_are_data_not_optimization (N >= 2^16)'] = dict(
        planned=planned_big, finished=len(fin_big), optimization_limited=opt, threshold=planned_big / 4,
        verdict=('inconclusive (fewer than half of the N >= 2^16 runs finished)' if len(fin_big) < planned_big / 2 else
                 'supported' if not opt else 'not supported' if len(opt) > planned_big / 4 else 'partly supported'))
    crit['descriptive_last_kl_vs_N'] = {N: table.get(f'last_N{N}', {}).get('test_kl', {}).get('median') for N in NS}
    crit['descriptive_online_reference'] = {t: table.get(f'{t}_Nonline') for t in ('root', 'last')}
    res = dict(health=hl, P_star=P_STAR, regimes=regimes, criteria=crit, table=table, lr=lr_info(a.runs), runs=S)
    dump(Path(a.results) / 'analysis.json', res)
    dump(Path(a.results) / 'curves.json', curves(runs, keys=('train_loss', 'diag_loss', 'diag_acc', 'diag_kl',
                                                             'fit_loss', 'fit_acc', 'fit_kl')))
    for k, v in crit.items():
        if isinstance(v, dict) and 'verdict' in v:
            print(f"{k}: {v['verdict']}")
    print('regimes:', {k: {kk: v[kk] for kk in ('finished', 'learned', 'data-limited', 'optimization-limited')}
                       for k, v in regimes.items() if v['finished']})


if __name__ == '__main__':
    main()
