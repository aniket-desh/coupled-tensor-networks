"""Tables for the README, computed from the saved results only; also written to results/summary.json.

  python -m studies.rhm.child_feature_enabling.summary
"""
import json
import statistics
from pathlib import Path

import numpy as np

from studies.rhm.child_feature_enabling import record

HERE = Path(__file__).resolve().parent
ARMS = ('intact', 'detached')


def first_at_least(steps, values, level=0.5):
    return next((s for s, v in zip(steps, values) if v >= level), None)


def planted():
    data = json.loads((HERE / 'results/planted/planted.json').read_text())
    out = dict(inv_lambda_2=1 / data['meta']['lambda_2'], rows=[])
    print('\n## Planted sweep\n')
    print('| arm | delta | T(r_L=1/2) | T(r_R=1/2) | T(D_h=1/2) | wait | delta^2 T_h | full-table |dT| |')
    print('|---|---|---|---|---|---|---|---|')
    for arm in data['arms']:
        e = arm['events']
        row = dict(tied=arm['tied'], s=arm['s'], delta=arm['delta'], **e, wait=e['hard_half'] - e['left_half'],
                   delta2_T=arm['delta'] ** 2 * e['hard_half'],
                   full_table_max_dt=max(arm['event_time_abs_diff_full_vs_reduced'].values()))
        out['rows'].append(row)
        print(f"| {'tied' if arm['tied'] else 'untied'}, s={arm['s']} | {arm['delta']} | {e['left_half']:.4f} | "
              f"{e['right_half']:.4f} | {e['hard_half']:.4f} | {row['wait']:.4f} | {row['delta2_T']:.4f} | "
              f"{row['full_table_max_dt']:.1e} |")
    for tied in (True, False):
        rows = [r for r in out['rows'] if r['tied'] == tied and r['s'] == 1]
        slopes = np.diff([r['wait'] for r in rows]) / np.diff(np.log([1 / r['delta'] for r in rows]))
        out[f"wait_slopes_{'tied' if tied else 'untied'}"] = slopes.tolist()
        print(f"\nadjacent slopes of the wait, {'tied' if tied else 'untied'}: "
              + ', '.join(f'{x:.4f}' for x in slopes) + f"  (predicted {0 if tied else out['inv_lambda_2']:.4f})")
    out['tolerance_sensitivity_max_change'] = max(
        abs(v) for s in data['tolerance_sensitivity'] for k in ('change', 'change_full_table') for v in s[k].values())
    out['gradient_check_max_error'] = max(data['gradient_check']['max_abs_errors'].values())
    print(f"largest event-time change at tolerances / 4 (delta = 0.005): {out['tolerance_sensitivity_max_change']:.1e}")
    return out


def checkpoints():
    data = json.loads((HERE / 'results/checkpoints/checkpoints.json').read_text())
    rows = [r for r in data['rows'] if r['status'] == 'ok']
    out = dict(n_ok=len(rows), missing=data['meta']['missing'], runs=[])
    print(f"\n## Checkpoint readout\n\n{len(rows)} checkpoints read, {len(out['missing'])} missing\n")
    print('| grammar | additive fraction easy | hard | easy gap | hard gap |')
    print('|---|---|---|---|---|')
    for g, spec in data['grammars'].items():
        print(f"| {g} | {spec['easy']['additive_fraction']:.4f} | {spec['hard']['additive_fraction']:.4f} | "
              f"{spec['easy']['neighbor_gap']:.4f} | {spec['hard']['neighbor_gap']:.4f} |")
    shown = (0, 100, 150, 200, 300, 40000)
    print('\n| run | ' + ' | '.join(f'A_h, A_e @ {s}' for s in shown) + ' | acc @ 200 | max A_h | final D_h^CE |')
    print('|---|' + '---|' * (len(shown) + 3))
    for name in sorted({r['run'] for r in rows}):
        run = {r['step']: r for r in rows if r['run'] == name}
        out['runs'].append(dict(run=name, A_h={s: run[s]['A_h'] for s in run}, A_e={s: run[s]['A_e'] for s in run},
                                ce={s: run[s]['ce'] for s in run}, acc={s: run[s]['acc'] for s in run},
                                D_h_ce={s: run[s]['D_h_ce'] for s in run}))
        print(f'| {name} | ' + ' | '.join(f"{run[s]['A_h']:.3f}, {run[s]['A_e']:.3f}" for s in shown)
              + f" | {run[200]['acc']:.3f} | {max(r['A_h'] for r in run.values()):.3f} | {run[40000]['D_h_ce']:.4f} |")
    out['A_h_step0_range'] = [min(r['A_h'] for r in rows if r['step'] == 0), max(r['A_h'] for r in rows if r['step'] == 0)]
    out['A_e_step0_range'] = [min(r['A_e'] for r in rows if r['step'] == 0), max(r['A_e'] for r in rows if r['step'] == 0)]
    late = [r for r in rows if r['step'] >= 300]
    out['A_h_after_300_range'] = [min(r['A_h'] for r in late), max(r['A_h'] for r in late)]
    out['A_e_after_300_range'] = [min(r['A_e'] for r in late), max(r['A_e'] for r in late)]
    out['easy_leads_hard'] = dict(  # is A_e > A_h at every checkpoint up to the transition?
        n=sum(r['step'] <= 300 for r in rows), n_true=sum(r['A_e'] > r['A_h'] for r in rows if r['step'] <= 300))
    out['cutoff_sensitivity_max'] = max(abs(r[f'A_{k}'] - r[f'A_{k}_cutoff10']) for r in rows for k in 'eh')
    out['min_rank'] = min(r['A_h_rank'] for r in rows)
    out['logged_loss_max_abs_diff'] = max(abs(r['ce'] - r['logged_diag_loss']) for r in rows if 'logged_diag_loss' in r)
    out['ce_reveal1_max_width'] = max(r['ce_reveal1_upper'] - r['ce_reveal1_lower'] for r in rows)
    out['ce_reveal1_min_width'] = min(r['ce_reveal1_upper'] - r['ce_reveal1_lower'] for r in rows)
    out['final_test'] = data['final_test']
    out['D_ce_after_300_min'] = {k: min(r[f'D_{k}_ce'] for r in late) for k in 'eh'}
    out['acc_range_by_step'] = {step: [min(r['acc'] for r in rows if r['step'] == step),
                                       max(r['acc'] for r in rows if r['step'] == step)] for step in (100, 150, 200, 300)}
    out['A_e_over_A_h'] = {step: [min(r['A_e'] / r['A_h'] for r in rows if r['step'] == step),
                                  max(r['A_e'] / r['A_h'] for r in rows if r['step'] == step)] for step in (100, 150)}
    # the contrasts' additive fractions against the committed audit of the same rule tables
    audit = json.loads((HERE / 'results/grammar_audit.json').read_text())['tables']
    out['contrast_vs_audit_max_abs_diff'] = max(
        max(abs(data['grammars'][str(t['seed'])]['easy']['additive_fraction'] - t['additive_fraction_max']),
            abs(data['grammars'][str(t['seed'])]['hard']['additive_fraction'] - t['additive_fraction_min']))
        for t in audit)
    out['contrast_eigenspace_dims'] = sorted({spec[k]['eigenspace_dim'] for spec in data['grammars'].values()
                                              for k in ('easy', 'hard')})
    for key in ('A_h_step0_range', 'A_e_step0_range', 'A_h_after_300_range', 'A_e_after_300_range', 'easy_leads_hard',
                'cutoff_sensitivity_max', 'min_rank', 'logged_loss_max_abs_diff', 'ce_reveal1_min_width',
                'ce_reveal1_max_width', 'D_ce_after_300_min', 'acc_range_by_step', 'A_e_over_A_h',
                'contrast_vs_audit_max_abs_diff', 'contrast_eigenspace_dims'):
        print(f'{key}: {out[key]}')
    print('final test (50,000 trees): acc ' + ', '.join(f"{t['acc']:.5f}" for t in data['final_test'])
          + ' | CE ' + ', '.join(f"{t['ce']:.2e}" for t in data['final_test']))
    return out


def intervention(folder):
    files = sorted((HERE / folder / 'pairs').glob('g*_s*.json'))
    if not files:
        return None
    pairs = [json.loads(f.read_text()) for f in files]
    horizon, lr = pairs[0]['steps'], pairs[0]['lr']
    out = dict(folder=folder, lr=lr, horizon=horizon, n_pairs=len(pairs), pairs=[],
               commits=sorted({p['meta']['commit'] for p in pairs}),
               uncommitted=sorted({str(p['meta']['uncommitted_diff_sha256']) for p in pairs}))
    print(f'\n## Intervention: {folder} (rate {lr:g}, horizon {horizon:,}, {len(pairs)} pairs)\n')
    print('| pair | T intact | T detached | ratio | T_A intact | T_A detached | final D_h intact (A_h) | '
          'final D_h detached (A_h) | status |')
    print('|---|---|---|---|---|---|---|---|---|')
    for p in pairs:
        row = dict(grammar=p['grammar'], seed=p['seed'], status=p['status'], init=p['init']['kind'],
                   seconds=p['seconds'])
        for arm in ARMS:
            cols, cross = p['arms'][arm]['columns'], p['arms'][arm]['crossing']
            row[arm] = dict(T=cross['T'], censored=cross['censored'], bracket=cross.get('bracket'),
                            T_ceiling=first_at_least(cols['step'], cols['A_h']),
                            T_easy=first_at_least(cols['step'], cols['D_e']),
                            final_D_h=cols['D_h'][-1], final_A_h=cols['A_h'][-1], final_D_e=cols['D_e'][-1],
                            final_A_e=cols['A_e'][-1], max_D_h=max(cols['D_h']),
                            max_D_minus_A=p['arms'][arm]['max_D_minus_A'], final_lower_shift=cols['lower_shift'][-1],
                            last_step=cols['step'][-1])
            if not cross['censored']:  # state of the arm at its own half-fit evaluation
                k = cols['step'].index(cross['T'])
                row[arm]['at_T'] = dict(A_h=cols['A_h'][k], A_e=cols['A_e'][k], D_e=cols['D_e'][k],
                                        lower_shift=cols['lower_shift'][k])
            row[arm]['min_rank'] = min(cols['rank_h'])
            row[arm]['cutoff_sensitivity'] = max(abs(cols[f'A_{k}'][j] - cols[f'A_{k}_cutoff10'][j])
                                                 for k in 'eh' for j in range(len(cols['step'])))
            gaps = [a - d for a, d in zip(cols['A_h'], cols['D_h'])]
            row[arm]['max_ceiling_gap'] = max(gaps)
            row[arm]['median_ceiling_gap'] = statistics.median(gaps)
            row[arm]['n_evaluations'] = len(gaps)
            row[arm]['n_gap_above_0.05'] = sum(g > 0.05 for g in gaps)
            # single-batch gradient diagnostics over the first 1,000 updates (evaluations 1..40; none at update 0,
            # where the zero heads send no gradient down)
            early = [k for k, st in enumerate(cols['step']) if 0 < st <= 1000 and cols['grad_easy'][k] is not None]
            if early:
                row[arm]['early_grad_ratio_easy_over_hard'] = statistics.median(
                    cols['grad_easy'][k] / cols['grad_hard'][k] for k in early)
                row[arm]['early_grad_cos'] = statistics.median(cols['grad_cos'][k] for k in early)
            # the largest single-evaluation fall of the trained hard fit: an instability shows up here
            row[arm]['largest_drop_D_h'] = float(-np.diff(cols['D_h']).min())
        a, b = row['intact'], row['detached']
        row['capped_difference'] = min(b['T'] or horizon, horizon) - min(a['T'] or horizon, horizon)
        row['ratio'] = None if a['censored'] or b['censored'] else b['T'] / a['T']
        row['ratio_lower_bound'] = None if a['censored'] or not b['censored'] else horizon / a['T']
        out['pairs'].append(row)
        fmt = lambda x: f"{x['T']:,}" if not x['censored'] else f'> {horizon:,}'
        tc = lambda x: f"{x['T_ceiling']:,}" if x['T_ceiling'] is not None else f'> {horizon:,}'
        ratio = (f"{row['ratio']:.2f}" if row['ratio'] else
                 f"> {row['ratio_lower_bound']:.2f}" if row['ratio_lower_bound'] else 'n/a')
        print(f"| g{p['grammar']} s{p['seed']} | {fmt(a)} | {fmt(b)} | {ratio} | {tc(a)} | {tc(b)} | "
              f"{a['final_D_h']:.4f} ({a['final_A_h']:.4f}) | {b['final_D_h']:.4f} ({b['final_A_h']:.4f}) | "
              f"{p['status']} |")
    rows = out['pairs']
    out['censored'] = {arm: sum(r[arm]['censored'] for r in rows) for arm in ARMS}
    out['nonfinite'] = sum(r['status'] != 'finished' for r in rows)
    out['intact_first'] = sum(r['capped_difference'] > 0 for r in rows)
    out['detached_first'] = sum(r['capped_difference'] < 0 for r in rows)
    out['capped_difference'] = dict(values=[r['capped_difference'] for r in rows],
                                    median=statistics.median(r['capped_difference'] for r in rows),
                                    min=min(r['capped_difference'] for r in rows),
                                    max=max(r['capped_difference'] for r in rows))
    ratios = [r['ratio'] for r in rows if r['ratio']]
    if ratios:
        out['ratio'] = dict(n=len(ratios), median=statistics.median(ratios), min=min(ratios), max=max(ratios))
    for arm in ARMS:
        ts = [r[arm]['T'] for r in rows if not r[arm]['censored']]
        if ts:
            out[f'T_{arm}'] = dict(n=len(ts), median=statistics.median(ts), min=min(ts), max=max(ts))
        lags = [r[arm]['T'] - r[arm]['T_ceiling'] for r in rows if not r[arm]['censored']]
        if lags:
            out[f'ceiling_to_fit_lag_{arm}'] = dict(n=len(lags), median=statistics.median(lags), min=min(lags),
                                                    max=max(lags))
    for arm in ARMS:
        at = [r[arm]['at_T'] for r in rows if 'at_T' in r[arm]]
        if at:
            out[f'at_T_{arm}'] = {k: dict(n=len(at), median=statistics.median(x[k] for x in at),
                                          min=min(x[k] for x in at), max=max(x[k] for x in at)) for k in at[0]}
    out['max_ceiling_gap'] = {arm: max(r[arm]['max_ceiling_gap'] for r in rows) for arm in ARMS}
    out['median_ceiling_gap'] = {arm: [min(r[arm]['median_ceiling_gap'] for r in rows),
                                       max(r[arm]['median_ceiling_gap'] for r in rows)] for arm in ARMS}
    out['evaluations_with_gap_above_0.05'] = {arm: [sum(r[arm]['n_gap_above_0.05'] for r in rows),
                                                    sum(r[arm]['n_evaluations'] for r in rows)] for arm in ARMS}
    for key in ('early_grad_ratio_easy_over_hard', 'early_grad_cos'):
        if all(key in r[arm] for r in rows for arm in ARMS):
            out[key] = {arm: dict(median=statistics.median(r[arm][key] for r in rows),
                                  min=min(r[arm][key] for r in rows), max=max(r[arm][key] for r in rows))
                        for arm in ARMS}
    out['min_rank'] = min(r[arm]['min_rank'] for r in rows for arm in ARMS)
    out['cutoff_sensitivity_max'] = max(r[arm]['cutoff_sensitivity'] for r in rows for arm in ARMS)
    out['max_final_or_any_D_h'] = {arm: max(r[arm]['max_D_h'] for r in rows) for arm in ARMS}
    out['steps_done'] = [min(r[arm]['last_step'] for r in rows for arm in ARMS),
                         max(r[arm]['last_step'] for r in rows for arm in ARMS)]
    out['seconds'] = dict(total=sum(r['seconds'] for r in rows), min=min(r['seconds'] for r in rows),
                          max=max(r['seconds'] for r in rows))
    out['init_kinds'] = sorted({r['init'] for r in rows})
    out['max_D_minus_A'] = max(r[arm]['max_D_minus_A'] for r in rows for arm in ARMS)
    out['largest_drop_D_h'] = {arm: max(r[arm]['largest_drop_D_h'] for r in rows) for arm in ARMS}
    out['final_D_h_range'] = {arm: [min(r[arm]['final_D_h'] for r in rows), max(r[arm]['final_D_h'] for r in rows)]
                              for arm in ARMS}
    out['final_A_h_range'] = {arm: [min(r[arm]['final_A_h'] for r in rows), max(r[arm]['final_A_h'] for r in rows)]
                              for arm in ARMS}
    out['final_A_e_range'] = {arm: [min(r[arm]['final_A_e'] for r in rows), max(r[arm]['final_A_e'] for r in rows)]
                              for arm in ARMS}
    for key in ('censored', 'nonfinite', 'intact_first', 'detached_first', 'capped_difference', 'ratio', 'T_intact',
                'T_detached', 'ceiling_to_fit_lag_intact', 'ceiling_to_fit_lag_detached', 'at_T_intact', 'at_T_detached', 'max_ceiling_gap',
                'median_ceiling_gap', 'evaluations_with_gap_above_0.05', 'early_grad_ratio_easy_over_hard',
                'early_grad_cos', 'min_rank', 'cutoff_sensitivity_max', 'max_final_or_any_D_h', 'steps_done',
                'seconds', 'init_kinds',
                'max_D_minus_A',
                'largest_drop_D_h', 'final_D_h_range', 'final_A_h_range', 'final_A_e_range', 'commits', 'uncommitted'):
        if key in out:
            print(f'{key}: {out[key]}')
    pilot = HERE / folder / 'pilot.json'
    if pilot.exists():
        pl = json.loads(pilot.read_text())
        out['pilot'] = dict(choice=pl['choice'], steps=pl['steps'], rates=[
            dict(lr=r['lr'], eligible=r['eligible'], max_pop_loss=r['max_pop_loss'],
                 mean_final_hard_mse=r['mean_final_hard_mse'], final_hard_mse=r['final_hard_mse'],
                 crossing={arm: r['pair']['arms'][arm]['crossing']['T'] for arm in ARMS},
                 largest_drop_D_e={arm: float(-np.diff(r['pair']['arms'][arm]['columns']['D_e']).min()) for arm in ARMS})
            for r in pl['results']])
        print(f"pilot ({pl['steps']} updates): choice {pl['choice']}")
        for r in out['pilot']['rates']:
            print('  ', r)
        # Does rate x updates act as a clock? The same pilot pair at equal rate x updates, for each rate whose
        # schedule has an evaluation there.
        out['pilot']['equal_flow_time'] = {}
        for t in (30, 120, 300):
            rows_t = {}
            for r in pl['results']:
                cols = {arm: r['pair']['arms'][arm]['columns'] for arm in ARMS}
                step = round(t / r['lr'])
                if step in cols['intact']['step'] and abs(step * r['lr'] - t) < 1e-9:
                    k = cols['intact']['step'].index(step)
                    rows_t[f"{r['lr']:g}"] = {arm: dict(D_e=cols[arm]['D_e'][k], D_h=cols[arm]['D_h'][k],
                                                        A_e=cols[arm]['A_e'][k], A_h=cols[arm]['A_h'][k]) for arm in ARMS}
            out['pilot']['equal_flow_time'][t] = rows_t
            print(f'  pilot pair at rate x updates = {t}: ' + '; '.join(
                f"rate {lr}: intact D_e {v['intact']['D_e']:.4f}, D_h {v['intact']['D_h']:.4f}" for lr, v in rows_t.items()))
    return out


def rate_comparison(a, b):
    """The pairs run at two rates: half-fit times in updates and in rate x updates, and the detached/intact ratio."""
    rows = []
    print(f"\n## Same pairs at rates {a['lr']:g} and {b['lr']:g}\n")
    print('| pair | rate | T intact | T detached | ratio | rate x T intact | rate x T detached |')
    print('|---|---|---|---|---|---|---|')
    for rb in b['pairs']:
        ra = next(r for r in a['pairs'] if (r['grammar'], r['seed']) == (rb['grammar'], rb['seed']))
        for src, r in ((a, ra), (b, rb)):
            ti, td = r['intact']['T'], r['detached']['T']
            rows.append(dict(grammar=r['grammar'], seed=r['seed'], lr=src['lr'], T_intact=ti, T_detached=td,
                             ratio=r['ratio'], flow_intact=None if ti is None else src['lr'] * ti,
                             flow_detached=None if td is None else src['lr'] * td))
            f = lambda x, d=0: 'censored' if x is None else f'{x:,.{d}f}'
            print(f"| g{r['grammar']} s{r['seed']} | {src['lr']:g} | {f(ti)} | {f(td)} | {f(r['ratio'], 2)} | "
                  f"{f(rows[-1]['flow_intact'])} | {f(rows[-1]['flow_detached'])} |")
    return rows


def main():
    out = dict(planted=planted(), checkpoints=checkpoints())
    for folder in ('results/intervention', 'results/intervention-extended', 'results/intervention-extended-lr0.1'):
        result = intervention(folder)
        if result is not None:
            out[Path(folder).name] = result
    if 'intervention-extended' in out and 'intervention-extended-lr0.1' in out:
        out['rate_comparison'] = rate_comparison(out['intervention-extended'], out['intervention-extended-lr0.1'])
    record.dump(HERE / 'results' / 'summary.json', out)


if __name__ == '__main__':
    main()
