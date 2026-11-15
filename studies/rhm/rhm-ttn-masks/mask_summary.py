"""Compare the three arms of rhm-ttn-masks and apply the criteria of EXPERIMENT.md.

    PYTHONPATH=. python studies/rhm/rhm-ttn-masks/mask_summary.py RESULTS_DIR RUNS_DIR [OLD_SUMMARY]

Reads RESULTS_DIR/<arm>/summary.json (written per arm by rhm-ttn-messages/msg_summary.py, whose measures, scopes and
H1-H4 rules are used unchanged) and RUNS_DIR/<arm>/<run>/result.json (losses), writes RESULTS_DIR/summary.json.
OLD_SUMMARY is rhm-ttn-messages/results/summary.json: the prefix arm repeats those runs, so its numbers are printed next
to them as a check on this experiment's training script.

Rules (see EXPERIMENT.md; all at the final checkpoint, parents of tree level >= 2, both-informative pairs):
  H_A  supported: the `mixed` arm's H4 verdict from msg_summary is "supported" (H4 holds in all three seeds of at least
       3 of the 4 grammars) while the `prefix` arm's is not.
       partly supported: `mixed` passes H4 in 1 or 2 grammars, or closes the gap over `prefix` on the both-informative
       R2 (1 - R2_mixed <= 0.5 (1 - max(R2_prefix, 0)), seed by seed) in all three seeds of at least 3 grammars.
       not supported: neither, AND the mixed arm learned the varied task (varied excess loss < VEXCESS_MAX in all three
       seeds) in at least 3 grammars. Without that, a failed H4 is inconclusive: the tree was not shown to predict on
       the patterns that are supposed to teach it the composition.
  H_B  over the (grammar, seed) pairs where both arms pass the prefix gate: the invalid pairs' read-out mass ratio
       (m3_invalid_q; 0 for exact messages) is lower in `mixed` than in `prefix` in at least 10 of 12 pairs AND the
       median of the paired ratios mixed / prefix is at most 0.5.
A grammar is judged for an arm only if all three seeds finished and reach prefix excess loss < 0.02 nats/token
(msg_summary's `learned`), so an arm cannot pass by predicting worse.
The run failed (`run_failed` in the output) if fewer than MIN_RUNS / 36 of the runs of the arms analysed so far were
analysed (the `varied` arm comes in a second step and may be censored for time), or if a prefix-arm run
differs from the same run of rhm-ttn-messages by more than REPRO_MAX nats/token of prefix excess loss."""
import json
import sys
from pathlib import Path

ARMS, GAP = ('prefix', 'varied', 'mixed'), 0.5
VEXCESS_MAX, MIN_RUNS, REPRO_MAX = 0.02, 31, 0.005
KEYS = ('m4_both', 'm4_both_ceil', 'm4_both_init', 'm4_emp', 'm4_left', 'm4_dir', 'm3_acc', 'm3_invalid_q',
        'm3_valid_q_cv', 'm1_proj', 'm2_ve', 'm4a_emp')


def main():
    res, runs_dir = Path(sys.argv[1]), Path(sys.argv[2])
    S = {a: json.load(open(res / a / 'summary.json')) for a in ARMS if (res / a / 'summary.json').exists()}
    R = {a: {q['run']: q for q in S[a]['runs']} for a in S}
    for a in S:
        for name, q in R[a].items():
            h = json.load(open(runs_dir / a / name / 'result.json'))['hist'][-1]
            q.update(vexcess=h['vexcess'])
    cells = sorted({c for a in S for c in S[a]['cells']})
    f = lambda v: '   -  ' if v is None else f'{v:6.2f}'
    print('per run, final checkpoint (both = M4 both-informative R2, level >= 2; inv_q = invalid / valid read-out mass)')
    print(f'{"run":20s} ' + ' | '.join(f'{a:>7s}: excess vexcess  both  ceil inv_q  h4' for a in S))
    for c in cells:
        for s in range(3):
            name, line = f'{c}_s{s}', []
            for a in S:
                q = R[a].get(name)
                line.append('    (missing)' if q is None else
                            f'         {q["excess"]:6.3f} {q["vexcess"]:7.3f}{f(q["m4_both"])}{f(q["m4_both_ceil"])}'
                            f'{f(q["m3_invalid_q"])}  {str(q.get("h4"))[0]}')
            print(f'{name:20s} ' + ' | '.join(line))
    out = dict(arms={a: dict(verdicts=S[a]['verdicts'], cells=S[a]['cells'], run_failed=S[a]['run_failed'],
                             runs={n: {k: q.get(k) for k in KEYS + ('excess', 'vexcess', 'learned', 'status', 'h1', 'h2', 'h3', 'h4')}
                                   for n, q in R[a].items()}) for a in S})
    print('\nH4 per arm (msg_summary): ' + '; '.join(
        f'{a}: {v["verdict"]} ({v["supported"]} of {len(v["judged"])})'
        for a in S for k, v in S[a]['verdicts'].items() if k.startswith('h4_')))
    if {'prefix', 'mixed'} <= set(S):
        h4 = {a: next(v for k, v in S[a]['verdicts'].items() if k.startswith('h4_')) for a in ('prefix', 'mixed')}
        pairs = [(n, R['prefix'][n], R['mixed'][n]) for n in sorted(R['prefix']) if n in R['mixed']]
        closes = lambda new, old: None not in (new, old) and 1 - new <= GAP * (1 - max(old, 0))
        both_ok = lambda c: all(R[a].get(f'{c}_s{s}', {}).get('learned') for a in ('prefix', 'mixed') for s in range(3))
        closed = [c for c in cells if both_ok(c) and all(
            closes(R['mixed'][f'{c}_s{s}']['m4_both'], R['prefix'][f'{c}_s{s}']['m4_both']) for s in range(3))]
        vlearned = [c for c in cells if all(
            (R['mixed'].get(f'{c}_s{s}') or {}).get('vexcess', 9) < VEXCESS_MAX for s in range(3))]
        if h4['mixed']['verdict'] == 'supported' and h4['prefix']['verdict'] != 'supported':
            ha = 'supported'
        elif h4['mixed']['verdict'].startswith('inconclusive'):
            ha = 'inconclusive (fewer than 3 grammars learned by the mixed arm)'
        elif h4['mixed']['supported'] >= 1 or len(closed) >= 3:
            ha = 'partly supported'
        elif len(vlearned) >= 3:
            ha = 'not supported'
        else:
            ha = f'inconclusive (H4 not met, and the mixed arm learned the varied task in only {len(vlearned)} grammars)'
        inv = [(n, p['m3_invalid_q'], m['m3_invalid_q']) for n, p, m in pairs
               if None not in (p['m3_invalid_q'], m['m3_invalid_q']) and p['learned'] and m['learned']]
        lower = sum(m < p for _, p, m in inv)
        ratios = sorted(m / p for _, p, m in inv if p > 0)
        med_ratio = ratios[len(ratios) // 2] if ratios else None
        hb = ('inconclusive (fewer than 12 learned pairs)' if len(inv) < 12 else
              'supported' if lower >= 10 and med_ratio <= 0.5 else 'not supported')
        out.update(h_a=dict(verdict=ha, mixed_h4=h4['mixed'], prefix_h4=h4['prefix'], closed_gap_over_prefix=closed,
                            varied_task_learned=vlearned),
                   h_b=dict(verdict=hb, lower=lower, pairs=len(inv), median_ratio=med_ratio,
                            ratios={n: m / p for n, p, m in inv if p > 0}))
        print(f'H_A: {ha}  (mixed passes H4 in {h4["mixed"]["supported"]} of {len(h4["mixed"]["judged"])} grammars, prefix in '
              f'{h4["prefix"]["supported"]} of {len(h4["prefix"]["judged"])}; mixed closes the gap over prefix in {closed}; '
              f'mixed learned the varied task in {vlearned})')
        print(f'H_B: {hb}  (invalid read-out mass lower in mixed than in prefix in {lower} of {len(inv)} learned pairs; '
              f'paired ratio mixed / prefix: median {f(med_ratio)}, range {f(ratios[0] if ratios else None)} to '
              f'{f(ratios[-1] if ratios else None)})')
    # counted over the arms analysed so far: the `varied` arm is analysed in a second step and may be censored for time
    n_runs, planned = sum(len(R[a]) for a in S), 12 * len(S)
    failed = [] if n_runs >= MIN_RUNS / 36 * planned else [f'only {n_runs} of {planned} runs analysed in arms {list(S)}']
    print(f'arms analysed: {list(S)}' + ('' if len(S) == len(ARMS) else f' (not yet or censored: {[a for a in ARMS if a not in S]})'))
    if len(sys.argv) > 3 and 'prefix' in S:
        old = {q['run']: q for q in json.load(open(sys.argv[3]))['runs']}
        print('\nprefix arm against rhm-ttn-messages (same script arguments and seeds): excess, m4_both, m3_invalid_q')
        rep = {}
        for n, q in sorted(R['prefix'].items()):
            if n in old:
                rep[n] = {k: (q[k], old[n][k]) for k in ('excess', 'm4_both', 'm3_invalid_q')}
                print(f'  {n:20s} ' + '  '.join(f'{k} {a:7.4f} / {b:7.4f}' for k, (a, b) in rep[n].items()))
                if abs(q['excess'] - old[n]['excess']) > REPRO_MAX:
                    failed.append(f'{n}: prefix excess {q["excess"]:.4f} against {old[n]["excess"]:.4f} in rhm-ttn-messages')
        out['prefix_vs_rhm_ttn_messages'] = rep
    out.update(n_runs=n_runs, run_failed=failed)
    print('run failed: ' + ('; '.join(failed) if failed else 'no'))
    json.dump(out, open(res / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
