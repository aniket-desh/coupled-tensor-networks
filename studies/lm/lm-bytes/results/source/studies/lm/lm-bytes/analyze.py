"""Summarise lm-bytes results: bits/byte per model (median and range over seeds), params, targets seen, and the
comparisons in EXPERIMENT.md. Writes <results>/analysis.json and prints a table.
Usage: PYTHONPATH=. python studies/lm/lm-bytes/analyze.py <results>

TTN and chain runs use the identity init (ctn.tree.identity_init) except the 'ttn_var' group (variance init).
Headline numbers are test bits/byte on every target of test.bin (validation is reported alongside). A comparison
A vs B uses the done seeds of each (an n-gram is a single deterministic value):
  'A better'   every A seed is below every B seed and the medians differ by at least MARGIN
  'B better'   the reverse
  'equal'      the medians differ by less than MARGIN and neither group's seed range exceeds MARGIN
  'unresolved' otherwise (ranges overlap, or a seed range wider than MARGIN makes 'equal' unsafe)
  'missing'    one side has no finished run
Also per run: the first curve step at which validation bits/byte drops below the KN unigram (100 MB counts; 4.42 if
the n-gram is missing), so a slow start can be told from a stall, and per group the median validation bits/byte at
fixed fractions of training (500, 1k, 2k, 5k, 10k updates of a 40k run).
Runs that were skipped (deadline), diverged or failed are counted per group and never imputed."""
import json
import re
import sys
from pathlib import Path

import numpy as np

MARGIN = 0.02  # bits/byte: a practical threshold set before the run (~1.5% of ~1.3 bits/byte); seed spread is not
# known in advance, so 'equal' also requires both seed ranges to be within MARGIN
UNIGRAM_FALLBACK = 4.42  # KN unigram validation bits/byte measured on the pilot set, used if the n-gram run is missing
EARLY_FRACS = [0.0125, 0.025, 0.05, 0.125, 0.25]  # 500, 1k, 2k, 5k, 10k updates of a 40k run
GROUP_LABEL = {
    'ttn_r32': 'TTN untied r=32', 'ttn_r16': 'TTN untied r=16', 'ttn_r64': 'TTN untied r=64',
    'ttn_r128': 'TTN untied r=128', 'ttn_var_r32': 'TTN untied r=32, variance init',
    'ttn_frozenW_r32': 'TTN r=32, fusion cores frozen', 'ttn_r32_long': 'TTN untied r=32, 160k updates', 'ttn_tied_r32': 'TTN level-tied r=32',
    'mlp_h192': 'MLP 192 hidden', 'chain_r93': 'chain r=93 (matched)', 'chain_r32': 'chain r=32 (small)',
    'transformer_d128': 'transformer d=128, 4 layers',
}


def load(res):
    runs = {}
    for f in sorted(Path(res, 'runs').glob('*.json')):
        r = json.load(open(f))
        runs[r['cfg']['name']] = r
    return runs


def group_of(name):
    return re.sub(r'_s\d+$', '', name)


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v), values=v)


def compare(a_name, a, b_name, b):
    if not a or not b:
        return dict(a=a_name, b=b_name, verdict='missing')
    d = a['median'] - b['median']
    if a['max'] < b['min'] and d <= -MARGIN:
        v = f'{a_name} better'
    elif b['max'] < a['min'] and d >= MARGIN:
        v = f'{b_name} better'
    elif abs(d) < MARGIN and a['max'] - a['min'] <= MARGIN and b['max'] - b['min'] <= MARGIN:
        v = 'equal'
    else:
        v = 'unresolved'
    return dict(a=a_name, b=b_name, a_median=a['median'], b_median=b['median'], diff=d, verdict=v)


def point(x):
    return dict(median=x, min=x, max=x, n=1, values=[x]) if x is not None else None


def first_below(curve, thr):
    """First curve step with validation bits/byte below thr, or None."""
    return next((st for st, v in zip(curve['step'], curve['val_bits']) if v < thr), None)


def early_points(rs):
    """Median validation bits/byte over seeds at the first curve step >= each EARLY_FRACS fraction of training."""
    out = {}
    for f in EARLY_FRACS:
        vals, st = [], None
        for r in rs:
            c = r.get('curve', {})
            i = next((k for k, s_ in enumerate(c.get('step', [])) if s_ >= f * r['cfg']['steps']), None)
            if i is not None:
                vals.append(c['val_bits'][i])
                st = c['step'][i]
        if vals and st is not None:
            out[str(st)] = float(np.median(vals))
    return out


def main(res):
    runs = load(res)
    ng_big = next((runs[k] for k in ('ngram_100M', 'ngram_large') if k in runs and runs[k]['status'] == 'done'), None)
    unigram_val = ng_big['val_bits'][0] if ng_big else UNIGRAM_FALLBACK
    below = {n: first_below(r['curve'], unigram_val) for n, r in runs.items() if r.get('curve')}
    groups = {}
    for n, r in runs.items():
        if n.startswith('pilot_') or r['cfg'].get('family') == 'ngram':
            continue
        groups.setdefault(group_of(n), []).append(r)
    table = {}
    for g, rs in sorted(groups.items()):
        done = [r for r in rs if r['status'] == 'done']
        table[g] = dict(
            label=GROUP_LABEL.get(g, g), family=rs[0]['cfg']['family'], model=rs[0]['cfg']['model'],
            init=rs[0]['cfg']['model'].get('init', '-'),
            params=rs[0].get('params'), lr=rs[0]['cfg']['lr'], steps=rs[0]['cfg']['steps'],
            targets_seen=rs[0]['cfg']['steps'] * rs[0]['cfg']['batch'],
            status={s: sum(r['status'] == s for r in rs) for s in sorted({r['status'] for r in rs})},
            seeds_done=sorted(r['cfg']['seed'] for r in done),
            test=stats([r['test_bits'] for r in done]), val=stats([r['val_bits'] for r in done]),
            train=stats([r['train_bits'] for r in done]),
            gap_val_minus_train=stats([r['val_bits'] - r['train_bits'] for r in done]),
            ms_per_step=stats([r.get('ms_per_step') for r in rs]),
            clip_frac_last=stats([r['curve']['clip_frac'][-1] for r in done if r['curve']['clip_frac']]),
            trainable_params=rs[0].get('trainable_params'),
            first_step_below_unigram={r['cfg']['name']: below.get(r['cfg']['name']) for r in rs},
            early_val_median=early_points([r for r in rs if r.get('curve')]),
            n_test=done[0]['n_test'] if done else None, device=rs[0].get('device'))
    ngrams = {n: r for n, r in runs.items() if r['cfg'].get('family') == 'ngram'}
    ng = {}
    for n, r in ngrams.items():
        if r['status'] != 'done':
            ng[n] = dict(status=r['status'])
            continue
        ng[n] = dict(status='done', train_bytes=r['cfg']['train_bytes'], orders=r['orders'], test=r['test_bits'],
                     val=r['val_bits'], train=r['train_bits'], distinct=r['distinct'], wall_s=r['wall_s'])
    big = next((v for k, v in ng.items() if v.get('status') == 'done' and k in ('ngram_100M', 'ngram_large')), None)
    small = next((v for k, v in ng.items() if v.get('status') == 'done' and k in ('ngram_10M', 'ngram_small')), None)

    def best_ngram(src, max_order):
        if src is None:
            return None, None
        i = int(np.argmin(src['test'][:max_order]))
        return src['test'][i], i + 1

    T = lambda g: table.get(g, {}).get('test')
    ttn = T('ttn_r32')
    b4, o4 = best_ngram(big, 4)
    b8, o8 = best_ngram(big, 8)
    s8, so8 = best_ngram(small, 8)
    comps = {
        'ttn_vs_ngram_le4': compare('TTN r=32', ttn, f'KN order {o4} (100M)', point(b4)),
        'ttn_vs_ngram_le8': compare('TTN r=32', ttn, f'KN order {o8} (100M)', point(b8)),
        'ttn_vs_ngram_le8_10M': compare('TTN r=32', ttn, f'KN order {so8} (10.24M bytes)', point(s8)),
        'ttn_vs_mlp': compare('TTN r=32', ttn, 'MLP', T('mlp_h192')),
        'ttn_vs_matched_chain': compare('TTN r=32', ttn, 'chain r=93', T('chain_r93')),
        'untied_vs_tied': compare('TTN untied r=32', ttn, 'TTN tied r=32', T('ttn_tied_r32')),
        'ttn_vs_transformer': compare('TTN r=32', ttn, 'transformer', T('transformer_d128')),
        'identity_vs_variance_init': compare('TTN identity init', ttn, 'TTN variance init', T('ttn_var_r32')),
        'ttn_vs_frozenW': compare('TTN r=32', ttn, 'frozen-fusion TTN r=32', T('ttn_frozenW_r32')),
        'chain_matched_vs_small': compare('chain r=93', T('chain_r93'), 'chain r=32', T('chain_r32')),
    }
    # Where the TTN sits on the n-gram ladder: the highest order it beats (by median).
    ladder = None
    if ttn and big:
        beaten = [o for o, b in zip(big['orders'], big['test']) if ttn['median'] < b]
        ladder = dict(highest_order_beaten=max(beaten) if beaten else 0, ngram_test=big['test'])
    # Is 40k converged? The 160k run against the range of the 40k seeds: 'not converged' if it is more than MARGIN
    # below the best 40k seed, 'converged' if it is within MARGIN of (or above) the best 40k seed.
    long_r = runs.get('ttn_r32_long_s0')
    conv = dict(verdict='missing')
    if long_r and long_r['status'] == 'done' and ttn:
        d = long_r['test_bits'] - ttn['min']
        conv = dict(test_40k_median=ttn['median'], test_40k_range=[ttn['min'], ttn['max']], n_40k=ttn['n'],
                    test_160k=long_r['test_bits'], diff_vs_best_40k=d,
                    verdict='not converged at 40k (160k beats the best 40k seed by >= margin)' if d <= -MARGIN
                    else 'converged at 40k (160k within margin of the best 40k seed)')
    # Plateau: the TTN starts near uniform (8 bits/byte) and must first learn the byte marginal, a degree-16 feature.
    # It has left the plateau if its median beats the KN unigram.
    plateau = None
    if ttn and big:
        plateau = dict(unigram_test=big['test'][0], ttn_median=ttn['median'], left_plateau=ttn['median'] < big['test'][0])
        if not plateau['left_plateau'] and 'test_160k' in conv:
            conv['verdict'] += ' -- but the TTN has not left the plateau, so this says nothing about capacity'
    scaling = {r: (T(f'ttn_r{r}') or {}).get('median') for r in (16, 32, 64, 128)}

    # Hypothesis verdicts (EXPERIMENT.md "Success and failure criteria").
    main_done = len(table.get('ttn_r32', {}).get('seeds_done', []))
    if main_done < 2 or big is None:
        h1 = 'inconclusive'
    elif ttn['max'] < b4:
        h1 = 'supported'
    elif ttn['median'] > b4:
        h1 = 'not supported'
    else:
        h1 = 'partly supported'
    lrs = json.load(open(Path(res, 'lr_choice.json'))) if Path(res, 'lr_choice.json').exists() else {}
    pilots = {n: dict(status=r['status'], val_bits=r.get('val_bits'), lr=r['cfg']['lr'], family=r['cfg']['family'])
              for n, r in runs.items() if n.startswith('pilot_')}
    censored = {g: {k: v for k, v in t['status'].items() if k != 'done'} for g, t in table.items()
                if set(t['status']) - {'done'}}
    out = dict(margin=MARGIN, groups=table, ngrams=ng, comparisons=comps, ngram_ladder=ladder, convergence=conv, plateau=plateau,
               unigram_val_threshold=unigram_val,
               ttn_scaling_test_median=scaling, H1_ttn_beats_order_le4=h1, lr_choice=lrs, pilots=pilots,
               censored=censored, run_failed=main_done == 0)
    json.dump(out, open(Path(res, 'analysis.json'), 'w'), indent=1, default=float)

    f = lambda s: f"{s['median']:.4f} [{s['min']:.4f}, {s['max']:.4f}] n={s['n']}" if s else '-'
    print(f"{'model':32s} {'init':>8s} {'params':>10s} {'targets':>10s} {'lr':>7s} {'test bits/byte (median [range])':>36s} "
          f"{'val':>30s} {'val-train':>8s} {'ms/step':>8s}  status")
    for g, t in table.items():
        gap = t['gap_val_minus_train']['median'] if t['gap_val_minus_train'] else float('nan')
        ms = t['ms_per_step']['median'] if t['ms_per_step'] else float('nan')
        print(f"{t['label']:32s} {t['init']:>8s} {t['params'] or 0:>10,d} {t['targets_seen']:>10,d} {t['lr']:>7g} {f(t['test']):>36s} "
              f"{f(t['val']):>30s} {gap:>8.4f} {ms:>8.2f}  {t['status']}")
    for n, v in ng.items():
        if v['status'] == 'done':
            print(f"{n} ({v['train_bytes']:,} train bytes) test by order 1..{len(v['test'])}: "
                  + ' '.join(f'{x:.3f}' for x in v['test']))
        else:
            print(f'{n}: {v["status"]}')
    print(f"\nH1 (TTN r=32 beats KN orders <= 4): {h1}")
    for k, c in comps.items():
        extra = f" ({c['a_median']:.4f} vs {c['b_median']:.4f}, diff {c['diff']:+.4f})" if 'diff' in c else ''
        print(f'{k:24s} {c["verdict"]}{extra}')
    print(f'n-gram ladder: {ladder["highest_order_beaten"] if ladder else "-"} = highest KN order (100M) the TTN r=32 '
          f'median beats')
    print(f"convergence: {conv['verdict']}" + (
        f" (40k median {conv['test_40k_median']:.4f} range [{conv['test_40k_range'][0]:.4f}, "
        f"{conv['test_40k_range'][1]:.4f}], 160k {conv['test_160k']:.4f})" if 'test_160k' in conv else ''))
    print(f'\nfirst curve step with validation below the KN unigram ({unigram_val:.3f}); None = never:')
    for g, t in table.items():
        print(f"  {t['label']:32s} {list(t['first_step_below_unigram'].values())}   early val (step: median) "
              + ', '.join(f'{k}: {v:.3f}' for k, v in t['early_val_median'].items()))
    if plateau:
        print(f"plateau: TTN r=32 median {plateau['ttn_median']:.4f} vs KN unigram {plateau['unigram_test']:.4f} -> "
              f"{'left the plateau' if plateau['left_plateau'] else 'STILL ON THE PLATEAU'}")
    print('TTN test median by r: ' + ', '.join(f'r={r}: {v:.4f}' if v else f'r={r}: -' for r, v in scaling.items()))
    print('LR choice: ' + ', '.join(f"{k} {v['lr']:g}" for k, v in lrs.items()))
    if censored:
        print(f'not done (skipped/diverged/failed): {censored}')


if __name__ == '__main__':
    main(sys.argv[1])
