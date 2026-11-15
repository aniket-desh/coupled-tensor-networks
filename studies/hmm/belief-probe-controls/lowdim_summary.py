"""Apply the rules of EXPERIMENT.md to results/lowdim/<family>/<run>.json; print the table and write
lowdim_summary.json. plot.py draws the figures from the same files.

    PYTHONPATH=. python studies/hmm/belief-probe-controls/lowdim_summary.py [RESULTS_SUBDIR]   # default: lowdim

Per run and position group, with R2 pooled over positions as 1 - sum SSE / sum SST:
  rev      reverse direction, over all eligible positions: the share of the state's variance (its top principal scores
           holding 90% of it) that is an affine function of the belief on held-out prefixes; rev_init for the initial
           network. full_all is the experiment's forward probe (all directions) on the same positions.
  k90(t)   the fewest principal directions with which the belief itself gives the belief at R2 >= 0.9 (the `oracle`
           curve): the dimension the belief needs at that position.
  judged   positions where k90(t) is at most half of the state's varying directions, so that "use only the top k90
           directions" really restricts the probe.
  G        the model's forward R2 from its top k90(t) principal directions, pooled over the judged positions; the same
           for the initial network (G_init) and the one-hot tokens (G_tok). G2 is the model at 2 k90(t), `full` its
           probe with all directions on the judged positions.
  scr      the scrambled target from the model's top k90(t) directions, next to the oracle's value there.
Rules, per run:
  dominant  rev >= 0.8, rev halves the unexplained variance of rev_init (1 - rev <= 0.5 (1 - rev_init)), and
            full_all >= 0.8: the belief is linearly readable and most of the state's variance is a function of it.
  compact   G >= 0.8 and G halves the unexplained variance of G_init and of G_tok: the belief is readable from as many
            top directions as it needs.
A cell (same grammar or process, model and width; three seeds, all learned by the experiment's own gate) gets each
label only if all three seeds do. `compact` fails for a state that is a stretched or duplicated image of the belief;
`dominant` does not, which is why it is the primary label."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
G_MIN, RATIO, ORACLE_MIN = 0.8, 0.5, 0.9
SUMMARIES = {'rhm-transformer': HERE.parents[1] / 'rhm' / 'rhm-belief-geometry' / 'results' / 'summary.json',
             'rhm-tree': HERE.parents[1] / 'rhm' / 'rhm-ttn-belief-geometry' / 'results' / 'summary.json',
             'hmm': HERE.parent / 'hmm-belief-geometry' / 'results' / 'summary.json'}
r2 = lambda q: 1 - q[0] / q[1] if q[1] > 0 else float('nan')


def pooled(curves, ts, label, target, k_of):
    """1 - sum SSE / sum SST over positions ts of set `label` at k = k_of(t) (the largest grid k not above it)."""
    sse = sst = 0.0
    for t in ts:
        c = {int(k): v for k, v in curves[t][label][target].items()}
        k = max([q for q in c if q <= k_of(t)], default=min(c))
        sse, sst = sse + c[k][0], sst + c[k][1]
    return 1 - sse / sst if sst > 0 else None


def analyse(d):
    out = {}
    for grp, gi in d['groups'].items():
        ts = [str(t) for t in gi['positions']]
        if not ts:
            out[grp] = dict(judged=[], positions=ts)
            continue
        rev = lambda label: 1 - sum(d['curves'][t][label]['rev'][1] for t in ts) / sum(d['curves'][t][label]['rev'][2] for t in ts)
        per_t = [1 - d['curves'][t]['model']['rev'][1] / d['curves'][t]['model']['rev'][2] for t in ts]
        q = out[grp] = dict(positions=ts, rev=rev('model'), rev_init=rev('init'), rev_mean=sum(per_t) / len(per_t),
                            full_all=pooled(d['curves'], ts, 'model', 'belief', lambda t: 10 ** 9))
        q['dominant'] = bool(q['rev'] >= G_MIN and 1 - q['rev'] <= RATIO * (1 - max(q['rev_init'], 0)) and q['full_all'] >= G_MIN)
        k90, pmax = {}, {}
        for t in ts:
            c = {int(k): r2(v) for k, v in d['curves'][t]['oracle']['belief'].items()}
            k90[t] = next((k for k in sorted(c) if c[k] >= ORACLE_MIN), None)
            pmax[t] = max(int(k) for k in d['curves'][t]['model']['belief'])
        judged = q['judged'] = [t for t in ts if k90[t] is not None and 2 * k90[t] <= pmax[t]]
        if not judged:
            continue
        P = lambda label, target, mult=1: pooled(d['curves'], judged, label, target, lambda t: mult * k90[t])
        q.update(k90=[k90[t] for t in judged], G=P('model', 'belief'), G2=P('model', 'belief', 2),
                 G_init=P('init', 'belief'), G_tok=P('tokens', 'belief'), oracle=P('oracle', 'belief'),
                 full=pooled(d['curves'], judged, 'model', 'belief', lambda t: 10 ** 9),
                 scr=P('model', 'scr'), scr_oracle=P('oracle', 'scr'),
                 scr_full=pooled(d['curves'], judged, 'model', 'scr', lambda t: 10 ** 9))
        q['compact'] = bool(q['G'] >= G_MIN and all(1 - q['G'] <= RATIO * (1 - max(c, 0)) for c in (q['G_init'], q['G_tok'])))
    return out


def load(sub):
    """{family: (runs {name: saved dict}, learned {name: bool})} for results/<sub>."""
    out = {}
    for fam_dir in sorted(p for p in (HERE / 'results' / sub).iterdir() if p.is_dir()):
        learned = {q['run']: q.get('learned', True) for q in json.load(open(SUMMARIES[fam_dir.name]))['runs']}
        out[fam_dir.name] = ({p.stem: json.load(open(p)) for p in sorted(fam_dir.glob('*.json'))}, learned)
    return out


def summarize(sub, quiet=False):
    """({family: {cell|group: dict(runs, dominant, compact)}}, largest reproduction difference per family)."""
    say = (lambda *a: None) if quiet else print
    f = lambda v: '  -  ' if v is None else f'{v:5.2f}'
    summary, repro = {}, {}
    for fam, (runs, learned) in load(sub).items():
        ana = {n: analyse(d) for n, d in runs.items()}
        cells = defaultdict(list)
        for n in runs:
            cells[re.sub(r'_s\d+$', '', n)].append(n)
        say(f'\n== {fam}: {len(runs)} runs')
        # reproduction: the full probe (all directions) per position against the experiment's saved number
        diffs = [abs(r2(d['curves'][t]['model']['belief'][max(d['curves'][t]['model']['belief'], key=int)]) - v)
                 for d in runs.values() if not d.get('smoke') for t, v in d['saved_full'].items()]
        if diffs:
            repro[fam] = max(diffs)
            say(f"reproduction of the experiment's full probe: largest |difference| in R2 over {len(diffs)} positions = "
                f'{max(diffs):.2e}' + ('  <-- above 0.01: CHECK' if max(diffs) > 0.01 else ''))
        say(f'{"run":24s} group  rev (mean) rev_init full_all dominant | judged   G     G2   full  init   tok  oracle | scr: model '
            f'oracle full | compact')
        fam_out = {}
        for cell, names in sorted(cells.items()):
            for grp in sorted({g for n in names for g in ana[n]}):
                dom, com, all_learned = [], [], all(learned.get(n, True) for n in names)
                for n in sorted(names):
                    q = ana[n].get(grp)
                    if not q or not q['positions']:
                        say(f'{n:24s} {grp:5s} no eligible position')
                        continue
                    ok = learned.get(n, True)
                    dom.append(q['dominant'] and ok)
                    line = (f'{n:24s} {grp:5s} {f(q["rev"])} ({f(q["rev_mean"])}) {f(q["rev_init"])}    {f(q["full_all"])}   '
                            f'{"yes" if q["dominant"] else "no ":3s}    | ')
                    if q['judged']:
                        com.append(q['compact'] and ok)
                        line += (f'{len(q["judged"]):2d}/{len(q["positions"]):<2d} {f(q["G"])} {f(q["G2"])} {f(q["full"])} {f(q["G_init"])} '
                                 f'{f(q["G_tok"])} {f(q["oracle"])}  |      {f(q["scr"])} {f(q["scr_oracle"])} {f(q["scr_full"])} | '
                                 f'{"yes" if q["compact"] else "no"}')
                    else:
                        line += f'no judged position of {len(q["positions"])}'
                    say(line + ('' if ok else '  (not learned)'))
                label = lambda v: None if len(v) < 3 else bool(all(v))
                fam_out[f'{cell}|{grp}'] = dict(runs={n: ana[n].get(grp) for n in names}, dominant=label(dom), compact=label(com),
                                                learned=all_learned)
                txt = lambda v: 'not judged (fewer than 3 seeds)' if v is None else ('yes' if v else 'no')
                say(f'  -> {cell} {grp}: dominant {txt(label(dom))}; compact {txt(label(com))}')
        summary[fam] = fam_out
        L = [v for v in fam_out.values() if v['learned']]  # counts are among learned cells
        say(f'{fam}, learned cells: dominant in {sum(bool(v["dominant"]) for v in L)} of {sum(v["dominant"] is not None for v in L)} '
            f'judged (cell, group) pairs; compact in {sum(bool(v["compact"]) for v in L)} of '
            f'{sum(v["compact"] is not None for v in L)}')
    return summary, repro


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else 'lowdim'
    summary, repro = summarize(sub)
    json.dump(dict(families=summary, reproduction=repro), open(HERE / 'results' / f'{sub}_summary.json', 'w'), indent=1)
    if any(v > 0.01 for v in repro.values()):
        raise SystemExit('reproduction difference above 0.01: do not interpret these numbers')


if __name__ == '__main__':
    main()
