"""Do the worker rerun and the coordinator run of the two toy-grammar experiments agree?

    python studies/rhm/rhm-toy-rerun/compare.py [out.json]

Run on the coordinator once both exist. Compares, per experiment, the verdict strings of summary.json and, run by run
(matched by name), whether each run is exact (excess loss below 1e-10 nats) in both. The two machines run the same
code, seeds and float64 arithmetic but different CPUs and thread scheduling, so trajectories can differ in the last
digits and a run near a saddle can end elsewhere: agreement is expected on verdicts and cell counts, not on every run."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = {'forced-synonyms': HERE.parent / 'rhm-forced-synonyms' / 'results', 'crossed-rank': HERE.parent / 'rhm-crossed-rank' / 'results'}
EXACT = 1e-10


def verdicts(summary):
    v = summary.get('verdicts', summary)
    out = {k: x for k, x in v.items() if k.startswith('H') and 'detail' not in k}
    for g, t in (v.get('trainability at the bound (pre-stated reading)') or {}).items():
        out[f'trainability on {g}'] = t['reading']
    out['smallest_exact_width'] = v.get('smallest_exact_width')
    return out


def main():
    report = {}
    for name, local in PAIRS.items():
        worker = HERE / 'results' / name
        if not (local / 'summary.json').exists() or not (worker / 'summary.json').exists():
            report[name] = 'missing: ' + ', '.join(str(p) for p in (local, worker) if not (p / 'summary.json').exists())
            continue
        va, vb = (verdicts(json.load(open(p / 'summary.json'))) for p in (local, worker))
        ra, rb = ({q['run']: q['kl'] for q in json.load(open(p / 'runs.json'))} for p in (local, worker))
        both = sorted(set(ra) & set(rb))
        flips = [(r, ra[r], rb[r]) for r in both if (ra[r] < EXACT) != (rb[r] < EXACT)]
        report[name] = dict(verdicts_agree=va == vb, differing_verdicts={k: [va.get(k), vb.get(k)] for k in va if va.get(k) != vb.get(k)},
                            runs_coordinator=len(ra), runs_worker=len(rb), runs_in_both=len(both),
                            exact_coordinator=sum(ra[r] < EXACT for r in both), exact_worker=sum(rb[r] < EXACT for r in both),
                            runs_exact_on_one_machine_only=flips)
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        json.dump(report, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
