"""Are the saved runs of the worker rerun and the coordinator run the same, bit for bit?

    PYTHONPATH=. uv run --frozen python studies/rhm/rhm-toy-rerun/compare_runs.py [out.json]

compare.py compares verdicts and which runs are exact. This goes one level down: for every run file present in both
runs/rhm-<experiment>/ (coordinator) and runs/rhm-toy-rerun/<experiment>/ (worker), it loads both and tests each saved
entry for exact equality (torch.equal on tensors, == on everything else, NaN equal to NaN). `seconds` is the run's
wall-clock time, so it is expected to differ and is reported separately as the per-run time on each machine.
It reads the coordinator's folders and writes only the file named on the command line."""
import json
import os
import statistics
import sys
import time

import torch

ROOT = 'runs'
TIMING = 'seconds'


def same(x, y):
    if torch.is_tensor(x):
        if not (torch.is_tensor(y) and x.shape == y.shape and x.dtype == y.dtype):
            return False
        if torch.equal(x, y):
            return True                            # torch.equal is False wherever there is a NaN, so compare those apart
        return x.is_floating_point() and torch.equal(x.isnan(), y.isnan()) and torch.equal(x.nan_to_num(), y.nan_to_num())
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
    if isinstance(x, float) and x != x:
        return isinstance(y, float) and y != y
    return x == y


def main():
    report = dict(compared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    for name in ('forced-synonyms', 'crossed-rank'):
        coord, worker = f'{ROOT}/rhm-{name}', f'{ROOT}/rhm-toy-rerun/{name}'
        fc, fw = ({f for f in os.listdir(d) if f.endswith('.pt')} for d in (coord, worker))
        both, unreadable, differing, keys, tc, tw = sorted(fc & fw), [], {}, set(), [], []
        for f in both:
            try:                                   # the coordinator may still be writing its newest file
                x, y = (torch.load(f'{d}/{f}', weights_only=False) for d in (coord, worker))
            except Exception as e:
                unreadable.append([f, repr(e)[:120]])
                continue
            keys |= set(x) | set(y)
            d = sorted(k for k in set(x) | set(y) if k != TIMING and not (k in x and k in y and same(x[k], y[k])))
            if d:
                differing[f] = d
            tc.append(float(x[TIMING])), tw.append(float(y[TIMING]))
        n = len(both) - len(unreadable)
        report[name] = dict(runs_coordinator=len(fc), runs_worker=len(fw), runs_compared=n, unreadable=unreadable,
                            entries_compared=sorted(keys - {TIMING}), runs_equal_in_every_entry=n - len(differing),
                            runs_with_a_differing_entry=differing,
                            seconds_per_run_coordinator=dict(median=statistics.median(tc), min=min(tc), max=max(tc)),
                            seconds_per_run_worker=dict(median=statistics.median(tw), min=min(tw), max=max(tw)))
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        json.dump(report, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
