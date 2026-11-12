"""rhm-controls: per-arm numbers the write-up quotes that analysis.json does not table directly.

  uv run --frozen python studies/rhm/rhm-controls/summary.py runs/rhm-controls studies/rhm/rhm-controls/results

Per (task, arm), over the 6 runs (grammars 0-2 x init seeds 0, 1): median [min, max] of
  excess_test_err   test error rate minus the Bayes-optimal error rate on the same 50,000 test trees
  cum_lr_to_thresh  sum of the LR over the updates before the threshold (0.9 x Bayes diagnostic accuracy), i.e. the
                    gradient-flow time t = sum_k eta_k; +inf if never reached
  diag_acc_3000     diagnostic accuracy at update 3,000
Written after the run by the supervising session (not part of the approved commands); reads only saved results.
"""
import json
import sys
from pathlib import Path

import numpy as np

from studies.rhm.analysis import cum_lr, t_thresh


def mr(v):
    return dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)), n=len(v))


def main(runs, out):
    res = {}
    for t in ('root', 'last'):
        for arm in ('tree', 'perm', 'tied', 'chain'):
            ex, cl, a3 = [], [], []
            for g in range(3):
                for s in range(2):
                    r = json.load(open(Path(runs) / f'{t}_{arm}_g{g}_s{s}' / 'result.json'))
                    ex.append(r['bayes_test']['acc'] - r['test']['acc'])
                    cl.append(cum_lr(r, t_thresh(r)))
                    a3.append(r['log']['diag_acc'][r['log']['step'].index(3000)])
            res[f'{t}_{arm}'] = dict(lr=r['lr'], excess_test_err=mr(ex), cum_lr_to_thresh=mr(cl), diag_acc_3000=mr(a3))
    json.dump(res, open(Path(out) / 'summary.json', 'w'), indent=1)
    for k, v in res.items():
        print(k, {m: round(v[m]['median'], 4) if isinstance(v[m], dict) else v[m] for m in v})


if __name__ == '__main__':
    main(*sys.argv[1:3])
