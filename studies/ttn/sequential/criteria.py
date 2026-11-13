"""Location-of-transfer criteria for sequential, from saved results only (analyze.py reports per-arm medians and the
family sign test but not these ratios). Usage: PYTHONPATH=. python studies/ttn/sequential/criteria.py <results>

For each (family, s2) and each frozen arm: the median over gate-passing seeds of the seed-paired ratio
t_half(c2, frozen arm) / t_half(c2, all), with censored half-times as +inf (ratio inf when only the frozen arm is
censored; seeds where both are censored drop out, as in studies.ttn.analysis.ratio_ci), a bootstrap 95% interval,
the ratio of the two arms' medians, and the fraction of the frozen arm that is censored.
Writes <results>/criteria.json."""
import json
import sys
from pathlib import Path

import numpy as np

from studies.ttn.analysis import fmt, inf_nan, load, ratio_ci

res = Path(sys.argv[1])
gate = np.array(load(res, 'phase1')[1]['gate_ok'], bool)
out = {}
for p in sorted(res.glob('*_s2*.json')):
    _, meta = load(res, p.stem)
    seeds, arms = meta['seeds'], meta['arms']
    h = np.array(meta['summary']['half'], float)[:, 1]
    H = {arm: inf_nan(h[a * seeds:(a + 1) * seeds])[gate] for a, arm in enumerate(arms)}
    for arm in ('freeze_q1', 'freeze_left'):
        r, ci = ratio_ci(H[arm], H['all'])
        ci = [c if np.isfinite(c) else np.inf for c in ci]  # percentiles over inf ratios come back nan: the bound is unbounded
        cens = float(np.isinf(H[arm]).mean())
        rom = float(np.median(H[arm]) / np.median(H['all']))
        out[f'{p.stem}_{arm}_over_all'] = dict(paired_median_ratio=r, ci=ci, ratio_of_medians=rom,
                                               censored_frac=cens, censored_all=float(np.isinf(H['all']).mean()),
                                               n=int(gate.sum()))
        print(f'{p.stem:16s} {arm:12s}/all  paired median {fmt(r, ci)}  ratio of medians {rom:.4g}  '
              f'censored {cens:.2f} (all arm {np.isinf(H["all"]).mean():.2f}), n={gate.sum()}')
json.dump(out, open(res / 'criteria.json', 'w'), indent=1, default=float)
