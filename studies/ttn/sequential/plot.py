"""Figures for sequential. Usage: PYTHONPATH=. python studies/ttn/sequential/plot.py <results> <figures>"""
import sys
from pathlib import Path

import numpy as np

from ctn.style import plt
from studies.ttn.plotting import COLORS, load, mobility_panel, save

res, fig_dir = Path(sys.argv[1]), sys.argv[2]
tags = sorted(p.stem for p in res.glob('*_s2*.json'))
fig, ax = plt.subplots(len(tags), 3, figsize=(12, 3.2 * len(tags)), constrained_layout=True, squeeze=False)
for i, tag in enumerate(tags):
    log, meta = load(res, tag)
    S, arms, t1 = meta['seeds'], meta['arms'], meta['phase1_t_end']
    gate = np.array(meta['phase1_gate_ok'], bool)
    s = np.array([c['s'] for c in meta['comps']])
    tt = np.maximum(log['t'] - t1, 1e-3)
    for a, arm in enumerate(arms):
        for j in range(S):
            if arm != 'fresh' and not gate[j]:
                continue
            r = a * S + j
            ax[i, 0].plot(tt, log['c'][:, r, 1] / s[1], color=COLORS[a], lw=0.7, alpha=0.6, label=arm if j == 0 else None)
            if arm != 'fresh':
                ax[i, 1].plot(tt, log['c'][:, r, 0] / s[0] - 1, color=COLORS[a], lw=0.7, alpha=0.6, label=arm if j == 0 else None)
    ax[i, 0].set(xscale='log', xlabel='time since output 2 was added', ylabel='$c_2/s_2$', title=f'{tag}: learning the new output')
    ax[i, 1].set(xscale='log', xlabel='time since output 2 was added', ylabel='$c_1/s_1 - 1$', title=f'{tag}: change in the learned output')
    ax[i, 0].legend(fontsize=7)
    ax[i, 1].legend(fontsize=7)
    mobility_panel(ax[i, 2], log, meta, 0, 1, runs=[j for j in range(S) if gate[j]], t_from=t1)
    ax[i, 2].set_title(f'{tag}: $K_{{12}}$ by subtree, arm "all"')
save(fig, fig_dir, 'sequential')
