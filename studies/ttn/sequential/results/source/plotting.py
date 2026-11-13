"""Plot helpers shared by the TTN experiments' plot.py scripts. Figures are PDFs; inputs are a results folder."""
import json
from pathlib import Path

import numpy as np

from ctn.style import C1, C2, C3, INK, MUTED, plt
from ctn.ttn import GROUPS, NODES

COLORS = [C1, C2, C3, INK]


def load(folder, name):
    z = np.load(Path(folder) / f'{name}.npz')
    meta = json.load(open(Path(folder) / f'{name}.json'))
    return {k: z[k] for k in z.files}, meta


def names(folder, prefix):
    return sorted(p.stem for p in Path(folder).glob(f'{prefix}*.json'))


def coefficient_panel(ax, log, meta, runs=None, t_from=0.0, title=None):
    """c_k/s_k against time, one color per component, thin line per run."""
    s = np.array([c['s'] for c in meta['comps']])
    runs = range(log['c'].shape[1]) if runs is None else runs
    for k in range(len(s)):
        for i, r in enumerate(runs):
            ax.plot(log['t'] - t_from, log['c'][:, r, k] / s[k], color=COLORS[k], lw=0.8, alpha=0.6,
                    label=f"$c_{k + 1}/s_{k + 1}$ (s={s[k]:g}, out {meta['comps'][k]['out'] + 1})" if i == 0 else None)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set(xlabel='time $\\eta\\,\\times$ step', ylabel='$c_k/s_k$', title=title or meta['name'])
    ax.legend(loc='lower right', fontsize=7)


def mobility_panel(ax, log, meta, i, j, runs=None, t_from=0.0):
    """Off-diagonal K_ij split by subtree group (median over runs), against time."""
    K = log['K'][:, :, i, j, :]  # (Td, R, G)
    K = K[:, list(runs)] if runs is not None else K
    for g, name in enumerate(GROUPS):
        ax.plot(log['t_K'] - t_from, np.median(K[..., g], 1), label=name, lw=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel='time', ylabel=f'$K_{{{i + 1}{j + 1}}}$ by subtree (median)', title=f"{meta['name']}: coupling by subtree")
    ax.legend(fontsize=7, ncol=2)


def alignment_panel(ax, log, meta, k, runs=None, t_from=0.0):
    """Fraction of component k's restricted feature in each node's activation span (median over runs)."""
    A = log['align'][:, :, k, :]
    A = A[:, list(runs)] if runs is not None else A
    for v, node in enumerate(NODES):
        ax.plot(log['t_detail'] - t_from, np.median(A[..., v], 1), label=node, lw=1.2)
    ax.set(xlabel='time', ylabel='alignment $R^2$ (median)', ylim=(-0.02, 1.02), title=f"{meta['name']}: component {k + 1} by node")
    ax.legend(fontsize=7, ncol=3)


def logy(ax, values, what='no run reached half its target'):
    """Log y-axis when there is positive data; otherwise say on the panel that nothing was reached (censored)."""
    v = np.asarray(values, float)
    if (np.isfinite(v) & (v > 0)).any():
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, what, transform=ax.transAxes, ha='center', color=MUTED)


def save(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder) / f'{name}.pdf')
    plt.close(fig)
