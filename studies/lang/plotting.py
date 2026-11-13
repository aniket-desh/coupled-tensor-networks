"""Small plotting helpers shared by the language-process experiments' plot.py scripts (they read only saved results)."""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ['#2a78d6', '#d65f2a', '#3a9a4a', '#9a3ab0', '#b0a030', '#30a0b0', '#666666', '#c0306a']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def load(res):
    res = Path(res)
    rd = lambda n: json.load(open(res / n)) if (res / n).exists() else {}
    return rd('analysis.json'), rd('curves.json'), rd('interp.json')


def num(x):
    if isinstance(x, str):
        return float(x)
    return float('nan') if x is None else x


def curve(ax, c, key='diag_kl', **kw):
    y = [num(v) for v in c[key]]
    s = [max(v, 1) for v in c['step']]
    ax.plot(s, y, **kw)


def t_points(ax, xs, ts, cap, **kw):
    """Steps to threshold; censored runs (inf) drawn as open markers at the cap."""
    for x, t in zip(xs, ts):
        t = num(t)
        if math.isfinite(t):
            ax.plot(x, max(t, 1), 'o', ms=4, **kw)
        else:
            ax.plot(x, cap, 'o', ms=5, mfc='none', **{k: v for k, v in kw.items() if k != 'mfc'})


def save(fig, out, name):
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(out) / f'{name}.pdf')
    plt.close(fig)
    print('wrote', Path(out) / f'{name}.pdf')


def traj(ax, tr, name, summary, levels, label_fmt='level {}', colors=COLORS, **kw):
    d = (tr.get(name) or {}).get(summary) or {}
    for i, L in enumerate(levels):
        x = d.get(str(L))
        if not x:
            continue
        st = sorted(x, key=int)
        ax.plot([max(int(s), 1) for s in st], [num(x[s]) for s in st], color=colors[i % len(colors)],
                label=label_fmt.format(L), **kw)


def legend(ax, **kw):
    """Legend with one entry per label."""
    h, l = ax.get_legend_handles_labels()
    d = dict(zip(l, h))
    if d:
        ax.legend(list(d.values()), list(d.keys()), **kw)
