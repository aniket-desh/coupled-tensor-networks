"""rhm-transfer figures from <results>/curves.json and analysis.json.

  PYTHONPATH=. python studies/rhm/rhm-transfer/plot.py studies/rhm/rhm-transfer/results studies/rhm/rhm-transfer/figures

phase2.pdf: unseen-production accuracy (left) and seen-production accuracy (right) vs phase-2 updates per arm, one
line per unit; color = where the frozen part comes from (or scratch / finetune / oracle), solid = C variant, dashed =
WC variant; dotted line = threshold. pretrain.pdf: seen and unseen accuracy during pretraining, and the other-grammar
pretraining's accuracy on its own grammar."""
import json
import sys
from pathlib import Path

from studies.rhm.plotting import CATEGORICAL, INK2, plt, save, steps_x

FAMILY = {'scratch': 0, 'finetune': 1, 'learned': 2, 'other': 3, 'random': 4, 'oracle': 5}


def style(arm):
    fam = next(k for k in FAMILY if k in arm)
    return dict(color=CATEGORICAL[FAMILY[fam]], ls='--' if arm.endswith('_WC') else '-')


def main():
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    curves = json.load(open(res / 'curves.json'))
    thr = json.load(open(res / 'analysis.json'))['threshold']
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    seen = set()
    for k, c in sorted(curves.items()):
        unit, arm = k.split('/')
        if arm.startswith('pretrain'):
            continue
        for ax, key in zip(axes, ('unseen_acc', 'seen_acc')):
            ax.plot(steps_x(c['step']), c[key], lw=0.9, alpha=0.8, label=arm if (arm not in seen and key == 'unseen_acc') else None,
                    **style(arm))
        seen.add(arm)
    for ax, t in zip(axes, ('unseen root productions (trained on)', 'seen root productions (pretraining set)')):
        ax.set_xscale('log')
        ax.axhline(thr, color=INK2, ls=':', lw=0.8)
        ax.set_title(t)
        ax.set_xlabel('phase-2 updates')
    axes[0].set_ylabel('diagnostic accuracy')
    if seen:
        axes[0].legend(fontsize=5)
    save(fig, figs / 'phase2.pdf')
    fig, ax = plt.subplots(figsize=(4, 3))
    for k, c in sorted(curves.items()):
        if k.endswith('/pretrain'):
            ax.plot(steps_x(c['step']), c['seen_acc'], color=CATEGORICAL[0], lw=0.9)
            ax.plot(steps_x(c['step']), c['unseen_acc'], color=CATEGORICAL[1], lw=0.9)
        elif k.endswith('/pretrain_other'):
            ax.plot(steps_x(c['step']), c['own_acc'], color=CATEGORICAL[3], lw=0.9, ls='--')
    ax.plot([], [], color=CATEGORICAL[0], label='seen productions')
    ax.plot([], [], color=CATEGORICAL[1], label='unseen productions (held out)')
    ax.plot([], [], color=CATEGORICAL[3], ls='--', label='other grammar, its own trees')
    ax.set_xscale('log')
    ax.set_xlabel('pretraining updates')
    ax.set_ylabel('diagnostic accuracy')
    ax.legend(fontsize=7)
    save(fig, figs / 'pretrain.pdf')


if __name__ == '__main__':
    main()
