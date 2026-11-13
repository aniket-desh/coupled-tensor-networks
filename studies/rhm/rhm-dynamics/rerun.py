"""Amendment (supervisor, 2026-11-13): the r = 64 runs at the shared peak LR 3e-3 against the rerun at 1e-3.

At 3e-3, 3 of the 6 r = 64 runs learned and then collapsed to the uniform output (diagnostic loss log 16) between
updates 3,000 and 10,000 and ended there; a fourth (root, grammar 1) touched log 16 at update 9,000 and recovered.
Variance-init runs have 13 interpretation checkpoints because they stop at update 5,000. The rerun (`run.py --only r64 --lr 1e-3 --out runs/rhm-dynamics-rerun`) is descriptive and
does not enter H2a or H2b. This script reads only saved result files and writes:

  results/rerun_r64.json  per run and LR: test loss / accuracy / KL from Bayes, steps to threshold (diagnostic accuracy
                          >= 0.9 x Bayes, the H2b definition), the highest diagnostic loss after update 1,000 (a
                          collapse shows as ~2.77 = log 16), and the r = 16 reference at 3e-3 (seeds 0-2);
                          also the checkpoint steps each run's interpret.json covers (the interpretation step was
                          cut by its timeout before it could record its status in analysis.json).
  figures/rerun_r64.pdf   diagnostic loss and accuracy vs updates, r = 64 at 3e-3 and at 1e-3.

  PYTHONPATH=. python studies/rhm/rhm-dynamics/rerun.py runs/rhm-dynamics runs/rhm-dynamics-rerun studies/rhm/rhm-dynamics
"""
import json
import math
import sys
from pathlib import Path

from studies.rhm.analysis import dump, load_runs, t_thresh
from studies.rhm.plotting import CATEGORICAL, INK2, plt, save, steps_x

LOG16 = math.log(16)


def summary(r):
    log = r['log']
    late = [l for s, l in zip(log['step'], log['diag_loss']) if s >= 1000]
    return dict(lr=r['lr'], status=r['status'], test=r['test'], bayes_test=r['bayes_test'], t_thresh=t_thresh(r),
                max_diag_loss_after_1000=max(late), collapsed=max(late) > LOG16 - 0.05)


def interp_steps(path):
    """Checkpoint steps an interpret.json covers. It is rewritten after every checkpoint, so a run cut off by the
    step's timeout leaves a partial file with fewer than its 15 checkpoints."""
    return sorted(c['step'] for c in json.load(open(path))['checkpoints']) if path.exists() else []


def main(orig_dir, rerun_dir, exp_dir):
    orig, rerun = load_runs(orig_dir), load_runs(rerun_dir)
    r64 = sorted(n for n in orig if '_r64_' in n)
    out = dict(
        r64={n: {'lr3e-3': summary(orig[n]), 'lr1e-3': summary(rerun[n])} for n in r64},
        r16_ref={n: summary(orig[n]) for n in sorted(orig) if '_m4_' in n and '_r16_' in n and 'varinit' not in n},
        interpret={n: interp_steps(Path(orig_dir) / n / 'interpret.json') for n in sorted(orig)})
    dump(Path(exp_dir) / 'results/rerun_r64.json', out)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    for j, task in enumerate(('root', 'last')):
        for n in (n for n in r64 if n.startswith(task)):
            for runs, ls, lab in ((orig, '-', 'peak LR 3e-3 (shared)'), (rerun, '--', 'peak LR 1e-3 (rerun)')):
                log, g = runs[n]['log'], int(n.split('_g')[1][0])
                for i, key in enumerate(('diag_loss', 'diag_acc')):
                    axes[i, j].plot(steps_x(log['step']), log[key], ls=ls, color=CATEGORICAL[g], lw=1.0,
                                    label=f'grammar {g}, {lab}' if i == 0 else None)
        axes[0, j].axhline(LOG16, color=INK2, ls=':', lw=0.8)
        axes[0, j].set_yscale('log')
        axes[0, j].set_title(f"task '{task}', r = 64, m = 4, init seed 0")
        axes[1, j].set_xlabel('updates')
        axes[1, j].set_xscale('log')
    axes[0, 0].set_ylabel('diagnostic loss (nats)\n(dotted: log 16, uniform output)')
    axes[1, 0].set_ylabel('diagnostic accuracy')
    axes[0, 1].legend(fontsize=6, frameon=False)
    fig.suptitle('r = 64 at the shared LR (solid) and at 1e-3 (dashed); 1 run per grammar and LR')
    save(fig, Path(exp_dir) / 'figures/rerun_r64.pdf')
    for n, v in out['r64'].items():
        print(n, {k: (round(s['test']['kl'], 4), s['t_thresh'], s['collapsed']) for k, s in v.items()})
    full = {n: len(v) for n, v in out['interpret'].items()}
    print('interpretation complete (15 checkpoints):', sum(v == 15 for v in full.values()), 'of', len(full),
          '| partial:', {n: v for n, v in full.items() if v != 15})


if __name__ == '__main__':
    main(*sys.argv[1:4])
