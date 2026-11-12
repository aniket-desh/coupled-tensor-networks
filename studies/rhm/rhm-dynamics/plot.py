"""rhm-dynamics figures from <results>/curves.json (written by analyze.py) and the interpretation JSONs.

  PYTHONPATH=. python studies/rhm/rhm-dynamics/plot.py studies/rhm/rhm-dynamics/results studies/rhm/rhm-dynamics/figures \
      [runs/rhm-dynamics]

widths.pdf: diagnostic loss, KL from Bayes and accuracy vs updates, columns = task, color = width (m = 4, init seed 0;
one line per grammar), dashed line = Bayes. seeds.pdf: r = 16, init seeds 0-2. m8.pdf: m = 4 vs m = 8 at r = 16. init.pdf: identity vs variance init.
With the runs folder given: interp_<task>_m<m>/ per (task, m), from `python -m studies.rhm.plot_interp` over the
<run>/interpret.json of the identity-init r = 16 runs of that group (median and range over grammars and seeds)."""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from studies.rhm.plotting import CATEGORICAL, INK2, WIDTH_COLORS, plt, save, steps_x

TASKS = ('root', 'last')
ROWS = (('diag_loss', 'diagnostic loss (nats)', 'loss'), ('diag_kl', 'KL from Bayes (nats)', None),
        ('diag_acc', 'diagnostic accuracy', 'acc'))


def panel_grid(curves, select, color, label, path, title):
    fig, axes = plt.subplots(len(ROWS), len(TASKS), figsize=(7, 7), sharex=True)
    for j, task in enumerate(TASKS):
        seen = set()
        for n, c in sorted(curves.items()):
            if c['cfg']['task'] != task or not select(c):
                continue
            k = label(c)
            for i, (key, ylab, bkey) in enumerate(ROWS):
                ax = axes[i, j]
                ax.plot(steps_x(c['step']), c[key], color=color(c), lw=1.0, alpha=0.85,
                        label=k if (k not in seen and i == 0) else None)
                if bkey and task == 'last':
                    ax.axhline(c['bayes_diag'][bkey], color=INK2, ls='--', lw=0.8)
            seen.add(k)
        for i, (key, ylab, _) in enumerate(ROWS):
            ax = axes[i, j]
            ax.set_xscale('log')
            if key != 'diag_acc':
                ax.set_yscale('log')
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"task '{task}'" + (' (dashed: Bayes)' if task == 'last' else ' (Bayes: loss 0, acc 1)'))
                ax.legend(fontsize=7)
            if i == len(ROWS) - 1:
                ax.set_xlabel('updates')
    fig.suptitle(title)
    save(fig, path)


def main():
    res, figs = Path(sys.argv[1]), Path(sys.argv[2])
    curves = json.load(open(res / 'curves.json'))
    ident = lambda c: c['cfg'].get('init') == 'identity'
    m4 = lambda c: c['cfg']['grammar']['m'] == 4 and ident(c)
    panel_grid(curves, lambda c: m4(c) and c['cfg']['seed'] == 0, lambda c: WIDTH_COLORS.get(c['cfg']['r'], 'k'),
               lambda c: f"r = {c['cfg']['r']}", figs / 'widths.pdf', 'Width, m = 4, init seed 0 (one line per grammar)')
    panel_grid(curves, lambda c: m4(c) and c['cfg']['r'] == 16, lambda c: CATEGORICAL[c['cfg']['seed']],
               lambda c: f"init seed {c['cfg']['seed']}", figs / 'seeds.pdf', 'Init seeds, r = 16, m = 4 (one line per grammar)')
    panel_grid(curves, lambda c: ident(c) and c['cfg']['r'] == 16 and c['cfg']['seed'] == 0,
               lambda c: CATEGORICAL[0 if m4(c) else 1], lambda c: f"m = {c['cfg']['grammar']['m']}", figs / 'm8.pdf',
               'Synonyms per symbol, r = 16, init seed 0 (one line per grammar)')
    sel = lambda c: (c['cfg']['grammar']['m'], c['cfg']['r'], c['cfg']['seed']) == (4, 16, 0)
    panel_grid(curves, sel, lambda c: CATEGORICAL[0 if ident(c) else 1], lambda c: f"{c['cfg'].get('init')} init",
               figs / 'init.pdf', 'Identity vs variance init, r = 16, m = 4, init seed 0 (one line per grammar)')
    if len(sys.argv) > 3:
        interp_figures(curves, Path(sys.argv[3]), figs)


def interp_figures(curves, runs, figs):
    groups = defaultdict(list)
    for n, c in sorted(curves.items()):
        f = runs / n / 'interpret.json'
        if c['cfg']['r'] == 16 and c['cfg'].get('init') == 'identity' and f.exists():
            groups[(c['cfg']['task'], c['cfg']['grammar']['m'])].append(str(f))
    if not groups:
        print('note: no interpret.json found; skipping the interpretation figures')
    for (task, m), files in sorted(groups.items()):
        out = figs / f'interp_{task}_m{m}'
        rc = subprocess.run([sys.executable, '-m', 'studies.rhm.plot_interp', *files, str(out)]).returncode
        print(f'interp_{task}_m{m}: {len(files)} runs, plot_interp ' + ('ok' if rc == 0 else f'failed (exit {rc})'))


if __name__ == '__main__':
    main()
