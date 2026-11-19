"""Root task only: the saved trajectories at r = 8, 16, 32, 64 (peak LR 3e-3, init seed 0, 3 grammars), with the
r = 64 rerun at 1e-3 beside them. Separates three things that the width table merges: the update at which a run first
reaches 0.9 accuracy, how far it falls afterwards, and where it ends.

  PYTHONPATH=. python studies/rhm/rhm-dynamics/root_widths.py --collect runs/rhm-dynamics-rerun   # writes the json
  PYTHONPATH=. python studies/rhm/rhm-dynamics/root_widths.py                                      # figure only

--collect copies the 15 root trajectories into results/root_widths.json: 12 from results/curves.json and the three
1e-3 reruns from <dir>/root_m4_g*_r64_s0/result.json (the rerun's curves are otherwise only in the run folder, which
is not in git). The figure reads only that file and writes figures/root_widths.pdf.

Per run: t90 = first logged update with diagnostic accuracy >= 0.9 (the definition's threshold, 0.9 x Bayes accuracy,
Bayes = 1 on root); low = the lowest logged diagnostic accuracy after t90; final = test accuracy at update 40,000.
The diagnostic set is logged every 50 updates up to 3,000 and every 1,000 after, so a dip shorter than 1,000 updates
late in training can be missed: `low` is an upper bound on the true minimum."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'root_widths.json'
ARMS = [(8, '3e-3'), (16, '3e-3'), (32, '3e-3'), (64, '3e-3'), (64, '1e-3')]


def record(res, lr):
    log = res['log'] if 'log' in res else res
    step, acc, loss = log['step'], log['diag_acc'], log['diag_loss']
    t90 = next((s for s, a in zip(step, acc) if a >= 0.9), None)
    after = [(a, s) for s, a in zip(step, acc) if t90 is not None and s > t90]
    low, low_step = min(after) if after else (None, None)
    return dict(lr=lr, step=step, diag_acc=acc, diag_loss=loss, t90=t90, low=low, low_step=low_step,
                n_below_090_after_t90=sum(a < 0.9 for a, _ in after), final_diag_acc=acc[-1],
                test_acc=res['test']['acc'], test_kl=res['test']['kl'])


def collect(rerun_dir):
    curves = json.loads((HERE / 'results' / 'curves.json').read_text())
    out = {}
    for g in range(3):
        for r in (8, 16, 32, 64):
            c = curves[f'root_m4_g{g}_r{r}_s0']
            assert c['status'] == 'finished' and abs(c['lr'] - 3e-3) < 1e-12
            out[f'g{g}_r{r}_lr3e-3'] = record(c, 3e-3)
        res = json.loads((Path(rerun_dir) / f'root_m4_g{g}_r64_s0' / 'result.json').read_text())
        assert res['status'] == 'finished' and abs(res['lr'] - 1e-3) < 1e-12 and res['cfg']['grammar']['seed'] == g
        out[f'g{g}_r64_lr1e-3'] = record(res, 1e-3)
    OUT.write_text(json.dumps(out))
    print('wrote', OUT)


def table(d):
    print(f"{'run':18s} {'t90':>5s} {'lowest acc after t90 (at update)':>34s} {'evals < 0.9':>12s} {'test acc':>9s} {'test KL':>9s}")
    for r, lr in ARMS:
        for g in range(3):
            x = d[f'g{g}_r{r}_lr{lr}']
            print(f"g{g} r={r:<3d} lr {lr:5s} {x['t90']:5d} {x['low']:22.4f} ({x['low_step']:5d}) "
                  f"{x['n_below_090_after_t90']:12d} {x['test_acc']:9.4f} {x['test_kl']:9.2e}")


def plot(d):
    from studies.rhm.plotting import CATEGORICAL, INK, INK2, plt, save, steps_x
    col = CATEGORICAL[:3]  # one hue per grammar, the same in every panel
    fig = plt.figure(figsize=(9.6, 5.0))
    gs = fig.add_gridspec(2, 15, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.6, top=0.86)
    top = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(5)]
    bot = [fig.add_subplot(gs[1, 5 * i:5 * i + 4]) for i in range(3)]
    name = lambda r, lr: f'r = {r}' + ('' if r < 64 else f', peak LR {lr}')
    for ax, (r, lr) in zip(top, ARMS):
        for y, lab in ((1.0, 'Bayes'), (0.9, '0.9'), (1 / 16, 'chance')):
            ax.axhline(y, color=INK2, lw=0.6, ls={'Bayes': '--', '0.9': '-', 'chance': ':'}[lab], zorder=1)
            if ax is top[0]:
                ax.text(0.6, y + (0.02 if lab != '0.9' else -0.075), lab, color=INK2, fontsize=6.5, va='bottom')
        for g in range(3):
            x = d[f'g{g}_r{r}_lr{lr}']
            ax.plot(steps_x(x['step']), x['diag_acc'], color=col[g], lw=1.0, label=f'grammar {g}', zorder=3)
            i = x['step'].index(x['t90'])
            ax.plot([x['t90']], [x['diag_acc'][i]], 'o', ms=4.5, mfc=col[g], mec='white', mew=0.8, zorder=4)
        ax.set_xscale('log')
        ax.set_xlim(0.5, 40000)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([1, 100, 10000])
        ax.set_title(name(r, lr), fontsize=8, color=INK)
        ax.set_xlabel('update')
        if ax is not top[0]:
            ax.set_yticklabels([])
    top[0].set_ylabel('diagnostic accuracy')
    h, l = top[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=7, handlelength=1.4, bbox_to_anchor=(0.5, 0.955))

    xs = range(len(ARMS))
    labels = ['8', '16', '32', '64', '64\n(LR 1e-3)']
    jit = (-0.16, 0.0, 0.16)
    panels = [('t90', 'updates to first reach 0.9', True),
              ('low', 'lowest logged accuracy after that', False),
              ('test_acc', 'test accuracy at update 40,000', False)]
    for ax, (key, title, logy) in zip(bot, panels):
        if not logy:
            for y, ls in ((1.0, '--'), (0.9, '-'), (1 / 16, ':')):
                ax.axhline(y, color=INK2, lw=0.6, ls=ls, zorder=1)
            ax.set_ylim(0, 1.06)
        ax.axvline(3.5, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
        for i, (r, lr) in enumerate(ARMS):
            for g in range(3):
                ax.plot([i + jit[g]], [d[f'g{g}_r{r}_lr{lr}'][key]], 'o', ms=5, mfc=col[g], mec='white', mew=0.8, zorder=3)
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(80, 800)
            ax.set_yticks([100, 200, 400, 800])
            ax.set_yticklabels(['100', '200', '400', '800'])
            ax.minorticks_off()
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, 4.5)
        ax.set_xlabel('bond dimension r')
        ax.set_title(title, fontsize=8, color=INK)
    fig.suptitle('RHM root task, aligned untied TreeTN, identity init, init seed 0, peak LR 3e-3 unless marked. '
                 'Dots on the curves: first logged accuracy of 0.9 or more.', fontsize=8, color=INK, y=1.0)
    save(fig, HERE / 'figures' / 'root_widths.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--collect', default=None, help='the rerun folder (runs/rhm-dynamics-rerun); rewrites the json')
    a = p.parse_args()
    if a.collect:
        collect(a.collect)
    d = json.loads(OUT.read_text())
    table(d)
    plot(d)
