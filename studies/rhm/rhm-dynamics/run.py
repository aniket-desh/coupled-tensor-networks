"""rhm-dynamics: train TreeTNs on the RHM 'root' and 'last' tasks with online data and save checkpoints.

Grid, in priority order (a run whose turn comes after the deadline is recorded as skipped):
  1. m = 4, r = 16, init seed 0, grammars 0-2, both tasks
  2. m = 4, r in {8, 32}, init seed 0
  3. m = 4, r = 16, init seeds 1 and 2
  4. m = 4, r = 16, grammars 0-2, init seed 0 with the default variance init, capped at 5,000 updates of the
     40,000-update schedule (documents the collapse to the uniform-output saddle; every other run uses the identity
     init, ctn.tree.identity_init)
  5. m = 4, r = 64, init seed 0
  6. m = 8, r = 16, init seed 0, grammars 0-2 (more synonyms: the last token is less predictable)
40,000 updates each. The peak LR per task comes from a pilot at r = 16 on grammar 0 (m = 4) over {1e-4, 3e-4, 1e-3, 3e-3}
on init seed 9 (no evaluation run uses it): 4,000 updates of the 40,000-update schedule, the lower diagnostic loss
wins (saved in <out>/lr.json). The same pilot also runs at r = 8 and r = 64, only to record which LR those widths
prefer; the grid uses the r = 16 choice for every width. --lr overrides the pilot (reruns go to a separate --out).
The online data stream depends on (grammar, init seed) only, so runs that differ in width see the same trees.

  PYTHONPATH=. python studies/rhm/rhm-dynamics/run.py --out runs/rhm-dynamics --hours 3 [--smoke]
"""
import json
import math
from pathlib import Path
from studies.rhm import train as T

TASKS = ('root', 'last')
STEPS, PILOT_STEPS, VAR_STEPS = 40_000, 4_000, 5_000
# Checkpoints besides common.CHECKPOINTS, to resolve the transition. CPU probe with the 100-update warmup (r = 16,
# grammar 0): root diagnostic accuracy 0.08 / 0.21 / 0.40 / 0.74 / 0.94 at updates 100 / 200 / 300 / 400 / 500;
# last-token accuracy 0.12 / 0.41 / 0.52 / 0.62 / 0.70 at 50 / 100 / 150 / 200 / 300, 0.80 near 700 (Bayes 0.885).
# 700 and 1,500 cover slower settings (r = 8, m = 8).
EXTRA_CKPTS = (50, 150, 200, 400, 500, 700, 1500)


def cfg(task, r, gseed, seed, m=4, init='identity'):
    return dict(model='tree', r=r, tie='none', perm=None, seed=seed, task=task, grammar=T.grammar_cfg(gseed, m=m),
                init=init)


def name(c):
    return (f"{c['task']}_m{c['grammar']['m']}_g{c['grammar']['seed']}_r{c['r']}_s{c['seed']}"
            + ('_varinit' if c['init'] == 'variance' else ''))


def spec(c, steps, lr=None, total=None):
    g = c['grammar']
    total = total or steps
    return dict(name=name(c), cfg=c, lr=lr, steps=min(steps, total), schedule_total=steps, data='online',
                extra_ckpts=EXTRA_CKPTS if steps == STEPS else (),
                data_seed=T.ONLINE_SEED + 1000 * g['seed'] + 100 * g['m'] + c['seed'])


def grid(smoke):
    if smoke:
        return ([cfg(t, r, 0, 0) for r in (16, 8) for t in TASKS] + [cfg(t, 16, 0, 0, init='variance') for t in TASKS]
                + [cfg(t, 16, 0, 0, m=8) for t in TASKS])
    G3 = (0, 1, 2)
    return ([cfg(t, 16, g, 0) for g in G3 for t in TASKS]
            + [cfg(t, r, g, 0) for r in (8, 32) for g in G3 for t in TASKS]
            + [cfg(t, 16, g, s) for s in (1, 2) for g in G3 for t in TASKS]
            + [cfg(t, 16, g, 0, init='variance') for g in G3 for t in TASKS]
            + [cfg(t, 64, g, 0) for g in G3 for t in TASKS]
            + [cfg(t, 16, g, 0, m=8) for g in G3 for t in TASKS])


def shared_lr(a, lr, widths):
    """The shared LR per task is the r = 16 pilot choice, unless another width's pilot diverged (non-finite diagnostic
    loss) at that LR: then step down the grid until no width's pilot diverged, before any grid run starts. Recorded in
    lr.json as 'shared'. (An --lr override is used as given.)"""
    path = Path(a.out) / 'lr.json'
    saved = json.load(open(path))
    for t in TASKS:
        if saved[t].get('override'):
            continue
        down = sorted(T.LRS, reverse=True)
        choice = lr[t]
        for cand in [x for x in down if x <= choice]:
            bad = [r for r in widths if r != 16 and not math.isfinite(saved.get(f'{t}_r{r}', {}).get('diag_loss', {}).get(f'{cand:g}', 0.0))]
            if not bad:
                break
            print(f'shared LR {t}: pilot at r = {bad} diverged at {cand:g}; stepping down', flush=True)
        else:
            cand = min(T.LRS)
        saved[t]['shared'] = cand
        lr[t] = cand
    T.dump(path, saved)
    return lr


def main():
    a = T.args(hours=3.0)
    steps, pilot_steps, var_steps = (300, 100, 150) if a.smoke else (STEPS, PILOT_STEPS, VAR_STEPS)
    cfgs = grid(a.smoke)
    T.prepare(a, list({T.gkey(c['grammar']): c['grammar'] for c in cfgs}.values()))
    widths = (16,) if a.smoke else (16, 8, 64)
    pil = {(t if r == 16 else f'{t}_r{r}'): spec(cfg(t, r, 0, T.PILOT_SEED), pilot_steps) for r in widths for t in TASKS}
    lr = T.pilot(a, pil, pilot_steps, steps)  # keys 'root', 'last' (r = 16, used) and 'root_r8', ... (recorded only)
    lr = shared_lr(a, lr, widths)
    T.start_clock(a)
    T.run_all(a, [spec(c, steps, lr[c['task']], var_steps if c['init'] == 'variance' else None) for c in cfgs],
              T.run_standard)


if __name__ == '__main__':
    main()
