"""Learning-rate pilot per grouping, on pilot seeds 0-3 (never used by run.py), run before the experiment.

    PYTHONPATH=. python studies/rhm/rhm-crossed-rank/lr_pilot.py studies/rhm/rhm-crossed-rank/results/lr_pilot.json --procs 2

toy.LR was chosen on `sym` aligned at r = 2. This repeats that choice for each grouping at its own width bound
(sym r = 2 in both groupings, mod aligned r = 3, mod crossed r = 4), so a lower share of exact runs in a crossed cell
cannot be put down to a rate tuned for the aligned tree. Per cell, init and rate: the excess loss after 20,000 Adam
updates and after the polish, for 4 seeds. A rate is "best" for (cell, init) if it has the most exact runs, ties broken
by the lower median Adam endpoint."""
import json
import sys
from multiprocessing import Pool

import numpy as np
import torch

from studies.rhm import toy as Y

CELLS = [('sym', 'aligned', 2), ('sym', 'crossed', 2), ('mod', 'aligned', 3), ('mod', 'crossed', 4)]
LRS, SEEDS = (1e-3, 3e-3, 1e-2, 3e-2), range(4)


def one(job):
    name, g, r, init, lr, seed = job
    torch.set_num_threads(1)
    pop = Y.population(name)
    model = Y.build(r, g, seed, init)
    log, _ = Y.train(model, pop, Y.STEPS, lr, log_every=Y.STEPS)
    kl, info = Y.polish(model, pop, Y.POLISH_ITERS)
    return dict(grammar=name, grouping=g, r=r, init=init, lr=lr, seed=seed, kl_adam=log[-1][1], kl=kl, capped=info['capped'])


if __name__ == '__main__':
    out, procs = sys.argv[1], int(sys.argv[sys.argv.index('--procs') + 1]) if '--procs' in sys.argv else 2
    jobs = [(n, g, r, init, lr, s) for n, g, r in CELLS for init in ('variance', 'identity') for lr in LRS for s in SEEDS]
    with Pool(procs) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    table = {}
    for n, g, r in CELLS:
        for init in ('variance', 'identity'):
            per = {}
            for lr in LRS:
                q = [x for x in rows if (x['grammar'], x['grouping'], x['init'], x['lr']) == (n, g, init, lr)]
                kl = np.array([x['kl'] for x in q])
                per[lr] = dict(n_exact=int((kl < Y.EXACT).sum()), median_adam=float(np.nanmedian([x['kl_adam'] for x in q])),
                               polished=sorted(float(k) for k in kl))
            best = max(LRS, key=lambda lr: (per[lr]['n_exact'], -per[lr]['median_adam']))
            table[f'{n}_{g}_r{r}_{init}'] = dict(per_lr={f'{lr:g}': per[lr] for lr in LRS}, best=best, used=Y.LR[init],
                                                 used_n_exact=per[Y.LR[init]]['n_exact'], best_n_exact=per[best]['n_exact'])
            print(f"{n} {g} r={r} {init:8s} " + '  '.join(f"{lr:g}: {per[lr]['n_exact']}/4 exact (Adam median {per[lr]['median_adam']:.1e})" for lr in LRS)
                  + f"   best {best:g}, used {Y.LR[init]:g}", flush=True)
    json.dump(dict(table=table, runs=rows, provenance=Y.provenance()), open(out, 'w'), indent=1)
