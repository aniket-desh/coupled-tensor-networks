"""Train aligned and crossed TreeTNs of width 1 to 6 on the `sym` and `mod` grammars (studies/rhm/toy.py).

    PYTHONPATH=. python studies/rhm/rhm-crossed-rank/run.py --out runs/rhm-crossed-rank --procs 4

Main grid: grammar in {mod, sym} x grouping in {aligned, crossed} x r in 1..6 x init in {variance, identity} x seeds
2000..2009 = 480 runs at each init's default rate (toy.LR), each 20,000 full-batch Adam updates on the 24 legal
prefixes and an L-BFGS polish. The two groupings of a cell share seeds, inits and learning rates.

Rate arm, after the main grid: the four cells where each grouping sits at its own width bound (TRAIN_CELLS) at the
three other rates of toy.LRS, same inits and seeds = 240 runs, so that the two groupings can be compared with each at
its own best rate.

One file per run, written when the run ends; finished runs are skipped, so the command can be repeated after an
interruption."""
import argparse

from studies.rhm import toy as Y

WIDTHS, SEEDS = (3, 4, 2, 5, 1, 6), range(2000, 2010)   # the widths that carry the criteria first


TRAIN_CELLS = (('mod', 'aligned', 3), ('mod', 'crossed', 4), ('sym', 'aligned', 2), ('sym', 'crossed', 2))


def grid(widths=WIDTHS, seeds=SEEDS, steps=Y.STEPS):
    main = [dict(grammar=name, grouping=g, r=r, init=init, lr=Y.LR[init], seed=s, steps=steps)
            for r in widths for name in ('mod', 'sym') for init in ('variance', 'identity') for s in seeds
            for g in ('aligned', 'crossed')]
    rates = [dict(grammar=name, grouping=g, r=r, init=init, lr=lr, seed=s, steps=steps)
             for name, g, r in TRAIN_CELLS if r in widths for init in ('variance', 'identity')
             for lr in Y.LRS if lr != Y.LR[init] for s in seeds]
    return main + rates


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='widths 2-4, seed 1000, 2,000 updates (a test of the pipeline, not a result)')
    a = ap.parse_args()
    cfgs = grid(widths=(2, 3, 4), seeds=range(1000, 1001), steps=2000) if a.smoke else grid()
    Y.run_grid(cfgs, a.out, a.procs)
