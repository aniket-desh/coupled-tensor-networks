"""Train aligned TreeTNs of width 1 to 4 on the `sym` grammar (studies/rhm/toy.py).

    PYTHONPATH=. python studies/rhm/rhm-forced-synonyms/run.py --out runs/rhm-forced-synonyms --procs 4

Grid: r in {1, 2, 3, 4} x init in {variance, identity} x seeds 2000..2019 = 160 runs, each 20,000 full-batch Adam
updates on the 24 legal prefixes and an L-BFGS polish. One file per run, written when the run ends; finished runs are
skipped, so the command can be repeated after an interruption."""
import argparse

from studies.rhm import toy as Y

WIDTHS, SEEDS = (1, 2, 3, 4), range(2000, 2020)   # pilots and the smoke test used seeds below 1100


def grid(widths=WIDTHS, seeds=SEEDS, steps=Y.STEPS):
    # r = 2 first: the criteria rest on it.
    return [dict(grammar='sym', grouping='aligned', r=r, init=init, lr=Y.LR[init], seed=s, steps=steps)
            for r in sorted(widths, key=lambda r: r != 2) for init in ('variance', 'identity') for s in seeds]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='seeds 1000 and 1001, 2,000 updates (a test of the pipeline, not a result)')
    a = ap.parse_args()
    cfgs = grid(seeds=range(1000, 1002), steps=2000) if a.smoke else grid()
    Y.run_grid(cfgs, a.out, a.procs)
