"""Three descriptive checks added by the supervisor after the run (no criterion rests on them; see README, "Observed after the run").

    PYTHONPATH=. python studies/rhm/rhm-crossed-rank/posthoc.py runs/rhm-crossed-rank studies/rhm/rhm-crossed-rank/results

1. Every exact run of the four cells at their bound (default rates), not only the cell's best: the largest weight and
   the largest centered-logit error over the 24 legal prefixes. H4 reports these for the best run alone.
2. `mod` crossed at r = 3 and r = 4, default rates: is a width-4 tree that is not exact at the same function as the
   best width-3 tree? Two numbers per run, both on legal prefixes only (the tree is unconstrained elsewhere) and both
   independent of the gauge of the bonds:
     - the largest difference between its centered logits and those of the best width-3 crossed tree on the 24 legal
       prefixes;
     - the singular values of its centered logits on the legal 4 x 4 block that forces width 4 (rows (x_1 x_3) = 00,
       01, 02, 10; columns (x_2 x_4, y) = (12, 0), (12, 1), (21, 0), (21, 1)), in units of log(2/3)/3. The target's
       block has determinant -27; a tree whose bond for (x_1, x_3) carries three channels has a singular block.
3. The width lower bound of every bond a tree over x_1 .. x_4 could have (toy.cut_bound for each single token and
   each pair of tokens), for both grammars: the bounds of the pairing the grid does not train, (x_1 x_4)(x_2 x_3).
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from studies.rhm import toy as Y

CELLS = (('sym', 'aligned', 2), ('sym', 'crossed', 2), ('mod', 'aligned', 3), ('mod', 'crossed', 4))


ROWS, COLS = [(0, 0), (0, 1), (0, 2), (1, 0)], [(1, 2), (2, 1)]   # (x_1, x_3) and (x_2, x_4) of the legal 4 x 4 block
UNIT = math.log(2 / 3) / 3


def block_singular_values(model):
    """Singular values of the centered logits on the legal 4 x 4 block, in units of log(2/3)/3."""
    x = torch.tensor([[a, b, c, d] + [Y.MASK] * 4 for a, c in ROWS for b, d in COLS])
    with torch.no_grad():
        M = Y.center(model(x))[:, :2].reshape(4, 4).numpy() / UNIT   # outputs y = 0, 1
    return np.linalg.svd(M, compute_uv=False)


def main(runs_dir, out_dir):
    out = dict(exact_runs={}, cut_rank={})
    lines = []
    for n, g, r in CELLS:
        pop = Y.population(n)
        w, e = [], []
        for f in sorted(Path(runs_dir).glob(f'{n}_{g}_r{r}_*_s2*.pt')):
            run = torch.load(f)
            if '_lr' in f.stem or not run['kl_polished'] < Y.EXACT:
                continue
            model = Y.load_model(run)
            with torch.no_grad():
                e.append((Y.center(model(pop['x'])) - Y.center(pop['p'].log())).abs().max().item())
            w.append(max(p.abs().max().item() for p in model.parameters()))
        out['exact_runs'][f'{n}_{g}_r{r}'] = dict(n=len(w), largest_weight=[min(w), float(np.median(w)), max(w)],
                                                  largest_centered_logit_error=[min(e), float(np.median(e)), max(e)])
        lines.append(f'{n} {g} r={r}: {len(w)} exact runs; largest |weight| min/median/max {min(w):.2f} / {np.median(w):.2f} / {max(w):.2f}; '
                     f'largest centered-logit error min/median/max {min(e):.1e} / {np.median(e):.1e} / {max(e):.1e}')
    pop = Y.population('mod')
    runs = {f.stem: torch.load(f) for r in (3, 4) for f in sorted(Path(runs_dir).glob(f'mod_crossed_r{r}_*_s2*.pt')) if '_lr' not in f.stem}
    ref = min((k for k in runs if '_r3_' in k), key=lambda k: runs[k]['kl_polished'])
    with torch.no_grad():
        F3 = Y.center(Y.load_model(runs[ref])(pop['x']))
        target = np.linalg.svd(Y.center(pop['p'].log())[[pop['x'][:, :4].tolist().index([a, b, c, d]) for a, c in ROWS for b, d in COLS]][:, :2]
                               .reshape(4, 4).numpy() / UNIT, compute_uv=False)
    out['block_of_target'] = target.tolist()
    lines.append(f'reference width-3 tree: {ref}; singular values of the target on the legal block: ' + ' '.join(f'{v:.3f}' for v in target))
    for k, run in runs.items():
        model = Y.load_model(run)
        with torch.no_grad():
            d = (Y.center(model(pop['x'])) - F3).abs().max().item()
        s = block_singular_values(model)
        out['cut_rank'][k] = dict(kl=run['kl_polished'], init=run['cfg']['init'], largest_logit_difference_from_reference=d, block_singular_values=s.tolist())
        lines.append(f"{k}: excess {run['kl_polished']:.6e}  largest logit difference from the reference {d:.1e}  block singular values " + ' '.join(f'{v:.3f}' for v in s))
    out['cut_bounds'] = {}
    for n in Y.GRAMMARS:
        cuts = [(i,) for i in range(4)] + list(itertools.combinations(range(4), 2))
        b = {''.join(f'x{i + 1}' for i in c): Y.cut_bound(n, c) for c in cuts}
        out['cut_bounds'][n] = b
        lines.append(f'{n}: width lower bound per bond  ' + '  '.join(f'({k}) {v}' for k, v in b.items()))
    json.dump(out, open(Path(out_dir) / 'posthoc.json', 'w'), indent=1)
    Path(out_dir, 'posthoc.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
