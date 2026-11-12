"""rhm-controls: architecture controls at r = 16, both tasks, grammars 0-2, init seeds 0 and 1.

Arms (same online data stream per (grammar, init seed) in every arm):
  tree   untied TreeTN on the correct leaf order (the reference, rerun here)
  perm   untied TreeTN with a fixed random leaf permutation (seeded per (grammar, seed), redrawn until no sibling
         pair of the data tree are siblings in the model; recorded in the config and in <out>/perms.json)
  tied   level-tied TreeTN: one leaf embedding A and one fusion core per level
  chain  shared-core chain (ChainTN) with its width chosen so its parameter count is closest to the r = 16 tree's
The peak LR is piloted per (task, arm) over {1e-4, 3e-4, 1e-3, 3e-3} on grammar 0, init seed 9 (no evaluation run uses
it), 4,000 updates of the 40,000-update schedule; --lr overrides it (reruns go to a separate --out). 40,000 updates
per run. Priority: init seed 0 (all arms, grammars, tasks), then init seed 1. Under the identity init the tied tree
and the chain start position-blind: one leaf embedding for all positions, so position enters only through the
structure.

  PYTHONPATH=. python studies/rhm/rhm-controls/run.py --out runs/rhm-controls --hours 4.5 [--smoke]
"""
import json
from pathlib import Path

import torch

from ctn.tree import ChainTN, n_params, tree_params
from studies.rhm import common as C
from studies.rhm import train as T

TASKS, ARMS = ('root', 'last'), ('tree', 'perm', 'tied', 'chain')
R, STEPS, PILOT_STEPS = 16, 40_000, 4_000


def chain_width(task, r_tree=R):
    """Chain width whose parameter count r + V r + r^3 + O r is closest to the untied r_tree tree's."""
    g = C.GRAMMAR
    V, O = g['v'] + (task == 'last'), (g['n'] if task == 'root' else g['v'])
    target = tree_params(16, V, r_tree, O)
    return min(range(2, 4 * r_tree), key=lambda r: abs(r + V * r + r ** 3 + O * r - target)), target


def perm_for(gseed, seed, n=16):
    """A seeded leaf permutation with no data-tree sibling pair kept as model siblings. Also reports, per level, how
    many model nodes cover exactly the leaves of some data-tree node."""
    g = torch.Generator().manual_seed(500 + 10 * gseed + seed)
    while True:
        p = torch.randperm(n, generator=g).tolist()
        if all({p[2 * j], p[2 * j + 1]} != {2 * k, 2 * k + 1} for j in range(n // 2) for k in range(n // 2)):
            break
    kept = {}
    for l in range(1, 4):
        w = 2 ** l
        data = {frozenset(range(k * w, (k + 1) * w)) for k in range(n // w)}
        kept[f'level{l}'] = sum(frozenset(p[j * w:(j + 1) * w]) in data for j in range(n // w))
    return p, kept


def cfg(task, arm, gseed, seed):
    c = dict(model='tree', r=R, tie='none', perm=None, seed=seed, task=task, grammar=T.grammar_cfg(gseed), arm=arm,
             init='identity')
    if arm == 'perm':
        c['perm'] = perm_for(gseed, seed)[0]
    elif arm == 'tied':
        c['tie'] = 'level'
    elif arm == 'chain':
        c.update(model='chain', r=chain_width(task)[0], tie=None)
    return c


def spec(c, steps, lr=None):
    g = c['grammar']
    return dict(name=f"{c['task']}_{c['arm']}_g{g['seed']}_s{c['seed']}", cfg=c, lr=lr, steps=steps, data='online',
                data_seed=T.ONLINE_SEED + 1000 * g['seed'] + 100 * g['m'] + c['seed'])


def main():
    a = T.args(hours=4.5)
    steps, pilot_steps = (300, 100) if a.smoke else (STEPS, PILOT_STEPS)
    G3, seeds = ((0,), (0,)) if a.smoke else ((0, 1, 2), (0, 1))
    cfgs = [cfg(t, arm, g, s) for s in seeds for g in G3 for t in TASKS for arm in ARMS]
    meta = dict(chain={t: dict(zip(('width', 'tree_params'), chain_width(t))) for t in TASKS},
                perms={f'g{g}_s{s}': dict(zip(('perm', 'model_nodes_matching_data_nodes'), perm_for(g, s)))
                       for g in G3 for s in seeds})
    for t in TASKS:
        meta['chain'][t]['chain_params'] = n_params(C.build(cfg(t, 'chain', 0, 0)))
        meta['chain'][t]['tied_params'] = n_params(C.build(cfg(t, 'tied', 0, 0)))
        assert isinstance(C.build(cfg(t, 'chain', 0, 0)), ChainTN)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(Path(a.out) / 'controls_meta.json', 'w'), indent=1)
    print(json.dumps(meta['chain']), flush=True)
    T.prepare(a, [T.grammar_cfg(g) for g in G3])
    lr = T.pilot(a, {f'{t}_{arm}': spec(cfg(t, arm, 0, T.PILOT_SEED), pilot_steps) for t in TASKS for arm in ARMS},
                 pilot_steps, steps)
    T.start_clock(a)
    T.run_all(a, [spec(c, steps, lr[f"{c['task']}_{c['arm']}"]) for c in cfgs], T.run_standard)


if __name__ == '__main__':
    main()
