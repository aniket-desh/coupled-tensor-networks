"""lang-induction: storing associations before the query is known (retrieval) and induction with accidental repeats.

Processes (V = 8, target noise 0.05; studies/lang/processes.py):
  retrieval    7 key-value pairs on leaves (2m, 2m+1) with distinct keys, query on leaf 14 (a pair's key), blank on 15;
               y = that pair's value. Strata: the target pair's index m. Pair 6 shares the level-2 node with the query,
               pairs 4-5 the level-3 node, pairs 0-3 only the root, so the left half must keep all four of its
               associations for every possible query (sampled centered-logit rank 29 with the key set fixed).
  multimatch   x_0..x_13 iid, a planted i in 0..12, query x_14 = x_i, blank on 15; y = x_{i+1}; Bayes votes over
               all matches. Strata: number of matches (1, 2, 3+). Sampled exact rank up to ~440 at level 2, with
               ~75 singular values above 1% of the largest.
Arms per process: tree at r in {16, 32, 64, 128} (untied), perm at r in {16, 64} (leaf permutation with no adjacent
pair kept as siblings, so key and value of a pair never share a level-1 node), chain with width matched to the r = 16
and r = 64 trees' parameter counts. Seeds 0-2 (perm and data per seed). 60,000 updates.
Peak LR piloted per (process, arm, width) on seed 9 over 12,000 updates of the 60,000-update schedule, excluding LRs
whose pilot learned and then collapsed; grids: {1e-4, 3e-4, 1e-3, 3e-3} for r <= 32 and the r = 16 chain,
{3e-5, 1e-4, 3e-4, 1e-3} for r >= 64 and the wide chain.

  PYTHONPATH=. python studies/lang/lang-induction/run.py --out runs/lang-induction --hours 6 [--smoke]
"""
import json
from pathlib import Path

from ctn.tree import n_params
from studies.lang import common as L
from studies.lang import train as LT
from studies.rhm import train as T

V, STEPS, PILOT_STEPS = 8, 60_000, 12_000
# linear checkpoints in the stretch where staging is decided (cheap at r <= 32)
STAGE_CKPTS = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000]
ARMS = [('tree', 16), ('tree', 32), ('tree', 64), ('tree', 128), ('perm', 16), ('perm', 64), ('chain', 16), ('chain', 64)]
NARROW, WIDE = (1e-4, 3e-4, 1e-3, 3e-3), (3e-5, 1e-4, 3e-4, 1e-3)


def cfg(proc, arm, r, seed):
    p = dict(proc, pseed=seed)
    c = dict(process=p, model='tree', r=r, tie='none', perm=None, seed=seed, init='identity', init_eps=0.1,
             arm=f'{arm}{r}', r_ref=r)
    if arm == 'perm':
        c['perm'] = L.perm_split(seed)
    elif arm == 'chain':
        c.update(model='chain', r=L.chain_width(p, r)[0], tie=None)
    return c


def main():
    a = LT.args(hours=6.0)
    steps, pilot_steps = (300, 100) if a.smoke else (STEPS, PILOT_STEPS)
    seeds = (0,) if a.smoke else (0, 1, 2)
    arms = [('tree', 16), ('perm', 16), ('chain', 16)] if a.smoke else ARMS
    procs = {'retrieval': dict(kind='retrieval', V=V), 'multimatch': dict(kind='multimatch', V=V)}
    specs = []
    # priority: seed 0 of everything first; within a seed, narrow arms before wide ones
    for s in seeds:
        for ci, (pn, proc) in enumerate(procs.items()):
            for arm, r in arms:
                ck = (sorted(set(LT.ckpt_list(steps, n=12)) | ({c for c in STAGE_CKPTS if c <= steps} if not a.smoke else set()))
                      if r <= 32 else LT.ckpt_list(steps, n=8)) if arm != 'chain' else ()
                specs.append(LT.spec(f'{pn}_{arm}{r}_s{s}', cfg(proc, arm, r, s), steps,
                                     data_seed=T.ONLINE_SEED + 1000 * s + ci, extra_ckpts=ck))
    base, grids = {}, {}
    for pn, proc in procs.items():
        for arm, r in arms:
            k = f'{pn}_{arm}{r}'
            base[k] = LT.spec(f'pilot_{k}', cfg(proc, arm, r, T.PILOT_SEED), pilot_steps, data_seed=T.ONLINE_SEED + 995)
            grids[k] = WIDE if r >= 64 else NARROW
    Path(a.out).mkdir(parents=True, exist_ok=True)
    meta = {k: dict(params=n_params(L.build(b['cfg'])), r=b['cfg']['r'], model=b['cfg']['model']) for k, b in base.items()}
    json.dump(dict(arms=meta, perms={s: L.perm_split(s) for s in seeds}), open(Path(a.out) / 'meta.json', 'w'), indent=1)
    print(json.dumps(meta), flush=True)
    lr = LT.pilot(a, base, pilot_steps, steps, lrs=grids)
    T.start_clock(a)
    for sp in specs:
        sp['lr'] = lr[f"{sp['cfg']['process']['kind']}_{sp['cfg']['arm']}"]
    T.run_all(a, specs, LT.run)


if __name__ == '__main__':
    main()
