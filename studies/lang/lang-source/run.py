"""lang-source: selecting one source token (fixed source) and combining two (two-source) in a TreeTN.

Processes (V = 8, target noise 0.05, iid uniform context; studies/lang/processes.py):
  fixed_j{j}    y = sigma(x_j), j in {0, 7, 8, 15}. All 16 leaves of the untied tree are equivalent under the tree's
                automorphisms, so the tree arms should not depend on j; the chain reads x_j 15 - j steps before the end.
  two_{i}_{j}   y = f(x_i, x_j), pairs (0,1) LCA level 1, (1,2) level 2, (3,4) level 3, (7,8), (0,8), (0,15) the root.
                (0,1)..(7,8) all have sequence distance 1; (7,8), (0,8), (0,15) share the root as LCA and differ in
                sequence distance (1, 8, 15).
Arms: tree (untied, r = 16), tied (tie='level', r = 16), chain (shared-core MPS, width matched to the r = 16 tree's
parameter count), tree32 (untied, r = 32, capacity check on fixed_j0, two_0_1, two_7_8).
Init seeds 0-2; the process instance (sigma, f) uses the same seed; every arm with the same (condition, seed) sees
the same data stream. 20,000 updates. Peak LR piloted per arm on two_3_4, seed 9, over the first 2,000 updates
(these processes are learned within a few hundred, so a longer pilot would choose by post-convergence loss).

  PYTHONPATH=. python studies/lang/lang-source/run.py --out runs/lang-source --hours 3 [--smoke]
"""
import json
from pathlib import Path

from ctn.tree import n_params
from studies.lang import common as L
from studies.lang import train as LT
from studies.rhm import train as T

V, R, STEPS, PILOT_STEPS = 8, 16, 20_000, 2_000
# dense early checkpoints: these processes are learned within a few hundred updates
CKPTS = [0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 2000, 5000, 10000, 20000]
FIXED = [0, 7, 8, 15]
PAIRS = [(0, 1), (1, 2), (3, 4), (7, 8), (0, 8), (0, 15)]
ARMS = ('tree', 'tied', 'chain')
CAP_CONDS = ('fixed_j0', 'two_0_1', 'two_7_8')


def conds():
    out = {f'fixed_j{j}': dict(kind='fixed', V=V, j=j) for j in FIXED}
    out.update({f'two_{i}_{j}': dict(kind='two', V=V, i=i, j=j) for i, j in PAIRS})
    return out


def cfg(proc, arm, seed):
    p = dict(proc, pseed=seed)
    c = dict(process=p, model='tree', r=R, tie='none', perm=None, seed=seed, init='identity', init_eps=0.1, arm=arm)
    if arm == 'tied':
        c['tie'] = 'level'
    elif arm == 'chain':
        c.update(model='chain', r=L.chain_width(p, R)[0], tie=None)
    elif arm == 'tree32':
        c['r'] = 32
    return c


def main():
    a = LT.args(hours=3.0)
    steps, pilot_steps = (300, 100) if a.smoke else (STEPS, PILOT_STEPS)
    seeds = (0,) if a.smoke else (0, 1, 2)
    C = conds()
    if a.smoke:
        C = {k: C[k] for k in ('fixed_j0', 'two_0_1')}
    ck = [c for c in CKPTS if c <= steps] if not a.smoke else LT.ckpt_list(steps)
    specs = []
    for s in seeds:
        for ci, (cn, proc) in enumerate(C.items()):
            arms = ARMS + (('tree32',) if cn in CAP_CONDS else ())
            for arm in arms:
                specs.append(LT.spec(f'{cn}_{arm}_s{s}', cfg(proc, arm, s), steps,
                                     data_seed=T.ONLINE_SEED + 1000 * s + ci, extra_ckpts=ck if arm != 'chain' else ()))
    pilot_proc = dict(kind='two', V=V, i=3, j=4)
    base = {arm: LT.spec(f'pilot_{arm}', cfg(pilot_proc, arm, T.PILOT_SEED), pilot_steps, data_seed=T.ONLINE_SEED + 999)
            for arm in ARMS + ('tree32',)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    meta = {arm: dict(params=n_params(L.build(base[arm]['cfg'])), r=base[arm]['cfg']['r']) for arm in base}
    json.dump(dict(arms=meta, conds=C, checkpoints=ck), open(Path(a.out) / 'meta.json', 'w'), indent=1)
    print(json.dumps(meta), flush=True)
    lr = LT.pilot(a, base, pilot_steps, steps)
    T.start_clock(a)
    for sp in specs:
        sp['lr'] = lr[sp['cfg']['arm']]
    T.run_all(a, specs, LT.run)


if __name__ == '__main__':
    main()
