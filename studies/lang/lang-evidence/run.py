"""lang-evidence: combining evidence from two directions (masked bigram) and detecting an event (reserved trigger).

Processes (V = 16; studies/lang/processes.py):
  mb_p{p}   stationary Markov chain, leaf p masked, y = x_p; p*(y|x) prop. to P(y|x_{p-1}) P(x_{p+1}|y). The two
            neighbours meet at tree level 2 for p = 5 (leaves 4..7), level 3 for p = 4 (leaves 0..7), and only at the
            root core for p = 8 (7 and 9 sit in different halves).
  trigger   x_0..x_14 iid; with probability 1/2 one holds the reserved trigger token; y = c(x_15) if a trigger is
            present else d(x_15), target noise 0.05.
Arms. Masked bigram: tree (untied, r = 16), perm (same tree on a fixed leaf permutation with no adjacent pair kept as
siblings, r = 16), chain (width matched to the r = 16 tree), tree32 (r = 32, every p; the causal test of the evidence mechanism
needs a probe subspace well below the bond width, which r = 16 can't give).
Trigger: tree16, tree32, perm32, chain (width matched to the r = 32 tree).
Init seeds 0-2 (process instance = seed; perm drawn per seed), same data stream across arms per (condition, seed).
30,000 updates; peak LR piloted per (process, arm) on seed 9 (masked bigram pilots on p = 4).

  PYTHONPATH=. python studies/lang/lang-evidence/run.py --out runs/lang-evidence --hours 4 [--smoke]
"""
import json
from pathlib import Path

from ctn.tree import n_params
from studies.lang import common as L
from studies.lang import train as LT
from studies.rhm import train as T

V, STEPS, PILOT_STEPS = 16, 30_000, 4_000
MB_P = (5, 4, 8)
MB_ARMS, TR_ARMS = ('tree', 'perm', 'chain'), ('tree16', 'tree32', 'perm32', 'chain')


def cfg(proc, arm, seed):
    p = dict(proc, pseed=seed)
    r = 32 if arm.endswith('32') else 16
    c = dict(process=p, model='tree', r=r, tie='none', perm=None, seed=seed, init='identity', init_eps=0.1, arm=arm)
    if arm.startswith('perm'):
        c['perm'] = L.perm_split(seed)
    elif arm == 'chain':
        c.update(model='chain', r=L.chain_width(p, 32 if p['kind'] == 'trigger' else 16)[0], tie=None)
    return c


def main():
    a = LT.args(hours=4.0)
    steps, pilot_steps = (300, 100) if a.smoke else (STEPS, PILOT_STEPS)
    seeds = (0,) if a.smoke else (0, 1, 2)
    conds = {f'mb_p{p}': (dict(kind='mbigram', V=V, p=p), MB_ARMS + ('tree32',)) for p in MB_P}
    conds['trigger'] = (dict(kind='trigger', V=V), TR_ARMS)
    if a.smoke:
        conds = {k: conds[k] for k in ('mb_p5', 'trigger')}
    ck = LT.ckpt_list(steps)
    specs = []
    for s in seeds:
        for ci, (cn, (proc, arms)) in enumerate(conds.items()):
            for arm in arms:
                specs.append(LT.spec(f'{cn}_{arm}_s{s}', cfg(proc, arm, s), steps,
                                     data_seed=T.ONLINE_SEED + 1000 * s + ci, extra_ckpts=ck if arm != 'chain' else ()))
    mbp, trp = dict(kind='mbigram', V=V, p=4), dict(kind='trigger', V=V)
    base = {f'mb_{arm}': LT.spec(f'pilot_mb_{arm}', cfg(mbp, arm, T.PILOT_SEED), pilot_steps, data_seed=T.ONLINE_SEED + 998)
            for arm in MB_ARMS + ('tree32',)}
    base.update({f'trigger_{arm}': LT.spec(f'pilot_tr_{arm}', cfg(trp, arm, T.PILOT_SEED), pilot_steps,
                                           data_seed=T.ONLINE_SEED + 997) for arm in TR_ARMS})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    meta = {k: dict(params=n_params(L.build(b['cfg'])), r=b['cfg']['r']) for k, b in base.items()}
    json.dump(dict(arms=meta, conds={k: v[0] for k, v in conds.items()}, checkpoints=ck,
                   perms={s: L.perm_split(s) for s in seeds}), open(Path(a.out) / 'meta.json', 'w'), indent=1)
    print(json.dumps(meta), flush=True)
    lr = LT.pilot(a, base, pilot_steps, steps)
    T.start_clock(a)
    for sp in specs:
        kind = 'trigger' if sp['cfg']['process']['kind'] == 'trigger' else 'mb'
        sp['lr'] = lr[f"{kind}_{sp['cfg']['arm']}"]
    T.run_all(a, specs, LT.run)


if __name__ == '__main__':
    main()
