"""rhm-sample-efficiency: r = 16 TreeTNs trained on fixed training sets of N trees, both tasks, grammars 0-2,
init seed 0.

N in {2^10, 2^14, 2^18, 2^20} (first priority: the sizes the mentor asked for), then {2^12, 2^16}. The training sets
are nested prefixes of one seeded 2^20-tree sample per grammar. Up to 100,000 updates (batch 256, drawn uniformly
with replacement; LR schedule over 100,000 updates). The fixed 20,000-tree diagnostic set is the validation set, with
two early-stopping selections: the highest diagnostic accuracy (for accuracy, 'learned' and the regimes) and the
lowest diagnostic loss (for loss and KL); both are reported next to the final numbers. A run stops early only when the diagnostic loss has not improved for 25,000 updates AND the loss on the
training set is below 1e-3 nats (it has fit the data, so going on cannot change the answer); otherwise it runs the
full 100,000 updates, so a high training loss means an optimization failure rather than too little training.
The peak LR per task comes from the online pilot (r = 16, grammar 0, init seed 9, {1e-4, 3e-4, 1e-3, 3e-3}, 4,000 of 40,000
updates), as in rhm-dynamics; --lr overrides it (reruns go to a separate --out). Last in priority: 6 online reference
runs (fresh trees every update, same model and 100,000-update schedule, capped at 20,000 updates), the N -> infinity
end of the curve. All results are for the TreeTN with the identity init.

  PYTHONPATH=. python studies/rhm/rhm-sample-efficiency/run.py --out runs/rhm-sample-efficiency --hours 5.5 [--smoke]
"""
from studies.rhm import train as T

TASKS = ('root', 'last')
STEPS, PILOT_STEPS, PATIENCE, TRAIN_TOL, ONLINE_CAP = 100_000, 4_000, 25_000, 1e-3, 20_000
NS_FIRST, NS_SECOND = (2 ** 10, 2 ** 14, 2 ** 18, 2 ** 20), (2 ** 12, 2 ** 16)


def cfg(task, gseed):
    return dict(model='tree', r=16, tie='none', perm=None, seed=0, task=task, grammar=T.grammar_cfg(gseed), init='identity')


def spec(c, steps, N, lr=None, patience=PATIENCE):
    g = c['grammar']
    return dict(name=f"{c['task']}_g{g['seed']}_N{N}", cfg=c, lr=lr, steps=steps, data=N,
                data_seed=9_000_000 + 1000 * g['seed'] + c['seed'], early_stop=dict(patience=patience, train_tol=TRAIN_TOL))


def online_ref(c, steps, cap, lr):
    g = c['grammar']
    return dict(name=f"{c['task']}_g{g['seed']}_online", cfg=c, lr=lr, steps=cap, schedule_total=steps, data='online',
                data_seed=T.ONLINE_SEED + 1000 * g['seed'] + 400 + c['seed'])


def main():
    a = T.args(hours=5.5)
    if a.smoke:
        steps, pilot_steps, patience, G3, Ns, cap = 300, 100, 100, (0,), (2 ** 10, 2 ** 12), 200
    else:
        steps, pilot_steps, patience, G3, Ns, cap = STEPS, PILOT_STEPS, PATIENCE, (0, 1, 2), NS_FIRST + NS_SECOND, ONLINE_CAP
    T.prepare(a, [T.grammar_cfg(g) for g in G3])
    online = lambda t: dict(name=t, cfg=dict(cfg(t, 0), seed=T.PILOT_SEED), data='online',
                            data_seed=T.ONLINE_SEED + 400 + T.PILOT_SEED)
    lr = T.pilot(a, {t: online(t) for t in TASKS}, pilot_steps, 40_000 if not a.smoke else 300)
    T.start_clock(a)
    specs = ([spec(cfg(t, g), steps, N, lr[t], patience) for N in Ns for g in G3 for t in TASKS]
             + [online_ref(cfg(t, g), steps, cap, lr[t]) for g in G3 for t in TASKS])
    T.run_all(a, specs, T.run_standard)


if __name__ == '__main__':
    main()
