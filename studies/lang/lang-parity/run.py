"""lang-parity: a tiny representation that is hard to discover. Parity of k bits across init scales and input bias.

Process (studies/lang/processes.py, Parity): x_t iid bits (P(1) = 1/2 + bias), y = parity of the first k leaves,
target noise 0.05. The first k leaves form an aligned subtree for k = 2, 4, 8, 16. Exact representation needs only a
scalar bond (sampled centered-logit rank 1 at every cut, studies/lang/results/rank.json), so failure to learn is a
learning-dynamics result, not a capacity one. With bias 0 every proper subset of the k bits is independent of y.
Grid (main): k in {2, 4, 8, 16} x identity-init noise eps in {0.01, 0.1, 0.3, 1.0} x bias in {0, 0.2} x seeds 0-2,
untied tree r = 16. Extra (descriptive): mod-4 sum (q = 4) at k in {2, 4, 8}, eps 0.1, bias 0; narrow r = 2 parity
at k in {4, 8}, eps 0.1, bias 0. 60,000 updates. One peak LR, 1e-3, fixed in advance for every run (a pilot would
choose by post-convergence noise on easy cells, and every hypothesis here is about escape time), so init scale,
degree and bias are the only factors that differ. Runs with bias also log accuracy on an unbiased diagnostic set,
which their threshold uses (studies/lang/analysis.t_thresh).
Every run with the same (k, bias, seed) sees the same data stream.

  PYTHONPATH=. python studies/lang/lang-parity/run.py --out runs/lang-parity --hours 5 [--smoke]
"""
import json
from pathlib import Path

from studies.lang import train as LT
from studies.rhm import train as T

STEPS, LR = 60_000, 1e-3
KS, EPS, BIAS = (2, 4, 8, 16), (0.01, 0.1, 0.3, 1.0), (0.0, 0.2)


def cfg(k, eps, bias, seed, q=2, r=16, arm='main'):
    p = dict(kind='parity', k=k, q=q, bias=bias, pseed=seed)
    return dict(process=p, model='tree', r=r, tie='none', perm=None, seed=seed, init='identity', init_eps=eps, arm=arm)


def name(c):
    p = c['process']
    return f"{c['arm']}_k{p['k']}_e{c['init_eps']:g}_b{p['bias']:g}_s{c['seed']}"


def main():
    a = LT.args(hours=5.0)
    steps = 300 if a.smoke else STEPS
    seeds = (0,) if a.smoke else (0, 1, 2)
    ks, eps_l, bias_l = ((2, 8), (0.1,), (0.0,)) if a.smoke else (KS, EPS, BIAS)
    ck = LT.ckpt_list(steps, n=14)
    cfgs = [cfg(k, e, b, s) for s in seeds for k in ks for b in bias_l for e in eps_l]
    cfgs += [cfg(k, 0.1, 0.0, s, q=4, arm='mod4') for s in seeds for k in ((2,) if a.smoke else (2, 4, 8))]
    cfgs += [cfg(k, 0.1, 0.0, s, r=2, arm='r2') for s in seeds for k in ((4,) if a.smoke else (4, 8))]
    data_seed = lambda c: T.ONLINE_SEED + 1000 * c['seed'] + 10 * c['process']['k'] + int(10 * c['process']['bias']) + 3 * (c['process']['q'] == 4)
    specs = [LT.spec(name(c), c, steps, data_seed=data_seed(c), extra_ckpts=ck) for c in cfgs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(checkpoints=ck, ks=ks, eps=eps_l, bias=bias_l, lr=LR), open(Path(a.out) / 'meta.json', 'w'), indent=1)
    lr = {arm: a.lr_over.get(arm, a.lr_over.get('*', LR)) for arm in ('main', 'mod4', 'r2')}
    T.dump(Path(a.out) / 'lr.json', {arm: dict(lr=v, fixed_in_advance=v == LR) for arm, v in lr.items()})
    T.start_clock(a)
    for sp in specs:
        sp['lr'] = lr[sp['cfg']['arm']]
    T.run_all(a, specs, LT.run)


if __name__ == '__main__':
    main()
