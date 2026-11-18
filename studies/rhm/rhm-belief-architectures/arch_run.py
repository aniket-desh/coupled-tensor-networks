"""Train one tensor-network arm on RHM next-token prediction (the task of rhm-belief-geometry and
rhm-ttn-belief-geometry), for the architecture comparison.

    PYTHONPATH=. python studies/rhm/rhm-belief-architectures/arch_run.py v8m4s2L4 --arm perm --seed 0 --out runs/rhm-belief-architectures

Arms (all identity init, eps 0.1; no biases or nonlinearities):
  tree   the aligned TreeTN of rhm-ttn-belief-geometry (ttn_run.build), r = R_TREE, masked prefix task: for each prefix
         length t the tree sees x_<t with positions >= t replaced by the mask token and predicts x_t.
  perm   the same TreeTN on a fixed random leaf permutation (model leaf i reads position perm[i]), same task. One
         seeded permutation per (config, seed), redrawn until no sibling pair of the data tree are siblings in the
         model (the rule of rhm-controls/run.py: perm_for, generalized to any n).
  chain  ctn.tree.ChainTN read causally (belief_common.chain_states): logits[:, t] = U h_t predicts x_t from x_<t, no
         masks, leaf vocabulary v. Its width is the one whose parameter count r + 2 v r + r^3 is closest to the
         aligned tree's (chain_width).
Every arm's update uses the same `batch` fresh sequences (same seeded stream per (config, seed)) and the same loss,
the mean cross-entropy over sequences and positions. --seed sets the grammar seed and the init seed, as in the other
belief experiments, so run <config>_<arm>_s<seed> shares its rules, test set and Bayes loss with them. The experiment
trains every arm at the default LR; a run at another LR goes to <config>_<arm>@lr<lr>_s<seed> so it cannot overwrite
one, and is not part of the experiment."""
import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from ctn.tree import ChainTN, TreeTN, identity_init, n_params, tree_params
from studies import belief_common as BC
from studies.rhm import grammar as G
from studies.rhm.train import lr_at

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent / 'rhm-ttn-belief-geometry'))
import ttn_run as TR  # noqa: E402
from run import N_TEST, TEST_SEED, bayes_loss, parse_cfg  # noqa: E402

R_TREE, ARMS, LR = 32, ('tree', 'perm', 'chain'), 1e-3


def perm_for(config, seed, n):
    """(perm, kept): a seeded leaf permutation with no data-tree sibling pair kept as model siblings, and per level
    how many model nodes cover exactly the leaves of some data-tree node."""
    g = torch.Generator().manual_seed(500 + 10 * (zlib.crc32(config.encode()) % 100_000) + seed)
    while True:
        p = torch.randperm(n, generator=g).tolist()
        if all({p[2 * j], p[2 * j + 1]} != {2 * k, 2 * k + 1} for j in range(n // 2) for k in range(n // 2)):
            break
    kept = {}
    for l in range(1, int(math.log2(n))):
        w = 2 ** l
        data = {frozenset(range(k * w, (k + 1) * w)) for k in range(n // w)}
        kept[f'level{l}'] = sum(frozenset(p[j * w:(j + 1) * w]) in data for j in range(n // w))
    return p, kept


def chain_params(r, v):
    return r + v * r + r ** 3 + v * r  # h0, A, W, U


def chain_width(T, v, r_tree=R_TREE):
    """(width, tree parameter count): the chain width whose parameter count is closest to the aligned tree's."""
    target = tree_params(T, v + 1, r_tree, v)
    return min(range(2, 8 * r_tree), key=lambda r: abs(chain_params(r, v) - target)), target


def build(cfg):
    g, arm = cfg['grammar'], cfg['arm']
    if arm == 'tree':
        return TR.build(cfg)
    if arm == 'perm':
        model = TreeTN(cfg['T'], g['v'] + 1, cfg['r'], g['v'], tie='none', perm=cfg['perm'], seed=g['seed'])
    else:
        model = ChainTN(cfg['T'], g['v'], cfg['r'], g['v'], seed=g['seed'])
    identity_init(model, cfg['init_eps'])
    return model


def logits_targets(model, arm, x, v):
    """(logits (B * T, v), targets (B * T,)), rows ordered (sequence, position)."""
    if arm == 'chain':
        return BC.chain_states(model, x)[1].flatten(0, 1), x.flatten()
    xm, y = TR.prefixes(x, v)
    return model(xm), y


@torch.no_grad()
def test_loss(model, arm, x, v, chunk=1024):
    """(T,) mean over sequences of -log q(x_t | x_<t)."""
    T = x.shape[1]
    out = torch.zeros(T, dtype=torch.float64, device=x.device)
    for i in range(0, len(x), chunk):
        lg, y = logits_targets(model, arm, x[i:i + chunk], v)
        out += -lg.log_softmax(-1).gather(-1, y[:, None])[:, 0].double().reshape(-1, T).sum(0)
    return (out / len(x)).cpu()


def make_cfg(config, arm, seed):
    g = parse_cfg(config)
    assert g['s'] == 2, 'binary grammars only'
    T, v = 2 ** g['L'], g['v']
    width, target = chain_width(T, v)
    cfg = dict(kind='tn-arch', arm=arm, config=config, grammar=dict(g, seed=seed), init_eps=0.1, T=T, V=v,
               r=width if arm == 'chain' else R_TREE, perm=None, chain_width=width, tree_params=target)
    if arm == 'perm':
        cfg['perm'], cfg['perm_kept'] = perm_for(config, seed, T)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/rhm-belief-architectures')
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--ckpts', default='', help='comma-separated checkpoint steps, replacing the default set (0 and --steps are always kept)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = make_cfg(a.config, a.arm, a.seed)
    g, v, T = cfg['grammar'], cfg['V'], cfg['T']
    L = g['L']
    rules = G.sample_rules(v, g['n'], g['m'], 2, L, seed=a.seed)
    variant = '' if a.lr == LR else f'@lr{a.lr:g}'  # a non-default LR gets its own folder
    run_dir = Path(a.out) / f'{a.config}_{a.arm}{variant}_s{a.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    xt = G.sample_trees(N_TEST, rules, seed=TEST_SEED + a.seed)[L]
    bayes = bayes_loss(xt, rules, v)
    xt = xt.to(dev)
    model = build(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(dev).manual_seed(1000 + a.seed)  # the stream of ttn_run.py: every arm sees the same sequences
    rules_dev = {l: r.to(dev) for l, r in rules.items()}
    ckpts = {0, a.steps} | {int(c) for c in a.ckpts.split(',')} if a.ckpts else {0, 300, 1000, 3000, 10000, a.steps}
    ckpts = sorted(ckpts & set(range(a.steps + 1)))
    logs = sorted(set(ckpts) | set(range(0, a.steps + 1, 1000)) | {100, 200, 500})
    hist, t0, status = [], time.time(), 'finished'
    for step in range(a.steps + 1):
        if step in logs:
            tl = test_loss(model, a.arm, xt, v)
            hist.append(dict(step=step, test=tl.mean().item(), excess=(tl - bayes).mean().item(),
                             test_pos=tl.tolist(), sec=time.time() - t0))
            print(f'{a.config} {a.arm} s{a.seed} step {step:6d} test {hist[-1]["test"]:.4f} '
                  f'excess {hist[-1]["excess"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if step in ckpts:
            torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, run_dir / f'ckpt_{step}.pt')
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step, a.lr, a.steps, warmup=a.warmup)
        lg, y = logits_targets(model, a.arm, G.sample_trees(a.batch, rules_dev, g=gen, device=dev)[L], v)
        loss = F.cross_entropy(lg, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            status = 'diverged'
            print('diverged', flush=True)
            break
    json.dump(dict(cfg=cfg, args=vars(a), status=status, n_params=n_params(model), bayes_pos=bayes.tolist(),
                   bayes=bayes.mean().item(), hist=hist, ms_per_update=1000 * (time.time() - t0) / max(1, step),
                   ckpts=[c for c in ckpts if c <= step]),
              open(run_dir / 'result.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
