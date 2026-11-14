"""Train the causal transformer (model.py) on next-token prediction for one RHM grammar, with fresh samples each update.

    PYTHONPATH=. python studies/rhm/rhm-belief-geometry/run.py v8m4s2L3 --seed 0 --out runs/rhm-belief-geometry

A config name vVmMsSLL (optionally nN, default n = v) fixes the grammar; --seed sets both the grammar seed and the init
seed. Loss is the mean cross-entropy over all T positions. Every log point records the per-position test loss next to
the exact Bayes loss -log P(x_t | x_<t) = -log P(x_<=t) + log P(x_<t), with prefix probabilities from
grammar.likelihoods with the unseen suffix marginalized, so the excess loss is exact at every position."""
import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from studies.rhm import grammar as G
from studies.rhm.train import lr_at

import sys
sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402

TEST_SEED, N_TEST = 30_000, 8192


def parse_cfg(name):
    d = {k: int(v) for k, v in re.findall(r'([vmsLn])(\d+)', name)}
    d.setdefault('n', d['v'])
    return d


def bayes_loss(x, rules, v, chunk=4096):
    """(T,) mean over sequences of -log P(x_t | x_<t), exact."""
    B, T = x.shape
    logp = torch.zeros(B, T + 1, dtype=torch.float64)  # logp[:, t] = log P(x_<t)
    for t in range(1, T + 1):
        for i in range(0, B, chunk):
            lik = G.likelihoods(x[i:i + chunk], rules, v, unknown=range(t, T))[0][:, 0]
            logp[i:i + chunk, t] = lik.mean(-1).log()  # uniform root prior
    return -(logp[:, 1:] - logp[:, :-1]).mean(0)


@torch.no_grad()
def test_loss(model, x, v, chunk=4096):
    out = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        lp = model(M.inputs(xb, v)).log_softmax(-1)
        out += -lp.gather(-1, xb.unsqueeze(-1))[..., 0].double().sum(0)
    return (out / len(x)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/rhm-belief-geometry')
    ap.add_argument('--steps', type=int, default=50_000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = parse_cfg(a.config)
    v, s, L = g['v'], g['s'], g['L']
    T = s ** L
    rules = G.sample_rules(v, g['n'], g['m'], s, L, seed=a.seed)
    run_dir = Path(a.out) / f'{a.config}_s{a.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    xt = G.sample_trees(N_TEST, rules, seed=TEST_SEED + a.seed)[L]
    bayes = bayes_loss(xt, rules, v)
    xt = xt.to(dev)
    torch.manual_seed(a.seed)
    model = M.Transformer(v, T, d=a.d, layers=a.layers, heads=a.heads).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(dev).manual_seed(1000 + a.seed)
    rules_dev = {l: r.to(dev) for l, r in rules.items()}
    ckpts = sorted({0, 300, 1000, 3000, 10000, 30000, a.steps} & set(range(a.steps + 1)))
    logs = sorted(set(ckpts) | set(range(0, a.steps + 1, 1000)) | {100, 200, 500})
    hist, t0 = [], time.time()
    cfg = dict(grammar=dict(g, seed=a.seed), d=a.d, layers=a.layers, heads=a.heads, T=T, V=v)
    for step in range(a.steps + 1):
        if step in logs:
            model.eval()
            tl = test_loss(model, xt, v)
            hist.append(dict(step=step, test=tl.mean().item(), excess=(tl - bayes).mean().item(),
                             test_pos=tl.tolist(), sec=time.time() - t0))
            print(f'{a.config} s{a.seed} step {step:6d} test {hist[-1]["test"]:.4f} '
                  f'excess {hist[-1]["excess"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
            model.train()
        if step in ckpts:
            torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, run_dir / f'ckpt_{step}.pt')
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step, a.lr, a.steps, warmup=a.warmup)
        x = G.sample_trees(a.batch, rules_dev, g=gen, device=dev)[L]
        loss = F.cross_entropy(model(M.inputs(x, v)).flatten(0, 1), x.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            print('diverged', flush=True)
            break
    json.dump(dict(cfg=cfg, args=vars(a), bayes_pos=bayes.tolist(), bayes=bayes.mean().item(), hist=hist,
                   ms_per_update=1000 * (time.time() - t0) / max(1, step), ckpts=ckpts),
              open(run_dir / 'result.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
