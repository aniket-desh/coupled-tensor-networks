"""Train a TreeTN on RHM prefix prediction: the same next-token task as the causal transformer of
rhm-belief-geometry, for a model without a causal mask.

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/ttn_run.py v8m4s2L4 --r 32 --seed 0 --out runs/rhm-ttn-belief-geometry

Task. For a sequence x_0 .. x_{T-1} and each prefix length t = 0 .. T-1, the tree sees x_<t with every leaf at
position >= t replaced by the mask token v (leaf vocabulary v + 1) and predicts x_t (v classes). One update uses every
prefix of `batch` fresh sequences (batch * T rows), so the loss is the transformer's: mean cross-entropy over sequences
and positions. The tree learns which token to predict from the mask pattern.

Model. ctn.tree.TreeTN on the grammar's own leaf order (binary grammars only), untied, identity init (eps 0.1), as in
the other RHM studies. --seed sets the grammar seed and the init seed, as in rhm-belief-geometry, so run
<config>_s<seed> here uses the same rules, test set and Bayes loss as the transformer run of the same name."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ctn.tree import TreeTN, identity_init, n_params
from studies.rhm import grammar as G
from studies.rhm.train import lr_at

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rhm-belief-geometry'))
from run import N_TEST, TEST_SEED, bayes_loss, parse_cfg  # noqa: E402


def prefixes(x, v):
    """x (B, T) -> (xm (B * T, T), y (B * T,)): row (b, t) is x_b with positions >= t masked, target x_b[t]."""
    B, T = x.shape
    ar = torch.arange(T, device=x.device)
    masked = ar[None, :] >= ar[:, None]  # (t, position)
    xm = torch.where(masked[None], torch.full_like(x[:, None, :], v), x[:, None, :])
    return xm.reshape(B * T, T), x.reshape(B * T)


def build(cfg):
    g = cfg['grammar']
    model = TreeTN(cfg['T'], g['v'] + 1, cfg['r'], g['v'], tie='none', seed=g['seed'])
    identity_init(model, cfg['init_eps'])
    return model


@torch.no_grad()
def test_loss(model, x, v, chunk=1024):
    """(T,) mean over sequences of -log q(x_t | x_<t masked input)."""
    T = x.shape[1]
    out = torch.zeros(T, dtype=torch.float64, device=x.device)
    for i in range(0, len(x), chunk):
        xm, y = prefixes(x[i:i + chunk], v)
        lp = model(xm).log_softmax(-1).gather(-1, y[:, None])[:, 0]
        out += -lp.double().reshape(-1, T).sum(0)
    return (out / len(x)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/rhm-ttn-belief-geometry')
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=100)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = parse_cfg(a.config)
    assert g['s'] == 2, 'TreeTN is a binary tree'
    v, L = g['v'], g['L']
    T = 2 ** L
    rules = G.sample_rules(v, g['n'], g['m'], 2, L, seed=a.seed)
    run_dir = Path(a.out) / f'{a.config}_r{a.r}_s{a.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    xt = G.sample_trees(N_TEST, rules, seed=TEST_SEED + a.seed)[L]
    bayes = bayes_loss(xt, rules, v)
    xt = xt.to(dev)
    cfg = dict(kind='ttn', grammar=dict(g, seed=a.seed), r=a.r, init_eps=0.1, T=T, V=v)
    model = build(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(dev).manual_seed(1000 + a.seed)
    rules_dev = {l: r.to(dev) for l, r in rules.items()}
    ckpts = sorted({0, 300, 1000, 3000, 10000, a.steps} & set(range(a.steps + 1)))
    logs = sorted(set(ckpts) | set(range(0, a.steps + 1, 1000)) | {100, 200, 500})
    hist, t0, status = [], time.time(), 'finished'
    for step in range(a.steps + 1):
        if step in logs:
            tl = test_loss(model, xt, v)
            hist.append(dict(step=step, test=tl.mean().item(), excess=(tl - bayes).mean().item(),
                             test_pos=tl.tolist(), sec=time.time() - t0))
            print(f'{a.config} r{a.r} s{a.seed} step {step:6d} test {hist[-1]["test"]:.4f} '
                  f'excess {hist[-1]["excess"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if step in ckpts:
            torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, run_dir / f'ckpt_{step}.pt')
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step, a.lr, a.steps, warmup=a.warmup)
        xm, y = prefixes(G.sample_trees(a.batch, rules_dev, g=gen, device=dev)[L], v)
        loss = F.cross_entropy(model(xm), y)
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
