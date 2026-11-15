"""Train a TreeTN on RHM token prediction under one of three observation patterns (the factor under test).

    PYTHONPATH=. python studies/rhm/rhm-ttn-masks/mask_run.py v8m4s2L4 --arm varied --r 32 --seed 0
    PYTHONPATH=. python studies/rhm/rhm-ttn-masks/mask_run.py --check

Task, the same for every arm: the tree sees a sequence with some leaves replaced by the mask token v and predicts the
token under the LEFTMOST mask. A row is (sequence, query position q): positions < q are observed, q is masked, and the
arms differ only in the positions > q:
  prefix  all masked. This is exactly rhm-ttn-belief-geometry/ttn_run.py (same rows, same order), where a parent's
          right child sees tokens only once its left child is fully observed.
  varied  each position > q is masked independently with probability u, u ~ U(0, 1) drawn per row. A parent whose left
          child holds the query then has a partially observed left child and an informative right child: the
          combination the prefix task never shows. A varied row is still a pure prefix row with probability
          1 / (T - q), 21% on average over q at T = 16.
  mixed   each row is a prefix row or a varied row with probability 1/2 (so about 61% prefix rows at T = 16).
The prefix rows are the u = 1 case of the varied rows, so the input format, the model and the read-out are identical
across arms and every arm can be scored on the prefix task.

One update uses every query position of `batch` fresh sequences (batch * T rows), as in ttn_run.py. The sequence stream
(generator 1000 + seed) and the init are the same in the three arms; the masks come from a separate generator
(2000 + seed), so the arms see the same sequences.

Losses logged against exact Bayes: `test` = the prefix task on ttn_run's test set (the comparison across arms), and
`vtest` = the varied task on the first N_VTEST of those sequences with fixed masks (seed 3000 + seed), with the exact
posterior from posterior_masked below."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ctn.tree import n_params
from studies.rhm import grammar as G
from studies.rhm.train import lr_at

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent / 'rhm-ttn-belief-geometry'))
import ttn_run as TR  # noqa: E402
from run import N_TEST, TEST_SEED, bayes_loss, parse_cfg  # noqa: E402

ARMS = ('prefix', 'varied', 'mixed')
N_VTEST = 2048


def rows(x, v, arm, gen):
    """x (B, T) -> (xm (B * T, T), y (B * T,)): row (b, q) has positions < q observed, q masked, and positions > q
    masked according to the arm; the target is x_b[q]. Row order is ttn_run.prefixes's."""
    B, T = x.shape
    ar = torch.arange(T, device=x.device)
    after = (ar[None, :] > ar[:, None])[None].expand(B, T, T)  # (b, q, position): position > q
    if arm == 'prefix':
        hide = after
    else:
        u = torch.rand(B, T, 1, generator=gen, device=x.device)
        hide = after & (torch.rand(B, T, T, generator=gen, device=x.device) < u)
        if arm == 'mixed':
            hide = torch.where(torch.rand(B, T, 1, generator=gen, device=x.device) < 0.5, after, hide)
    masked = hide | (ar[None, :] == ar[:, None])[None]
    xm = torch.where(masked, torch.full_like(x[:, None, :], v), x[:, None, :])
    return xm.reshape(B * T, T), x.reshape(B * T)


def posterior_masked(xm, rules, v, q):
    """Exact P(x_q | observed tokens), (N, v), for rows xm (N, T) in which the value v marks an unobserved leaf and q
    (N,) is the queried position: one upward pass (grammar.likelihoods's recursion) per candidate token at q."""
    N = xm.shape[0]
    s, L = rules[0].shape[2], len(rules)
    base = torch.where((xm == v)[..., None], torch.ones((), dtype=torch.float64, device=xm.device),
                       F.one_hot(xm.clamp_max(v - 1), v).double())  # (N, T, v): all-ones at unobserved leaves
    out = torch.zeros(N, v, dtype=torch.float64, device=xm.device)
    for c in range(v):
        lik = base.clone()
        lik[torch.arange(N), q] = F.one_hot(torch.tensor(c), v).double().to(xm.device)
        for l in range(L - 1, -1, -1):
            R = rules[l].to(xm.device)
            child = lik.reshape(N, -1, s, lik.shape[-1])
            prod_ = torch.ones(N, child.shape[1], R.shape[0], R.shape[1], dtype=torch.float64, device=xm.device)
            for j in range(s):
                prod_ = prod_ * child[:, :, j][:, :, R[:, :, j]]
            lik = prod_.mean(-1)
        out[:, c] = lik[:, 0].sum(-1)  # the uniform root prior cancels
    return out / out.sum(-1, keepdim=True)


@torch.no_grad()
def masked_loss(model, xm, y, chunk=16384):
    return torch.cat([-model(xm[i:i + chunk]).log_softmax(-1).gather(-1, y[i:i + chunk, None])[:, 0].double()
                      for i in range(0, len(xm), chunk)]).mean().item()


def check():
    """posterior_masked against the two exact references the project already has, and the row construction."""
    g = parse_cfg('v4m2s2L3')
    v, L = g['v'], g['L']
    T = 2 ** L
    rules = G.sample_rules(v, g['n'], g['m'], 2, L, seed=0)
    x = G.sample_trees(64, rules, seed=1)[L]
    q = torch.arange(T).repeat(len(x))
    # prefix rows: -log P(x_q | x_<q) must be run.bayes_loss
    xm, y = rows(x, v, 'prefix', None)
    assert torch.equal(xm, TR.prefixes(x, v)[0])
    p = posterior_masked(xm, rules, v, q)
    nll = -p.gather(1, y[:, None])[:, 0].log().reshape(-1, T).mean(0)
    assert torch.allclose(nll, bayes_loss(x, rules, v), atol=1e-12), (nll, bayes_loss(x, rules, v))
    # everything but q observed: must be grammar.posterior_token
    for pos in (0, 3, T - 1):
        xo = x.clone()
        xo[:, pos] = v
        ref = G.posterior_token(x, rules, v, pos)
        assert torch.allclose(posterior_masked(xo, rules, v, torch.full((len(x),), pos)), ref, atol=1e-12)
    # varied rows: the leftmost mask is q, positions < q are the sequence's, and the arms share the sequences
    gen = torch.Generator().manual_seed(0)
    for arm in ('varied', 'mixed'):
        xv, yv = rows(x, v, arm, gen)
        assert torch.equal((xv == v).int().argmax(1), q) and torch.equal(yv, y)
        assert torch.equal(xv[xv != v], x.repeat_interleave(T, 0)[xv != v])
        n_masked = (xv == v).sum(1)
        assert (n_masked < (xm == v).sum(1)).any() and (n_masked <= (xm == v).sum(1)).all()
    print('mask_run checks passed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--arm', choices=ARMS, default='prefix')
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/rhm-ttn-masks')
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=100)
    a = ap.parse_args()
    if a.check:
        return check()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = parse_cfg(a.config)
    assert g['s'] == 2, 'TreeTN is a binary tree'
    v, L = g['v'], g['L']
    T = 2 ** L
    rules = G.sample_rules(v, g['n'], g['m'], 2, L, seed=a.seed)
    run_dir = Path(a.out) / a.arm / f'{a.config}_r{a.r}_s{a.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    xt = G.sample_trees(N_TEST, rules, seed=TEST_SEED + a.seed)[L]
    bayes = bayes_loss(xt, rules, v)
    xv, yv = rows(xt[:N_VTEST], v, 'varied', torch.Generator().manual_seed(3000 + a.seed))
    qv = torch.arange(T).repeat(N_VTEST)
    vbayes = torch.cat([-posterior_masked(xv[i:i + 8192], rules, v, qv[i:i + 8192]).gather(1, yv[i:i + 8192, None])[:, 0].log()
                        for i in range(0, len(xv), 8192)]).mean().item()
    xt, xv, yv = xt.to(dev), xv.to(dev), yv.to(dev)
    cfg = dict(kind='ttn', grammar=dict(g, seed=a.seed), r=a.r, init_eps=0.1, T=T, V=v, arm=a.arm)
    model = TR.build(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(dev).manual_seed(1000 + a.seed)  # sequences: ttn_run's stream
    mgen = torch.Generator(dev).manual_seed(2000 + a.seed)  # masks
    rules_dev = {l: r.to(dev) for l, r in rules.items()}
    ckpts = sorted({0, 3000, a.steps} & set(range(a.steps + 1)))  # the criteria use the first and the last
    logs = sorted(set(ckpts) | set(range(0, a.steps + 1, 1000)) | {100, 200, 500})
    hist, t0, status = [], time.time(), 'finished'
    for step in range(a.steps + 1):
        if step in logs:
            tl = TR.test_loss(model, xt, v)
            vt = masked_loss(model, xv, yv)
            hist.append(dict(step=step, test=tl.mean().item(), excess=(tl - bayes).mean().item(), vtest=vt,
                             vexcess=vt - vbayes, test_pos=tl.tolist(), sec=time.time() - t0))
            print(f'{a.config} {a.arm} r{a.r} s{a.seed} step {step:6d} prefix excess {hist[-1]["excess"]:.4f} '
                  f'varied excess {hist[-1]["vexcess"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if step in ckpts:
            torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, run_dir / f'ckpt_{step}.pt')
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step, a.lr, a.steps, warmup=a.warmup)
        xm, y = rows(G.sample_trees(a.batch, rules_dev, g=gen, device=dev)[L], v, a.arm, mgen)
        loss = F.cross_entropy(model(xm), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            status = 'diverged'
            print('diverged', flush=True)
            break
    json.dump(dict(cfg=cfg, args=vars(a), status=status, n_params=n_params(model), bayes_pos=bayes.tolist(),
                   bayes=bayes.mean().item(), vbayes=vbayes, hist=hist,
                   ms_per_update=1000 * (time.time() - t0) / max(1, step), ckpts=[c for c in ckpts if c <= step]),
              open(run_dir / 'result.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
