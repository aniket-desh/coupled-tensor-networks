"""Train one model on next-token prediction for one hidden Markov process (studies/hmm/processes.py), T = 16.

    PYTHONPATH=. python studies/hmm/hmm-belief-geometry/hmm_run.py mess3 chain --seed 0 --out runs/hmm-belief-geometry

Arms: transformer, chain (r = 32), tree (r = 32), tree8 (the tree at r = 8; its product features [h_L, h_R, h_L (x) h_R]
number 80, few enough to be probed on processes with few distinct prefixes). Models, all trained on the same loss, the mean over sequences and positions t = 0 .. T-1 of -log q(x_t | x_<t), with
fresh sequences every update:
  transformer  rhm-belief-geometry/model.py: causal, pre-LN, input [BOS, x_0 .. x_{T-2}] (4 layers, d 128, 4 heads).
  tree         ctn.tree.TreeTN, untied, identity init: for each t it sees x_<t with positions >= t replaced by the mask
               token V and predicts x_t (the prefix task of rhm-ttn-belief-geometry/ttn_run.py), batch * T rows.
  chain        ctn.tree.ChainTN, identity init, read causally: logits_t = U h_t with h_t the state after t tokens
               (belief_common.chain_states), so one pass gives every position.
--seed sets the init and the data stream; the process itself has no seed, and the test set and its exact Bayes loss are
the same for every model and seed of a process."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ctn.tree import ChainTN, TreeTN, identity_init, n_params
from studies import belief_common as BC
from studies.hmm import processes as P
from studies.rhm.train import lr_at

RHM = Path(__file__).resolve().parents[2] / 'rhm'
sys.path.insert(0, str(RHM / 'rhm-ttn-belief-geometry'))
sys.path.insert(0, str(RHM / 'rhm-belief-geometry'))
import model as M  # noqa: E402
from ttn_run import prefixes  # noqa: E402

T_CTX, TEST_SEED, N_TEST = 16, 30_000, 8192
MODELS = ('transformer', 'tree', 'chain')  # model kinds
ARMS = dict(transformer='transformer', chain='chain', tree='tree', tree8='tree')  # arm -> kind; tree8 is the tree at r = 8


def build(cfg):
    V, T, seed = cfg['V'], cfg['T'], cfg['seed']
    if cfg['model'] == 'transformer':
        torch.manual_seed(seed)
        return M.Transformer(V, T, d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'])
    m = (TreeTN(T, V + 1, cfg['r'], V, tie='none', seed=seed) if cfg['model'] == 'tree'
         else ChainTN(T, V, cfg['r'], V, seed=seed))
    identity_init(m, cfg['init_eps'])
    return m


def logits(model, kind, x, V):
    """(B, T, V): logits[:, t] predicts x_t from x_<t."""
    if kind == 'transformer':
        return model(M.inputs(x, V))
    if kind == 'chain':
        return BC.chain_states(model, x)[1]
    return model(prefixes(x, V)[0]).reshape(*x.shape, V)


@torch.no_grad()
def test_loss(model, kind, x, V, chunk=1024):
    out = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        out += -logits(model, kind, xb, V).log_softmax(-1).gather(-1, xb[..., None])[..., 0].double().sum(0)
    return (out / len(x)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('process', choices=P.NAMES)
    ap.add_argument('arm', choices=list(ARMS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/hmm-belief-geometry')
    ap.add_argument('--steps', type=int, default=30_000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--r', type=int, default=32)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = P.get(a.process)
    V = proc['V']
    kind = ARMS[a.arm]
    r = 8 if a.arm == 'tree8' else a.r
    run_dir = Path(a.out) / f'{a.process}_{a.arm}_s{a.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    xt = P.sample(proc, N_TEST, T_CTX, g=torch.Generator().manual_seed(TEST_SEED))
    bayes = P.bayes_loss(proc, xt)
    xt = xt.to(dev)
    cfg = dict(process=a.process, model=kind, arm=a.arm, seed=a.seed, T=T_CTX, V=V, S=proc['S'])
    cfg.update(dict(d=128, layers=4, heads=4) if kind == 'transformer' else dict(r=r, init_eps=0.1))
    warmup = 500 if kind == 'transformer' else 100  # as in rhm-belief-geometry and in the tree-network studies
    model = build(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(dev).manual_seed(1000 + a.seed)
    ckpts = sorted({0, 300, 1000, 3000, 10000, a.steps} & set(range(a.steps + 1)))
    logs = sorted(set(ckpts) | set(range(0, a.steps + 1, 1000)) | {100, 200, 500})
    hist, t0, status, step = [], time.time(), 'finished', 0
    for step in range(a.steps + 1):
        if step in logs:
            model.eval()
            tl = test_loss(model, kind, xt, V)
            hist.append(dict(step=step, test=tl.mean().item(), excess=(tl - bayes).mean().item(),
                             test_pos=tl.tolist(), sec=time.time() - t0))
            print(f'{a.process} {a.arm} s{a.seed} step {step:6d} test {hist[-1]["test"]:.4f} '
                  f'excess {hist[-1]["excess"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
            model.train()
        if step in ckpts:
            torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, run_dir / f'ckpt_{step}.pt')
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step, a.lr, a.steps, warmup=warmup)
        x = P.sample(proc, a.batch, T_CTX, g=gen, device=dev)
        loss = F.cross_entropy(logits(model, kind, x, V).flatten(0, 1), x.flatten())
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
