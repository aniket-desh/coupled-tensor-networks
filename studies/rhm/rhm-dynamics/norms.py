"""Per-tensor activation, gradient and update norms through training, to look at the r = 64 collapse.

Replays one rhm-dynamics run (same cfg, peak LR, 40,000-update cosine schedule, data seed and init) for the first
`--steps` updates and records, every `--every` updates, on the training batch of that update:
  act   ||z|| per node, mean over the batch: leaves (16), levels 1-3 (8, 4, 2), and the logits vector
  grad  Frobenius norm of the gradient of each node's slice: A_i (16), W^(1)_k (8), W^(2)_k (4), W^(3)_k (2), C
  upd   Frobenius norm of the Adam step applied to that slice (parameters after the step minus before)
  param Frobenius norm of the slice
The run is not bit-identical to the GPU run: the online data comes from a device generator, so the CPU stream differs,
and the collapse step (or whether it happens) may differ too.

  PYTHONPATH=. python studies/rhm/rhm-dynamics/norms.py --task last --gseed 0 --r 64 --steps 10000
  python studies/rhm/rhm-dynamics/norms.py --plot results/norms_last_g0_r64_lr3e-3.json
"""
import argparse
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = ['leaf', 'L1', 'L2', 'L3', 'root']


def slices(model):
    """{level: list of per-node parameter tensors}; untied cores keep every node of a level on the leading axis."""
    return {'leaf': list(model.A), 'L1': list(model.W[0]), 'L2': list(model.W[1]), 'L3': list(model.W[2]),
            'root': [model.C]}


def run(a):
    import torch
    import torch.nn.functional as F
    from studies.rhm import common as C
    from studies.rhm import train as T
    torch.set_num_threads(a.threads)
    g = T.grammar_cfg(a.gseed)
    cfg = dict(model='tree', r=a.r, tie='none', perm=None, seed=0, task=a.task, grammar=g, init='identity')
    torch.manual_seed(0)
    model = C.build(cfg)
    opt = T.make_opt(list(model.parameters()), a.lr)
    rules = C.rules(g)
    batch_fn = T.online_batches(cfg, rules, 'cpu', T.ONLINE_SEED + 1000 * g['seed'] + 100 * g['m'] + cfg['seed'])
    rec = dict(cfg=cfg, lr=a.lr, schedule_total=40_000, step=[], loss=[],
               act={k: [] for k in LEVELS}, grad={k: [] for k in LEVELS}, upd={k: [] for k in LEVELS},
               param={k: [] for k in LEVELS})
    total = 40_000
    out = HERE / 'results' / f"norms_{a.task}_g{a.gseed}_r{a.r}_lr{a.lr:.0e}.json".replace('e-0', 'e-')
    t0 = time.time()
    for step in range(a.steps + 1):
        lr = T.lr_at(step, a.lr, total)
        for gr in opt.param_groups:
            gr['lr'] = lr
        x, y = batch_fn(step)
        logits, acts = model(x, acts=True)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        log = step % a.every == 0
        if log:
            with torch.no_grad():
                rec['step'].append(step)
                rec['loss'].append(loss.item())
                for l, z in enumerate(acts):  # z: (B, nodes, r)
                    rec['act'][LEVELS[l]].append(z.norm(dim=-1).mean(0).tolist())
                rec['act']['root'].append([logits.norm(dim=-1).mean().item()])
                before = {k: [p.detach().clone() for p in v] for k, v in slices(model).items()}
                grads = {'leaf': model.A.grad, 'L1': model.W[0].grad, 'L2': model.W[1].grad, 'L3': model.W[2].grad,
                         'root': model.C.grad[None]}
                for k in LEVELS:
                    rec['grad'][k].append([gi.norm().item() for gi in grads[k]])
                    rec['param'][k].append([p.norm().item() for p in before[k]])
        opt.step()
        if log:
            with torch.no_grad():
                for k, v in slices(model).items():
                    rec['upd'][k].append([(p - b).norm().item() for p, b in zip(v, before[k])])
            if not math.isfinite(rec['loss'][-1]):
                break
        if step % 500 == 0:
            print(f'{step:6d} loss {loss.item():.4f}  lr {lr:.2e}  {time.time() - t0:.0f}s', flush=True)
            out.write_text(json.dumps(rec))
    out.write_text(json.dumps(rec))
    print('wrote', out)
    return out


def plot(path):
    """Rows: training loss, activation norm, gradient norm, Adam update norm. Columns: tree level. One line per node."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rec = json.loads(Path(path).read_text())
    s = rec['step']
    rows = [('act', r'activation $\langle\|z\|\rangle_{\rm batch}$'), ('grad', r'gradient $\|\nabla\|_F$'),
            ('upd', r'Adam update $\|\Delta\|_F$'), ('param', r'parameter $\|\theta\|_F$')]
    titles = {'leaf': r'leaves $A_i$ / $z_i$', 'L1': r'level 1 $W^{(1)}_k$', 'L2': r'level 2 $W^{(2)}_k$',
              'L3': r'level 3 $W^{(3)}_k$', 'root': r'root $C$ / logits'}
    fig, ax = plt.subplots(len(rows) + 1, 5, figsize=(16, 13), sharex=True)
    for j, k in enumerate(LEVELS):
        ax[0, j].plot(s, rec['loss'], 'k-', lw=0.6)
        ax[0, j].axhline(math.log(16), color='gray', ls=':', lw=0.8)
        ax[0, j].set_title(titles[k], fontsize=10)
        for i, (q, lab) in enumerate(rows, start=1):
            vals = list(zip(*rec[q][k]))  # one series per node
            for v in vals:
                ax[i, j].plot(s[:len(v)], v, lw=0.6, alpha=0.8)
            ax[i, j].set_yscale('log')
            if j == 0:
                ax[i, 0].set_ylabel(lab)
    ax[0, 0].set_ylabel('train loss (batch)')
    for j in range(5):
        ax[-1, j].set_xlabel('update')
    c = rec['cfg']
    fig.suptitle(f"TreeTN r = {c['r']}, task {c['task']}, grammar {c['grammar']['seed']}, peak LR {rec['lr']:g} "
                 f"(cosine over 40,000); one line per node; dotted = log 16", fontsize=11)
    fig.tight_layout()
    dst = HERE / 'figures' / (Path(path).stem + '.pdf')
    fig.savefig(dst)
    fig.savefig(dst.with_suffix('.png'), dpi=110)
    print('wrote', dst)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='last')
    p.add_argument('--gseed', type=int, default=0)
    p.add_argument('--r', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--steps', type=int, default=10_000)
    p.add_argument('--every', type=int, default=10)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--plot', default=None)
    a = p.parse_args()
    if a.plot:
        plot(a.plot if Path(a.plot).is_absolute() else HERE / a.plot)
    else:
        run(a)  # plot separately (needs matplotlib, not torch): --plot <json>
