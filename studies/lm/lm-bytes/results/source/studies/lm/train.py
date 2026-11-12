"""Training and evaluation of one next-byte model; shared by the lm experiments.

A run config (cfg) is a flat dict:
  name, family, model (spec for models.build), seed (init; the batch stream uses seed + 10_000, so equal seeds see the
  same batches in every family), lr (peak), steps, batch, warmup, floor (final LR as a fraction of peak), clip
  (gradient-norm clip, 0 = off), train ('train' | 'pilot'), eval_every, n_curve (validation and train positions used
  for the curves), final ('full' or an int: positions for the final validation and test numbers), ckpt_steps,
  frozen (optional list of top-level parameter names that stay at their init, e.g. ['W'] for the TreeTN fusion cores).
Schedule: LR rises linearly from 0 to the peak over `warmup` updates, then follows a cosine from the peak down to
floor * peak at the last update. Adam (0.9, 0.999, eps 1e-8), no weight decay, fp32 (TF32 matmuls allowed on GPU).
Loss: cross-entropy of the next byte; bits/byte = CE / ln 2.

`train_run(cfg, out, deadline)` writes `<out>/runs/<name>.json` and returns the result. Status 'done', 'diverged'
(non-finite loss; the run stops there) or 'skipped' (the projected finish, including the final evaluation, falls after
the deadline; the partial curve is kept)."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from studies.lm import data as D
from studies.lm import models

LN2 = math.log(2.0)


def lr_at(step, cfg):
    """LR for update number `step` (0-based)."""
    peak, w, T = cfg['lr'], cfg['warmup'], cfg['steps']
    if step < w:
        return peak * (step + 1) / w
    frac = (step - w) / max(T - 1 - w, 1)
    return peak * (cfg['floor'] + (1 - cfg['floor']) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


@torch.no_grad()
def mean_ce(model, data, pos, bs):
    """Mean cross-entropy (nats) over target positions pos (long tensor on data's device)."""
    tot = 0.0
    for i in range(0, len(pos), bs):
        x, y = D.windows(data, pos[i:i + bs])
        tot += F.cross_entropy(model(x), y, reduction='sum').item()
    return tot / len(pos)


def device_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'


def train_run(cfg, out, deadline, root=D.DIR, ckpt_dir=None, log=print):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(out)
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    tr = torch.from_numpy(D.load(cfg['train'], root)).to(dev)
    va = torch.from_numpy(D.load('val', root)).to(dev)
    te = torch.from_numpy(D.load('test', root)).to(dev)
    curve_va = torch.from_numpy(D.subsample(va.numel(), cfg['n_curve'], seed=1)).to(dev)
    curve_tr = torch.from_numpy(D.subsample(tr.numel(), cfg['n_curve'], seed=2)).to(dev)
    final = cfg['final']
    fin_va = torch.from_numpy(D.positions(va.numel()) if final == 'full' else D.subsample(va.numel(), final, seed=3)).to(dev)
    fin_te = torch.from_numpy(D.positions(te.numel()) if final == 'full' else D.subsample(te.numel(), final, seed=4)).to(dev)

    model = models.build(cfg['model'], cfg['seed']).to(dev)
    # cfg['frozen']: top-level parameter names kept at their init (e.g. ['W'], every fusion core of a TreeTN).
    frozen = set(cfg.get('frozen') or [])
    for name, p in model.named_parameters():
        if name.split('.')[0] in frozen:
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                           **({'fused': True} if dev == 'cuda' else {'foreach': True}))
    gen = torch.Generator(device=dev).manual_seed(cfg['seed'] + 10_000)
    ebs = models.eval_batch(cfg['model'])
    steps, B = cfg['steps'], cfg['batch']
    ckpts = set(cfg.get('ckpt_steps') or [])

    res = dict(cfg=cfg, params=models.n_params(model), trainable_params=sum(p.numel() for p in params),
               device=device_name(), status='running',
               curve=dict(step=[], train_bits=[], val_bits=[], lr=[], clip_frac=[], grad_norm_max=[]))
    C = res['curve']

    def save():
        res['wall_s'] = round(time.time() - t_start, 1)
        tmp = out / 'runs' / f".{cfg['name']}.json.tmp"
        json.dump(res, open(tmp, 'w'), indent=1)
        tmp.replace(out / 'runs' / f"{cfg['name']}.json")

    def checkpoint(step):
        if ckpt_dir is not None and step in ckpts:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, Path(ckpt_dir) / f'step_{step:06d}.pt')

    def sync():
        if dev == 'cuda':
            torch.cuda.synchronize()

    eval_s_per_pos, train_s = None, 0.0
    loss_sum = torch.zeros((), device=dev)
    n_clip = torch.zeros((), device=dev)
    gmax = torch.zeros((), device=dev)
    n_int = 0
    checkpoint(0)
    sync()
    t_mark = time.time()
    probe = min(200, max(steps // 10, 1))
    step = 0
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step - 1, cfg)
        t = torch.randint(D.CTX, tr.numel(), (B,), device=dev, generator=gen)
        x, y = D.windows(tr, t)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg['clip'] > 0:
            gn = torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        else:
            gn = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]))
        n_clip += (gn > cfg['clip']).float() if cfg['clip'] > 0 else 0.0
        gmax = torch.maximum(gmax, gn.detach())
        opt.step()
        loss_sum += loss.detach()
        n_int += 1
        checkpoint(step)
        at_eval = step % cfg['eval_every'] == 0 or step == steps
        if not (at_eval or step == probe):
            continue
        sync()
        train_s += time.time() - t_mark
        res['ms_per_step'] = round(1000 * train_s / step, 3)
        if at_eval:
            tl = loss_sum.item() / n_int / LN2
            te0 = time.time()
            model.eval()
            vb = mean_ce(model, va, curve_va, ebs) / LN2
            model.train()
            sync()
            eval_s_per_pos = (time.time() - te0) / len(curve_va)
            C['step'].append(step), C['train_bits'].append(tl), C['val_bits'].append(vb)
            C['lr'].append(lr_at(step - 1, cfg)), C['clip_frac'].append(n_clip.item() / n_int)
            C['grad_norm_max'].append(gmax.item())
            loss_sum.zero_(), n_clip.zero_(), gmax.zero_()
            n_int = 0
            if not (math.isfinite(tl) and math.isfinite(vb)):
                res.update(status='diverged', stopped_at=step)
                log(f"{cfg['name']}: diverged at step {step}", flush=True)
                save()
                return res
            if step % (10 * cfg['eval_every']) == 0:
                log(f"{cfg['name']}: step {step} train {tl:.3f} val {vb:.3f} bits/byte, "
                    f"{res['ms_per_step']:.1f} ms/step", flush=True)
        # Deadline: the rest of training at the measured speed plus the final evaluation (~ curve-eval speed).
        eval_rate = eval_s_per_pos if eval_s_per_pos is not None else 1e-6
        need = train_s / step * (steps - step) + eval_rate * (len(fin_va) + len(fin_te) + len(curve_tr)) * 1.2 + 10
        if time.time() + need > deadline:
            res.update(status='skipped', stopped_at=step,
                       reason=f'projected finish {need / 60:.1f} min after step {step} is past the deadline')
            log(f"{cfg['name']}: skipped at step {step} ({res['reason']})", flush=True)
            save()
            return res
        sync()
        t_mark = time.time()

    model.eval()
    t0 = time.time()
    res['val_bits'] = mean_ce(model, va, fin_va, ebs) / LN2
    res['test_bits'] = mean_ce(model, te, fin_te, ebs) / LN2
    res['train_bits'] = mean_ce(model, tr, curve_tr, ebs) / LN2
    res['eval_s'] = round(time.time() - t0, 1)
    res.update(status='done', targets_seen=steps * B, n_val=len(fin_va), n_test=len(fin_te), n_train_eval=len(curve_tr))
    if not (math.isfinite(res['val_bits']) and math.isfinite(res['test_bits'])):
        res['status'] = 'diverged'
    log(f"{cfg['name']}: {res['status']} val {res['val_bits']:.4f} test {res['test_bits']:.4f} bits/byte "
        f"({res['params']} params, {res['ms_per_step']:.1f} ms/step)", flush=True)
    save()
    return res


def ngram_run(name, out, train_bytes, final, n_curve, root=D.DIR, max_order=8, log=print):
    """Kneser-Ney orders 1..max_order counted on the first `train_bytes` bytes of train.bin (None = all of it); scored
    on the same final validation/test positions as the neural models and on the train curve positions."""
    from studies.lm import ngram
    out = Path(out)
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr = D.load('train', root)
    if train_bytes is not None:
        tr = tr[:train_bytes]
    va, te = D.load('val', root), D.load('test', root)
    fin_va = D.positions(va.size) if final == 'full' else D.subsample(va.size, final, seed=3)
    fin_te = D.positions(te.size) if final == 'full' else D.subsample(te.size, final, seed=4)
    kn = ngram.KneserNey(max_order, log=log).fit(tr)
    t_fit = time.time() - t0
    res = dict(cfg=dict(name=name, family='ngram', model=dict(kind='ngram', smoothing='interpolated modified Kneser-Ney',
                                                              max_order=max_order),
                        train='train', train_bytes=int(tr.size), final=final),
               status='done', orders=list(range(1, max_order + 1)), targets_seen=int(tr.size),
               val_bits=kn.bits(va, fin_va).tolist(), test_bits=kn.bits(te, fin_te).tolist(),
               train_bits=kn.bits(tr, D.subsample(tr.size, n_curve, seed=2)).tolist(),
               n_val=len(fin_va), n_test=len(fin_te), fit_s=round(t_fit, 1), wall_s=round(time.time() - t0, 1),
               device='cpu', **kn.summary())
    json.dump(res, open(out / 'runs' / f'{name}.json', 'w'), indent=1)
    log(f"{name}: test bits/byte by order {np.round(res['test_bits'], 3).tolist()} ({res['wall_s']}s)", flush=True)
    return res
