"""Training driver for the language-process experiments: one run per spec, an LR pilot, and the shared CLI.

It reuses studies/rhm/train.py for the optimizer, LR schedule, logging and checkpoint schedule (T.train), the
process-pool runner (T.run_all) and the CLI (T.args); only the data (studies/lang/processes.py) and the evaluation
sets differ. A spec is dict(name, cfg, lr, steps, data_seed, schedule_total=None, ckpts=True, extra_ckpts=(),
subdir=''). Each run writes <out>/<subdir>/<name>/result.json and its checkpoints."""
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

from studies.lang import common as L
from studies.rhm import train as T

ROOT = Path(__file__).resolve().parents[2]
DENSE_EVERY, DENSE_UNTIL = 10, 2000
_SETS = {}


def eval_sets(spec, a):
    k = (json.dumps(spec, sort_keys=True), a.n_diag, a.n_test)
    if k not in _SETS:
        _SETS[k] = dict(diag=L.fixed_set(spec, a.n_diag, L.DIAG_SEED), test=L.fixed_set(spec, a.n_test, L.TEST_SEED))
    return _SETS[k]


@torch.no_grad()
def by_stratum(model, x, y, pb, s, chunk=8192):
    """Accuracy, cross-entropy and KL(Bayes || model) per stratum value, with the Bayes accuracy and loss there."""
    if s is None:
        return None
    lq = torch.cat([torch.log_softmax(model(x[i:i + chunk]).float(), -1) for i in range(0, len(y), chunk)])
    out = {}
    for v in sorted(set(s.tolist())):
        m = s == v
        p, l, yy = pb[m], lq[m], y[m]
        out[str(v)] = dict(n=int(m.sum()), acc=(l.argmax(1) == yy).float().mean().item(),
                           loss=-l.gather(1, yy[:, None]).mean().item(),
                           kl=(torch.xlogy(p, p) - p * l).sum(1).mean().item(),
                           bayes_acc=(p.argmax(1) == yy).float().mean().item(),
                           bayes_loss=-torch.log(p.gather(1, yy[:, None]).clamp_min(1e-30)).mean().item())
    return out


def run(a, spec):
    cfg = spec['cfg']
    run_dir = Path(a.out) / spec.get('subdir', '') / spec['name']
    run_dir.mkdir(parents=True, exist_ok=True)
    sets = eval_sets(cfg['process'], a)
    dev = a.device
    torch.manual_seed(0)
    model = L.build(cfg).to(dev)
    proc = L.process(cfg).to(dev)
    gen = torch.Generator(dev).manual_seed(spec['data_seed'])
    batch_fn = lambda step: proc.sample_xy(T.BATCH, gen)
    xd, yd, pd, _ = sets['diag']
    evals = {'diag': (xd.to(dev), yd.to(dev), pd.to(dev))}
    unb = None
    if cfg['process'].get('bias'):  # parity with input bias: also log on unbiased inputs (see analysis.t_thresh)
        xu, yu, pu, _ = L.fixed_set(dict(cfg['process'], bias=0.0), a.n_diag, L.DIAG_SEED)
        evals['unb'] = unb = (xu.to(dev), yu.to(dev), pu.to(dev))
    t = time.time()
    # logging: the shared log-spaced schedule plus every DENSE_EVERY updates up to DENSE_UNTIL, since several
    # processes are learned within a few hundred updates (t would otherwise sit at the 50-update resolution)
    log_at = set(T.log_steps(spec['steps'])) | set(range(0, min(spec['steps'], DENSE_UNTIL) + 1, DENSE_EVERY))
    log, status, steps, ms = T.train(model, cfg, spec['lr'], spec['steps'], batch_fn, evals, run_dir=run_dir,
                                     deadline=a.deadline, ckpts=spec.get('ckpts', True),
                                     schedule_total=spec.get('schedule_total'), log_at=sorted(log_at),
                                     extra_ckpts=spec.get('extra_ckpts', ()))
    xt, yt, pt, st = (v.to(dev) if v is not None else None for v in sets['test'])
    res = dict(name=spec['name'], cfg=cfg, lr=spec['lr'], steps_planned=spec['steps'], steps_done=steps,
               status=status, data_seed=spec['data_seed'], schedule_total=spec.get('schedule_total') or spec['steps'],
               batch=T.BATCH, warmup=T.WARMUP, lr_floor=T.LR_FLOOR,
               n_params=sum(p.numel() for p in model.parameters()), ms_per_step=ms, seconds=time.time() - t,
               device=T.device_note(dev), n_diag=len(yd), n_test=len(yt),
               bayes_diag=T.bayes_ref('lang', evals['diag'][1], evals['diag'][2]),
               bayes_unb=T.bayes_ref('lang', unb[1], unb[2]) if unb else None,
               initial_diag_loss=log['diag_loss'][0] if log['diag_loss'] else None,
               bayes_test=T.bayes_ref('lang', yt, pt), test=T.evaluate(model, xt, yt, pt, 'lang'),
               strata=by_stratum(model, xt, yt, pt, st), log=log,
               spec={k: v for k, v in spec.items() if k != 'cfg'})
    T.dump(run_dir / 'result.json', res)
    print(f"{spec['name']}: {status} at {steps} updates, {ms:.2f} ms/update, test loss {res['test']['loss']:.4f} "
          f"acc {res['test']['acc']:.4f} kl {res['test']['kl']:.4f} (Bayes {res['bayes_test']['loss']:.4f} / "
          f"{res['bayes_test']['acc']:.4f})", flush=True)
    return status


def pilot(a, base, steps, schedule_total, lrs=T.LRS):
    """Peak LR per key by the lowest diagnostic loss after `steps` updates of the real schedule. An LR whose run
    collapsed (see `collapsed`) is excluded, since the r = 64 RHM runs learned and then fell back to uniform output. base: {key: spec without lr, on an init seed no evaluation run uses}; lrs may be a dict {key: grid}.
    --lr overrides apply. Saved to <out>/lr.json. Returns {key: lr}."""
    path = Path(a.out) / 'lr.json'
    saved = json.load(open(path)) if path.exists() else {}
    over = getattr(a, 'lr_over', {}) or {}
    for k in base:
        if k in over or '*' in over:
            saved[k] = dict(lr=over.get(k, over.get('*')), override=True)
    grid = lambda k: lrs[k] if isinstance(lrs, dict) else lrs
    todo = {k: s for k, s in base.items() if k not in saved}
    specs = [dict(s, name=f'{k}_lr{lr:g}', lr=lr, steps=steps, schedule_total=schedule_total, ckpts=False,
                  subdir='pilot') for k, s in todo.items() for lr in grid(k)]
    if specs:
        T.run_all(a, specs, run, deadline=None, planned=False)
    for k in todo:
        loss, unstable = {}, {}
        O = L.process(base[k]['cfg']).O
        for lr in grid(k):
            f = Path(a.out) / 'pilot' / f'{k}_lr{lr:g}' / 'result.json'
            r = json.load(open(f)) if f.exists() else None
            ok = r is not None and r['status'] == 'finished' and math.isfinite(r['log']['diag_loss'][-1])
            if r is None or r['status'] != 'finished':
                # a pilot run that crashed (e.g. out of GPU memory) must not silently drop its LR from the grid
                sys.exit(f'pilot {k} at lr {lr:g} did not finish ({"no result" if r is None else r["status"]}); '
                         f'rerun with fewer --procs')
            unstable[f'{lr:g}'] = ok and collapsed(r['log']['diag_loss'], O)
            loss[f'{lr:g}'] = r['log']['diag_loss'][-1] if ok and not unstable[f'{lr:g}'] else float('inf')
        if not any(math.isfinite(v) for v in loss.values()):
            sys.exit(f'no usable pilot loss for {k}: pass --lr or copy lr.json into --out')
        choice = min(grid(k), key=lambda lr: loss[f'{lr:g}'])
        saved[k] = dict(lr=choice, diag_loss=loss, unstable=unstable, steps=steps, schedule_total=schedule_total,
                        grid=list(grid(k)), at_grid_edge=choice in (min(grid(k)), max(grid(k))),
                        seed=base[k]['cfg']['seed'])
        print(f'pilot {k}: diag loss {loss} -> lr {choice:g}' + (' (grid edge)' if saved[k]['at_grid_edge'] else ''),
              flush=True)
    T.dump(path, saved)
    return {k: saved[k]['lr'] for k in base}


def collapsed(diag_loss, O):
    """Learned, then fell back: some logged diagnostic loss is below max(log(O) / 2, log(O) - 0.5) (a run counts as
    having learned even when the Bayes loss is high, e.g. multi-match induction at 0.87 nats with log 8 = 2.08), and
    a later one is above log(O) - 0.1 (within 0.1 nats of the uniform-output loss)."""
    u = math.log(O)
    first = next((i for i, v in enumerate(diag_loss) if v < max(u / 2, u - 0.5)), None)
    return first is not None and any(v > u - 0.1 for v in diag_loss[first:])


def args(hours, extra=None):
    """studies.rhm.train.args, plus copies of studies/lang/*.py into <out>/source/lang_*.py."""
    a = T.args(hours=hours, extra=extra)
    a.n_diag, a.n_test = (2000, 2000) if a.smoke else (L.N_DIAG, L.N_TEST)
    for f in (ROOT / 'studies/lang').glob('*.py'):
        shutil.copy(f, Path(a.out) / 'source' / f'lang_{f.name}')
    return a


def spec(name, cfg, steps, data_seed, lr=None, **kw):
    return dict(name=name, cfg=dict(cfg, task='lang'), lr=lr, steps=steps, data_seed=data_seed, **kw)


def ckpt_list(total, n=12):
    """Checkpoint steps for probing through training: 0 and n log-spaced steps up to `total`."""
    import numpy as np
    return sorted({0, total, *np.unique(np.round(np.logspace(1.5, np.log10(total), n)).astype(int)).tolist()})
