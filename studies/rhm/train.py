"""Shared training code for the RHM experiments: data, evaluation, the training loop, the LR pilot and a
deadline-aware runner.

Training. Cross-entropy, Adam, batch 256, float32. Learning rate: linear warmup over the first 100 updates, then
cosine decay to 0.1x the peak at the run's last update (`lr_at`). Data is either online (256 fresh trees per update,
sampled on the device from the grammar with a seeded torch generator) or a fixed training set of N trees (batches
drawn uniformly with replacement).

Evaluation. Every run logs against the fixed diagnostic set of its grammar (`common.diag_trees`, 20,000 trees,
seeded on the CPU) at a log-spaced schedule plus every 50 updates up to 3,000, every 1,000 updates and every
checkpoint, and reports the separate
50,000-tree test set (`common.test_trees`) at the end. Metrics are mean cross-entropy (nats), accuracy, and the KL
from the Bayes posterior, E_x KL(p_Bayes(.|x) || q(.|x)). For 'root' the Bayes posterior is a point mass on the true
root (the grammar is unambiguous), so its loss is 0, its accuracy 1, and KL = cross-entropy. For 'last' it is
`grammar.posterior_token` at leaf 15, computed exactly in float64 once per grammar and saved in <out>/data/.

Layout of an output folder <out>:
  <out>/<run name>/result.json   config, LR, status, log (per log step), final test metrics, timing
  <out>/<run name>/ckpt_<step>.pt   `common.save` checkpoints at CHECKPOINTS (scaled to the run's length)
  <out>/data/diag_<key>.pt        the diagnostic trees of each grammar (= common.diag_trees; key e.g. m4_g0)
  <out>/data/bayes_<key>.pt       the Bayes posteriors of the last token on the diagnostic and test sets
  <out>/lr.json                   the LR pilot: loss per candidate and the choice
  <out>/status.json               every run's status: finished, early_stopped, truncated (deadline), diverged,
                                  skipped (deadline passed before it started), failed (exception)
  <out>/source/                   the code that produced the results
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from studies.rhm import common as C
from studies.rhm import grammar as G

ROOT = Path(__file__).resolve().parents[2]
BATCH, WARMUP, LR_FLOOR = 256, 100, 0.1
LRS = (1e-4, 3e-4, 1e-3, 3e-3)
PILOT_SEED = 9  # LR pilots run on this init seed, which no evaluation run uses
TRAIN_SEED, ONLINE_SEED = 40_000, 7_000_000
N_TRAIN_MAX = 2 ** 20


def gkey(g):
    return f"m{g['m']}_g{g['seed']}"


def grammar_cfg(seed, **over):
    return dict(C.GRAMMAR, seed=seed, **over)


# ----------------------------------------------------------------------------------------------- schedules

def lr_at(step, peak, total, warmup=WARMUP):
    """Linear warmup to `peak` over `warmup` updates, then cosine to LR_FLOOR * peak at update `total`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    u = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * u)))


def ckpt_steps(total):
    """common.CHECKPOINTS scaled to a run of `total` updates (identical for 40,000)."""
    base = C.CHECKPOINTS[-1]
    return sorted({int(round(c * total / base)) for c in C.CHECKPOINTS})


def log_steps(total, n=60, every=1000, dense=50, dense_until=3000):
    """Log-spaced steps, plus every `dense` updates up to `dense_until` (where the transitions happen, so steps to
    threshold are resolved to 50 updates), every `every` updates, and the checkpoints."""
    s = set(np.unique(np.round(np.logspace(0, np.log10(max(total, 1)), n)).astype(int)).tolist())
    s |= set(range(0, min(total, dense_until) + 1, dense))
    s |= set(range(0, total + 1, every)) | set(ckpt_steps(total)) | {0, total}
    return sorted(x for x in s if 0 <= x <= total)


# ------------------------------------------------------------------------------------------------ data

def bayes_last(tokens, rules, v, device, chunk=2048):
    """P(token 15 | tokens 0..14), (B, v) float32 on the CPU, exact (float64 upward passes)."""
    R = {l: r.to(device) for l, r in rules.items()}
    out = [G.posterior_token(tokens[i:i + chunk].to(device), R, v, tokens.shape[1] - 1).float().cpu()
           for i in range(0, len(tokens), chunk)]
    return torch.cat(out)


def prepare(a, gcfgs):
    """Write, for each grammar, <out>/data/diag_<key>.pt (the diagnostic trees: a plain trees dict, identical to
    common.diag_trees(gcfg)) and <out>/data/bayes_<key>.pt (the Bayes posteriors of the last token on the diagnostic and
    test sets). Run once in the main process before the runs start."""
    d = Path(a.out) / 'data'
    d.mkdir(parents=True, exist_ok=True)
    for g in gcfgs:
        k = gkey(g)
        if (d / f'bayes_{k}.pt').exists() and torch.load(d / f'bayes_{k}.pt')['n_diag'] == a.n_diag:
            continue
        t = time.time()
        rules = C.rules(g)
        diag, test = C.diag_trees(g, a.n_diag), C.test_trees(g, a.n_test)
        torch.save(diag, d / f'diag_{k}.pt')
        torch.save(dict(gcfg=g, n_diag=a.n_diag, n_test=a.n_test, diag_seed=C.DIAG_SEED + g['seed'],
                        test_seed=C.TEST_SEED + g['seed'],
                        bayes_diag=bayes_last(diag[g['L']], rules, g['v'], a.device),
                        bayes_test=bayes_last(test[g['L']], rules, g['v'], a.device)), d / f'bayes_{k}.pt')
        print(f'prepared data for {k} in {time.time() - t:.0f}s', flush=True)


_BUNDLES = {}


def bundle(a, g):
    """Per-process cache of a grammar's rules (on the device), eval trees and Bayes posteriors."""
    k = gkey(g)
    if k not in _BUNDLES:
        saved = torch.load(Path(a.out) / 'data' / f'bayes_{k}.pt')
        diag = torch.load(Path(a.out) / 'data' / f'diag_{k}.pt')
        _BUNDLES[k] = dict(rules={l: r.to(a.device) for l, r in C.rules(g).items()}, diag=diag,
                           test=C.test_trees(g, saved['n_test']), bayes_diag=saved['bayes_diag'],
                           bayes_test=saved['bayes_test'])
    return _BUNDLES[k]


def evalset(trees, cfg, device, bayes=None):
    """(x, y, p_Bayes or None) on the device; p_Bayes is used only for 'last' (for 'root' it is the point mass)."""
    x, y = C.task_batch(trees, cfg)
    pb = bayes.to(device) if (bayes is not None and cfg['task'] == 'last') else None
    return x.to(device), y.to(device), pb


_TRAIN_SETS = {}


def train_set(g, N):
    """The first N trees of a fixed 2^20-tree training set per grammar (CPU, seeded): smaller sets are prefixes of
    larger ones, so training sets are nested across N. Only levels 0 (root) and L (tokens) are kept."""
    k = gkey(g)
    if k not in _TRAIN_SETS:
        t = G.sample_trees(N_TRAIN_MAX, C.rules(g), seed=TRAIN_SEED + g['seed'])
        _TRAIN_SETS[k] = {0: t[0], g['L']: t[g['L']]}
    t = _TRAIN_SETS[k]
    return {0: t[0][:N], g['L']: t[g['L']][:N]}


def online_batches(cfg, rules, device, seed, sampler=None):
    """batch_fn for online training: fresh trees every update from a device generator seeded with `seed`."""
    gen = torch.Generator(device).manual_seed(seed)
    sampler = sampler or (lambda B, g: G.sample_trees(B, rules, g=g, device=device))
    return lambda step: C.task_batch(sampler(BATCH, gen), cfg)


def fixed_batches(x, y, device, seed):
    gen = torch.Generator(device).manual_seed(seed)
    return lambda step: (lambda i: (x[i], y[i]))(torch.randint(0, len(y), (BATCH,), generator=gen, device=device))


# ------------------------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, x, y, pb=None, task='root', chunk=8192):
    """Mean cross-entropy, accuracy and KL(p_Bayes || q) (nan when no Bayes posterior applies)."""
    was = model.training
    model.eval()
    ce = acc = kl = 0.0
    for i in range(0, len(y), chunk):
        lq = F.log_softmax(model(x[i:i + chunk]).float(), -1)
        yy = y[i:i + chunk]
        ce += -lq.gather(1, yy[:, None]).sum().item()
        acc += (lq.argmax(1) == yy).sum().item()
        if pb is not None:
            p = pb[i:i + chunk]
            kl += (torch.xlogy(p, p) - p * lq).sum().item()
    model.train(was)
    n = len(y)
    kl = ce / n if task == 'root' else (kl / n if pb is not None else float('nan'))
    return dict(loss=ce / n, acc=acc / n, kl=kl)


def bayes_ref(task, y, pb):
    """Cross-entropy and accuracy of the Bayes predictor on (y, p_Bayes)."""
    if task == 'root':
        return dict(loss=0.0, acc=1.0)
    lp = torch.log(pb.gather(1, y[:, None]).clamp_min(1e-30))
    return dict(loss=-lp.mean().item(), acc=(pb.argmax(1) == y).float().mean().item())


# ----------------------------------------------------------------------------------------------- training

def make_opt(params, lr):
    return torch.optim.Adam(params, lr=lr)


def train(model, cfg, peak_lr, total, batch_fn, evals, run_dir=None, deadline=None, params=None, on_eval=None,
          ckpts=True, schedule_total=None, log_at=None, extra_ckpts=()):
    """Train `model` for `total` updates. batch_fn(step) -> (x, y) on the device. evals: {name: (x, y, p_Bayes)},
    evaluated at the log steps (diagnostics; never used for gradients). Checkpoints go to run_dir. on_eval(step, log)
    may return True to stop early. Only `params` (default: those with requires_grad) are given to Adam.
    Returns (log, status, steps done, ms per update)."""
    params = [p for p in model.parameters() if p.requires_grad] if params is None else params
    opt = make_opt(params, peak_lr)
    sched_T = schedule_total or total
    log_at = set(log_at if log_at is not None else log_steps(total))
    # checkpoints: common.CHECKPOINTS scaled to the schedule's length, those within this run, the extras and the end
    ck = ({c for c in ckpt_steps(sched_T) if c <= total} | set(extra_ckpts) | {total}
          if (ckpts and run_dir is not None) else set())
    log_at |= ck
    log = dict(step=[], lr=[], train_loss=[], **{f'{k}_{m}': [] for k in evals for m in ('loss', 'acc', 'kl')})
    recent, status, step = [], 'finished', 0
    sync = torch.cuda.synchronize if next(model.parameters()).is_cuda else (lambda: None)
    t_eval, t0 = 0.0, time.time()

    def do_eval(step):
        nonlocal t_eval
        sync()
        te = time.time()
        log['step'].append(step)
        log['lr'].append(lr_at(step, peak_lr, sched_T))
        log['train_loss'].append(torch.stack(recent).mean().item() if recent else float('nan'))
        recent.clear()
        for k, (x, y, pb) in evals.items():
            for m, val in evaluate(model, x, y, pb, cfg['task']).items():
                log[f'{k}_{m}'].append(val)
        t_eval += time.time() - te
        return on_eval(step, log) if on_eval else False

    for step in range(total + 1):
        stop = False
        if step in log_at or step == total:
            stop = do_eval(step)
            if not math.isfinite(log['train_loss'][-1]) and step > 0:
                status = 'diverged'
                break
        if step in ck or (stop and run_dir is not None):
            C.save(Path(run_dir) / f'ckpt_{step}.pt', model, cfg, step)
        if stop:
            status = 'early_stopped'
            break
        if step == total:
            break
        if deadline and step % 50 == 0 and time.time() > deadline:
            status = 'truncated'
            if log['step'][-1] != step:
                do_eval(step)
            if run_dir is not None:
                C.save(Path(run_dir) / f'ckpt_{step}.pt', model, cfg, step)
            break
        lr = lr_at(step, peak_lr, sched_T)
        for gr in opt.param_groups:
            gr['lr'] = lr
        x, y = batch_fn(step)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        recent.append(loss.detach())
    sync()
    ms = (time.time() - t0 - t_eval) / max(step, 1) * 1e3
    return log, status, step, ms


# ----------------------------------------------------------------------------------------------- one run

def run_standard(a, spec):
    """One run from a spec: dict(name, cfg, lr, steps, data='online' | N, data_seed, schedule_total=None,
    ckpts=True, extra_ckpts=(), early_stop=None | dict(patience, train_tol), subdir=''). Writes <out>/<subdir>/<name>/result.json."""
    cfg, g = spec['cfg'], spec['cfg']['grammar']
    run_dir = Path(a.out) / spec.get('subdir', '') / spec['name']
    run_dir.mkdir(parents=True, exist_ok=True)
    b = bundle(a, g)
    torch.manual_seed(0)
    model = C.build(cfg).to(a.device)
    evals = {'diag': evalset(b['diag'], cfg, a.device, b['bayes_diag'])}
    data = spec.get('data', 'online')
    if data == 'online':
        batch_fn = online_batches(cfg, b['rules'], a.device, spec['data_seed'])
    else:
        trees = train_set(g, data)
        xt, yt = C.task_batch(trees, cfg)
        xt, yt = xt.to(a.device), yt.to(a.device)
        batch_fn = fixed_batches(xt, yt, a.device, spec['data_seed'])
        k = min(data, a.n_diag)
        # the training set itself (its first k trees), with the Bayes posterior so 'last' gets a KL to Bayes on it
        pb_fit = bayes_last(trees[g['L']][:k], C.rules(g), g['v'], a.device) if cfg['task'] == 'last' else None
        evals['fit'] = evalset({0: trees[0][:k], g['L']: trees[g['L']][:k]}, cfg, a.device, pb_fit)
    best = dict(loss=float('inf'), step=0, state=None)  # lowest diagnostic loss (for loss and KL)
    best_acc = dict(acc=-1.0, step=0, state=None)  # highest diagnostic accuracy (for accuracy-based tests)
    es = spec.get('early_stop')

    def on_eval(step, log):
        if data != 'online':
            if log['diag_loss'][-1] < best['loss']:
                best.update(loss=log['diag_loss'][-1], step=step, state=copy.deepcopy(model.state_dict()))
            if log['diag_acc'][-1] > best_acc['acc']:
                best_acc.update(acc=log['diag_acc'][-1], step=step, state=copy.deepcopy(model.state_dict()))
        if es is None:
            return False
        tl = log['fit_loss'][-1] if 'fit' in evals else log['train_loss'][-1]
        return step - best['step'] >= es['patience'] and tl < es['train_tol']

    t = time.time()
    log, status, steps, ms = train(model, cfg, spec['lr'], spec['steps'], batch_fn, evals, run_dir=run_dir,
                                   deadline=a.deadline, on_eval=on_eval, ckpts=spec.get('ckpts', True),
                                   schedule_total=spec.get('schedule_total'), extra_ckpts=spec.get('extra_ckpts', ()))
    xs, ys, ps = evalset(b['test'], cfg, a.device, b['bayes_test'])
    res = dict(name=spec['name'], cfg=cfg, lr=spec['lr'], steps_planned=spec['steps'], steps_done=steps, status=status,
               data=data, data_seed=spec['data_seed'], schedule_total=spec.get('schedule_total') or spec['steps'],
               batch=BATCH, warmup=WARMUP, lr_floor=LR_FLOOR, n_params=sum(p.numel() for p in model.parameters()),
               ms_per_step=ms, seconds=time.time() - t, device=device_note(a.device),
               diag_file=str(Path('data') / f'diag_{gkey(g)}.pt'), n_diag=len(evals['diag'][1]), n_test=len(ys),
               bayes_diag=bayes_ref(cfg['task'], evals['diag'][1], evals['diag'][2]),
               bayes_test=bayes_ref(cfg['task'], ys, ps),
               # Bayes cross-entropy on the fit subset of a fixed training set: fit is judged by the excess training
               # loss over it (a memorizing model has excess <= 0; its KL to Bayes is large, so KL can't measure fit)
               bayes_fit=bayes_ref(cfg['task'], evals['fit'][1], evals['fit'][2]) if 'fit' in evals else None,
               test=evaluate(model, xs, ys, ps, cfg['task']), log=log, spec={k: v for k, v in spec.items() if k != 'cfg'})
    for key, sel in (('best', best), ('best_acc', best_acc)):
        if sel['state'] is not None and data != 'online':
            # early-stopping selections: lowest diagnostic loss ('best') and highest diagnostic accuracy ('best_acc')
            model.load_state_dict(sel['state'])
            res[key] = dict(step=sel['step'], test=evaluate(model, xs, ys, ps, cfg['task']),
                            fit=evaluate(model, *evals['fit'], cfg['task']),
                            diag=evaluate(model, *evals['diag'], cfg['task']))
    dump(run_dir / 'result.json', res)
    print(f"{spec['name']}: {status} at {steps} updates, {ms:.2f} ms/update, test loss {res['test']['loss']:.4f} "
          f"acc {res['test']['acc']:.4f} (Bayes {res['bayes_test']['loss']:.4f} / {res['bayes_test']['acc']:.4f})",
          flush=True)
    return status


def dump(path, obj):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    os.replace(tmp, path)


def device_note(device):
    return torch.cuda.get_device_name() if str(device).startswith('cuda') else 'cpu'


# ------------------------------------------------------------------------------------------------ LR pilot

def pilot(a, base, steps, schedule_total, lrs=LRS):
    """Choose the peak LR per key by diagnostic loss after `steps` updates of a run whose schedule is
    `schedule_total` long (so the pilot sees the real warmup and the start of the real decay).
    base: {key: spec without lr}; callers put the pilot on init seed PILOT_SEED, which no evaluation run uses.
    `--lr` overrides (a.lr_over: {key or '*': lr}) replace the pilot for those keys. Results, and whether the choice
    sits at an edge of the grid, are saved in <out>/lr.json (reused by reruns into the same --out). Returns {key: lr}."""
    path = Path(a.out) / 'lr.json'
    saved = json.load(open(path)) if path.exists() else {}
    over = getattr(a, 'lr_over', {}) or {}
    for k in base:
        if k in over or '*' in over:
            saved[k] = dict(lr=over.get(k, over.get('*')), override=True)
    todo = {k: s for k, s in base.items() if k not in saved}
    specs = [dict(s, name=f'{k}_lr{lr:g}', lr=lr, steps=steps, schedule_total=schedule_total, ckpts=False,
                  subdir='pilot') for k, s in todo.items() for lr in lrs]
    if specs:
        run_all(a, specs, run_standard, deadline=None, planned=False)
    for k in todo:
        loss = {}
        for lr in lrs:
            f = Path(a.out) / 'pilot' / f'{k}_lr{lr:g}' / 'result.json'
            r = json.load(open(f)) if f.exists() else None
            ok = r is not None and r['status'] in ('finished',) and math.isfinite(r['log']['diag_loss'][-1])
            loss[f'{lr:g}'] = r['log']['diag_loss'][-1] if ok else float('inf')
        if not any(math.isfinite(v) for v in loss.values()):
            # e.g. a rerun with --only into a fresh --out (pilots filtered out, no lr.json): never fall back silently
            sys.exit(f'no finite pilot loss for {k}: copy the original lr.json into --out, or pass --lr')
        choice = min(lrs, key=lambda lr: loss[f'{lr:g}'])
        saved[k] = dict(lr=choice, diag_loss=loss, steps=steps, schedule_total=schedule_total, grid=list(lrs),
                        at_grid_edge=choice in (min(lrs), max(lrs)), seed=base[k]['cfg']['seed'],
                        ms_per_step={f'{lr:g}': _ms(Path(a.out) / 'pilot' / f'{k}_lr{lr:g}') for lr in lrs})
        print(f'pilot {k}: diag loss {loss} -> lr {choice:g}' + (' (grid edge)' if saved[k]['at_grid_edge'] else ''),
              flush=True)
    dump(path, saved)
    return {k: saved[k]['lr'] for k in base}


def _ms(run_dir):
    f = Path(run_dir) / 'result.json'
    return json.load(open(f))['ms_per_step'] if f.exists() else None


def start_clock(a):
    """Start the training deadline now (after setup and the LR pilots) and record when it started."""
    a.deadline = time.time() + 3600 * a.hours
    dump(Path(a.out) / 'clock.json', dict(start=time.time(), hours=a.hours, deadline=a.deadline,
                                          setup_seconds=time.time() - a.t_launch))
    print(f'training deadline in {a.hours:g} h (setup and pilots took {time.time() - a.t_launch:.0f} s)', flush=True)


# ------------------------------------------------------------------------------------------------- runner

def args(hours=8.0, extra=None):
    """Command-line knobs every run.py shares. Copies the code that produces the results into <out>/source."""
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--hours', type=float, default=hours,
                   help='wall-clock deadline for the main grid, counted from the end of setup and the LR pilots')
    p.add_argument('--lr', default='', help="override the piloted peak LR: '3e-4' for every key, or per pilot key as written in lr.json (e.g. 'root=3e-4,last=1e-3' in rhm-dynamics, 'root_chain=1e-4' in rhm-controls)"
                   " (use with a separate --out for reruns)")
    p.add_argument('--smoke', action='store_true', help='tiny CPU test: few runs, few updates, small eval sets')
    p.add_argument('--procs', type=int, default=0, help='concurrent run processes (default: 6 on a GPU, 1 on CPU)')
    p.add_argument('--only', default='', help='comma-separated substrings: run only matching run names (reruns)')
    p.add_argument('--allow-cpu', action='store_true', help='run the full grid without a GPU (local tests only)')
    for f in (extra or []):
        f(p)
    a = p.parse_args()
    a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.device == 'cpu' and not (a.smoke or a.allow_cpu):
        raise SystemExit('no CUDA device: refusing to run the full grid on CPU (a driver or torch build problem)')
    a.t_launch, a.deadline = time.time(), None  # the deadline starts with start_clock(a), after setup and pilots
    a.lr_over = {kv.split('=')[0] if '=' in kv else '*': float(kv.split('=')[-1]) for kv in a.lr.split(',') if kv}
    a.n_diag, a.n_test = (2000, 2000) if a.smoke else (C.N_DIAG, C.N_TEST)
    a.procs = a.procs or (1 if a.device == 'cpu' else 6)
    out = Path(a.out)
    (out / 'source').mkdir(parents=True, exist_ok=True)
    script = Path(sys.argv[0]).resolve()
    for f in [ROOT / 'ctn/tree.py', *(ROOT / 'studies/rhm').glob('*.py'), *script.parent.glob('*.py'), ROOT / 'uv.lock']:
        dst = out / 'source' / (f.name if f.parent.name in ('rhm', 'ctn') or f.name == 'uv.lock' else f'{f.parent.name}_{f.name}')
        shutil.copy(f, dst)
    torch.set_num_threads(1 if a.device == 'cpu' else 4)  # containers report host CPUs above their cgroup quota
    return a


def run_all(a, specs, fn, deadline='a', planned=True):
    """fn(a, spec) for every spec, in the given (priority) order, `a.procs` at a time sharing the device. A spec whose
    turn comes after the deadline is recorded as skipped; a finished run (result.json with status finished or
    early_stopped) is not rerun. Statuses are merged into <out>/status.json as runs end. With planned=True the full
    list (before --only) is saved as <out>/planned.json: analysis counts run health against it."""
    if planned:
        dump(Path(a.out) / 'planned.json', [s['name'] for s in specs])
    if a.only:
        keys = a.only.split(',')
        specs = [s for s in specs if any(k in s['name'] for k in keys)]
    jobs = [(fn, a, s, a.deadline if deadline == 'a' else deadline) for s in specs]
    print(f'{len(jobs)} runs, {min(a.procs, max(len(jobs), 1))} at a time', flush=True)
    out = []
    if a.procs == 1 or len(jobs) <= 1:
        it = (_guarded(*j) for j in jobs)
        for r in it:
            out.append(r)
            _record(a, [r])
    else:
        with mp.get_context('spawn').Pool(min(a.procs, len(jobs)), maxtasksperchild=4) as pool:
            for r in pool.imap(_guarded_star, jobs, chunksize=1):
                out.append(r)
                _record(a, [r])
    return out


def _record(a, results):
    path = Path(a.out) / 'status.json'
    st = json.load(open(path)) if path.exists() else {}
    for name, status in results:
        st[name] = status
    dump(path, st)


def _orphan_watch(parent):
    while os.getppid() == parent:  # a spawn worker would outlive a killed parent and overwrite a rerun's results
        time.sleep(5)
    os._exit(1)


def _guarded_star(job):
    if not getattr(_guarded_star, 'watching', False):
        threading.Thread(target=_orphan_watch, args=(os.getppid(),), daemon=True).start()
        _guarded_star.watching = True
    torch.set_num_threads(1)
    return _guarded(*job)


def _guarded(fn, a, spec, deadline):
    name = spec['name']
    done = Path(a.out) / spec.get('subdir', '') / name / 'result.json'
    if done.exists():
        st = json.load(open(done)).get('status')
        if st in ('finished', 'early_stopped'):
            return name, st
    if deadline is not None and time.time() > deadline:
        print(f'{name}: skipped (deadline passed before it started)', flush=True)
        return name, 'skipped'
    a = copy.copy(a)
    a.deadline = deadline
    try:
        return name, fn(a, spec)
    except Exception as e:  # keep the other runs going; the failure is printed and recorded
        import traceback
        traceback.print_exc()
        return name, f'failed: {e!r}'


# ------------------------------------------------------------------------------- transfer (rhm-transfer)

def split_root_productions(rules, seed, max_tries=100_000):
    """Split the n*m root productions into seen and unseen: exactly one unseen production per root class (so every
    class keeps m-1 seen productions and every unseen production's class is one the model was trained on), redrawn
    until every level-1 symbol that occurs on a side (left or right child) in an unseen production also occurs on that
    side in a seen one. Returns (seen mask (n, m) bool, number of redraws)."""
    R0 = rules[0].cpu()
    n, m, s = R0.shape
    g = torch.Generator().manual_seed(seed)
    for tries in range(max_tries):
        seen = torch.ones(n, m, dtype=torch.bool)
        seen[torch.arange(n), torch.randint(0, m, (n,), generator=g)] = False
        if all(set(R0[~seen][:, j].tolist()) <= set(R0[seen][:, j].tolist()) for j in range(s)):
            return seen, tries
    raise RuntimeError('no valid seen/unseen split found')


def sample_restricted(B, rules, allowed, g, device):
    """Trees whose root production is allowed: root uniform over classes, production uniform over the class's allowed
    productions, everything below as in the grammar. allowed: (n, m) bool, at least one per class."""
    n, m = allowed.shape
    root = torch.randint(0, n, (B, 1), generator=g, device=device)
    k = torch.multinomial(allowed.to(device).float()[root[:, 0]], 1, generator=g)
    trees = {0: root, ('rule', 0): k, 1: rules[0].to(device)[root, k].flatten(1)}
    return G.expand(trees, rules, 1, g, device)


def freeze(model, groups):
    """Freeze the named parameter groups of `model.groups()` (e.g. 'leaf', 'L1'); returns the trainable parameters."""
    for k in groups:
        for p in model.groups()[k]:
            p.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


def rescale_frozen(model, variant, x):
    """Scale the last frozen core so the bonds the fresh cores receive have unit RMS on x (the fresh cores keep their
    own init and so the same relative step size in every arm): 'C' scales W[-1] (level-3 cores) by 1/rho with
    rho = RMS of the level-3 bonds; 'WC' scales W[-2] (level-2 cores) by 1/rho with rho = RMS of the level-2 bonds.
    Returns rho and the RMS of the level-2 and level-3 bonds afterwards."""
    with torch.no_grad():
        _, acts = model(x, acts=True)
        k = -1 if variant == 'C' else -2
        rho = acts[k].pow(2).mean().sqrt()
        model.W[k].div_(rho)
        _, acts = model(x, acts=True)
    return dict(rho=float(rho), rms_level2=float(acts[-2].pow(2).mean().sqrt()),
                rms_level3=float(acts[-1].pow(2).mean().sqrt()))


class OracleTop(torch.nn.Module):
    """Ceiling for the transfer test: a trainable root core C applied to fixed top bonds that code the true level-1
    symbols, z = sqrt(r) Q e_a with a from the exact parse (needs r >= v) and Q a fixed seeded random orthogonal matrix.
    The code is dense on purpose: under Adam each entry of C moves ~lr per step, so a one-hot code (each sample touching
    one entry of C) learns ~r times slower than dense unit-RMS bonds and would not be a ceiling. C starts as `C0`."""

    def __init__(self, rules, v, r, C0, seed=0):
        super().__init__()
        self.rules, self.v, self.r = {l: t.cpu() for l, t in rules.items()}, v, r
        self.C = torch.nn.Parameter(C0.detach().clone())
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))
        self.register_buffer('code', (Q * math.sqrt(r)).to(C0.dtype))  # column a = sqrt(r) Q e_a

    def forward(self, x):
        lat = G.parse(x, self.rules, self.v)[1].clamp_min(0)  # (B, 2) level-1 symbols
        z = self.code.T[lat.to(self.code.device)]  # (B, 2, r)
        return torch.einsum('oab,Ba,Bb->Bo', self.C, z[:, 0], z[:, 1])
