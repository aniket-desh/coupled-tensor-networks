"""Shared runner for the TTN experiments: data, one batched group of runs, summaries, and saving.

A group is R runs that share a teacher and hyperparameters and differ in init (and data) seed; they train together
in one batched computation. Every group is saved as soon as it ends, so a run cut short by its time cap keeps
everything finished before it. Runs still training when a group stops are censored: reported, never dropped.

Everything trains in float64: at small init scale the per-step relative update shrinks like alpha^13 and falls below
float32 resolution (measured: at alpha = 0.6, 72% of float32 updates round to no change).
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ctn.measure import half_time
from ctn.ttn import TTN, Population, loss_per_run, sample, stack, targets, train

O = 10  # student outputs; outputs no teacher component uses are masked out of the loss
DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[2]


def args(**defaults):
    """Command-line knobs every run.py shares; `--smoke` shrinks everything for a CPU test. Copies the code that
    produces the results into <out>/source first."""
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--hours', type=float, default=defaults.get('hours', 4.5), help='wall-clock deadline for training')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--groups', default='', help='comma-separated group names to run (default: all), e.g. for a rerun')
    p.add_argument('--eta', type=float, default=None, help='override the step size (recorded in each group json)')
    p.add_argument('--allow-cpu', action='store_true', help='run without a GPU (local validation only)')
    p.add_argument('--seeds', type=int, default=0, help='override seeds per group (testing)')
    p.add_argument('--t-max', type=float, default=0, help='override t_max (testing)')
    p.add_argument('--procs', type=int, default=0, help='parallel group processes (default: min(groups, CPUs - 1, 8))')
    a = p.parse_args()
    a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.device == 'cpu' and not (a.smoke or a.allow_cpu):
        raise SystemExit('no CUDA device: refusing to run the full grid on CPU (a driver or torch build problem)')
    a.deadline = time.time() + 3600 * a.hours
    out = Path(a.out)
    (out / 'source').mkdir(parents=True, exist_ok=True)
    for f in [*(ROOT / 'ctn').glob('*.py'), *(ROOT / 'studies/ttn').glob('*.py'), Path(sys.argv[0]).resolve(),
              ROOT / 'uv.lock']:
        shutil.copy(f, out / 'source' / f.name)
    return a


def data(comps, R, N, N_probe, seed, device):
    """Training set (per run, independent; none when N = 0, i.e. population training) and a fresh probe set for
    alignment and spectra (and, in sampled mode, coefficients and K)."""
    phi = y = None
    if N:
        x, phi = sample(R, N, 8, seed, device, DTYPE)
        y, _ = targets(x, comps, O)
    xp, phip = sample(R, N_probe, 8, seed + 10_000, device, DTYPE)
    _, gp = targets(xp, comps, O)
    return phi, y, (xp, phip, gp)


def active_mask(comps, R, device, outputs=None):
    m = torch.zeros(R, O, device=device, dtype=DTYPE)
    for c in comps:
        if outputs is None or c['out'] in outputs:
            m[:, c['out']] = 1.0
    return m


def summarize(log, comps, t_from=0.0):
    """Per run and component: half-time measured from t_from (nan if never reached: right-censored at the run's end,
    not missing; see median_censored), peak relative overshoot, final c/s, and |c| at t_from."""
    t, c = log['t'], log['c']  # (T,), (T, R, K)
    s = np.array([cc['s'] for cc in comps])
    sel = t >= t_from
    half = np.array([[half_time(t[sel] - t_from, c[sel, r_, k], s[k]) for k in range(len(comps))] for r_ in range(c.shape[1])])
    return dict(half=half, overshoot=np.maximum((c[sel] / s - 1).max(0), 0.0), final=c[-1] / s,
                c_start=np.abs(c[sel][0]), diverged=log['diverged'][-1])


def median_censored(h, axis=0):
    """Median over runs, with never-reached half-times (nan) counted as +inf: censored runs are slower than every run
    that learned. The result is inf when half or more of the runs are censored."""
    return np.median(np.where(np.isnan(h), np.inf, h), axis=axis)


def save_group(out, name, log, meta):
    """<out>/<name>.npz holds the time series; <out>/<name>.json the metadata and per-run summaries."""
    log = stack(log)
    np.savez_compressed(Path(out) / f'{name}.npz', **{k: v for k, v in log.items() if isinstance(v, np.ndarray) and v.size})
    json.dump(meta, open(Path(out) / f'{name}.json', 'w'), indent=1,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


def run_group(a, name, comps, R, r, alpha, eta, t_max, N, N_probe, seed, outputs=None, net=None, t0=0.0, prev=None,
              stop_tol=1e-3, stop_loss=1e-8, data_=None, extra_meta=None, groups_left=1, active=None,
              check_every=10, net_kw=None):
    """Train one group to t_max (or convergence, or its share of the remaining wall-clock time: 1/groups_left) and
    save it. N = 0 trains on the exact population loss (Population), N > 0 on N sampled inputs per run.
    Returns (net, log, data)."""
    phi, y, probe = data_ or data(comps, R, N, N_probe, seed, a.device)
    net = net or TTN(R=R, r=r, d=8, O=O, alpha=alpha, seed=seed, device=a.device, dtype=DTYPE, **(net_kw or {}))
    active = active if active is not None else active_mask(comps, R, a.device, outputs)
    pop = Population(comps, R, r, 8, O, a.device, DTYPE) if not N else None
    loss_fn = (lambda: pop.loss(net, active)) if pop is not None else (lambda: loss_per_run(net(phi), y, active))
    step_s = _step_time(net, loss_fn)
    budget = max(int((a.deadline - time.time()) / groups_left / max(step_s, 1e-6)), 1)
    wanted = int(round((t_max - t0) / eta))
    steps = min(wanted, budget)
    print(f'{name}: {step_s * 1e3:.2f} ms/step, budget {budget:.3g} steps (t <= {t0 + eta * steps:.4g}), wanted {wanted:.3g}',
          flush=True)
    start = time.time()
    log = train(net, active, eta, steps, probe, comps, phi=phi, y=y, pop=pop, check_every=check_every if not N else 50,
                t0=t0, stop_tol=stop_tol, stop_loss=stop_loss, log=prev)
    meta = dict(name=name, comps=comps, R=R, r=r, alpha=alpha, eta=eta, t_max=t_max, N=N, N_probe=N_probe, seed=seed,
                loss='population' if pop is not None else 'sampled', outputs=outputs, t_end=log['t_end'],
                ms_per_step=step_s * 1e3, seconds=time.time() - start, device=gpu_note(), **(extra_meta or {}))
    finished = np.asarray(log['converged']) | np.asarray(log['diverged'][-1])
    meta['converged'] = np.asarray(log['converged']).tolist()
    meta['censored_runs'] = int((~finished).sum())
    meta['diverged_runs'] = int(np.asarray(log['diverged'][-1]).sum())
    meta['stopped_by'] = ('convergence' if finished.all() and meta['converged'].count(True) else 'divergence' if finished.all()
                          else 'deadline' if steps < wanted else 't_max')
    meta['summary'] = summarize(stack(dict(log)), comps, t0 if prev is None else 0.0)
    save_group(a.out, name, log, meta)
    torch.save([p.detach().cpu() for p in net.params], Path(a.out) / f'{name}_params.pt')  # for branching and reruns
    print(f"{name}: t_end {log['t_end']:.4g} in {meta['seconds']:.0f}s, stopped by {meta['stopped_by']}, "
          f"{meta['censored_runs']}/{R} runs censored, {meta['diverged_runs']}/{R} diverged", flush=True)
    return net, log, (phi, y, probe)


def _step_time(net, loss_fn, n=3):
    """Seconds per GD step for this group (gradients only, parameters untouched)."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    torch.autograd.grad(loss_fn().sum(), net.params)  # warm-up
    sync()
    t = time.time()
    for _ in range(n):
        torch.autograd.grad(loss_fn().sum(), net.params)
    sync()
    return (time.time() - t) / n * 1.15  # logging and detail measurements add roughly 15%


def gpu_note():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'


def run_parallel(a, specs, fn):
    """Run fn(a, spec) for every spec in its own process, `a.procs` at a time, all sharing the one GPU.
    Population steps are launch-bound, so concurrent groups barely slow each other and each gets the whole deadline
    (groups that wait in the queue get what is left when they start). specs: list of dicts with a 'name'.
    `--groups` restricts the run to the named specs. A failing group is reported and the others continue."""
    if a.groups:
        wanted = a.groups.split(',')
        specs = [sp for sp in specs if sp['name'] in wanted]
    procs = a.procs or max(1, min(len(specs), (os.cpu_count() or 2) - 1, 8))
    print(f'{len(specs)} groups, {procs} at a time: {[sp["name"] for sp in specs]}', flush=True)
    if procs == 1:
        return [_guarded(fn, a, sp) for sp in specs]
    with mp.get_context('spawn').Pool(procs, maxtasksperchild=1) as pool:
        return pool.starmap(_guarded, [(fn, a, sp, procs) for sp in specs], chunksize=1)


def _orphan_watch(parent):
    # A spawn worker outlives a killed parent and would later overwrite results of a rerun; exit once orphaned.
    while os.getppid() == parent:
        time.sleep(5)
    os._exit(1)


def _guarded(fn, a, spec, procs=1):
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // procs))  # concurrent groups must not oversubscribe the CPU
    if procs > 1:
        threading.Thread(target=_orphan_watch, args=(os.getppid(),), daemon=True).start()
    try:
        fn(a, spec)
        return spec['name'], 'ok'
    except Exception as e:  # keep the other groups running; the failure is printed and returned
        import traceback
        traceback.print_exc()
        return spec['name'], f'failed: {e!r}'
