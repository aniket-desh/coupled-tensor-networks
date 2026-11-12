"""lm-bytes: next-byte prediction on TinyStories from 16 context bytes. TTN against n-grams, MLP, chain and transformer.

Order of work (later stages are skipped, and recorded as skipped, if they cannot finish before the deadline):
  1. LR pilots: each neural family at peak LR in {3e-4, 1e-3}, 5k updates on the 10 MB pilot set, seed 0; the LR with
     the lower final validation bits/byte (2^17 fixed validation positions) is used for every run of that family
     (the untied-TTN choice, made at r = 32, also serves r = 16, 64, 128 and the long run; the matched-chain choice also
     serves the small chain).
  2. Main comparison, 40k updates on the 100 MB train set, seeds 0, 1, 2 (seed-major, so one seed of every model
     finishes first): untied TTN r = 32, MLP (192 hidden), matched chain (r = 93), transformer (d = 128, 4 layers),
     and the frozen-fusion TTN r = 32 (identity init, fusion cores W frozen, only A and C trained, the TTN's LR).
     Kneser-Ney n-grams (orders 1..8, counted on all of train.bin and on its first 10.24M bytes, the neural models'
     target count) run at the start in their own CPU process.
  3. The rest: untied TTN r = 32 with the variance init (2 seeds, the TTN's piloted LR), level-tied TTN r = 32 (3 seeds), untied TTN r = 16 (3), r = 64 (2), small chain r = 32 (3), untied TTN
     r = 128 (1), and one untied r = 32 run of 160k updates (seed 0).
TTN and chain init: identity (ctn.tree.identity_init, eps = 0.1) unless the run name says 'var' (the classes'
variance-preserving init, which stalls near 8 bits/byte under Adam on CPU pilots).
Every neural run: batches of 256 random windows, Adam, no weight decay, 1k-update linear warmup then cosine to 0.1x
peak, gradient-norm clipping at 1.0 (needed by the TTN, which diverges without it; applied to every family so the
schedule is identical), fp32 with TF32 matmuls. Final validation and test bits/byte on every target of val.bin and
test.bin. Checkpoints of the untied r = 32 seed-0 main run at updates CKPT_STEPS go to --ckpt.
GPU runs share the device through a pool of --procs processes, taking runs in the priority order above.

Usage: PYTHONPATH=. python studies/lm/lm-bytes/run.py --out DIR --hours H [--smoke] [--procs P] [--ckpt DIR]
         [--only NAME,...] [--lr FAMILY=VALUE,...]
--only runs just the named runs (pilots, neural runs, ngram_10M / ngram_100M; smoke: ngram_small / ngram_large). The
peak LRs come from --lr, then from DIR/lr_choice.json if it exists, then from the named pilots; a neural run with no
LR from any of these is an error. Use it for reruns into a fresh DIR (copy lr_choice.json there, or pass --lr).
Outputs: DIR/runs/<name>.json per run (see studies/lm/train.py), DIR/plan.json, DIR/lr_choice.json, DIR/status.json,
DIR/source/ (copies of the code)."""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FAMILIES = {  # the model each family's LR pilot uses
    'ttn': dict(kind='tree', r=32, tie='none', init='identity', eps=0.1),
    'ttn_tied': dict(kind='tree', r=32, tie='level', init='identity', eps=0.1),
    'mlp': dict(kind='mlp', hidden=192),
    'chain': dict(kind='chain', r=93, init='identity', eps=0.1),
    'transformer': dict(kind='transformer', d=128, layers=4, heads=4),
}
SMOKE_SIZE = {('tree', 16): 2, ('tree', 32): 4, ('tree', 64): 6, ('tree', 128): 8, ('chain', 93): 6, ('chain', 32): 4}
LRS = [3e-4, 1e-3]
CKPT_STEPS = [0, 100, 300, 1000, 3000, 10000, 20000, 40000]
FULL = dict(steps=40_000, pilot_steps=5_000, long_steps=160_000, warmup=1_000, eval_every=500, n_curve=2 ** 17,
            final='full', ckpt_steps=CKPT_STEPS, ngram_order=8)
SMOKE = dict(steps=300, pilot_steps=100, long_steps=1_200, warmup=30, eval_every=50, n_curve=2048, final=8192,
             ckpt_steps=[0, 10, 100, 300], ngram_order=8, ngram_train_bytes=1_000_000)


def model_spec(family, smoke, **over):
    spec = dict(FAMILIES[family], **over)
    if smoke:
        if spec['kind'] in ('tree', 'chain'):
            spec['r'] = SMOKE_SIZE[(spec['kind'], spec['r'])]
        elif spec['kind'] == 'mlp':
            spec['hidden'] = 16
        else:
            spec.update(d=16, layers=1, heads=2)
    return spec


def tag(spec):
    k = spec['kind']
    return {'tree': f"r{spec.get('r')}", 'chain': f"r{spec.get('r')}", 'mlp': f"h{spec.get('hidden')}",
            'transformer': f"d{spec.get('d')}"}[k]


def cfg_for(family, seed, lr, P, smoke, steps=None, train='train', name=None, ckpt=False, **over):
    spec = model_spec(family, smoke, **over)
    steps = steps or P['steps']
    nominal = tag(model_spec(family, False, **over))  # smoke runs keep the full-size names, so analysis works unchanged
    return dict(name=name or f'{family}_{nominal}_s{seed}', family=family, model=spec, seed=seed, lr=lr, steps=steps,
                batch=256, warmup=P['warmup'], floor=0.1, clip=1.0, train=train, eval_every=P['eval_every'],
                n_curve=P['n_curve'], final=P['final'], ckpt_steps=P['ckpt_steps'] if ckpt else [])


def pilots(P, smoke):
    return [cfg_for(f, 0, lr, P, smoke, steps=P['pilot_steps'], train='pilot', name=f'pilot_{f}_lr{lr:g}')
            | dict(final=P['n_curve'])  # pilots are judged on the fixed 2^17 validation positions
            for f in FAMILIES for lr in LRS]


def main_and_rest(P, smoke, lr):
    runs = []
    for s in range(3):
        runs += [cfg_for('ttn', s, lr['ttn'], P, smoke, ckpt=(s == 0)), cfg_for('mlp', s, lr['mlp'], P, smoke),
                 cfg_for('chain', s, lr['chain'], P, smoke), cfg_for('transformer', s, lr['transformer'], P, smoke),
                 # Control for what training the fusion cores adds beyond the identity init's starting function
                 # class: W frozen at the identity init, only the leaves A and the root C train.
                 cfg_for('ttn', s, lr['ttn'], P, smoke, name=f'ttn_frozenW_r32_s{s}') | dict(family='ttn_frozenW',
                                                                                           frozen=['W'])]
    # The variance-init arm: same LR, data and schedule as the identity-init TTN; documents the zero-saddle stall.
    runs += [cfg_for('ttn', s, lr['ttn'], P, smoke, init='variance', name=f'ttn_var_r32_s{s}') | dict(family='ttn_var')
             for s in range(2)]
    runs += [cfg_for('ttn_tied', s, lr['ttn_tied'], P, smoke) for s in range(3)]
    runs += [cfg_for('ttn', s, lr['ttn'], P, smoke, r=16) for s in range(3)]
    runs += [cfg_for('ttn', s, lr['ttn'], P, smoke, r=64) for s in range(2)]
    runs += [cfg_for('chain', s, lr['chain'], P, smoke, r=32) for s in range(3)]
    runs += [cfg_for('ttn', 0, lr['ttn'], P, smoke, r=128)]
    long = cfg_for('ttn', 0, lr['ttn'], P, smoke, steps=P['long_steps'])
    runs += [long | dict(name=long['name'].replace('_s0', '_long_s0'))]
    return runs


def gpu_job(cfg, out, deadline, ckpt_dir, threads):
    """One neural run in a pool process. Exceptions are recorded as status 'failed'."""
    import torch
    torch.set_num_threads(threads)
    from studies.lm.train import train_run
    path = Path(out) / 'runs' / f"{cfg['name']}.json"
    if time.time() > deadline - 60:
        res = dict(cfg=cfg, status='skipped', reason='not started: deadline reached', stopped_at=0)
        json.dump(res, open(path, 'w'), indent=1)
        return res
    try:
        return train_run(cfg, out, deadline, ckpt_dir=ckpt_dir if cfg['ckpt_steps'] else None)
    except Exception:
        res = dict(cfg=cfg, status='failed', error=traceback.format_exc())
        json.dump(res, open(path, 'w'), indent=1)
        print(f"{cfg['name']}: FAILED\n{res['error']}", flush=True)
        return res


def ngram_names(smoke):
    return ['ngram_small', 'ngram_large'] if smoke else ['ngram_10M', 'ngram_100M']


def ngram_job(out, P, smoke, names):
    from studies.lm.train import ngram_run
    tb = P.get('ngram_train_bytes')
    small, large = ngram_names(smoke)
    if small in names:
        ngram_run(small, out, train_bytes=10_240_000 if not smoke else tb // 4, final=P['final'], n_curve=P['n_curve'],
                  max_order=P['ngram_order'])
    if large in names:
        ngram_run(large, out, train_bytes=tb, final=P['final'], n_curve=P['n_curve'], max_order=P['ngram_order'])


def run_pool(ex, cfgs, out, deadline, ckpt_dir, threads):
    futs = {ex.submit(gpu_job, c, out, deadline, ckpt_dir, threads): c['name'] for c in cfgs}
    res = {}
    for f in as_completed(futs):
        try:
            res[futs[f]] = f.result()
        except Exception:  # the pool itself broke (e.g. a worker was killed)
            res[futs[f]] = dict(status='failed', error=traceback.format_exc())
            print(f'{futs[f]}: pool failure\n{res[futs[f]]["error"]}', flush=True)
    return res


def choose_lr(pres, fams):
    """Per family in fams (the families whose pilots were submitted): the LR of the finished pilot with the lowest
    validation bits/byte."""
    choice = {}
    for fam in fams:
        cands = [(r['val_bits'], r['cfg']['lr']) for r in pres.values()
                 if r.get('status') == 'done' and r['cfg']['family'] == fam]
        choice[fam] = dict(lr=min(cands)[1] if cands else min(LRS), pilot_val_bits={str(lr): v for v, lr in cands},
                           note='' if cands else 'no pilot finished; the smaller LR is used')
    return choice


def copy_source(out):
    dst = Path(out) / 'source'
    for f in [*sorted((ROOT / 'studies/lm').glob('*.py')), *sorted(Path(__file__).parent.glob('*.py')),
              ROOT / 'ctn/tree.py', Path(__file__).parent / 'EXPERIMENT.md']:
        if f.exists():
            rel = f.relative_to(ROOT)
            (dst / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--hours', type=float, required=True, help='deadline for all training, from start')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--procs', type=int, default=3, help='concurrent GPU runs')
    p.add_argument('--threads', type=int, default=None, help='torch CPU threads per run (default 2 on GPU, 1 on CPU)')
    p.add_argument('--ckpt', default='artifacts/lm-bytes/ttn_r32_s0', help='checkpoint folder of the r = 32 seed-0 run')
    p.add_argument('--only', default='', help='comma-separated run names to run (default: everything)')
    p.add_argument('--lr', default='', help='comma-separated FAMILY=VALUE peak LRs that override the pilots')
    a = p.parse_args()
    only = set(n for n in a.only.split(',') if n)
    lr_over = {k: float(v) for k, v in (kv.split('=') for kv in a.lr.split(',') if kv)}
    unknown = set(lr_over) - set(FAMILIES)
    assert not unknown, f'--lr: unknown families {unknown}; use {list(FAMILIES)}'
    t0 = time.time()
    deadline = t0 + 3600 * a.hours
    out = Path(a.out)
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    copy_source(out)
    P = SMOKE if a.smoke else FULL
    import torch
    threads = a.threads or (2 if torch.cuda.is_available() else 1)
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    print(f"device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; procs {a.procs}; "
          f"deadline in {a.hours} h", flush=True)

    want = lambda name: not only or name in only
    ng_names = [n for n in ngram_names(a.smoke) if want(n)]
    ctx = mp.get_context('spawn')
    ng = ctx.Process(target=ngram_job, args=(str(out), P, a.smoke, ng_names))
    if ng_names:
        ng.start()
    with ProcessPoolExecutor(a.procs, mp_context=ctx) as ex:
        pcfgs = [c for c in pilots(P, a.smoke) if want(c['name'])]
        pres = run_pool(ex, pcfgs, str(out), deadline, a.ckpt, threads)
        lr_file = out / 'lr_choice.json'
        lr = json.load(open(lr_file)) if only and lr_file.exists() else {}  # a rerun keeps the original pilots' LRs
        lr.update(choose_lr(pres, [f for f in FAMILIES if any(c['family'] == f for c in pcfgs)]))
        for fam, v in lr_over.items():
            lr[fam] = dict(lr=v, note='set by --lr')
        if pres or lr_over:
            json.dump(lr, open(lr_file, 'w'), indent=1)
        print('LR choice: ' + ', '.join(f"{k} {v['lr']:g}" for k, v in lr.items()), flush=True)
        lrs = {f: lr.get(f, {}).get('lr') for f in FAMILIES}
        runs = [c for c in main_and_rest(P, a.smoke, lrs) if want(c['name'])]
        missing = sorted({c['name'] for c in runs if c['lr'] is None})
        assert not missing, f'no LR for {missing}: pass --lr or put lr_choice.json in --out'
        json.dump(dict(pilots=pilots(P, a.smoke), runs=runs, only=sorted(only), params=P, procs=a.procs, hours=a.hours),
                  open(out / 'plan.json', 'w'), indent=1, default=str)
        mres = run_pool(ex, runs, str(out), deadline, a.ckpt, threads)
    if ng_names:
        ng.join(timeout=max(0.0, deadline - time.time()))
        if ng.is_alive():
            ng.terminate()
            print('n-grams: stopped at the deadline', flush=True)
    for n in ng_names:
        if not (out / 'runs' / f'{n}.json').exists():
            json.dump(dict(cfg=dict(name=n, family='ngram'), status='skipped', reason='not finished by the deadline'),
                      open(out / 'runs' / f'{n}.json', 'w'), indent=1)
    status = {n: r.get('status') for n, r in {**pres, **mres}.items()}
    json.dump(dict(status=status, wall_h=round((time.time() - t0) / 3600, 3)), open(out / 'status.json', 'w'), indent=1)
    print(f"done in {(time.time() - t0) / 3600:.2f} h: " +
          ', '.join(f'{s} {list(status.values()).count(s)}' for s in sorted(set(map(str, status.values())))), flush=True)


if __name__ == '__main__':
    main()
