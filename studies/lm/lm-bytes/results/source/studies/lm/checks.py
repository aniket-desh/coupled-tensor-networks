"""Correctness checks for the lm experiments. Exits non-zero if any check fails.
Usage: python -m studies.lm.checks [out.json]

  data       the split files match meta.json (sizes, separators, sha256); windows return the right bytes
  uniform    bits/byte of the uniform model is exactly 8
  params     parameter counts of the full-size models match their formulas, and the matched chain is within 0.1%
  schedule   warmup and cosine endpoints of the LR schedule
  ttn_norms  per-level activation RMS and per-group gradient norms of the r = 16 untied TTN on real windows, at init
             and after 20 Adam steps at LR 1e-3, for the variance init and the identity init: finite, within loose
             bounds (the variance-init numbers show the shrinking toward the zero saddle)
  identity   full-size identity-init networks (TTN r = 32 untied and tied, chain r = 93) give finite activations at
             every level / step on real windows, with channel 0 carrying the constant
  overfit    every neural family (small instances) fits 128 fixed random (context, byte) pairs to near-zero loss;
             tree and chain with both inits
  ngram      the vectorised counts equal a naive dictionary count on a short text; KN distributions sum to 1 for
             sample contexts at every order; the per-target path agrees with the full distribution"""
import json
import math
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from ctn.io import save_json
from ctn.tree import tree_params
from studies.lm import data as D
from studies.lm import models, ngram
from studies.lm.train import lr_at

torch.set_num_threads(1)
results, failed = {}, []


def check(name, ok, **info):
    results[name] = dict(ok=bool(ok), **info)
    print(f"[{'ok' if ok else 'FAIL'}] {name} {json.dumps(info, default=float)}", flush=True)
    if not ok:
        failed.append(name)


def check_data():
    problems = D.verify()
    check('data_files', not problems, problems=problems)
    va = D.load('val')
    t = torch.tensor([D.CTX, 1000, va.size - 1])
    x, y = D.windows(torch.from_numpy(va), t)
    ok = all(np.array_equal(x[i].numpy(), va[ti - D.CTX:ti]) and y[i].item() == va[ti] for i, ti in enumerate(t.tolist()))
    check('windows', ok)


def check_uniform():
    y = torch.randint(0, 256, (1000,))
    bits = F.cross_entropy(torch.zeros(1000, 256, dtype=torch.float64), y).item() / math.log(2)
    check('uniform_8_bits', abs(bits - 8.0) < 1e-12, bits=bits)


def check_params():
    got = {f'tree_r{r}': models.n_params(models.build(dict(kind='tree', r=r), 0)) for r in (16, 32, 64)}
    want = {f'tree_r{r}': tree_params(16, 256, r, 256) for r in (16, 32, 64)}
    tied = models.n_params(models.build(dict(kind='tree', r=32, tie='level'), 0))
    chain = models.n_params(models.build(dict(kind='chain', r=93), 0))
    mlp = models.n_params(models.build(dict(kind='mlp', hidden=192), 0))
    tf = models.n_params(models.build(dict(kind='transformer', d=128, layers=4, heads=4), 0))
    ok = (got == want and want['tree_r32'] == 851_968 and tied == 256 * 32 + 3 * 32 ** 3 + 256 * 32 ** 2
          and chain == 93 + 2 * 256 * 93 + 93 ** 3 and abs(chain / want['tree_r32'] - 1) < 1e-3 and mlp == 836_032
          and abs(tf / 0.85e6 - 1) < 0.05)
    check('param_counts', ok, trees=got, tied_r32=tied, chain_r93=chain, mlp=mlp, transformer=tf)


def check_schedule():
    cfg = dict(lr=1e-3, warmup=1000, steps=40000, floor=0.1)
    v = [lr_at(0, cfg), lr_at(999, cfg), lr_at(39999, cfg), lr_at(20499, cfg)]
    ok = abs(v[0] - 1e-6) < 1e-12 and abs(v[1] - 1e-3) < 1e-12 and abs(v[2] - 1e-4) < 1e-12 and abs(v[3] - 5.5e-4) < 1e-7
    check('schedule', ok, lr_first_warm_last_mid=v)


def level_stats(model, x, y):
    logits, acts = model(x, acts=True)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    act = [a.pow(2).mean().sqrt().item() for a in acts] + [logits.pow(2).mean().sqrt().item()]
    grad = {k: sum(p.grad.pow(2).sum() for p in ps).sqrt().item() for k, ps in model.groups().items()}
    return loss.item(), act, grad


def check_ttn_norms():
    for init in ('variance', 'identity'):
        ttn_norms(init)


def ttn_norms(init):
    print(f'  {init} init:')
    torch.manual_seed(0)
    tr = torch.from_numpy(D.load('pilot'))
    model = models.build(dict(kind='tree', r=16, init=init, eps=0.1), seed=0)
    t = torch.randint(D.CTX, tr.numel(), (256,))
    x, y = D.windows(tr, t)
    l0, a0, g0 = level_stats(model, x, y)
    print('  init: loss %.3f; activation RMS by level (0..3, logits) %s; grad norms %s' %
          (l0, np.round(a0, 3).tolist(), {k: round(v, 4) for k, v in g0.items()}))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(20):
        xb, yb = D.windows(tr, torch.randint(D.CTX, tr.numel(), (256,)))
        opt.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
    l1, a1, g1 = level_stats(model, x, y)
    print('  after 20 Adam steps: loss %.3f; activation RMS %s; grad norms %s' %
          (l1, np.round(a1, 3).tolist(), {k: round(v, 4) for k, v in g1.items()}))
    vals = a0 + a1 + list(g0.values()) + list(g1.values())
    lo = 0.3 if init == 'variance' else 0.05  # identity: channel 0 is ~1 and the rest ~eps, so RMS ~ 1/sqrt(r)
    ok = (all(math.isfinite(v) for v in vals + [l0, l1]) and all(lo < a < 3 for a in a0)
          and all(1e-3 < a < 1e3 for a in a1) and all(1e-6 < g < 1e3 for g in list(g0.values()) + list(g1.values())))
    check(f'ttn_level_norms_{init}', ok, init_act=a0, init_grad=g0, step20_act=a1, step20_grad=g1, loss=[l0, l1])


def check_identity_init():
    tr = torch.from_numpy(D.load('pilot'))
    x, _ = D.windows(tr, torch.randint(D.CTX, tr.numel(), (512,), generator=torch.Generator().manual_seed(1)))
    for name, spec in {'tree_r32': dict(kind='tree', r=32), 'tree_tied_r32': dict(kind='tree', r=32, tie='level'),
                       'chain_r93': dict(kind='chain', r=93)}.items():
        model = models.build(dict(spec, init='identity', eps=0.1), seed=0)
        with torch.no_grad():
            logits, acts = model(x, acts=True)
        rms = [a.pow(2).mean().sqrt().item() for a in acts] + [logits.pow(2).mean().sqrt().item()]
        ch0 = [a[..., 0].mean().item() for a in acts]  # the constant channel
        ok = all(math.isfinite(v) and 1e-3 < v < 1e3 for v in rms) and all(0.5 < c < 2 for c in ch0)
        check(f'identity_init_finite_{name}', ok, act_rms=[round(v, 4) for v in rms],
              channel0_mean=[round(c, 3) for c in ch0][:6])


def check_overfit():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randint(0, 256, (128, 16), generator=g), torch.randint(0, 256, (128,), generator=g)
    specs = {'tree_variance': dict(kind='tree', r=8, init='variance'),
             'tree_tied_variance': dict(kind='tree', r=8, tie='level', init='variance'),
             'chain_variance': dict(kind='chain', r=16, init='variance'),
             'tree_identity': dict(kind='tree', r=8, init='identity', eps=0.1),
             'chain_identity': dict(kind='chain', r=16, init='identity', eps=0.1), 'mlp': dict(kind='mlp', hidden=192),
             'transformer': dict(kind='transformer', d=32, layers=2, heads=2)}
    for name, spec in specs.items():
        model = models.build(spec, seed=0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        t0 = time.time()
        for step in range(1, 1501):
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if loss.item() < 0.01:
                break
        check(f'overfit_{name}', loss.item() < 0.05, loss=loss.item(), steps=step, s=round(time.time() - t0, 1))


def check_ngram():
    text = np.frombuffer(b'the cat sat on the mat. the cat ate the rat. a mat, a cat, a hat.' * 3, np.uint8)
    kn = ngram.KneserNey(max_order=4, log=lambda *a, **k: None).fit(text)
    ok = True
    for m in (1, 2, 3, 4):
        naive = Counter(bytes(text[t - m + 1:t + 1]) for t in range(m - 1, text.size))
        L = kn.raw[m - 1]
        got = {int(k).to_bytes(8, 'little')[:m][::-1]: int(c) for k, c in zip(L.g, L.a)}
        ok &= got == dict(naive)
        if m < 4:  # continuation counts: distinct left neighbours of each m-gram
            cont = Counter(g[1:] for g in Counter(bytes(text[t - m:t + 1]) for t in range(m, text.size)))
            Lc = kn.cont[m - 1]
            ok &= {int(k).to_bytes(8, 'little')[:m][::-1]: int(c) for k, c in zip(Lc.g, Lc.a)} == dict(cont)
    check('ngram_counts_vs_naive', ok)
    tr = D.load('pilot')[:300_000]
    kn = ngram.KneserNey(max_order=8, log=lambda *a, **k: None).fit(tr)
    va = D.load('val')
    err, agree = 0.0, 0.0
    for t in [16, 5000, 123456, 999_999, va.size - 1]:
        p = kn.dist(va[t - 16:t])
        err = max(err, np.abs(p.sum(1) - 1).max())
        agree = max(agree, np.abs(p[:, va[t]] - kn.probs(va, np.array([t]))[:, 0]).max())
    p = kn.dist(np.full(16, 7, np.uint8))  # a context never seen (byte 7 does not occur)
    err = max(err, np.abs(p.sum(1) - 1).max())
    check('ngram_sums_to_1', err < 1e-9 and agree < 1e-12, max_abs_sum_err=err, max_target_disagreement=agree)


if __name__ == '__main__':
    t0 = time.time()
    for f in (check_data, check_uniform, check_params, check_schedule, check_ttn_norms, check_identity_init, check_overfit, check_ngram):
        try:
            f()
        except Exception as e:  # a crash is a failure, not a skip
            check(f.__name__, False, error=repr(e))
    results['seconds'] = round(time.time() - t0, 1)
    if len(sys.argv) > 1:
        save_json(sys.argv[1], results)  # creates the parent folder (a fresh worker has no results/ yet)
    print(f"{len(failed)} failed: {failed}" if failed else f"all checks passed ({results['seconds']}s)")
    sys.exit(1 if failed else 0)
