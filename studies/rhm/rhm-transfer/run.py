"""rhm-transfer: do a pretrained TreeTN's lower subtrees hold side-symbol codes that a new readout can reuse on unseen
root productions?

Root task, main grammar (m = 4), grammars 0-2, r = 16, init seeds 0-2 (priority: seed 0 on every grammar first),
identity init (ctn.tree.identity_init) for every model. The 64 root productions are split per grammar into 48 seen and
16 unseen (`train.split_root_productions`: one unseen production per class; every level-1 symbol that occurs on a side
of an unseen production occurs on that side in a seen one). Productions are distinct pairs, so an unseen pair can be
told apart from the seen ones by its two side symbols: phase 2 tests reuse of side-symbol codes by a new readout, not
conjunctive recombination.

One unit = one (grammar g, init seed s), all models from the same init state:
  pretrain        online trees of grammar g whose root production is seen, PRE updates; gate: seen accuracy >= GATE
  pretrain_other  online trees of a different grammar (seed 100 + g, all productions), PRE updates; gate on its own
                  root accuracy. Its lower subtrees are trained and well scaled but code the wrong symbols.
  phase 2, POST updates per arm on unseen-production trees of grammar g, in this order:
    scratch             every core trainable, from the init
    finetune            every core trainable, from the pretrained state (C included)
    frozen_learned_C    leaves and fusion levels 1-3 frozen at the pretrained values, fresh root core C
    frozen_other_C      the same with pretrain_other's frozen part (a trained floor)
    frozen_random_C     the same with the init's frozen part (an untrained, near-additive map of the tokens)
    oracle_C            fresh C on fixed top bonds sqrt(r) e_a of the true level-1 symbols (ceiling, descriptive)
    frozen_learned_WC, frozen_other_WC, frozen_random_WC: leaves and levels 1-2 frozen, fresh level-3 cores and C
  Fresh cores are variance-init draws (TreeTN seed 1000 + s), the same draw in every arm. The last frozen core is
  scaled so the fresh cores receive unit-RMS bonds (`train.rescale_frozen`): W[-1] for the C arms, W[-2] for the WC
  arms (the variance-init level-3 cores then give level-3 bonds of RMS about 1, recorded). Trainable cores are never
  rescaled, so every arm's fresh cores take the same relative steps.
Each phase and arm has its own Adam and schedule (warmup 100, cosine to 0.1x over its length). The peak LR is the
root-task online pilot's (r = 16, grammar 0, init seed 9, {1e-4, 3e-4, 1e-3, 3e-3}); --lr overrides it.
Diagnostic sets: 20,000 seen-production and 20,000 unseen-production trees per grammar, CPU-seeded.

  PYTHONPATH=. python studies/rhm/rhm-transfer/run.py --out runs/rhm-transfer --hours 3 [--smoke]
"""
import copy
import json
import time
from pathlib import Path

import torch

from studies.rhm import common as C
from studies.rhm import train as T

R, PRE, POST, PILOT_STEPS, GATE, OTHER = 16, 5_000, 10_000, 4_000, 0.95, 100
ARMS = ('scratch', 'finetune', 'frozen_learned_C', 'frozen_other_C', 'frozen_random_C', 'oracle_C',
        'frozen_learned_WC', 'frozen_other_WC', 'frozen_random_WC')
FROZEN = {'C': ['leaf', 'L1', 'L2', 'L3'], 'WC': ['leaf', 'L1', 'L2']}
SPLIT_SEED, DIAG_SEED = 60_000, 61_000


def cfg(gseed, seed):
    return dict(model='tree', r=R, tie='none', perm=None, seed=seed, task='root', grammar=T.grammar_cfg(gseed),
                init='identity')


def diag_sets(a, g, rules, seen):
    gen = torch.Generator().manual_seed(DIAG_SEED + g['seed'])
    c = dict(task='root', grammar=g)
    return {k: T.evalset(T.sample_restricted(a.n_diag, rules, allowed, gen, 'cpu'), c, a.device)
            for k, allowed in (('seen', seen), ('unseen', ~seen))}


def fresh(model, variant, seed):
    """Replace the upper cores by variance-init draws (seed 1000 + init seed): C, and for 'WC' also the level-3 cores."""
    new = C.build(dict(model='tree', r=model.r, tie='none', perm=None, seed=1000 + seed, task='root',
                       grammar=dict(C.GRAMMAR, seed=0), init='variance'))
    with torch.no_grad():
        model.C.copy_(new.C.to(model.C.device))
        if variant == 'WC':
            model.W[-1].copy_(new.W[-1].to(model.C.device))
    return new.C


def pretrain(a, unit, name, c, rules, lr, steps, batch_fn, evals, gate_key):
    """Train from the current state, or reload a finished earlier attempt. Returns (state, result, gate passed)."""
    d = unit / name
    f = d / 'result.json'
    model = C.build(c).to(a.device)
    if f.exists() and json.load(open(f))['status'] == 'finished':
        res = json.load(open(f))
        model.load_state_dict(torch.load(d / f"ckpt_{res['steps_done']}.pt")['state'])
    else:
        d.mkdir(parents=True, exist_ok=True)
        t = time.time()
        log, status, steps_done, ms = T.train(model, c, lr, steps, batch_fn, evals, run_dir=d, deadline=a.deadline)
        res = dict(phase=name, cfg=c, lr=lr, status=status, steps_done=steps_done, ms_per_step=ms,
                   seconds=time.time() - t, log=log, schedule_total=steps, warmup=T.WARMUP)
        C.save(d / f'ckpt_{steps_done}.pt', model, c, steps_done)
        T.dump(f, res)
    passed = res['status'] == 'finished' and res['log'][f'{gate_key}_acc'][-1] >= a.gate
    return copy.deepcopy(model.state_dict()), res, passed


def run_unit(a, spec):
    c, lr = spec['cfg'], spec['lr']
    g, seed = c['grammar'], c['seed']
    unit = Path(a.out) / spec['name']
    unit.mkdir(parents=True, exist_ok=True)
    rules = {l: r.to(a.device) for l, r in C.rules(g).items()}
    seen, redraws = T.split_root_productions(C.rules(g), SPLIT_SEED + g['seed'])
    evals = diag_sets(a, g, rules, seen)
    torch.manual_seed(0)
    init_state = copy.deepcopy(C.build(c).to(a.device).state_dict())
    sampler = lambda rl, allowed: (lambda B, gen: T.sample_restricted(B, rl, allowed, gen, a.device))
    data_seed = T.ONLINE_SEED + 1000 * g['seed'] + 500 + seed
    meta = dict(cfg=c, lr=lr, seen=seen.tolist(), split_redraws=redraws, root_productions=C.rules(g)[0].tolist(),
                pre=spec['pre'], post=spec['post'], gate=a.gate, device=T.device_note(a.device))
    # pretraining on seen productions of grammar g, and on all productions of another grammar
    pre_state, pre, gate = pretrain(a, unit, 'pretrain', c, rules, lr, spec['pre'],
                                    T.online_batches(c, rules, a.device, data_seed, sampler(rules, seen)), evals, 'seen')
    go = T.grammar_cfg(OTHER + g['seed'])
    co = dict(c, grammar=go)
    rules_o = {l: r.to(a.device) for l, r in C.rules(go).items()}
    eval_o = {'own': T.evalset(C.diag_trees(go, a.n_diag), co, a.device)}
    other_state, pre_o, gate_o = pretrain(a, unit, 'pretrain_other', co, rules_o, lr, spec['pre'],
                                          T.online_batches(co, rules_o, a.device, data_seed + 7), eval_o, 'own')
    print(f"{spec['name']} pretrain: seen acc {pre['log']['seen_acc'][-1]:.4f} (gate {'passed' if gate else 'FAILED'}); "
          f"other grammar: own acc {pre_o['log']['own_acc'][-1]:.4f} (gate {'passed' if gate_o else 'FAILED'})", flush=True)
    if not gate:
        T.dump(unit / 'unit.json', dict(meta, gate_passed=False, other_gate_passed=gate_o, arms={}))
        return 'gate_failed' if pre['status'] == 'finished' else pre['status']
    arms = {}
    xg = evals['unseen'][0][:4096]
    for arm in ARMS:
        f = unit / arm / 'result.json'
        if f.exists() and json.load(open(f))['status'] == 'finished':
            arms[arm] = 'finished'
            continue
        if 'other' in arm and not gate_o:
            arms[arm] = 'dropped (other-grammar pretraining failed its gate)'
            continue
        (unit / arm).mkdir(exist_ok=True)
        info = {}
        if arm == 'oracle_C':
            m = T.OracleTop(C.rules(g), g['v'], R, fresh(C.build(c), 'C', seed)).to(a.device)
            params = list(m.parameters())
        else:
            m = C.build(c).to(a.device)
            m.load_state_dict(pre_state if arm in ('finetune', 'frozen_learned_C', 'frozen_learned_WC') else
                              other_state if 'other' in arm else init_state)
            params = list(m.parameters())
            if arm.startswith('frozen'):
                variant = arm.split('_')[-1]
                fresh(m, variant, seed)
                info = T.rescale_frozen(m, variant, xg)
                params = T.freeze(m, FROZEN[variant])
        t = time.time()
        log, status, steps, ms = T.train(m, c, lr, spec['post'],
                                         T.online_batches(c, rules, a.device, data_seed + 50, sampler(rules, ~seen)),
                                         evals, params=params, deadline=a.deadline)
        torch.save({'arm': arm, 'step': steps, 'state': {k: v.cpu() for k, v in m.state_dict().items()}},
                   unit / arm / f'ckpt_{steps}.pt')
        T.dump(f, dict(meta, phase='post', arm=arm, status=status, steps_done=steps, ms_per_step=ms, rescale=info,
                       n_trainable=sum(p.numel() for p in params), seconds=time.time() - t, log=log,
                       schedule_total=spec['post'], warmup=T.WARMUP))
        arms[arm] = status
        print(f"{spec['name']} {arm}: {status}, unseen acc {log['unseen_acc'][-1]:.4f}, seen acc {log['seen_acc'][-1]:.4f}, "
              f"{ms:.2f} ms/update", flush=True)
    T.dump(unit / 'unit.json', dict(meta, gate_passed=True, other_gate_passed=gate_o, arms=arms))
    return 'finished' if all(s == 'finished' for s in arms.values()) else 'partial: ' + json.dumps(arms)


def main():
    a = T.args(hours=3.0)
    if a.smoke:
        pre, post, pilot_steps, G3, seeds, a.gate = 300, 200, 100, (0,), (0,), 0.0  # smoke: no gate, so the arms run
    else:
        pre, post, pilot_steps, G3, seeds, a.gate = PRE, POST, PILOT_STEPS, (0, 1, 2), (0, 1, 2), GATE
    T.prepare(a, [T.grammar_cfg(0)])  # for the pilot only (it trains and evaluates on the full grammar)
    pilot = dict(name='root', cfg=cfg(0, T.PILOT_SEED), data='online', data_seed=T.ONLINE_SEED + 400 + T.PILOT_SEED)
    lr = T.pilot(a, {'root': pilot}, pilot_steps, 40_000 if not a.smoke else 300)['root']
    T.start_clock(a)
    specs = [dict(name=f'g{g}_s{s}', cfg=cfg(g, s), lr=lr, pre=pre, post=post) for s in seeds for g in G3]
    T.run_all(a, specs, run_unit)


if __name__ == '__main__':
    main()
