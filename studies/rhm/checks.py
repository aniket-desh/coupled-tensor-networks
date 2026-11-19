"""Correctness checks for the RHM experiments; exits non-zero on any failure. Optional argument: a JSON path for the
results. Run before every experiment: python -m studies.rhm.checks [out.json]"""
import itertools
import json
import sys
import tempfile
from pathlib import Path

import torch

from ctn.tree import n_params
from studies.rhm import common as C
from studies.rhm import grammar as G
from studies.rhm import train as T

torch.set_num_threads(1)
results = {}


def check(name, ok, detail=''):
    results[name] = dict(ok=bool(ok), detail=str(detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def brute_force(rules, v, n, m, s, L):
    """Every tree of a tiny grammar with its probability: [(tokens tuple, root, prob)]."""
    out = []
    for root in range(n):
        layers = [[((root,), 1.0 / n)]]
        for l in range(L):
            nxt = []
            for syms, p in layers[-1]:
                for ks in itertools.product(range(m), repeat=len(syms)):
                    child = sum((rules[l][a, k].tolist() for a, k in zip(syms, ks)), [])
                    nxt.append((tuple(child), p / m ** len(syms)))
            layers.append(nxt)
        out += [(tok, root, p) for tok, p in layers[-1]]
    return out


def tiny_posteriors():
    v, n, m, s, L = 4, 3, 2, 2, 2
    rules = G.sample_rules(v, n, m, s, L, seed=3)
    trees = brute_force(rules, v, n, m, s, L)
    tok = sorted({t for t, _, _ in trees})
    x = torch.tensor(tok)
    joint = torch.zeros(len(tok), n, dtype=torch.float64)
    for t, root, p in trees:
        joint[tok.index(t), root] += p
    pr = joint / joint.sum(1, keepdim=True)
    err_root = (G.posterior_root(x, rules, v) - pr).abs().max().item()
    # P(x_last | others): marginal of each full token tuple, normalized over the last token.
    px = joint.sum(1)
    ref = torch.zeros(len(tok), v, dtype=torch.float64)
    for i, t in enumerate(tok):
        for c in range(v):
            tt = t[:-1] + (c,)
            if tt in tok:
                ref[i, c] = px[tok.index(tt)]
    ref = ref / ref.sum(1, keepdim=True)
    err_tok = (G.posterior_token(x, rules, v, s ** L - 1) - ref).abs().max().item()
    # With the last leaf unobserved the root posterior is not a point mass: P(root | tokens 0..s^L-2).
    pre = sorted({t[:-1] for t in tok})
    jr = torch.zeros(len(pre), n, dtype=torch.float64)
    for t, root, p in trees:
        jr[pre.index(t[:-1]), root] += p
    xr = torch.tensor([t + (0,) for t in pre])
    err_marg = (G.posterior_root(xr, rules, v, unknown=(s ** L - 1,)) - jr / jr.sum(1, keepdim=True)).abs().max().item()
    check('posterior_root with an unobserved leaf vs brute force', err_marg < 1e-12,
          f'max err {err_marg:.2e}, mean max-posterior {(jr / jr.sum(1, keepdim=True)).max(1).values.mean():.3f}')
    total = sum(p for _, _, p in trees)
    check('posterior_root vs brute force (tiny grammar)', err_root < 1e-12, f'max err {err_root:.2e}')
    check('posterior_token vs brute force (tiny grammar)', err_tok < 1e-12, f'max err {err_tok:.2e}, total prob {total:.12f}')


def parse_and_tasks():
    g = T.grammar_cfg(0)
    rules = C.rules(g)
    trees = G.sample_trees(500, rules, seed=1)
    lat = G.parse(trees[g['L']], rules, g['v'])
    check('parse recovers every latent level', all(torch.equal(lat[l], trees[l]) for l in range(g['L'] + 1)))
    x, y = C.task_batch(trees, dict(task='last', grammar=g))
    ok = (x[:, -1] == g['v']).all() and torch.equal(x[:, :-1], trees[g['L']][:, :-1]) and torch.equal(y, trees[g['L']][:, -1])
    check("task_batch 'last': leaf 15 is the mask token, target is token 15", ok)
    x, y = C.task_batch(trees, dict(task='root', grammar=g))
    check("task_batch 'root': all tokens in, root out", torch.equal(x, trees[g['L']]) and torch.equal(y, trees[0][:, 0]))
    d1, d2 = C.diag_trees(g, 300), C.diag_trees(g, 300)
    check('diag_trees is deterministic', all(torch.equal(d1[k], d2[k]) for k in d1))
    # Bayes posterior of the last token puts mass only on tokens consistent with a valid parse, and contains the truth.
    pb = T.bayes_last(trees[g['L']][:200], rules, g['v'], 'cpu')
    check('Bayes last-token posterior is normalized and covers the true token',
          (pb.sum(1) - 1).abs().max() < 1e-5 and (pb.gather(1, trees[g['L']][:200, -1:]) > 0).all())


def checkpoints():
    g = T.grammar_cfg(0)
    trees = G.sample_trees(64, C.rules(g), seed=2)
    cfgs = [dict(model='tree', r=8, tie='none', perm=None, seed=1, task='root', grammar=g),
            dict(model='tree', r=8, tie='level', perm=None, seed=1, task='last', grammar=g),
            dict(model='tree', r=8, tie='none', perm=list(torch.randperm(16, generator=torch.Generator().manual_seed(0)).tolist()),
                 seed=1, task='root', grammar=g),
            dict(model='chain', r=8, seed=1, task='last', grammar=g),
            dict(model='tree', r=8, tie='none', perm=None, seed=1, task='root', grammar=g, init='identity'),
            dict(model='chain', r=8, seed=1, task='root', grammar=g, init='identity')]
    with tempfile.TemporaryDirectory() as d:
        for cfg in cfgs:
            m = C.build(cfg)
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.01 * torch.randn_like(p))  # not the init: the load must restore trained weights
            x, _ = C.task_batch(trees, cfg)
            C.save(Path(d) / 'ckpt_7.pt', m, cfg, 7)
            m2, cfg2, step = C.load(Path(d) / 'ckpt_7.pt')
            ok = torch.equal(m(x), m2(x)) and cfg2 == cfg and step == 7
            check(f"checkpoint round trip reproduces logits ({cfg['model']}, tie={cfg.get('tie')}, "
                  f"perm={cfg.get('perm') is not None}, init={cfg.get('init', 'variance')})", ok)


def training_step():
    g = T.grammar_cfg(0)
    rules = C.rules(g)
    for task in ('root', 'last'):
        cfg = dict(model='tree', r=8, tie='none', perm=None, seed=0, task=task, grammar=g)
        m = C.build(cfg)
        x, y = C.task_batch(G.sample_trees(T.BATCH, rules, seed=5), cfg)
        log, status, steps, _ = T.train(m, cfg, 1e-2, 60, lambda s: (x, y), {'b': (x, y, None)}, log_at=[0, 60])
        l0, l1 = log['b_loss'][0], log['b_loss'][-1]
        check(f'training decreases the loss on a fixed batch ({task})', status == 'finished' and l1 < 0.9 * l0,
              f'{l0:.3f} -> {l1:.3f}')


def freezing():
    g = T.grammar_cfg(0)
    rules = C.rules(g)
    seen, tries = T.split_root_productions(rules, seed=0)
    R0 = rules[0]
    sides_ok = all(set(R0[~seen][:, j].tolist()) <= set(R0[seen][:, j].tolist()) for j in range(2))
    check('production split: one unseen per class, unseen children occur on the same side in seen productions',
          ((~seen).sum(1) == 1).all() and sides_ok, f'{int(seen.sum())} seen, {int((~seen).sum())} unseen, {tries} redraws')
    gen = torch.Generator().manual_seed(0)
    for name, allowed in (('seen', seen), ('unseen', ~seen)):
        t = T.sample_restricted(400, rules, allowed, gen, 'cpu')
        lat = G.parse(t[g['L']], rules, g['v'])
        ok = torch.equal(lat[0], t[0]) and allowed[t[0][:, 0], t[('rule', 0)][:, 0]].all()
        check(f'restricted sampler ({name}): only allowed root productions, trees parse', ok)
    cfg = dict(model='tree', r=8, tie='none', perm=None, seed=0, task='root', grammar=g)
    x, y = C.task_batch(G.sample_trees(T.BATCH, rules, seed=6), cfg)
    for frozen in (['leaf', 'L1', 'L2', 'L3'], ['leaf', 'L1', 'L2']):
        m = C.build(cfg)
        before = {k: [p.detach().clone() for p in ps] for k, ps in m.groups().items()}
        params = T.freeze(m, frozen)
        T.train(m, cfg, 1e-2, 30, lambda s: (x, y), {}, params=params, log_at=[])
        same = all(torch.equal(p, q) for k in frozen for p, q in zip(m.groups()[k], before[k]))
        moved = all(not torch.equal(p, q) for k in m.groups() if k not in frozen for p, q in zip(m.groups()[k], before[k]))
        check(f'freezing {frozen}: frozen bitwise unchanged, the rest moved', same and moved,
              f'{sum(p.numel() for p in params)} of {n_params(m)} parameters trainable')


def transfer_parts():
    g = T.grammar_cfg(0)
    rules = C.rules(g)
    cfg = dict(model='tree', r=16, tie='none', perm=None, seed=0, task='root', grammar=g, init='identity')
    x, y = C.task_batch(G.sample_trees(512, rules, seed=7), cfg)
    for variant in ('C', 'WC'):
        m = C.build(cfg)
        with torch.no_grad():
            m.W[-1 if variant == 'C' else -2].mul_(5.0)  # badly scaled frozen part
        info = T.rescale_frozen(m, variant, x)
        rms = info['rms_level3'] if variant == 'C' else info['rms_level2']
        check(f'rescale_frozen ({variant}): fresh cores receive unit-RMS bonds', abs(rms - 1) < 1e-4,
              f"rho {info['rho']:.3f} -> rms {rms:.6f}")
    tr = G.sample_trees(256, rules, seed=8)
    o = T.OracleTop(rules, g['v'], 16, torch.zeros(16, 16, 16))
    z_ok = torch.equal(G.parse(tr[g['L']], rules, g['v'])[1], tr[1])
    with torch.no_grad():
        o.C.zero_()
        Lk = torch.zeros(16, 16, 16)
        Lk[tr[0][:, 0], tr[1][:, 0], tr[1][:, 1]] = 1.0  # a lookup table reading off the root from the true symbol pair
        Q = o.code / 4.0  # sqrt(r) = 4: o.code = 4 Q, Q orthogonal
        o.C.copy_(Q @ Lk @ Q.T / 16)  # the lookup expressed in the dense code: z_a^T C z_b = Lk[a, b]
    acc = (o(tr[g['L']]).argmax(1) == tr[0][:, 0]).float().mean().item()
    check('oracle top: level-1 symbols from the parse, a lookup C classifies every tree', z_ok and acc == 1.0, f'acc {acc}')


def schedule():
    w = T.WARMUP
    lr = [T.lr_at(s, 1.0, 40000) for s in (0, w - 1, w, 40000)]
    ok = abs(lr[0] - 1 / w) < 1e-12 and abs(lr[1] - 1) < 1e-12 and abs(lr[2] - 1) < 1e-12 and abs(lr[3] - 0.1) < 1e-12
    check(f'LR schedule: warmup to peak at {w}, cosine to 0.1x at the end', ok, lr)
    check('checkpoint steps for 40,000 updates are common.CHECKPOINTS', T.ckpt_steps(40000) == C.CHECKPOINTS)


def all_trees(rules, n, m, s, L):
    """Every tree of a tiny grammar as ({level: symbols tuple}, prob)."""
    out = [({0: (r,)}, 1.0 / n) for r in range(n)]
    for l in range(L):
        nxt = []
        for tree, p in out:
            for ks in itertools.product(range(m), repeat=len(tree[l])):
                child = sum((rules[l][a, k].tolist() for a, k in zip(tree[l], ks)), [])
                nxt.append(({**tree, l + 1: tuple(child)}, p / m ** len(ks)))
        out = nxt
    return out


def stack_beliefs():
    """stack_belief against direct enumeration of P(prefix, pending symbols), for every prefix of two tiny grammars
    (binary depth 3, ternary depth 2), plus the next-token distribution it implies against P(x_t | x_<t)."""
    for v, n, m, s, L in ((3, 3, 2, 2, 3), (3, 3, 2, 3, 2)):
        rules = G.sample_rules(v, n, m, s, L, seed=1)
        trees = all_trees(rules, n, m, s, L)
        err_b = err_next = 0.0
        for t in range(s ** L + 1):
            nodes = G.pending_nodes(t, s, L)
            sizes = [n if l == 0 else v for l, _ in nodes]
            joint, nxt = {}, {}
            for tree, p in trees:
                pre = tree[L][:t]
                idx = 0
                for (l, q), k in zip(nodes, sizes):
                    idx = idx * k + tree[l][q]
                joint.setdefault(pre, torch.zeros(max(1, int(torch.tensor(sizes).prod())), dtype=torch.float64))[idx] += p
                if t < s ** L:
                    nxt.setdefault(pre, torch.zeros(v, dtype=torch.float64))[tree[L][t]] += p
            pres = sorted(joint)
            x = torch.tensor([list(q) + [0] * (s ** L - t) for q in pres])
            ref = torch.stack([joint[q] / joint[q].sum() for q in pres])
            b = G.stack_belief(x, rules, v, t)
            err_b = max(err_b, (b - ref).abs().max().item())
            if t < s ** L:
                refn = torch.stack([nxt[q] / nxt[q].sum() for q in pres])
                err_next = max(err_next, (G.next_token_from_belief(b, rules, v, t) - refn).abs().max().item())
        check(f'stack_belief_s{s}_L{L}', err_b < 1e-12 and err_next < 1e-12,
              f'max |b - enum| = {err_b:.1e}, max |P(next) - enum| = {err_next:.1e}, {len(trees)} trees')


def grammar_rules():
    """The generator's three structural properties, for the grammars the experiments train on (v16m4s2L4 and
    v8m2s2L4, n = v, grammar seeds 0-2):

    1. unique parent: at each level the v_l * m productions are distinct s-tuples, so each legal tuple has exactly one
       parent symbol (and one production index), and `inverse` returns that parent;
    2. one table per level: in sampled trees, the children at every position p of level l are rules[l][a, k] for the
       parent a and recorded production k, with the same rules[l] at every p, and `parse` recovers every level;
    3. uniform productions: for each (level, position, parent symbol), the production index decoded from the observed
       child pair (not the recorded k) is uniform over the m allowed tuples. Pearson chi-square per cell with m - 1
       degrees of freedom; the minimum p-value must exceed 1e-3 / (number of cells) (Bonferroni), and the chi-square
       summed over all cells (df = cells * (m - 1)) must have p > 1e-3. Chi-square survival: Q(df/2, x/2)."""
    B = 200_000
    for v, m in ((16, 4), (8, 2)):
        s, L, n = 2, 4, v
        for seed in range(3):
            name = f'v{v}m{m}s{s}L{L} seed {seed}'
            rules = G.sample_rules(v, n, m, s, L, seed=seed)
            inv = G.inverse(rules, v)
            uniq = True
            for l, R in rules.items():
                code = (R * v ** torch.arange(s - 1, -1, -1)).sum(-1)  # (k, m) tuple codes
                k = R.shape[0]
                uniq &= code.unique().numel() == k * m and bool((R >= 0).all() and (R < v).all())
                uniq &= bool((inv[l][code] == torch.arange(k)[:, None]).all())  # each tuple maps back to its parent
                uniq &= int((inv[l] >= 0).sum()) == k * m  # and no other tuple is legal
            check(f'grammar {name}: each legal pair has a unique parent at its level', uniq)

            tr = G.sample_trees(B, rules, seed=100 + seed)
            same, chi_tot, df_tot, p_min, cells = True, 0.0, 0, 1.0, 0
            for l in range(L):
                a, kr = tr[l], tr[('rule', l)]  # (B, s^l)
                pairs = tr[l + 1].reshape(B, -1, s)  # (B, s^l, s): children of each node at level l
                same &= bool((pairs == rules[l][a, kr]).all())  # same table rules[l] at every position
                # Decode the production index from the observed pair: the unique k with rules[l][a, k] == pair.
                match = (rules[l][a] == pairs[:, :, None, :]).all(-1)  # (B, s^l, m)
                same &= bool((match.sum(-1) == 1).all())
                kd = match.float().argmax(-1)
                for p in range(a.shape[1]):
                    cnt = torch.zeros(rules[l].shape[0], m, dtype=torch.float64)
                    cnt.index_put_((a[:, p], kd[:, p]), torch.ones(B, dtype=torch.float64), accumulate=True)
                    tot = cnt.sum(1, keepdim=True)
                    ok_rows = tot[:, 0] > 0
                    exp = tot / m
                    chi = ((cnt - exp) ** 2 / exp)[ok_rows].sum(1)  # (parents present,)
                    pv = torch.special.gammaincc(torch.tensor((m - 1) / 2, dtype=torch.float64), chi / 2)
                    chi_tot += chi.sum().item()
                    df_tot += int(ok_rows.sum()) * (m - 1)
                    p_min = min(p_min, pv.min().item())
                    cells += int(ok_rows.sum())
            parsed = G.parse(tr[L], rules, v)
            same &= all(bool((parsed[l] == tr[l]).all()) for l in range(L + 1))
            check(f'grammar {name}: one rule table per level, reused at every position; parse recovers every level',
                  same, f'{B} trees')
            p_tot = torch.special.gammaincc(torch.tensor(df_tot / 2, dtype=torch.float64),
                                            torch.tensor(chi_tot / 2, dtype=torch.float64)).item()
            check(f'grammar {name}: each parent picks uniformly among its {m} allowed pairs',
                  p_min > 1e-3 / cells and p_tot > 1e-3,
                  f'{cells} (level, position, parent) cells, min p {p_min:.3g} (Bonferroni bound {1e-3 / cells:.1e}), '
                  f'pooled chi2 {chi_tot:.0f} on {df_tot} df, p {p_tot:.3g}')


if __name__ == '__main__':
    for f in (grammar_rules, tiny_posteriors, stack_beliefs, parse_and_tasks, checkpoints, training_step, freezing, transfer_parts, schedule):
        try:
            f()
        except Exception as e:  # a crashing check is a failed check
            check(f.__name__, False, repr(e))
    bad = [k for k, v in results.items() if not v['ok']]
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(sys.argv[1], 'w'), indent=1)
    print(f'{len(results) - len(bad)}/{len(results)} checks passed' + (f'; FAILED: {bad}' if bad else ''))
    sys.exit(1 if bad else 0)
