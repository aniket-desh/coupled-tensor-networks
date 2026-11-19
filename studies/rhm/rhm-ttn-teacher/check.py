"""The exact aligned TreeTN teacher for the RHM root task, and the population objective it induces. Checks for NOTE.md.

  PYTHONPATH=. python studies/rhm/rhm-ttn-teacher/check.py        # writes results/check.{json,txt}, exits 1 on failure

Teacher (binary grammar, depth L, bond dimension r = v). Grammar levels are root-first (level 0 = root, level L =
tokens); TreeTN levels are leaf-first (level j holds the 2^(L-j) bonds made of grammar level L - j). So
  A        = I_v                                      leaf token c -> e_c
  W[j]_abc = 1[ inv[L-1-j](b, c) = a ],  j = 0..L-2   TreeTN core j fuses grammar level L-j into level L-j-1
  C_obc    = 1[ inv[0](b, c) = o ]                    root core, n outputs
with inv[l] = grammar.inverse (the parent of an ordered pair at grammar level l, -1 if no symbol produces it, in which
case the core's response is 0). One table per level, so the teacher is a tie='level' TreeTN; untied, every node of a
level gets a copy.

Checked here, by enumeration on tiny grammars (float64):
  1. teacher(x) = e_root(x) = posterior_root(x) on every legal string, = 0 on every other string of the product space,
     and = n m^(2^L - 1) P(x, root) everywhere (the generative joint, from grammar.likelihoods);
  2. the input law: every derivation has probability 1/(n m^(2^L-1)) and derivations -> strings is injective, so x is
     uniform on the support S, |S| = n m^(2^L-1);
  3. for a fixed generic student, the squared loss averaged over derivations (one-hot labels) equals
     ||P_S (T - T*)||_F^2 / (2|S|), and so do all parameter gradients;
  4. the uniform-product objective ||T - T*||_F^2 / (2|X|) with the zero-filled teacher equals
     (|S|/|X|) L_S + sum_{x not in S} ||T(x)||^2 / (2|X|), loss and gradients, and its gradient is not parallel to
     that of L_S; a teacher whose forbidden-pair entries are filled with noise has L_S = 0 and grad L_S = 0 exactly
     but L_X > 0 and grad L_X != 0;
  5. cross-entropy: population CE over derivations = mean KL(p_Bayes || softmax f) on S; at f = beta * teacher it is
     log(1 + (n-1) e^-beta) (> 0 for every finite beta), with accuracy 1.
And on the main grammar (v = n = 16, m = 4, L = 4): the teacher built through common.build (tie='level' and untied,
r = 16) gives the one-hot root on the 20,000 diagnostic trees and 0 on strings with one token changed that no longer
parse."""
import itertools
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ctn.tree import TreeTN
from studies.rhm import common as C
from studies.rhm import grammar as G
from studies.rhm.checks import brute_force

HERE = Path(__file__).resolve().parent
results, lines = {}, []


def check(name, ok, detail=''):
    results[name] = dict(ok=bool(ok), detail=str(detail))
    lines.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print(lines[-1], flush=True)


def teacher_cores(rules, v):
    """(A, [W_0 .. W_{L-2}], C) as in the module docstring: A (v, v), W_j (v, v, v), C (n, v, v), float64."""
    L, inv = len(rules), G.inverse(rules, v)
    table = lambda l: F.one_hot(inv[l].clamp_min(0), rules[l].shape[0]).double() * (inv[l] >= 0)[:, None]  # (v^2, k_l)
    core = lambda l: table(l).T.reshape(rules[l].shape[0], v, v)  # [a, b, c] = 1[inv_l(b, c) = a]
    return torch.eye(v, dtype=torch.float64), [core(L - 1 - j) for j in range(L - 1)], core(0)


def teacher(rules, v, tie='level', model=None):
    """The teacher as a TreeTN of bond dimension v (a new one, or `model` filled in place)."""
    L, n = len(rules), rules[0].shape[0]
    assert rules[0].shape[2] == 2, 'binary grammars only'
    A, Ws, Cc = teacher_cores(rules, v)
    m = model if model is not None else TreeTN(2 ** L, v, v, n, tie=tie).double()
    with torch.no_grad():
        m.A.copy_(A.expand_as(m.A))
        for W, Wt in zip(m.W, Ws):
            W.copy_(Wt.expand_as(W))
        m.C.copy_(Cc)
    return m


def grads(loss, model):
    return torch.cat([g.flatten() for g in torch.autograd.grad(loss, list(model.parameters()), retain_graph=True)])


def tiny(v, n, m, L, seed):
    tag = f'v{v}n{n}m{m}L{L}'
    s, nl = 2, 2 ** L
    rules = G.sample_rules(v, n, m, s, L, seed=seed)
    T = teacher(rules, v)
    X = torch.tensor(list(itertools.product(range(v), repeat=nl)))  # the full product space, (v^nl, nl)
    root = G.parse(X, rules, v)[0][:, 0]  # -1 off the support
    on = root >= 0
    S, nS, nX = X[on], int(on.sum()), len(X)
    size = n * m ** (nl - 1)

    # 1. teacher against parse, posterior_root and the generative joint
    with torch.no_grad():
        tX = T(X)
    onehot = F.one_hot(root.clamp_min(0), n).double() * on[:, None]
    post = G.posterior_root(S, rules, v)
    joint = G.likelihoods(X, rules, v)[0][:, 0] / n  # P(x, root): likelihood times the uniform root prior
    e1 = (tX - onehot).abs().max().item()
    e2 = (tX[on] - post).abs().max().item()
    e3 = (tX - size * joint).abs().max().item()
    check(f'{tag}: teacher = e_root on S (parse), = posterior_root on S, = 0 off S, = n m^(2^L-1) P(x, root) on X',
          max(e1, e2, e3) == 0.0 or max(e1, e2, e3) < 1e-12,
          f'max errors {e1:.1e}, {e2:.1e}, {e3:.1e}; |S| = {nS}, |X| = {nX}')
    Tu = teacher(rules, v, tie='none')
    with torch.no_grad():
        check(f'{tag}: untied copy of the teacher is the same function', (Tu(X) - tX).abs().max().item() == 0.0)

    # 2. the input law, from the enumerated derivations
    der = brute_force(rules, v, n, m, s, L)  # [(tokens, root, prob)]
    toks = {t for t, _, _ in der}
    pd = torch.tensor([p for _, _, p in der], dtype=torch.float64)
    check(f'{tag}: {len(der)} derivations, each of probability 1/(n m^(2^L-1)), map one-to-one onto S',
          len(der) == size == len(toks) == nS and toks == {tuple(t) for t in S.tolist()}
          and (pd - 1.0 / size).abs().max().item() < 1e-15, f'n m^(2^L-1) = {size}')

    # 3. squared loss and gradients for a fixed generic student: derivations vs the support-restricted Frobenius form
    xd, yd = torch.tensor([t for t, _, _ in der]), torch.tensor([r for _, r, _ in der])
    for tie, r in (('none', 3), ('level', 2)):
        st = TreeTN(nl, v, r, n, tie=tie, seed=11).double()
        L_der = (pd * 0.5 * ((st(xd) - F.one_hot(yd, n).double()) ** 2).sum(1)).sum()
        L_S = ((st(S) - tX[on]) ** 2).sum() / (2 * nS)
        g_der, g_S = grads(L_der, st), grads(L_S, st)
        check(f'{tag}: student (tie={tie}, r={r}): E_derivations 1/2 |T(x) - e_y|^2 = |P_S(T - T*)|_F^2 / (2|S|), '
              f'loss and gradient',
              abs(L_der.item() - L_S.item()) < 1e-12 * max(1, L_S.item())
              and (g_der - g_S).abs().max().item() < 1e-12 * max(1, g_S.abs().max().item()),
              f'loss {L_S.item():.6g}, |diff| {abs(L_der.item() - L_S.item()):.1e}; grad norm {g_S.norm().item():.4g}, '
              f'max |diff| {(g_der - g_S).abs().max().item():.1e} over {len(g_S)} parameters')

        # 4. the uniform-product objective with the zero-filled teacher
        out = st(X)
        L_X = ((out - tX) ** 2).sum() / (2 * nX)
        L_off = (out[~on] ** 2).sum() / (2 * nX)
        g_X, g_off = grads(L_X, st), grads(L_off, st)
        cos = F.cosine_similarity(g_X, g_S, dim=0).item()
        check(f'{tag}: student (tie={tie}, r={r}): L_X = (|S|/|X|) L_S + sum_(x not in S) |T(x)|^2 / (2|X|), loss and '
              f'gradient; grad L_X not parallel to grad L_S',
              abs(L_X.item() - (nS / nX * L_S.item() + L_off.item())) < 1e-12 * max(1, L_X.item())
              and (g_X - (nS / nX * g_S + g_off)).abs().max().item() < 1e-12 * max(1, g_X.abs().max().item())
              and abs(cos) < 0.999,
              f'L_S {L_S.item():.5g}, L_X {L_X.item():.5g}, off-support share of L_X {L_off.item() / L_X.item():.3f}, '
              f'of |grad L_X| {g_off.norm().item() / g_X.norm().item():.3f}; cos(grad L_X, grad L_S) = {cos:.3f}')

    # 4b. counterexample: fill the teacher's forbidden-pair entries with noise
    Tf = teacher(rules, v)
    inv = G.inverse(rules, v)
    gen = torch.Generator().manual_seed(5)
    free = 0
    with torch.no_grad():
        for core, l in [(W, L - 1 - j) for j, W in enumerate(Tf.W)] + [(Tf.C, 0)]:
            bad = (inv[l] < 0).reshape(v, v)  # forbidden ordered pairs at grammar level l
            noise = torch.randn(core.shape, generator=gen, dtype=torch.float64)
            core += noise * bad  # broadcasts over the leading axes
            free += int(bad.sum()) * core.shape[-3]
    oS, oX = Tf(S), Tf(X)
    L_S = ((oS - tX[on]) ** 2).sum() / (2 * nS)
    L_X = ((oX - tX) ** 2).sum() / (2 * nX)
    g_S, g_X = grads(L_S, Tf), grads(L_X, Tf)
    check(f'{tag}: teacher with its {free} forbidden-pair entries filled with N(0, 1) noise: L_S = 0 and grad L_S = 0 '
          f'exactly, L_X > 0 and grad L_X != 0',
          L_S.item() == 0.0 and g_S.abs().max().item() == 0.0 and L_X.item() > 1e-3 and g_X.norm().item() > 1e-3,
          f'L_S {L_S.item():.1e}, |grad L_S| {g_S.norm().item():.1e}, L_X {L_X.item():.4g}, |grad L_X| '
          f'{g_X.norm().item():.4g}; nonzero outputs on {int((oX[~on].abs().sum(1) > 0).sum())} of {nX - nS} '
          f'off-support strings')

    # 5. cross-entropy
    st = TreeTN(nl, v, 3, n, tie='none', seed=11).double()
    with torch.no_grad():
        ce_der = (pd * F.cross_entropy(st(xd), yd, reduction='none')).sum().item()
        lq = F.log_softmax(st(S), -1)
        kl = (torch.xlogy(post, post) - post * lq).sum(1).mean().item()
        errs, accs = [], []
        for beta in (1.0, 5.0, 20.0):
            lo = beta * tX[on]
            ce = F.cross_entropy(lo, root[on]).item()
            errs.append(abs(ce - math.log1p((n - 1) * math.exp(-beta))))
            accs.append((lo.argmax(1) == root[on]).double().mean().item())
    check(f'{tag}: cross-entropy over derivations = mean KL(p_Bayes || softmax T) on S; at beta * teacher it is '
          f'log(1 + (n-1) e^-beta), accuracy 1',
          abs(ce_der - kl) < 1e-12 and max(errs) < 1e-12 and min(accs) == 1.0,
          f'CE {ce_der:.6f}, |CE - KL| {abs(ce_der - kl):.1e}; beta = 1, 5, 20 errors {errs[0]:.1e}, {errs[1]:.1e}, '
          f'{errs[2]:.1e}')


def main_grammar():
    """The teacher inside the experiments' model class: common.build with r = 16 on the main grammar."""
    for gseed in range(3):
        g = dict(C.GRAMMAR, seed=gseed)
        rules, v, L = C.rules(g), g['v'], g['L']
        trees = C.diag_trees(g)
        x, y = trees[L], trees[0][:, 0]
        # one token changed at random; keep the strings that no longer parse
        gen = torch.Generator().manual_seed(gseed)
        xb = x.clone()
        pos = torch.randint(0, x.shape[1], (len(x),), generator=gen)
        xb[torch.arange(len(x)), pos] = (xb[torch.arange(len(x)), pos] + torch.randint(1, v, (len(x),), generator=gen)) % v
        bad = G.parse(xb, rules, v)[0][:, 0] < 0
        for tie in ('level', 'none'):
            cfg = dict(model='tree', r=v, tie=tie, perm=None, seed=0, task='root', grammar=g)
            T = teacher(rules, v, model=C.build(cfg).double())
            with torch.no_grad():
                out, outb = T(x), T(xb[bad])
            check(f'main grammar seed {gseed}, common.build tie={tie}, r = {v}: teacher = one-hot root on the diagnostic '
                  f'trees, 0 on one-token corruptions that do not parse',
                  (out - F.one_hot(y, g['n']).double()).abs().max().item() == 0.0 and outb.abs().max().item() == 0.0,
                  f'{len(x)} trees, {int(bad.sum())} corrupted strings off the support, '
                  f'{sum(p.numel() for p in T.parameters())} parameters')
        size, nX = g['n'] * g['m'] ** (2 ** L - 1), v ** (2 ** L)
        if gseed == 0:
            lines.append(f"main grammar: |S| = n m^(2^L-1) = {size} = 2^{math.log2(size):.0f}, |X| = v^(2^L) = "
                         f"2^{math.log2(nX):.0f}, |S|/|X| = 2^{math.log2(size / nX):.0f}")
            print(lines[-1])


if __name__ == '__main__':
    torch.set_num_threads(1)
    tiny(3, 3, 2, 3, seed=1)   # 8 leaves, |X| = 6561, |S| = 384
    tiny(4, 2, 2, 2, seed=3)   # 4 leaves, n != v, |X| = 256, |S| = 16
    main_grammar()
    bad = [k for k, r in results.items() if not r['ok']]
    lines.append(f'{len(results) - len(bad)}/{len(results)} checks passed' + (f'; FAILED: {bad}' if bad else ''))
    print(lines[-1])
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'check.json').write_text(json.dumps(results, indent=1))
    (HERE / 'results' / 'check.txt').write_text('\n'.join(lines) + '\n')
    sys.exit(1 if bad else 0)
