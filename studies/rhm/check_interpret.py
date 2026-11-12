"""Checks for studies/rhm/interpret.py; exits non-zero on failure. Small and CPU-only (about a minute).

 1. gauge: a random invertible G on one bond of an untied TreeTN, with G^{-1} absorbed by the parent core, leaves the
    logits unchanged, and the whitened measures (invariance ratios, variance explained, LDA and whole-bond ablation,
    probe, patching) agree to 1e-6 between the two parameterizations;
 2. the probe reaches ~100% on a synthetic linearly separable bond;
 3. mean-ablating the whole bond of each of the root's children changes the loss;
 4. subtree masks select exactly the parameters a bond depends on, and the coupling prediction -eta <g_A, g_B> matches
    the actual change of L_A to first order as eta -> 0 (relative error shrinks ~linearly);
 5. donor selection yields grammar-valid chains: the spliced input parses at every level, to the expected root and the
    donor latent; whole-bond patching equals running the network on the spliced input;
 6. the 'last' task path runs, with Bayes ceilings <= 1 and < 1 on the masked spine.
Run: python -m studies.rhm.check_interpret"""
import sys

import torch

from ctn.tree import TreeTN
from studies.rhm import common
from studies.rhm import grammar as G
from studies.rhm import interpret as I

torch.set_num_threads(1)
FAIL = []


def check(name, ok, info=''):
    print(f'{"ok  " if ok else "FAIL"} {name} {info}')
    if not ok:
        FAIL.append(name)


def random_G(r, seed):
    g = torch.Generator().manual_seed(seed)
    Q1 = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=I.DT))[0]
    Q2 = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=I.DT))[0]
    s = torch.exp(torch.empty(r, dtype=I.DT).uniform_(-1.5, 1.5, generator=g))  # singular values in [0.22, 4.5]
    return Q1 @ torch.diag(s) @ Q2


def regauge(model, l, k, Gm):
    """Copy of an untied TreeTN with bond (l, k) -> G z and G^{-1} absorbed by its parent core."""
    m = TreeTN(model.n, model.V, model.r, model.O).to(I.DT)
    m.load_state_dict(model.state_dict())
    Gi = torch.linalg.inv(Gm)
    with torch.no_grad():
        if l == 0:
            m.A[k] = m.A[k] @ Gm.T
        else:
            m.W[l - 1][k] = torch.einsum('ia,abc->ibc', Gm, m.W[l - 1][k])
        if l < model.D - 1:
            P = m.W[l][k // 2]
            m.W[l][k // 2] = torch.einsum('abc,bj->ajc', P, Gi) if k % 2 == 0 else torch.einsum('abc,cj->abj', P, Gi)
        else:
            m.C.copy_(torch.einsum('oab,aj->ojb', m.C, Gi) if k == 0 else torch.einsum('oab,bj->oaj', m.C, Gi))
    return m


def main():
    cfg = dict(model='tree', r=8, tie='none', perm=None, seed=3, task='root', grammar=dict(common.GRAMMAR, seed=0))
    rules = common.rules(cfg['grammar'])
    trees = G.sample_trees(800, rules, seed=7)
    ctx = I.prepare(cfg, trees, rules, n_pairs=300, n_patch=300, n_group=64)
    model = common.build(cfg).to(I.DT)
    base = I.analyze(model, ctx, coupling=False)
    rec = {(nd['l'], nd['k']): nd for nd in base['nodes']}

    # 1. gauge invariance
    keys = ['inv_same', 'inv_sym', 'var_explained', 'abl_whole', 'abl_lda', 'abl_lda4', 'probe', 'pred_same',
            'patch_whole', 'patch_lda', 'patch_same_stay']
    x = ctx['x']
    for i, (l, k) in enumerate([(1, 2), (2, 1), (3, 0), (3, 1)]):
        m2 = regauge(model, l, k, random_G(cfg['r'], 100 + i))
        with torch.no_grad():
            dl = ((m2(x) - model(x)).abs().max() / model(x).abs().max()).item()
            z1, z2 = model(x, acts=True)[1][l][:, k], m2(x, acts=True)[1][l][:, k]
        check(f'gauge ({l},{k}): network function unchanged', dl < 1e-10, f'rel max |dlogits| = {dl:.1e}')
        check(f'gauge ({l},{k}): bond actually changed', (z1 - z2).abs().max().item() > 1e-2)
        r2 = {(nd['l'], nd['k']): nd for nd in I.analyze(m2, ctx, coupling=False)['nodes']}[(l, k)]
        dev = {key: abs(rec[(l, k)][key] - r2[key]) for key in keys if rec[(l, k)].get(key) is not None}
        worst = max(dev, key=dev.get)
        check(f'gauge ({l},{k}): whitened measures agree', dev[worst] < 1e-6,
              f'max dev {dev[worst]:.1e} ({worst}); inv_same {r2["inv_same"]:.4f}, var_expl {r2["var_explained"]:.4f}')

    # 2. probe on a synthetic linearly separable bond
    g = torch.Generator().manual_seed(0)
    C, d, n = 16, 8, 4000
    means = 3 * torch.randn(C, d, generator=g, dtype=I.DT)
    yy = torch.randint(0, C, (n,), generator=g)
    X = means[yy] + 0.1 * torch.randn(n, d, generator=g, dtype=I.DT)
    mu, Wh, _ = I.whitener(X @ random_G(d, 5).T)  # in an arbitrary gauge, then whitened
    U = (X @ random_G(d, 5).T - mu) @ Wh.T
    acc = I.probe_acc(U[:n // 2], yy[:n // 2], U[n // 2:], yy[n // 2:], C)
    check('probe: synthetic separable bond', acc > 0.99, f'held-out acc {acc:.4f}')

    # 3. whole-bond ablation at the root's children
    for k in (0, 1):
        a = rec[(3, k)]['abl_whole']
        check(f'ablation: whole bond (3,{k}) changes the loss', abs(a) > 1e-3, f'dloss {a:+.4f}')

    # 4. masks and coupling
    small = TreeTN(16, 16, 4, 16, seed=1).to(I.DT)
    xs = torch.randint(0, 16, (64, 16), generator=g)
    for (l, k) in [(0, 5), (1, 3), (2, 2), (3, 1)]:
        grads = torch.autograd.grad(small(xs, acts=True)[1][l][:, k].pow(2).sum(), list(small.parameters()),
                                    allow_unused=True)
        mk = I.subtree_masks(small, l, k)
        ok = True
        for (nme, p), gr in zip(small.named_parameters(), grads):
            gr = torch.zeros_like(p) if gr is None else gr
            nz = gr.flatten(1).abs().sum(1) > 0 if gr.dim() > 1 else gr.abs() > 0
            ok &= torch.equal(nz, mk[nme].flatten(1).all(1) if p.dim() > 1 else mk[nme])
        check(f'masks: subtree ({l},{k}) = parameters the bond depends on', bool(ok))
    yA, yB = torch.randint(0, 16, (64,), generator=g), torch.randint(0, 16, (64,), generator=g)
    xB = xs.clone()
    xB[:, 8:] = torch.randint(0, 16, (64, 8), generator=g)  # B shares the left half of the inputs with A
    yB[:32] = yA[:32]
    for sname, mk in I.coupling_masks(small, 2, 0).items():
        errs = [I.coupling_step(small, xs, yA, xB, yB, mk, rel=rel) for rel in (1e-2, 1e-3, 1e-4)]
        e = [abs(r['actual'] / r['predicted'] - 1) for r in errs]
        lin = all(0.05 < e[i + 1] / e[i] < 0.2 for i in range(2))  # error ~ eta: the residual is second order
        check(f'coupling: first order as eta -> 0 ({sname})', lin and e[2] < 1e-2,
              'rel err ' + ', '.join(f'{v:.1e}' for v in e) + f'; cos {errs[0]["cos"]:+.3f}')

    # 5. donors give grammar-valid chains; patching = splicing
    tok = trees[cfg['grammar']['L']]
    L, v = cfg['grammar']['L'], cfg['grammar']['v']
    for (l, k), pt in ctx['patch'].items():
        if not len(pt['rec']):
            check(f'donors ({l},{k}): none available', l < 2, f'frac with donor {pt["frac_with_donor"]:.2f}')
            continue
        h = I.splice(tok, pt['rec'], pt['donor'], l, k)
        p = G.parse(h, rules, v)
        valid = all((p[lev] >= 0).all().item() for lev in range(L + 1))
        same_root = torch.equal(p[0][:, 0], pt['expected'])
        lat_ok = torch.equal(p[L - l][:, k], trees[L - l][pt['donor'], k])
        changed = (pt['expected'] != trees[0][pt['rec'], 0]).all().item()
        with torch.no_grad():
            src = model(x[pt['donor']], acts=True)[1][l][:, k]

            def edit(lev, z, l=l, k=k, src=src):
                if lev == l:
                    z = z.clone()
                    z[:, k] = src
                return z
            dpatch = (model(x[pt['rec']], edit=edit) - model(h)).abs().max().item()
        check(f'donors ({l},{k}): spliced inputs parse to the expected root',
              valid and same_root and lat_ok and changed and dpatch < 1e-10,
              f'n {len(pt["rec"])}, frac with donor {pt["frac_with_donor"]:.2f}, |patch - splice| {dpatch:.0e}')

    # 6. 'last' task path
    cfg_l = dict(cfg, task='last')
    ctx_l = I.prepare(cfg_l, G.sample_trees(400, rules, seed=8), rules, n_pairs=100, n_patch=100, n_group=32)
    out = I.analyze(common.build(cfg_l), ctx_l)
    ce = {(nd['l'], nd['k']): nd['ceiling'] for nd in out['nodes']}
    spine = [ce[(l, 15 >> l)] for l in range(L)]
    others = [c for nd, c in ce.items() if nd[1] != 15 >> nd[0]]
    check("'last': ceilings <= 1, = 1 off the masked spine", max(spine) <= 1 and min(others) == 1,
          f'spine ceilings by level {[round(c, 3) for c in spine]}; Bayes loss {ctx_l["bayes"]["loss"]:.3f}')
    check("'last': patched KL measures computed", any(nd.get('kl_patch') is not None for nd in out['nodes'])
          and len(out['coupling']) > 0)

    print('all checks passed' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
