"""Checks of the message-passing analysis against a tree that passes the exact messages (no training).

    PYTHONPATH=. python studies/rhm/rhm-ttn-messages/msg_check.py

exact_tree builds the reference by hand: bond dimension v, leaf embedding = one-hot token, mask = all-ones, core
W_abc = P(b, c | a) from the rules, so every bond equals the exact unnormalized message. msg_analyze.analyze_model must
then return the ideal value of every measure: projective probe R2 = 1 on held-out local prefixes (partially and fully
observed, at eligible nodes); invariance ve = 1 for every grouping (the bond is a function of lam_u, and rows with the
same extended message have the same lam_u); production table accuracy and mass = 1 at parents of level >= 2 and
overall, zero norm and zero read-out mass on invalid pairs, equal read-out mass on valid pairs; mixture update R2 = 1
on every subset of pairs (seen, left-masked, both informative) and code linearity R2 = 1. The same must hold after a
random gauge transformation z -> s G z of every bond (compensated in its parent core), which also rescales each node.
A tree with random cores must fail M3 and M4, so the measures are not trivially high. Also checked directly: the bonds
equal grammar.likelihoods, exact_update equals the next level's exact message, and the extended message sums over the
next token to the message.

What this does and does not validate. The exact tree validates the analysis code against known answers. It is not a
solution of the training task: its bonds hold lam_u only, so it cannot tell apart two local prefixes with the same
message and different next-token predictions, which a tree at the Bayes loss must. That shows up here as the M1x probe
to the extended message scoring below 1 on the exact tree (checked), and it is why H2 is judged on the extended
grouping."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msg_analyze as MA  # noqa: E402
from ctn.tree import TreeTN  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

ok = True


def check(name, cond, note=''):
    global ok
    ok &= bool(cond)
    print(f'{"PASS" if cond else "FAIL"}  {name}  {note}')


def exact_tree(rules, v, L):
    T = 2 ** L
    model = TreeTN(T, v + 1, v, v, seed=0).double()
    with torch.no_grad():
        model.A.zero_()
        model.A[:, :v] = torch.eye(v, dtype=torch.float64)
        model.A[:, v] = 1.0
        for l in range(1, L):
            R = rules[L - l]
            W = torch.zeros(v, v, v, dtype=torch.float64)
            for a in range(R.shape[0]):
                W[a, R[a, :, 0], R[a, :, 1]] += 1.0 / R.shape[1]
            model.W[l - 1].copy_(W.expand_as(model.W[l - 1]))
    return model


def gauge(model, seed):
    """z_u -> M_u z_u for every bond, M_u = s_u (I + 0.4 noise) invertible, compensated in the parent core."""
    g = torch.Generator().manual_seed(seed)
    r, D = model.r, model.D
    M = {(l, k): torch.exp(torch.randn((), generator=g, dtype=torch.float64))
         * (torch.eye(r, dtype=torch.float64) + 0.4 * torch.randn(r, r, generator=g, dtype=torch.float64) / r ** 0.5)
         for l in range(D) for k in range(model.n >> l)}
    with torch.no_grad():
        for k in range(model.n):
            model.A[k] = model.A[k] @ M[(0, k)].T
        for l in range(1, D):
            for k in range(model.n >> l):
                Li, Ri = torch.linalg.inv(M[(l - 1, 2 * k)]), torch.linalg.inv(M[(l - 1, 2 * k + 1)])
                model.W[l - 1][k] = torch.einsum('ax,xyz,yb,zc->abc', M[(l, k)], model.W[l - 1][k], Li, Ri)
    return model


for v, m, L, N in ((4, 2, 4, 1500), (3, 2, 4, 1500)):
    gcfg = dict(v=v, n=v, m=m, s=2, L=L, seed=1)
    rules = G.sample_rules(v, v, m, 2, L, seed=1)
    trees = G.sample_trees(N, rules, seed=2)
    x = trees[L]
    tg = MA.targets(trees, rules, gcfg)
    tag = f'v{v}m{m}L{L}'

    model = exact_tree(rules, v, L)
    Z = MA.bonds(model, x, v)
    err = max((Z[n] - t['lam']).abs().max().item() for n, t in tg.items())
    check(f'{tag}: exact tree bonds = grammar.likelihoods messages', err < 1e-12, f'max err {err:.1e}')
    # exact_update of the children's exact messages = the parent's exact message (normalized), at a level >= 2 node
    t = 2 ** (L - 1) - 1  # node (L-1, 0) partially observed
    lik = G.likelihoods(x, rules, v, unknown=range(t, 2 ** L))
    nrm = lambda z: z / z.sum(1, keepdim=True)
    up, _ = MA.exact_update(nrm(lik[2][:, 0]), nrm(lik[2][:, 1]), rules[1])
    check(f'{tag}: exact_update(children) = parent message', (up - nrm(lik[1][:, 0])).abs().max() < 1e-12)
    # extended message: summed over the next token it is the message (up to the float32 it is stored in), and rows with
    # the same extended id have the same message id
    errx, split = 0.0, 0
    for n_, t_ in tg.items():
        if len(t_['xrows']):
            v_ = t_['lam'].shape[1]
            marg = t_['xmsgn'].double().reshape(-1, v_, v_).sum(2)
            errx = max(errx, (marg / marg.sum(1, keepdim=True) - t_['lamn'][t_['xrows']]).abs().max().item())
        pairs = torch.unique(torch.stack([t_['xid'], t_['mid']], 1), dim=0)
        split += len(pairs) - len(torch.unique(t_['xid']))
    check(f'{tag}: extended message sums over the next token to the message; it refines the message grouping',
          errx < 1e-6 and split == 0, f'max err {errx:.1e}, {split} extended ids spanning two messages')

    # the local projective probe is the shared one: same predictions on a held-out half of a partially observed node
    n0 = max(tg, key=lambda n: (n[0], -n[1]))
    zz, bb = Z[n0], tg[n0]['lamn']
    h = len(zz) // 2
    diff = (MA.fit_projective(zz[:h], bb[:h])(zz[h:], rectify=False) - MA.BC.projective_fit_predict(zz[:h], bb[:h], zz[h:])[0]).abs().max().item()
    check(f'{tag}: fit_projective = belief_common.projective_fit_predict', diff < 1e-8, f'max diff {diff:.1e}')

    for name, mdl in (('exact', model), ('exact + gauge', gauge(exact_tree(rules, v, L), 5))):
        nodes, _ = MA.analyze_model(mdl, x, tg, rules, gcfg, v, controls=True)
        P = MA.pooled(nodes)['all']
        m1 = [n['m1'] for n in nodes.values() if n['info']['eligible']]  # other nodes have too few local prefixes to fit on
        worst = lambda key: min(q[key]['R2'] for q in m1 if key in q)
        check(f'{tag} {name}: M1 projective R2 = 1 on held-out local prefixes (every eligible node)',
              len(m1) >= 2 and worst('proj_partial') > 1 - 1e-6 and worst('proj_full') > 1 - 1e-6,
              f"{len(m1)} eligible nodes, worst partial {worst('proj_partial'):.8f}, full {worst('proj_full'):.8f}; "
              f"affine partial {worst('aff_partial'):.3f} (the scale of an unnormalized message varies)")
        Phi = MA.pooled(nodes)['l2+']
        ve = [q['ve'] for n in nodes.values() for key, q in n['m2'].items() if key.endswith('_msg') or key.endswith('_xmsg')]
        nx = sum(key.endswith('_xmsg') for n in nodes.values() for key in n['m2'])
        check(f'{tag} {name}: M2 ve = 1 for the unnormalized and the extended message', min(ve) > 1 - 1e-9 and nx > 0,
              f'min {min(ve):.10f} over {len(ve)} ({nx} extended)')
        xr = [q['xproj_partial']['R2'] for q in m1 if 'xproj_partial' in q]
        check(f'{tag} {name}: M1x is computed, and the exact tree does not hold the extended message (R2 < 1)',
              len(xr) >= 1 and max(xr) < 0.999, f'extended-message projective R2 {min(xr):.3f} .. {max(xr):.3f} at {len(xr)} eligible nodes')
        for scope, m3 in (('all levels', P['m3']), ('level >= 2', Phi['m3'])):
            check(f'{tag} {name}: M3 production table exact ({scope}), zero norm and read-out on invalid pairs',
                  m3['acc_proj'] == 1 and abs(m3['mass_proj'] - 1) < 1e-4 and m3['invalid_norm_ratio'] < 1e-6
                  and m3['invalid_q_ratio'] < 1e-3 and m3['valid_q_cv'] < 1e-3,
                  f"acc {m3['acc_proj']:.3f}, mass {m3['mass_proj']:.6f}, invalid/valid norm {m3['invalid_norm_ratio']:.1e}, "
                  f"read-out mass invalid/valid {m3['invalid_q_ratio']:.1e}, valid cv {m3['valid_q_cv']:.1e}, {m3['n_valid']} valid pairs")
        r4 = {key: Phi[key]['R2'] for key in ('m4_emp_proj', 'm4_emp_seen_proj', 'm4_emp_leftmask_proj', 'm4_emp_both_proj', 'm4_dir_proj')}
        r4.update({key: P[key]['R2'] for key in ('m4a_emp_proj', 'm4a_dir_proj')})
        check(f'{tag} {name}: M4 mixture update (every subset, level >= 2) and M4a code linearity R2 = 1', min(r4.values()) > 1 - 1e-6,
              ' '.join(f'{key[3:]} {val:.7f}' for key, val in r4.items())
              + f" | additive ceiling both {Phi['m4_emp_both_addceil']['R2']:.3f} leftmask {Phi['m4_emp_leftmask_addceil']['R2']:.3f} "
              f"dir {Phi['m4_dir_addceil']['R2']:.3f}; both-informative pairs scored: "
              f"{sum(n['m4'].get('emp_both_n', 0) for n in nodes.values() if 'm4' in n)}")
        check(f'{tag} {name}: controls: shuffled ~ 0, token readout and scrambled target reported',
              all(abs(q['shuffle_all']['R2']) < 0.2 for q in m1), f"tok proj partial {P.get('m1_tok_proj_partial', {}).get('R2')}, "
              f"scr proj partial {P.get('m1_scr_proj_partial', {}).get('R2')}")

    rnd = exact_tree(rules, v, L)
    with torch.no_grad():
        for W in rnd.W:
            W.copy_(torch.randn(W.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64) / v)
    P = MA.pooled(MA.analyze_model(rnd, x, tg, rules, gcfg, v, controls=False)[0])['all']
    check(f'{tag} random cores: M3 and M4 fail', P['m3']['acc_proj'] < 0.8 and P['m4_dir_proj']['R2'] < 0.5,
          f"M3 acc {P['m3']['acc_proj']:.3f}, M4 dir R2 {P['m4_dir_proj']['R2']:.3f}, emp R2 {P['m4_emp_proj']['R2']:.3f}")
print('all pass' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
