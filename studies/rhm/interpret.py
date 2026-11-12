"""Interpretation of a TreeTN trained on the RHM: does each bond encode its node's latent symbol, invariantly to the
surface realization, and does the network use it?

Run:  python -m studies.rhm.interpret <run dir> --out <json> [--diag <file>] [--device cuda]
It analyzes every <run dir>/ckpt_<step>.pt (an untrained model built from the config stands in for a missing step 0)
and writes one JSON with every measure per checkpoint and per node, plus per-level medians.

Correspondence. Tree level l (bond activations acts[l], l = 0 leaf bonds .. L-1 the root's two children) is grammar
level g = L - l; tree node k at level l carries the latent trees[g][:, k], which spans tokens [k 2^l, (k+1) 2^l).

Gauge. A bond can be reparameterized z -> G z with G^{-1} absorbed by the parent core, so raw coordinates mean nothing.
Every measure uses whitened bonds u = Wh (z - mu), with mu and the covariance S taken over the diagnostic set and
Wh = Lam^{-1/2} E^T over the eigendirections of S with eigenvalue > TOL * max (a relative cutoff, which is gauge
covariant whenever the spectrum has a gap there; d = number of kept directions, = r for a full-rank bond). Under
z -> G z the whitened coordinates change only by an orthogonal rotation, so distances, between-class traces,
L2-regularized probes and projector-based interventions are gauge invariant. Analysis runs in float64 on a copy of
each checkpoint; the files are never modified.

Measures, per node (l, k):
 1. probe: held-out accuracy of a multinomial logistic regression (L2 = L2 on the weights, LBFGS) from u to the latent,
    trained on half the diagnostic set. Baselines: the same probe at step 0 (untrained), with shuffled labels, and on
    the one-hot tokens the node sees ('local', additive in the tokens so it cannot form conjunctions). For the 'last'
    task, nodes whose span holds the masked leaf have a Bayes ceiling: accuracy of argmax_a P(latent = a | tokens 0..14),
    enumerating the masked token with grammar.posterior_token and parsing each completion; probe_rel = probe / ceiling.
 2. invariance (internal nodes): same = E||u(x) - u(x')||^2 / (2 d) over pairs from grammar.resample_rules at the node
    (same latent, new realization); 2 d is the expected squared distance of independent diagnostic pairs, so 0 means
    invariant and 1 means no invariance. sym: the same for grammar.resample_symbols (new latent). var_explained =
    tr(between-class covariance of u) / d. pred_same / acc_same: the prediction is unchanged / correct on x'.
 3. ablation (internal nodes): mean-ablate the top-k LDA directions of u (eigenvectors of the whitened between-class
    covariance), z -> z - Wp Q Q^T Wh (z - mu), and report the diagnostic loss increase; k = min(classes - 1, d)
    ('lda') and k = min(K_SMALL, d) ('lda4'), each against N_RAND random orthonormal k-subspaces of u ('rand',
    'rand4'), and the whole bond z -> mu ('whole').
    patching (internal nodes, identity perm only): for a recipient, a donor latent a' != a at the node such that every
    parent pair up to the root remains a production (inverse rule map) defines an expected root (required to differ
    from the recipient's root for the 'root' task). The whole donor bond is patched in ('whole') or only its
    k = min(classes-1, d) LDA subspace ('lda'); controls are same-latent donors (prediction should stay) and random
    donors. In a tree network a node's bond depends only on the tokens in its span, so whole-bond patching is exactly
    running the network on the spliced input (recipient tokens outside the span, donor tokens inside): it tests the
    network on a grammatical spliced sentence. For the 'last' task: KL(Bayes || model) for token 15 on the patched
    recipients, with the Bayes posterior of the spliced input.
 4. coupling (nodes (l, 0), l >= 1, untied trees only): group A = latent a and root rho at the node (the most frequent
    pair); B_same = latent a, root != rho; B_diff = latent != a, root != rho. For parameter subsets (cores of the subtree
    under the node, of the sibling subtree, of the path above, all) the predicted change of L_A after one SGD step on
    L_B restricted to the subset is -eta <grad L_A, grad L_B>_subset, with eta set so the subset's relative parameter
    change is REL_STEP; the step is then taken on a copy and the actual change of L_A measured.

Diagnostic set: --diag <file> (a dict of trees from grammar.sample_trees, or {'trees': ...}); otherwise
grammar.sample_trees(DIAG_N, rules, seed=DIAG_SEED). Other random choices (split, pairs, donors, groups, random
subspaces) use fixed seeds derived from SEED, so all checkpoints see the same examples."""
import argparse
import copy
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from ctn.tree import TreeTN
from studies.rhm import common
from studies.rhm import grammar as G

DIAG_SEED = 20261112  # regenerated diagnostic set: grammar.sample_trees(DIAG_N, rules, seed=DIAG_SEED)
DIAG_N = 20000
SEED = 12345  # split, resampled pairs, donors, groups, random subspaces
L2 = 1e-4  # probe weight penalty (on unit-variance whitened features), added to the mean cross-entropy
TOL = 1e-9  # whitening keeps covariance eigenvalues > TOL * largest
K_SMALL = 4
N_RAND = 5
REL_STEP = 1e-3  # coupling: ||eta * grad_B restricted|| / ||theta restricted||
DT = torch.float64


# ---------------------------------------------------------------- basic pieces

def forward(model, x, edit=None, acts=False, bs=4096):
    """model(x) in chunks (the edit must not depend on the example index)."""
    outs, A = [], None
    for i in range(0, len(x), bs):
        o = model(x[i:i + bs], acts=acts, edit=edit)
        if acts:
            o, a = o
            A = [[t] for t in a] if A is None else [A[j] + [t] for j, t in enumerate(a)]
        outs.append(o)
    logits = torch.cat(outs)
    return (logits, [torch.cat(t) for t in A]) if acts else logits


def ce(logits, y):
    return F.cross_entropy(logits, y).item()


def whitener(Z, tol=TOL):
    """Z (N, r). mu (r,), Wh (d, r) with Wh S Wh^T = I_d, Wp (r, d) with Wh Wp = I_d (Wp maps whitened back)."""
    mu = Z.mean(0)
    S = (Z - mu).T @ (Z - mu) / len(Z)
    lam, E = torch.linalg.eigh(S)
    keep = lam > tol * lam.max() if lam.max() > 0 else torch.zeros_like(lam, dtype=torch.bool)
    E, lam = E[:, keep], lam[keep]
    return mu, (E / lam.sqrt()).T, E * lam.sqrt()


def lda(U, y, C):
    """U (N, d) whitened (mean 0, identity covariance). Q (d, d): eigenvectors of the between-class covariance
    S_B = sum_c p_c m_c m_c^T, by decreasing eigenvalue; var_explained = tr S_B / d."""
    d = U.shape[1]
    if d == 0:
        return U.new_zeros(0, 0), float('nan')
    cnt = torch.bincount(y, minlength=C).to(U.dtype)
    M = torch.zeros(C, d, dtype=U.dtype, device=U.device).index_add_(0, y, U) / cnt.clamp_min(1)[:, None]
    SB = (M.T * (cnt / len(y))) @ M
    ev, Q = torch.linalg.eigh(SB)
    return Q.flip(1), (ev.sum() / d).item()


def probe_acc(Xtr, ytr, Xte, yte, C, l2=L2, iters=200):
    """Held-out accuracy of multinomial logistic regression with weight penalty l2 * ||W||^2, fit by LBFGS."""
    W = torch.zeros(Xtr.shape[1], C, dtype=Xtr.dtype, device=Xtr.device, requires_grad=True)
    b = torch.zeros(C, dtype=Xtr.dtype, device=Xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return ((Xte @ W + b).argmax(1) == yte).double().mean().item()


def latent(trees, L, l, k):
    return trees[L - l][:, k]


def expected_roots(trees, inv, v, g, k, cand):
    """cand (B, K): symbols substituted at (grammar level g, position k) of each tree. Returns the root implied by
    the inverse rule map going up, or -1 where some parent pair is not a production. Binary productions (s = 2)."""
    sym, pos = cand.clone(), k
    for lev in range(g, 0, -1):
        sib = trees[lev][:, pos ^ 1].unsqueeze(1).expand_as(sym)
        left, right = (sym, sib) if pos % 2 == 0 else (sib, sym)
        par = inv[lev - 1][left.clamp_min(0) * v + right]
        par[sym < 0] = -1
        sym, pos = par, pos // 2
    return sym


def splice(tok, rec, don, l, k):
    """Recipient tokens with the span of tree node (l, k) taken from the donors (identity perm)."""
    h = tok[rec].clone()
    h[:, k << l:(k + 1) << l] = tok[don, k << l:(k + 1) << l]
    return h


def subtree_masks(model, l, k):
    """{parameter name: bool mask} selecting the cores of the subtree under tree node (l, k) of an untied TreeTN:
    the leaf embeddings A[k 2^l : (k+1) 2^l] and, for each level l' = 1..l, W[l'-1][k 2^(l-l') : (k+1) 2^(l-l')]
    (the node's own core included)."""
    assert isinstance(model, TreeTN) and model.tie == 'none', 'per-node masks need an untied TreeTN'
    m = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    m['A'][k << l:(k + 1) << l] = True
    for lp in range(1, l + 1):
        m[f'W.{lp - 1}'][k << (l - lp):(k + 1) << (l - lp)] = True
    return m


def coupling_masks(model, l, k):
    """Masks for the coupling test: 'subtree' under (l, k), 'sibling' subtree under (l, k ^ 1), 'above' (the cores
    of the ancestors of (l, k), root core included), 'all'."""
    above = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    for lp in range(l + 1, model.D):
        above[f'W.{lp - 1}'][k >> (lp - l)] = True
    above['C'][:] = True
    return {'subtree': subtree_masks(model, l, k), 'sibling': subtree_masks(model, l, k ^ 1), 'above': above,
            'all': {n: torch.ones_like(p, dtype=torch.bool) for n, p in model.named_parameters()}}


def coupling_step(model, xA, yA, xB, yB, mask, rel=REL_STEP, eta=None):
    """Predicted and actual change of L_A after one SGD step on L_B restricted to `mask`. Works on a copy."""
    m = copy.deepcopy(model)
    params = dict(m.named_parameters())
    with torch.enable_grad():
        gA = torch.autograd.grad(F.cross_entropy(m(xA), yA), list(params.values()))
        gB = torch.autograd.grad(F.cross_entropy(m(xB), yB), list(params.values()))
    mk = [mask[n].to(p.dtype) for n, p in params.items()]
    dot = sum((a * b * w).sum() for a, b, w in zip(gA, gB, mk)).item()
    nA = math.sqrt(sum(((a * w) ** 2).sum().item() for a, w in zip(gA, mk)))
    nB = math.sqrt(sum(((b * w) ** 2).sum().item() for b, w in zip(gB, mk)))
    nth = math.sqrt(sum(((p * w) ** 2).sum().item() for p, w in zip(params.values(), mk)))
    if eta is None:
        eta = rel * nth / nB if nB > 0 else 0.0
    with torch.no_grad():
        L0 = F.cross_entropy(m(xA), yA).item()
        for p, b, w in zip(params.values(), gB, mk):
            p -= eta * b * w
        L1 = F.cross_entropy(m(xA), yA).item()
    pred, act = -eta * dot, L1 - L0
    return dict(eta=eta, predicted=pred, actual=act, ratio=act / pred if pred != 0 else None,
                sign_agree=bool(np.sign(pred) == np.sign(act)), cos=dot / (nA * nB) if nA * nB > 0 else None)


# ---------------------------------------------------------------- checkpoint-independent preparation

def load_diag(path, rules, n=None):
    if path:
        d = torch.load(path, map_location='cpu')
        trees = d['trees'] if 'trees' in d else d
        src = os.path.abspath(path)
    else:
        trees, src = G.sample_trees(DIAG_N, rules, seed=DIAG_SEED), f'grammar.sample_trees({DIAG_N}, seed={DIAG_SEED})'
    if n:
        trees = {key: t[:n] for key, t in trees.items()}
    return trees, src


def bayes_last(tok, rules, v, L, bs=1000):
    """'last' task, exact: P15 (N, v) = P(token 15 | tokens 0..14), and for each tree level l the posterior (N, v)
    of the latent of the node holding leaf 15 (by enumerating token 15 and parsing every completion)."""
    N, n = tok.shape
    P15 = torch.cat([G.posterior_token(tok[i:i + bs], rules, v, n - 1) for i in range(0, N, bs)])
    post = {l: torch.zeros(N, v, dtype=DT) for l in range(L)}
    for t in range(v):
        c = tok.clone()
        c[:, -1] = t
        p = G.parse(c, rules, v)
        for l in range(L):
            a = p[L - l][:, (n - 1) >> l]
            ok = a >= 0
            post[l][ok, a[ok]] += P15[ok, t]
    return P15, post


def prepare(cfg, trees, rules, n_pairs=2000, n_patch=2000, n_group=512, seed=SEED, device='cpu'):
    gc = cfg['grammar']
    v, L, task = gc['v'], gc['L'], cfg['task']
    assert gc['s'] == 2, 'binary trees only'
    n = 2 ** L
    perm = cfg.get('perm') or list(range(n))
    ident = list(perm) == list(range(n))
    x, y = common.task_batch(trees, cfg)
    tok = trees[L]
    N = len(x)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = torch.randperm(N, generator=g)
    ctx = dict(cfg=cfg, rules=rules, inv=G.inverse(rules, v), v=v, L=L, task=task, trees=trees, x=x, y=y, N=N,
               tr=order[:N // 2], te=order[N // 2:], ident=ident, notes=[])
    nodes = [(l, k) for l in range(L) for k in range(n >> l)]
    ctx['nodes'] = nodes
    ctx['lat'] = {nd: latent(trees, L, *nd) for nd in nodes}
    Vin = common.vocab_in(cfg)
    if not ident:
        ctx['notes'].append('perm is not the identity: tree node (l, k) is still labelled with grammar node k, which is '
                            'not the span it sees; patching is skipped')

    # Bayes references and probe ceilings.
    if task == 'last':
        P15, post = bayes_last(tok, rules, v, L)
        ctx['P15'] = P15
        ctx['bayes'] = dict(loss=-P15[torch.arange(N), y].clamp_min(1e-300).log().mean().item(),
                            acc=(P15.argmax(1) == y).double().mean().item())
    else:
        ctx['bayes'] = dict(loss=0.0, acc=1.0)
    ctx['ceiling'] = {}
    for (l, k) in nodes:
        c = 1.0
        if task == 'last' and k == (n - 1) >> l:
            te = ctx['te']
            c = (post[l][te].argmax(1) == ctx['lat'][(l, k)][te]).double().mean().item()
        ctx['ceiling'][(l, k)] = c

    # Local-token baseline: linear probe on the one-hot tokens the node sees (checkpoint independent).
    ctx['local'] = {}
    for (l, k) in nodes:
        cols = torch.as_tensor(perm[k << l:(k + 1) << l])
        X = F.one_hot(x[:, cols], Vin).flatten(1).to(device, DT)
        yl, tr, te = ctx['lat'][(l, k)].to(device), ctx['tr'].to(device), ctx['te'].to(device)
        ctx['local'][(l, k)] = probe_acc(X[tr], yl[tr], X[te], yl[te], v)

    # Invariance pairs: x' from resample_rules (same latent) and resample_symbols (new latent) at each internal node.
    sub = lambda idx: {key: t[idx] for key, t in trees.items()}
    ctx['pairs'] = {}
    for (l, k) in nodes:
        if l == 0:
            continue
        idx = ctx['te'][:n_pairs]
        t0 = sub(idx)
        ts = G.resample_rules(t0, rules, L - l, k, g=g)
        tn = G.resample_symbols(t0, rules, L - l, k, g=g)
        xs, ys = common.task_batch(ts, cfg)
        xn, _ = common.task_batch(tn, cfg)
        ctx['pairs'][(l, k)] = dict(idx=idx, x_same=xs, y_same=ys, x_sym=xn)

    # Patching donors.
    ctx['patch'] = {}
    if ident:
        root = trees[0][:, 0]
        for (l, k) in nodes:
            if l == 0:
                continue
            gl, a = L - l, ctx['lat'][(l, k)]
            pool = {s: torch.nonzero(a == s).flatten().numpy() for s in range(v)}
            rec = ctx['te'][:n_patch]
            cand = torch.arange(v).expand(len(rec), v)
            ex = expected_roots(sub(rec), ctx['inv'], v, gl, k, cand)  # (R, v)
            ok = (ex >= 0) & (cand != a[rec, None]) & torch.tensor([len(pool[s]) > 0 for s in range(v)])
            if task == 'root':
                ok &= ex != root[rec, None]
            R, D, E, S = [], [], [], []
            Rd = torch.as_tensor(rng.integers(N, size=len(rec)))
            Re = expected_roots(sub(rec), ctx['inv'], v, gl, k, a[Rd][:, None])[:, 0]
            for i, ri in enumerate(rec.tolist()):
                same = pool[a[ri].item()]
                same = same[same != ri]
                S.append(int(rng.choice(same)) if len(same) else ri)
                valid = torch.nonzero(ok[i]).flatten().numpy()
                if len(valid):
                    s = int(rng.choice(valid))
                    R.append(ri)
                    D.append(int(rng.choice(pool[s])))
                    E.append(ex[i, s].item())
            pt = dict(rec_all=rec, same=torch.tensor(S), rand=Rd, rand_expected=Re,
                      rec=torch.tensor(R, dtype=torch.long), donor=torch.tensor(D, dtype=torch.long),
                      expected=torch.tensor(E, dtype=torch.long), frac_with_donor=len(R) / len(rec))
            if task == 'last':
                for key, (rr, dd) in dict(hyb=(pt['rec'], pt['donor']), hyb_same=(rec, pt['same'])).items():
                    h = splice(tok, rr, dd, l, k)
                    pt['P_' + key] = G.posterior_token(h, rules, v, n - 1) if len(h) else torch.zeros(0, v, dtype=DT)
            ctx['patch'][(l, k)] = pt

    # Coupling groups at nodes (l, 0).
    ctx['groups'] = {}
    root = trees[0][:, 0]
    for l in range(1, L):
        a = ctx['lat'][(l, 0)]
        pair = a * cfg['grammar']['n'] + root
        av, rho = divmod(torch.bincount(pair).argmax().item(), cfg['grammar']['n'])
        pick = lambda m: torch.nonzero(m).flatten()[:n_group]
        A = pick((a == av) & (root == rho))
        Bs = pick((a == av) & (root != rho))
        Bd = pick((a != av) & (root != rho))
        ctx['groups'][l] = dict(latent=av, root=rho, A=A, B_same=Bs, B_diff=Bd)
    return ctx


# ---------------------------------------------------------------- per-checkpoint analysis

def analyze(model, ctx, device='cpu', coupling=True):
    """All measures for one model (a float64 copy is made). Returns a dict for the JSON."""
    model = copy.deepcopy(model).to(device=device, dtype=DT).eval()
    v, task, N = ctx['v'], ctx['task'], ctx['N']
    x, y = ctx['x'].to(device), ctx['y'].to(device)
    tr, te = ctx['tr'].to(device), ctx['te'].to(device)
    g = torch.Generator().manual_seed(SEED + 1)
    out = {}
    with torch.no_grad():
        logits, acts = forward(model, x, acts=True)
        base = ce(logits, y)
        pred = logits.argmax(1)
        out.update(loss=base, acc=(pred == y).double().mean().item())
        if not isinstance(model, TreeTN):
            out['note'] = 'not a TreeTN: only loss and accuracy'
            return out
        nodes = []
        for (l, k) in ctx['nodes']:
            Z = acts[l][:, k]
            mu, Wh, Wp = whitener(Z)
            d = Wh.shape[0]
            U = (Z - mu) @ Wh.T
            yl = ctx['lat'][(l, k)].to(device)
            rec = dict(l=l, k=k, d=d, ceiling=ctx['ceiling'][(l, k)], local=ctx['local'][(l, k)])
            rec['probe'] = probe_acc(U[tr], yl[tr], U[te], yl[te], v)
            ysh = yl[torch.randperm(N, generator=g).to(device)]
            rec['probe_shuffled'] = probe_acc(U[tr], ysh[tr], U[te], ysh[te], v)
            rec['probe_rel'] = rec['probe'] / rec['ceiling'] if rec['ceiling'] > 0 else None
            Q, rec['var_explained'] = lda(U, yl, v)
            if l == 0:
                nodes.append(rec)
                continue

            # 2. invariance
            P = ctx['pairs'][(l, k)]
            idx = P['idx'].to(device)
            for key in ('same', 'sym'):
                lg, a2 = forward(model, P['x_' + key].to(device), acts=True)
                dist = (((Z[idx] - a2[l][:, k]) @ Wh.T) ** 2).sum(1).mean().item()
                rec['inv_' + key] = dist / (2 * d) if d else None
                if key == 'same':
                    p2 = lg.argmax(1)
                    rec['pred_same'] = (p2 == pred[idx]).double().mean().item()
                    rec['acc_same'] = (p2 == P['y_same'].to(device)).double().mean().item()

            # 3a. ablation
            def ablate(Pop):
                def edit(lev, z):
                    if lev != l:
                        return z
                    z = z.clone()
                    z[:, k] = z[:, k] - (z[:, k] - mu) @ Pop.T
                    return z
                return ce(forward(model, x, edit=edit), y) - base

            rec['abl_whole'] = ablate(torch.eye(Z.shape[1], dtype=DT, device=device))
            for name, kk in (('', min(v - 1, d)), ('4', min(K_SMALL, d))):
                rec['k_lda' + name] = kk
                if kk == 0:
                    continue
                rec['abl_lda' + name] = ablate(Wp @ Q[:, :kk] @ Q[:, :kk].T @ Wh)
                rnd = []
                for _ in range(N_RAND):
                    R = torch.linalg.qr(torch.randn(d, kk, generator=g, dtype=DT))[0].to(device)
                    rnd.append(ablate(Wp @ R @ R.T @ Wh))
                rec['abl_rand' + name] = float(np.mean(rnd))
                rec['abl_rand' + name + '_sd'] = float(np.std(rnd))

            # 3b. patching (whole bond, and the k = min(v-1, d) LDA subspace)
            if (l, k) in ctx['patch']:
                pt = ctx['patch'][(l, k)]
                rec['patch_frac_with_donor'] = pt['frac_with_donor']
                Plda = Wp @ Q[:, :rec['k_lda']] @ Q[:, :rec['k_lda']].T @ Wh if d else None

                def patched(rr, src, Pop=None):
                    def edit(lev, z):
                        if lev != l:
                            return z
                        z = z.clone()
                        z[:, k] = src if Pop is None else z[:, k] + (src - z[:, k]) @ Pop.T
                        return z
                    return model(x[rr], edit=edit)

                rec_all = pt['rec_all'].to(device)
                if task == 'root':
                    if len(pt['rec']):
                        rr, dn, ex = pt['rec'].to(device), pt['donor'].to(device), pt['expected'].to(device)
                        rec['patch_whole'] = (patched(rr, Z[dn]).argmax(1) == ex).double().mean().item()
                        if Plda is not None:
                            rec['patch_lda'] = (patched(rr, Z[dn], Plda).argmax(1) == ex).double().mean().item()
                    rec['patch_same_stay'] = (patched(rec_all, Z[pt['same'].to(device)]).argmax(1)
                                              == pred[rec_all]).double().mean().item()
                    pr = patched(rec_all, Z[pt['rand'].to(device)]).argmax(1)
                    re_, valid = pt['rand_expected'].to(device), pt['rand_expected'].to(device) >= 0
                    rec['patch_rand_changed'] = (pr != pred[rec_all]).double().mean().item()
                    rec['patch_rand_expected'] = (pr[valid] == re_[valid]).double().mean().item() if valid.any() else None
                else:
                    kl = lambda P, lg: (P.to(device) * (P.to(device).clamp_min(1e-300).log()
                                                        - F.log_softmax(lg, 1))).sum(1).mean().item()
                    rec['kl_base'] = kl(ctx['P15'][pt['rec_all']], logits[rec_all])
                    rec['kl_same'] = kl(pt['P_hyb_same'], patched(rec_all, Z[pt['same'].to(device)]))
                    if len(pt['rec']):
                        rr, dn = pt['rec'].to(device), pt['donor'].to(device)
                        rec['kl_patch_base'] = kl(ctx['P15'][pt['rec']], logits[rr])
                        rec['kl_patch'] = kl(pt['P_hyb'], patched(rr, Z[dn]))
            nodes.append(rec)
    out['nodes'] = nodes

    # 4. coupling
    if coupling and model.tie == 'none':
        cp = []
        for l, gr in ctx['groups'].items():
            xA, yA = x[gr['A'].to(device)], y[gr['A'].to(device)]
            masks = coupling_masks(model, l, 0)
            for setting in ('B_same', 'B_diff'):
                if len(gr[setting]) == 0 or len(gr['A']) == 0:
                    continue
                xB, yB = x[gr[setting].to(device)], y[gr[setting].to(device)]
                for sname, mk in masks.items():
                    r = coupling_step(model, xA, yA, xB, yB, mk)
                    cp.append(dict(l=l, k=0, setting=setting[2:], subset=sname, nA=len(xA), nB=len(xB), **r))
        out['coupling'] = cp
    return out


def summarize(nodes):
    """Median over the nodes of each tree level, for every numeric field."""
    out = {}
    for l in sorted({nd['l'] for nd in nodes}):
        rows = [nd for nd in nodes if nd['l'] == l]
        s = {}
        for key in dict.fromkeys(key for r in rows for key in r if key not in ('l', 'k')):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] is not None
                    and not (isinstance(r[key], float) and math.isnan(r[key]))]
            if vals:
                s[key] = float(np.median(vals))
        out[str(l)] = s
    return out


def clean(o):
    """JSON-safe: NaN / inf -> None, tensors -> lists, tuple keys -> strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, torch.Tensor):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def checkpoints(run):
    found = {int(m.group(1)): os.path.join(run, f) for f in os.listdir(run) if (m := re.fullmatch(r'ckpt_(\d+)\.pt', f))}
    return dict(sorted(found.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--diag', default=None, help='saved diagnostic trees (dict from grammar.sample_trees)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n-diag', type=int, default=None, help='use only the first n diagnostic examples (CPU tests)')
    ap.add_argument('--n-pairs', type=int, default=2000)
    ap.add_argument('--n-patch', type=int, default=2000)
    ap.add_argument('--n-group', type=int, default=512)
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    ck = checkpoints(args.run)
    assert ck, f'no ckpt_<step>.pt in {args.run}'
    _, cfg, _ = common.load(next(iter(ck.values())))
    rules = common.rules(cfg['grammar'])
    trees, src = load_diag(args.diag, rules, args.n_diag)
    t0 = time.time()
    ctx = prepare(cfg, trees, rules, args.n_pairs, args.n_patch, args.n_group, device=args.device)
    t_prep = time.time() - t0
    print(f'prepared {ctx["N"]} diagnostic examples in {t_prep:.1f}s', flush=True)
    res = dict(run=os.path.abspath(args.run), cfg=cfg, diag=dict(source=src, N=ctx['N'], n_train=len(ctx['tr'])),
               settings=dict(seed=SEED, l2=L2, tol=TOL, k_small=K_SMALL, n_rand=N_RAND, rel_step=REL_STEP,
                             n_pairs=args.n_pairs, n_patch=args.n_patch, n_group=args.n_group),
               bayes=ctx['bayes'], notes=ctx['notes'], prep_seconds=t_prep,
               groups={str(l): dict(latent=gr['latent'], root=gr['root'], nA=len(gr['A']), nB_same=len(gr['B_same']),
                                    nB_diff=len(gr['B_diff'])) for l, gr in ctx['groups'].items()},
               checkpoints=[])
    if 0 not in ck:
        ck = {0: None, **ck}
        res['notes'].append('no ckpt_0.pt: step 0 is common.build(cfg), the seeded initialization')
    for step, path in ck.items():
        t = time.time()
        model = common.build(cfg) if path is None else common.load(path)[0]
        r = analyze(model, ctx, args.device)
        r['step'] = step
        if 'nodes' in r:
            r['levels'] = summarize(r['nodes'])
        r['seconds'] = time.time() - t
        res['checkpoints'].append(r)
        lv = r.get('levels', {})
        print(f'step {step:6d}  loss {r["loss"]:.4f}  acc {r["acc"]:.3f}  probe by level '
              + ' '.join(f'{lv[s].get("probe_rel", float("nan")):.2f}' for s in sorted(lv)) + f'  ({r["seconds"]:.1f}s)',
              flush=True)
        with open(args.out, 'w') as f:
            json.dump(clean(res), f, indent=1)


if __name__ == '__main__':
    main()
