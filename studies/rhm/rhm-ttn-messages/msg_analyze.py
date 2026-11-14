"""Does a TreeTN trained on RHM prefix prediction pass the grammar's upward messages? Analysis of one run.

    PYTHONPATH=. python studies/rhm/rhm-ttn-messages/msg_analyze.py RUN_DIR OUT_DIR [N_ANA]

Exact inference. For a prefix x_<t, node u of the tree has the upward message
    lam_u(a) = P(observed leaves under u | Z_u = a),   lam_u(a) = sum_bc P(b, c | a) lam_L(b) lam_R(c),
with P(b, c | a) = 1/m if (b, c) is a production of a (else 0), a point mass at an observed leaf and all-ones at a
masked one. The TTN merge z_a = sum_bc W_abc zL_b zR_c has the same form. Messages come from grammar.likelihoods with
the suffix unknown (tree level l, node k <-> grammar level L - l, position k). They are defined up to scale, so the
target is lamn_u = lam_u / sum_a lam_u(a), and the probe that fits a message code is projective: q = W z + c fit so
that q / sum(q) matches lamn (the model and fit of belief_common.projective_fit_predict, repeated in fit_projective
below because the fitted map is needed on new inputs). Messages are nonnegative, so a bond is decoded as the normalized
nonnegative part of q, max(q, 0) / sum max(q, 0) (uniform if nothing is positive): identical to q / sum(q) for a true
message code, and always a point of the simplex, where the bare ratio is unbounded once sum(q) comes near zero on
inputs outside the fit set (`projraw` keeps the bare ratio on M1's test rows). The affine probe is reported next to
it.

Rows. A bond depends only on its span's tokens and on how many of them are observed, its local prefix. Node (l, k) is
collected at the prefix lengths t = k 2^l + j, j = 0 .. 2^l (t <= T - 1, the training range), for every analysis
sequence: j = 0 is the masked case (one constant bond), j = 2^l the fully observed case (message = point mass on the
node's symbol), the rest the partially observed case. The last node of each level contains the leaf that is always
masked in training, so it has no fully observed case.

Measures, per internal node (tree levels 1 .. L-1) and pooled (R2 by summed SSE / SST, the rest by row counts):
 M1 decodable   bond -> lamn_u, affine and projective, fit on half of the distinct local prefixes and scored on the
                other half, by case (R2 of a case is against the fit mean of that case). Eligible nodes: at least
                ELIG_FACTOR * (r + 1) distinct fit local prefixes and partially observed test rows. Controls (final
                checkpoint): the same probes on the one-hot local tokens with the mask (`tok`), the affine probe on
                those plus all their pairwise products where that is eligible (`tok2`), the probes on a scrambled target
                (an independent Dirichlet(1) vector per distinct normalized message; `scr`), and the affine probe on
                shuffled fit targets. Checkpoint 0 is the identity-init tree.
    M1x extended  (first and last checkpoint, partially observed rows only) the same probes to the normalized extended
                message Mn_u = M_u / sum(M_u), a v x v table (`xaff`, `xproj`; token readout `xtok_aff`). See below.

The extended message. When node u is partially observed, the token to predict lies inside its span, and everything the
output learns about the span passes through u's bond. The message lam_u is then not enough for the task: the tree
needs M_u(a, c) = P(observed leaves under u, next token = c | Z_u = a), whose rows sum over c to lam_u, and (to know
which token to predict) the number of observed leaves j. M_u comes from one upward pass per candidate next token.
Two local prefixes with the same lam_u and different M_u or j must get different bonds in a tree at the Bayes loss.
A fully observed or masked node has no next token inside it, so its extended message is lam_u itself.
 M2 invariance  on whitened bonds u (interpret.whitener, so gauge invariant): same = E||u - u'||^2 over pairs of rows
                with different local prefixes and the same exact message, divided by the same quantity for random
                pairs of the case with different local prefixes; ve = 1 - same / rand is the share of bond variance
                that the message accounts for among rows that have such a partner (`cover` = their share). Three
                groupings: `xmsg` = the same extended message and the same number of observed leaves (the criterion:
                what the task lets a bond forget), `msg` = the same unnormalized lam_u, `nmsg` = the same normalized
                lam_u (both descriptive: the task itself forces some of these pairs apart).
                Pooled over nodes with at least ELIG_FACTOR * (r + 1) distinct local prefixes: with fewer points than
                bond dimensions, whitening maps any network's bonds to the same configuration, so ve says nothing
                (in the pilot, level-1 nodes gave the same ve to three decimals before and after training).
 M3 productions for a parent whose two children both have a fully observed case: the class-mean bonds mu_b, mu_c of
                the children's symbols (the leaf embeddings for leaf children) are pushed through the learned core,
                z = W(mu_b, mu_c), and decoded with the parent's probe (fit on all its rows). For the m v valid pairs
                (b, c): accuracy of the decoded symbol and the decoded mass on the true parent. For the invalid pairs
                (no production): the norm of z in the metric of the parent's bonds, z^T S^+ z with S = E[z z^T]
                (invariant to the gauge), as a ratio of medians invalid / valid; exact message passing gives 0.
                Also the probe's unnormalized read-out q_bc = W z_bc + c of every pair: its total nonnegative mass
                sum_a max(q_bc(a), 0), as the ratio of medians invalid / valid (`invalid_q_ratio`; 0 for exact
                messages) and its coefficient of variation over the valid pairs (`valid_q_cv`; 0 for exact messages).
                The merge is bilinear and q affine, so the read-out of any mixture bond in M4 is exactly
                sum_bc lamL(b) lamR(c) q_bc: these two numbers are what M4 depends on.
 M4 update      child bonds are built as mixtures sum_b lamL(b) mu_b and sum_c lamR(c) mu_c, merged with the learned
                core and decoded; the target is the exact normalized update of (lamL, lamR). Pairs are drawn
                independently from the two children's empirical messages (`emp`) and from Dirichlet(1) (`dir`); pairs
                whose exact update is zero are dropped (counted). In training the right child sees tokens only once the
                left one is fully observed, so the independent `emp` pairs split by the children's observation cases:
                `emp_seen` (left full or right masked: combinations of cases that occur in training), and the rest,
                `emp_unseen`, itself split into `emp_leftmask` (left masked, right informative: the exact update is
                then linear in the right message, no product involved) and `emp_both` (left partially observed, right
                informative: the product matters; only at parents of level >= 2, whose children can be partially
                observed). Each subset is scored on the second half of its pairs against the mean of its first half,
                next to its additive ceiling: the affine least-squares fit of the target on [lamL, lamR] from the
                first half of the same subset, the best a merge without the product could do there.
    M4a code     the mixture sum_b lam(b) mu_b decoded with the node's own probe against lam itself, for the node's
                empirical partially observed messages and for Dirichlet(1): is the bond linear in the message on the
                basis of its class means?
M3 and M4 use inputs no probe was fit on. The decoder there is the projective probe; the affine one is reported too.

Saved for msg_plot.py at the first and last checkpoints (level-wise, node 0): the production tables, a sample of the
M4 decoded/exact pairs and, for v = 3, exact and decoded messages of test rows."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-ttn-belief-geometry'))
import ttn_run as TR  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402
from studies.rhm.interpret import whitener  # noqa: E402

N_ANA, ANA_SEED = 20000, 50_000
ELIG_FACTOR, FIT_CAP, N_PAIRS = 10, 60000, 8000
DT = torch.float64
MASK, PARTIAL, FULL = 0, 1, 2


def dirichlet(n, v, gen):
    e = -torch.rand(n, v, generator=gen, dtype=DT).clamp_min(1e-300).log()
    return e / e.sum(1, keepdim=True)


def fit_projective(A_fit, B_fit, iters=300, ridge=1e-8):
    """The projective probe of belief_common.projective_fit_predict (same standardization, affine start and L-BFGS),
    returned as decode(A, rectify=True) so it can be applied to inputs it was not fit on."""
    mu = A_fit.mean(0)
    X = A_fit - mu
    evals, evecs = torch.linalg.eigh(X.T @ X / len(X))
    keep = evals > 1e-12 * evals.max()
    P = evecs[:, keep] / evals[keep].sqrt()
    Zf = X @ P
    c0 = B_fit.mean(0).contiguous()
    Gm = Zf.T @ Zf
    Gm += ridge * Gm.diagonal().mean() * torch.eye(len(Gm), dtype=Gm.dtype, device=Gm.device)
    W0 = torch.linalg.solve(Gm, Zf.T @ (B_fit - c0)).contiguous()
    W, c = W0.clone().requires_grad_(True), c0.clone().requires_grad_(True)

    def loss_of(W, c):
        q = Zf @ W + c
        return ((q / q.sum(1, keepdim=True) - B_fit) ** 2).sum(1).mean()
    base = loss_of(W0, c0).item()
    opt = torch.optim.LBFGS([W, c], max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=20,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = loss_of(W, c)
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        final = loss_of(W, c).item()
    if not np.isfinite(final) or final > base:
        W, c = W0, c0
    W, c = W.detach(), c.detach()

    def decode(A, rectify=True, raw=False):
        q = ((A - mu) @ P) @ W + c
        if raw:
            return q
        if not rectify:
            return q / q.sum(1, keepdim=True)
        q = q.clamp_min(0)
        tot = q.sum(1, keepdim=True)
        return torch.where(tot > 0, q / tot.clamp_min(1e-300), torch.full_like(q, 1.0 / q.shape[1]))
    return decode


def targets(trees, rules, gcfg):
    """Everything that does not depend on the network, per internal node (l, k): rows ordered by prefix length, then
    sequence. lam (rows, v) exact messages, lamn normalized, case, sym (the node's symbol), toks (the span with the
    unobserved leaves set to the mask token v), and ids of the local prefix (lp), the unnormalized message (mid), the
    normalized message (nid) and the extended message with the number of observed leaves (xid). For the partially
    observed rows (indices xrows) xmsgn (len(xrows), v * v) float32 is the normalized extended message, flattened
    (symbol, next token)."""
    L, v = gcfg['L'], gcfg['v']
    x = trees[L]
    N, T = x.shape
    ident = lambda z: torch.unique(z, dim=0, return_inverse=True)[1]
    key_of = lambda m: torch.cat([(m / m.sum(1, keepdim=True)).round(decimals=9), m.sum(1, keepdim=True).log().round(decimals=7)], 1)
    acc = {(l, k): dict(lam=[], j=[], xid=[], xrows=[], xmsgn=[], n_x=0, rows=0) for l in range(1, L) for k in range(T >> l)}
    for t in range(T):
        lik = G.likelihoods(x, rules, v, unknown=range(t, T))
        partial = [(l, k) for (l, k) in acc if 0 < t - (k << l) < (1 << l)]
        ext = {}
        if partial:  # M_u(a, c) = P(observed leaves under u, x_t = c | Z_u = a): one upward pass per candidate token
            xc, passes = x.clone(), []
            for c in range(v):
                xc[:, t] = c
                passes.append(G.likelihoods(xc, rules, v, unknown=range(t + 1, T)))
            ext = {(l, k): torch.stack([p[L - l][:, k] for p in passes], -1).flatten(1) for (l, k) in partial}  # (N, v v)
        for (l, k), a in acc.items():
            j = t - (k << l)
            if 0 <= j <= (1 << l):
                lam = lik[L - l][:, k]
                a['lam'].append(lam)
                a['j'].append(torch.full((N,), j))
                if (l, k) in ext:
                    M = ext[(l, k)]
                    a['xmsgn'].append((M / M.sum(1, keepdim=True)).float())
                    a['xrows'].append(a['rows'] + torch.arange(N))
                    ids = ident(key_of(M))
                else:
                    ids = ident(key_of(lam))
                a['xid'].append(a['n_x'] + ids)  # blocks have different j, so ids never collide across j
                a['n_x'] += int(ids.max()) + 1
                a['rows'] += N
    out = {}
    for (l, k), a in acc.items():
        size = 1 << l
        lam, j = torch.cat(a['lam']), torch.cat(a['j'])
        seq = torch.arange(N).repeat(len(a['lam']))
        toks = x[seq, k * size:(k + 1) * size].clone()
        toks[torch.arange(size)[None] >= j[:, None]] = v
        lamn = lam / lam.sum(1, keepdim=True)
        out[(l, k)] = dict(lam=lam, lamn=lamn, sym=trees[L - l][seq, k], toks=toks,
                           case=torch.where(j == 0, MASK, torch.where(j == size, FULL, PARTIAL)),
                           lp=ident(toks), nid=ident(lamn.round(decimals=9)), mid=ident(key_of(lam)), xid=torch.cat(a['xid']),
                           xrows=torch.cat(a['xrows']) if a['xrows'] else torch.zeros(0, dtype=torch.long),
                           xmsgn=torch.cat(a['xmsgn']) if a['xmsgn'] else torch.zeros(0, v * v))
    return out


@torch.no_grad()
def bonds(model, x, v, chunk=8192):
    """{(l, k): bonds (rows, r) float64} in the row order of targets(), from the tree run on its training inputs."""
    T, L = x.shape[1], model.D
    out = {(l, k): [] for l in range(1, L) for k in range(T >> l)}
    for t in range(T):
        parts = []
        for i in range(0, len(x), chunk):
            xm = x[i:i + chunk].clone()
            xm[:, t:] = v
            parts.append(model(xm, acts=True)[1])
        for (l, k), lst in out.items():
            if 0 <= t - (k << l) <= (1 << l):
                lst.append(torch.cat([p[l][:, k] for p in parts]))
    return {n: torch.cat(z).to(DT) for n, z in out.items()}


def case_scores(b, bhat, case, b_fit, case_fit):
    """Score dicts for all test rows and for the partially and fully observed ones (each against its own fit mean)."""
    out = {'all': BC.score(b, bhat, b_fit.mean(0))}
    for name, c in (('partial', PARTIAL), ('full', FULL)):
        if (case == c).any() and (case_fit == c).any():
            out[name] = BC.score(b[case == c], bhat[case == c], b_fit[case_fit == c].mean(0))
    return out


def m1(Z, tg, r, seed, controls, extended):
    lp, case, b = tg['lp'], tg['case'], tg['lamn']
    dev = Z.device
    gen = torch.Generator().manual_seed(seed)
    side = torch.rand(int(lp.max()) + 1, generator=gen) < 0.5
    side[lp[tg['case'] == MASK]] = True  # the masked case is one local prefix; it goes to the fit side
    fit, test = torch.nonzero(side[lp])[:, 0], torch.nonzero(~side[lp])[:, 0]
    n_fit_lp = int(side.sum())
    if len(fit) > FIT_CAP:
        fit = fit[torch.randperm(len(fit), generator=gen)[:FIT_CAP]]
    info = dict(n_fit_prefixes=n_fit_lp, n_prefixes=len(side), n_messages=int(tg['nid'].max()) + 1,
                n_partial_messages=int(torch.unique(tg['nid'][case == PARTIAL]).numel()),
                n_partial_xmessages=int(torch.unique(tg['xid'][case == PARTIAL]).numel()),
                eligible=bool(n_fit_lp >= ELIG_FACTOR * (r + 1) and (case[test] == PARTIAL).any() and (case[fit] == PARTIAL).any()),
                m2_eligible=bool(len(side) >= ELIG_FACTOR * (r + 1)))
    if len(test) == 0:
        return info, {}, None
    f, s = fit.to(dev), test.to(dev)
    bf, bt, cf, ct = b[fit].to(dev), b[test].to(dev), case[fit].to(dev), case[test].to(dev)
    res = {}
    pa = BC.fit_predict(Z[f], bf, Z[s])
    dec = fit_projective(Z[f], bf)
    pp = dec(Z[s])
    for name, pred in (('aff', pa), ('proj', pp), ('projraw', dec(Z[s], rectify=False))):
        for cname, sc in case_scores(bt, pred, ct, bf, cf).items():
            res[f'{name}_{cname}'] = sc
    oh = torch.nn.functional.one_hot(tg['toks'], int(tg['toks'].max()) + 1).flatten(1).to(DT).to(dev)
    fx, sx = fit[case[fit] == PARTIAL], test[case[test] == PARTIAL]
    if extended and len(fx) > 10 and len(sx) > 10:  # M1x: the normalized extended message, partially observed rows
        pos = torch.full((len(lp),), -1, dtype=torch.long)
        pos[tg['xrows']] = torch.arange(len(tg['xrows']))
        xf, xs = tg['xmsgn'][pos[fx]].to(DT).to(dev), tg['xmsgn'][pos[sx]].to(DT).to(dev)
        fxd, sxd = fx.to(dev), sx.to(dev)
        res['xaff_partial'] = BC.score(xs, BC.fit_predict(Z[fxd], xf, Z[sxd]), xf.mean(0))
        res['xproj_partial'] = BC.score(xs, fit_projective(Z[fxd], xf)(Z[sxd]), xf.mean(0))
        if controls:
            res['xtok_aff_partial'] = BC.score(xs, BC.fit_predict(oh[fxd], xf, oh[sxd]), xf.mean(0))
    if controls:
        for name, pred in (('tok_aff', BC.fit_predict(oh[f], bf, oh[s])), ('tok_proj', fit_projective(oh[f], bf)(oh[s]))):
            for cname, sc in case_scores(bt, pred, ct, bf, cf).items():
                res[f'{name}_{cname}'] = sc
        size, V1 = tg['toks'].shape[1], int(tg['toks'].max()) + 1
        n_tok2 = size * V1 + size * (size - 1) // 2 * V1 * V1
        info['tok2_eligible'] = bool(n_fit_lp >= ELIG_FACTOR * (n_tok2 + 1))
        if info['tok2_eligible']:  # token readout with all pairwise products inside the span (test rows capped: wide)
            def with_pairs(rows):
                o3 = oh[rows].reshape(len(rows), size, V1)
                return torch.cat([oh[rows]] + [(o3[:, i, :, None] * o3[:, j, None, :]).flatten(1)
                                               for i in range(size) for j in range(i + 1, size)], 1)
            sub = torch.randperm(len(s), generator=gen)[:20000].to(dev)
            pred = BC.fit_predict(with_pairs(f), bf, with_pairs(s[sub]))
            for cname, sc in case_scores(bt[sub], pred, ct[sub], bf, cf).items():
                res[f'tok2_aff_{cname}'] = sc
        scr = dirichlet(int(tg['nid'].max()) + 1, b.shape[1], gen)[tg['nid']].to(dev)
        for name, pred in (('scr_aff', BC.fit_predict(Z[f], scr[f], Z[s])), ('scr_proj', fit_projective(Z[f], scr[f])(Z[s]))):
            for cname, sc in case_scores(scr[s], pred, ct, scr[f], cf).items():
                res[f'{name}_{cname}'] = sc
        perm = torch.randperm(len(fit), generator=gen).to(dev)
        res['shuffle_all'] = BC.score(bt, BC.fit_predict(Z[f], bf[perm], Z[s]), bf.mean(0))
    keep = torch.randperm(len(test), generator=gen)[:3000]
    return info, res, dict(true=bt[keep].cpu().numpy().astype(np.float32), proj=pp[keep].cpu().numpy().astype(np.float32),
                           case=ct[keep].cpu().numpy().astype(np.int8))


def same_pairs(gid, lp, rows, n, gen):
    """Pairs (i, i') of rows among `rows` with the same gid and different local prefixes: i uniform over the rows,
    i' uniform in i's group (8 tries). Returns (i, i', share of sampled rows that found a partner)."""
    idx = torch.nonzero(rows)[:, 0]
    if len(idx) < 2:
        return None
    g = gid[idx]
    order = torch.argsort(g)
    uniq, counts = torch.unique_consecutive(g[order], return_counts=True)
    starts = counts.cumsum(0) - counts
    grp = torch.searchsorted(uniq, g)
    pick = torch.randint(len(idx), (n,), generator=gen)
    partner = torch.full((n,), -1)
    for _ in range(8):
        cand = order[starts[grp[pick]] + (torch.rand(n, generator=gen) * counts[grp[pick]]).long()]
        ok = (partner < 0) & (lp[idx[cand]] != lp[idx[pick]])
        partner[ok] = cand[ok]
    found = partner >= 0
    return idx[pick[found]], idx[partner[found]], found.double().mean().item()


def m2(Z, tg, seed):
    mu, Wh, _ = whitener(Z)
    d = Wh.shape[0]
    if d == 0:
        return {}
    U = (Z - mu) @ Wh.T
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for cname, rows in (('partial', tg['case'] == PARTIAL), ('full', tg['case'] == FULL), ('observed', tg['case'] != MASK)):
        if rows.sum() < 2:
            continue
        idx = torch.nonzero(rows)[:, 0]
        a, b = (idx[torch.randint(len(idx), (N_PAIRS,), generator=gen)] for _ in range(2))
        differ = tg['lp'][a] != tg['lp'][b]  # like the same-message pairs, random pairs have different local prefixes
        if not differ.any():
            continue
        a, b = a[differ].to(Z.device), b[differ].to(Z.device)
        rand = ((U[a] - U[b]) ** 2).sum(1).mean().item()
        for gname, name in (('xid', 'xmsg'), ('mid', 'msg'), ('nid', 'nmsg')):
            sp = same_pairs(tg[gname], tg['lp'], rows, N_PAIRS, gen)
            if sp is None or len(sp[0]) == 0 or rand <= 0:
                continue
            i, j, cover = sp
            same = ((U[i.to(Z.device)] - U[j.to(Z.device)]) ** 2).sum(1).mean().item()
            out[f'{cname}_{name}'] = dict(same=same / (2 * d), rand=rand / (2 * d), ve=1 - same / rand, cover=cover,
                                           n=len(i), d=d)
    return out


def class_means(model, Z, tg, l, k, v):
    """(mu (v, r), present (v,) bool): mean bond of each symbol over the fully observed rows of node (l, k); for a leaf
    (l = 0) the embedding of each token."""
    if l == 0:
        return model.A[k, :v].detach().to(DT), torch.ones(v, dtype=torch.bool)
    rows = tg[(l, k)]['case'] == FULL
    sym = tg[(l, k)]['sym'][rows].to(Z[(l, k)].device)
    cnt = torch.bincount(sym, minlength=v).to(DT)
    mu = torch.zeros(v, Z[(l, k)].shape[1], dtype=DT, device=sym.device).index_add_(0, sym, Z[(l, k)][rows.to(sym.device)])
    return mu / cnt.clamp_min(1)[:, None], (cnt > 0).cpu()


def child_messages(tg, x, l, k, v, n, gen):
    """n normalized messages of node (l, k) drawn from its rows, with their observation case (for a leaf: the observed
    token or the mask, equally likely)."""
    if l == 0:
        lam = torch.nn.functional.one_hot(x[torch.randint(len(x), (n,), generator=gen), k], v).to(DT)
        masked = torch.rand(n, generator=gen) < 0.5
        lam[masked] = 1.0 / v
        return lam, torch.where(masked, MASK, FULL)
    i = torch.randint(len(tg[(l, k)]['lamn']), (n,), generator=gen)
    return tg[(l, k)]['lamn'][i], tg[(l, k)]['case'][i]


def exact_update(lamL, lamR, R):
    """Normalized exact update for production rules R (v_parent, m, 2); also which rows have a nonzero update."""
    u = (lamL[:, R[:, :, 0]] * lamR[:, R[:, :, 1]]).mean(2)
    ok = u.sum(1) > 1e-300
    return u / u.sum(1, keepdim=True).clamp_min(1e-300), ok


def analyze_model(model, x, tg, rules, gcfg, r, controls, extended=True, seed=0):
    """All measures for one network. Returns (per-node dict, arrays for the figures)."""
    L, v = gcfg['L'], gcfg['v']
    T = x.shape[1]
    dev = next(model.parameters()).device
    Z = bonds(model, x.to(dev), v)
    nodes, arrays = {}, {}
    means = {}
    for l in range(0, L):
        for k in range(T >> l):
            if (l == 0 and k < T - 1) or (l > 0 and (tg[(l, k)]['case'] == FULL).any()):  # leaf T-1 is never observed
                means[(l, k)] = class_means(model, Z, tg, l, k, v)
    for (l, k), t in tg.items():
        gen = torch.Generator().manual_seed(1000 * seed + 97 * l + k)
        rec = dict(l=l, k=k)
        info, res, tri = m1(Z[(l, k)], t, r, 7 + 97 * l + k, controls, extended)
        rec.update(info=info, m1=res, m2=m2(Z[(l, k)], t, 11 + 97 * l + k))
        if tri is not None and k == 0 and v == 3:
            arrays.update({f'tri_l{l}_{key}': a for key, a in tri.items()})

        # Synthetic bonds decoded with this node's probe (fit on all its rows): M3, M4 (as a parent), M4a (as itself).
        batch, spans = [], {}

        def add(name, z):
            spans[name] = (sum(len(b) for b in batch), len(z))
            batch.append(z)
        cl, cr = (l - 1, 2 * k), (l - 1, 2 * k + 1)
        parent_ok = cl in means and cr in means
        if parent_ok:
            W = model.W[l - 1][k].detach().to(DT)
            (muL, okL), (muR, okR) = means[cl], means[cr]
            R = rules[L - l]
            add('pairs', torch.einsum('abc,ib,jc->ija', W, muL, muR).reshape(v * v, -1))
            lam = {}
            for name in ('emp', 'dir'):
                if name == 'emp':
                    (lL, caseL), (lR, caseR) = child_messages(tg, x, *cl, v, N_PAIRS, gen), child_messages(tg, x, *cr, v, N_PAIRS, gen)
                    # combinations of cases that occur in training: the right child sees tokens only once the left is full
                    seen = (caseL == FULL) | (caseR == MASK)
                    subsets = dict(seen=seen, unseen=~seen, leftmask=~seen & (caseL == MASK), both=~seen & (caseL == PARTIAL))
                else:
                    lL, lR = dirichlet(N_PAIRS, v, gen), dirichlet(N_PAIRS, v, gen)
                lam[name] = (lL, lR)
                add('mix_' + name, torch.einsum('abc,nb,nc->na', W, lL.to(dev) @ muL, lR.to(dev) @ muR))
        if (l, k) in means:
            mu, _ = means[(l, k)]
            own = {'emp': t['lamn'][torch.nonzero(t['case'] == PARTIAL)[:, 0][torch.randint(int((t['case'] == PARTIAL).sum()), (N_PAIRS,), generator=gen)]],
                   'dir': dirichlet(N_PAIRS, v, gen)}
            for name, lm in own.items():
                add('own_' + name, lm.to(dev) @ mu)
        if batch:
            fit = torch.randperm(len(t['lamn']), generator=gen)[:FIT_CAP].to(dev)
            b_all, Zb = t['lamn'].to(dev)[fit], torch.cat(batch)
            decoder = fit_projective(Z[(l, k)][fit], b_all)
            dec = {'proj': decoder(Zb), 'aff': BC.fit_predict(Z[(l, k)][fit], b_all, Zb)}
            get = lambda kind, name: dec[kind][spans[name][0]:spans[name][0] + spans[name][1]]
        if parent_ok:
            parent = torch.full((v, v), -1, dtype=torch.long)
            for a in range(R.shape[0]):
                parent[R[a, :, 0], R[a, :, 1]] = a
            valid = (parent >= 0) & okL[:, None] & okR[None, :]
            invalid = (parent < 0) & okL[:, None] & okR[None, :]
            m3 = dict(n_valid=int(valid.sum()), n_invalid=int(invalid.sum()), missing_classes=int((~okL).sum() + (~okR).sum()))
            for kind in ('proj', 'aff'):
                d = get(kind, 'pairs').reshape(v, v, -1).cpu()
                m3[f'acc_{kind}'] = (d.argmax(-1)[valid] == parent[valid]).double().mean().item()
                m3[f'mass_{kind}'] = d[valid].gather(1, parent[valid][:, None]).mean().item()
            S = Z[(l, k)].T @ Z[(l, k)] / len(Z[(l, k)])
            ev, E = torch.linalg.eigh(S)
            keep = ev > 1e-9 * ev.max()
            zp = Zb[spans['pairs'][0]:spans['pairs'][0] + v * v]  # the merged class-mean bonds themselves
            nrm = (((zp @ E[:, keep]) ** 2) / ev[keep]).sum(1).sqrt().reshape(v, v).cpu()
            m3['invalid_norm_ratio'] = (nrm[invalid].median() / nrm[valid].median()).item() if invalid.any() else None
            # the probe's unnormalized read-out of every pair: M4's mixtures read out as sum_bc lamL(b) lamR(c) q_bc
            q = decoder(zp, raw=True).reshape(v, v, -1).cpu()
            mass = q.clamp_min(0).sum(-1)
            m3['invalid_q_ratio'] = (mass[invalid].median() / mass[valid].median().clamp_min(1e-300)).item() if invalid.any() else None
            m3['valid_q_cv'] = (mass[valid].std() / mass[valid].mean().abs().clamp_min(1e-300)).item() if valid.sum() > 1 else None
            rec['m3'] = m3
            if k == 0:
                arrays[f'prod_l{l}_true'] = parent.numpy().astype(np.int16)
                arrays[f'prod_l{l}_decoded'] = get('proj', 'pairs').reshape(v, v, -1).argmax(-1).cpu().numpy().astype(np.int16)
                arrays[f'prod_l{l}_mass'] = get('proj', 'pairs').reshape(v, v, -1).cpu().numpy().astype(np.float32)
                arrays[f'prod_l{l}_norm'] = nrm.numpy().astype(np.float32)
                arrays[f'prod_l{l}_q'] = q.numpy().astype(np.float32)
            m4 = {}
            first = torch.arange(N_PAIRS) < N_PAIRS // 2
            for name in ('emp', 'dir'):
                lL, lR = lam[name]
                tgt, ok = exact_update(lL, lR, R)
                feats = torch.cat([lL, lR], 1)
                m4[f'{name}_dropped'] = 1 - ok.double().mean().item()
                for sname, sel in [('', torch.ones(N_PAIRS, dtype=torch.bool))] + (list(subsets.items()) if name == 'emp' else []):
                    key = f'{name}_{sname}_' if sname else f'{name}_'
                    a_, b_ = ok & sel & first, ok & sel & ~first  # fit the ceiling and the baseline mean on a_, score on b_
                    m4[key + 'n'] = int(b_.sum())
                    if a_.sum() > 2 * feats.shape[1] and b_.sum() > 10:
                        mu_t = tgt[a_].mean(0)
                        m4[key + 'addceil'] = BC.score(tgt[b_], BC.fit_predict(feats[a_], tgt[a_], feats[b_]), mu_t)
                        for kind in ('proj', 'aff'):
                            m4[key + kind] = BC.score(tgt[b_], get(kind, 'mix_' + name).cpu()[b_], mu_t)
                if k == 0:
                    sub = torch.nonzero(ok & ~first)[:800, 0]
                    arrays[f'm4_l{l}_{name}_true'] = tgt[sub].numpy().astype(np.float32)
                    arrays[f'm4_l{l}_{name}_proj'] = get('proj', 'mix_' + name).cpu()[sub].numpy().astype(np.float32)
                    if name == 'emp':
                        arrays[f'm4_l{l}_emp_both'] = subsets['both'][sub].numpy()
            rec['m4'] = m4
        if (l, k) in means:
            rec['m4a'] = {f'{name}_{kind}': BC.score(lm, get(kind, 'own_' + name).cpu(), lm.mean(0))
                          for name, lm in own.items() for kind in ('proj', 'aff')}
        nodes[f'{l},{k}'] = rec
    return nodes, arrays


def pooled(nodes):
    """Pooled numbers over nodes, overall and per tree level: R2 = 1 - sum SSE / sum SST for score dicts (M1 over the
    eligible nodes only), means weighted by counts for M2 (its eligible nodes only) and M3."""
    out = {}
    for scope in ['all', 'l2+'] + sorted({f"l{n['l']}" for n in nodes.values()}):
        ns = [n for n in nodes.values() if scope == 'all' or (scope == 'l2+' and n['l'] >= 2) or f"l{n['l']}" == scope]
        P = {}
        for block, cond in (('m1', lambda n: n['info']['eligible']), ('m4', lambda n: True), ('m4a', lambda n: True)):
            keys = sorted({key for n in ns if cond(n) for key, val in n.get(block, {}).items() if isinstance(val, dict)})
            for key in keys:
                vals = [n[block][key] for n in ns if cond(n) and key in n.get(block, {})]
                sst = sum(q['sst'] for q in vals)
                P[f'{block}_{key}'] = dict(R2=1 - sum(q['sse'] for q in vals) / sst if sst > 0 else None, nodes=len(vals))
        P['m1_n_eligible'] = sum(n['info']['eligible'] for n in ns)
        # the bond probes pooled over only the nodes where the pairwise token readout exists, to compare like with like
        t2 = [n['m1'] for n in ns if n['info']['eligible'] and 'tok2_aff_partial' in n.get('m1', {})]
        for key in ('proj_partial', 'aff_partial', 'tok2_aff_partial'):
            vals = [q[key] for q in t2 if key in q]
            sst = sum(q['sst'] for q in vals)
            if vals:
                P[f'm1_{key}@tok2'] = dict(R2=1 - sum(q['sse'] for q in vals) / sst if sst > 0 else None, nodes=len(vals))
        for key in sorted({key for n in ns if n['info']['m2_eligible'] for key in n.get('m2', {})}):
            vals = [n['m2'][key] for n in ns if n['info']['m2_eligible'] and key in n['m2']]
            w = sum(q['n'] for q in vals)
            P[f'm2_{key}'] = dict(ve=sum(q['ve'] * q['n'] for q in vals) / w, cover=float(np.mean([q['cover'] for q in vals])),
                                  nodes=len(vals))
        m3 = [n['m3'] for n in ns if 'm3' in n]
        if m3:
            w = sum(q['n_valid'] for q in m3)
            P['m3'] = dict(nodes=len(m3), n_valid=w,
                           **{key: sum(q[key] * q['n_valid'] for q in m3) / w for key in ('acc_proj', 'acc_aff', 'mass_proj', 'mass_aff')},
                           **{key: (float(np.median([q[key] for q in m3 if q.get(key) is not None]))
                                    if any(q.get(key) is not None for q in m3) else None) for key in ('invalid_q_ratio', 'valid_q_cv')},
                           invalid_norm_ratio=(float(np.median([q['invalid_norm_ratio'] for q in m3 if q['invalid_norm_ratio'] is not None]))
                                               if any(q['invalid_norm_ratio'] is not None for q in m3) else None))
        out[scope] = P
    return out


def strip(nodes):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(val) for k, val in o.items() if k not in ('sse', 'sst')}
        return o
    return {name: clean(n) for name, n in nodes.items()}


def main():
    global N_ANA
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3:
        N_ANA = int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    g, r, v = cfg['grammar'], cfg['r'], cfg['V']
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    trees = G.sample_trees(N_ANA, rules, seed=ANA_SEED + g['seed'])
    x = trees[g['L']]
    tg = targets(trees, rules, g)
    ckpts = res['ckpts']
    metrics = {}
    for ck in ckpts:
        model = TR.build(cfg).to(dev)
        model.load_state_dict(torch.load(run_dir / f'ckpt_{ck}.pt', map_location=dev)['state'])
        model.eval()
        nodes, arrays = analyze_model(model, x, tg, rules, g, r, controls=ck == ckpts[-1],
                                      extended=ck in (ckpts[0], ckpts[-1]))  # the projective probe needs grad
        P = pooled(nodes)
        metrics[ck] = dict(pooled=P, nodes=strip(nodes))
        if ck in (ckpts[0], ckpts[-1]):
            np.savez_compressed(out_dir / f'arrays_ckpt{ck}.npz', **arrays)
        a, hi = P['all'], P['l2+']
        show = lambda key, f='R2', P_=a: '-' if P_.get(key) is None or P_[key].get(f) is None else f'{P_[key][f]:.3f}'
        print(f'{run_dir.name} ckpt {ck}: M1 partial proj={show("m1_proj_partial")} aff={show("m1_aff_partial")} '
              f'x={show("m1_xproj_partial")} ({a["m1_n_eligible"]} nodes) | M2 ve xmsg={show("m2_observed_xmsg", "ve")} '
              f'msg={show("m2_observed_msg", "ve")} | M3 (level >= 2) acc={show("m3", "acc_proj", hi)} '
              f'invalid norm={show("m3", "invalid_norm_ratio", hi)} q={show("m3", "invalid_q_ratio", hi)} | M4 (level >= 2) '
              f'emp={show("m4_emp_proj", P_=hi)} both={show("m4_emp_both_proj", P_=hi)} ceil={show("m4_emp_both_addceil", P_=hi)} '
              f'leftmask={show("m4_emp_leftmask_proj", P_=hi)} dir={show("m4_dir_proj", P_=hi)} | M4a emp={show("m4a_emp_proj")}',
              flush=True)
    hist = res['hist']
    json.dump(dict(run=run_dir.name, cfg=cfg, status=res['status'], bayes=res['bayes'], final_excess=hist[-1]['excess'],
                   hist=[{k: h[k] for k in ('step', 'test', 'excess')} for h in hist], n_ana=N_ANA, metrics=metrics),
              open(out_dir / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
