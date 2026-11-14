"""Probes from a tensor network's hidden states to the exact RHM belief states, for the three arms of arch_run.py.
The splits, metrics and controls are those of rhm-belief-geometry/analyze.py and rhm-ttn-belief-geometry/ttn_analyze.py
(same 60,000 analysis sequences, same seeds for the prefix and belief splits, so every arm here, the TTN run and the
transformer run are scored on the same held-out prefixes); only what differs is described here.

    PYTHONPATH=. python studies/rhm/rhm-belief-architectures/arch_analyze.py RUN_DIR OUT_DIR [N_ANA]

Hidden states at prefix length t, and the probes on them:
  tree, perm  the bonds of the tree run on x_<t with positions >= t masked. `top` = [h_A, h_B], the two bonds under the
              root (affine probe, 2r features); `top_bil` = [h_A, h_B, h_A (x) h_B] (bilinear probe, 2r + r^2 features,
              first and last checkpoint); `all` = every bond of tree levels 1 .. D-1 (affine).
              In the aligned tree h_A, h_B summarize the left and right half of the sequence. In the permuted tree they
              summarize two scattered leaf sets A = perm[:T/2], B = perm[T/2:], so both vary from the first token on:
              there is no first half in which one bond carries the whole prefix.
  chain       `state` = h_t, the chain's state after t tokens (affine probe, width features). It has seen the whole
              prefix, like a residual stream at position t.
  ~proj       chain only: the projective probe on `state` (belief_common.projective_fit_predict: the belief up to
              normalization), at the first and last checkpoint, prefix split, where D_t <= PROJ_MAX_D. The chain's
              update h_t = W(h_{t-1}, token) has the form of unnormalized filtering, for which this is the natural
              probe. The trees get no projective fits: they enter no verdict and are slow on a CPU.
  ~soft       chain only, same positions and checkpoints: the readout-matched probe bhat = softmax(W h + c)
              (softmax_fit_predict). The chain is trained through a softmax of U h_t, so its state is fit to make a
              linear map of it the log of a distribution; the belief may be read the same way without h_t being an
              unnormalized belief.

Controls (final checkpoint unless noted):
  checkpoint 0     the identity-init network under every probe.
  tokens           affine probe on the one-hot prefix; for the chain also tokens~proj and tokens~soft, the projective
                   and softmax probes on it (where D_t <= PROJ_MAX_D), so each probe family has a token control that
                   sees what it sees.
  tokprod@prefix   the token readout with cross-set products, the fair baseline for the bilinear probe:
                   [onehot(prefix tokens in A), onehot(prefix tokens in B), their products] with A, B the two leaf sets
                   the top bonds see (the halves for the aligned tree); where it is eligible, listed otherwise.
  addceil@prefix,  the bound on any affine probe on [h_A, h_B]: the best fit f(what A saw) + g(what B saw).
  addup@prefix     Aligned tree, t > T/2: f(left-half symbol) + g(right-half parse state), ttn_analyze.additive_design.
                   Permuted tree: f(id of the prefix tokens in A) + g(id of the prefix tokens in B), fit by backfitting
                   (additive_backfit), where the tables fit in memory.
                   addup is the fit's R2 on its own fit rows: an upper bound on what any additive function reaches in
                   the population (it is uninformative, near 1, once the ids are almost as many as the prefixes; the
                   id counts are stored). addceil is the same fit scored on the held-out prefixes: a lower estimate.
                   For the aligned tree the two agree, because few symbols parametrize the fit. For the permuted tree
                   the sub-prefix ids are many and each co-occurs with few others, so addceil can fall far below the
                   truth (in the pilot even below the affine token readout, which is itself additive); it is reported
                   only where both ids of at least ADD_COVER of the test rows occur in the fit rows, the other
                   positions being listed in `addceil_uncomputable`, and the bound to read is addup.
  nexttok, shuffle, <set>~scr@prefix (scrambled target), <set>@belief with span@belief (belief split): as in
  ttn_analyze.py. The shuffled fit uses `all` (trees) or `state` (chain).

Eligibility: at least ELIG_FACTOR * (features + 1) distinct fit prefixes. Position groups, each pooled as R2_<group>
for every <key>@... entry that covers all of the group's positions:
  ea     eligible for this arm's affine probe (`top`: 2r features; `state`: the chain's width)
  eb     eligible for the trees' bilinear probe (2r + r^2 features). Defined in every arm, the chain included: it is
         the common set on which the three arms are compared.
  eb_lo, eb_hi  the eb positions with t <= T/2 and t > T/2. For t <= T/2 the aligned tree's second top bond is
         constant, so its bilinear probe has far fewer effective features than the permuted tree's; the two regimes are
         reported apart and H_B's representation measure uses eb_hi.
  ebt    the eb positions where tokprod exists
  bel    the ea positions of the belief split
  pa, pb the ea / eb positions with D_t <= PROJ_MAX_D (the chain's projective and softmax probes)"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent / 'rhm-ttn-belief-geometry'))
sys.path.insert(0, str(HERE))
import analyze as AZ  # noqa: E402
import arch_run as AR  # noqa: E402
import beliefs as BL  # noqa: E402
import ttn_analyze as TA  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

N_ANA = 60000
MIN_BELIEFS, PROJ_MAX_D, ADD_COVER, ADD_MAX_ENTRIES = 20, 64, 0.95, 5e7


def leaf_sets(cfg):
    """(A, B): the sequence positions under the two top bonds (the halves for the aligned tree and, nominally, the chain)."""
    T = cfg['T']
    p = cfg['perm'] if cfg['arm'] == 'perm' else list(range(T))
    return sorted(p[:T // 2]), sorted(p[T // 2:])


def sub_ids(x, pos):
    """Integer id of each row's tokens at `pos` (N,), and the number of ids; (None, 0) for an empty set."""
    if not pos:
        return None, 0
    u, ids = torch.unique(x[:, pos], dim=0, return_inverse=True)
    return ids, len(u)


def additive_backfit(ia, ib, na, nb, y, fi, ti, iters=200, tol=1e-10):
    """Least-squares fit y ~ c + f(a) + g(b) on rows fi with categorical a, b (ids ia, ib; either may be None), by
    backfitting; returns the prediction for rows ti. An id unseen in the fit rows contributes 0."""
    mu = y[fi].mean(0)
    r = y[fi] - mu
    D = y.shape[1]
    tabs = []
    for ids, n in ((ia, na), (ib, nb)):
        if ids is not None:
            cnt = torch.zeros(n, dtype=y.dtype, device=y.device).index_add_(0, ids[fi], torch.ones(len(fi), dtype=y.dtype, device=y.device))
            tabs.append([ids, torch.zeros(n, D, dtype=y.dtype, device=y.device), cnt.clamp_min(1)[:, None]])
    for _ in range(iters if len(tabs) > 1 else 1):
        delta = 0.0
        for j, (ids, tab, cnt) in enumerate(tabs):
            other = sum((t2[i2[fi]] for k, (i2, t2, _) in enumerate(tabs) if k != j), torch.zeros_like(r))
            new = torch.zeros_like(tab).index_add_(0, ids[fi], r - other) / cnt
            delta = max(delta, (new - tab).abs().max().item())
            tabs[j][1] = new
        if delta < tol:
            break
    return mu + sum((tab[ids[ti]] for ids, tab, _ in tabs), torch.zeros(len(ti), D, dtype=y.dtype, device=y.device))


def set_token_product(x, t, A, B, v):
    """[onehot(prefix tokens in A), onehot(prefix tokens in B), their products]; number of features alongside."""
    oh = lambda pos: torch.nn.functional.one_hot(x[:, pos], v).flatten(1).double()
    a, b = oh([p for p in A if p < t]), oh([p for p in B if p < t])
    return torch.cat([a, b, (a[:, :, None] * b[:, None, :]).flatten(1)], 1)


def n_token_product(t, A, B, v):
    na, nb = sum(p < t for p in A) * v, sum(p < t for p in B) * v
    return na + nb + na * nb


def softmax_fit_predict(A_fit, B_fit, A_test, iters=300, ridge=1e-8):
    """Readout-matched probe bhat = softmax(W a + c), fit by least squares on the belief with L-BFGS; returns
    (bhat for A_test, final fit loss / fit loss at the start). Features are projected on their principal axes and
    scaled to unit variance, as in belief_common.projective_fit_predict, which leaves the model class unchanged. The
    start is the affine least-squares fit to the centered log-belief, log(b + 1e-6) minus its row mean: the affine
    probe's solution in the coordinates where this model is linear. If the optimizer leaves the finite region or ends
    above its start, the start is returned (ratio 1)."""
    mu = A_fit.mean(0)
    X = A_fit - mu
    evals, evecs = torch.linalg.eigh(X.T @ X / len(X))
    keep = evals > 1e-12 * evals.max()
    P = evecs[:, keep] / evals[keep].sqrt()
    Z, Zt = X @ P, (A_test - mu) @ P
    logb = (B_fit + 1e-6).log()
    logb = logb - logb.mean(1, keepdim=True)
    c0 = logb.mean(0)
    G_ = Z.T @ Z
    G_ += ridge * G_.diagonal().mean() * torch.eye(len(G_), dtype=G_.dtype, device=G_.device)
    W0 = torch.linalg.solve(G_, Z.T @ (logb - c0)).contiguous()
    W, c = W0.clone().requires_grad_(True), c0.contiguous().clone().requires_grad_(True)
    loss_of = lambda W, c: (((Z @ W + c).softmax(1) - B_fit) ** 2).sum(1).mean()
    base = loss_of(W0, c0).item()
    opt = torch.optim.LBFGS([W, c], max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=20,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = loss_of(W, c)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        final = loss_of(W, c).item()
        if not np.isfinite(final) or final > base:
            return (Zt @ W0 + c0).softmax(1), 1.0
        return (Zt @ W + c).softmax(1), final / max(base, 1e-300)


@torch.no_grad()
def chain_all_states(model, x, chunk=8192):
    return torch.cat([BC.chain_states(model, x[i:i + chunk])[0] for i in range(0, len(x), chunk)])


def main():
    global N_ANA
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3:
        N_ANA = int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    g, arm, T, v = cfg['grammar'], cfg['arm'], cfg['T'], cfg['V']
    H, L = T // 2, g['L']
    AFF, BIL, SHUF = ('state', None, 'state') if arm == 'chain' else ('top', 'top_bil', 'all')
    A_set, B_set = leaf_sets(cfg)
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], L, seed=g['seed'])
    trees = G.sample_trees(N_ANA, rules, seed=AZ.ANA_SEED + g['seed'])  # the sequences of the other belief analyses
    x = trees[L]
    bel = BL.beliefs(x, rules, g)
    half = N_ANA // 2
    fit, test = slice(0, half), slice(half, None)
    positions = [t for t in sorted(bel) if t >= 1]
    skipped = [t for t in range(1, T) if t not in bel]
    score, put = BC.score, (lambda d, key, t, val: d.setdefault(key, {}).__setitem__(t, val))

    geom, basis = {}, {}
    for t in positions:
        b = bel[t][0]
        uniq, uid = torch.unique(b.round(decimals=9), dim=0, return_inverse=True)
        mu = b.mean(0)
        _, sv, Vh = torch.linalg.svd(b - mu, full_matrices=False)
        basis[t] = (mu, Vh[:3].T)
        ev = sv ** 2 / (sv ** 2).sum().clamp_min(1e-300)
        geom[t] = dict(D=b.shape[1], n_unique=len(uniq), rank=int((sv > 1e-9 * sv[0]).sum()) if sv[0] > 0 else 0,
                       k99=int((ev.cumsum(0) < 0.99).sum()) + 1, uid=uid)
    psplit, bsplit = {}, {}
    for t in positions:  # the seeds of analyze.py / ttn_analyze.py
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        n_pre = int(pid.max()) + 1
        if n_pre >= AZ.MIN_PREFIXES:
            side = torch.rand(n_pre, generator=torch.Generator().manual_seed(100 + t)) < 0.5
            f_idx, t_idx = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
            if len(f_idx) and len(t_idx):
                psplit[t] = (f_idx, t_idx, n_pre, int(side.sum()))
        nu = geom[t]['n_unique']
        if nu >= MIN_BELIEFS:
            side = torch.rand(nu, generator=torch.Generator().manual_seed(300 + t)) < 0.5
            bsplit[t] = (torch.nonzero(side[geom[t]['uid']])[:, 0], torch.nonzero(~side[geom[t]['uid']])[:, 0])
    p_aff = cfg['r'] if arm == 'chain' else 2 * cfg['r']
    p_bil = 2 * AR.R_TREE + AR.R_TREE ** 2
    enough = lambda t, p: t in psplit and psplit[t][3] >= AZ.ELIG_FACTOR * (p + 1)
    ea = [t for t in positions if enough(t, p_aff)]
    eb = [t for t in positions if enough(t, p_bil)]
    small = lambda t: geom[t]['D'] <= PROJ_MAX_D

    # Controls that do not need the network.
    ctrl, tokprod_skipped, addceil_uncomputable = {}, [], {}
    for t in positions:
        b, p = bel[t]
        bd = b.to(dev)
        oh = torch.nn.functional.one_hot(x[:, :t], v).flatten(1).double().to(dev)
        mu = bd[fit].mean(0)
        put(ctrl, 'nexttok', t, score(bd[test], AZ.nexttok_ceiling(p[fit], b[fit], p[test]).to(dev), mu))
        put(ctrl, 'tokens', t, score(bd[test], BC.fit_predict(oh[fit], bd[fit], oh[test]), mu))
        if t in psplit:
            fc, tc = psplit[t][:2]
            fi, ti = fc.to(dev), tc.to(dev)
            mu2 = bd[fi].mean(0)
            put(ctrl, 'nexttok@prefix', t, score(bd[ti], AZ.nexttok_ceiling(p[fc], b[fc], p[tc]).to(dev), mu2))
            put(ctrl, 'tokens@prefix', t, score(bd[ti], BC.fit_predict(oh[fi], bd[fi], oh[ti]), mu2))
            if arm == 'chain' and small(t):
                put(ctrl, 'tokens~proj@prefix', t, score(bd[ti], BC.projective_fit_predict(oh[fi], bd[fi], oh[ti])[0], mu2))
                put(ctrl, 'tokens~soft@prefix', t, score(bd[ti], softmax_fit_predict(oh[fi], bd[fi], oh[ti])[0], mu2))
            if arm == 'tree' and t > H:
                A = TA.additive_design(trees, t, L).to(dev)
                put(ctrl, 'addceil@prefix', t, score(bd[ti], BC.fit_predict(A[fi], bd[fi], A[ti]), mu2))
                put(ctrl, 'addup@prefix', t, score(bd[fi], BC.fit_predict(A[fi], bd[fi], A[fi]), mu2))
            if arm == 'perm':
                (ia, na), (ib, nb) = sub_ids(x, [q for q in A_set if q < t]), sub_ids(x, [q for q in B_set if q < t])
                cover = torch.ones(len(tc), dtype=torch.bool)
                for ids, n in ((ia, na), (ib, nb)):
                    if ids is not None:
                        seen = torch.zeros(n, dtype=torch.bool)
                        seen[ids[fc]] = True
                        cover &= seen[ids[tc]]
                cov = cover.float().mean().item()
                ids = dict(ids_A=na, ids_B=nb, fit_prefixes=psplit[t][3])
                if (na + nb) * bd.shape[1] > ADD_MAX_ENTRIES:
                    addceil_uncomputable[t] = f'tables too large: ({na} + {nb}) ids x D = {bd.shape[1]}'
                else:
                    ja, jb = (None if i is None else i.to(dev) for i in (ia, ib))
                    put(ctrl, 'addup@prefix', t, dict(score(bd[fi], additive_backfit(ja, jb, na, nb, bd, fi, fi), mu2), **ids))
                    if cov < ADD_COVER:
                        addceil_uncomputable[t] = f'held-out estimate skipped: coverage {cov:.2f} < {ADD_COVER} ({na} and {nb} ids)'
                    else:
                        put(ctrl, 'addceil@prefix', t, dict(score(bd[ti], additive_backfit(ja, jb, na, nb, bd, fi, ti), mu2),
                                                            coverage=cov, **ids))
            if arm != 'chain' and (arm == 'perm' or t > H):
                if enough(t, n_token_product(t, A_set, B_set, v)):
                    A = set_token_product(x, t, A_set, B_set, v).to(dev)
                    put(ctrl, 'tokprod@prefix', t, score(bd[ti], BC.fit_predict(A[fi], bd[fi], A[ti]), mu2))
                else:
                    tokprod_skipped.append(t)
        if t in bsplit:
            fi, ti = (i.to(dev) for i in bsplit[t])
            mu3 = bd[fi].mean(0)
            U = torch.unique(bd[fi].round(decimals=9), dim=0) - mu3
            _, sv, Vh = torch.linalg.svd(U, full_matrices=False)
            Bs = Vh[sv > 1e-9 * sv[0]].T
            put(ctrl, 'span@belief', t, score(bd[ti], mu3 + (bd[ti] - mu3) @ Bs @ Bs.T, mu3))
    groups = {'ea': ea, 'eb': eb, 'eb_lo': [t for t in eb if t <= H], 'eb_hi': [t for t in eb if t > H],
              'ebt': [t for t in eb if t in ctrl.get('tokprod@prefix', {})],
              'bel': [t for t in ea if t in bsplit], 'pa': [t for t in ea if small(t)], 'pb': [t for t in eb if small(t)]}

    ckpts = res['ckpts']
    xd = x.to(dev)
    metrics, proj = {}, {}
    for ck in ckpts:
        model = AR.build(cfg).to(dev)
        model.load_state_dict(torch.load(run_dir / f'ckpt_{ck}.pt', map_location=dev)['state'])
        model.eval()
        keep, last = ck in (ckpts[0], ckpts[-1]), ck == ckpts[-1]
        states = chain_all_states(model, xd) if arm == 'chain' else None
        m = {k: dict(d) for k, d in ctrl.items()} if last else {}
        pr = {}
        for t in positions:
            if arm == 'chain':
                S = {'state': states[:, t]}
            else:
                S = TA.bond_sets(model, xd, t, v, bilinear=keep)
                S = {k: S[k] for k in ('top', 'all', 'top_bil') if k in S}
            bd = bel[t][0].to(dev)
            bf, bt = bd[fit], bd[test]
            mu = bf.mean(0)
            preds = {}
            for k, A in S.items():
                A = A.double()
                pk = BC.fit_predict(A[fit], bf, A[test])
                put(m, k, t, score(bt, pk, mu))
                if keep and k in (AFF, BIL):
                    preds[k] = pk
            perm_rows = torch.randperm(half, generator=torch.Generator().manual_seed(7)).to(dev)
            A = S[SHUF].double()
            pk = BC.fit_predict(A[fit], bf[perm_rows], A[test])
            put(m, 'shuffle', t, score(bt, pk, mu))
            if keep:
                preds['shuffle'] = pk
            if t in psplit:
                fi, ti = (i.to(dev) for i in psplit[t][:2])
                mu2 = bd[fi].mean(0)
                for k, A in S.items():
                    A = A.double()
                    put(m, k + '@prefix', t, score(bd[ti], BC.fit_predict(A[fi], bd[fi], A[ti]), mu2))
                    if keep and arm == 'chain' and small(t):
                        for tag, fn in (('~proj', BC.projective_fit_predict), ('~soft', softmax_fit_predict)):
                            ph, ratio = fn(A[fi], bd[fi], A[ti])
                            put(m, k + tag + '@prefix', t, dict(score(bd[ti], ph, mu2), loss_ratio=ratio))
                if last:
                    scr = torch.randn(geom[t]['n_unique'], 8, generator=torch.Generator().manual_seed(500 + t),
                                      dtype=torch.float64)[geom[t]['uid']].to(dev)
                    for k in (AFF, BIL):
                        if k:
                            A = S[k].double()
                            put(m, k + '~scr@prefix', t, score(scr[ti], BC.fit_predict(A[fi], scr[fi], A[ti]), scr[fi].mean(0)))
            if keep and t in bsplit:
                fi, ti = (i.to(dev) for i in bsplit[t])
                mu3 = bd[fi].mean(0)
                for k in (AFF, BIL):
                    if k:
                        A = S[k].double()
                        put(m, k + '@belief', t, score(bd[ti], BC.fit_predict(A[fi], bd[fi], A[ti]), mu3))
            if keep:
                bmu, V = (z.to(dev) for z in basis[t])
                P = lambda z: ((z - bmu) @ V).cpu().numpy().astype(np.float32)
                h = S[AFF][test].double()
                h = h - h.mean(0)
                _, _, Vh = torch.linalg.svd(h, full_matrices=False)
                e = dict(uid=geom[t]['uid'][test].numpy(), true=P(bt), raw_pca=(h @ Vh[:3].T).cpu().numpy().astype(np.float32),
                         pred_aff=P(preds[AFF]), pred_shuffle=P(preds['shuffle']))
                if BIL:
                    e['pred_bil'] = P(preds[BIL])
                for j, (l, q) in enumerate(G.pending_nodes(t, g['s'], L)):
                    if (g['n'] if l == 0 else v) == 3:
                        e[f'tri{j}_L{l}_true'] = BL.node_marginal(bt, t, g, j).cpu().numpy().astype(np.float32)
                        for name, k in (('aff', AFF), ('bil', BIL)):
                            if k:
                                e[f'tri{j}_L{l}_{name}'] = BL.node_marginal(preds[k], t, g, j).cpu().numpy().astype(np.float32)
                pr[t] = e
            del S, preds
        pooled = {}
        for k, mk in m.items():
            sse, sst = sum(q['sse'] for q in mk.values()), sum(q['sst'] for q in mk.values())
            pooled[k] = dict(R2=1 - sse / sst, rmse=float(np.sqrt(sse / sum(q['n'] for q in mk.values()))), positions=sorted(mk))
            if '@' in k:
                for gname, ts in groups.items():
                    if ts and all(t in mk for t in ts):
                        pooled[k]['R2_' + gname] = 1 - sum(mk[t]['sse'] for t in ts) / sum(mk[t]['sst'] for t in ts)
        metrics[ck] = dict(pooled=pooled, per_pos={k: {t: {kk: vv for kk, vv in q.items() if kk not in ('sse', 'sst')}
                                                       for t, q in mk.items()} for k, mk in m.items()})
        if keep:
            proj[ck] = pr
        show = lambda k, f: f'{pooled[k][f]:.3f}' if k in pooled and f in pooled[k] else '-'
        bil = (f' bil={show(BIL + "@prefix", "R2_eb")} (t<=T/2 {show(BIL + "@prefix", "R2_eb_lo")}, t>T/2 '
               f'{show(BIL + "@prefix", "R2_eb_hi")})' if BIL else
               f' | pa proj={show(AFF + "~proj@prefix", "R2_pa")} soft={show(AFF + "~soft@prefix", "R2_pa")}')
        print(f'{run_dir.name} ckpt {ck}: seq {AFF}={show(AFF, "R2")} shuffle={show("shuffle", "R2")} | ea {AFF}='
              f'{show(AFF + "@prefix", "R2_ea")} tokens={show("tokens@prefix", "R2_ea")} | eb {AFF}='
              f'{show(AFF + "@prefix", "R2_eb")}{bil} '
              f'addup={show("addup@prefix", "R2_eb")} | belief split {AFF}={show(AFF + "@belief", "R2_bel")} '
              f'span={show("span@belief", "R2_bel")}', flush=True)
    for ck, pr in proj.items():
        np.savez_compressed(out_dir / f'proj_ckpt{ck}.npz', **{f't{t}_{k}': a for t, e in pr.items() for k, a in e.items()})
    hist = res['hist']
    json.dump(dict(run=run_dir.name, cfg=cfg, arm=arm, n_params=res['n_params'], status=res['status'], bayes=res['bayes'],
                   final_test=hist[-1]['test'], final_excess=hist[-1]['excess'],
                   hist=[{k: h[k] for k in ('step', 'test', 'excess')} for h in hist], test_pos=hist[-1]['test_pos'],
                   bayes_pos=res['bayes_pos'], positions=positions, skipped=skipped, half=H,
                   probes=dict(affine=AFF, bilinear=BIL), features=dict(affine=p_aff, bilinear=p_bil),
                   leaf_sets=dict(A=A_set, B=B_set),
                   prefix_split={t: dict(n_prefixes=n, n_fit_prefixes=nf, n_fit=len(f), n_test=len(tt))
                                 for t, (f, tt, n, nf) in psplit.items()},
                   prefix_split_skipped=[t for t in positions if t not in psplit],
                   belief_split=sorted(bsplit), tokprod_skipped=tokprod_skipped, addceil_uncomputable=addceil_uncomputable,
                   eligible=groups, geom={t: {k: vv for k, vv in d.items() if k != 'uid'} for t, d in geom.items()},
                   metrics=metrics, n_ana=N_ANA),
              open(out_dir / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
