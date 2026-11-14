"""Probes from a model's hidden states to the exact HMM belief states, per run (the method of
rhm-belief-geometry/analyze.py and rhm-ttn-belief-geometry/ttn_analyze.py, whose docstrings define the splits, the
eligibility rule and the controls; what differs is here).

    PYTHONPATH=. python studies/hmm/hmm-belief-geometry/hmm_analyze.py RUN_DIR OUT_DIR [N_ANA]

Target at position t = 1 .. T-1: the belief eta_t = P(state | x_<t) (S entries, the same space at every t), and for the
product processes also `marg` = the two factor marginals (keys `marg|<set>`).

Hidden states at position t (primary set first):
  transformer  L4 = last block's output at input position t (before the final LayerNorm); concat = all six sets.
  chain        state = h_t, the state after t tokens.
  tree         top = [h_L, h_R], the two bonds under the root, on the masked prefix input; top_bil = [h_L, h_R,
               h_L (x) h_R] (first and last checkpoint); all = every bond of levels 1 .. D-1.
Probe families, each fit separately per position by least squares on the fit rows:
  affine      <set>        bhat = W a + c.
  projective  <set>^proj   bhat = q / sum(q), q = W a + c (first and last checkpoint; the probe of belief_common
              with a second, algebraic starting point, see projective_fit_predict below). Exact filtering is u_{t+1} = u_t T[x_t], eta = u / sum(u): a chain whose state is a linear
              image of the unnormalized u carries the belief projectively, not affinely. For the tree at t > T/2,
              eta_t ~ eta_H M_R is a ratio of forms bilinear in (left half, right prefix): projective on top_bil.
              For `marg` each factor's marginal is normalized separately.
Splits: sequence split (figures), prefix split (criteria; `@prefix`), as in the RHM analyses, same eligibility rule:
a position is eligible for a set with p features if it has at least ELIG_FACTOR * (p + 1) distinct fit prefixes.

Controls (final checkpoint unless noted):
  checkpoint 0   the initial network, probed identically (first checkpoint has every probe family).
  tokens         the affine probe on the one-hot prefix (t * V features).
  nexttok        ceiling of any function of the exact next-token distribution p_t: the conditional mean E[belief | p_t],
                 estimated as the mean belief of ALL analysis rows with the same p_t (not of the fit prefixes only: a
                 ceiling must not lose to a probe because a test prefix's p_t was absent from the fit side), or an
                 affine function of p_t fit on all rows, whichever has the lower error. It is 1 wherever the belief is
                 a function of p_t: on mess3, factored and coupled (a linear image, the emission matrix is square and
                 full rank) and on coins3 (one-to-one). rrxor is the process where it is well below 1.
  context        the excess loss of the optimal predictors from only the last 2 and the last 3 tokens
                 (processes.context_loss) on the training runs' test set, saved as `context_excess`: the yardstick
                 for "learned" in hmm_summary.py.
  shuffle        fit-half beliefs permuted across sequences (sequence split), scored against the truth.
  <set>~scr      the same affine probe fit to a scrambled target: an independent N(0, 1) 8-vector per distinct belief
                 of the position. With few distinct beliefs a probe that identifies the belief state fits any
                 labelling; n_unique per position is saved next to it.
  addceil        tree, t > H = T/2: the population ceiling of every predictor additive in the two halves,
                 f(left half) + g(right prefix), computed exactly by enumerating all halves (alternating conditional
                 means on the exact joint table), where the table is small enough (V^t * S <= MAX_TABLE); positions
                 where it is not are listed. Any affine probe on bonds from disjoint spans is bounded by it.
  tokprod        tree, t > H: least squares on [onehot(left tokens), onehot(right prefix tokens), their products],
                 where eligible.
Pooled map (Simplex convention; transformer and chain only; `pool` in the output): one affine map for all positions
t >= 1, fit on the first POOL_N sequences, scored on the sequence split and on the union of the per-position prefix
splits; projective too for the primary set at the first and last checkpoint.

Position groups, pooled as R2_<group> for every `@prefix` key that covers the whole group:
  e    eligible for the primary set (L4 / state / top);   nt  = e with nexttok R2 < NT_ROOM;
  tree: h1 = e, t <= H;  h2b = eligible for top_bil, t > H;  h2c = h2b with addceil available and below ADD_ROOM
        (where the belief is not already additive in the two halves);  h2t = h2b with tokprod eligible;
        h2n = h2b with nexttok R2 < NT_ROOM.
Saved predictions (proj_ckpt*.npz, seed 0 only, first and last checkpoint): the true beliefs and the one probe that
hmm_plot.py draws for the model, for the first SAVE_N test rows of the sequence split."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RHM = HERE.parents[1] / 'rhm'
sys.path.insert(0, str(RHM / 'rhm-ttn-belief-geometry'))
sys.path.insert(0, str(RHM / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE))
import analyze as AZ  # noqa: E402
import hmm_run as HR  # noqa: E402
import ttn_analyze as TA  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.hmm import processes as P  # noqa: E402

N_ANA, ANA_SEED, POOL_N, SAVE_N = 60000, 40_000, 20000, 4000
MAX_TABLE, NT_ROOM, ADD_ROOM = 2e8, 0.9, 0.9
PRIMARY = dict(transformer='L4', chain='state', tree='top')
PROJ = dict(transformer=['L4'], chain=['state'], tree=['top', 'top_bil'])
SHUFFLE = dict(transformer='concat', chain='state', tree='all')
score = BC.score


def exact_additive(proc, t, H, iters=2000, tol=1e-13):
    """Population (mse, var) of the best f(left half) + g(right prefix) fit to eta_t, t > H, from the exact joint table
    P(L, R, state) = pi T[L] T[R]; both are per-row squared errors summed over the S belief entries. None if the table
    has more than MAX_TABLE entries.

    With w = P(L, R) and d = belief - mean, the normal equations are f_L = (sum_R w d - (w g)_L) / w_L and
    g_R = (sum_L w d - (w^T f)_R) / w_R, iterated to a fixed point (alternating conditional means), and
    mse = sum w |d|^2 - 2 sum w d . (f + g) + sum w |f + g|^2, so only the (L, R, state) table itself is stored."""
    T, V, S = proc['T'], proc['V'], proc['S']
    if V ** t * S > MAX_TABLE:
        return None
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = T.to(dev)
    u = proc['pi'].to(dev)[None]
    for _ in range(H):
        u = torch.einsum('ns,xsj->nxj', u, T).reshape(-1, S)
    u = u[u.sum(1) > 0]
    Mr = torch.eye(S, dtype=P.F64, device=dev)[None]
    for _ in range(t - H):
        Mr = torch.einsum('nij,xjk->nxik', Mr, T).reshape(-1, S, S)
    d = torch.einsum('ls,rsj->lrj', u, Mr)  # P(L, R, state); becomes d in place
    w = d.sum(-1)
    mean = d.sum((0, 1))  # = sum w * belief, since sum w = 1
    d /= w.clamp_min(1e-300)[..., None]
    d -= mean
    d[w == 0] = 0.0
    wl, wr = w.sum(1, keepdim=True).clamp_min(1e-300), w.sum(0)[:, None].clamp_min(1e-300)
    dL, dR = torch.einsum('lr,lrs->ls', w, d), torch.einsum('lr,lrs->rs', w, d)
    var = torch.einsum('lr,lrs,lrs->', w, d, d).item()
    del d
    f, g = torch.zeros_like(dL), torch.zeros_like(dR)
    for _ in range(iters):
        f_new = (dL - w @ g) / wl
        g_new = (dR - w.T @ f_new) / wr
        delta = max((f_new - f).abs().max().item(), (g_new - g).abs().max().item())
        f, g = f_new, g_new
        if delta < tol:
            break
    quad = (wl * f ** 2).sum() + (wr * g ** 2).sum() + 2 * (f * (w @ g)).sum()
    mse = var - 2 * ((dL * f).sum() + (dR * g).sum()).item() + quad.item()
    return max(mse, 0.0), var


def nexttok_ceiling(p_all, b_all, p_test, b_test):
    """Prediction of b_test from p_test by E[belief | p], estimated on all rows (group means, or affine in p, whichever
    has the lower error on the test rows)."""
    grp = AZ.nexttok_ceiling(p_all.cpu(), b_all.cpu(), p_test.cpu()).to(b_test.device)
    aff = BC.fit_predict(p_all, b_all, p_test)
    return grp if ((b_test - grp) ** 2).sum() <= ((b_test - aff) ** 2).sum() else aff


def projective_fit_predict(A_fit, B_fit, A_test, iters=300, ridge=1e-8):
    """The projective probe of belief_common (bhat = q / sum(q), q = Theta^T [z, 1] on whitened features z), with a
    second starting point. belief_common starts L-BFGS from the affine solution only, and on a tree built to be exact
    (hmm_check.py) that stalls at R2 0.995 where 1 is attainable. The second start is the algebraic solution: an exact
    projective code satisfies q_n - b_n (1^T q_n) = 0 for every row, which is linear in Theta, so minimize
        sum_n ||(I - b_n 1^T) Theta^T z_n||^2   subject to   mean_n 1^T Theta^T z_n = 1,
    a linear solve, theta = M^-1 a / (a^T M^-1 a) with M[(i,d),(j,e)] = sum_n z_ni z_nj G_n[d,e],
    G_n = I - 1 b_n^T - b_n 1^T + |b_n|^2 1 1^T, and a[(i,d)] = mean_n z_ni. L-BFGS then minimizes the actual squared
    error from each start, and the solution with the lower fit loss is returned (never worse in fit than the affine
    probe, which is the first start with zero iterations). A solution whose denominator changes sign on the fit rows is
    rejected: it cannot be the total mass of a message, and it blows up on new rows."""
    mu = A_fit.mean(0)
    X = A_fit - mu
    evals, evecs = torch.linalg.eigh(X.T @ X / len(X))
    keep = evals > 1e-12 * evals.max()
    Pm = evecs[:, keep] / evals[keep].sqrt()
    one = lambda Z: torch.cat([Z, torch.ones(len(Z), 1, dtype=Z.dtype, device=Z.device)], 1)
    Z, Zt = one(X @ Pm), one((A_test - mu) @ Pm)
    N, p1 = Z.shape
    D = B_fit.shape[1]
    eye = lambda n: torch.eye(n, dtype=Z.dtype, device=Z.device)
    ZZ = Z.T @ Z
    th_aff = torch.linalg.solve(ZZ + ridge * ZZ.diagonal().mean() * eye(p1), Z.T @ B_fit).contiguous()
    Zb = [Z.T @ (B_fit[:, d:d + 1] * Z) for d in range(D)]  # sum_n z z^T b_nd
    Zn = Z.T @ ((B_fit ** 2).sum(1, keepdim=True) * Z)
    M = torch.einsum('ij,de->idje', ZZ, eye(D)) + Zn[:, None, :, None].expand(p1, D, p1, D).clone()
    for d in range(D):
        M[:, d, :, :] -= Zb[d][:, :, None]  # - b_n[d] in row index d  (b 1^T)
        M[:, :, :, d] -= Zb[d][:, None, :]  # - b_n[e] in column index e (1 b^T)
    M = M.reshape(p1 * D, p1 * D)
    a = Z.mean(0)[:, None].expand(p1, D).reshape(-1)
    v = torch.linalg.solve(M + ridge * M.diagonal().mean() * eye(p1 * D), a)
    th_alg = (v / (a @ v)).reshape(p1, D).contiguous()

    def predict(Zx, th):
        q = Zx @ th
        return q / q.sum(1, keepdim=True)

    loss_of = lambda th: ((predict(Z, th) - B_fit) ** 2).sum(1).mean()
    best, best_loss = th_aff, loss_of(th_aff).item()
    base = best_loss
    for th0 in (th_aff, th_alg):
        th = th0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([th], max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=20,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = loss_of(th)
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except RuntimeError:  # a start whose denominators cross zero can produce non-finite values
            continue
        with torch.no_grad():
            final = loss_of(th).item()
            den = (Z @ th).sum(1)
            positive = bool((den * den.mean().sign() > 0).all())  # an unnormalized message has positive total mass
        if np.isfinite(final) and final < best_loss and positive:
            best, best_loss = th.detach(), final
    with torch.no_grad():
        return predict(Zt, best), best_loss / max(base, 1e-300)


def projective(A_fit, B_fit, A_test, blocks):
    return torch.cat([projective_fit_predict(A_fit, B_fit[:, a:b], A_test)[0] for a, b in blocks], 1)


@torch.no_grad()
def hidden(model, kind, xd, V, bilinear):
    """at(t) -> {set: (N, d) tensor on xd's device} for the model's activation sets at position t."""
    if kind == 'transformer':
        acts = AZ.activations(model, xd, V)  # {set: (N, T, d)} float32 on the CPU
        return lambda t: {'L4': acts['L4'][:, t].to(xd.device), 'concat': AZ.at(acts, 'concat', t, xd.device)}
    if kind == 'chain':
        st = torch.cat([BC.chain_states(model, xd[i:i + 8192])[0] for i in range(0, len(xd), 8192)])
        return lambda t: {'state': st[:, t]}

    def at(t):
        s = TA.bond_sets(model, xd, t, V, bilinear)
        return {k: s[k] for k in ('top', 'all', 'top_bil') if k in s}
    return at


def main():
    n_ana = int(sys.argv[3]) if len(sys.argv) > 3 else N_ANA
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    kind, T, V, S = cfg['model'], cfg['T'], cfg['V'], cfg['S']
    H = T // 2
    proc = P.get(cfg['process'])
    x = P.sample(proc, n_ana, T, g=torch.Generator().manual_seed(ANA_SEED))  # the same sequences for every model
    eta, pnext = (z.to(dev) for z in P.beliefs(proc, x))
    targets = {'': (eta, [(0, S)])}  # key prefix -> (target (N, T, D), blocks that each sum to 1)
    if proc['factors']:
        ma, mb = P.marginals(proc, eta)
        targets['marg|'] = (torch.cat([ma, mb], -1), [(0, ma.shape[-1]), (ma.shape[-1], ma.shape[-1] + mb.shape[-1])])
    xd = x.to(dev)
    half = n_ana // 2
    fit, test = slice(0, half), slice(half, None)
    positions = list(range(1, T))
    put = lambda d, key, t, val: d.setdefault(key, {}).__setitem__(t, val)

    geom, psplit = {}, {}
    for t in positions:
        b = eta[:, t]
        _, uid = torch.unique(b.round(decimals=9), dim=0, return_inverse=True)
        sv = torch.linalg.svdvals(b - b.mean(0))
        geom[t] = dict(n_unique=int(uid.max()) + 1, rank=int((sv > 1e-9 * sv[0]).sum()) if sv[0] > 0 else 0, uid=uid)
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        n_pre = int(pid.max()) + 1
        if n_pre >= AZ.MIN_PREFIXES:
            side = torch.rand(n_pre, generator=torch.Generator().manual_seed(100 + t)) < 0.5
            f_idx, t_idx = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
            if len(f_idx) and len(t_idx):
                psplit[t] = (f_idx.to(dev), t_idx.to(dev), n_pre, int(side.sum()))
    enough = lambda t, p: t in psplit and psplit[t][3] >= AZ.ELIG_FACTOR * (p + 1)

    # Controls that do not need the network.
    ctrl, addceil_skipped, tokprod_skipped = {}, [], []
    for t in positions:
        oh = torch.nn.functional.one_hot(xd[:, :t], V).flatten(1).double()
        for pre, (tg, _) in targets.items():
            b = tg[:, t]
            put(ctrl, pre + 'tokens', t, score(b[test], BC.fit_predict(oh[fit], b[fit], oh[test]), b[fit].mean(0)))
            if t in psplit:
                fi, ti = psplit[t][:2]
                put(ctrl, pre + 'tokens@prefix', t, score(b[ti], BC.fit_predict(oh[fi], b[fi], oh[ti]), b[fi].mean(0)))
        b, p = eta[:, t], pnext[:, t]
        put(ctrl, 'nexttok', t, score(b[test], nexttok_ceiling(p, b, p[test], b[test]), b[fit].mean(0)))
        if t in psplit:
            fi, ti = psplit[t][:2]
            put(ctrl, 'nexttok@prefix', t, score(b[ti], nexttok_ceiling(p, b, p[ti], b[ti]), b[fi].mean(0)))
            if kind == 'tree' and t > H:
                ex = exact_additive(proc, t, H)
                if ex is None:
                    addceil_skipped.append(t)
                else:  # a population value, written as sums over the test rows so it pools like the probes
                    n = len(ti)
                    put(ctrl, 'addceil@prefix', t, dict(sse=ex[0] * n, sst=ex[1] * n, n=n, R2=1 - ex[0] / ex[1],
                                                         rmse=float(np.sqrt(ex[0]))))
                if enough(t, H * V + (t - H) * V + H * V * (t - H) * V):
                    A = TA.token_product(x, t, V).to(dev)
                    put(ctrl, 'tokprod@prefix', t, score(b[ti], BC.fit_predict(A[fi], b[fi], A[ti]), b[fi].mean(0)))
                else:
                    tokprod_skipped.append(t)

    # the probe hmm_plot.py draws at position t (its `stacked`)
    drawn = lambda t: dict(transformer='L4', chain='state^proj')[kind] if kind != 'tree' else ('top^proj' if t <= H else 'top_bil^proj')
    xt = P.sample(proc, HR.N_TEST, T, g=torch.Generator().manual_seed(HR.TEST_SEED))  # the training runs' test set
    bl = P.bayes_loss(proc, xt)
    context = {m: (P.context_loss(proc, xt, m) - bl).mean().item() for m in (2, 3)}
    p_feat = dict(L4=cfg.get('d', 0), concat=6 * cfg.get('d', 0), state=cfg.get('r', 0), top=2 * cfg.get('r', 0),
                  top_bil=2 * cfg.get('r', 0) + cfg.get('r', 0) ** 2)
    prim = PRIMARY[kind]
    e = [t for t in positions if enough(t, p_feat[prim])]
    nt = ctrl.get('nexttok@prefix', {})
    groups = {'e': e, 'nt': [t for t in e if nt[t]['R2'] < NT_ROOM]}
    if kind == 'transformer':
        groups['eC'] = [t for t in positions if enough(t, p_feat['concat'])]
    if kind == 'tree':
        e_bil = [t for t in positions if enough(t, p_feat['top_bil'])]
        addc = ctrl.get('addceil@prefix', {})
        groups.update(h1=[t for t in e if t <= H], h2=[t for t in e if t > H], h2b=[t for t in e_bil if t > H],
                      h2c=[t for t in e_bil if t > H and t in addc and addc[t]['R2'] < ADD_ROOM],
                      h2t=[t for t in e_bil if t > H and t in ctrl.get('tokprod@prefix', {})],
                      h2n=[t for t in e_bil if t > H and nt[t]['R2'] < NT_ROOM])

    # Rows of the pooled-over-positions map: the first POOL_N sequences.
    pool_n = min(POOL_N, n_ana // 3)
    pool_seq = (torch.arange(pool_n, device=dev) < pool_n // 2)
    in_pool = lambda idx: idx[idx < pool_n]

    ckpts = res['ckpts']
    metrics, proj = {}, {}
    for ck in ckpts:
        model = HR.build(cfg).to(dev)
        model.load_state_dict(torch.load(run_dir / f'ckpt_{ck}.pt', map_location=dev)['state'])
        model.eval()
        keep = ck in (ckpts[0], ckpts[-1])
        save = keep and cfg['seed'] == 0  # predictions are saved only for what hmm_plot.py draws
        last = ck == ckpts[-1]
        at = hidden(model, kind, xd, V, bilinear=keep)
        m = {k: dict(d) for k, d in ctrl.items()} if last else {}
        pr, pool_rows = {}, {}
        for t in positions:
            Sx = {k: A.double() for k, A in at(t).items()}
            e_out = {}
            for pre, (tg, blocks) in targets.items():
                b = tg[:, t]
                bf, bt, mu = b[fit], b[test], b[fit].mean(0)
                for k, A in Sx.items():
                    fams = [('', lambda Af, Bf, At: BC.fit_predict(Af, Bf, At))]
                    if keep and k in PROJ[kind]:
                        fams.append(('^proj', lambda Af, Bf, At: projective(Af, Bf, At, blocks)))
                    for suf, fn in fams:
                        pk = fn(A[fit], bf, A[test])
                        put(m, pre + k + suf, t, score(bt, pk, mu))
                        if save and pre == '' and k + suf == drawn(t):
                            e_out[f'pred_{k}{suf}'] = pk[:SAVE_N].cpu().numpy().astype(np.float32)
                        if t in psplit:
                            fi, ti = psplit[t][:2]
                            put(m, pre + k + suf + '@prefix', t, score(b[ti], fn(A[fi], b[fi], A[ti]), b[fi].mean(0)))
            b = eta[:, t]
            perm = torch.randperm(half, generator=torch.Generator().manual_seed(7)).to(dev)
            A = Sx[SHUFFLE[kind]]
            pk = BC.fit_predict(A[fit], b[fit][perm], A[test])
            put(m, 'shuffle', t, score(b[test], pk, b[fit].mean(0)))
            if last and t in psplit:
                fi, ti = psplit[t][:2]
                scr = torch.randn(geom[t]['n_unique'], 8, generator=torch.Generator().manual_seed(500 + t),
                                  dtype=torch.float64)[geom[t]['uid'].cpu()].to(dev)
                for k in [prim] + (['top_bil'] if kind == 'tree' else []):
                    put(m, k + '~scr@prefix', t, score(scr[ti], BC.fit_predict(Sx[k][fi], scr[fi], Sx[k][ti]), scr[fi].mean(0)))
            if save:
                e_out['true'] = b[test][:SAVE_N].cpu().numpy().astype(np.float32)
                pr[t] = e_out
            if kind != 'tree':
                pool_rows[t] = {k: A[:pool_n] for k, A in Sx.items()}
            del Sx
        pooled = {}
        for k, mk in m.items():
            sse, sst = sum(q['sse'] for q in mk.values()), sum(q['sst'] for q in mk.values())
            pooled[k] = dict(R2=1 - sse / sst if sst > 0 else float('nan'),
                             rmse=float(np.sqrt(sse / sum(q['n'] for q in mk.values()))), positions=sorted(mk))
            if '@' in k:
                for gname, ts in groups.items():
                    if ts and all(t in mk for t in ts):
                        gs = sum(mk[t]['sst'] for t in ts)
                        pooled[k]['R2_' + gname] = 1 - sum(mk[t]['sse'] for t in ts) / gs if gs > 0 else float('nan')
        out = dict(pooled=pooled, per_pos={k: {t: {kk: vv for kk, vv in q.items() if kk not in ('sse', 'sst')}
                                               for t, q in mk.items()} for k, mk in m.items()})
        if kind != 'tree':  # one map for all positions
            B = torch.cat([eta[:pool_n, t] for t in positions])
            seq_fit = pool_seq.repeat(len(positions))
            pre_fit = torch.cat([torch.zeros(pool_n, dtype=torch.bool, device=dev).index_fill_(0, in_pool(psplit[t][0]), True)
                                 if t in psplit else torch.zeros(pool_n, dtype=torch.bool, device=dev) for t in positions])
            pre_test = torch.cat([torch.zeros(pool_n, dtype=torch.bool, device=dev).index_fill_(0, in_pool(psplit[t][1]), True)
                                  if t in psplit else torch.zeros(pool_n, dtype=torch.bool, device=dev) for t in positions])
            pool = {}
            for k in pool_rows[positions[0]]:
                A = torch.cat([pool_rows[t][k] for t in positions])
                fams = [('', BC.fit_predict)]
                if keep and k == prim:
                    fams.append(('^proj', lambda Af, Bf, At: projective_fit_predict(Af, Bf, At)[0]))
                for suf, fn in fams:
                    for name, fm, tm in (('', seq_fit, ~seq_fit), ('@prefix', pre_fit, pre_test)):
                        if fm.sum() <= 10 * (A.shape[1] + 1) or tm.sum() == 0:  # too few rows to fit one map
                            continue
                        sc = score(B[tm], fn(A[fm], B[fm], A[tm]), B[fm].mean(0))
                        pool[k + suf + name] = dict(R2=sc['R2'], rmse=sc['rmse'], n=sc['n'])
                del A
            out['pool'] = pool
        metrics[ck] = out
        if save:
            proj[ck] = pr
        show = lambda k, f='R2': f'{pooled[k][f]:.3f}' if k in pooled and f in pooled[k] else '-'
        first = 'h1' if kind == 'tree' else 'e'
        line = (f'{run_dir.name} ckpt {ck}: seq {prim}={show(prim)} shuffle={show("shuffle")} | prefix[{first}] {prim}='
                f'{show(prim + "@prefix", "R2_" + first)} proj={show(prim + "^proj@prefix", "R2_" + first)} '
                f'tokens={show("tokens@prefix", "R2_" + first)} scr={show(prim + "~scr@prefix", "R2_" + first)}')
        if kind == 'tree':
            line += (f' | h2b top={show("top@prefix", "R2_h2b")} bil={show("top_bil@prefix", "R2_h2b")} bil^proj='
                     f'{show("top_bil^proj@prefix", "R2_h2b")} addceil={show("addceil@prefix", "R2_h2b")}')
        else:
            line += f' | one map: {out["pool"].get(prim + "@prefix", {}).get("R2", float("nan")):.3f}'
        print(line, flush=True)
    for ck, pr in proj.items():
        np.savez_compressed(out_dir / f'proj_ckpt{ck}.npz', **{f't{t}_{k}': a for t, e in pr.items() for k, a in e.items()})
    hist = res['hist']
    json.dump(dict(run=run_dir.name, cfg=cfg, n_params=res['n_params'], status=res['status'], bayes=res['bayes'],
                   final_test=hist[-1]['test'], final_excess=hist[-1]['excess'],
                   hist=[{k: h[k] for k in ('step', 'test', 'excess')} for h in hist], test_pos=hist[-1]['test_pos'],
                   bayes_pos=res['bayes_pos'], positions=positions, half=H, primary=prim, features=p_feat,
                   prefix_split={t: dict(n_prefixes=n, n_fit_prefixes=nf, n_fit=len(f), n_test=len(tt))
                                 for t, (f, tt, n, nf) in psplit.items()},
                   prefix_split_skipped=[t for t in positions if t not in psplit],
                   addceil_skipped=addceil_skipped, tokprod_skipped=tokprod_skipped, eligible=groups,
                   geom={t: {k: vv for k, vv in d.items() if k != 'uid'} for t, d in geom.items()},
                   context_excess=context, metrics=metrics, n_ana=n_ana, pool_n=pool_n),
              open(out_dir / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
