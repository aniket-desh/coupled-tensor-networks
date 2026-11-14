"""Linear probes from the residual stream to the exact RHM belief states (Shai et al. 2024 method), per run.

    PYTHONPATH=. python studies/rhm/rhm-belief-geometry/analyze.py runs/rhm-belief-geometry/v8m4s2L3_s0 OUT_DIR [N_ANA]

Analysis set: N_ANA sequences sampled from the grammar (fixed seed), split by sequence into a fit half and a test half.
Sampling makes every row count in proportion to its probability, which is the prefix-probability weighting of the
later Simplex code (Riechers et al. 2025) without deduplication. For each input position t = 1 .. T-1 (t = 0 is BOS,
whose belief is the constant prior) with D_t <= beliefs.MAX_D, and each activation set
  embed (block input), L1..L4 (block outputs, before the final LayerNorm), lnf (after it), concat (all six),
fit b_t ~ W a + c by least squares on the fit half (float64, ridge 1e-8 relative to the mean diagonal, which only
guards rank deficiency) and score on the test half:
  R2 = 1 - sum ||b - bhat||^2 / sum ||b - mean_fit b||^2,   RMSE = sqrt(mean_rows ||b - bhat||^2)  (Riechers 2025),
and pooled over positions as 1 - sum_t SSE_t / sum_t SST_t. Controls, scored the same way on the same test rows:
  shuffle  the fit half's belief rows permuted across sequences (Shai 2024: error against the unshuffled truth);
  ckpt 0   the randomly initialized network;
  tokens   the best linear function of the raw prefix: the probe applied to the one-hot encoding of x_<t (t * v
           features). Where this is high the belief is a linear readout of the tokens and the probe shows nothing
           computed; the informative positions are where the trained network beats it.
  nexttok  the best any function of the exact next-token distribution P(x_t | x_<t) can do: the mean belief of fit
           rows with the same next-token distribution (rounded to 9 decimals), the fit mean for unseen ones.
           If R2 (concat) exceeds this, the residual stream carries belief information beyond the next-token output.
Prefix split (key=<set>@prefix): the sequence split above lets test rows share their prefix with fit rows, and when a
grammar has fewer distinct prefixes than features, any injective embedding of the prefix (a random network's included)
lets a linear probe interpolate any function of it. So each position's distinct prefixes x_<t are also assigned at
random to fit or test (seeded per position) and the probe is scored only on prefixes it never saw. A network that maps
prefixes to an affine image of their belief passes this; a memorizing probe on an injective embedding does not.
Positions with fewer than MIN_PREFIXES distinct prefixes are left out of the prefix split (listed in the output).
Even then a probe with p features is underdetermined on few prefixes, so the criteria use only eligible positions:
at least ELIG_FACTOR * (p + 1) distinct fit prefixes, p = d_model for one layer (L4) or 6 d_model (concat). Every
<key>@prefix entry is also pooled over the L4-eligible positions (R2_eL4) and the concat-eligible ones (R2_eC), so the
trained network, random init, the token readout and the next-token ceiling are compared on the same positions.
Pairwise distances (Shai 2024 Fig. 7D): for up to 1500 test rows per position, Euclidean distances between predicted
beliefs (concat probe) against distances between true beliefs and between next-token distributions; R^2 = corr^2.

Saved for plot.py (every checkpoint's metrics; projections for the first and last checkpoints): per position, the
test rows' unique-belief id, their coordinates on the top-3 PCs of the true beliefs over the sampled rows (so weighted
by prefix probability; Shai Fig. 7 used the distinct beliefs unweighted, which here lets a few rare beliefs set the axes)
for the truth and each probe, the top-3 PCs of the raw final-layer residual stream, and, where a pending node has 3
symbols, that node's marginal (truth and probe) for a triangle plot."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import beliefs as BL  # noqa: E402
import model as M  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

MIN_PREFIXES, ELIG_FACTOR = 20, 10
N_ANA, ANA_SEED = 60000, 40_000  # N_ANA can be overridden by the third argument (smaller for CPU tests)
SETS = ['embed', 'L1', 'L2', 'L3', 'L4', 'lnf', 'concat']


def fit_predict(A_fit, B_fit, A_test, ridge=1e-8):
    """Affine least squares: returns bhat for A_test."""
    mu_a, mu_b = A_fit.mean(0), B_fit.mean(0)
    X, Y = A_fit - mu_a, B_fit - mu_b
    G_ = X.T @ X
    G_ += ridge * G_.diagonal().mean().clamp_min(1e-30) * torch.eye(len(G_), dtype=G_.dtype, device=G_.device)
    W = torch.linalg.solve(G_, X.T @ Y)
    return (A_test - mu_a) @ W + mu_b


def score(b, bhat, mu):
    sse, sst = ((b - bhat) ** 2).sum().item(), ((b - mu) ** 2).sum().item()
    return dict(sse=sse, sst=sst, n=len(b), R2=1 - sse / sst if sst > 0 else float('nan'),
                rmse=float(np.sqrt(sse / len(b))))


def nexttok_ceiling(p_fit, b_fit, p_test):
    keys_fit = [tuple(r) for r in np.round(p_fit.numpy(), 9)]
    groups = {}
    for i, k in enumerate(keys_fit):
        groups.setdefault(k, []).append(i)
    means = {k: b_fit[idx].mean(0) for k, idx in groups.items()}
    mu = b_fit.mean(0)
    return torch.stack([means.get(tuple(r), mu) for r in np.round(p_test.numpy(), 9)])


def dist_corr(b, bhat, p, n=1500, seed=0):
    idx = torch.randperm(len(b), generator=torch.Generator().manual_seed(seed))[:n]
    iu = torch.triu_indices(len(idx), len(idx), 1)
    d = lambda z: torch.cdist(z[idx], z[idx])[iu[0], iu[1]]
    dp, db, dn = d(bhat), d(b), d(p)
    r2 = lambda u, w: float(np.corrcoef(u.numpy(), w.numpy())[0, 1] ** 2) if u.std() > 0 and w.std() > 0 else float('nan')
    sub = torch.randperm(len(dp), generator=torch.Generator().manual_seed(seed))[:20000]
    return dict(R2_belief=r2(db, dp), R2_nexttok=r2(dn, dp)), dict(d_pred=dp[sub], d_belief=db[sub], d_next=dn[sub])


@torch.no_grad()
def activations(model, x, v, chunk=4096):
    """{set: (B, T, d)} float32 on the CPU for the six single sets; 'concat' is assembled per position by at()."""
    out = {k: [] for k in SETS}
    for i in range(0, len(x), chunk):
        _, hs = model(M.inputs(x[i:i + chunk], v), resid=True)
        layers = [hs[0]] + [hs[j] for j in range(1, len(hs))] + [model.ln(hs[-1])]
        for k, h in zip(SETS[:-1], layers):
            out[k].append(h.float().cpu())  # float32 storage; cast to float64 per position when fitting
    return {k: torch.cat(vs) for k, vs in out.items() if vs}


def at(acts, k, t, dev):
    """Activations of set k at position t, (B, d_k) float64 on dev."""
    if k == 'concat':
        return torch.cat([acts[j][:, t] for j in SETS[:-1]], -1).to(dev).double()
    return acts[k][:, t].to(dev).double()


def main():
    global N_ANA
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3:
        N_ANA = int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    g = cfg['grammar']
    v, T = g['v'], cfg['T']
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(N_ANA, rules, seed=ANA_SEED + g['seed'])[g['L']]
    bel = BL.beliefs(x, rules, g)
    half = N_ANA // 2
    fit, test = slice(0, half), slice(half, None)
    positions = [t for t in sorted(bel) if t >= 1]
    skipped = [t for t in range(1, T) if t not in bel]

    # Ground-truth geometry: distinct beliefs, rank, PCA basis on unique beliefs (unweighted).
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
    psplit = {}
    for t in positions:
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        n_pre = int(pid.max()) + 1
        if n_pre < MIN_PREFIXES:
            continue
        side = torch.rand(n_pre, generator=torch.Generator().manual_seed(100 + t)) < 0.5
        f_idx, t_idx = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
        if len(f_idx) and len(t_idx):
            psplit[t] = (f_idx, t_idx, n_pre, int(side.sum()))
    d = cfg['d']
    elig = {'eL4': [t for t, sp in psplit.items() if sp[3] >= ELIG_FACTOR * (d + 1)],
            'eC': [t for t, sp in psplit.items() if sp[3] >= ELIG_FACTOR * (6 * d + 1)]}
    ckpts = res['ckpts']
    metrics, proj = {}, {}
    for ck in ckpts:
        state = torch.load(run_dir / f'ckpt_{ck}.pt', map_location=dev)
        model = M.Transformer(v, T, d=cfg['d'], layers=cfg['layers'], heads=cfg['heads']).to(dev)
        model.load_state_dict(state['state'])
        model.eval()
        acts = activations(model, x.to(dev), v)
        m = {k: {} for k in SETS + ['shuffle', 'nexttok', 'tokens'] + [k + '@prefix' for k in SETS + ['nexttok', 'tokens']]}
        keep = ck in (ckpts[0], ckpts[-1])
        pr = {}
        dists = {}
        for t in positions:
            b, p = bel[t]
            bf, bt = b[fit].to(dev), b[test].to(dev)
            mu = bf.mean(0)
            preds = {}
            for k in SETS:
                A = at(acts, k, t, dev)
                preds[k] = fit_predict(A[fit], bf, A[test])
                m[k][t] = score(bt, preds[k], mu)
            perm = torch.randperm(half, generator=torch.Generator().manual_seed(7))
            A = at(acts, 'concat', t, dev)
            preds['shuffle'] = fit_predict(A[fit], bf[perm.to(dev)], A[test])
            m['shuffle'][t] = score(bt, preds['shuffle'], mu)
            if ck == ckpts[-1]:
                preds['nexttok'] = nexttok_ceiling(p[fit], b[fit], p[test]).to(dev)
                m['nexttok'][t] = score(bt, preds['nexttok'], mu)
                oh = torch.nn.functional.one_hot(x[:, :t], v).flatten(1).double().to(dev)
                m['tokens'][t] = score(bt, fit_predict(oh[fit], bf, oh[test]), mu)
                dc, ds = dist_corr(bt.cpu(), preds['concat'].cpu(), p[test])
                m['concat'][t].update(dist=dc)
                dists[t] = {k: vv.numpy().astype(np.float32) for k, vv in ds.items()}
            if t in psplit:
                fi, ti = psplit[t][:2]
                bf2, bt2 = b[fi].to(dev), b[ti].to(dev)
                mu2 = bf2.mean(0)
                for k in SETS:
                    A = at(acts, k, t, dev)
                    m[k + '@prefix'][t] = score(bt2, fit_predict(A[fi.to(dev)], bf2, A[ti.to(dev)]), mu2)
                if ck == ckpts[-1]:
                    m['nexttok@prefix'][t] = score(bt2, nexttok_ceiling(p[fi], b[fi], p[ti]).to(dev), mu2)
                    oh = torch.nn.functional.one_hot(x[:, :t], v).flatten(1).double().to(dev)
                    m['tokens@prefix'][t] = score(bt2, fit_predict(oh[fi.to(dev)], bf2, oh[ti.to(dev)]), mu2)
            if keep:
                bmu, V = basis[t]
                bmu, V = bmu.to(dev), V.to(dev)
                P = lambda z: ((z - bmu) @ V).cpu().numpy().astype(np.float32)
                h = acts['L4'][test, t].double()
                h = h - h.mean(0)
                _, _, Vh = torch.linalg.svd(h, full_matrices=False)
                e = dict(uid=geom[t]['uid'][test].numpy(), true=P(bt), raw_pca=(h @ Vh[:3].T).numpy().astype(np.float32),
                         **{f'pred_{k}': P(preds[k]) for k in ('concat', 'L4', 'shuffle')})
                nodes = G.pending_nodes(t, g['s'], g['L'])
                for j, (l, q) in enumerate(nodes):
                    if (g['n'] if l == 0 else v) == 3:
                        e[f'tri{j}_L{l}_true'] = BL.node_marginal(bt, t, g, j).cpu().numpy().astype(np.float32)
                        e[f'tri{j}_L{l}_concat'] = BL.node_marginal(preds['concat'], t, g, j).cpu().numpy().astype(np.float32)
                pr[t] = e
        pooled = {}
        for k, mk in m.items():
            if mk:
                sse, sst = sum(r['sse'] for r in mk.values()), sum(r['sst'] for r in mk.values())
                pooled[k] = dict(R2=1 - sse / sst, rmse=float(np.sqrt(sse / sum(r['n'] for r in mk.values()))),
                                 positions=sorted(mk))
                if k.endswith('@prefix'):
                    for e, ts in elig.items():
                        ts = [t for t in ts if t in mk]
                        if ts:
                            pooled[k]['R2_' + e] = 1 - sum(mk[t]['sse'] for t in ts) / sum(mk[t]['sst'] for t in ts)
        metrics[ck] = dict(pooled=pooled, per_pos={k: {t: {kk: vv for kk, vv in r.items() if kk not in ('sse', 'sst')}
                                                       for t, r in mk.items()} for k, mk in m.items() if mk})
        if keep:
            proj[ck] = pr
        if dists:
            np.savez_compressed(out_dir / 'dists.npz', **{f't{t}_{k}': a for t, d in dists.items() for k, a in d.items()})
        print(f'{run_dir.name} ckpt {ck}: pooled R2 ' +
              ' '.join(f'{k}={pooled[k]["R2"]:.3f}' for k in pooled if k in ('L4', 'concat', 'shuffle', 'nexttok', 'tokens')
                       or k in ('L4@prefix', 'concat@prefix', 'tokens@prefix', 'nexttok@prefix')), flush=True)
    for ck, pr in proj.items():
        np.savez_compressed(out_dir / f'proj_ckpt{ck}.npz',
                            **{f't{t}_{k}': a for t, e in pr.items() for k, a in e.items()})
    hist = res['hist']
    json.dump(dict(run=run_dir.name, cfg=cfg, bayes=res['bayes'], final_test=hist[-1]['test'],
                   final_excess=hist[-1]['excess'], hist=[{k: h[k] for k in ('step', 'test', 'excess')} for h in hist],
                   test_pos=hist[-1]['test_pos'], bayes_pos=res['bayes_pos'], positions=positions, skipped=skipped,
                   prefix_split={t: dict(n_prefixes=n, n_fit_prefixes=nf, n_fit=len(f), n_test=len(tt))
                                 for t, (f, tt, n, nf) in psplit.items()},
                   eligible=elig,
                   prefix_split_skipped=[t for t in positions if t not in psplit],
                   geom={t: {k: vv for k, vv in d.items() if k != 'uid'} for t, d in geom.items()},
                   metrics=metrics, n_ana=N_ANA),
              open(out_dir / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
