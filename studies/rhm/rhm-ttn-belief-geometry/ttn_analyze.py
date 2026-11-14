"""Probes from a TreeTN's bond states to the exact RHM belief states, per run (the method of
rhm-belief-geometry/analyze.py, whose docstring defines the splits, metrics and controls; only what differs is here).

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/ttn_analyze.py RUN_DIR OUT_DIR [N_ANA]

Hidden state. For prefix length t the tree runs on x_<t with positions >= t masked (the training input), and its bond
states are collected: acts[l] (B, T / 2^l, r), l = 0 leaves .. D-1 the two bonds h_L, h_R under the root. A bond
depends only on the tokens in its span. Activation sets at position t:
  top      [h_L, h_R], 2r features: the two bonds the root contracts into logits. Primary probe.
  lev<l>   all bonds of tree level l = 1 .. D-2, concatenated.
  all      levels 1 .. D-1 concatenated.
  done     the maximal fully observed nodes (the binary decomposition of [0, t), e.g. spans 0:4, 4:6, 6:7 for t = 7).
  top_bil  [h_L, h_R, h_L (x) h_R], 2r + r^2 features (first and last checkpoint only).
Why top_bil: for t <= T/2 the right bond h_R sees only masks and is constant, so h_L carries the whole prefix and an
affine probe on `top` can express any linear code of the belief. For t > T/2 the belief depends jointly on the left
half (through its symbol) and the right half's partial evidence, and no bond has seen both: an affine probe on
[h_L, h_R] can only capture the part additive in the two halves, while the interaction is a function of the pair, which
the bilinear features can carry (the root core contracts exactly h_L (x) h_R).

Affine and bilinear probe scores are invariant to the bond gauge z -> G z; the raw PCA of the top bonds is not.

Eligible positions: at least ELIG_FACTOR * (features + 1) distinct fit prefixes, for `top` (e_top) and for `top_bil`
(e_bil). Each <key>@prefix entry is pooled over four groups, with H = T / 2:
  R2_h1   e_top, t <= H     (the first-half criterion)
  R2_h2   e_top, t > H
  R2_h2b  e_bil, t > H      (affine and bilinear compared on the same positions)
  R2_e    e_top, all t
Checkpoint 0 is the identity-init network (bonds near an additive function of the tokens), the init control.

Controls beyond the transformer analysis (final checkpoint unless noted; none of the first three needs the network):
  addceil@prefix  t > H: the best fit additive in the two halves, f(left-half symbol) + g(right-half parse state), by
                  least squares on their one-hot codes. Every affine probe on bonds from disjoint spans (top, all, done)
                  is bounded by it, so the affine-bilinear gap is meaningful only against this ceiling.
                  Group h2c = the h2b positions where it is below ADD_ROOM.
  tokprod@prefix  t > H: least squares on [onehot(left tokens), onehot(right prefix tokens), their products], the
                  token readout that sees cross-half products like top_bil does; computed where it is eligible
                  (ELIG_FACTOR * (features + 1) distinct fit prefixes), listed otherwise. Group h2t = the h2b
                  positions where it exists.
  span@belief     see the belief split below.
  <set>~scr@prefix  the same probe fit to a scrambled target: an independent N(0, 1) 8-vector per distinct belief
                  (seeded per position). A position has few distinct beliefs, so a probe that linearly identifies the
                  belief state fits any labelling of the states; R2 minus this is what the belief's geometry adds.
Belief split (<set>@belief; first and last checkpoint; positions with at least MIN_BELIEFS distinct beliefs): the
distinct beliefs are assigned at random to fit or test, so the probe must extrapolate to belief vectors it never saw,
which a code affine in the belief allows and a mere identification of states does not (the scrambled target scores
about 0 here by construction). span@belief is its ceiling: the test beliefs projected on the affine span of the fit
beliefs. Pooled as R2_bel over the e_top positions with enough beliefs."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE))
import analyze as AZ  # noqa: E402
import beliefs as BL  # noqa: E402
import ttn_run as TR  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

N_ANA = 60000
ADD_ROOM, MIN_BELIEFS = 0.8, 20


def done_nodes(t, D):
    """Maximal fully observed nodes for prefix length t: [(tree level, node index)], left to right."""
    out, pos = [], 0
    for l in range(D - 1, -1, -1):
        if pos + 2 ** l <= t:
            out.append((l, pos >> l))
            pos += 2 ** l
    return out


@torch.no_grad()
def bond_sets(model, x, t, v, bilinear, chunk=8192):
    """{set: (B, d_set) float32 on x's device} for prefix length t."""
    D, parts = model.D, []
    for i in range(0, len(x), chunk):
        xm = x[i:i + chunk].clone()
        xm[:, t:] = v
        parts.append(model(xm, acts=True)[1])
    acts = [torch.cat([p[l] for p in parts]) for l in range(D)]
    s = {'top': acts[D - 1].flatten(1)}
    for l in range(1, D - 1):
        s[f'lev{l}'] = acts[l].flatten(1)
    s['all'] = torch.cat([acts[l].flatten(1) for l in range(1, D)], 1)
    s['done'] = torch.cat([acts[l][:, k] for l, k in done_nodes(t, D)], 1)
    if bilinear:
        hl, hr = acts[D - 1][:, 0], acts[D - 1][:, 1]
        s['top_bil'] = torch.cat([s['top'], (hl[:, :, None] * hr[:, None, :]).flatten(1)], 1)
    return s


def onehot_ids(*cols):
    """One-hot code of the joint value of integer columns (N,), as (N, n_distinct) float64."""
    _, ids = torch.unique(torch.stack(cols, 1), dim=0, return_inverse=True)
    return torch.nn.functional.one_hot(ids).double()


def additive_design(trees, t, L):
    """[onehot(left-half symbol), onehot(right-half parse state)] for t > T/2: the right state is the tuple of symbols
    of the fully observed nodes inside the right half (tree level l, node k <-> grammar level L - l, position k)."""
    right = [trees[L - l][:, k] for l, k in done_nodes(t, L) if (k << l) >= 2 ** (L - 1)]
    return torch.cat([onehot_ids(trees[1][:, 0]), onehot_ids(*right)], 1)


def token_product(x, t, v):
    H = x.shape[1] // 2
    a = torch.nn.functional.one_hot(x[:, :H], v).flatten(1).double()
    b = torch.nn.functional.one_hot(x[:, H:t], v).flatten(1).double()
    return torch.cat([a, b, (a[:, :, None] * b[:, None, :]).flatten(1)], 1)


def main():
    global N_ANA
    run_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3:
        N_ANA = int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    g, r, T, v = cfg['grammar'], cfg['r'], cfg['T'], cfg['V']
    H, L = T // 2, g['L']
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], L, seed=g['seed'])
    trees = G.sample_trees(N_ANA, rules, seed=AZ.ANA_SEED + g['seed'])  # the transformer analysis's sequences
    x = trees[L]
    bel = BL.beliefs(x, rules, g)
    half = N_ANA // 2
    fit, test = slice(0, half), slice(half, None)
    positions = [t for t in sorted(bel) if t >= 1]
    skipped = [t for t in range(1, T) if t not in bel]
    score = AZ.score

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
    psplit = {}  # same seeds as the transformer analysis, so both models are scored on the same held-out prefixes
    for t in positions:
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        n_pre = int(pid.max()) + 1
        if n_pre < AZ.MIN_PREFIXES:
            continue
        side = torch.rand(n_pre, generator=torch.Generator().manual_seed(100 + t)) < 0.5
        f_idx, t_idx = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
        if len(f_idx) and len(t_idx):
            psplit[t] = (f_idx, t_idx, n_pre, int(side.sum()))
    bsplit = {}  # belief split: distinct beliefs to fit or test
    for t in positions:
        nu = geom[t]['n_unique']
        if nu >= MIN_BELIEFS:
            side = torch.rand(nu, generator=torch.Generator().manual_seed(300 + t)) < 0.5
            bsplit[t] = (torch.nonzero(side[geom[t]['uid']])[:, 0], torch.nonzero(~side[geom[t]['uid']])[:, 0])
    p_top, p_bil = 2 * r, 2 * r + r * r
    enough = lambda t, p: t in psplit and psplit[t][3] >= AZ.ELIG_FACTOR * (p + 1)
    e_top = [t for t in positions if enough(t, p_top)]
    e_bil = [t for t in positions if enough(t, p_bil)]

    # Controls that do not need the network.
    ctrl, tokprod_skipped = {}, []
    put = lambda d, key, t, val: d.setdefault(key, {}).__setitem__(t, val)
    for t in positions:
        b, p = bel[t]
        bd = b.to(dev)
        oh = torch.nn.functional.one_hot(x[:, :t], v).flatten(1).double().to(dev)
        mu = bd[fit].mean(0)
        put(ctrl, 'nexttok', t, score(bd[test], AZ.nexttok_ceiling(p[fit], b[fit], p[test]).to(dev), mu))
        put(ctrl, 'tokens', t, score(bd[test], AZ.fit_predict(oh[fit], bd[fit], oh[test]), mu))
        if t in psplit:
            fc, tc = psplit[t][:2]
            fi, ti = fc.to(dev), tc.to(dev)
            mu2 = bd[fi].mean(0)
            put(ctrl, 'nexttok@prefix', t, score(bd[ti], AZ.nexttok_ceiling(p[fc], b[fc], p[tc]).to(dev), mu2))
            put(ctrl, 'tokens@prefix', t, score(bd[ti], AZ.fit_predict(oh[fi], bd[fi], oh[ti]), mu2))
            if t > H:
                A = additive_design(trees, t, L).to(dev)
                put(ctrl, 'addceil@prefix', t, score(bd[ti], AZ.fit_predict(A[fi], bd[fi], A[ti]), mu2))
                if enough(t, H * v + (t - H) * v + H * v * (t - H) * v):
                    A = token_product(x, t, v).to(dev)
                    put(ctrl, 'tokprod@prefix', t, score(bd[ti], AZ.fit_predict(A[fi], bd[fi], A[ti]), mu2))
                else:
                    tokprod_skipped.append(t)
        if t in bsplit:
            fi, ti = (i.to(dev) for i in bsplit[t])
            mu3 = bd[fi].mean(0)
            U = torch.unique(bd[fi].round(decimals=9), dim=0) - mu3
            _, sv, Vh = torch.linalg.svd(U, full_matrices=False)
            B = Vh[sv > 1e-9 * sv[0]].T
            put(ctrl, 'span@belief', t, score(bd[ti], mu3 + (bd[ti] - mu3) @ B @ B.T, mu3))
    addc = ctrl.get('addceil@prefix', {})
    groups = {'h1': [t for t in e_top if t <= H], 'h2': [t for t in e_top if t > H],
              'h2b': [t for t in e_bil if t > H], 'e': e_top,
              'h2c': [t for t in e_bil if t > H and addc[t]['R2'] < ADD_ROOM],
              'h2t': [t for t in e_bil if t > H and t in ctrl.get('tokprod@prefix', {})],
              'bel': [t for t in e_top if t in bsplit]}

    ckpts = res['ckpts']
    xd = x.to(dev)
    metrics, proj = {}, {}
    for ck in ckpts:
        model = TR.build(cfg).to(dev)
        model.load_state_dict(torch.load(run_dir / f'ckpt_{ck}.pt', map_location=dev)['state'])
        model.eval()
        keep = ck in (ckpts[0], ckpts[-1])
        last = ck == ckpts[-1]
        m = {k: dict(d) for k, d in ctrl.items()} if last else {}
        pr = {}
        for t in positions:
            S = bond_sets(model, xd, t, v, bilinear=keep)
            bd = bel[t][0].to(dev)
            bf, bt = bd[fit], bd[test]
            mu = bf.mean(0)
            preds = {}
            for k, A in S.items():
                A = A.double()
                pk = AZ.fit_predict(A[fit], bf, A[test])
                put(m, k, t, score(bt, pk, mu))
                if keep and k in ('all', 'top', 'top_bil'):
                    preds[k] = pk
            perm = torch.randperm(half, generator=torch.Generator().manual_seed(7)).to(dev)
            A = S['all'].double()
            pk = AZ.fit_predict(A[fit], bf[perm], A[test])
            put(m, 'shuffle', t, score(bt, pk, mu))
            if keep:
                preds['shuffle'] = pk
            if t in psplit:
                fi, ti = (i.to(dev) for i in psplit[t][:2])
                mu2 = bd[fi].mean(0)
                for k, A in S.items():
                    A = A.double()
                    put(m, k + '@prefix', t, score(bd[ti], AZ.fit_predict(A[fi], bd[fi], A[ti]), mu2))
                if last:
                    scr = torch.randn(geom[t]['n_unique'], 8, generator=torch.Generator().manual_seed(500 + t),
                                      dtype=torch.float64)[geom[t]['uid']].to(dev)
                    for k in ('top', 'top_bil', 'all'):
                        A = S[k].double()
                        put(m, k + '~scr@prefix', t, score(scr[ti], AZ.fit_predict(A[fi], scr[fi], A[ti]), scr[fi].mean(0)))
            if keep and t in bsplit:
                fi, ti = (i.to(dev) for i in bsplit[t])
                mu3 = bd[fi].mean(0)
                for k in ('top', 'top_bil', 'all'):
                    A = S[k].double()
                    put(m, k + '@belief', t, score(bd[ti], AZ.fit_predict(A[fi], bd[fi], A[ti]), mu3))
            if keep:
                bmu, V = (z.to(dev) for z in basis[t])
                P = lambda z: ((z - bmu) @ V).cpu().numpy().astype(np.float32)
                h = S['top'][test].double()
                h = h - h.mean(0)
                _, _, Vh = torch.linalg.svd(h, full_matrices=False)
                e = dict(uid=geom[t]['uid'][test].numpy(), true=P(bt), raw_pca=(h @ Vh[:3].T).cpu().numpy().astype(np.float32),
                         **{f'pred_{k}': P(preds[k]) for k in ('all', 'top', 'top_bil', 'shuffle')})
                for j, (l, q) in enumerate(G.pending_nodes(t, g['s'], L)):
                    if (g['n'] if l == 0 else v) == 3:
                        e[f'tri{j}_L{l}_true'] = BL.node_marginal(bt, t, g, j).cpu().numpy().astype(np.float32)
                        for k in ('top', 'top_bil'):
                            e[f'tri{j}_L{l}_{k}'] = BL.node_marginal(preds[k], t, g, j).cpu().numpy().astype(np.float32)
                pr[t] = e
            del S, preds
        pooled = {}
        for k, mk in m.items():
            sse, sst = sum(q['sse'] for q in mk.values()), sum(q['sst'] for q in mk.values())
            pooled[k] = dict(R2=1 - sse / sst, rmse=float(np.sqrt(sse / sum(q['n'] for q in mk.values()))), positions=sorted(mk))
            if '@' in k:
                for gname, ts in groups.items():
                    ts = [t for t in ts if t in mk]
                    if ts and len(ts) == len(groups[gname]):  # a key that misses some of a group's positions is not pooled
                        pooled[k]['R2_' + gname] = 1 - sum(mk[t]['sse'] for t in ts) / sum(mk[t]['sst'] for t in ts)
        metrics[ck] = dict(pooled=pooled, per_pos={k: {t: {kk: vv for kk, vv in q.items() if kk not in ('sse', 'sst')}
                                                       for t, q in mk.items()} for k, mk in m.items()})
        if keep:
            proj[ck] = pr
        show = lambda k, f='R2': f'{pooled[k][f]:.3f}' if k in pooled and f in pooled[k] else '-'
        print(f'{run_dir.name} ckpt {ck}: seq top={show("top")} all={show("all")} shuffle={show("shuffle")} | h1 top='
              f'{show("top@prefix", "R2_h1")} scr={show("top~scr@prefix", "R2_h1")} tokens={show("tokens@prefix", "R2_h1")} | h2b '
              f'top={show("top@prefix", "R2_h2b")} bil={show("top_bil@prefix", "R2_h2b")} scr={show("top_bil~scr@prefix", "R2_h2b")} '
              f'addceil={show("addceil@prefix", "R2_h2b")} | belief split top={show("top@belief", "R2_bel")} '
              f'bil={show("top_bil@belief", "R2_bel")} span={show("span@belief", "R2_bel")}', flush=True)
    for ck, pr in proj.items():
        np.savez_compressed(out_dir / f'proj_ckpt{ck}.npz', **{f't{t}_{k}': a for t, e in pr.items() for k, a in e.items()})
    hist = res['hist']
    sets = ['done'] + [f'lev{l}' for l in range(1, int(np.log2(T)) - 1)] + ['top', 'all']
    json.dump(dict(run=run_dir.name, cfg=cfg, n_params=res['n_params'], status=res['status'], bayes=res['bayes'],
                   final_test=hist[-1]['test'], final_excess=hist[-1]['excess'],
                   hist=[{k: h[k] for k in ('step', 'test', 'excess')} for h in hist], test_pos=hist[-1]['test_pos'],
                   bayes_pos=res['bayes_pos'], positions=positions, skipped=skipped, half=H, sets=sets,
                   features=dict(top=p_top, top_bil=p_bil),
                   prefix_split={t: dict(n_prefixes=n, n_fit_prefixes=nf, n_fit=len(f), n_test=len(tt))
                                 for t, (f, tt, n, nf) in psplit.items()},
                   prefix_split_skipped=[t for t in positions if t not in psplit],
                   belief_split=sorted(bsplit), tokprod_skipped=tokprod_skipped,
                   eligible=dict(e_top=e_top, e_bil=e_bil, **groups),
                   geom={t: {k: vv for k, vv in d.items() if k != 'uid'} for t, d in geom.items()},
                   metrics=metrics, n_ana=N_ANA),
              open(out_dir / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
