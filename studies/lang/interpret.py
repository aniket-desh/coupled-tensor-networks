"""Look for each process's predicted intermediate computation in a trained TreeTN, at every checkpoint of a run.

  python -m studies.lang.interpret <run dir> --out <json> [--device cuda] [--n 6000]

For every tree node (l, k) (its leaf span S follows the model's perm; studies.lang.processes.spans) and every summary
the process defines over S (processes.Process.summary), three levels of evidence:

1. Decodability. Bonds are whitened over the probe set (studies.rhm.interpret.whitener: u = Wh (z - mu), gauge
   covariant), then a held-out probe is fit on half the set: multinomial logistic regression for 'class' summaries
   (accuracy; chance = majority-class rate) and ridge regression for 'reg' summaries (R^2 over all output dims; with
   'ceiling', the R^2 of the same ridge from the model's own centered output logits).
   Baselines: the same probe on shuffled targets (final checkpoint), and a 'local' probe on the one-hot tokens of S
   (additive in the tokens, so it can't read conjunctions such as f(x_i, x_j), a parity or a trigger flag;
   computed once per node). The step-0 checkpoint is the untrained baseline.
2. Invariance (primary summary; where the process defines resample_same). x' keeps x outside S and the summary over S
   but changes the surface tokens inside S. sub = E||P(u - u')||^2 / E||P(u - u_ind)||^2 in the probe subspace
   (P projects on the top LDA directions for a class summary, k = min(C - 1, d), or on the top ridge directions
   holding 90% of the predicted variance, at most d / 2, for a reg summary), with u_ind the bond of an independent example; 0 means invariant, 1 means no more invariant
   than an unrelated input. whole = the same on the whole whitened bond. pred_same: argmax unchanged. Where the
   process restricts the test (Process.inv_mask, e.g. the trigger: only examples with a trigger in the span), only
   those examples enter, and n is reported.
3. Intervention (internal nodes, every summary). A donor example with a different summary over S defines the
   spliced input s = (recipient outside S, donor inside S); only pairs with a valid spliced input are used. The
   donor's bond is patched into the recipient: 'whole' (the whole bond, which in a tree equals running the model
   on s; checks.py verifies this), 'sub' (only the probe subspace: z_rec + Wp Q Q^T (u_don - u_rec)) and 'rand'
   (N_RAND random subspaces of the same dimension). The counterfactual for 'whole' is p*(s); for 'sub', 'rand' and
   'none' (no patch) it is p*(s_name), where s_name splices only the summary's support (the tokens in S that
   determine it, Process.support), so a subspace is asked to carry that summary and nothing else. Reported:
   KL(counterfactual || q_patched), KL(p*(recipient) || q_patched), and 'moved' = the fraction of pairs whose
   patched argmax equals the counterfactual's argmax, among pairs where that argmax differs from the recipient's.

Forward passes run in float32 and everything downstream (whitening, probes, KLs) in float64; the random choices use fixed seeds, so every checkpoint sees the
same probe examples, pairs and subspaces."""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from studies.lang import common as L
from studies.lang import processes as P
from studies.rhm.interpret import lda, probe_acc, whitener

SEED = 4242
N_RAND = 3
RIDGE = 1e-3  # ridge penalty on unit-variance whitened features (relative to N)
REG_VAR = 0.9  # reg probe subspace: top ridge directions holding this share of the predicted variance (<= d / 2)
DT = torch.float64


def checkpoints(run):
    ck = sorted(Path(run).glob('ckpt_*.pt'), key=lambda p: int(re.findall(r'\d+', p.name)[-1]))
    return [(int(re.findall(r'\d+', p.name)[-1]), p) for p in ck]


def ridge(Xtr, Ytr, Xte, Yte, lam=RIDGE):
    """R^2 (1 - SSE / SST over all output dims) of ridge regression with intercept; returns (r2, W (d, C))."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A, B = Xtr - mx, Ytr - my
    W = torch.linalg.solve(A.T @ A + lam * len(A) * torch.eye(A.shape[1], dtype=DT, device=A.device), A.T @ B)
    pred = (Xte - mx) @ W + my
    sst = ((Yte - Yte.mean(0)) ** 2).sum()
    return (1 - ((pred - Yte) ** 2).sum() / sst).item() if sst > 0 else float('nan'), W


def onehot_tokens(x, S, V):
    return torch.nn.functional.one_hot(x[:, S], V).reshape(len(x), -1).to(DT)


def fit(kind, Xtr, ytr, Xte, yte, C):
    """(score, basis Q (d, k) of the probe subspace)."""
    if kind == 'class':
        acc = probe_acc(Xtr, ytr, Xte, yte, C)
        Q, _ = lda(Xtr - Xtr.mean(0), ytr, C)
        return acc, Q[:, :min(C - 1, Xtr.shape[1])]
    r2, W = ridge(Xtr, ytr, Xte, yte)
    U, s, _ = torch.linalg.svd(W, full_matrices=False)
    if not s.numel() or s.max() <= 0:
        return r2, U[:, :0]
    # the probe subspace for a reg target: the fewest top directions of the ridge map holding REG_VAR of the
    # predicted variance (on whitened features it is proportional to s^2), at most half the bond, so that a random
    # subspace of the same dimension is a real control even for wide targets (e.g. the 64-dim dictionary)
    frac = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    k = min(int((frac < REG_VAR).sum()) + 1, max(1, Xtr.shape[1] // 2))
    return r2, U[:, :k]


def chance(kind, ytr, yte, C):
    if kind == 'class':
        return (yte == torch.bincount(ytr, minlength=C).argmax()).double().mean().item()
    return 0.0


def kl(p, logq):
    return (torch.xlogy(p, p) - p * logq).sum(1).mean().item()


class Context:
    """Fixed probe examples, splits, resampled pairs, donors and random directions for one process and model layout."""

    def __init__(self, cfg, n, device):
        self.proc = L.process(cfg).to(device)
        g = torch.Generator(device).manual_seed(L.PROBE_SEED)
        self.x, _ = self.proc.sample_xy(n, g)
        self.half = n // 2
        self.spans = P.spans(cfg.get('perm'))
        self.dev = device
        gs = torch.Generator(device).manual_seed(SEED)
        xt = self.x[self.half:]
        self.ind = torch.randperm(len(xt), generator=gs, device=device)  # independent partner for invariance
        self.don = torch.randperm(len(xt), generator=gs, device=device)  # donor for patching
        self.res, self.local, self.pairs = {}, {}, {}
        for node, S in self.spans.items():
            self.res[node] = self.proc.resample_same(xt, S, gs)
            spl = xt.clone()
            spl[:, S] = xt[self.don][:, S]
            self.pairs[node] = spl
        self.sum_tr = {node: self.proc.summary(self.x[:self.half], S) for node, S in self.spans.items()}
        self.inv_mask = {node: self.proc.inv_mask(xt, S) for node, S in self.spans.items()}
        self.sum_te = {node: self.proc.summary(xt, S) for node, S in self.spans.items()}
        self.gr = torch.Generator(device).manual_seed(SEED + 1)


@torch.no_grad()
def acts(model, x, edit=None):
    """Forward in the model's own precision (float32: consumer GPUs run float64 at 1/64 rate); activations and
    logits are returned in float64 for the whitening, probes and KLs."""
    logits, a = model(x, acts=True, edit=edit)
    return logits.to(DT), [t.to(DT) for t in a]


def analyze(model, ctx, final=False):
    proc, xt, h = ctx.proc, ctx.x[ctx.half:], ctx.half
    logit_all, A = acts(model, ctx.x)
    logit_te = logit_all[h:]
    out = {}
    for (l, k), S in ctx.spans.items():
        Z = A[l][:, k]
        mu, Wh, Wp = whitener(Z[:h])
        U = (Z - mu) @ Wh.T
        Utr, Ute = U[:h], U[h:]
        node = {}
        res = ctx.res[(l, k)]
        Ures = None
        if res is not None:
            _, Ar = acts(model, res)
            Ures = (Ar[l][:, k] - mu) @ Wh.T
            logit_res, _ = acts(model, res)
        for name, (kind, ytr, C) in ctx.sum_tr[(l, k)].items():
            yte = ctx.sum_te[(l, k)][name][1]
            if kind == 'reg':
                ytr, yte = ytr.to(DT), yte.to(DT)
            d = {}
            if U.shape[1] == 0:
                node[name] = dict(probe=float('nan'), d=0)
                continue
            score, Q = fit(kind, Utr, ytr, Ute, yte, C)
            d.update(probe=score, chance=chance(kind, ytr, yte, C), d=U.shape[1], k=Q.shape[1])
            if kind == 'reg':
                # ceiling: the same ridge fit from the model's own centered output logits. In a tree the logits are
                # linear in each bond (others fixed), so a bond can't be expected to hold more of a nonlinear target
                # (e.g. clipped evidence) than the output itself expresses
                Lc = logit_all - logit_all.mean(1, keepdim=True)
                d['ceiling'] = ridge(Lc[:h], ytr, Lc[h:], yte)[0]
            if final:
                perm = torch.randperm(len(ytr), generator=ctx.gr, device=ctx.dev)
                d['shuffle'] = fit(kind, Utr, ytr[perm], Ute, yte, C)[0]
            key = ((l, k), name)
            if key not in ctx.local:
                Xl = onehot_tokens(ctx.x, S, proc.V_in)
                ctx.local[key] = fit(kind, Xl[:h], ytr, Xl[h:], yte, C)[0]
            d['local'] = ctx.local[key]
            # invariance (primary summary only)
            if name == proc.primary_of(ctx.sum_tr[(l, k)]) and Ures is not None and Q.shape[1] > 0:
                mk = ctx.inv_mask[(l, k)]
                sel = (lambda V: V[mk]) if mk is not None else (lambda V: V)
                if mk is None or mk.sum() >= 50:
                    Ua, Ub, Ui = sel(Ute), sel(Ures), sel(Ute[ctx.ind])
                    P_ = lambda V: V @ Q
                    den = ((P_(Ua) - P_(Ui)) ** 2).sum(1).mean()
                    d['inv'] = dict(sub=(((P_(Ua) - P_(Ub)) ** 2).sum(1).mean() / den).item(),
                                    whole=(((Ua - Ub) ** 2).sum(1).mean() / ((Ua - Ui) ** 2).sum(1).mean()).item(),
                                    pred_same=(sel(logit_te).argmax(1) == sel(logit_res).argmax(1)).double().mean().item(),
                                    n=int(len(Ua)))
            # intervention (internal nodes)
            if l >= 1 and Q.shape[1] > 0:
                d['patch'] = patch(model, ctx, l, k, S, name, kind, mu, Wh, Wp, Q, Z[h:], logit_te)
            node[name] = d
        out[f'{l}_{k}'] = node
    return out


def patch(model, ctx, l, k, S, name, kind, mu, Wh, Wp, Q, Zte, logit_te):
    proc, xt = ctx.proc, ctx.x[ctx.half:]
    spl, don = ctx.pairs[(l, k)], ctx.don
    tgt_rec, tgt_don = ctx.sum_te[(l, k)][name][1], ctx.sum_te[(l, k)][name][1][don]
    differ = (tgt_rec != tgt_don) if kind == 'class' else (tgt_rec != tgt_don).any(1)
    sup = proc.support(name, S)
    spl_sup = xt.clone()
    spl_sup[:, sup] = xt[don][:, sup]
    m = proc.valid(spl) & proc.valid(spl_sup) & differ
    if m.sum() < 50:
        return dict(n=int(m.sum()))
    rec, zr, zd = xt[m], Zte[m], Zte[don][m]
    pS, pT, pB = proc.bayes(spl[m]).to(DT), proc.bayes(spl_sup[m]).to(DT), proc.bayes(rec).to(DT)

    def score(lq, target):
        hit = target.argmax(1) != pB.argmax(1)
        return dict(kl_target=kl(target, lq), kl_base=kl(pB, lq),
                    moved=(lq.argmax(1) == target.argmax(1))[hit].double().mean().item() if hit.any() else float('nan'))

    def run(znew, target):
        def edit(level, z):
            if level == l:
                z = z.clone()
                z[:, k] = znew.to(z.dtype)
            return z
        return score(torch.log_softmax(acts(model, rec, edit)[0], 1), target)

    du = (zd - mu) @ Wh.T - (zr - mu) @ Wh.T
    out = dict(n=int(m.sum()), n_moved=int((pT.argmax(1) != pB.argmax(1)).sum()), whole=run(zd, pS),
               sub=run(zr + (du @ Q @ Q.T) @ Wp.T, pT))
    rands = []
    for _ in range(N_RAND):
        R, _ = torch.linalg.qr(torch.randn(Q.shape[0], Q.shape[1], generator=ctx.gr, device=ctx.dev, dtype=DT))
        rands.append(run(zr + (du @ R @ R.T) @ Wp.T, pT))
    out['rand'] = {key: sum(r[key] for r in rands) / N_RAND for key in rands[0]}
    out['none'] = score(torch.log_softmax(logit_te[m], 1), pT)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n', type=int, default=6000)
    a = ap.parse_args()
    cks = checkpoints(a.run)
    if not cks:
        raise SystemExit(f'no checkpoints in {a.run}')
    t = time.time()
    res = dict(run=str(a.run), n=a.n, steps={})
    ctx = None
    for i, (step, path) in enumerate(cks):
        model, cfg, _ = L.load(path, a.device)
        if cfg['model'] != 'tree':
            raise SystemExit('interpretation is defined for tree models only')
        model = model.eval()
        ctx = ctx or Context(cfg, a.n, a.device)
        res['cfg'] = cfg
        res['steps'][str(step)] = analyze(model, ctx, final=i == len(cks) - 1)
    res['seconds'] = time.time() - t
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f)
    print(f'{a.run}: {len(cks)} checkpoints in {res["seconds"]:.0f}s -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
