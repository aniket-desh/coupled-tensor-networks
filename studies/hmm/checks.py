"""Checks of studies/hmm/processes.py:  python -m studies.hmm.checks [out.json]
  kernels      sum_x T[x] is row-stochastic and pi is stationary; mess3(0.05, 0.85) equals the matrix printed in Shai et
               al. (App. A.3); the rrxor mixed-state presentation from pi has 36 distinct belief states.
  filtering    beliefs() and the next-token distribution against brute-force enumeration of every hidden path
               (every process, short sequences), and the Bayes loss against the entropy of the enumerated distribution.
  sampler      empirical frequencies of the first three tokens against the exact distribution.
  products     factored: the joint belief is the outer product of the factors' own beliefs; coupled: it is not, and
               factor A's marginal process is still mess3(0.05, 0.85); coins3: the belief is the posterior from counts."""
import os
import itertools
import json
import sys

import torch

from studies.hmm import processes as P

results = []


def check(name, cond, note=''):
    results.append(dict(name=name, ok=bool(cond), note=note))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {note}')


def brute(proc, n):
    """{prefix tuple (length <= n): P(prefix, state after it) (S,)} by summing over every hidden path."""
    T, pi, V, S = proc['T'], proc['pi'], proc['V'], proc['S']
    out = {(): pi.clone()}
    for xs in itertools.product(range(V), repeat=n):
        for k in range(1, n + 1):
            if xs[:k] in out:
                continue
            acc = torch.zeros(S, dtype=P.F64)
            for path in itertools.product(range(S), repeat=k + 1):
                pr = pi[path[0]]
                for i in range(k):
                    pr = pr * T[xs[i], path[i], path[i + 1]]
                acc[path[-1]] += pr
            out[xs[:k]] = acc
    return out


def kernels():
    for name in P.NAMES:
        p = P.get(name)
        M = p['T'].sum(0)
        check(f'{name}: row-stochastic kernel and stationary start',
              (M.sum(1) - 1).abs().max() < 1e-12 and (p['pi'] @ M - p['pi']).abs().max() < 1e-12 and (p['T'] >= 0).all())
    ref = torch.tensor([[.765, .00375, .00375], [.0425, .0675, .00375], [.0425, .00375, .0675]], dtype=P.F64)
    T = P.mess3(0.05, 0.85)
    perm = lambda e: [e] + [i for i in range(3) if i != e]
    same = all((T[e][perm(e)][:, perm(e)] - ref).abs().max() < 1e-12 for e in range(3))
    check('mess3(0.05, 0.85): T^(0) is the printed matrix, T^(1), T^(2) its permutations', same)
    p = P.get('rrxor')
    seen, frontier = {tuple(p['pi'].round(decimals=9).tolist())}, [p['pi']]
    while frontier:
        nxt = []
        for eta in frontier:
            for x in range(2):
                j = eta @ p['T'][x]
                if j.sum() > 1e-12:
                    e2 = j / j.sum()
                    key = tuple(e2.round(decimals=9).tolist())
                    if key not in seen:
                        seen.add(key)
                        nxt.append(e2)
        frontier = nxt
    check('rrxor: 36 distinct belief states from the stationary start', len(seen) == 36, f'{len(seen)} states')


def filtering():
    for name, n in (('mess3', 4), ('rrxor', 5), ('z1r', 5), ('coins3', 5), ('factored', 2), ('coupled', 2)):
        p = P.get(name)
        ref = brute(p, n)
        seqs = [xs for xs in itertools.product(range(p['V']), repeat=n) if ref[xs].sum() > 0]
        x = torch.tensor(seqs)
        eta, pn = P.beliefs(p, x)
        err_b = err_p = 0.0
        for i, xs in enumerate(seqs):
            for t in range(n):
                j = ref[xs[:t]]
                err_b = max(err_b, (eta[i, t] - j / j.sum()).abs().max().item())
                nt = torch.stack([ref[xs[:t] + (c,)].sum() if xs[:t] + (c,) in ref else torch.zeros((), dtype=P.F64)
                                  for c in range(p['V'])]) / j.sum()
                err_p = max(err_p, (pn[i, t] - nt).abs().max().item())
        w = torch.stack([ref[xs].sum() for xs in seqs])
        H = -(w * w.log()).sum().item() / n  # entropy per token of the length-n distribution
        loss = (w[:, None] * -pn.gather(2, x[..., None])[..., 0].log()).sum(0).mean().item()
        check(f'{name}: filtering vs enumeration of hidden paths (n = {n})', err_b < 1e-12 and err_p < 1e-12 and abs(H - loss) < 1e-12,
              f'max |eta - enum| {err_b:.1e}, |P(next) - enum| {err_p:.1e}, {len(seqs)} sequences, total prob {w.sum():.12f}')


def sampler():
    for name in P.NAMES:
        p = P.get(name)
        ref = brute(p, 3) if p['S'] < 9 else None
        N = 200_000
        x = P.sample(p, N, 3, g=torch.Generator().manual_seed(0))
        V = p['V']
        emp = torch.bincount(x[:, 0] * V * V + x[:, 1] * V + x[:, 2], minlength=V ** 3).double() / N
        if ref is not None:
            exact = torch.stack([ref[xs].sum() if xs in ref else torch.zeros((), dtype=P.F64)
                                 for xs in itertools.product(range(V), repeat=3)])
        else:  # 9-state processes: the exact distribution from the filter's own chain rule (checked above)
            xs = torch.tensor(list(itertools.product(range(V), repeat=3)))
            exact = P.beliefs(p, xs)[1].gather(2, xs[..., None])[..., 0].prod(1)
        tol = 5 * (exact.max() / N).sqrt().item()
        check(f'{name}: sampler frequencies of the first three tokens', (emp - exact).abs().max() < tol,
              f'max |emp - exact| {(emp - exact).abs().max():.1e} (tolerance {tol:.1e})')


def products():
    g = torch.Generator().manual_seed(1)
    f, c = P.get('factored'), P.get('coupled')
    x = P.sample(f, 2000, 8, g=g)
    eta = P.beliefs(f, x)[0]
    A = dict(T=P.mess3(0.05, 0.85)); A.update(pi=P.stationary(A['T']))
    B = dict(T=P.mess3(0.15, 0.6)); B.update(pi=P.stationary(B['T']))
    ea, eb = P.beliefs(A, x // 3)[0], P.beliefs(B, x % 3)[0]
    outer = (ea[..., :, None] * eb[..., None, :]).flatten(-2)
    ma, mb = P.marginals(f, eta)
    check('factored: joint belief = outer product of the factors\' own beliefs',
          (eta - outer).abs().max() < 1e-12 and (ma - ea).abs().max() < 1e-12 and (mb - eb).abs().max() < 1e-12)
    xc = P.sample(c, 2000, 8, g=g)
    ec = P.beliefs(c, xc)[0]
    ca, cb = P.marginals(c, ec)
    dev = (ec - (ca[..., :, None] * cb[..., None, :]).flatten(-2)).abs().max().item()
    TA = c['T'].reshape(3, 3, 3, 3, 3, 3).sum((1, 5))  # sum over xb and b' -> (xa, a, b, a'), which must not depend on b
    check('coupled: joint belief is not a product; factor A alone is still mess3(0.05, 0.85)',
          dev > 0.05 and (TA - A['T'][:, :, None, :]).abs().max() < 1e-12, f'max |joint - product| {dev:.3f}')
    k = P.get('coins3')
    xk = P.sample(k, 500, 10, g=g)
    ek = P.beliefs(k, xk)[0]
    n1 = torch.cat([torch.zeros(500, 1), xk[:, :-1].cumsum(1).float()], 1).double()
    t = torch.arange(10).double()
    pr = torch.tensor([0.2, 0.5, 0.8], dtype=P.F64)
    post = pr ** n1[..., None] * (1 - pr) ** (t[None, :, None] - n1[..., None])
    check('coins3: belief = posterior over the coin from the counts', (ek - post / post.sum(-1, keepdim=True)).abs().max() < 1e-12)


def contexts():
    """processes.context_loss: the one-token-context predictor against brute-force P(x_1 | x_0); a context longer than
    the sequence gives the Bayes loss; the excess shrinks with the context. The excesses for 2 and 3 tokens of context
    are printed (hmm_summary.py's learned gate uses the 2-token one)."""
    g = torch.Generator().manual_seed(2)
    for name in P.NAMES:
        p = P.get(name)
        x = P.sample(p, 8192, 16, g=g)
        bl = P.bayes_loss(p, x)
        full = (P.context_loss(p, x, 16) - bl).abs().max().item()
        e1, e2, e3 = ((P.context_loss(p, x, m) - bl).mean().item() for m in (1, 2, 3))
        note = f'|16-token context - Bayes| {full:.1e}; excess with 1, 2, 3 tokens of context {e1:.4f}, {e2:.4f}, {e3:.4f}'
        ok = full < 1e-12 and -1e-12 <= e3 <= e2 + 1e-12 <= e1 + 2e-12
        if p['S'] < 9:
            ref = brute(p, 2)
            V = p['V']
            pairs = torch.tensor([[a, b] for a in range(V) for b in range(V) if ref[(a,)].sum() > 0])
            q = P.beliefs(p, pairs)[1][:, 1]  # filter the one-token window from pi
            exact = torch.stack([ref[(a, b)].sum() / ref[(a,)].sum() for a, b in pairs.tolist()])
            err = (q.gather(1, pairs[:, 1:])[:, 0] - exact).abs().max().item()
            ok, note = ok and err < 1e-12, note + f'; |one-token predictor - enum| {err:.1e}'
        check(f'{name}: optimal short-context predictors', ok, note)


if __name__ == '__main__':
    for fn in (kernels, filtering, sampler, products, contexts):
        try:
            fn()
        except Exception as e:  # a crashing check is a failed check
            check(fn.__name__, False, f'crashed: {e!r}')
    n_ok = sum(r['ok'] for r in results)
    print(f'{n_ok}/{len(results)} checks passed')
    if len(sys.argv) > 1:
        os.makedirs(os.path.dirname(os.path.abspath(sys.argv[1])), exist_ok=True)  # $EXP_DIR/results does not exist on a fresh worker
        json.dump(results, open(sys.argv[1], 'w'), indent=1)
    sys.exit(0 if n_ok == len(results) else 1)
