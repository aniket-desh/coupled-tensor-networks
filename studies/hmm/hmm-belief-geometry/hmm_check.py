"""Checks of the HMM probe pipeline (no training):  PYTHONPATH=. python studies/hmm/hmm-belief-geometry/hmm_check.py
  causality     logits[:, t] of each model depends only on x_<t.
  exact chain   a ChainTN built to run the exact unnormalized filter u_{t+1} = u_t T[x_t] on mess3 (r = S,
                W[a, b, c] = T[c, b, a], token embedding = one-hot, h_0 = pi): its state at position t is u_t, so the
                projective probe must recover eta_t exactly at every position (R2 = 1) and the affine probe must not.
                This also checks the alignment of states[:, t] with eta[:, t].
  exact tree    a TreeTN built to multiply transfer matrices (r = S^2: a leaf embeds vec(T[x]), a masked leaf vec(I),
                a merge is the matrix product), on the masked prefix input: for t <= T/2 the projective probe on the top
                bonds recovers eta_t exactly; for t > T/2 it needs the product features (top_bil), and the top bonds
                alone do not suffice.
  ceilings      exact_additive against a least-squares fit of one-hot(left half) + one-hot(right prefix) on a large
                sample; the next-token ceiling is 1 on mess3 (the belief is a linear image of P(next)) and below 1
                somewhere on rrxor."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hmm_analyze as HA  # noqa: E402
import hmm_run as HR  # noqa: E402
from ctn.tree import ChainTN, TreeTN  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.hmm import processes as P  # noqa: E402

ok = True


def check(name, cond, note=''):
    global ok
    ok &= bool(cond)
    print(f'{"PASS" if cond else "FAIL"}  {name}  {note}')


def r2(A, b, proj):
    f, s = slice(0, len(b) // 2), slice(len(b) // 2, None)
    pred = HA.projective_fit_predict(A[f], b[f], A[s])[0] if proj else BC.fit_predict(A[f], b[f], A[s])
    return BC.score(b[s], pred, b[f].mean(0))['R2']


g = torch.Generator().manual_seed(0)
proc = P.get('mess3')
V, S, T = proc['V'], proc['S'], 8
x = P.sample(proc, 4000, T, g=g)
eta, pnext = P.beliefs(proc, x)

x2 = x.clone()
x2[:, 5:] = (x2[:, 5:] + 1) % V
for kind in HR.MODELS:
    cfg = dict(model=kind, V=V, T=T, seed=0, d=32, layers=2, heads=2, r=6, init_eps=1.0)
    m = HR.build(cfg).eval()
    with torch.no_grad():
        a, b = HR.logits(m, kind, x[:64], V), HR.logits(m, kind, x2[:64], V)
    check(f'causality, {kind}', (a[:, :6] - b[:, :6]).abs().max() < 1e-5 and (a[:, 6:] - b[:, 6:]).abs().max() > 1e-5)

chain = ChainTN(T, V, S, V, seed=0)
with torch.no_grad():
    chain.h0.copy_(proc['pi'].float())
    chain.A.copy_(torch.eye(V))
    chain.W.copy_(proc['T'].permute(2, 1, 0).float())  # W[a, b, c] = T[x = c, s = b, s' = a]
    st = BC.chain_states(chain, x)[0].double()
worst_p = min(r2(st[:, t], eta[:, t], True) for t in range(1, T))
worst_a = min(r2(st[:, t], eta[:, t], False) for t in range(1, T))
u = st / st.sum(-1, keepdim=True)
check('exact chain: state = unnormalized filter; projective probe exact, affine probe not',
      (u - eta).abs().max() < 1e-5 and worst_p > 0.9999 and worst_a < 0.999,
      f'min over positions: projective R2 {worst_p:.6f}, affine R2 {worst_a:.4f}')

r = S * S
tree = TreeTN(T, V + 1, r, V, tie='none', seed=0)
with torch.no_grad():
    leaf = torch.cat([proc['T'].reshape(V, r), torch.eye(S, dtype=P.F64).reshape(1, r)]).float()  # token x -> vec(T[x]); mask -> vec(I)
    tree.A.copy_(leaf.expand(T, -1, -1))
    W = torch.zeros(S, S, S, S, S, S)  # (i, k | i, j | j, k)
    for i in range(S):
        for j in range(S):
            for k in range(S):
                W[i, k, i, j, j, k] = 1.0
    for w in tree.W:
        w.copy_(W.reshape(r, r, r).expand_as(w))
H = T // 2
res = {}
for t in (2, H, H + 1, T - 1):
    s = HA.TA.bond_sets(tree, x, t, V, bilinear=True)
    res[t] = {k: r2(s[k].double(), eta[:, t], True) for k in ('top', 'top_bil')}
check('exact tree: projective on the top bonds exact up to T/2; after it only with the product features',
      all(res[t]['top'] > 0.9999 for t in (2, H)) and all(res[t]['top_bil'] > 0.9999 for t in res)
      and all(res[t]['top'] < 0.999 for t in (H + 1, T - 1)),
      ' '.join(f"t={t}: top {v['top']:.4f} top_bil {v['top_bil']:.4f}" for t, v in res.items()))

xs = P.sample(proc, 300_000, 6, g=g)  # T = 6, H = 3, t = 5: 27 left halves, 9 right prefixes
b5 = P.beliefs(proc, xs)[0][:, 5]
ids = lambda z: torch.unique(z, dim=0, return_inverse=True)[1]
D = torch.cat([torch.nn.functional.one_hot(ids(xs[:, :3])).double(), torch.nn.functional.one_hot(ids(xs[:, 3:5])).double()], 1)
fit = BC.fit_predict(D, b5, D)
samp = 1 - ((b5 - fit) ** 2).sum().item() / ((b5 - b5.mean(0)) ** 2).sum().item()
mse, var = HA.exact_additive(proc, 5, 3)
check('exact_additive vs least squares on one-hot(left) + one-hot(right), 300,000 samples', abs((1 - mse / var) - samp) < 5e-3,
      f'exact R2 {1 - mse / var:.4f}, sampled {samp:.4f}')

f, s = slice(0, 2000), slice(2000, None)
# the ceiling is E[belief | next-token distribution], estimated on all rows and scored on the test rows
ceil = lambda pr, t: BC.score(pr[0][s, t], HA.nexttok_ceiling(pr[1][:, t], pr[0][:, t], pr[1][s, t], pr[0][s, t]), pr[0][f, t].mean(0))['R2']
rr = P.get('rrxor')
xr = P.sample(rr, 4000, 16, g=g)
br = P.beliefs(rr, xr)
c_m = min(ceil((eta, pnext), t) for t in range(1, T))
c_r = [ceil(br, t) for t in range(1, 16)]
cc = P.get('coins3')
bc = P.beliefs(cc, P.sample(cc, 4000, 16, g=g))
c_c = min(ceil(bc, t) for t in range(1, 16))
check('next-token ceiling: 1 on mess3 and coins3 (belief a function of the prediction), below 0.9 somewhere on rrxor',
      c_m > 0.9999 and c_c > 0.9999 and min(c_r) < 0.9,
      f'mess3 min {c_m:.5f}; coins3 min {c_c:.5f}; rrxor min {min(c_r):.3f} max {max(c_r):.3f}')
print('all pass' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
