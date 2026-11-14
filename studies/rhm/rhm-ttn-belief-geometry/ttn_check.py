"""Checks of the TTN probe code (no training):
  prefixes()      row (b, t) is x_b with positions >= t masked and target x_b[t], against an explicit loop;
  done_nodes()    the spans tile [0, t) exactly, largest first;
  structure       on a random tree: h_R is constant across inputs for t <= T/2, h_L does not change when the right half's
                  tokens change, and h_R does not change when the left half's tokens change (what the hypotheses assume);
  probes          on synthetic targets from those bonds: an affine target is recovered by the affine probe; a bilinear
                  target sum_ab M_kab hL_a hR_b is recovered by the bilinear probe (R2 ~ 1) and not by the affine one;
  controls        the additive design fits a target additive in (left symbol, right state) exactly and not a pure
                  interaction; the token-product features have the stated width."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE))
import analyze as AZ  # noqa: E402
import ttn_analyze as TA  # noqa: E402
import ttn_run as TR  # noqa: E402

ok = True


def check(name, cond, note=''):
    global ok
    ok &= bool(cond)
    print(f'{"PASS" if cond else "FAIL"}  {name}  {note}')


v, T, r = 5, 16, 6
g = torch.Generator().manual_seed(0)
x = torch.randint(0, v, (4000, T), generator=g)

xm, y = TR.prefixes(x[:3], v)
ref = [(torch.cat([x[b, :t], torch.full((T - t,), v)]), x[b, t]) for b in range(3) for t in range(T)]
check('prefixes: masks and targets', all((xm[i] == a).all() and y[i] == c for i, (a, c) in enumerate(ref)))

tiles = True
for t in range(1, T):
    spans = [(k << l, (k + 1) << l) for l, k in TA.done_nodes(t, 4)]
    tiles &= spans[0][0] == 0 and spans[-1][1] == t and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    tiles &= all(a[1] - a[0] > b[1] - b[0] for a, b in zip(spans, spans[1:]))
check('done_nodes: spans tile [0, t), largest first', tiles)

model = TR.build(dict(grammar=dict(v=v, seed=3), r=r, init_eps=1.0, T=T))  # eps = 1: far from the pass-through init
top = lambda xx, t: TA.bond_sets(model, xx, t, v, bilinear=True)
s8 = top(x, 8)['top']
check('h_R constant for t <= T/2', s8[:, r:].std(0).max() < 1e-6 and s8[:, :r].std(0).min() > 1e-4)
x2 = x.clone()
x2[:, 8:] = torch.randint(0, v, (4000, 8), generator=g)
x3 = x.clone()
x3[:, :8] = torch.randint(0, v, (4000, 8), generator=g)
a, b2, b3 = top(x, 13)['top'], top(x2, 13)['top'], top(x3, 13)['top']
check('h_L blind to the right half, h_R blind to the left half',
      (a[:, :r] - b2[:, :r]).abs().max() < 1e-6 and (a[:, r:] - b3[:, r:]).abs().max() < 1e-6
      and (a[:, r:] - b2[:, r:]).abs().max() > 1e-4)

S = top(x, 13)
hl, hr = S['top'][:, :r].double(), S['top'][:, r:].double()
f, s = slice(0, 2000), slice(2000, None)
aff = S['top'].double() @ torch.randn(2 * r, 7, generator=g, dtype=torch.float64)
bil = torch.einsum('kab,Ba,Bb->Bk', torch.randn(7, r, r, generator=g, dtype=torch.float64), hl, hr)
R2 = lambda feats, target: AZ.score(target[s], AZ.fit_predict(feats[f].double(), target[f], feats[s].double()), target[f].mean(0))['R2']
r_aa, r_ab, r_bb = R2(S['top'], aff), R2(S['top'], bil), R2(S['top_bil'], bil)
check('affine target: affine probe recovers it', r_aa > 0.999, f'R2 {r_aa:.5f}')
check('bilinear target: bilinear probe recovers it, affine probe does not', r_bb > 0.999 and r_ab < 0.9,
      f'R2 bilinear {r_bb:.5f}, affine {r_ab:.3f}')
# additive ceiling: a target f(left symbol) + g(right state) is fit exactly; a pure interaction (XOR-like) is not
trees = {1: torch.randint(0, 3, (4000, 2), generator=g), 4: None,
         2: torch.randint(0, 3, (4000, 4), generator=g), 3: torch.randint(0, 3, (4000, 8), generator=g)}
A = TA.additive_design(trees, 14, 4)  # right-half observed nodes for t = 14: span 8:12 (grammar level 2, node 2), 12:14 (level 3, node 6)
zl, s1, s2 = trees[1][:, 0], trees[2][:, 2], trees[3][:, 6]
add = torch.stack([zl.double() ** 2 + (3 * s1 + s2).double().sin(), s2.double() - zl], 1)
inter = ((zl + s1) % 3 == 0).double()[:, None]
check('additive design: additive target fit exactly, interaction not', R2(A, add) > 0.999 and R2(A, inter) < 0.5,
      f'R2 additive {R2(A, add):.5f}, interaction {R2(A, inter):.3f}, {A.shape[1]} columns')
tp = TA.token_product(x, 13, v)
check('token_product: width H v + (t - H) v + their product', tp.shape[1] == 8 * v + 5 * v + 8 * v * 5 * v and tp.sum(1).eq(8 + 5 + 40).all())
print('all pass' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
