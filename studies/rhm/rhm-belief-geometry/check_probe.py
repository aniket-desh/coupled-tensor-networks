"""Probe machinery on synthetic activations with known answers (no training):
  a = b_t P + noise, P random (D_t, 128): concat-style probe R2 ~ 1;  shuffled labels R2 ~ 0 (about -p/n from overfitting);
  the next-token ceiling never exceeds 1 and equals 1 where b_t is a function of P(x_t | x_<t)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import analyze as A  # noqa: E402
import beliefs as BL  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

g = dict(v=8, n=8, m=4, s=2, L=3, seed=0)
rules = G.sample_rules(8, 8, 4, 2, 3, seed=0)
x = G.sample_trees(6000, rules, seed=1)[3]
bel = BL.beliefs(x, rules, g)
ok = True
for t in range(1, 8):
    b, p = bel[t]
    gen = torch.Generator().manual_seed(t)
    a = b @ torch.randn(b.shape[1], 128, generator=gen, dtype=torch.float64) + 1e-3 * torch.randn(len(b), 128, generator=gen, dtype=torch.float64)
    f, s = slice(0, 3000), slice(3000, None)
    mu = b[f].mean(0)
    r = A.score(b[s], A.fit_predict(a[f], b[f], a[s]), mu)['R2']
    rs = A.score(b[s], A.fit_predict(a[f], b[f][torch.randperm(3000, generator=gen)], a[s]), mu)['R2']
    rn = A.score(b[s], A.nexttok_ceiling(p[f], b[f], p[s]), mu)['R2']
    # sum-to-one is preserved by an affine fit to simplex-valued targets
    s1 = (A.fit_predict(a[f], b[f], a[s]).sum(-1) - 1).abs().max().item()
    good = r > 0.999 and abs(rs) < 0.1 and rn <= 1 + 1e-12 and s1 < 1e-6
    ok &= good
    print(f'{"PASS" if good else "FAIL"} t={t} D={b.shape[1]}: R2 linear={r:.5f} shuffled={rs:.4f} nexttok ceiling={rn:.4f} |sum-1|={s1:.1e}')
print('all pass' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
