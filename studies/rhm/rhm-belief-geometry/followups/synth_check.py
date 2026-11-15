"""Read-only rerun of the synthetic belief-split check described in followups/README.md."""
import sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, 'studies/rhm/rhm-belief-geometry')
import analyze as AZ
import beliefs as BL
from studies.rhm import grammar as G

g = dict(v=8, n=8, m=4, s=2, L=3, seed=0)
rules = G.sample_rules(8, 8, 4, 2, 3, seed=0)
x = G.sample_trees(8000, rules, seed=1)[3]
bel = BL.beliefs(x, rules, g)
for t in (5, 7):
    b = bel[t][0]
    uniq, uid = torch.unique(b.round(decimals=9), dim=0, return_inverse=True)
    bside = torch.rand(len(uniq), generator=torch.Generator().manual_seed(300 + t)) < 0.5
    f2, t2 = torch.nonzero(bside[uid])[:, 0], torch.nonzero(~bside[uid])[:, 0]
    mu = b[f2].mean(0)
    U = torch.unique(b[f2].round(decimals=9), dim=0) - mu
    _, sv, Vh = torch.linalg.svd(U, full_matrices=False)
    B = Vh[sv > 1e-9 * sv[0]].T
    ceil = AZ.score(b[t2], mu + (b[t2] - mu) @ B @ B.T, mu)['R2']
    print(f't={t} D={b.shape[1]} n_beliefs={len(uniq)} fit beliefs={int(bside.sum())} test beliefs={int((~bside).sum())} '
          f'fit rows={len(f2)} test rows={len(t2)} ceiling={ceil:.4f}')
    for seed in range(6):
        gen = torch.Generator().manual_seed(seed)
        P = torch.randn(b.shape[1], 128, generator=gen, dtype=torch.float64)
        code = torch.randn(len(uniq), 128, generator=gen, dtype=torch.float64)
        for noise in (0.0, 1e-3):
            eps = noise * torch.randn(len(b), 128, generator=gen, dtype=torch.float64)
            a_lin, a_code = b @ P + eps, code[uid] + eps
            r_lin = AZ.score(b[t2], AZ.fit_predict(a_lin[f2], b[f2], a_lin[t2]), mu)['R2']
            r_code = AZ.score(b[t2], AZ.fit_predict(a_code[f2], b[f2], a_code[t2]), mu)['R2']
            print(f'   seed {seed} noise {noise:g}: linear image {r_lin:.3f}   random code per belief {r_code:.3f}')
