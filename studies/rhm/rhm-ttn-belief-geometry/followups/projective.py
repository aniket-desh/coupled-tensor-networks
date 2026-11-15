"""Follow-up to rhm-ttn-belief-geometry (not pre-registered): does a probe "up to normalization" recover what the
affine probe on the top bonds misses?

    PYTHONPATH=. python studies/rhm/rhm-ttn-belief-geometry/followups/projective.py RUN_DIR OUT_JSON [N_ANA]

A tree's merge has the form of the grammar's upward message, so a bond may hold an unnormalized message whose scale
varies from prefix to prefix; the belief is then a ratio of linear functions of the bond, which an affine probe cannot
express and the projective probe can (studies/belief_common.py: bhat = q / sum(q), q = W a + c). At the final
checkpoint, on held-out prefixes, at every position that is eligible for the top-bond probe (>= 10 (2r + 1) distinct
fit prefixes) and whose belief has at most MAX_D entries (the projective fit is nonlinear, so large beliefs are left
out), this script fits to the two top bonds [h_L, h_R]:
  affine      the experiment's probe;
  projective  the same features, belief up to normalization.
Both are also fit on the identity-init tree (checkpoint 0). If the bond's dimension, not its normalization, limits the
affine probe, the projective probe scores about the same.

Sample: N_ANA sequences (default 20,000, a third of the experiment's), so the numbers are not the experiment's own."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent))
import analyze as AZ  # noqa: E402
import beliefs as BL  # noqa: E402
import ttn_analyze as TA  # noqa: E402
import ttn_run as TR  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

MAX_D = 64


def main():
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    g, r, T, v = cfg['grammar'], cfg['r'], cfg['T'], cfg['V']
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(N, rules, seed=AZ.ANA_SEED + g['seed'])[g['L']]
    bel = BL.beliefs(x, rules, g)
    models = {}
    for ck in (res['ckpts'][0], res['ckpts'][-1]):
        m = TR.build(cfg)
        m.load_state_dict(torch.load(run_dir / f'ckpt_{ck}.pt', map_location='cpu')['state'])
        models[ck] = m.eval()
    first, last = res['ckpts'][0], res['ckpts'][-1]
    rows, tot = [], {}
    add = lambda key, half, sc: tot.setdefault((key, half), []).append((sc['sse'], sc['sst']))
    for t in sorted(bel):
        b = bel[t][0]
        if t < 1 or b.shape[1] > MAX_D:
            continue
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        side = torch.rand(int(pid.max()) + 1, generator=torch.Generator().manual_seed(100 + t)) < 0.5
        if int(side.sum()) < AZ.ELIG_FACTOR * (2 * r + 1):
            continue
        fi, ti = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
        mu = b[fi].mean(0)
        half = 'first' if t <= T // 2 else 'second'
        row = dict(t=t, half=half, D=b.shape[1], n_fit_prefixes=int(side.sum()),
                   n_beliefs=len(torch.unique(b.round(decimals=9), dim=0)))
        for name, ck in (('', last), ('init_', first)):
            A = TA.bond_sets(models[ck], x, t, v, bilinear=False)['top'].double()
            sa = BC.score(b[ti], BC.fit_predict(A[fi], b[fi], A[ti]), mu)
            ph, ratio = BC.projective_fit_predict(A[fi], b[fi], A[ti])
            sp = BC.score(b[ti], ph, mu)
            row.update({name + 'affine': sa['R2'], name + 'projective': sp['R2'], name + 'fit_loss_ratio': ratio})
            add(name + 'affine', half, sa)
            add(name + 'projective', half, sp)
        rows.append(row)
    pooled = {f'{k}@{h}': 1 - sum(a for a, _ in vals) / sum(c for _, c in vals) for (k, h), vals in tot.items()}
    json.dump(dict(run=run_dir.name, n_ana=N, final_excess=res['hist'][-1]['excess'], pooled=pooled, positions=rows),
              open(out, 'w'), indent=1)
    f = lambda k: f'{pooled[k]:.3f}' if k in pooled else '  -  '
    n = lambda h: sum(q['half'] == h for q in rows)
    print(f"{run_dir.name}: first half ({n('first')} pos) affine {f('affine@first')} projective {f('projective@first')} "
          f"[init {f('init_affine@first')} / {f('init_projective@first')}] | second half ({n('second')} pos) affine "
          f"{f('affine@second')} projective {f('projective@second')} [init {f('init_affine@second')} / "
          f"{f('init_projective@second')}]", flush=True)


if __name__ == '__main__':
    main()
