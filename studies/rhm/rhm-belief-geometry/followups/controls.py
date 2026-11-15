"""Follow-up to rhm-belief-geometry (not part of its pre-registered criteria): two controls that the reviews of the
later experiments showed are needed to read a high probe R2.

    PYTHONPATH=. python studies/rhm/rhm-belief-geometry/followups/controls.py RUN_DIR OUT_JSON [N_ANA]

A position has few distinct beliefs compared with the probe's 129 parameters per output, so a probe that linearly
identifies the belief state fits any labelling of the states. For the last-layer probe (L4) at the final checkpoint,
on the positions eligible under the experiment's rule (>= 10 (d + 1) distinct fit prefixes), this script reports:
  belief     R2 on held-out prefixes (the experiment's H1 number, recomputed on this smaller sample);
  scrambled  the same probe fit to an independent N(0, 1) 8-vector per distinct belief. belief - scrambled
             ("selectivity") is what the belief's geometry adds over identifying the state;
  belief split  the distinct beliefs are assigned at random to fit or test (positions with >= 20 distinct beliefs),
             so the probe must extrapolate to belief vectors it never saw. A code affine in the belief allows that;
             a code that only identifies states does not. The scrambled target on this split is saved too
             (split_scrambled); a probe cannot predict a random vector for a state it never saw, and in the results it
             scores below zero.
             Reported with its ceiling (the test beliefs projected on the affine span of the fit beliefs) and with
             the same probe on the randomly initialized network (checkpoint 0).
Caveat (from the TTN review): the belief split fits on few distinct targets, so it is a noisy measure; read it with
the ceiling and the random-init number, not alone.

Sample: N_ANA sequences (default 20,000, a third of the experiment's 60,000, to fit the coordinator's CPU and
memory), so each position is scored on fewer held-out prefixes than in the experiment. In the five seed-0 runs
analysed, the eligible positions and the distinct-belief counts came out the same as in the experiment."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as AZ  # noqa: E402
import beliefs as BL  # noqa: E402
import model as M  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

MIN_BELIEFS = 20


@torch.no_grad()
def last_layer(model, x, v, chunk=2000):
    """(B, T, d) float64: the last block's output, before the final LayerNorm."""
    return torch.cat([model(M.inputs(x[i:i + chunk], v), resid=True)[1][-1].double() for i in range(0, len(x), chunk)])


def main():
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    res = json.load(open(run_dir / 'result.json'))
    cfg = res['cfg']
    g, v, T, d = cfg['grammar'], cfg['V'], cfg['T'], cfg['d']
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(N, rules, seed=AZ.ANA_SEED + g['seed'])[g['L']]
    bel = BL.beliefs(x, rules, g)
    acts = {}
    for ck in (res['ckpts'][0], res['ckpts'][-1]):
        model = M.Transformer(v, T, d=d, layers=cfg['layers'], heads=cfg['heads'])
        model.load_state_dict(torch.load(run_dir / f'ckpt_{ck}.pt', map_location='cpu')['state'])
        acts[ck] = last_layer(model.eval(), x, v)
    first, last = res['ckpts'][0], res['ckpts'][-1]
    rows, tot = [], {}
    add = lambda key, sc: tot.setdefault(key, []).append((sc['sse'], sc['sst']))
    for t in sorted(bel):
        if t < 1:
            continue
        b = bel[t][0]
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        side = torch.rand(int(pid.max()) + 1, generator=torch.Generator().manual_seed(100 + t)) < 0.5
        if int(side.sum()) < AZ.ELIG_FACTOR * (d + 1):
            continue
        fi, ti = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
        uniq, uid = torch.unique(b.round(decimals=9), dim=0, return_inverse=True)
        A, A0 = acts[last][:, t], acts[first][:, t]
        row = dict(t=t, D=b.shape[1], n_beliefs=len(uniq), n_fit_prefixes=int(side.sum()))
        sc = AZ.score(b[ti], AZ.fit_predict(A[fi], b[fi], A[ti]), b[fi].mean(0))
        scr = torch.randn(len(uniq), 8, generator=torch.Generator().manual_seed(500 + t), dtype=torch.float64)[uid]
        ss = AZ.score(scr[ti], AZ.fit_predict(A[fi], scr[fi], A[ti]), scr[fi].mean(0))
        row.update(belief=sc['R2'], scrambled=ss['R2'])
        add('belief', sc)
        add('scrambled', ss)
        if len(uniq) >= MIN_BELIEFS:
            bside = torch.rand(len(uniq), generator=torch.Generator().manual_seed(300 + t)) < 0.5
            f2, t2 = torch.nonzero(bside[uid])[:, 0], torch.nonzero(~bside[uid])[:, 0]
            mu = b[f2].mean(0)
            U = torch.unique(b[f2].round(decimals=9), dim=0) - mu
            _, sv, Vh = torch.linalg.svd(U, full_matrices=False)
            B = Vh[sv > 1e-9 * sv[0]].T
            parts = dict(split=AZ.score(b[t2], AZ.fit_predict(A[f2], b[f2], A[t2]), mu),
                         split_init=AZ.score(b[t2], AZ.fit_predict(A0[f2], b[f2], A0[t2]), mu),
                         split_ceiling=AZ.score(b[t2], mu + (b[t2] - mu) @ B @ B.T, mu),
                         split_scrambled=AZ.score(scr[t2], AZ.fit_predict(A[f2], scr[f2], A[t2]), scr[f2].mean(0)))
            for k, sc2 in parts.items():
                row[k] = sc2['R2']
                add(k, sc2)
        rows.append(row)
    pooled = {k: 1 - sum(a for a, _ in v_) / sum(b_ for _, b_ in v_) for k, v_ in tot.items()}
    pooled['n_positions'] = len(rows)
    pooled['n_split_positions'] = sum('split' in r for r in rows)
    json.dump(dict(run=run_dir.name, n_ana=N, final_excess=res['hist'][-1]['excess'], pooled=pooled, positions=rows),
              open(out, 'w'), indent=1)
    f = lambda k: f'{pooled[k]:.3f}' if k in pooled else '  -  '
    print(f"{run_dir.name}: {len(rows)} positions | belief {f('belief')} scrambled {f('scrambled')} | belief split "
          f"{f('split')} (init {f('split_init')}, ceiling {f('split_ceiling')}, scrambled {f('split_scrambled')}) on "
          f"{pooled['n_split_positions']} positions", flush=True)


if __name__ == '__main__':
    main()
