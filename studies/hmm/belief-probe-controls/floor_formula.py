"""Closed form of the factor-code floor of part 1, checked against the measured floor (run after the experiment, CPU).

    PYTHONPATH=. python studies/hmm/belief-probe-controls/floor_formula.py

On `factored` the two factor beliefs b^A_t, b^B_t are independent at every position. Write b^A = mu_A + d_A and
b^B = mu_B + d_B with E d = 0. Then
    b^A (x) b^B = mu_A (x) mu_B + d_A (x) mu_B + mu_A (x) d_B + d_A (x) d_B .
The first three terms are affine in the state [b^A, b^B]; the last has mean zero and is uncorrelated with every affine
function of the state (by independence), so it is exactly the residual of the best affine probe. With v_A = E|d_A|^2,
v_B = E|d_B|^2, the cross terms of the total variance vanish for the same reason, and at one position
    SSE_t = v_A v_B ,    SST_t = |mu_B|^2 v_A + |mu_A|^2 v_B + v_A v_B ,
    floor = 1 - sum_t SSE_t / sum_t SST_t     (pooled like the experiment's R2).
This script evaluates v_A, v_B, mu on an independent sample (seed below, not the analysis sample) and prints the formula
next to the floor that factor_product.py measured with a held-out-prefix affine probe (results/factor_product.json)."""
import json
from pathlib import Path

import torch

from studies.hmm import processes as P

HERE = Path(__file__).resolve().parent
N, T, SEED = 60000, 16, 12345
GROUPS = {'transformer': ('e', range(4, 16)), 'chain': ('e', range(3, 16)), 'tree': ('h1', range(4, 9)), 'tree8': ('h1', range(3, 9))}


def main():
    proc = P.get('factored')
    eta, _ = P.beliefs(proc, P.sample(proc, N, T, g=torch.Generator().manual_seed(SEED)))
    A, B = (m.double() for m in P.marginals(proc, eta))
    runs = json.load(open(HERE / 'results' / 'factor_product.json'))['runs']
    out = {}
    for arm, (grp, ts) in GROUPS.items():
        q = runs[f'factored_{arm}_s0']
        assert list(ts) == q['positions'][grp], (arm, q['positions'])  # the positions the measured floor was pooled over
        sse = sst = 0.0
        for t in ts:
            (ma, va), (mb, vb) = ((m[:, t].mean(0), ((m[:, t] - m[:, t].mean(0)) ** 2).sum(1).mean()) for m in (A, B))
            sse, sst = sse + va * vb, sst + (mb @ mb) * va + (ma @ ma) * vb + va * vb
        out[arm] = dict(positions=list(ts), formula=float(1 - sse / sst), measured=q['floor_aff'])
        print(f'{arm:12s} t = {ts[0]:2d}..{ts[-1]}: formula {out[arm]["formula"]:.4f}   measured floor {q["floor_aff"]:.4f}')
    t = 15
    out['at_t15'] = dict(v_A=float(((A[:, t] - A[:, t].mean(0)) ** 2).sum(1).mean()), v_B=float(((B[:, t] - B[:, t].mean(0)) ** 2).sum(1).mean()),
                         mu_A=A[:, t].mean(0).tolist(), mu_B=B[:, t].mean(0).tolist())
    print('t = 15:', {k: (round(v, 4) if isinstance(v, float) else [round(u, 3) for u in v]) for k, v in out['at_t15'].items()})
    json.dump(dict(n=N, seed=SEED, arms=out), open(HERE / 'results' / 'floor_formula.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
