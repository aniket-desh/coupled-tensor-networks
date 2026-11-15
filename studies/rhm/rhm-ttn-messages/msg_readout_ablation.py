"""Exploratory, written after the run: how much of M4's shortfall is in the read-out table q_bc?

    python studies/rhm/rhm-ttn-messages/msg_readout_ablation.py RESULTS_DIR

M4 decodes the merge of two mixture bonds. The merge is bilinear and the probe's read-out affine, so for normalized
child messages the read-out of the merged mixture is exactly
    q(lamL, lamR) = sum_bc lamL(b) lamR(c) q_bc,        q_bc = read-out of the merged class means of the pair (b, c),
and the exact update is the same sum with the production tensor T_bc in place of q_bc. The decoded message equals the
exact one for all pairs iff q_bc = kappa T_bc (kappa > 0): invalid pairs read out as zero, valid pairs with equal mass
on the right parent. msg_analyze.py saved q_bc for node 0 of each tree level (arrays_ckpt*.npz, prod_l<l>_q), so the
decoded update can be recomputed from the table alone, and the table can be edited:
    as saved        q_bc unchanged (must reproduce the saved M4 on Dirichlet pairs, up to the sample of pairs)
    invalid zeroed  q_bc = 0 on invalid pairs
    equal mass      valid q_bc rescaled to the median nonnegative mass sum_a max(q_bc(a), 0) of the valid pairs
    both            both edits
Pairs are Dirichlet(1), as in M4's `dir` subset (the empirical message pairs were not saved). R2 is against the exact
normalized update, scored on the second half of the pairs against the mean of the first half, as in msg_analyze.py, and
pooled over node 0 of each level >= 2 by summed SSE / SST. Final checkpoint. Writes RESULTS_DIR/readout_ablation.json
and prints per-cell medians over the three seeds with the range."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

N_PAIRS = 8000


def decode(q):
    """The decoder of msg_analyze.fit_projective: the normalized nonnegative part of q (uniform if nothing is positive)."""
    q = np.clip(q, 0, None)
    tot = q.sum(1, keepdims=True)
    return np.where(tot > 0, q / np.clip(tot, 1e-300, None), 1.0 / q.shape[1])


def tables(q, valid):
    mass = np.clip(q, 0, None).sum(-1)                     # (v, v) nonnegative read-out mass of each pair
    zero = q.copy()
    zero[~valid] = 0
    equal = q.copy()
    equal[valid] *= (np.median(mass[valid]) / np.clip(mass[valid], 1e-12, None))[:, None]
    both = equal.copy()
    both[~valid] = 0
    return {'as_saved': q, 'invalid_zeroed': zero, 'equal_mass': equal, 'both': both}


def main():
    res = Path(sys.argv[1])
    out, by = {}, defaultdict(list)
    for d in sorted(p for p in res.iterdir() if (p / 'analysis.json').exists()):
        a = json.load(open(d / 'analysis.json'))
        ck = max(int(c) for c in a['metrics'])
        z = np.load(d / f'arrays_ckpt{ck}.npz')
        rng = np.random.default_rng(0)
        sse, sst, saved = defaultdict(float), 0.0, {}
        for l in sorted({int(k.split('_')[1][1:]) for k in z.files if k.startswith('prod_') and k.endswith('_q')}):
            if l < 2:
                continue
            q, parent = z[f'prod_l{l}_q'].astype(np.float64), z[f'prod_l{l}_true']
            v, valid = q.shape[0], parent >= 0
            T = np.zeros((v, v, v))
            b, c = np.nonzero(valid)
            T[b, c, parent[b, c]] = 1.0                    # 1/m cancels in the normalized update
            lL, lR = rng.dirichlet(np.ones(v), N_PAIRS), rng.dirichlet(np.ones(v), N_PAIRS)
            w = lL[:, :, None] * lR[:, None, :]            # (n, v, v) weights lamL(b) lamR(c), summing to 1
            exact = np.einsum('nbc,bca->na', w, T)
            exact /= exact.sum(1, keepdims=True)
            fit, test = slice(0, N_PAIRS // 2), slice(N_PAIRS // 2, None)
            sst += ((exact[test] - exact[fit].mean(0)) ** 2).sum()
            for name, qt in tables(q, valid).items():
                sse[name] += ((exact[test] - decode(np.einsum('nbc,bca->na', w, qt))[test]) ** 2).sum()
            saved[l] = a['metrics'][str(ck)]['nodes'][f'{l},0']['m4']['dir_proj']['R2']
        row = {name: 1 - s / sst for name, s in sse.items()}
        row['saved_dir_R2_per_level'] = saved
        out[d.name] = row
        by[d.name.rsplit('_s', 1)[0]].append(row)
    json.dump(out, open(res / 'readout_ablation.json', 'w'), indent=1)
    names = ['as_saved', 'invalid_zeroed', 'equal_mass', 'both']
    mr = lambda rr, k: f'{median(x[k] for x in rr):.2f} [{min(x[k] for x in rr):.2f}, {max(x[k] for x in rr):.2f}]'
    print('| grammar | r | ' + ' | '.join(n.replace('_', ' ') for n in names) + ' |')
    print('| --- | --- | ' + ' | '.join('---' for _ in names) + ' |')
    for cell in sorted(by, key=lambda c: (-int(c.rsplit('_r', 1)[1]), c)):
        g, r = cell.rsplit('_r', 1)
        print(f'| `{g}` | {r} | ' + ' | '.join(mr(by[cell], n) for n in names) + ' |')
    print('\nper run and level, saved M4 Dirichlet R2 at node 0 (for comparison with `as saved`, which pools levels >= 2):')
    for run, row in out.items():
        print(run, {l: round(x, 2) for l, x in row['saved_dir_R2_per_level'].items()}, 'as saved, pooled:', round(row['as_saved'], 2))


if __name__ == '__main__':
    main()
