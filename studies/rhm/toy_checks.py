"""Checks for studies/rhm/toy.py: the two hand-built grammars, their exact targets, the rank bounds, the hand-built
exact trees and the optimizer.

    python -m studies.rhm.toy_checks [out.json]
"""
import json
import math
import sys

import numpy as np
import torch

from studies.rhm import grammar as G
from studies.rhm import toy as Y

torch.set_num_threads(1)
results = {}


def check(name, ok, detail=''):
    results[name] = dict(ok=bool(ok), detail=str(detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def main():
    K3 = torch.linalg.matrix_power((torch.ones(3, 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64)) / 2, 3)
    for name in Y.GRAMMARS:
        pop = Y.population(name)
        check(f'{name}: 384 sequences, 24 legal prefixes, each of probability 1/24',
              pop['n_sequences'] == 384 and len(pop['x']) == 24 and np.allclose(pop['prefix_prob'], 1 / 24))
        check(f'{name}: P(x_5 | prefix) = (K^3)[A] = 1/4 at A, 3/8 elsewhere',
              torch.allclose(pop['p'], K3[pop['A']], atol=1e-15), K3[0].tolist())
        # independent path: the left-to-right belief code of grammar.py on the same prefixes
        b = G.stack_belief(pop['x'][:, :Y.T_OBS], Y.rules(name), Y.V, Y.T_OBS)
        q = G.next_token_from_belief(b, Y.rules(name), Y.V, Y.T_OBS)
        check(f'{name}: targets equal grammar.stack_belief -> next_token_from_belief', torch.allclose(q, pop['p'], atol=1e-14),
              (q - pop['p']).abs().max().item())
        uni = Y.excess(torch.zeros(24, 3, dtype=torch.float64), pop['p']).item()
        check(f'{name}: a uniform predictor is UNIFORM_KL = 0.016417 nats above Bayes', abs(uni - Y.UNIFORM_KL) < 1e-15 and abs(uni - 0.016417) < 1e-6, uni)
        F = math.log(2 / 3) * (torch.eye(3, dtype=torch.float64) - 1 / 3)
        Fc = Y.center(pop['p'].log())
        check(f'{name}: centered Bayes logits are log(2/3) (e_A - 1/3), norm F_NORM, pairwise distance F_DIST',
              torch.allclose(Fc, F[pop['A']], atol=1e-14) and abs(F[0].norm().item() - Y.F_NORM) < 1e-14
              and abs((F[0] - F[1]).norm().item() - Y.F_DIST) < 1e-14)
        miss = lambda b, c: (-(b + c)) % 3
        check(f'{name}: the half symbol is the missing symbol of the two pair symbols, which always differ',
              bool((pop['cL'] != pop['cR']).all() and (miss(pop['cL'], pop['cR']) == pop['A']).all()))
    pop = Y.population('sym')
    check('sym: A = x_1 + x_2 + x_3 + x_4 mod 3 on legal prefixes', bool((pop['x'][:, :4].sum(1) % 3 == pop['A']).all()))
    check('sym: the pair symbol is the missing symbol of its two tokens',
          bool(((-(pop['x'][:, 0] + pop['x'][:, 1])) % 3 == pop['cL']).all()))

    # mod: the first child's distribution is unchanged, and the 4 x 4 legal submatrix of the crossed cut is regular
    first = lambda prods: sorted(sorted(b for b, _ in prods[a]) for a in range(3))
    check('mod: bottom-level first-child sets equal sym\'s', first(Y.MODB) == first(Y.SYM), first(Y.MODB))
    pm = Y.population('mod')
    Fc = {tuple(k[:4].tolist()): Y.center(q.log()).numpy() for k, q in zip(pm['x'], pm['p'])}
    rows, cols = [(0, 0), (0, 1), (0, 2), (1, 0)], [((1, 2), 0), ((1, 2), 1), ((2, 1), 0), ((2, 1), 1)]
    legal = all((r[0], c[0][0], r[1], c[0][1]) in Fc for r in rows for c in cols)
    M = np.array([[Fc[(r[0], c[0][0], r[1], c[0][1])][c[1]] for c in cols] for r in rows]) / (math.log(2 / 3) / 3) if legal else np.zeros((4, 4))
    check('mod: rows (x_1 x_3) = 00, 01, 02, 10 against columns (x_2 x_4, y) = (12, 0), (12, 1), (21, 0), (21, 1): all 16 '
          'entries legal, determinant -27 in units of log(2/3)/3', legal and abs(np.linalg.det(M) + 27) < 1e-9, np.linalg.det(M))

    bounds = {(n, g): Y.tree_bound(n, g) for n in Y.GRAMMARS for g in Y.PERMS}
    check('width lower bounds: sym aligned 2, sym crossed 2, mod aligned 3, mod crossed 4',
          bounds == {('sym', 'aligned'): 2, ('sym', 'crossed'): 2, ('mod', 'aligned'): 3, ('mod', 'crossed'): 4}, bounds)
    check('mod: the crossed bond (x_1, x_3) alone needs width 4; the aligned bond (x_1, x_2) needs 2; leaves x_1, x_3 need 3',
          Y.cut_bound('mod', (0, 2)) == 4 and Y.cut_bound('mod', (0, 1)) == 2 and Y.cut_bound('mod', (0,)) == 3
          and Y.cut_bound('mod', (2,)) == 3)

    # hand-built exact trees
    for g in Y.PERMS:
        e = Y.excess(Y.exact_complex(g)(pop['x']), pop['p']).item()
        check(f'sym {g}: the width-2 complex tree is exact', e < 1e-14, e)
    e = Y.excess(Y.exact_complex('aligned')(pm['x']), pm['p']).item()
    check('mod: the width-2 complex tree is NOT exact (it needs width 3)', e > 1e-4, e)
    for name in Y.GRAMMARS:
        p_ = Y.population(name)
        e = Y.excess(Y.exact_onehot(name)(p_['x']), p_['p']).item()
        check(f'{name} aligned: the width-3 one-hot message-passing tree is exact', e < 1e-14, e)
    m = Y.exact_complex('aligned')
    before = m(pop['x'])
    after = Y.regauge(m, seed=1)(pop['x'])
    check('regauge changes every core and leaves the logits unchanged', torch.allclose(before, after, atol=1e-12)
          and not torch.allclose(m.W[0], Y.exact_complex('aligned').W[0]), (before - after).abs().max().item())

    # optimizer: the loss gradient against a finite difference, a short run, and the polish
    model = Y.build(2, 'aligned', seed=5)
    loss = Y.excess(model(pop['x']), pop['p'])
    g = torch.autograd.grad(loss, model.C)[0]
    with torch.no_grad():
        model.C[1, 0, 1] += 1e-6
        up = Y.excess(model(pop['x']), pop['p']).item()
        model.C[1, 0, 1] -= 2e-6
        dn = Y.excess(model(pop['x']), pop['p']).item()
        model.C[1, 0, 1] += 1e-6
    fd = (up - dn) / 2e-6
    check('excess-loss gradient matches a central finite difference', abs(fd - g[1, 0, 1].item()) < 1e-8 * max(1, abs(fd)),
          (fd, g[1, 0, 1].item()))
    log, ck = Y.train(model, pop, 2000, 1e-2, ckpt_at=(0, 100))
    check('2,000 Adam updates lower the excess loss and save the requested checkpoints',
          log[-1][1] < log[0][1] and sorted(ck) == [0, 100, 2000], (log[0][1], log[-1][1]))
    k, info = Y.polish(model, pop)
    check('the L-BFGS polish does not raise the excess loss and reports its iterations', k <= log[-1][1] + 1e-15 and 0 < info['n_iter'] <= Y.POLISH_ITERS
          and not info['capped'], (log[-1][1], k, info))
    bad = Y.build(2, 'aligned', seed=5)
    with torch.no_grad():
        bad.C.fill_(float('nan'))
    k, info = Y.polish(bad, pop)
    check('the polish leaves a non-finite tree alone', not math.isfinite(k) and info['n_iter'] == 0)
    k, info = Y.polish(Y.build(2, 'aligned', seed=5), pop, iters=3)
    check('a polish stopped by its limit is marked capped', info['capped'], info)

    bad = [n for n, r in results.items() if not r['ok']]
    print(f'{len(results) - len(bad)}/{len(results)} checks passed')
    if len(sys.argv) > 1:
        json.dump(results, open(sys.argv[1], 'w'), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
