"""Checks of what this experiment adds (no training):
  perm_for            a permutation, deterministic, no data-tree sibling pair kept as model siblings, different per seed;
  permuted tree       the top bond h_A depends only on the tokens at positions perm[:T/2], h_B on the others, and for a
                      short prefix both already vary (no first half), unlike the aligned tree;
  chain               the causal read-out on RHM tokens: logits[:, t] and state t do not change when tokens >= t change;
                      the loss rows are ordered like the tree's (sequence, position); the width is the closest in
                      parameter count; the three arms draw the same training sequences;
  additive_backfit    equals the dense least-squares fit on [onehot(a), onehot(b)]; fits an additive target exactly and
                      not a pure interaction; handles an empty factor;
  set_token_product   has the stated width and equals ttn_analyze.token_product for the aligned halves;
  direction, compare  arch_summary's pairing rule on hand-made win/loss lists: the 80% rule with grammar consistency and
                      a minimum of pairs; the identity-init gap is subtracted from the representation gap; pairs whose
                      permuted tree did not learn leave the representation measure; a grammar with an unfinished run
                      is censored whole;
  softmax probe       softmax_fit_predict recovers a log-linear code of the belief that the affine probe misses, its
                      rows sum to 1, and it is unchanged by an invertible linear change of the state."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arch_analyze as AA  # noqa: E402
import arch_run as AR  # noqa: E402
import ttn_analyze as TA  # noqa: E402
import ttn_run as TR  # noqa: E402
from ctn.tree import n_params  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

ok = True


def check(name, cond, note=''):
    global ok
    ok &= bool(cond)
    print(f'{"PASS" if cond else "FAIL"}  {name}  {note}')


for n in (8, 16, 32):
    p, kept = AR.perm_for('v8m4s2L4', 1, n)
    sib = any({p[2 * j], p[2 * j + 1]} == {2 * k, 2 * k + 1} for j in range(n // 2) for k in range(n // 2))
    check(f'perm_for n={n}: permutation, no sibling pair kept, deterministic, seed-dependent',
          sorted(p) == list(range(n)) and not sib and kept['level1'] == 0 and p == AR.perm_for('v8m4s2L4', 1, n)[0]
          and p != AR.perm_for('v8m4s2L4', 2, n)[0], f'kept {kept}')

g = torch.Generator().manual_seed(0)
cfg = AR.make_cfg('v8m4s2L4', 'perm', 0)
cfg['init_eps'] = 1.0  # far from the pass-through init
v, T, r = cfg['V'], cfg['T'], cfg['r']
model = AR.build(cfg)
A, B = AA.leaf_sets(cfg)
x = torch.randint(0, v, (2000, T), generator=g)
top = lambda xx, t: TA.bond_sets(model, xx, t, v, bilinear=False)['top']
xb = x.clone()
xb[:, B] = torch.randint(0, v, (2000, len(B)), generator=g)
xa = x.clone()
xa[:, A] = torch.randint(0, v, (2000, len(A)), generator=g)
t0, tb, ta = top(x, T), top(xb, T), top(xa, T)
check('permuted tree: h_A sees only positions perm[:T/2], h_B only the others',
      (t0[:, :r] - tb[:, :r]).abs().max() < 1e-6 and (t0[:, r:] - ta[:, r:]).abs().max() < 1e-6
      and (t0[:, r:] - tb[:, r:]).abs().max() > 1e-4 and (t0[:, :r] - ta[:, :r]).abs().max() > 1e-4)
t_early = max(min(A), min(B)) + 1
se = top(x, t_early)
cfg_t = AR.make_cfg('v8m4s2L4', 'tree', 0)
cfg_t['init_eps'] = 1.0
tree = AR.build(cfg_t)
st = TA.bond_sets(tree, x, t_early, v, bilinear=False)['top']
check(f'permuted tree: both top bonds vary at t = {t_early} <= T/2; aligned tree: h_R constant there',
      t_early <= T // 2 and se[:, :r].std(0).max() > 1e-4 and se[:, r:].std(0).max() > 1e-4 and st[:, r:].std(0).max() < 1e-6)

cc = AR.make_cfg('v8m4s2L4', 'chain', 0)
chain = AR.build(cc)
rules = G.sample_rules(v, 8, 4, 2, 4, seed=0)
xr = G.sample_trees(64, rules, seed=1)[4]
states, logits = BC.chain_states(chain, xr)
x2 = xr.clone()
x2[:, 9:] = (x2[:, 9:] + 1) % v
s2, l2 = BC.chain_states(chain, x2)
check('chain: state and logits at t <= 9 unchanged by tokens >= 9, later ones change',
      (states[:, :10] - s2[:, :10]).abs().max() < 1e-6 and (logits[:, :10] - l2[:, :10]).abs().max() < 1e-6
      and (logits[:, 10:] - l2[:, 10:]).abs().max() > 1e-6)
lg, y = AR.logits_targets(chain, 'chain', xr, v)
_, yt = TR.prefixes(xr, v)
check('chain: loss rows ordered (sequence, position) like the tree arms', (y == yt).all() and torch.allclose(lg.reshape(64, T, v), logits))
w, target = AR.chain_width(T, v)
check('chain width closest in parameters to the aligned tree',
      n_params(chain) == AR.chain_params(w, v) and n_params(tree) == target
      and all(abs(AR.chain_params(w, v) - target) <= abs(AR.chain_params(q, v) - target) for q in (w - 1, w + 1)),
      f'width {w}: {n_params(chain)} against {target}')

N = 3000
ia, ib = torch.randint(0, 7, (N,), generator=g), torch.randint(0, 11, (N,), generator=g)
fi, ti = torch.arange(0, 2000), torch.arange(2000, N)
add = torch.stack([ia.double() ** 2 + (ib * 1.3).double().sin(), ib.double() - ia], 1)
inter = ((ia + ib) % 3 == 0).double()[:, None]
noisy = add + inter + 0.3 * torch.randn(N, 2, generator=g, dtype=torch.float64)
dense = torch.cat([torch.nn.functional.one_hot(ia, 7), torch.nn.functional.one_hot(ib, 11)], 1).double()
R2 = lambda pred, tgt: BC.score(tgt[ti], pred, tgt[fi].mean(0))['R2']
bf = lambda tgt, a=ia, b=ib, na=7, nb=11: AA.additive_backfit(a, b, na, nb, tgt, fi, ti)
d_noisy = (bf(noisy) - BC.fit_predict(dense[fi], noisy[fi], dense[ti])).abs().max().item()
check('additive_backfit: equals dense least squares on the one-hot design', d_noisy < 1e-6, f'max difference {d_noisy:.1e}')
check('additive_backfit: additive target exact, interaction not', R2(bf(add), add) > 1 - 1e-9 and R2(bf(inter), inter) < 0.3,
      f'R2 additive {R2(bf(add), add):.6f}, interaction {R2(bf(inter), inter):.3f}')
one = AA.additive_backfit(None, ib, 0, 11, noisy, fi, ti)
means = torch.stack([noisy[fi][ib[fi] == k].mean(0) for k in range(11)])
check('additive_backfit: one empty factor gives the group means of the other', (one - means[ib[ti]]).abs().max() < 1e-9)

xs = torch.randint(0, v, (50, T), generator=g)
tp = AA.set_token_product(xs, 11, A, B, v)
check('set_token_product: stated width', tp.shape[1] == AA.n_token_product(11, A, B, v))
half_a, half_b = list(range(T // 2)), list(range(T // 2, T))
check('set_token_product: equals ttn_analyze.token_product for the aligned halves',
      torch.equal(AA.set_token_product(xs, 13, half_a, half_b, v), TA.token_product(xs, 13, v)))

gens = [torch.Generator().manual_seed(1000) for _ in range(2)]
check('arms draw the same training sequences (same generator seed and call)',
      torch.equal(G.sample_trees(8, rules, g=gens[0])[4], G.sample_trees(8, rules, g=gens[1])[4]))
import arch_summary as AS  # noqa: E402
gs = ['a', 'b', 'c', 'd']
mk = lambda signs: [(gs[i // 3], s) for i, s in enumerate(signs)]
d1 = AS.direction(mk([1] * 10 + [0, -1]))                          # 10 of 12; grammar d: 1 win, 1 tie, 1 loss -> net 0
d2 = AS.direction(mk([1] * 9 + [0] * 3))                           # 9 of 12: below 80%
d3 = AS.direction(mk([-1] * 12))                                   # all to the other arm
d4 = AS.direction(mk([1] * 10 + [-1] * 2))                         # 10 of 12 but grammar d reversed
d5 = AS.direction(mk([1] * 8))                                     # fewer than 9 usable pairs
d6 = AS.direction(mk([1] * 9))                                     # three whole grammars, all wins
check('direction: 80% rule, grammar consistency, minimum pairs, two-sided',
      [d['to'] for d in (d1, d2, d3, d4, d5, d6)] == ['ref', None, 'other', None, 'untestable', 'ref'],
      f"{[d['to'] for d in (d1, d2, d3, d4, d5, d6)]}, sign p for 10-1: {d1['sign_p']:.4f}")


def run_row(c, arm, s, excess, hi, hi0):
    return dict(config=c, arm=arm, seed=s, excess=excess, learned=excess < AS.EXCESS_MAX, finished=True,
                eb_hi_best=hi, eb_hi_init=hi0, eb_lo_best=None, eb_lo_init=None, eb_best=hi, eb_init=hi0)


cellx = {}
for c in gs:
    cellx[(c, 'tree')] = [run_row(c, 'tree', s, 0.005 + 0.001 * s, 0.9, 0.5) for s in range(3)]
    # grammars a, b: the gap is 0.3 at init and 0.3 at the end (nothing learned in the gap); c: the gap opens by 0.3;
    # d: the permuted tree does not learn
    hi = dict(a=0.6, b=0.6, c=0.6, d=0.2)[c]
    hi0 = dict(a=0.2, b=0.2, c=0.5, d=0.2)[c]
    cellx[(c, 'perm')] = [run_row(c, 'perm', s, 0.5 if c == 'd' else 0.012, hi, hi0) for s in range(3)]
full = {k: True for k in cellx}
cmp1 = AS.compare(cellx, gs, full, 'tree', 'perm')
pr = {p['pair']: p for p in cmp1['pairs']}
part = {k: v and k[0] != 'b' for k, v in full.items()}
cmp2 = AS.compare(cellx, gs, part, 'tree', 'perm')
check('compare: the identity-init gap is subtracted, unlearned permuted trees leave the representation measure, a grammar '
      'with an unfinished run is censored whole',
      pr['a_s0']['eb_hi']['sign'] == 0 and pr['c_s0']['eb_hi']['sign'] == 1 and cmp1['representation']['usable'] == 9
      and cmp1['representation']['ref_wins'] == 3 and cmp1['learning']['usable'] == 12 and cmp1['learning']['to'] == 'ref'
      and abs(cmp1['median_learning_gap'] - 0.007) < 2e-3 and cmp2['censored_grammars'] == ['b']
      and cmp2['learning']['usable'] == 9 and cmp2['representation']['to'] == 'untestable',
      f"representation {cmp1['representation']['ref_wins']}-{cmp1['representation']['other_wins']}-{cmp1['representation']['ties']} "
      f"of {cmp1['representation']['usable']}; verdict: {cmp1['verdict']}")

# softmax-of-affine probe on a synthetic state whose linear readout is the log-belief
gq = torch.Generator().manual_seed(5)
hs = torch.randn(4000, 20, generator=gq, dtype=torch.float64)
bs = (hs @ torch.randn(20, 6, generator=gq, dtype=torch.float64) * 0.8).softmax(1)
f_, t_ = slice(0, 2000), slice(2000, None)
r2 = lambda pred: BC.score(bs[t_], pred, bs[f_].mean(0))['R2']
soft, ratio = AA.softmax_fit_predict(hs[f_], bs[f_], hs[t_])
gauge = torch.randn(20, 20, generator=gq, dtype=torch.float64)
soft_g, _ = AA.softmax_fit_predict(hs[f_] @ gauge, bs[f_], hs[t_] @ gauge)
r_aff, r_soft = r2(BC.fit_predict(hs[f_], bs[f_], hs[t_])), r2(soft)
check('softmax_fit_predict: recovers a log-linear belief code the affine probe misses, rows sum to 1, gauge invariant',
      r_soft > 0.999 and r_aff < 0.95 and (soft.sum(1) - 1).abs().max() < 1e-9 and (soft - soft_g).abs().max() < 1e-4,
      f'R2 softmax {r_soft:.5f}, affine {r_aff:.3f}, loss ratio {ratio:.1e}, gauge change moves predictions by {(soft - soft_g).abs().max():.1e}')
print('all pass' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
