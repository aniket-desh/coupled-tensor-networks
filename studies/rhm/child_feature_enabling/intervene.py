"""Paired two-head intervention on the easy contrast's gradient into the lower cores (EXPERIMENT.md, section 3).

Both arms of a pair start from the same lower tree (leaf embeddings A and fusion cores W of an rhm-dynamics step-0
checkpoint; the categorical root is discarded) and two zero linear heads w_e, w_h on the root's product features
Z = z_left (x) z_right (row-major, left index first). Both minimize, by plain SGD on the same stream of batches,

    L = 1/2 E[(w_e.Z - c_e(Y))^2] + 1/2 E[(w_h.Z - c_h(Y))^2],

with c_e, c_h the most and least additive unit-variance root contrasts of the grammar's top rule table.
  intact    both terms are differentiated through Z;
  detached  the easy prediction is w_e.stopgrad(Z): the easy head still trains on the current features, but the
            easy error no longer moves the lower cores. Forward predictions are unchanged by the detach.

Every EVAL updates (and at step 0), exactly, from the population moments of Z (population.root_moments):
  A_c = 1 - min_w E[(c_Y - w.Z)^2]      the frozen-child ceiling: best linear root on the current features;
  D_c = 2 w_c.b_c - w_c^T F w_c          the trained head's fit, b_c = E[Z c_Y], F = E[Z Z^T]; D_c <= A_c.
The population loss of an arm is 1 - (D_e + D_h)/2. Gradient diagnostics at an evaluation use the next training
batch: the norms and cosine of the two terms' pre-detach gradients with respect to the lower cores.

Primary endpoint per arm: T = first scheduled evaluation with D_h >= 1/2 (censored at the horizon otherwise).

  python -m studies.rhm.child_feature_enabling.intervene --check
  python -m studies.rhm.child_feature_enabling.intervene --pilot
  python -m studies.rhm.child_feature_enabling.intervene --pairs 0:0 0:1 ...      (after the pilot)
Defaults are the protocol as written (rates 1e-4..1e-3, 300 pilot updates, 4,000 updates). The amended runs in
EXPERIMENT.md pass --out, --rates, --pilot-steps, --steps and --lr; run_all.sh has the exact commands.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from ctn.tree import TreeTN
from studies.rhm import common, grammar
from studies.rhm.child_feature_enabling import population, record
from studies.rhm.child_feature_enabling.checkpoints import contrasts_for

HERE = Path(__file__).resolve().parent
R, BATCH, STEPS, EVAL = 16, 256, 4000, 25
RATES, PILOT_STEPS, PILOT_SEED, LOSS_CAP = (1e-4, 3e-4, 1e-3), 300, 9, 10.0
ARMS = ('intact', 'detached')


def data_seed(g, i):
    return 7_000_400 + 1000 * g + i


def run_cfg(g, i):
    return dict(model='tree', r=R, tie='none', perm=None, seed=i, task='root', grammar=dict(common.GRAMMAR, seed=g),
                init='identity')


def initial_model(g, i, runs):
    """(float64 model, provenance). The stored step-0 checkpoint when it exists, else a rebuild from the config."""
    cfg = run_cfg(g, i)
    path = None if runs is None else Path(runs) / f'root_m4_g{g}_r{R}_s{i}' / 'ckpt_0.pt'
    if path is not None and path.exists():
        model, saved, step = common.load(path, device='cpu')
        assert saved == cfg and step == 0, (path, saved, step)
        source = dict(kind='stored step-0 checkpoint', file=str(path), sha256=record.sha256(path))
    else:
        model = common.build(cfg)
        source = dict(kind='rebuilt with common.build from the identity-init configuration', file=None, sha256=None)
    model = model.double()
    source['lower_sha256'] = record.tensor_sha256(torch.cat([p.flatten() for p in lower(model)]))
    return model, source


def lower(model):
    return [model.A, *model.W]


def features(model, x):
    """Z (B, r^2): outer product of the two bonds the root fuses, left index first."""
    _, levels = model(x, acts=True)
    left, right = levels[-1].unbind(dim=1)
    return (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(1)


def two_losses(model, heads, x, y, c, detach_easy):
    """(easy term, hard term) of the batch loss; c is (2, n) with rows c_e, c_h and heads is (w_e, w_h)."""
    Z = features(model, x)
    easy = (Z.detach() if detach_easy else Z) @ heads[0]
    hard = Z @ heads[1]
    return (easy - c[0, y]).square().mean() / 2, (hard - c[1, y]).square().mean() / 2


def sgd_step(model, heads, x, y, c, lr, detach_easy):
    params = lower(model) + list(heads)
    easy, hard = two_losses(model, heads, x, y, c, detach_easy)
    grads = torch.autograd.grad(easy + hard, params)
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= lr * g


def lower_gradients(model, heads, x, y, c):
    """Pre-detach gradients of the two loss terms with respect to the lower cores, each flattened to one vector."""
    easy, hard = two_losses(model, heads, x, y, c, detach_easy=False)
    return tuple(torch.cat([g.flatten() for g in torch.autograd.grad(term, lower(model), retain_graph=True)])
                 for term in (easy, hard))


@torch.no_grad()
def evaluate(model, heads, rules, c, start):
    moments = population.root_moments(model, rules)
    row, default = {}, len(moments.gram) * torch.finfo(torch.float64).eps
    for name, target, w in zip('eh', c, heads):
        fit, loose = (population.regression_oracle(moments, target, rtol=k * default) for k in (1, 10))
        b = moments.class_means.T @ target / len(target)
        row[f'A_{name}'], row[f'A_{name}_cutoff10'], row[f'rank_{name}'] = fit.explained, loose.explained, fit.rank
        row[f'D_{name}'] = (2 * w @ b - w @ moments.gram @ w).item()
    row['pop_loss'] = 1 - (row['D_e'] + row['D_h']) / 2
    row['lower_shift'] = torch.cat([(p - q).flatten() for p, q in zip(lower(model), start)]).norm().item()
    return row


def run_pair(g, i, lr, steps, runs, seed=None, save_dir=None):
    """Train both arms on one batch stream. Returns the per-evaluation rows of each arm and the pair's record."""
    t0 = time.time()
    gcfg = dict(common.GRAMMAR, seed=g)
    rules = common.rules(gcfg)
    spec, c = contrasts_for(gcfg)
    base, source = initial_model(g, i, runs)
    start = [p.detach().clone() for p in lower(base)]
    models = {arm: copy.deepcopy(base) for arm in ARMS}
    heads = {arm: [torch.zeros(R * R, dtype=torch.float64, requires_grad=True) for _ in range(2)] for arm in ARMS}
    seed = data_seed(g, i) if seed is None else seed
    gen = torch.Generator().manual_seed(seed)
    rows = {arm: [] for arm in ARMS}
    status = 'finished'
    for step in range(steps + 1):
        batch = None
        if step < steps:  # one batch per update, drawn once and given to both arms
            trees = grammar.sample_trees(BATCH, rules, g=gen)
            batch = (trees[gcfg['L']], trees[0][:, 0])
        if step % EVAL == 0:
            for arm in ARMS:
                row = dict(step=step, **evaluate(models[arm], heads[arm], rules, c, start))
                if batch is not None:
                    ge, gh = lower_gradients(models[arm], heads[arm], *batch, c)
                    row.update(grad_easy=ge.norm().item(), grad_hard=gh.norm().item(),
                               grad_cos=(ge @ gh / (ge.norm() * gh.norm())).item())
                rows[arm].append(row)
            if not all(math.isfinite(rows[arm][-1]['pop_loss']) for arm in ARMS):
                status = 'nonfinite'
                break
            if step % (20 * EVAL) == 0:
                print(f'g{g} s{i} lr {lr:g} step {step:>6}  ' + '  '.join(
                    f"{arm}: D_h {rows[arm][-1]['D_h']:.4f} A_h {rows[arm][-1]['A_h']:.4f} "
                    f"D_e {rows[arm][-1]['D_e']:.4f} A_e {rows[arm][-1]['A_e']:.4f}" for arm in ARMS), flush=True)
        if batch is not None:
            for arm in ARMS:
                sgd_step(models[arm], heads[arm], *batch, c, lr, detach_easy=(arm == 'detached'))
    out = dict(grammar=g, seed=i, lr=lr, steps=steps, batch=BATCH, eval_every=EVAL, data_seed=seed, status=status,
               init=source, contrasts=spec, seconds=time.time() - t0, arms={})
    for arm in ARMS:
        cols = {k: [row.get(k) for row in rows[arm]] for k in rows[arm][0]}
        slack = max(d - a for d, a in zip(cols['D_h'] + cols['D_e'], cols['A_h'] + cols['A_e']))
        out['arms'][arm] = dict(columns=cols, crossing=crossing(cols['step'], cols['D_h'], steps),
                                max_D_minus_A=slack)
        if save_dir is not None and status == 'finished':
            path = Path(save_dir) / f'g{g}_s{i}_{arm}_final.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(lower=[p.detach() for p in lower(models[arm])],
                            heads=[w.detach() for w in heads[arm]], step=steps, lr=lr), path)
            out['arms'][arm]['final_state'] = dict(file=str(path), sha256=record.sha256(path))
    return out


def crossing(steps, fits, horizon, level=0.5):
    """First scheduled evaluation with fit >= level, with its bracketing evaluations; censored if none."""
    for k, (step, fit) in enumerate(zip(steps, fits)):
        if fit >= level:
            return dict(T=step, censored=False, bracket=[steps[max(k - 1, 0)], step],
                        bracket_fits=[fits[max(k - 1, 0)], fit])
    return dict(T=None, censored=True, bound=f'T > {steps[-1]}', last_step=steps[-1], last_fit=fits[-1],
                max_fit=max(fits), horizon=horizon)


def check():
    """The detach is a path intervention: with identical parameters, nonzero heads and one batch, the two arms give
    the same predictions and head gradients, and their lower-core gradients differ by exactly the easy term's."""
    g, i = 0, PILOT_SEED
    gcfg = dict(common.GRAMMAR, seed=g)
    rules = common.rules(gcfg)
    _, c = contrasts_for(gcfg)
    model, _ = initial_model(g, i, None)
    gen = torch.Generator().manual_seed(1)
    heads = [(0.05 * torch.randn(R * R, generator=gen, dtype=torch.float64)).requires_grad_() for _ in range(2)]
    trees = grammar.sample_trees(BATCH, rules, seed=2)
    x, y = trees[gcfg['L']], trees[0][:, 0]
    params = lower(model) + heads
    grads = {}
    for arm in ARMS:
        easy, hard = two_losses(model, heads, x, y, c, detach_easy=(arm == 'detached'))
        grads[arm] = (easy.item(), hard.item(), torch.autograd.grad(easy + hard, params))
    assert grads['intact'][:2] == grads['detached'][:2]  # identical forward predictions, hence identical losses
    for a, b in zip(grads['intact'][2][-2:], grads['detached'][2][-2:]):  # both head gradients coincide
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    ge, gh = lower_gradients(model, heads, x, y, c)
    flat = {arm: torch.cat([t.flatten() for t in grads[arm][2][:-2]]) for arm in ARMS}
    torch.testing.assert_close(flat['intact'] - flat['detached'], ge, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(flat['detached'], gh, rtol=1e-12, atol=1e-14)
    assert ge.norm() > 1e-6 and gh.norm() > 1e-6  # the test point exercises both paths
    # D from the exact moments equals the enumerated fit on a grammar small enough to list every string.
    small = grammar.sample_rules(v=3, n=2, m=2, s=2, L=2, seed=7)
    strings = torch.cartesian_prod(*[torch.arange(3) for _ in range(4)])
    label = grammar.parse(strings, small, 3)[0].flatten()
    xs, ys = strings[label >= 0], label[label >= 0]
    tiny = TreeTN(4, 3, 5, 2, seed=17).double()
    cs = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
    ws = [0.1 * torch.randn(25, generator=gen, dtype=torch.float64) for _ in range(2)]
    row = evaluate(tiny, ws, small, cs, [p.detach().clone() for p in lower(tiny)])
    with torch.no_grad():
        Z = features(tiny, xs)
        for name, target, w in zip('eh', cs, ws):
            direct = 1 - (target[ys] - Z @ w).square().mean().item()
            best = 1 - (target[ys] - Z @ torch.linalg.lstsq(Z, target[ys], driver='gelsd').solution).square().mean().item()
            assert abs(row[f'D_{name}'] - direct) < 1e-12 and abs(row[f'A_{name}'] - best) < 1e-10
            assert row[f'D_{name}'] <= row[f'A_{name}'] + 1e-12
    print('PASS: equal predictions and head gradients across arms; lower-gradient difference = easy term '
          f'(|g_e| {ge.norm():.3e}, |g_h| {gh.norm():.3e}); D and A match enumeration on a 16-string grammar')


def pilot(a, out):
    """Grammar 0, init seed 9, 300 updates per rate. Eligible: finite, both arms' population loss < 10 at every
    evaluation. Choice: smallest mean final hard MSE (1 - D_h) over the two arms, ties toward the smaller rate."""
    results = []
    for lr in a.rates:
        pair = run_pair(0, PILOT_SEED, lr, a.pilot_steps, None)
        losses = [v for arm in ARMS for v in pair['arms'][arm]['columns']['pop_loss']]
        eligible = pair['status'] == 'finished' and all(math.isfinite(v) and v < LOSS_CAP for v in losses)
        final = [1 - pair['arms'][arm]['columns']['D_h'][-1] for arm in ARMS]
        results.append(dict(lr=lr, eligible=eligible, max_pop_loss=max(losses), final_hard_mse=dict(zip(ARMS, final)),
                            mean_final_hard_mse=sum(final) / 2, pair=pair))
        print(f"pilot lr {lr:g}: eligible {eligible}, mean final hard MSE {results[-1]['mean_final_hard_mse']:.6f}",
              flush=True)
    ok = [r for r in results if r['eligible']]
    choice = min(ok, key=lambda r: (r['mean_final_hard_mse'], r['lr']))['lr'] if ok else None
    record.dump(out / 'pilot.json', dict(meta=record.provenance(__file__, population.__file__), rates=list(a.rates),
                                         steps=a.pilot_steps, seed=PILOT_SEED, grammar=0, loss_cap=LOSS_CAP,
                                         choice=choice, results=results), indent=None)
    if choice is None:
        raise SystemExit('no rate qualified in the pilot; stopping as the protocol requires')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pairs', nargs='*', default=[], help='grammar:seed, e.g. 0:0 0:1')
    parser.add_argument('--runs', default='runs/rhm-dynamics', help='where the stored step-0 checkpoints are')
    parser.add_argument('--out', default=str(HERE / 'results' / 'intervention'))
    parser.add_argument('--states', help='folder for final weights (default runs/child-feature-enabling/<out name>; '
                                         'runs/ is not in git)')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--rates', type=float, nargs='+', default=RATES)
    parser.add_argument('--pilot-steps', type=int, default=PILOT_STEPS)
    parser.add_argument('--lr', type=float, help='override the pilot choice (recorded)')
    parser.add_argument('--threads', type=int, default=1)
    a = parser.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    states = a.states or f'runs/child-feature-enabling/{out.name}'
    if a.check:
        check()
    if a.pilot:
        pilot(a, out)
    for pair in a.pairs:
        g, i = map(int, pair.split(':'))
        lr = a.lr if a.lr is not None else json.loads((out / 'pilot.json').read_text())['choice']
        result = run_pair(g, i, lr, a.steps, a.runs, save_dir=states)
        result['meta'] = dict(record.provenance(__file__, population.__file__), lr_source='--lr' if a.lr else 'pilot',
                              precision='float64 CPU', optimizer='plain SGD, constant rate, no momentum/decay/clipping')
        record.dump(out / 'pairs' / f'g{g}_s{i}.json', result, indent=None)


if __name__ == '__main__':
    main()
