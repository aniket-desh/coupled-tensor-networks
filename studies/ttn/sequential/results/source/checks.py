"""Correctness checks for ctn/ttn.py, each aimed at an error that would corrupt a learning-time or coupling measurement.
Usage: python -m studies.ttn.checks [out.json]  (CPU, float64; default output studies/ttn/checks.json)"""
import sys
from pathlib import Path

import numpy as np
import torch

from ctn import io
from ctn.ttn import GROUPS, TTN, Population, alignment, coeffs, gram, loss_per_run, mobility, sample, targets, teacher_cores

from . import teachers

torch.manual_seed(0)
f64 = torch.float64
out = {}

# 1. Core-wise evaluation equals the dense coefficient tensor T in R^{O x p^8}, built only here (p = 2, r = 3).
net = TTN(R=2, r=3, d=1, O=3, seed=1, dtype=f64)
x, phi = sample(2, 64, 1, seed=2, device='cpu', dtype=f64)
A, Wq, Wh, C = [t.detach().numpy() for t in net.params]
dense = []
for run in range(2):
    Q = [np.einsum('abc,bi,cj->aij', Wq[run, k], A[run, 2 * k], A[run, 2 * k + 1]) for k in range(4)]
    H = [np.einsum('abc,bij,ckl->aijkl', Wh[run, h], Q[2 * h], Q[2 * h + 1]) for h in range(2)]
    T = np.einsum('obc,bijkl,cmnpq->oijklmnpq', C[run], H[0], H[1])  # (O, 2, ..., 2)
    P = phi[run].numpy()  # (N, 8, 2)
    dense.append(np.einsum('oijklmnpq,zi,zj,zk,zl,zm,zn,zp,zq->zo', T, *[P[:, j] for j in range(8)]))
out['core_contraction_vs_dense_T'] = float(np.abs(net(phi).detach().numpy() - np.array(dense)).max())

# 2. Autograd gradient against centered differences (float64, small network).
comps = teachers.agreement('quarter')
net = TTN(R=1, r=3, d=8, O=3, seed=3, dtype=f64)
x, phi = sample(1, 32, 8, seed=4, device='cpu', dtype=f64)
y, g = targets(x, comps, 3)
active = torch.ones(1, 3, dtype=f64)
L = lambda: loss_per_run(net(phi), y, active).sum()
grads = torch.autograd.grad(L(), net.params)
err = 0.0
rng = np.random.default_rng(0)
with torch.no_grad():
    for i, p in enumerate(net.params):
        for _ in range(5):
            idx = tuple(int(rng.integers(n)) for n in p.shape)
            old = p[idx].item()
            p[idx] = old + 1e-6
            lp = L().item()
            p[idx] = old - 1e-6
            lm = L().item()
            p[idx] = old
            err = max(err, abs((lp - lm) / 2e-6 - grads[i][idx].item()))
out['autograd_vs_finite_difference'] = err

# 3. Every teacher used is exactly representable at r = 4 (the smallest bond dimension the experiments use).
x, phi = sample(1, 2048, 8, seed=5, device='cpu', dtype=f64)
worst = 0.0
for comps in [teachers.agreement(f) for f in teachers.SETS] + [teachers.degree(D) for D in (1, 2, 4, 6, 8)]:
    net = TTN(R=1, r=4, d=8, O=10, dtype=f64)
    with torch.no_grad():
        for p, w in zip(net.params, teacher_cores(comps, 4, 8, 10)):
            p[0] = torch.tensor(w)
    y, _ = targets(x, comps, 10)
    worst = max(worst, (net(phi) - y).abs().max().item())
out['teacher_cores_reproduce_targets_max_err'] = worst

# 4. Teacher components are orthonormal on a large sample (sampling error ~ 1/sqrt(N) = 0.004).
x, phi = sample(1, 65536, 8, seed=6, device='cpu', dtype=f64)
for fam in teachers.SETS:
    _, g = targets(x, teachers.agreement(fam), 3)
    out[f'gram_minus_identity_{fam}'] = float((g[0].T @ g[0] / g.shape[1] - torch.eye(3, dtype=f64)).abs().max())

# 5. Variance-controlled init gives second moment ~1 at every layer for alpha = 1; alpha scales the output's
#    second moment by alpha^30. Averaged over 32 independent inits; the per-init spread is reported too.
x, phi = sample(32, 2048, 8, seed=7, device='cpu', dtype=f64)
for r in (4, 8, 16):
    for alpha in (1.0, 0.7):
        net = TTN(R=32, r=r, d=8, O=10, alpha=alpha, seed=8, dtype=f64)
        with torch.no_grad():
            a = net.forward(phi)
        for layer in ('leaf', 'quarter', 'half', 'out'):
            m2 = (a[layer] ** 2).mean(tuple(range(1, a[layer].dim())))
            out[f'init_m2_r{r}_alpha{alpha}_{layer}'] = [float(m2.mean()), float(m2.std())]
out['init_m2_out_alpha0.7_predicted'] = 0.7 ** 30

# 6. Mobility: groups sum to <grad c_i, grad c_j> over all parameters, and one small GD step moves c by
#    -eta <grad c, grad L> to first order (checks the masks, groups and the sign convention together).
comps = teachers.agreement('quarter')
net = TTN(R=2, r=4, d=8, O=3, seed=9, dtype=f64)
x, phi = sample(2, 512, 8, seed=10, device='cpu', dtype=f64)
y, g = targets(x, comps, 3)
active = torch.ones(2, 3, dtype=f64)
K = mobility(net, lambda: coeffs(net, phi, g, comps), 3)
gc = [torch.autograd.grad(coeffs(net, phi, g, comps)[:, k].sum(), net.params) for k in range(3)]
full = torch.tensor([[[sum((a[i][run] * b[i][run]).sum() for i in range(4)) for b in gc] for a in gc] for run in range(2)])
out['mobility_groups_sum_to_full'] = float((K.sum(-1) - full).abs().max())
gL = torch.autograd.grad(loss_per_run(net(phi), y, active).sum(), net.params)
pred = -torch.stack([sum((gc[k][i] * gL[i]).flatten(1).sum(1) for i in range(4)) for k in range(3)], -1)
_, v = mobility(net, lambda: coeffs(net, phi, g, comps), 3, lambda: loss_per_run(net(phi), y, active))
out['velocity_vs_grad_inner_product'] = float(((v.sum(-1) - pred) / pred.abs().max()).abs().max())
c0 = coeffs(net, phi, g, comps).detach()
eta = 1e-6
with torch.no_grad():
    for p, gr in zip(net.params, gL):
        p -= eta * gr
dc = (coeffs(net, phi, g, comps).detach() - c0) / eta
out['first_order_dc_rel_err'] = float(((dc - pred) / pred.abs().max()).abs().max())

# 7. Freezing: frozen cores get no update and drop out of K; masked outputs don't influence the gradient.
net = TTN(R=2, r=4, d=8, O=3, seed=11, dtype=f64)
net.freeze([0], ['q1', 'L'])
before = [p.detach().clone() for p in net.params]
gL = torch.autograd.grad(loss_per_run(net(phi), y, active).sum(), net.params)
with torch.no_grad():
    for p, gr, m in zip(net.params, gL, net.frozen):
        p -= 0.1 * gr * m
moved = lambda i, sl: float((net.params[i][sl] - before[i][sl]).abs().max().detach())
out['frozen_q1_L_moved'] = max(moved(0, (0, slice(0, 2))), moved(1, (0, 0)), moved(2, (0, 0)))
out['unfrozen_run_moved_q1'] = moved(0, (1, slice(0, 2)))
K = mobility(net, lambda: coeffs(net, phi, g, comps), 3)
out['frozen_groups_K_abs'] = float(K[0, :, :, [list(GROUPS).index('q1'), list(GROUPS).index('L')]].abs().max())
_, v = mobility(net, lambda: coeffs(net, phi, g, comps), 3, lambda: loss_per_run(net(phi), y, active))
out['frozen_groups_velocity_abs'] = float(v[0, :, [list(GROUPS).index('q1'), list(GROUPS).index('L')]].abs().max())
mask = torch.tensor([[1.0, 0.0, 1.0]] * 2, dtype=f64)
y2 = y.clone()
y2[..., 1] += 5.0
g1 = torch.autograd.grad(loss_per_run(net(phi), y, mask).sum(), net.params)
g2 = torch.autograd.grad(loss_per_run(net(phi), y2, mask).sum(), net.params)
out['masked_output_target_changes_grad'] = max(float((a - b).abs().max()) for a, b in zip(g1, g2))

# 8. Alignment is 1 at the teacher and ~r/N for a feature the node cannot carry.
comps = teachers.agreement('quarter')
net = TTN(R=1, r=4, d=8, O=3, dtype=f64)
with torch.no_grad():
    for p, w in zip(net.params, teacher_cores(comps, 4, 8, 3)):
        p[0] = torch.tensor(w)
x, phi = sample(1, 4096, 8, seed=12, device='cpu', dtype=f64)
with torch.no_grad():
    acts = net.forward(phi)
out['alignment_at_teacher_min'] = float(alignment(acts, x, comps).min())
other = [dict(out=0, s=1.0, coords=((0, 5), (1, 5)))]
out['alignment_unrelated_feature_q1'] = float(alignment(acts, x, other)[0, 0, 0])

# 9. Exact population quantities (the training loss of every experiment): Gram contraction against the dense T,
#    population loss and coefficients against Monte Carlo, and c_k = s_k exactly at the teacher.
net = TTN(R=2, r=3, d=1, O=3, seed=1, dtype=f64)
A, Wq, Wh, C = [t.detach().numpy() for t in net.params]
err = 0.0
for run in range(2):
    Q = [np.einsum('abc,bi,cj->aij', Wq[run, k], A[run, 2 * k], A[run, 2 * k + 1]) for k in range(4)]
    H = [np.einsum('abc,bij,ckl->aijkl', Wh[run, h], Q[2 * h], Q[2 * h + 1]) for h in range(2)]
    T = np.einsum('obc,bijkl,cmnpq->oijklmnpq', C[run], H[0], H[1])
    err = max(err, float(np.abs((T.reshape(3, -1) ** 2).sum(1) - gram(net.params, net.params)[run].detach().numpy()).max()))
out['gram_vs_dense_T_norm'] = err
comps = teachers.agreement('quarter')
net = TTN(R=4, r=8, d=8, O=3, seed=2, dtype=f64)
pop = Population(comps, 4, 8, 8, 3, 'cpu', f64)
act = torch.ones(4, 3, dtype=f64)
x, phi = sample(4, 2 ** 16, 8, seed=3, device='cpu', dtype=f64)
y, g = targets(x, comps, 3)
with torch.no_grad():
    Lp, Lm = pop.loss(net, act), loss_per_run(net(phi), y, act)
    out['population_vs_mc_loss_max_rel'] = float(((Lp - Lm) / Lp).abs().max())  # MC error ~ 1e-2 at N = 2^16
    out['population_vs_mc_coeff_max_abs'] = float((pop.coeffs(net) - coeffs(net, phi, g, comps)).abs().max())  # MC sd ~ 0.004
    with torch.no_grad():
        for p, w in zip(net.params, teacher_cores(comps, 8, 8, 3)):
            p[:] = torch.tensor(w)
    out['population_at_teacher_coeff_err'] = float((pop.coeffs(net) - torch.tensor([c['s'] for c in comps], dtype=f64)).abs().max())
    out['population_at_teacher_loss'] = float(pop.loss(net, act).abs().max())

# 10. Basis-permutation symmetry: E[phi phi^T] = I and the leaf init is isotropic, so permuting a leaf's basis
#     (constant <-> x_1) is an exact symmetry of population GD. A degree-2 product and a degree-8 product are then the
#     same problem; what the dynamics see is which leaves two outputs share a basis index on. Experiments are
#     designed around this; if it ever fails, those designs need revisiting.
def c1_path(comps, swap_leaves, steps=200, eta=0.05):
    net = TTN(R=2, r=8, d=8, O=10, seed=3000, dtype=f64)
    with torch.no_grad():
        for j in swap_leaves:
            net.params[0][:, j, :, [0, 1]] = net.params[0][:, j, :, [1, 0]]
    act = torch.zeros(2, 10, dtype=f64)
    act[:, comps[0]['out']] = 1
    pop = Population(comps, 2, 8, 8, 10, 'cpu', f64)
    path = []
    for i in range(steps):
        L = pop.loss(net, act)
        path.append(pop.coeffs(net)[:, 0].detach().clone())
        g = torch.autograd.grad(L.sum(), net.params)
        with torch.no_grad():
            for p, gr in zip(net.params, g):
                p -= eta * gr
    return torch.stack(path)
deg2 = [dict(out=0, s=1.0, coords=((0, 0), (4, 0)))]
deg8 = [dict(out=0, s=1.0, coords=tuple((j, 0) for j in range(8)))]
out['permutation_symmetry_deg2_vs_deg8'] = float((c1_path(deg2, []) - c1_path(deg8, [1, 2, 3, 5, 6, 7])).abs().max())

# Tolerances: a value outside its bound fails the run (exit 1), so `set -euo pipefail` stops before training.
# Monte Carlo bounds are several standard errors (N = 2^16: loss ~1e-2 relative, coefficients sd ~0.004).
TOL = {'core_contraction_vs_dense_T': ('<', 1e-12), 'autograd_vs_finite_difference': ('<', 1e-6),
       'teacher_cores_reproduce_targets_max_err': ('<', 1e-12), 'mobility_groups_sum_to_full': ('<', 1e-12),
       'first_order_dc_rel_err': ('<', 1e-3), 'frozen_q1_L_moved': ('<', 1e-15), 'unfrozen_run_moved_q1': ('>', 1e-6),
       'frozen_groups_K_abs': ('<', 1e-15),
       'velocity_vs_grad_inner_product': ('<', 1e-12), 'frozen_groups_velocity_abs': ('<', 1e-15), 'masked_output_target_changes_grad': ('<', 1e-15),
       'alignment_at_teacher_min': ('>', 0.999), 'alignment_unrelated_feature_q1': ('<', 0.02),
       'gram_vs_dense_T_norm': ('<', 1e-12), 'population_vs_mc_loss_max_rel': ('<', 0.05),
       'population_vs_mc_coeff_max_abs': ('<', 0.02), 'population_at_teacher_coeff_err': ('<', 1e-12),
       'population_at_teacher_loss': ('<', 1e-12), 'permutation_symmetry_deg2_vs_deg8': ('<', 1e-12),
       **{f'gram_minus_identity_{f}': ('<', 0.03) for f in teachers.SETS}}
for r in (8, 16):
    TOL[f'init_m2_r{r}_alpha1.0_leaf'] = ('mean in', (0.8, 1.25))
    TOL[f'init_m2_r{r}_alpha1.0_quarter'] = ('mean in', (0.8, 1.25))
failed = []
for k, v in out.items():
    rule = TOL.get(k)
    ok = True
    if rule:
        op, b = rule
        ok = (v < b) if op == '<' else (v > b) if op == '>' else (b[0] <= v[0] <= b[1])
        if not ok:
            failed.append(k)
    print(f'{k:44s}', v if isinstance(v, list) else f'{v:.3g}', '' if rule is None else ('ok' if ok else f'FAIL {rule}'))
io.save_json(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'checks.json', dict(out, failed=failed))
if failed:
    sys.exit(f'checks failed: {failed}')
print('all checks passed')
