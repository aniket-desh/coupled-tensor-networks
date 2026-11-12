"""Checks for the hierarchy model. Each targets an error that would corrupt a measured ordering or overshoot.
Usage: python -m studies.hierarchy.checks  (writes results/checks.json)"""
from pathlib import Path

import numpy as np

from ctn import io
from ctn.factors import Pattern
from ctn.integrate import capped
from ctn.measure import half_times, loss, overshoot

from .model import ARMS, NET, K_matrix, balanced_state, run_dense, run_reduced, single_output_net

rng = np.random.default_rng(0)
s = np.array([1.0, 0.3, 0.8, 0.25])
out = {}


def fd(th, f, g, step=1e-6):
    num = np.array([(f(th + step * e) - f(th - step * e)) / (2 * step) for e in np.eye(th.size)])
    return np.abs(num - g(th)).max()


# 1. Dense 58-parameter gradient against centered differences, and the dense independent copy.
target = NET.teacher(s)
out['dense_grad_fd'] = fd(NET.random(0.6, rng), lambda y: NET.loss(y, target), lambda y: NET.grad(y, target))
net1 = single_output_net(2)
t1 = net1.teacher(s[[2]])
out['dense_single_output_grad_fd'] = fd(net1.random(0.6, rng), lambda y: net1.loss(y, t1), lambda y: net1.grad(y, t1))

# 2. Teacher's four rank-one components are orthonormal directions, and the population loss claim holds.
units = [NET.teacher(np.eye(4)[k]) for k in range(4)]
out['teacher_components_orthonormal'] = np.abs(np.array([[(a * b).sum() for b in units] for a in units]) - np.eye(4)).max()

# 3. Factor flow: gradient against differences in raw parameters, on all four arms.
for name, arm in ARMS.items():
    x = rng.uniform(-1, 0.5, arm.A.shape[1])
    L = lambda th_: loss(np.exp(arm.A @ np.log(th_)), s)
    num = np.array([(L(np.exp(x) * (1 + 1e-6 * e)) - L(np.exp(x) * (1 - 1e-6 * e))) / (2e-6 * np.exp(x[j])) for j, e in enumerate(np.eye(x.size))])
    out[f'factor_flow_fd_{name}'] = np.abs(-num / np.exp(x) - arm.rhs(x, s)).max()

# 4. Cross-mobility from dense parameter gradients equals the closed K at an aligned balanced point,
#    so "siblings share 3 cores, cousins 1" is checked on the real tensor rather than assumed.
th = NET.aligned(0.5)
grads = [NET.pull_back(th, units[k]) for k in range(4)]  # grad of c_k = <T, unit_k>
Kd = np.array([[a @ b for b in grads] for a in grads])
cd = NET.coeffs(th)
out['K_dense_vs_closed'] = np.abs(Kd - K_matrix(cd)).max()
a, q2, p = balanced_state(cd)
out['K12_over_K13_expected_ratio_error'] = abs((Kd[0, 1] / Kd[0, 2]) - (1 + 2 * a ** 2 / q2[0]))
# The aligned sector derived from the einsum tree is the hand-written 'hierarchy' arm (factor order aside).
pat = NET.pattern()
out['network_pattern_vs_arm_mobility'] = np.abs(pat.mobility(pat.x0(0.5)) - ARMS['hierarchy'].mobility(ARMS['hierarchy'].x0(0.5))).max()

# 5. Exact reduction: dense flow, 13-factor flow and the 4D K equation must be one trajectory.
delta, arm = 0.2, ARMS['hierarchy']
t_a, c_a, _ = arm.run(s, delta)
t_r, c_r, _ = run_reduced(arm.coeffs(arm.x0(delta)), s)
t_d, c_d, _ = run_dense(NET, NET.aligned(delta), s, stop_loss=1e-16)
grid = np.linspace(0, min(t_a[-1], t_r[-1], t_d[-1]), 3000)
ci = lambda t, c: np.array([np.interp(grid, t, ck) for ck in c])
out['factor_vs_reduced_max_diff'] = np.abs(ci(t_a, c_a) - ci(t_r, c_r)).max()
out['factor_vs_dense_max_diff'] = np.abs(ci(t_a, c_a) - ci(t_d, c_d)).max()
out['half_times_factor'] = half_times(t_a, c_a, s).tolist()
out['half_times_dense'] = half_times(t_d, c_d, s).tolist()

# 6. Conserved balances along the factor flow (they define the balanced state the reduction uses).
_, x, _ = capped(lambda y: arm.rhs(y, s), arm.x0(0.2), lambda y: loss(arm.coeffs(y), s) < 1e-14)
v = {lab: np.exp(x[arm.index(lab)]) ** 2 for lab in arm.labels}
b = [v['u'] - v['p0'] - v['p1'], v['f0'] - v['p0'], v['f1'] - v['p1'], v['p0'] - v['r0'] - v['r1'], v['p1'] - v['r2'] - v['r3'],
     *(v[f'r{k}'] - v[f'b{k}'] for k in range(4))]
out['balances_max_drift'] = np.abs(b).max()

# 7. Independent reference: a 2-component, 3-factor pattern must give the published overshoot for
#    s = (1, 0.2), delta = 0.03 (note sec 25; also the shared_feature study).
s2 = np.array([1.0, 0.2])
t2, c2, _ = Pattern.from_paths([('a', 'r1', 'b1'), ('a', 'r2', 'b2')]).run(s2, 0.03)
out['two_component_overshoot_pct (0.3104)'] = 100 * overshoot(c2, s2)[0]
out['two_component_half_times (27.9670, 37.8080)'] = half_times(t2, c2, s2).tolist()

# 8. Independent arm against the scalar depth-5 law: cdot = 5 c^(8/5) (s - c).
t_i, c_i, _ = ARMS['independent'].run(s, 0.05)
t_l, x_l, _ = capped(lambda x: 5 * np.exp(x) ** (3 / 5) * (s - np.exp(x)), np.full(4, 5 * np.log(0.05)), lambda x: loss(np.exp(x), s) < 1e-16)
out['independent_vs_scalar_law_half_time_rel_diff'] = np.abs(half_times(t_i, c_i, s) / half_times(t_l, np.exp(x_l), s) - 1).max()
out['independent_overshoot'] = overshoot(c_i, s).max()

# 9. Tolerance sensitivity of the delta = 0.001 ordering, the most demanding quantity in the sweep.
base = {}
for rt, at in [(1e-12, 1e-13), (1e-13, 1e-14)]:
    t, c, _ = arm.run(s, 0.001, rtol=rt, atol=at)
    base[rt] = half_times(t, c, s)
out['delta_1e-3_half_time_abs_diff_tol'] = np.abs(base[1e-12] - base[1e-13]).max()
out['delta_1e-3_half_times'] = base[1e-13].tolist()

# 10. Radau, used only to rerun runs that exhausted DOP853's budget: its Jacobian, and agreement with DOP853.
for name in ['hierarchy', 'independent']:
    x = rng.uniform(-1, 0.3, ARMS[name].A.shape[1])
    num = np.array([(ARMS[name].rhs(x + 1e-6 * e, s) - ARMS[name].rhs(x - 1e-6 * e, s)) / 2e-6 for e in np.eye(x.size)]).T
    out[f'jac_fd_{name}'] = np.abs(num - ARMS[name].jac(x, s)).max()
t_e, c_e, _ = arm.run(s, .01)
t_s, c_s, _ = arm.run(s, .01, stiff=True)
out['radau_vs_dop853_half_time_abs_diff'] = np.abs(half_times(t_e, c_e, s) - half_times(t_s, c_s, s)).max()
out['radau_vs_dop853_overshoot_abs_diff'] = np.abs(overshoot(c_e, s) - overshoot(c_s, s)).max()

for k, v in out.items():
    print(f'{k:48s}', np.round(v, 6).tolist() if isinstance(v, list) else f'{v:.6g}')
io.save_json(Path(__file__).parent / 'results' / 'checks.json', out)
