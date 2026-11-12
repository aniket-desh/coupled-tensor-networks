"""Checks for the hierarchy model. Each targets an error that would corrupt a measured ordering or overshoot."""
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

import hierarchy as h
import model as m

rng = np.random.default_rng(0)
s = np.array([1.0, 0.3, 0.8, 0.25])
out = {}


def fd(th, f, g, step=1e-6):
    num = np.array([(f(th + step * e) - f(th - step * e)) / (2 * step) for e in np.eye(th.size)])
    return np.abs(num - g(th)).max()


# 1. Dense 58-parameter gradient against centered differences, and the dense independent copy.
target = h.d_target(s)
th = 0.6 * rng.standard_normal(58)
out['dense_grad_fd'] = fd(th, lambda y: 0.5 * ((h.d_tensor(y, 4) - target) ** 2).sum(), lambda y: h.d_grad(y, target))
t1 = h.d_target(s, [2])
th = 0.6 * rng.standard_normal(34)
out['dense_single_output_grad_fd'] = fd(th, lambda y: 0.5 * ((h.d_tensor(y, 1) - t1) ** 2).sum(), lambda y: h.d_grad(y, t1))

# 2. Teacher's four rank-one components are orthonormal directions, and the population loss claim holds.
units = [h.d_target(np.eye(4)[k]) for k in range(4)]
gram = np.array([[(a * b).sum() for b in units] for a in units])
out['teacher_components_orthonormal'] = np.abs(gram - np.eye(4)).max()

# 3. Factor flow: gradient against differences in raw parameters, on all four arms.
for name in ['hierarchy', 'u_only', 'group_only', 'independent']:
    arm = h.Arm(name)
    x = rng.uniform(-1, 0.5, arm.A.shape[1])
    L = lambda th_: 0.5 * ((s - np.exp(arm.A @ np.log(th_))) ** 2).sum()
    num = np.array([(L(np.exp(x) * (1 + 1e-6 * e)) - L(np.exp(x) * (1 - 1e-6 * e))) / (2e-6 * np.exp(x[j])) for j, e in enumerate(np.eye(x.size))])
    out[f'factor_flow_fd_{name}'] = np.abs(-num / np.exp(x) - arm.rhs(x, s)).max()

# 4. Cross-mobility from dense parameter gradients equals the closed K at an aligned balanced point,
#    so "siblings share 3 cores, cousins 1" is checked on the real tensor rather than assumed.
th = h.d_aligned(0.5)
E = [h.d_target(np.eye(4)[k]) for k in range(4)]
grads = []
for k in range(4):
    # grad of c_k is the gradient of <T, E_k>, obtainable as d_grad of 1/2||T - T*||^2 at T* = T - E_k.
    grads.append(h.d_grad(th, h.d_tensor(th, 4) - E[k]))
Kd = np.array([[a @ b for b in grads] for a in grads])
cd = h.d_coeffs(h.d_tensor(th, 4))
out['K_dense_vs_closed'] = np.abs(Kd - h.K_matrix(cd)).max()
a, q2, p = h.balanced_state(cd)
out['K12_over_K13_expected_ratio_error'] = abs((Kd[0, 1] / Kd[0, 2]) - (1 + 2 * a ** 2 / q2[0]))

# 5. Exact reduction: dense flow, 13-factor flow and the 4D K equation must be one trajectory.
delta = 0.2
t_a, c_a, _ = h.run_arm(h.Arm('hierarchy'), s, delta)
t_r, c_r, _ = h.run_reduced(np.exp(h.Arm('hierarchy').A @ h.Arm('hierarchy').x0(delta)), s)
th0 = h.d_aligned(delta)
t_d, c_d, _ = h.run_dense(th0, target, stop_loss=1e-16)
grid = np.linspace(0, min(t_a[-1], t_r[-1], t_d[-1]), 3000)
ci = lambda t, c: np.array([np.interp(grid, t, ck) for ck in c])
out['factor_vs_reduced_max_diff'] = np.abs(ci(t_a, c_a) - ci(t_r, c_r)).max()
out['factor_vs_dense_max_diff'] = np.abs(ci(t_a, c_a) - ci(t_d, c_d)).max()
out['half_times_factor'] = h.half_times(t_a, c_a, s).tolist()
out['half_times_dense'] = h.half_times(t_d, c_d, s).tolist()

# 6. Conserved balances along the factor flow (they define the balanced state the reduction uses).
arm = h.Arm('hierarchy')
_, x, _ = h.integrate(lambda y: arm.rhs(y, s), arm.x0(0.2), lambda y: 0.5 * ((s - arm.coeffs(y)) ** 2).sum() < 1e-14)
th_ = np.exp(x)
b = np.array([th_[0] ** 2 - th_[3] ** 2 - th_[4] ** 2, th_[1] ** 2 - th_[3] ** 2, th_[2] ** 2 - th_[4] ** 2,
              th_[3] ** 2 - th_[5] ** 2 - th_[6] ** 2, th_[4] ** 2 - th_[7] ** 2 - th_[8] ** 2,
              *(th_[5 + k] ** 2 - th_[9 + k] ** 2 for k in range(4))])
out['balances_max_drift'] = np.abs(b).max()

# 7. Independent reference: the Arm machinery with a 2-component, 3-factor pattern must give the
#    published overshoot for s = (1, 0.2), delta = 0.03 (note sec 25; also model.py).
arm2 = h.Arm('independent')
arm2.A = np.zeros((2, 5))
arm2.A[0, [0, 1, 2]] = 1
arm2.A[1, [0, 3, 4]] = 1
arm2.amp = np.array([np.sqrt(2), 1, 1, 1, 1.0])
s2 = np.array([1.0, 0.2])
t2, c2, _ = h.run_arm(arm2, s2, 0.03)
out['two_component_overshoot_pct (0.3104)'] = 100 * h.overshoot(c2, s2)[0]
out['two_component_half_times (27.9670, 37.8080)'] = h.half_times(t2, c2, s2).tolist()

# 8. Independent arm against the scalar depth-5 law: cdot = 5 c^(8/5) (s - c), half-time in closed form limit.
t_i, c_i, _ = h.run_arm(h.Arm('independent'), s, 0.05)
sol = h.integrate(lambda x: 5 * np.exp(x) ** (3 / 5) * (s - np.exp(x)), np.full(4, 5 * np.log(0.05)), lambda x: 0.5 * ((s - np.exp(x)) ** 2).sum() < 1e-16)
out['independent_vs_scalar_law_half_time_rel_diff'] = np.abs(h.half_times(t_i, c_i, s) / h.half_times(sol[0], np.exp(sol[1]), s) - 1).max()
out['independent_overshoot'] = h.overshoot(c_i, s).max()

# 9. Tolerance sensitivity of the delta = 0.001 ordering, the most demanding quantity in the sweep.
base = {}
for rt, at in [(1e-12, 1e-13), (1e-13, 1e-14)]:
    h.RTOL, h.ATOL = rt, at
    t, c, _ = h.run_arm(h.Arm('hierarchy'), s, 0.001)
    base[rt] = h.half_times(t, c, s)
h.RTOL, h.ATOL = 1e-12, 1e-13
out['delta_1e-3_half_time_abs_diff_tol'] = np.abs(base[1e-12] - base[1e-13]).max()
out['delta_1e-3_half_times'] = base[1e-13].tolist()

# 10. Radau, used only to rerun runs that exhausted DOP853's budget: its Jacobian, and agreement with DOP853.
for name in ['hierarchy', 'independent']:
    arm = h.Arm(name)
    x = rng.uniform(-1, 0.3, arm.A.shape[1])
    num = np.array([(arm.rhs(x + 1e-6 * e, s) - arm.rhs(x - 1e-6 * e, s)) / 2e-6 for e in np.eye(x.size)]).T
    out[f'jac_fd_{name}'] = np.abs(num - arm.jac(x, s)).max()
t_e, c_e, _ = h.run_arm(h.Arm('hierarchy'), s, .01)
t_i, c_i, _ = h.run_arm(h.Arm('hierarchy'), s, .01, stiff=True)
out['radau_vs_dop853_half_time_abs_diff'] = np.abs(h.half_times(t_e, c_e, s) - h.half_times(t_i, c_i, s)).max()
out['radau_vs_dop853_overshoot_abs_diff'] = np.abs(h.overshoot(c_e, s) - h.overshoot(c_i, s)).max()

for k, v in out.items():
    print(f'{k:48s}', np.round(v, 6).tolist() if isinstance(v, list) else f'{v:.6g}')
json.dump(out, open('results/h_checks.json', 'w'), indent=1, default=float)
