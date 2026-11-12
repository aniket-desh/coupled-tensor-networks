"""Correctness checks, each aimed at an error that would corrupt an overshoot measurement."""
import json

import numpy as np

import model as m

rng = np.random.default_rng(0)
s = np.array([1.0, 0.2])
out = {}


def fd_error(theta, f, g, h=1e-6):
    fd = np.array([(f(theta + h * e) - f(theta - h * e)) / (2 * h) for e in np.eye(theta.size)])
    return np.abs(fd - g(theta)).max()


# 1. Gradients against centered differences, including the moment-tensor loss.
x1, x2 = rng.standard_normal((64, 2)), rng.standard_normal((64, 2))
z = np.einsum('ni,nj->nij', x1, x2).reshape(64, 4)
G4 = z.T @ z / 64
for name, shared in [('shared', True), ('unshared', False)]:
    th = m.init_random(0.7, rng, shared)
    out[f'grad_fd_population_{name}'] = fd_error(th, lambda t: m.loss(t, s), lambda t: m.grad(t, s))
    out[f'grad_fd_moments_{name}'] = fd_error(th, lambda t: m.loss(t, s, G4), lambda t: m.grad(t, s, G4))
    # The sample-wise minibatch gradient and the moment gradient are the same object.
    out[f'grad_batch_vs_moments_{name}'] = np.abs(m.grad_batch(th, s, x1, x2) - m.grad(th, s, G4)).max()

# 2. Dense tensor loss against the note's closed form, and pull_back against its boxed flow.
th = m.init_random(0.7, rng, True)
U, R, B = m.unpack(th)
u, M, S, g = U[0], R @ B, np.diag(s), np.array([1.0, 0.0])
a, n = g @ u, u @ u
closed = 0.5 * (n * (M * M).sum() - 2 * a * (S * M).sum() + (S * S).sum())
out['loss_closed_form'] = abs(closed - m.loss(th, s))
E = a * S - n * M
boxed = m.pack((g * (S * M).sum() - (M * M).sum() * u)[None], E @ B.T, R.T @ E)
out['flow_vs_note_box'] = np.abs(boxed + m.grad(th, s)).max()

# 3. Large-sample moments recover the population loss (whitening assumption of the note).
x1, x2 = rng.standard_normal((2_000_000, 2)), rng.standard_normal((2_000_000, 2))
z = np.einsum('ni,nj->nij', x1, x2).reshape(-1, 4)
out['moments_minus_identity_2e6_samples'] = np.abs(z.T @ z / len(z) - np.eye(4)).max()

# 4. The independent reference: reproduce the numbers printed in the note (sec 25).
delta = 0.03
sol_red = m.solve_reduced(m.c0_balanced(delta), s, t_max=85)
sol_full = m.flow(m.init_aligned(s, delta), s, 85)
t = m.refine(sol_red)
c_red = np.exp(sol_red.sol(t))
c_full = np.array([m.coeffs(y) for y in sol_full.sol(t).T]).T
out['full_vs_reduced_max_diff'] = np.abs(c_red - c_full).max()
out['note_half_time_1 (27.9670)'] = m.half_time(t, c_red[0], s[0])
out['note_half_time_2 (37.8080)'] = m.half_time(t, c_red[1], s[1])
out['note_overshoot_pct (0.3104)'] = 100 * m.overshoot(c_red, s)[0]

# 5. Control has zero cross-mobility; shared model's <grad c1, grad c2> = M11 M22 anywhere.
E1, E2 = np.zeros((2, 2, 2)), np.zeros((2, 2, 2))
E1[0, 0, 0], E2[1, 0, 1] = 1, 1
th = m.init_random(0.7, rng, True)
U, R, B = m.unpack(th)
M = R @ B
out['cross_mobility_shared_identity'] = abs(m.pull_back(E1, U, R, B) @ m.pull_back(E2, U, R, B) - M[0, 0] * M[1, 1])
th = m.init_random(0.7, rng, False)
out['cross_mobility_control_is_zero'] = abs(m.pull_back(E1, *m.unpack(th)) @ m.pull_back(E2, *m.unpack(th)))

# 6. Reduced control is the unshared parameter flow on the aligned sector.
sol_red = m.solve_reduced(np.full(2, delta ** 3), s, m.rhs_independent, t_max=300)
sol_full = m.flow(m.init_aligned(s, delta, shared=False), s, 300)
print('control end times (solver stops on convergence):', sol_red.t[-1], sol_full.t[-1])
t = np.linspace(0, min(sol_red.t[-1], sol_full.t[-1]), 2000)  # never evaluate past the solved interval
c_red = np.exp(sol_red.sol(t))
c_full = np.array([m.coeffs(y) for y in sol_full.sol(t).T]).T
out['control_full_vs_reduced_max_diff'] = np.abs(c_red - c_full).max()
out['control_overshoot'] = m.overshoot(c_red, s).max()

# 7. Scale invariance used to fix s1 = 1: (c, s, t) -> (lam c, lam s, t/lam^(4/3)) with delta -> lam^(1/3) delta.
lam = 8.0
base = m.solve_reduced(m.c0_balanced(0.03), np.array([1.0, 0.5]))
scaled = m.solve_reduced(m.c0_balanced(0.03 * lam ** (1 / 3)), lam * np.array([1.0, 0.5]))
out['scale_invariance_overshoot_diff'] = abs(m.overshoot(np.exp(base.sol(m.refine(base))), np.array([1.0, 0.5]))[0]
                                             - m.overshoot(np.exp(scaled.sol(m.refine(scaled))), lam * np.array([1.0, 0.5]))[0])
tb = m.half_time(m.refine(base), np.exp(base.sol(m.refine(base)))[0], 1.0)
ts = m.half_time(m.refine(scaled), np.exp(scaled.sol(m.refine(scaled)))[0], lam)
out['scale_invariance_time_ratio (expect 16)'] = tb / ts

# 8. Optimizers. Imported here because experiments.py holds the step used in the sweep.
from experiments import opt_step

# Momentum 0 is plain GD on the real model, exactly.
th, th_gd, st = m.init_random(0.3, rng, True), None, {}
th_gd = th.copy()
for _ in range(50):
    th = opt_step(th, m.grad(th, s), st, ('hb', 0.01, 0.0))
    th_gd = th_gd - 0.01 * m.grad(th_gd, s)
out['hb_beta0_equals_gd'] = np.abs(th - th_gd).max()
# Heavy ball on x -> K x^2 / 2 against the closed form of x_(n+1) = (1 + beta - eta K) x_n - beta x_(n-1).
K_, eta_, beta_ = 3.0, 0.01, 0.9
roots = np.roots([1, -(1 + beta_ - eta_ * K_), beta_])
x, st = np.array([1.0]), {}
xs = [1.0]
for _ in range(200):
    x = opt_step(x, K_ * x, st, ('hb', eta_, beta_))
    xs.append(x[0])
A = np.linalg.solve(np.array([[1, 1], roots]), np.array([xs[0], xs[1]], complex))
closed = (A[0] * roots[0] ** np.arange(201) + A[1] * roots[1] ** np.arange(201)).real
out['hb_quadratic_vs_closed_form'] = np.abs(closed - np.array(xs)).max()
out['hb_complex_roots_iff_eta_K_above_line'] = float((np.abs(roots.imag).max() > 0) == (eta_ * K_ > (1 - np.sqrt(beta_)) ** 2)) - 1
# Adam against two steps worked by hand with g1 = 2, g2 = 1, lr = 0.1, b1 = 0.5, b2 = 0.5:
# step 1: m_hat = 2, w_hat = 4 -> dx = -0.1
# step 2: m = 0.5*1 + 0.5*1 = 1.0, m_hat = 1/0.75; w = 0.5*2 + 0.5*1 = 1.5, w_hat = 1.5/0.75 = 2 -> dx = -0.1*(4/3)/sqrt(2)
x, st = np.array([0.0]), {}
x = opt_step(x, np.array([2.0]), st, ('adam', 0.1, 0.5, 0.5))
x1 = x[0]
x = opt_step(x, np.array([1.0]), st, ('adam', 0.1, 0.5, 0.5))
out['adam_hand_step_error'] = abs(x1 + 0.1 * 2 / (2 + 1e-8)) + abs(x[0] - x1 + 0.1 * (4 / 3) / (np.sqrt(2) + 1e-8))
# Adam keeps the aligned sector closed: zero entries get zero gradient and stay zero.
th, st = m.init_aligned(s, 0.03), {}
for _ in range(500):
    th = opt_step(th, m.grad(th, s), st, ('adam', 1e-3, 0.9, 0.999))
U_, R_, B_ = m.unpack(th)
out['adam_aligned_sector_offdiag'] = max(abs(U_[0, 1]), abs(R_[0, 1]), abs(R_[1, 0]), abs(B_[0, 1]), abs(B_[1, 0]))

for k, v in out.items():
    print(f'{k:42s} {v:.6g}')
json.dump({k: float(v) for k, v in out.items()}, open('results/checks.json', 'w'), indent=1)
