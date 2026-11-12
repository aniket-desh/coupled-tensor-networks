"""Correctness checks, each aimed at an error that would corrupt an overshoot measurement.
Usage: python -m studies.shared_feature.checks  (writes results/checks.json)"""
from pathlib import Path

import numpy as np

from ctn import io, optim
from ctn.measure import half_time, overshoot
from ctn.network import features

from .model import (CONTROL, K_INDEPENDENT, SHARED, TARGET, K_matrix, c0_balanced, flow, reduced_c, solve_reduced)

rng = np.random.default_rng(0)
s = np.array([1.0, 0.2])
T = TARGET(s)
out = {}


def fd_error(theta, f, g, h=1e-6):
    fd = np.array([(f(theta + h * e) - f(theta - h * e)) / (2 * h) for e in np.eye(theta.size)])
    return np.abs(fd - g(theta)).max()


# 1. Gradients against centered differences, including the moment-tensor loss.
x1, x2 = rng.standard_normal((64, 2)), rng.standard_normal((64, 2))
z = features([x1, x2])
G4 = z.T @ z / 64
for name, net in [('shared', SHARED), ('unshared', CONTROL)]:
    th = net.random(0.7, rng)
    out[f'grad_fd_population_{name}'] = fd_error(th, lambda t: net.loss(t, T), lambda t: net.grad(t, T))
    out[f'grad_fd_moments_{name}'] = fd_error(th, lambda t: net.loss(t, T, G4), lambda t: net.grad(t, T, G4))
    # The sample-wise minibatch gradient and the moment gradient are the same object.
    out[f'grad_batch_vs_moments_{name}'] = np.abs(net.grad_batch(th, T, [x1, x2]) - net.grad(th, T, G4)).max()

# 2. Dense tensor loss against the note's closed form, and the gradient against its boxed flow.
th = SHARED.random(0.7, rng)
u, R, B = SHARED.unpack(th)
M, S, g = R @ B, np.diag(s), np.array([1.0, 0.0])
a, n = g @ u, u @ u
closed = 0.5 * (n * (M * M).sum() - 2 * a * (S * M).sum() + (S * S).sum())
out['loss_closed_form'] = abs(closed - SHARED.loss(th, T))
E = a * S - n * M
boxed = SHARED.pack([g * (S * M).sum() - (M * M).sum() * u, E @ B.T, R.T @ E])
out['flow_vs_note_box'] = np.abs(boxed + SHARED.grad(th, T)).max()

# 3. Large-sample moments recover the population loss (whitening assumption of the note).
z = features([rng.standard_normal((2_000_000, 2)), rng.standard_normal((2_000_000, 2))])
out['moments_minus_identity_2e6_samples'] = np.abs(z.T @ z / len(z) - np.eye(4)).max()

# 4. The independent reference: reproduce the numbers printed in the note (sec 25).
delta = 0.03
sol_red = solve_reduced(c0_balanced(delta), s, t_max=85)
sol_full = flow(SHARED, SHARED.aligned(delta), s, 85)
t, c_red = reduced_c(sol_red)
out['full_vs_reduced_max_diff'] = np.abs(c_red - SHARED.coeffs(sol_full.sol(t))).max()
out['note_half_time_1 (27.9670)'] = half_time(t, c_red[0], s[0])
out['note_half_time_2 (37.8080)'] = half_time(t, c_red[1], s[1])
out['note_overshoot_pct (0.3104)'] = 100 * overshoot(c_red, s)[0]

# 5. Control has zero cross-mobility; the shared model's <grad c1, grad c2> = M11 M22 anywhere.
E1, E2 = np.zeros((2, 2, 2)), np.zeros((2, 2, 2))
E1[0, 0, 0], E2[1, 0, 1] = 1, 1
th = SHARED.random(0.7, rng)
_, R, B = SHARED.unpack(th)
M = R @ B
out['cross_mobility_shared_identity'] = abs(SHARED.pull_back(th, E1) @ SHARED.pull_back(th, E2) - M[0, 0] * M[1, 1])
th = CONTROL.random(0.7, rng)
out['cross_mobility_control_is_zero'] = abs(CONTROL.pull_back(th, E1) @ CONTROL.pull_back(th, E2))

# 6. Reduced control is the unshared parameter flow on the aligned sector.
sol_red = solve_reduced(np.full(2, delta ** 3), s, K_INDEPENDENT, t_max=300)
sol_full = flow(CONTROL, CONTROL.aligned(delta), s, 300)
t = np.linspace(0, min(sol_red.t[-1], sol_full.t[-1]), 2000)  # never evaluate past the solved interval
c_red = np.exp(sol_red.sol(t))
out['control_full_vs_reduced_max_diff'] = np.abs(c_red - CONTROL.coeffs(sol_full.sol(t))).max()
out['control_overshoot'] = overshoot(c_red, s).max()

# 7. Scale invariance used to fix s1 = 1: (c, s, t) -> (lam c, lam s, t/lam^(4/3)) with delta -> lam^(1/3) delta.
lam, sb = 8.0, np.array([1.0, 0.5])
tb, cb = reduced_c(solve_reduced(c0_balanced(0.03), sb))
ts, cs = reduced_c(solve_reduced(c0_balanced(0.03 * lam ** (1 / 3)), lam * sb))
out['scale_invariance_overshoot_diff'] = abs(overshoot(cb, sb)[0] - overshoot(cs, lam * sb)[0])
out['scale_invariance_time_ratio (expect 16)'] = half_time(tb, cb[0], 1.0) / half_time(ts, cs[0], lam)

# 8. Closed-form K against the factor-flow mobility of the network's own aligned sector, at a balanced point.
#    This ties model.K_matrix (used by every reduced run) to the einsum network, not to a hand derivation.
pat = SHARED.pattern()
x = pat.x0(0.4)
out['K_closed_vs_network_mobility'] = np.abs(K_matrix(pat.coeffs(x)) - pat.mobility(x)).max()
out['aligned_init_vs_c0_balanced'] = np.abs(SHARED.coeffs(SHARED.aligned(0.4)) - c0_balanced(0.4)).max()

# 9. Optimizers. Momentum 0 is plain GD on the real model, exactly.
th, st = SHARED.random(0.3, rng), {}
th_gd = th.copy()
for _ in range(50):
    th = optim.step(th, SHARED.grad(th, T), st, ('hb', 0.01, 0.0))
    th_gd = th_gd - 0.01 * SHARED.grad(th_gd, T)
out['hb_beta0_equals_gd'] = np.abs(th - th_gd).max()
# Heavy ball on x -> K x^2 / 2 against the closed form of x_(n+1) = (1 + beta - eta K) x_n - beta x_(n-1).
K_, eta_, beta_ = 3.0, 0.01, 0.9
roots = np.roots([1, -(1 + beta_ - eta_ * K_), beta_])
x, st = np.array([1.0]), {}
xs = [1.0]
for _ in range(200):
    x = optim.step(x, K_ * x, st, ('hb', eta_, beta_))
    xs.append(x[0])
A = np.linalg.solve(np.array([[1, 1], roots]), np.array([xs[0], xs[1]], complex))
closed = (A[0] * roots[0] ** np.arange(201) + A[1] * roots[1] ** np.arange(201)).real
out['hb_quadratic_vs_closed_form'] = np.abs(closed - np.array(xs)).max()
out['hb_complex_roots_iff_eta_K_above_line'] = float((np.abs(roots.imag).max() > 0) == (eta_ * K_ > (1 - np.sqrt(beta_)) ** 2)) - 1
# Adam against two steps worked by hand with g1 = 2, g2 = 1, lr = 0.1, b1 = 0.5, b2 = 0.5:
# step 1: m_hat = 2, w_hat = 4 -> dx = -0.1
# step 2: m = 0.5*1 + 0.5*1 = 1.0, m_hat = 1/0.75; w = 0.5*2 + 0.5*1 = 1.5, w_hat = 1.5/0.75 = 2 -> dx = -0.1*(4/3)/sqrt(2)
x, st = np.array([0.0]), {}
x = optim.step(x, np.array([2.0]), st, ('adam', 0.1, 0.5, 0.5))
x1 = x[0]
x = optim.step(x, np.array([1.0]), st, ('adam', 0.1, 0.5, 0.5))
out['adam_hand_step_error'] = abs(x1 + 0.1 * 2 / (2 + 1e-8)) + abs(x[0] - x1 + 0.1 * (4 / 3) / (np.sqrt(2) + 1e-8))
# Adam keeps the aligned sector closed: zero entries get zero gradient and stay zero.
th, st = SHARED.aligned(0.03), {}
for _ in range(500):
    th = optim.step(th, SHARED.grad(th, T), st, ('adam', 1e-3, 0.9, 0.999))
u_, R_, B_ = SHARED.unpack(th)
out['adam_aligned_sector_offdiag'] = max(abs(u_[1]), abs(R_[0, 1]), abs(R_[1, 0]), abs(B_[0, 1]), abs(B_[1, 0]))

for k, v in out.items():
    print(f'{k:42s} {v:.6g}')
io.save_json(Path(__file__).parent / 'results' / 'checks.json', {k: float(v) for k, v in out.items()})
