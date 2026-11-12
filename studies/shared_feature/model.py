"""Shared-feature three-core TTN from the Zulip note, plus its no-sharing control.

Tensor:  T_oij = U_oi M_oj,  M = R B.

U is a single left feature u (shape (2,), broadcast over outputs) in the shared model, the note's model with
10 parameters, and has shape (2, 2) in the control, where each output owns its left feature (12 parameters).
That is the only difference, so a gap between them is attributable to sharing u.

Teacher: T*_oij = g_i S_oj with S = diag(s1, s2). Whitened inputs are rotation invariant, so g = (1, 0) and
e_k = h_k = standard basis lose no generality. Coefficients are the projections c_k = <T, e_k x g x h_k>, which
are well defined at arbitrary (misaligned) parameters.

Balanced aligned sector: c_k = a r_k b_k with a^2 = r1^2 + r2^2 and r_k = b_k, so a = (c1 + c2)^(1/3) and

    dc/dt = K(c) (s - c),   K(c) = 2a diag(c) + c c^T / a^2.

The flow is scale invariant: (c, s, t) -> (lam c, lam s, t / lam^(4/3)) with delta -> lam^(1/3) delta, so the
experiments fix s1 = 1 and vary rho = s2/s1.
"""
import numpy as np

from ctn import integrate
from ctn.factors import K_independent, log_flow
from ctn.measure import loss, overshoot
from ctn.network import Network

COMPONENTS = [dict(o=k, i=0, a=k, j=k) for k in range(2)]
DIMS = dict(o=2, i=2, a=2, j=2)
SHARED = Network('i,oa,aj->oij', DIMS, COMPONENTS)
CONTROL = Network('oi,oa,aj->oij', DIMS, COMPONENTS)
TARGET = lambda s: SHARED.teacher(s)  # same tensor for both networks


def net(shared=True):
    return SHARED if shared else CONTROL


def S(rho):
    return np.array([1.0, rho])


def flow(network, theta0, s, t_max, stop_loss=1e-18, fun=None, events=()):
    """Population gradient flow (or `fun`) on the full parameters until 1/2||T - T*||^2 < stop_loss."""
    T = TARGET(s)
    fun = fun or (lambda y: -network.grad(y, T))
    return integrate.solve(fun, theta0, t_max, done=lambda y: network.loss(y, T) - stop_loss, events=events)


# ------------------------------------------------------------ reduced equations in x = log c
def K_matrix(c):
    a = c.sum() ** (1 / 3)
    return 2 * a * np.diag(c) + np.outer(c, c) / a ** 2


K_INDEPENDENT = K_independent(3)  # the control's aligned sector: dc_k/dt = 3 c_k^(4/3) e_k


def solve_reduced(c0, s, K=K_matrix, t_max=1e8, stop_loss=1e-18, max_step=0.5):
    rhs = log_flow(K)
    return integrate.solve(lambda x: rhs(x, s), np.log(c0), t_max, done=lambda x: loss(np.exp(x), s) - stop_loss,
                           max_step=max_step)


def reduced_c(sol, sub=20):
    """Refined time grid and coefficients of a reduced solution."""
    t = integrate.refine(sol, sub)
    return t, np.exp(sol.sol(t))


def c0_balanced(delta):
    """Shared aligned init: a = sqrt(2) delta, r_k = b_k = delta."""
    return np.full(2, np.sqrt(2) * delta ** 3)


def heteroclinic_overshoot(rho, seed=1e-10):
    """Small-init limit: start next to the saddle (s1, 0) and follow it to (s1, s2)."""
    s = S(rho)
    _, c = reduced_c(solve_reduced(np.array([1.0, seed]), s, stop_loss=1e-16))
    return overshoot(c, s)[0]
