"""Shared-feature three-core TTN from the Zulip note, plus its no-sharing control.

Tensor:  T_oij = U_oi M_oj,  M = R B.

U has shape (1, 2) in the shared model (one left feature u used by both outputs,
the note's model, 10 parameters) and (2, 2) in the control (each output owns its
left feature, 12 parameters). That is the only difference between the two, so a
gap between them is attributable to sharing u.

Teacher: T*_oij = g_i S_oj with S = diag(s1, s2). Whitened inputs are
rotation invariant, so we take g = (1, 0) and e_k = h_k = standard basis without
loss of generality. Coefficients are the projections c_k = <T, e_k x g x h_k>,
which are well defined at arbitrary (misaligned) parameters.
"""
import numpy as np
from scipy.integrate import solve_ivp

RTOL, ATOL = 1e-12, 1e-14
MAX_STEP = 0.5  # DOP853 otherwise takes steps so long that its dense interpolant dominates the error


# ---------------------------------------------------------------- full parameter model
def pack(U, R, B):
    return np.concatenate([U.ravel(), R.ravel(), B.ravel()])


def unpack(theta):
    # 10 parameters -> shared u, 12 -> per-output u.
    K = (theta.size - 8) // 2
    return theta[:2 * K].reshape(K, 2), theta[2 * K:2 * K + 4].reshape(2, 2), theta[2 * K + 4:].reshape(2, 2)


def tensor(U, R, B):
    M = R @ B
    return U[:, :, None] * M[:, None, :]  # broadcasts K=1 over outputs o


def target(s):
    T = np.zeros((2, 2, 2))
    T[0, 0, 0], T[1, 0, 1] = s[0], s[1]
    return T


def coeffs(theta):
    T = tensor(*unpack(theta))
    return np.array([T[0, 0, 0], T[1, 0, 1]])


def loss(theta, s, G4=None):
    """1/2 ||T - T*||^2, or 1/2 D:G:D with the sample moment tensor G4 when given."""
    D = (tensor(*unpack(theta)) - target(s)).reshape(2, 4)
    return 0.5 * (D * D).sum() if G4 is None else 0.5 * (D * (D @ G4)).sum()


def pull_back(dT, U, R, B):
    """Chain rule: dX/dT -> dX/dtheta for any scalar X, given its gradient wrt T."""
    M = R @ B
    dU = (dT * M[:, None, :]).sum(2)
    if U.shape[0] == 1:
        dU = dU.sum(0, keepdims=True)  # the one u receives both outputs' gradient
    dM = (dT * U[:, :, None]).sum(1)
    return pack(dU, dM @ B.T, R.T @ dM)


def grad(theta, s, G4=None):
    U, R, B = unpack(theta)
    D = tensor(U, R, B) - target(s)
    dT = D if G4 is None else (D.reshape(2, 4) @ G4).reshape(2, 2, 2)
    return pull_back(dT, U, R, B)


def grad_batch(theta, s, x1, x2):
    """Minibatch gradient of 1/(2b) sum_n ||f(x1,x2) - y||^2, written out sample-wise."""
    U, R, B = unpack(theta)
    D = tensor(U, R, B) - target(s)
    z = (x1[:, :, None] * x2[:, None, :]).reshape(-1, 4)  # features x1 (kron) x2
    r = z @ D.reshape(2, 4).T  # prediction - label, shape (batch, outputs)
    dT = (r.T @ z / z.shape[0]).reshape(2, 2, 2)
    return pull_back(dT, U, R, B)


def init_aligned(s, delta, shared=True):
    """Balanced aligned init: r_k = b_k = delta, a = sqrt(2) delta (shared) or a_k = delta."""
    U = np.array([[np.sqrt(2) * delta, 0.0]]) if shared else np.array([[delta, 0.0], [delta, 0.0]])
    return pack(U, delta * np.eye(2), delta * np.eye(2))


def init_random(sigma, rng, shared=True):
    return sigma * rng.standard_normal(10 if shared else 12)


def flow(theta0, s, t_max, stop_loss=1e-18, grad_fn=None, event=None):
    """Integrate gradient flow; stop at loss < stop_loss (and optionally at `event`)."""
    g = grad_fn or (lambda t, y: -grad(y, s))
    done = lambda t, y: loss(y, s) - stop_loss
    done.terminal = True
    events = [done] + ([event] if event else [])
    return solve_ivp(g, (0, t_max), theta0, method='DOP853', rtol=RTOL, atol=ATOL,
                     max_step=MAX_STEP, dense_output=True, events=events)


# ---------------------------------------------------------------- reduced equations
def K_matrix(c):
    """Mobility K(c) = 2a diag(c) + c c^T / a^2, a = (c1+c2)^(1/3), so that cdot = K eps."""
    a = c.sum() ** (1 / 3)
    return 2 * a * np.diag(c) + np.outer(c, c) / a ** 2


def rhs_shared(t, x, s):
    """Balanced 2D flow in x = log c, which keeps precision when c ~ delta^3 is tiny."""
    c = np.exp(x)
    a = c.sum() ** (1 / 3)
    e = s - c
    return 2 * a * e + (c * e).sum() / a ** 2


def rhs_independent(t, x, s):
    """Independent balanced depth-3 modes: cdot_k = 3 c_k^(4/3) eps_k."""
    c = np.exp(x)
    return 3 * c ** (1 / 3) * (s - c)


def solve_reduced(c0, s, rhs=rhs_shared, t_max=1e8, stop_loss=1e-18):
    done = lambda t, x, *_: 0.5 * ((s - np.exp(x)) ** 2).sum() - stop_loss
    done.terminal = True
    return solve_ivp(rhs, (0, t_max), np.log(c0), args=(s,), method='DOP853',
                     rtol=RTOL, atol=ATOL, max_step=MAX_STEP, dense_output=True, events=done)


def refine(sol, sub=20):
    """Time grid that keeps every adaptive step and subdivides it, so peaks are resolved."""
    t = sol.t
    return np.unique(np.concatenate([np.linspace(t[i], t[i + 1], sub, endpoint=False)
                                     for i in range(len(t) - 1)] + [t[-1:]]))


def overshoot(c, s):
    """Max relative excursion above target for each component, clipped at zero."""
    return np.maximum((c / s[:, None] - 1).max(axis=1), 0.0)


def half_time(t, c_k, s_k):
    return np.interp(0.5 * s_k, np.maximum.accumulate(c_k), t)


def c0_balanced(delta):
    return np.full(2, np.sqrt(2) * delta ** 3)


def heteroclinic_overshoot(rho, seed=1e-10):
    """Small-init limit: start next to the saddle (s1, 0) and follow it to (s1, s2)."""
    s = np.array([1.0, rho])
    sol = solve_reduced(np.array([1.0, seed]), s, stop_loss=1e-16)
    return overshoot(np.exp(sol.sol(refine(sol))), s)[0]
