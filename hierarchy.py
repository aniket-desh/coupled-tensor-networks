"""Hierarchical sharing: four orthogonal components in two groups on a five-core tree.

Tensor:  T_oijl = u_i F_gj P_ag R_oab B_bl,  bonds: u-P dim 1, P-F dim 2, P-R dim 2, R-B dim 4.
Teacher: T* = sum_k s_k e_k x g x h_G(k) x w_k, groups G = {0,1} and {2,3}.

Diagonal sector: c_k = a f_G p_G r_k b_k. Under balance f_G = p_G = q_G, r_k = b_k = sqrt(p_k),
a^2 = sum_G q_G^2, q_G^2 = sum_(k in G) p_k. Writing C_G = sum_(k in G) c_k, these close exactly:

    a = (sum_G sqrt(C_G))^(2/5),  q_G^2 = sqrt(C_G / a),  p_k = c_k / (a q_G^2),
    cdot = K(c) eps,  K_ij = c_i c_j [1/a^2 + same_group(i,j) 2/q_G^2 + (i==j) 2/p_i].

The three terms count shared cores: every pair shares u, siblings also share f and p.
"""
import numpy as np
from scipy.integrate import Radau
from scipy.integrate._ivp.rk import DOP853

G = np.array([0, 0, 1, 1])
RTOL, ATOL = 1e-12, 1e-13
RTOL_STIFF, ATOL_STIFF = 1e-10, 1e-11
SAME = G[:, None] == G[None, :]


# ------------------------------------------------------------ exact 4D reduction
def balanced_state(c):
    C = np.array([c[G == g].sum() for g in (0, 1)])
    a = np.sqrt(C).sum() ** (2 / 5)
    q2 = np.sqrt(C / a)
    return a, q2, c / (a * q2[G])


def K_matrix(c):
    a, q2, p = balanced_state(c)
    return np.outer(c, c) * (1 / a ** 2 + SAME * 2 / q2[G][:, None]) + np.diag(2 * c ** 2 / p)


def rhs_reduced(x, s):
    """Flow of x = log c."""
    c = np.exp(x)
    return K_matrix(c) @ (s - c) / c


# ------------------------------------------------------------ sharing patterns as factor lists
class Arm:
    """c_k = prod of five scalar factors; `idx[k]` lists which parameters component k uses.
    Removing sharing is removing index overlap, so arms differ only in what is shared.
    `amp` gives balanced init amplitudes in units of delta (conserved bond balances = 0)."""

    def __init__(self, name):
        self.name = name
        r, s2 = range(4), np.sqrt(2)
        if name == 'hierarchy':  # u shared by all; f, p per group
            idx = [[0, 1 + G[k], 3 + G[k], 5 + k, 9 + k] for k in r]
            amp = [2.0, s2, s2, s2, s2] + [1.0] * 8
        elif name == 'u_only':  # far coupling only: f, p private
            idx = [[0, 1 + k, 5 + k, 9 + k, 13 + k] for k in r]
            amp = [2.0] + [1.0] * 16
        elif name == 'group_only':  # near coupling only: each group owns its u, f, p
            idx = [[G[k], 2 + G[k], 4 + G[k], 6 + k, 10 + k] for k in r]
            amp = [s2] * 6 + [1.0] * 8
        elif name == 'independent':
            idx = [[5 * k + j for j in range(5)] for k in r]
            amp = [1.0] * 20
        else:
            raise ValueError(name)
        self.idx = np.array(idx)
        self.amp = np.array(amp)
        self.A = np.zeros((4, len(amp)))  # incidence: component k uses parameter j
        for k in r:
            self.A[k, self.idx[k]] = 1.0

    def x0(self, delta):
        return np.log(self.amp * delta)

    def coeffs(self, x):
        return np.exp(self.A @ x)

    def rhs(self, x, s):
        """Gradient flow of 1/2 sum (s_k - c_k)^2 in x = log(theta): dx_j = sum_k c_k e_k A_kj / theta_j^2."""
        c = np.exp(self.A @ x)
        return (self.A.T @ (c * (s - c))) * np.exp(-2 * x)

    def jac(self, x, s):
        """Jacobian of rhs; only the implicit solver uses it."""
        c = np.exp(self.A @ x)
        inner = np.exp(-2 * x)[:, None] * (self.A.T * (c * (s - 2 * c))) @ self.A
        return inner - 2 * np.diag(self.rhs(x, s))


# ------------------------------------------------------------ integrator
def integrate(fun, y0, stop, cap=0.05, t_max=1e15, sub=6, max_steps=400_000, scale=None, jac=None):
    """DOP853 with a step cap that limits how far the state moves per step, so plateaus of length
    ~1/delta^3 cost almost nothing and transitions stay finely sampled. `scale` turns the cap from
    log-coordinates (cap/|f|_inf) into a relative cap for raw parameters (cap*|y|/|f|).
    Passing `jac` selects Radau: once a component has converged, its fast contraction limits DOP853 to
    steps of about 0.2 for the whole wait until the next component, which can exceed max_steps."""
    if jac is None:
        solver = DOP853(lambda t, y: fun(y), 0.0, np.array(y0, float), t_max, rtol=RTOL, atol=ATOL)
    else:
        solver = Radau(lambda t, y: fun(y), 0.0, np.array(y0, float), t_max, rtol=RTOL_STIFF, atol=ATOL_STIFF,
                       jac=lambda t, y: jac(y))
    T, Y = [0.0], [np.array(y0, float)]
    n = 0
    while solver.status == 'running' and n < max_steps:
        size = np.abs(solver.f).max() if scale is None else np.linalg.norm(solver.f) / max(np.linalg.norm(solver.y), 1e-300)
        solver.max_step = cap / max(size, 1e-300)
        solver.step()
        n += 1
        local = solver.dense_output()
        for tt in np.linspace(solver.t_old, solver.t, sub + 1)[1:]:
            T.append(tt)
            Y.append(local(tt))
        if stop(solver.y):
            break
    return np.array(T), np.array(Y).T, bool(stop(solver.y))


def run_arm(arm, s, delta, stop_loss=1e-16, stiff=False, **kw):
    if stiff:
        kw['jac'] = lambda y: arm.jac(y, s)
    t, x, done = integrate(lambda y: arm.rhs(y, s), arm.x0(delta), lambda y: 0.5 * ((s - arm.coeffs(y)) ** 2).sum() < stop_loss, **kw)
    return t, np.exp(arm.A @ x), done


def run_reduced(c0, s, stop_loss=1e-16, **kw):
    t, x, done = integrate(lambda y: rhs_reduced(y, s), np.log(c0), lambda y: 0.5 * ((s - np.exp(y)) ** 2).sum() < stop_loss, **kw)
    return t, np.exp(x), done


def half_times(t, c, s):
    out = []
    for ck, sk in zip(c, s):
        i = np.argmax(ck >= 0.5 * sk)
        out.append(t[i - 1] + (0.5 * sk - ck[i - 1]) / (ck[i] - ck[i - 1]) * (t[i] - t[i - 1]) if i else np.nan)
    return np.array(out)


def overshoot(c, s):
    return np.maximum((c / s[:, None] - 1).max(1), 0.0)


# ------------------------------------------------------------ dense 58-parameter tree
def d_unpack(th, O):
    u, F, P = th[:2], th[2:6].reshape(2, 2), th[6:10].reshape(2, 2)
    R = th[10:10 + 8 * O].reshape(O, 2, 4)
    return u, F, P, R, th[10 + 8 * O:].reshape(4, 4)


def d_tensor(th, O):
    u, F, P, R, B = d_unpack(th, O)
    left = u[None, :, None] * (P @ F)[:, None, :]  # a i j
    return np.einsum('aij,oal->oijl', left, R @ B, optimize=True)


def d_target(s, rows=None):
    rows = range(4) if rows is None else rows
    T = np.zeros((len(rows), 2, 2, 4))
    for o, k in enumerate(rows):
        T[o, 0, G[k], k] = s[k]
    return T


def d_coeffs(T, rows=None):
    rows = range(4) if rows is None else rows
    return np.array([T[o, 0, G[k], k] for o, k in enumerate(rows)])


def d_grad(th, target):
    """Gradient of 1/2 ||T - T*||^2 (whitened inputs make the population loss this Frobenius norm)."""
    O = target.shape[0]
    u, F, P, R, B = d_unpack(th, O)
    PF = P @ F
    left = u[None, :, None] * PF[:, None, :]
    right = R @ B
    D = np.einsum('aij,oal->oijl', left, right, optimize=True) - target
    dleft = np.einsum('oijl,oal->aij', D, right, optimize=True)
    dright = np.einsum('oijl,aij->oal', D, left, optimize=True)
    du = (dleft * PF[:, None, :]).sum((0, 2))
    dPF = (dleft * u[None, :, None]).sum(1)
    return np.concatenate([du, (P.T @ dPF).ravel(), (dPF @ F.T).ravel(), (dright @ B.T).ravel(),
                           np.einsum('oab,oal->bl', R, dright, optimize=True).ravel()])


def d_aligned(delta):
    """Balanced aligned init in the 58-parameter representation, matching Arm('hierarchy')."""
    s2 = np.sqrt(2) * delta
    R = np.zeros((4, 2, 4))
    for k in range(4):
        R[k, G[k], k] = delta
    return np.concatenate([[2 * delta, 0.0], (s2 * np.eye(2)).ravel(), (s2 * np.eye(2)).ravel(), R.ravel(), (delta * np.eye(4)).ravel()])


def run_dense(th0, target, rows=None, stop_loss=1e-14, **kw):
    fun = lambda y: -d_grad(y, target)
    L = lambda y: 0.5 * ((d_tensor(y, target.shape[0]) - target) ** 2).sum()
    t, y, done = integrate(fun, th0, lambda y: L(y) < stop_loss, cap=0.03, scale=True, **kw)
    # Coefficients from the tensor at every sample; thin first so this stays cheap.
    stride = max(y.shape[1] // 40000, 1)
    t, y = t[::stride], y[:, ::stride]
    c = np.array([d_coeffs(d_tensor(v, target.shape[0]), rows) for v in y.T]).T
    return t, c, done
