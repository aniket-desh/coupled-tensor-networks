"""Aligned-sector factor flows: every coefficient is a product of scalar parameters, c_k = prod_j theta_j^A_kj.

A is a 0/1 incidence matrix (component k uses parameter j). Two components are coupled exactly when their rows
overlap, so sharing patterns differ only in A. The state is x = log theta, which keeps precision while
c ~ delta^depth is tiny.

Gradient flow of L = 1/2 sum_k (s_k - c_k)^2 gives dtheta_j/dt = sum_k e_k c_k A_kj / theta_j with e = s - c, so

    dx_j/dt = sum_k e_k c_k A_kj / theta_j^2,      dc/dt = K e,   K_ik = c_i c_k sum_j A_ij A_kj / theta_j^2.

K is the mobility: K_ik counts the parameters shared by components i and k, each weighted by 1/theta_j^2.
"""
import numpy as np

from .integrate import capped
from .measure import loss


class Pattern:
    def __init__(self, A, labels=None):
        self.A = np.asarray(A, float)
        self.labels = list(labels) if labels is not None else list(range(self.A.shape[1]))
        # Balanced init with zero bond balances on a tree: a factor used by m components starts at sqrt(m) delta.
        self.amp = np.sqrt(self.A.sum(0))

    @classmethod
    def from_paths(cls, paths):
        """paths[k] lists the factor labels component k multiplies; repeated labels across components are shared."""
        labels = list(dict.fromkeys(f for p in paths for f in p))
        A = np.zeros((len(paths), len(labels)))
        for k, p in enumerate(paths):
            A[k, [labels.index(f) for f in p]] = 1.0
        return cls(A, labels)

    def index(self, label):
        return self.labels.index(label)

    def x0(self, delta):
        return np.log(self.amp * delta)

    def coeffs(self, x):
        return np.exp(self.A @ x)

    def rhs(self, x, s):
        c = np.exp(self.A @ x)
        return (self.A.T @ (c * (s - c))) * np.exp(-2 * x)

    def jac(self, x, s):
        """Jacobian of rhs; only the implicit (Radau) solver uses it."""
        c = np.exp(self.A @ x)
        inner = np.exp(-2 * x)[:, None] * (self.A.T * (c * (s - 2 * c))) @ self.A
        return inner - 2 * np.diag(self.rhs(x, s))

    def mobility(self, x):
        c = np.exp(self.A @ x)
        return np.outer(c, c) * ((self.A * np.exp(-2 * x)) @ self.A.T)

    def run(self, s, delta, stop_loss=1e-16, stiff=False, **kw):
        """Flow from the balanced aligned init until L < stop_loss. Returns t, c (components x times), done."""
        if stiff:
            kw['jac'] = lambda y: self.jac(y, s)
        t, x, done = capped(lambda y: self.rhs(y, s), self.x0(delta), lambda y: loss(self.coeffs(y), s) < stop_loss, **kw)
        return t, np.exp(self.A @ x), done


# ------------------------------------------------------------ closed reduced flows in log c
def log_flow(K):
    """Reduced flow dc/dt = K(c) (s - c), written for x = log c. K(c) is a closed-form mobility."""
    def rhs(x, s):
        c = np.exp(x)
        return K(c) @ (s - c) / c
    return rhs


def K_independent(depth):
    """Uncoupled balanced depth-n modes: dc_k/dt = n c_k^(2 - 2/n) e_k."""
    return lambda c: np.diag(depth * c ** (2 - 2 / depth))
