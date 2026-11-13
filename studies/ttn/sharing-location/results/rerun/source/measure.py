"""Observables of coefficient trajectories c, shape (components, times), against targets s."""
import numpy as np


def loss(c, s):
    """1/2 sum_k (s_k - c_k)^2; for c of shape (components, times) returns one value per time."""
    s = np.asarray(s)
    return 0.5 * ((s.reshape(s.shape + (1,) * (np.ndim(c) - 1)) - c) ** 2).sum(0)


def overshoot(c, s):
    """Max relative excursion above target for each component, clipped at zero."""
    return np.maximum((c / np.asarray(s)[:, None] - 1).max(1), 0.0)


def half_time(t, ck, sk):
    """First time ck reaches sk/2, linearly interpolated between samples; nan if it never does."""
    i = np.argmax(ck >= 0.5 * sk)
    if i == 0:
        return np.nan if ck[0] < 0.5 * sk else t[0]
    return t[i - 1] + (0.5 * sk - ck[i - 1]) / (ck[i] - ck[i - 1]) * (t[i] - t[i - 1])


def half_times(t, c, s):
    return np.array([half_time(t, ck, sk) for ck, sk in zip(c, s)])


def dissipation(K, e):
    """Pair terms of -dL/dt = e^T K e: D[i, j] = K_ij e_i e_j over time, for K of shape (times, n, n)
    and e of shape (n, times). Summing D over any set of (i, j) gives that set's share of the loss decrease."""
    return np.einsum('tij,it,jt->ijt', K, e, e)
