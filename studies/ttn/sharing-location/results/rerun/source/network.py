"""Dense tensor networks written as one einsum, with exact gradients of the Frobenius and sample losses.

A network is a spec such as 'i,oa,aj->oij' plus a size for every index letter. The parameter vector theta is
the cores flattened and concatenated in spec order. Each core appears once and has no repeated letter, so T is
linear in every core, and the gradient of <dT, T> with respect to core k is the same contraction with core k
removed and dT in its place. That one rule replaces a hand-written chain rule for every new tree.

The first output letter is the output o of f_o(x); the remaining output letters are input legs, and
f_o(x) = sum T_{o,i,j,...} x1_i x2_j ... for whitened inputs x1, x2, ...

Aligned sector: a component k is a dict {letter: index} that picks one entry of every core. Components fix the
teacher direction T* = sum_k s_k (one-hot at the output entry), the coefficient readout c_k = <T, that one-hot>,
the balanced aligned init, and the factor flow (each distinct core entry used is one scalar factor).
"""
from collections import Counter

import numpy as np

from .factors import Pattern


class Network:
    def __init__(self, spec, dims, components=()):
        ins, self.out = spec.replace(' ', '').split('->')
        self.cores = ins.split(',')
        self.dims = dict(dims)
        self.shapes = [tuple(self.dims[ch] for ch in core) for core in self.cores]
        self.offsets = np.cumsum([0] + [int(np.prod(sh)) for sh in self.shapes])
        self.n = int(self.offsets[-1])
        self.shape = tuple(self.dims[ch] for ch in self.out)
        self.components = [dict(c) for c in components]
        self._spec = spec
        # Gradient of <dT, T> wrt core k: contract dT with every other core, output core k's letters.
        self._grad_specs = [','.join([self.out] + self.cores[:k] + self.cores[k + 1:]) + '->' + core
                            for k, core in enumerate(self.cores)]
        # Contraction orders are found once; einsum would otherwise search for one on every call.
        dummy = [np.ones(sh) for sh in self.shapes]
        self._path = np.einsum_path(spec, *dummy, optimize='greedy')[0]
        self._grad_paths = [np.einsum_path(g, np.ones(self.shape), *(dummy[:k] + dummy[k + 1:]), optimize='greedy')[0]
                            for k, g in enumerate(self._grad_specs)]

    # -------------------------------------------------------------- parameters and tensor
    def unpack(self, theta):
        return [theta[a:b].reshape(sh) for a, b, sh in zip(self.offsets[:-1], self.offsets[1:], self.shapes)]

    def pack(self, cores):
        return np.concatenate([np.ravel(c) for c in cores])

    def tensor(self, theta):
        return np.einsum(self._spec, *self.unpack(theta), optimize=self._path)

    def pull_back(self, theta, dT):
        """d<dT, T(theta)>/dtheta: turns the gradient of any scalar wrt T into its gradient wrt theta."""
        cores = self.unpack(theta)
        return np.concatenate([np.einsum(g, dT, *(cores[:k] + cores[k + 1:]), optimize=p).ravel()
                               for k, (g, p) in enumerate(zip(self._grad_specs, self._grad_paths))])

    # -------------------------------------------------------------- losses (whitened inputs)
    def loss(self, theta, target, G=None):
        """1/2 ||T - T*||^2, the population loss; or 1/2 D:G:D with a sample moment matrix G of the input features."""
        D = (self.tensor(theta) - target).reshape(self.shape[0], -1)
        return 0.5 * (D * D).sum() if G is None else 0.5 * (D * (D @ G)).sum()

    def grad(self, theta, target, G=None):
        D = self.tensor(theta) - target
        dT = D if G is None else (D.reshape(self.shape[0], -1) @ G).reshape(D.shape)
        return self.pull_back(theta, dT)

    def grad_batch(self, theta, target, xs):
        """Minibatch gradient of 1/(2b) sum_n ||f(x_n) - y_n||^2 with noiseless labels y = f*(x), written sample-wise."""
        z = features(xs)
        D = (self.tensor(theta) - target).reshape(self.shape[0], -1)
        r = z @ D.T  # prediction - label, shape (batch, outputs)
        return self.pull_back(theta, (r.T @ z / len(z)).reshape(self.shape))

    # -------------------------------------------------------------- aligned sector
    def entry(self, k, comp):
        return tuple(comp[ch] for ch in self.cores[k])

    def teacher(self, s):
        T = np.zeros(self.shape)
        for comp, sk in zip(self.components, s):
            T[tuple(comp[ch] for ch in self.out)] = sk
        return T

    def coeffs(self, theta):
        """c_k for one theta, or shape (components, times) for theta of shape (n, times)."""
        if np.ndim(theta) == 2:
            return np.array([self.coeffs(th) for th in theta.T]).T
        T = self.tensor(theta)
        return np.array([T[tuple(comp[ch] for ch in self.out)] for comp in self.components])

    def random(self, sigma, rng):
        return sigma * rng.standard_normal(self.n)

    def aligned(self, delta):
        """Balanced aligned init: an entry used by m components starts at sqrt(m) delta, everything else at 0.
        On a tree this zeroes every conserved bond balance |parent|^2 - sum |children|^2."""
        cores = [np.zeros(sh) for sh in self.shapes]
        for k in range(len(self.cores)):
            for e, m in Counter(self.entry(k, comp) for comp in self.components).items():
                cores[k][e] = np.sqrt(m) * delta
        return self.pack(cores)

    def pattern(self):
        """Factor flow of the aligned sector: factor = (core, entry), shared when components pick the same entry."""
        return Pattern.from_paths([[(self.cores[k], self.entry(k, comp)) for k in range(len(self.cores))]
                                   for comp in self.components])


def features(xs):
    """Per-sample Kronecker product of the input legs, shape (batch, prod of leg sizes)."""
    z = np.ones((len(xs[0]), 1))
    for x in xs:
        z = (z[:, :, None] * x[:, None, :]).reshape(len(x), -1)
    return z
