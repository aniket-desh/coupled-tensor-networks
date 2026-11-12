"""Hierarchical sharing: four orthogonal components in two groups on a five-core tree.

Tensor:  T_oijl = u_i F_gj P_ag R_oab B_bl,  bonds: u-P dim 1, P-F dim 2, P-R dim 2, R-B dim 4.
Teacher: T* = sum_k s_k e_k x g x h_G(k) x w_k, groups G = {0,1} and {2,3}.

Diagonal sector: c_k = a f_G p_G r_k b_k. Under balance f_G = p_G = q_G, r_k = b_k = sqrt(p_k),
a^2 = sum_G q_G^2, q_G^2 = sum_(k in G) p_k. Writing C_G = sum_(k in G) c_k, these close exactly:

    a = (sum_G sqrt(C_G))^(2/5),  q_G^2 = sqrt(C_G / a),  p_k = c_k / (a q_G^2),
    cdot = K(c) eps,  K_ij = c_i c_j [1/a^2 + same_group(i,j) 2/q_G^2 + (i==j) 2/p_i].

The three terms count shared cores: every pair shares u, siblings also share f and p.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from ctn import io
from ctn.factors import Pattern, log_flow
from ctn.integrate import capped
from ctn.measure import loss
from ctn.network import Network

G = np.array([0, 0, 1, 1])
SAME = G[:, None] == G[None, :]
SPEC = 'i,gj,ag,oab,bl->oijl'
DIMS = dict(i=2, g=2, j=2, a=2, o=4, b=4, l=4)


def component(k, o=None):
    return dict(o=k if o is None else o, i=0, g=G[k], j=G[k], a=G[k], b=k, l=k)


NET = Network(SPEC, DIMS, [component(k) for k in range(4)])  # 58 parameters


def single_output_net(k):
    """One independent 34-parameter tree that sees only component k: nothing can couple it to the others."""
    return Network(SPEC, {**DIMS, 'o': 1}, [component(k, o=0)])


# ------------------------------------------------------------ sharing patterns of the aligned sector
# Component k multiplies five scalar factors; a label repeated across components is a shared parameter.
ARMS = {
    'hierarchy': Pattern.from_paths([('u', f'f{G[k]}', f'p{G[k]}', f'r{k}', f'b{k}') for k in range(4)]),
    'u_only': Pattern.from_paths([('u', f'f{k}', f'p{k}', f'r{k}', f'b{k}') for k in range(4)]),  # far coupling only
    'group_only': Pattern.from_paths([(f'u{G[k]}', f'f{G[k]}', f'p{G[k]}', f'r{k}', f'b{k}') for k in range(4)]),  # near only
    'independent': Pattern.from_paths([(f'u{k}', f'f{k}', f'p{k}', f'r{k}', f'b{k}') for k in range(4)]),
}


# ------------------------------------------------------------ exact 4D reduction
def balanced_state(c):
    C = np.array([c[G == g].sum() for g in (0, 1)])
    a = np.sqrt(C).sum() ** (2 / 5)
    q2 = np.sqrt(C / a)
    return a, q2, c / (a * q2[G])


def K_matrix(c):
    a, q2, p = balanced_state(c)
    return np.outer(c, c) * (1 / a ** 2 + SAME * 2 / q2[G][:, None]) + np.diag(2 * c ** 2 / p)


rhs_reduced = log_flow(K_matrix)


def run_reduced(c0, s, stop_loss=1e-16, **kw):
    t, x, done = capped(lambda y: rhs_reduced(y, s), np.log(c0), lambda y: loss(np.exp(y), s) < stop_loss, **kw)
    return t, np.exp(x), done


def run_dense(net, th0, s, stop_loss=1e-14, **kw):
    """Population gradient flow on the full parameters; coefficients read off the tensor at (thinned) samples."""
    target = net.teacher(s)
    t, y, done = capped(lambda y: -net.grad(y, target), th0, lambda y: net.loss(y, target) < stop_loss,
                        cap=0.03, scale=True, **kw)
    stride = max(y.shape[1] // 40000, 1)
    return t[::stride], net.coeffs(y[:, ::stride]), done


# ------------------------------------------------------------ race theory (P1, P2)
def I_x(x):
    return quad(lambda u: 1 / (u ** 2 + u ** (1 + x)), 1, np.inf)[0]


def tau(top, sib, delta=1.0):
    """Blow-up time of a group with top strength `top` and sibling strength `sib`, in tau = int a dt."""
    return I_x(sib / top) / (2 * top * delta ** 2)


# ------------------------------------------------------------ stored results
RESULTS = Path(__file__).parent / 'results'


def load(name):
    return io.load_json(RESULTS / f'{name}.json')


def merged(name):
    """Sweep rows with runs that exhausted DOP853's budget replaced by their Radau reruns (results/rerun.json)."""
    key = lambda r: tuple(r.get(k) for k in ('arm', 'x', 'y', 'delta', 'seed'))
    redo = {key(r): r for r in load('rerun')[name]}
    return [redo.get(key(r), r) for r in load(name)['results']]
