"""Eight-leaf binary tree tensor network trained by ordinary backprop, evaluated through its cores.

    z^(j)   = A_j phi_j,                    phi_j = (1, x^(j)),  leaves j = 1..8, A_j in R^{r x (1+d)}
    z^(v)_a = sum_bc W^(v)_abc z^(vL)_b z^(vR)_c,   quarters q1..q4 (leaf pairs), halves L, R
    f_o     = sum_bc C_obc z^(L)_b z^(R)_c

The coefficient tensor T in R^{O x (1+d)^8} is never formed: each batch is contracted leaf to root. Every core
carries a leading run axis R, so independent networks (seeds, arms) train in one batched computation. Their losses
are summed, which leaves each run's gradient its own.

Init: entries N(0, (alpha sigma)^2) with sigma = 1/sqrt(1+d) for leaves and 1/r for fusions and the root. For
independent unit-second-moment inputs every activation then has second moment ~1 at alpha = 1, and since each
output is a product through all 15 cores, scaling every core by alpha scales the output by alpha^15.

Teacher: components k = (output o_k, strength s_k, coordinates), g_k(x) = prod over its (patch, coord) of x^(patch)_coord
on independent +-1 inputs, y_o = sum_{k: o_k = o} s_k g_k. Distinct coordinate sets give orthonormal g_k, so the
coefficient c_k = E[f_{o_k} g_k] is the student's projection on component k and equals s_k at the teacher.
A product using at most one coordinate per patch is exactly representable (teacher_cores builds it).
"""
import math

import numpy as np
import torch

# Core names, their slots in the four parameter tensors, and the subtree each belongs to.
CORES = [f'A{j + 1}' for j in range(8)] + [f'W{k + 1}' for k in range(4)] + ['WL', 'WR', 'C']
GROUPS = {'q1': ['A1', 'A2', 'W1'], 'q2': ['A3', 'A4', 'W2'], 'q3': ['A5', 'A6', 'W3'], 'q4': ['A7', 'A8', 'W4'],
          'L': ['WL'], 'R': ['WR'], 'root': ['C']}
NODES = ['q1', 'q2', 'q3', 'q4', 'L', 'R']  # internal nodes whose activations are measured
LEAVES_OF = {'q1': [0, 1], 'q2': [2, 3], 'q3': [4, 5], 'q4': [6, 7], 'L': [0, 1, 2, 3], 'R': [4, 5, 6, 7]}


def slot(name):
    """(parameter tensor index, position along its core axis) of a core."""
    if name[0] == 'A':
        return 0, int(name[1:]) - 1
    if name in ('WL', 'WR'):
        return 2, 'LR'.index(name[1])
    if name == 'C':
        return 3, None
    return 1, int(name[1:]) - 1


# ------------------------------------------------------------ network
class TTN:
    def __init__(self, R, r, d=8, O=10, alpha=1.0, seed=0, device='cpu', dtype=torch.float32, const_weight=None):
        """const_weight w: fraction of each leaf row's variance on the constant phi_0 (w = 1/(1+d) is the isotropic
        default). w > 1/(1+d) starts every leaf mostly carrying the constant, i.e. ignoring its patch, which breaks
        the basis-permutation symmetry and makes the number of patches a feature uses matter. The leaf second moment
        stays 1 for every w."""
        self.R, self.r, self.d, self.O = R, r, d, O
        self.device, self.dtype = device, dtype
        g = torch.Generator(device='cpu').manual_seed(seed)
        p = d + 1
        shapes = [(R, 8, r, p), (R, 4, r, r, r), (R, 2, r, r, r), (R, O, r, r)]
        leaf_std = torch.full((p,), 1 / math.sqrt(p), dtype=torch.float64)
        if const_weight is not None:
            leaf_std = torch.tensor([math.sqrt(const_weight)] + [math.sqrt((1 - const_weight) / d)] * d, dtype=torch.float64)
        stds = [leaf_std, 1 / r, 1 / r, 1 / r]
        self.params = [(alpha * s * torch.randn(sh, generator=g, dtype=torch.float64)).to(device, dtype).requires_grad_()
                       for sh, s in zip(shapes, stds)]
        self.frozen = [torch.ones_like(t) for t in self.params]  # gradient masks, 0 where frozen

    def freeze(self, runs, cores):
        """Stop training `cores` (names from CORES or GROUPS) in the given runs."""
        names = [c for x in cores for c in (GROUPS.get(x, [x]))]
        for name in names:
            i, k = slot(name)
            if k is None:
                self.frozen[i][runs] = 0.0
            else:
                self.frozen[i][runs, k] = 0.0

    def forward(self, phi):
        """phi: (R, N, 8, 1+d). Returns activations: leaf (R,N,8,r), quarter (R,N,4,r), half (R,N,2,r), out (R,N,O)."""
        A, Wq, Wh, C = self.params
        leaf = torch.einsum('Rjap,Rnjp->Rnja', A, phi)
        quarter = fuse(Wq, leaf[:, :, 0::2], leaf[:, :, 1::2])
        half = fuse(Wh, quarter[:, :, 0::2], quarter[:, :, 1::2])
        out = torch.einsum('Robc,Rnb,Rnc->Rno', C, half[:, :, 0], half[:, :, 1])
        return dict(leaf=leaf, quarter=quarter, half=half, out=out)

    def __call__(self, phi):
        return self.forward(phi)['out']

    def copy_runs(self, src, runs_dst, runs_src):
        """Overwrite runs_dst of this network with runs_src of `src` (branching arms from a common state)."""
        with torch.no_grad():
            for p, q in zip(self.params, src.params):
                p[runs_dst] = q[runs_src]


def fuse(W, zl, zr):
    """W (R,K,r,r,r), zl, zr (R,N,K,r) -> (R,N,K,r): z_a = sum_bc W_abc zl_b zr_c."""
    return torch.einsum('Rkabc,Rnkb,Rnkc->Rnka', W, zl, zr)


# ------------------------------------------------------------ teacher and data
def features(x, comps):
    """g_k(x) for each component: x (R, N, 8, d) of +-1 -> (R, N, K)."""
    return torch.stack([torch.ones_like(x[..., 0, 0]) if not c['coords'] else
                        math.prod(x[..., p, i] for p, i in c['coords']) for c in comps], -1)


def restrict(comp, node):
    """Coordinates of a component that lie inside a node's subtree; its feature at that node."""
    return tuple(sorted((p, i) for p, i in comp['coords'] if p in LEAVES_OF[node]))


def targets(x, comps, O):
    g = features(x, comps)
    y = torch.zeros(*g.shape[:2], O, dtype=g.dtype, device=g.device)
    for k, c in enumerate(comps):
        y[..., c['out']] += c['s'] * g[..., k]
    return y, g


def sample(R, N, d, seed, device, dtype):
    """Independent, centered, whitened +-1 patches, and phi = (1, x) per patch."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = (2 * torch.randint(0, 2, (R, N, 8, d), generator=g) - 1).to(device, dtype)
    return x, torch.cat([torch.ones_like(x[..., :1]), x], -1)


def teacher_cores(comps, r, d, O):
    """Exact student weights for a product teacher (one run). Each node carries one bond direction per distinct
    restricted feature (the empty feature is the constant); raises if a node needs more than r."""
    nodes = {f'leaf{j}': [j] for j in range(8)} | LEAVES_OF
    feats = {v: list(dict.fromkeys(tuple(sorted((p, i) for p, i in c['coords'] if p in leaves)) for c in comps))
             for v, leaves in nodes.items()}
    for v, f in feats.items():
        if len(f) > r:
            raise ValueError(f'node {v} needs {len(f)} > r = {r} directions')
    A = np.zeros((8, r, d + 1))
    for j in range(8):
        for a, f in enumerate(feats[f'leaf{j}']):
            A[j, a, 0 if not f else 1 + f[0][1]] = 1.0
    def fusion(v, left, right):  # each direction of v is the product of its restrictions to the two children
        W = np.zeros((r, r, r))
        for a, f in enumerate(feats[v]):
            fl, fr = (tuple(x for x in f if x[0] in nodes[ch]) for ch in (left, right))
            W[a, feats[left].index(fl), feats[right].index(fr)] = 1.0
        return W
    Wq = np.stack([fusion(f'q{k + 1}', f'leaf{2 * k}', f'leaf{2 * k + 1}') for k in range(4)])
    Wh = np.stack([fusion('L', 'q1', 'q2'), fusion('R', 'q3', 'q4')])
    C = np.zeros((O, r, r))
    for c in comps:
        fl, fr = restrict(c, 'L'), restrict(c, 'R')
        C[c['out'], feats['L'].index(fl), feats['R'].index(fr)] += c['s']
    return [A, Wq, Wh, C]


# ------------------------------------------------------------ exact population quantities
# For independent inputs with E[phi_j phi_j^T] = I (here phi = (1, x), x independent +-1), the population inner
# product of two networks' outputs is the Frobenius inner product of their coefficient tensors, and it contracts
# through the cores: leaf Grams G_j = A_j A'_j^T, then G_v = sum W_abc W'_a'b'c' G_vL[b,b'] G_vR[c,c'] up the tree.
# Nothing of size (1+d)^8 is formed, and no sample is drawn.
def gram(P, Q):
    """<T_o(P), T_o(Q)> for every output o, shape (R, O). P, Q are core lists [A, Wq, Wh, C] with matching R."""
    A, Wq, Wh, C = P
    B, Vq, Vh, D = Q
    G = torch.einsum('Rjap,Rjbp->Rjab', A, B)
    Gq = torch.einsum('Rkabc,Rkbd,Rkce,Rkfde->Rkaf', Wq, G[:, 0::2], G[:, 1::2], Vq)
    Gh = torch.einsum('Rkabc,Rkbd,Rkce,Rkfde->Rkaf', Wh, Gq[:, 0::2], Gq[:, 1::2], Vh)
    return torch.einsum('Robc,Rbd,Rce,Rode->Ro', C, Gh[:, 0], Gh[:, 1], D)


def teacher_batch(comps, R, r, d, O, device, dtype, unit=None):
    """Teacher cores repeated over R runs; with `unit=k`, the network of component k alone at strength 1."""
    cs = comps if unit is None else [dict(comps[unit], s=1.0)]
    return [torch.tensor(w, device=device, dtype=dtype).expand(R, *w.shape).contiguous() for w in teacher_cores(cs, r, d, O)]


class Population:
    """Exact population loss and coefficients for a product teacher: L = 1/2 sum_o m_o ||T_o - T*_o||^2 and
    c_k = <T_{o_k}, e_k>, where e_k is component k's unit tensor."""

    def __init__(self, comps, R, r, d, O, device, dtype):
        self.comps = comps
        self.star = teacher_batch(comps, R, r, d, O, device, dtype)
        self.units = [teacher_batch(comps, R, r, d, O, device, dtype, unit=k) for k in range(len(comps))]
        self.star_norm = gram(self.star, self.star)

    def loss(self, net, active):
        P = net.params
        return 0.5 * ((gram(P, P) - 2 * gram(P, self.star) + self.star_norm) * active).sum(-1)

    def coeffs(self, net):
        return torch.stack([gram(net.params, u)[:, c['out']] for u, c in zip(self.units, self.comps)], -1)


# ------------------------------------------------------------ measurements
def coeffs(net, phi, g, comps):
    """c_k = mean_n f_{o_k}(x_n) g_k(x_n), shape (R, K); differentiable."""
    out = net(phi)
    return torch.stack([(out[..., c['out']] * g[..., k]).mean(1) for k, c in enumerate(comps)], -1)


def loss_per_run(out, y, active):
    """1/2 mean_n sum over active outputs (f_o - y_o)^2, per run; inactive outputs are masked, not targeted."""
    return 0.5 * (((out - y) ** 2) * active[:, None, :]).sum(-1).mean(-1)


def mobility(net, coeff_fn, K, loss_fn=None):
    """K_ij restricted to each subtree group: sum over trainable cores in the group of <grad c_i, grad c_j>,
    shape (R, K, K, len(GROUPS)), where coeff_fn() returns c of shape (R, K) differentiably.
    Summed over groups it is the full trainable-parameter K.

    With loss_fn (per-run loss, shape (R,)), also returns the velocity v_i = <grad c_i, -grad L> per group, (R, K, G):
    the exact dc_i/dt of gradient flow. It differs from sum_j K_ij e_j because the loss residual also has the student's
    coefficients on every non-teacher basis function, whose pull on c_i is v_i - sum_j K_ij e_j."""
    grads = []
    for k in range(K):
        c = coeff_fn()[:, k].sum()
        grads.append([gr * m for gr, m in zip(torch.autograd.grad(c, net.params), net.frozen)])
    if loss_fn is not None:
        grads.append([-gr * m for gr, m in zip(torch.autograd.grad(loss_fn().sum(), net.params), net.frozen)])
    out = torch.zeros(net.R, K, K, len(GROUPS), dtype=torch.float64)
    vel = torch.zeros(net.R, K, len(GROUPS), dtype=torch.float64)
    for gi, (gname, names) in enumerate(GROUPS.items()):
        for name in names:
            i, k = slot(name)
            vec = [(gr[i] if k is None else gr[i][:, k]).reshape(net.R, -1).double() for gr in grads]
            for a in range(K):
                for b in range(K):
                    out[:, a, b, gi] += (vec[a] * vec[b]).sum(1).cpu()
                if loss_fn is not None:
                    vel[:, a, gi] += (vec[a] * vec[K]).sum(1).cpu()
    return (out, vel) if loss_fn is not None else out


def node_activations(acts):
    """Activations of q1..q4, L, R: (R, N, 6, r)."""
    return torch.cat([acts['quarter'], acts['half']], 2)


def alignment(acts, x, comps):
    """Fraction of each component's restricted feature that lies in the span of a node's r activations:
    R^2 = |P_Z h|^2 / |h|^2, shape (R, K, 6). The restricted feature of a component with no coordinates in a
    subtree is the constant, which the node must also carry."""
    Z = node_activations(acts).double()
    bad = ~torch.isfinite(Z).flatten(1).all(1)  # diverged runs: measured as nan instead of crashing the solver
    Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    R_, N = Z.shape[:2]
    out = torch.zeros(R_, len(comps), len(NODES), dtype=torch.float64)
    for k, c in enumerate(comps):
        for v, node in enumerate(NODES):
            h = features(x, [dict(coords=restrict(c, node))]).double()  # (R, N, 1)
            sol = torch.linalg.lstsq(Z[:, :, v], h).solution
            proj = Z[:, :, v] @ sol
            out[:, k, v] = ((proj ** 2).sum((1, 2)) / (h ** 2).sum((1, 2))).cpu()
    out[bad.cpu()] = float('nan')
    return out


def spectra(acts):
    """Singular values of each node's activation matrix / sqrt(N): (R, 6, r)."""
    Z = node_activations(acts).double()
    bad = ~torch.isfinite(Z).flatten(1).all(1)
    sv = torch.linalg.svdvals(torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).permute(0, 2, 1, 3) / math.sqrt(Z.shape[1])).cpu()
    sv[bad.cpu()] = float('nan')
    return sv


# ------------------------------------------------------------ training
def log_steps(ratio):
    """Geometric schedule: true at step 0 and whenever floor(log_ratio(step)) changes."""
    lr = math.log(ratio)
    return lambda step: step < 2 or int(math.log(step) / lr) != int(math.log(step - 1) / lr)


def train(net, active, eta, steps, probe, comps, phi=None, y=None, pop=None, check_every=10, dc=0.01, dlog=0.1,
          base_ratio=1.01, detail_ratio=1.1, t0=0.0, stop_tol=1e-3, stop_loss=1e-8, log=None):
    """Full-batch GD, theta <- theta - eta grad L; time t = t0 + eta * step.
    Loss: the exact population loss when `pop` (a Population) is given, else the sampled loss on (phi, y).
    Coefficients and the monitored loss are exact with `pop`; otherwise they are measured on `probe` = (x, phi, g),
    a fresh sample, which also always supplies the activations for alignment and spectra.

    Adaptive logging. Escapes are short (a few time units) and happen at seed-dependent times, so a fixed or
    geometric grid misses them. c is evaluated every `check_every` steps and a log is recorded when any run's c_k/s_k
    moved by more than `dc`, or its log|c_k| by more than `dlog` (plateau growth), since the last record, and at least
    on a geometric schedule (`base_ratio`). The mobility K is taken at every record while any active component of any
    run is mid-rise (0.02 < c/s < 0.98), and on the coarse schedule otherwise; alignment and spectra only on the
    coarse schedule (`detail_ratio`) and at the end.
    Stops when every run has |c_k/s_k - 1| < stop_tol on its active components and monitored loss < stop_loss (or
    has diverged), or after `steps`. Returns the log dict (lists of numpy arrays), appending to `log` when given."""
    keys = ['t', 'c', 'L_train', 'L_probe', 'diverged', 't_K', 'K', 'v', 't_detail', 'align', 'spec']
    log = log if log is not None else {k: [] for k in keys}
    xp, phip, gp = probe
    dev, dt = phip.device, phip.dtype
    s = torch.tensor([c['s'] for c in comps], dtype=dt, device=dev)
    on = torch.stack([active[:, c['out']] > 0 for c in comps], -1)  # (R, K): component's output is trained
    yp, _ = targets(xp, comps, net.O)
    loss_fn = (lambda: pop.loss(net, active)) if pop is not None else (lambda: loss_per_run(net(phi), y, active))
    exact_c = (lambda: pop.coeffs(net)) if pop is not None else (lambda: coeffs(net, phip, gp, comps))
    diverged = torch.zeros(net.R, dtype=torch.bool, device=dev)
    base, coarse = log_steps(base_ratio), log_steps(detail_ratio)
    last = None
    for step in range(steps + 1):
        L = loss_fn()
        final = step == steps
        if step % check_every == 0 or final:
            with torch.no_grad():
                c = exact_c()
                rel = c / s
                moved = last is None or bool(((rel - last[0]).abs() > dc).any()) or bool(
                    ((c.abs().clamp_min(1e-300).log() - last[1]).abs() > dlog).any())
                Lp = L.detach() if pop is not None else loss_per_run(net(phip), yp, active)
                diverged |= ~torch.isfinite(L) | ~torch.isfinite(Lp) | (Lp > 1e6)
                done = (((rel - 1).abs() < stop_tol) | ~on).all(1) & (Lp < stop_loss)
                stop = bool((done | diverged).all())
            if moved or base(step) or final or stop:
                last = (rel, c.abs().clamp_min(1e-300).log())
                t = t0 + eta * step
                log['t'].append(t)
                log['c'].append(c.cpu().numpy())
                log['L_train'].append(L.detach().cpu().numpy())
                log['L_probe'].append(Lp.cpu().numpy())
                log['diverged'].append(diverged.cpu().numpy())
                rising = bool((((rel > 0.02) & (rel < 0.98)) & on).any())
                if rising or coarse(step) or final or stop:
                    log['t_K'].append(t)
                    K_, v_ = mobility(net, exact_c, len(comps), loss_fn)
                    log['K'].append(K_.numpy())
                    log['v'].append(v_.numpy())
                if coarse(step) or final or stop:
                    with torch.no_grad():
                        acts = net.forward(phip)
                        log['t_detail'].append(t)
                        log['align'].append(alignment(acts, xp, comps).numpy())
                        log['spec'].append(spectra(acts).numpy())
            if stop:
                break
        grads = torch.autograd.grad(L[~diverged].sum(), net.params)
        with torch.no_grad():
            for p, gr, m in zip(net.params, grads, net.frozen):
                p -= eta * gr * m
    log['t_end'] = t0 + eta * step
    log['converged'] = done.cpu().numpy()  # per run, at the last check: every active component within stop_tol
    return log


def stack(log):
    """Lists of per-log arrays -> arrays with time first."""
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in log.items()}
