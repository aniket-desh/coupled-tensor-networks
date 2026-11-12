"""Overshoot experiments. Usage: python -m studies.shared_feature.experiments [e1 e1a e2 e3 e4 e5a e5b e5c e6 e6b]

Everything uses s1 = 1 (see the scale invariance in model.py), so rho = s2/s1 and the init size are the only
parameters. Writes results/<exp>.{json,npz} next to this file.
"""
from pathlib import Path

import numpy as np

from ctn import integrate, io, optim
from ctn.factors import log_flow
from ctn.measure import dissipation, half_time, overshoot

from .model import (K_INDEPENDENT, SHARED, TARGET, K_matrix, S, c0_balanced, flow, heteroclinic_overshoot,
                    net, reduced_c, solve_reduced)

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
E1 = np.zeros((2, 2, 2))
E1[0, 0, 0] = 1.0  # one-hot of c1, so pull_back(theta, E1) = grad c1


def save(name, obj, **arrays):
    io.save_json(RESULTS / f'{name}.json', obj)
    if arrays:
        np.savez_compressed(RESULTS / f'{name}.npz', **arrays)


def train_gd(network, theta, s, eta, t_max):
    """Plain gradient descent; returns time and coefficients at every step."""
    T = TARGET(s)
    n = int(t_max / eta)
    c = np.zeros((2, n + 1))
    c[:, 0] = network.coeffs(theta)
    for i in range(n):
        theta = theta - eta * network.grad(theta, T)
        c[:, i + 1] = network.coeffs(theta)
    return eta * np.arange(n + 1), c


# ---------------------------------------------------------------- E1: anatomy of one run
def e1():
    delta, eta = 0.01, 0.02
    arrays, summary = {}, {}
    for rho in [0.2, 0.5]:
        s = S(rho)
        t, c = reduced_c(solve_reduced(c0_balanced(delta), s))
        e = s[:, None] - c
        # Loss rate decomposition dL/dt = -(K11 e1^2 + K22 e2^2 + 2 K12 e1 e2).
        K = np.array([K_matrix(ci) for ci in c.T])
        ti, ci = reduced_c(solve_reduced(np.full(2, delta ** 3), s, K_INDEPENDENT))
        # Real training: GD on the 10 parameters, aligned init and one random init.
        tg, cg = train_gd(SHARED, SHARED.aligned(delta), s, eta, t[np.argmax(c[1] > 0.999 * s[1])] * 1.5)
        rnd = flow(SHARED, SHARED.random(0.1, np.random.default_rng(1)), s, 1e5, stop_loss=1e-16)
        tr = integrate.refine(rnd, 5)
        cr = SHARED.coeffs(rnd.sol(tr))
        tag = f'rho{rho}'
        arrays.update({f'{tag}_t': t, f'{tag}_c': c, f'{tag}_e': e,
                       f'{tag}_K11': K[:, 0, 0], f'{tag}_K22': K[:, 1, 1], f'{tag}_K12': K[:, 0, 1],
                       f'{tag}_ti': ti, f'{tag}_ci': ci, f'{tag}_tg': tg[::10], f'{tag}_cg': cg[:, ::10],
                       f'{tag}_tr': tr, f'{tag}_cr': cr})
        i_peak = np.argmax(c[0])
        L = 0.5 * (e ** 2).sum(0)
        i_cross = np.argmax(c[0] >= s[0])
        summary[tag] = dict(
            overshoot_ode=overshoot(c, s)[0], overshoot_gd=overshoot(cg, s)[0],
            overshoot_random_init=overshoot(cr, s)[0],
            overshoot_independent=overshoot(ci, s).max(),
            t_half_1=half_time(t, c[0], s[0]), t_half_2=half_time(t, c[1], s[1]),
            t_half_1_independent=half_time(ti, ci[0], s[0]),
            t_half_2_independent=half_time(ti, ci[1], s[1]),
            t_cross=t[i_cross], t_peak=t[i_peak], c2_at_peak_over_s2=c[1, i_peak] / s[1],
            L_at_cross=L[i_cross], L1_at_peak=0.5 * e[0, i_peak] ** 2, L_at_peak=L[i_peak],
            L1_frac_of_L_at_peak=0.5 * e[0, i_peak] ** 2 / L[i_peak],
            max_L1_frac=(0.5 * e[0, i_cross:] ** 2 / L[i_cross:]).max(),
            monotone_L_violation=np.maximum(np.diff(L), 0).max(),
        )
    save('e1', dict(delta=delta, eta=eta, summary=summary), **arrays)


# ---------------------------------------------------------------- E1a: the shared entry a over training
def e1a():
    """Same runs as E1 (delta = 0.01), but on the 10 raw parameters, so a is read off u = (a, 0) rather than
    inferred. Balance (a^2 = r1^2 + r2^2, r_k = b_k, all conserved at zero from this init) predicts a^3 = c1 + c2."""
    delta = 0.01
    arrays, summary = {}, {}
    for rho in [0.5, 0.2]:
        s = S(rho)
        T = TARGET(s)
        sol = flow(SHARED, SHARED.aligned(delta), s, 1e5, stop_loss=1e-16)
        t = integrate.refine(sol, 5)
        y = sol.sol(t)
        U = y[:2]
        R, B = y[2:6].reshape(2, 2, -1), y[6:10].reshape(2, 2, -1)
        a, c = U[0], SHARED.coeffs(y)
        a_inf = (1 + rho) ** (1 / 3)
        adot = np.array([-SHARED.grad(v, T)[0] for v in y.T])
        i_cross, i_peak = np.argmax(c[0] >= 1), np.argmax(c[0])
        tag = f'rho{rho}'
        arrays.update({f'{tag}_t': t, f'{tag}_a': a, f'{tag}_c': c, f'{tag}_r': R[[0, 1], [0, 1]], f'{tag}_b': B[[0, 1], [0, 1]]})
        summary[tag] = dict(
            a_inf_predicted=a_inf, a_end=a[-1], a_over_final_max_minus_1=a.max() / a_inf - 1, u2_max_abs=np.abs(U[1]).max(),
            balance_a3_max_rel_err=np.abs(a ** 3 / c.sum(0) - 1).max(), adot_min=adot.min(),
            t_half_a=half_time(t, a, a_inf), t_half_c=[half_time(t, c[k], s[k]) for k in range(2)],
            t_cross=t[i_cross], t_peak=t[i_peak], a_over_final_at_cross=a[i_cross] / a_inf, a_over_final_at_peak=a[i_peak] / a_inf,
            r1_over_final_max=(R[0, 0] / np.sqrt(1 / a_inf)).max(), overshoot_c1=overshoot(c, s)[0])
    save('e1a', dict(delta=delta, summary=summary), **arrays)


# ---------------------------------------------------------------- E2: size of overshoot
def e2():
    rhos = np.array([.02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, .995])
    deltas = np.array([.3, .1, .03, .01, .003, .001, 1e-4])
    O1, O2, peak_c2, gap = [np.zeros((len(deltas), len(rhos))) for _ in range(4)]
    for i, d in enumerate(deltas):
        for j, rho in enumerate(rhos):
            s = S(rho)
            t, c = reduced_c(solve_reduced(c0_balanced(d), s))
            O1[i, j], O2[i, j] = overshoot(c, s)
            peak_c2[i, j] = c[1, np.argmax(c[0])] / s[1]
            gap[i, j] = half_time(t, c[1], s[1]) - half_time(t, c[0], s[0])
    # delta -> 0 limit along the saddle connection, two seeds to check seed independence.
    het = np.array([heteroclinic_overshoot(r) for r in rhos])
    het2 = np.array([heteroclinic_overshoot(r, seed=1e-13) for r in rhos])
    # Note's adiabatic estimates: rho^2/12 and the max of x(rho-x)/(3+2x) over x in (0, rho).
    x = np.linspace(0, 1, 200001)[None] * rhos[:, None]
    adiabatic_full = (x * (rhos[:, None] - x) / (3 + 2 * x)).max(1)
    # Solver convergence on the hardest cell (smallest overshoot): a 10x smaller step cap.
    s = S(rhos[0])
    O_fine = overshoot(reduced_c(solve_reduced(c0_balanced(0.03), s, max_step=0.05))[1], s)[0]
    save('e2', dict(solver_check=dict(rho=rhos[0], delta=.03, O_default=O1[2, 0], O_fine=O_fine),
                    het_seed_max_rel_diff=(np.abs(het - het2) / het).max()),
         rhos=rhos, deltas=deltas, O1=O1, O2=O2, peak_c2=peak_c2, gap=gap, het=het, het2=het2, adiabatic_full=adiabatic_full)


# ---------------------------------------------------------------- E3: is overshoot generic?
def slow_fast(s):
    """Eigenvectors of K at the solution; eps -> slow eigenvector at late times."""
    return np.linalg.eigh(K_matrix(s))


def separatrix(s, eta=1e-6, c_floor=1e-10):
    """Strong stable manifold below s: backward integration along the fast eigenvector.
    Backward in time this manifold attracts, so the integration is stable."""
    w, v = slow_fast(s)
    fast = v[:, 1] * np.sign(v[0, 1])  # all-positive Perron vector of K
    rhs = log_flow(K_matrix)
    sol = integrate.solve(lambda x: -rhs(x, s), np.log(s - eta * fast), 1e7, done=lambda x: np.exp(x).min() - c_floor)
    return np.exp(sol.y)


def e3_one(args):
    s, c0, sep_log = args
    t, c = reduced_c(solve_reduced(c0, s), 5)
    e = s[:, None] - c
    O = overshoot(c, s)
    # Predicted winner from which side of the separatrix the init lies on.
    predicted = 2 if np.log(c0[1]) > np.interp(np.log(c0[0]), sep_log[0], sep_log[1]) else 1
    observed = 1 if O[0] > O[1] else 2
    # Trapping claim: after leaving the both-below quadrant the sign of e1 e2 never returns.
    prod = e[0] * e[1]
    big = np.abs(e).min(0) > 1e-7
    mixed = prod[big] < 0
    first = np.argmax(mixed) if mixed.any() else len(mixed)
    trapped = mixed[first:].all()
    # Late-time ratio, measured in the window where e is well above solver noise.
    L = 0.5 * (e ** 2).sum(0)
    win = (L < 1e-9) & (L > 1e-12)
    ratio = np.median(e[0, win] / e[1, win]) if win.sum() else np.nan
    return dict(c0=c0.tolist(), O1=O[0], O2=O[1], predicted=predicted, observed=observed, trapped=bool(trapped),
                late_ratio=ratio)


def e3():
    n, out, arrays = 400, {}, {}
    for rho in [0.2, 0.5, 0.8]:
        s = S(rho)
        sep = separatrix(s)
        sep = sep[:, np.argsort(sep[0])]
        rng = np.random.default_rng(int(rho * 100))
        c0s = s * 10 ** rng.uniform(-4, np.log10(0.95), (n, 2))
        res = io.pmap(e3_one, [(s, c0, np.log(sep)) for c0 in c0s])
        w, v = slow_fast(s)
        theory_ratio = v[0, 0] / v[1, 0]
        # Separatrix near the origin: early-time theory says log p2 = rho log p1 + log kappa.
        p = sep / sep.sum(0) ** (1 / 3)
        lowest = sep[0] < 1e-5
        slope = np.polyfit(np.log(p[0, lowest]), np.log(p[1, lowest]), 1)
        tag = f'rho{rho}'
        ratios = np.array([r['late_ratio'] for r in res], float)
        arrays.update({f'{tag}_sep': sep, f'{tag}_c0': c0s, f'{tag}_observed': np.array([r['observed'] for r in res]),
                       f'{tag}_O1': np.array([r['O1'] for r in res]), f'{tag}_O2': np.array([r['O2'] for r in res]),
                       f'{tag}_ratios': ratios})
        out[tag] = dict(
            n=n, K_star=K_matrix(s).tolist(), eigs=w.tolist(),
            predicted_equals_observed=int(sum(r['predicted'] == r['observed'] for r in res)),
            no_overshoot_runs=int(sum(max(r['O1'], r['O2']) < 1e-8 for r in res)),
            both_overshoot_runs=int(sum(min(r['O1'], r['O2']) > 1e-8 for r in res)),
            c1_overshoots=int(sum(r['observed'] == 1 for r in res)),
            trapped_runs=int(sum(r['trapped'] for r in res)),
            theory_late_ratio=theory_ratio,
            late_ratio_median_over_c1_overshooters=float(np.nanmedian(ratios)),
            late_ratio_max_abs_dev=float(np.nanmax(np.abs(ratios - theory_ratio))),
            late_ratio_nan=int(np.isnan(ratios).sum()),
            sep_slope_near_origin=slope[0], sep_log_kappa=slope[1], smallest_c_on_sep=sep.min(),
        )
    save('e3', out, **arrays)


# ---------------------------------------------------------------- E4: does overshoot cost loss?
def e4():
    delta, out, arrays = 0.01, {}, {}
    for rho in [0.1, 0.2, 0.5, 0.8]:
        s = S(rho)
        T = TARGET(s)
        hit = lambda y: SHARED.coeffs(y)[0] - s[0]
        hit.terminal, hit.direction = True, 1
        pre = flow(SHARED, SHARED.aligned(delta), s, 1e6, events=[hit])
        t_star, th_star = pre.t[-1], pre.y[:, -1]
        assert pre.t_events[1].size == 1

        def grad_full(y):
            return -SHARED.grad(y, T)

        def grad_frozen_u(y):  # stop training the shared feature, private factors keep going
            g = -SHARED.grad(y, T)
            g[:2] = 0.0
            return g

        def grad_clamped(y):  # shared feature keeps training; remove any motion of c1
            g = -SHARED.grad(y, T)
            gc = SHARED.pull_back(y, E1)
            return g - (g @ gc) / (gc @ gc) * gc

        arms = {}
        for name, fn in [('full', grad_full), ('frozen_u', grad_frozen_u), ('clamp_c1', grad_clamped)]:
            sol = flow(SHARED, th_star, s, 1e6, stop_loss=1e-16, fun=fn)
            t = integrate.refine(sol, 5)
            c = SHARED.coeffs(sol.sol(t))
            L = 0.5 * ((s[:, None] - c) ** 2).sum(0)
            arms[name] = dict(
                overshoot=overshoot(c, s).tolist(), final_a=float(SHARED.unpack(sol.y[:, -1])[0][0]),
                t_half_2=half_time(t, c[1], s[1]),
                t_L_1e_2=float(t[np.argmax(L < 1e-2 * L[0])]), t_L_1e_6=float(t[np.argmax(L < 1e-6 * L[0])]),
                int_L=float(np.trapezoid(L, t)),
                final_rel_c_err=float(np.abs(c[:, -1] / s - 1).max()),
            )
            arrays[f'rho{rho}_{name}_t'], arrays[f'rho{rho}_{name}_c'] = t, c
        # Loss budget on the reduced flow, integrating -dL/dt term by term before and after crossing.
        t, c = reduced_c(solve_reduced(c0_balanced(delta), s, stop_loss=1e-20), 200)
        e = s[:, None] - c
        D = dissipation(np.array([K_matrix(ci) for ci in c.T]), e)
        diag, cross = D[0, 0] + D[1, 1], D[0, 1] + D[1, 0]
        i = np.argmax(c[0] >= s[0])
        L = 0.5 * (e ** 2).sum(0)
        budget = dict(L_start=L[0], L_end=L[-1], L_cross=L[i],
                      pre_diag=np.trapezoid(diag[:i + 1], t[:i + 1]), pre_cross=np.trapezoid(cross[:i + 1], t[:i + 1]),
                      post_diag=np.trapezoid(diag[i:], t[i:]), post_cross=np.trapezoid(cross[i:], t[i:]))
        budget['closure_pre'] = L[0] - L[i] - budget['pre_diag'] - budget['pre_cross']
        budget['closure_post'] = L[i] - L[-1] - budget['post_diag'] - budget['post_cross']
        # Baseline: independent modes from the same parameter scale over the whole run.
        ti, ci = reduced_c(solve_reduced(np.full(2, delta ** 3), s, K_INDEPENDENT))
        out[f'rho{rho}'] = dict(t_star=t_star, L_star=SHARED.loss(th_star, T), c2_star_over_s2=SHARED.coeffs(th_star)[1] / s[1],
                                arms=arms, budget=budget,
                                t_half_shared=[half_time(t, c[0], s[0]), half_time(t, c[1], s[1])],
                                t_half_independent=[half_time(ti, ci[0], s[0]), half_time(ti, ci[1], s[1])])
    save('e4', dict(delta=delta, results=out), **arrays)


# ---------------------------------------------------------------- E5: controls for real training
def e5a_one(args):
    """Finite step size on the aligned sector, shared vs unshared."""
    rho, eta, shared, delta = args
    s, network = S(rho), net(shared)
    T = TARGET(s)
    theta = network.aligned(delta)
    c_max = np.full(2, -np.inf)
    for _ in range(int(min(4000 / eta, 400000))):
        theta = theta - eta * network.grad(theta, T)
        c_ = network.coeffs(theta)
        c_max = np.maximum(c_max, c_)
        if not np.isfinite(c_).all() or np.abs(c_).max() > 1e6:
            return dict(rho=rho, eta=eta, shared=shared, O=[np.nan, np.nan], diverged=True)
        if ((s - c_) ** 2).sum() < 1e-20:
            break
    return dict(rho=rho, eta=eta, shared=shared, O=(c_max / s - 1).clip(0).tolist(), diverged=False)


def e5a():
    """Scalar independent mode crosses only if eta K(c) >= 1, i.e. eta >= 1/3 at s1 = 1."""
    etas = [.01, .05, .1, .2, .3, .33, .36, .4, .45, .5, .55, .6]
    grid = [(rho, eta, sh, 0.05) for rho in [.2, .5] for eta in etas for sh in [True, False]]
    save('e5a', dict(delta=0.05, results=io.pmap(e5a_one, grid)))


def e5b_one(args):
    """Population gradient flow from a random dense init in all parameters."""
    rho, sigma, shared, seed = args
    s, network = S(rho), net(shared)
    rng = np.random.default_rng([int(rho * 100), int(sigma * 1e4), int(shared), seed])
    sol = flow(network, network.random(sigma, rng), s, 1e6, stop_loss=1e-16)
    O = overshoot(network.coeffs(sol.sol(integrate.refine(sol, 5))), s)
    return dict(rho=rho, sigma=sigma, shared=shared, seed=seed, O1=O[0], O2=O[1],
                converged=bool(network.loss(sol.y[:, -1], TARGET(s)) < 1e-14), T=sol.t[-1])


def e5b():
    grid = [(rho, sg, sh, seed) for rho in [.2, .5, .8] for sg in [1.0, .3, .1, .03, .01]
            for sh in [True, False] for seed in range(50)]
    save('e5b', dict(results=io.pmap(e5b_one, grid, chunksize=4)))


def e5c_one(args):
    """Full-batch GD on a finite sample (noiseless labels), or fresh-minibatch SGD.
    N = 0 means the exact population loss. batch > 0 selects SGD."""
    rho, N, batch, shared, seed, sigma, eta, t_max = args
    s, network = S(rho), net(shared)
    T = TARGET(s)
    rng = np.random.default_rng([int(rho * 100), N, batch, int(shared), seed])
    theta = network.random(sigma, rng)
    G4 = None
    if N:
        z = np.einsum('ni,nj->nij', rng.standard_normal((N, 2)), rng.standard_normal((N, 2))).reshape(N, 4)
        G4 = z.T @ z / N
    n = int(t_max / eta)
    c = np.zeros((2, n))
    ok = True
    for i in range(n):
        if batch:
            g = network.grad_batch(theta, T, [rng.standard_normal((batch, 2)), rng.standard_normal((batch, 2))])
        else:
            g = network.grad(theta, T, G4)
        theta = theta - eta * g
        c[:, i] = network.coeffs(theta)
        if np.abs(c[:, i]).max() > 1e6 or not np.isfinite(c[:, i]).all():
            ok = False
            c = c[:, :i + 1]
            break
    # Raw overshoot, and overshoot of a running mean over one time unit so SGD jitter doesn't count.
    w = max(int(1 / eta), 1)
    smooth = np.array([np.convolve(ck, np.ones(w) / w, 'valid') for ck in c]) if c.shape[1] > w else c
    return dict(rho=rho, N=N, batch=batch, shared=shared, seed=seed, stable=ok, O_raw=overshoot(c, s).tolist(),
                O_smooth=overshoot(smooth, s).tolist(), final_err=float(np.abs(c[:, -1] / s - 1).max()))


def e5c():
    sigma, eta, t_max = 0.1, 0.005, 400
    grid = []
    for rho in [.2, .5]:
        for shared in [True, False]:
            for seed in range(24):
                grid += [(rho, N, 0, shared, seed, sigma, eta, t_max) for N in [0, 16, 64, 256, 1024, 4096, 16384]]
                grid += [(rho, 0, b, shared, seed, sigma, eta, t_max) for b in [8, 32, 128, 512]]
    save('e5c', dict(sigma=sigma, eta=eta, t_max=t_max, results=io.pmap(e5c_one, grid, chunksize=4)))


# ---------------------------------------------------------------- E6: momentum and Adam
# Predictions written before running:
# - Heavy ball, linearized at a target with curvature K: x'' roots go complex when
#   eta K > (1 - sqrt(beta))^2, so the control should ring (cross) above that line. Below it,
#   shared overshoot should approach the gradient-flow value and the control should stay near 0,
#   though a fast arrival could still cross once in the overdamped regime.
# - Adam normalizes each parameter, so a, r_k, b_k all start rising at about lr per step regardless
#   of s_k. Then c_1 and c_2 rise together and the WEAK mode reaches its target first, so in the
#   shared model the coefficient pushed past target should be c_2, the reverse of gradient flow.
def budget(opt):
    if opt[0] == 'hb':
        return int(min(600 * (1 - opt[2]) / opt[1], 2_000_000))
    # Tiny lr needs more steps. Set per opt, not through a global: spawned workers would not see it.
    return int(min(max(40 / opt[1], 20_000), 3_200_000 if opt[1] < 3e-6 else 500_000))


def e6_run(args):
    rho, opt, shared, init, seed, keep = args
    s, network = S(rho), net(shared)
    T = TARGET(s)
    if init == 'aligned':
        theta = network.aligned(0.03)
    else:
        theta = network.random(0.03, np.random.default_rng([int(rho * 100), int(shared), seed]))
    n_max = budget(opt)
    stride = max(n_max // 6000, 1)
    state, traj = {}, []
    c_max = np.full(2, -np.inf)
    first_peak, returned, above = np.full(2, -np.inf), [False, False], [False, False]
    crossings = np.zeros(2, int)
    first_hit, late, L_prev, L_rise, t_L6, i = [None, None], 0.0, None, 0.0, None, 0
    L0 = 0.5 * (s * s).sum()
    for i in range(n_max):
        theta = optim.step(theta, network.grad(theta, T), state, opt)
        c = network.coeffs(theta)
        if not np.isfinite(c).all() or np.abs(c).max() > 1e6:
            return dict(rho=rho, opt=list(opt), shared=shared, init=init, seed=seed, diverged=True)
        e = s - c
        L = 0.5 * (e * e).sum()
        if L_prev is not None:
            L_rise = max(L_rise, L - L_prev)
        L_prev = L
        if t_L6 is None and L < 1e-6 * L0:
            t_L6 = i + 1
        c_max = np.maximum(c_max, c)
        for k in range(2):
            now_above = c[k] > s[k]
            if now_above != above[k]:
                crossings[k] += abs(e[k]) > 1e-9 * s[k] or now_above
                if now_above and first_hit[k] is None:
                    first_hit[k] = i + 1
                if not now_above and first_hit[k] is not None:
                    returned[k] = True
            above[k] = now_above
            if now_above and not returned[k]:
                first_peak[k] = max(first_peak[k], c[k])
        if i >= 0.9 * n_max:
            late = max(late, np.abs(e / s).max())
        if keep and i % stride == 0:
            traj.append((i, c[0], c[1], L))
        if L < 1e-20:
            break
    out = dict(rho=rho, opt=list(opt), shared=shared, init=init, seed=seed, diverged=False, steps=i + 1, n_max=n_max,
               O=(c_max / s - 1).clip(0).tolist(), first_excursion=(first_peak / s - 1).clip(0).tolist(),
               crossings=crossings.tolist(), first_hit=first_hit, late_amp=late, L_rise=L_rise, t_L6=t_L6,
               final_err=float(np.abs(e / s).max()))
    if keep:
        out['traj'] = traj
    return out


def e6():
    etas = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, .3]
    hb = [('hb', e, b) for b in [0.0, .5, .9, .99] for e in etas if e / (1 - b) >= 3e-4 and e / (1 - b) <= 3.0]
    adam = [('adam', lr, b1, b2) for b1, b2 in [(.9, .999), (0.0, .999), (.9, .9), (0.0, .9)]
            for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1]]
    saved = {('hb', 1e-2, .9), ('hb', 3e-4, .9), ('adam', 1e-3, .9, .999), ('adam', 1e-2, .9, .999), ('adam', 1e-3, .9, .9)}
    grid = [(rho, o, sh, 'aligned', 0, o in saved and rho == .5) for rho in [.2, .5] for o in hb + adam for sh in [True, False]]
    rnd = [('hb', 3e-4, .9), ('hb', 1e-2, .9), ('adam', 1e-3, .9, .999), ('adam', 1e-2, .9, .999)]
    grid += [(rho, o, sh, 'random', seed, False) for rho in [.2, .5] for o in rnd for sh in [True, False] for seed in range(24)]
    res = io.pmap(e6_run, grid)
    arrays = {}
    for r in res:
        if 'traj' in r:
            arrays['_'.join(map(str, r['opt'])) + ('_shared' if r['shared'] else '_control')] = np.array(r.pop('traj')).T
    save('e6', dict(delta=0.03, sigma=0.03, results=res), **arrays)


def e6b():
    """Small-lr tail for Adam: does the overshoot reach a finite limit as lr -> 0 at fixed beta2?
    On the clock lr*steps the gradient EMAs become instantaneous, so the update tends to lr*sign(g),
    whose sliding mode would pin a coefficient at its target and make the overshoot vanish."""
    grid = [(rho, ('adam', lr, b1, .999), sh, 'aligned', 0, False) for rho in [.2, .5]
            for lr in [3e-5, 1e-5, 3e-6, 1e-6] for b1 in [.9, 0.0] for sh in [True, False]]
    save('e6b', dict(delta=0.03, results=io.pmap(e6_run, grid)))


if __name__ == '__main__':
    io.main(globals(), ['e1', 'e1a', 'e2', 'e3', 'e4', 'e5a', 'e5b', 'e5c', 'e6', 'e6b'], HERE)
