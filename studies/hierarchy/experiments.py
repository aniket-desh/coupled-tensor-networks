"""Hierarchy sweeps. Usage: python -m studies.hierarchy.experiments [order race scaling overshoot loss dense rerun signs stages]

Predictions, written before these sweeps. (The cousin prefactor was corrected after seeing one number,
delay 7.6e4 at delta=1e-3, in checks.py; everything else here is untested at time of writing.)

With tau = int a dt, early growth is p_k' = 2 q_G^2 s_k p_k, where q_G^2 = sum_(k in G) p_k. A group with
top strength s_top and sibling ratio x = s_sib/s_top blows up at
    tau_G = I(x) / (2 s_top delta^2),   I(x) = int_1^inf du / (u^2 + u^(1+x)).
P1 Race: the group with smaller tau_G learns first, as delta -> 0. Stronger siblings speed a group up.
P2 Cousin delay: dt_cousin = (tau_B - tau_A) / a, a = C_A^(1/5), C_A = sum of the winning group's targets. ~ delta^-2.
P3 Sibling delay: dt_sibling ~ ((1-x)/x) log(1/delta). Only logarithmic.
P4 Consequence: a weak sibling beats a stronger cousin unless tau_B - tau_A is O(delta^2 log(1/delta)) of tau_A.
   Group sharing is what does this: u_only should order by strength, group_only should keep the order.
P5 Escape laws after the first component saturates: d log c_sib/dt = 2 a q^2 s_sib (constant),
   d log c_cous/dt = 4 sqrt(a) s_cous sqrt(c_cous) (power law).
P6 Late-time signs of eps follow the slowest eigenvector of K(s). Overshoot in stages follows
   eps_learned = -K_ll^-1 K_lj eps_j while component j is learning.
"""
from pathlib import Path

import numpy as np

from ctn import io
from ctn.measure import dissipation, half_times
from ctn.measure import overshoot as overshoot_of

from .model import ARMS, G, NET, K_matrix, load, run_dense, run_reduced, single_output_net, tau

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
FLOOR = 1e-5  # overshoot below this is indistinguishable from zero (see checks.py)
S_MAIN = [1.0, 0.3, 0.8, 0.25]
S_ALT = [1.0, 0.6, 0.9, 0.3]


def save(name, obj, **arrays):
    io.save_json(RESULTS / f'{name}.json', obj)
    if arrays:
        np.savez_compressed(RESULTS / f'{name}.npz', **arrays)


def L_of(c, s):
    return 0.5 * ((s[:, None] - c) ** 2).sum(0)


def times_to_loss(t, c, s):
    L = L_of(c, s)
    return {f'{f:g}': float(t[np.argmax(L < f * L[0])]) for f in [.5, .1, .01, 1e-3, 1e-6]}


# ------------------------------------------------------------ order map
def order_one(args):
    arm, x, y, delta = args
    s = np.array([1.0, x, y, y / 3])
    t, c, done = ARMS[arm].run(s, delta, stop_loss=1e-12)
    return dict(arm=arm, x=x, y=y, delta=delta, half=half_times(t, c, s).tolist(), done=done)


def order():
    xs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ys = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, .995]
    grid = [('hierarchy', x, y, d) for d in [.1, .03, .01, .003, .001] for x in xs for y in ys]
    grid += [(a, x, y, .01) for a in ['u_only', 'group_only', 'independent'] for x in xs for y in ys]
    save('order', dict(results=io.pmap(order_one, grid, chunksize=4)))


# ------------------------------------------------------------ race between groups, random strengths
def race_one(args):
    seed, delta = args
    s = np.random.default_rng([seed, 7]).uniform(0.1, 1.0, 4)
    t, c, done = ARMS['hierarchy'].run(s, delta, stop_loss=1e-12)
    half = half_times(t, c, s)
    tops = [int(np.argmax(np.where(G == g, s, -1))) for g in (0, 1)]
    sibs = [int(np.argmin(np.where(G == g, s, 9))) for g in (0, 1)]
    taus = [tau(s[tops[g]], s[sibs[g]]) for g in (0, 1)]
    win = np.argmin(taus)
    C = s[G == win].sum()
    return dict(seed=seed, delta=delta, s=s.tolist(), half=half.tolist(), taus=taus, done=done, tops=tops,
                pred=int(taus[1] < taus[0]),  # predicted first group
                obs=int(half[tops[1]] < half[tops[0]]), strength_first=int(s[tops[1]] > s[tops[0]]),
                gap=half[tops[1 - win]] - half[tops[win]], gap_pred=(max(taus) - min(taus)) / (C ** (1 / 5) * delta ** 2))


def race():
    save('race', dict(results=io.pmap(race_one, [(seed, d) for d in [.01, .003, .001] for seed in range(500)], chunksize=4)))


# ------------------------------------------------------------ delay scaling and escape laws
def scaling():
    s = np.array(S_MAIN)
    rows = []
    for d in np.geomspace(.1, .001, 11):
        t, c, _ = ARMS['hierarchy'].run(s, d)
        rows.append(dict(delta=d, half=half_times(t, c, s).tolist(), O=overshoot_of(c, s).tolist()))
    # Escape-law curves at the smallest delta: measured d log c/dt from the flow's own samples.
    t, c, _ = ARMS['hierarchy'].run(s, .001)
    keep = np.unique(np.linspace(0, t.size - 1, 60000).astype(int))
    rate = np.array([np.gradient(np.log(ck), t) for ck in c])
    C_A = s[:2].sum()
    theory = dict(sib_rate=2 * s[1], cous_coef=4 * np.sqrt(C_A ** (1 / 5)) * s[2], a_cousin=C_A ** (1 / 5))
    save('scaling', dict(s=S_MAIN, rows=rows, theory=theory, x=s[1] / s[0], tau_gap=tau(s[2], s[3]) - tau(s[0], s[1]),
                         a=C_A ** (1 / 5)),
         t=t[keep], c=c[:, keep], rate=rate[:, keep], half=half_times(t, c, s))


# ------------------------------------------------------------ overshoot pattern
def adiabatic_stage(s, order_, stage):
    """Peak of eps_learned while component j = order_[stage] learns: solve K_ll e_l + K_lj e_j = 0
    self-consistently along c_j in (0, s_j). Unlearned components are held at 1e-12 of their targets."""
    j = order_[stage]
    learned = list(order_[:stage])
    peak = np.zeros(4)
    for cj in np.linspace(0.005, 0.995, 199) * s[j]:
        e = np.zeros(4)
        for _ in range(30):
            c = np.maximum(1e-12 * s, 0.0)
            c[learned] = s[learned] - e[learned]
            c[j] = cj
            K = K_matrix(c)
            e[learned] = np.linalg.solve(K[np.ix_(learned, learned)], -K[learned, j] * (s[j] - cj))
        peak[learned] = np.maximum(peak[learned], -e[learned] / s[learned])
    return peak


def stage_table(s_list, delta=.001):
    """Superseded by stages(): the windows overlap, so a learned component's earlier peak leaks into later stages."""
    out = []
    for s_ in s_list:
        s = np.array(s_)
        t, c, _ = ARMS['hierarchy'].run(s, delta)
        half = half_times(t, c, s)
        order_ = np.argsort(half)
        for stage in range(1, 4):
            lo = half[order_[stage - 1]]
            hi = half[order_[stage + 1]] if stage < 3 else t[-1]
            mask = (t > lo) & (t < hi)
            measured = np.maximum((c[:, mask] / s[:, None] - 1).max(1), 0.0)
            out.append(dict(s=s_, stage=stage, learning=int(order_[stage]), learned=order_[:stage].tolist(),
                            measured=measured.tolist(), adiabatic=adiabatic_stage(s, order_, stage).tolist()))
    return out


def late_signs(c, s, lo, hi):
    """Sign of eps = s - c in the window lo < L < hi, against the slowest eigenvector of K(s) up to overall sign."""
    e = s[:, None] - c
    L = 0.5 * (e ** 2).sum(0)
    win = (L < hi) & (L > lo)
    return np.sign(np.median(e[:, win], axis=1)) if win.any() else np.zeros(4)


def overshoot_one(args):
    seed, delta = args
    s = np.random.default_rng([seed, 11]).uniform(0.1, 1.0, 4)
    t, c, done = ARMS['hierarchy'].run(s, delta)
    O, half = overshoot_of(c, s), half_times(t, c, s)
    sign_obs = late_signs(c, s, 1e-11, 1e-9)
    w, v = np.linalg.eigh(K_matrix(s))
    sign_th = np.sign(v[:, 0])
    return dict(seed=seed, s=s.tolist(), O=O.tolist(), half=half.tolist(), done=done, count=int((O > FLOOR).sum()),
                order=np.argsort(half).tolist(), last_overshoots=bool(O[np.argmax(half)] > FLOOR),
                match=bool((sign_obs == sign_th).all() or (sign_obs == -sign_th).all()), gap=(w[1] - w[0]) / w[0],
                sign_obs=sign_obs.tolist(), sign_th=sign_th.tolist())


def overshoot():
    res = io.pmap(overshoot_one, [(i, .01) for i in range(1500)], chunksize=8)
    arms = {}
    for a in ARMS:
        for name, s_ in [('main', S_MAIN), ('alt', S_ALT)]:
            s = np.array(s_)
            t, c, _ = ARMS[a].run(s, .01)
            arms[f'{a}_{name}'] = dict(O=overshoot_of(c, s).tolist(), half=half_times(t, c, s).tolist())
    save('overshoot', dict(delta=.01, floor=FLOOR, results=res, arms=arms, stages=stage_table([S_MAIN, S_ALT])))


# ------------------------------------------------------------ loss
def loss():
    delta, out, arrays = .01, {}, {}
    for name, s_ in [('main', S_MAIN), ('alt', S_ALT)]:
        s = np.array(s_)
        for a in ARMS:
            t, c, _ = ARMS[a].run(s, delta)
            out[f'{a}_{name}'] = dict(half=half_times(t, c, s).tolist(), t_L=times_to_loss(t, c, s), O=overshoot_of(c, s).tolist(), s=s_)
            keep = np.unique(np.linspace(0, t.size - 1, 8000).astype(int))
            arrays[f'{a}_{name}_t'], arrays[f'{a}_{name}_c'] = t[keep], c[:, keep]
        # Loss budget on the exact 4D flow: which pairs' coupling carries how much dissipation.
        arm = ARMS['hierarchy']
        t, c, _ = run_reduced(arm.coeffs(arm.x0(delta)), s, stop_loss=1e-20, sub=20)
        D = dissipation(np.array([K_matrix(ci) for ci in c.T]), s[:, None] - c)
        diag = sum(D[k, k] for k in range(4))
        near = D[0, 1] + D[1, 0] + D[2, 3] + D[3, 2]
        far = sum(D[i, j] + D[j, i] for i in (0, 1) for j in (2, 3))
        L = L_of(c, s)
        integ = lambda v: float(np.trapezoid(v, t))
        pos = lambda v: integ(np.maximum(v, 0))
        neg = lambda v: integ(np.minimum(v, 0))
        out[f'budget_{name}'] = dict(L_start=L[0], L_end=L[-1], diag=integ(diag), near=integ(near), far=integ(far),
                                     near_pos=pos(near), near_neg=neg(near), far_pos=pos(far), far_neg=neg(far),
                                     closure=L[0] - L[-1] - integ(diag) - integ(near) - integ(far))
    save('loss', dict(delta=delta, results=out), **arrays)


# ------------------------------------------------------------ dense random init
def dense_one(args):
    s_, sigma, shared, seed = args
    s = np.array(s_)
    rng = np.random.default_rng([seed, int(sigma * 1000), int(100 * s_[1]), int(shared)])
    if shared:
        t, c, done = run_dense(NET, NET.random(sigma, rng), s, max_steps=60000)
        half, O = half_times(t, c, s), overshoot_of(c, s)
    else:
        half, O, done = np.zeros(4), np.zeros(4), True
        for k in range(4):
            net = single_output_net(k)
            t, ck, dk = run_dense(net, net.random(sigma, rng), s[[k]], max_steps=60000)
            half[k], O[k] = half_times(t, ck, s[[k]])[0], overshoot_of(ck, s[[k]])[0]
            done = done and dk
    return dict(s=s_, sigma=sigma, shared=shared, seed=seed, half=half.tolist(), O=O.tolist(), done=done)


def dense():
    grid = [(s_, sg, sh, seed) for s_ in [S_MAIN, S_ALT] for sg in [.2, .1, .05] for sh in [True, False] for seed in range(24)]
    save('dense', dict(results=io.pmap(dense_one, grid)))


# ------------------------------------------------------------ rerun of runs that exhausted DOP853's step budget
def rerun_order(r):
    s = np.array([1.0, r['x'], r['y'], r['y'] / 3])
    t, c, done = ARMS[r['arm']].run(s, r['delta'], stop_loss=1e-12, stiff=True, max_steps=2_000_000)
    return dict(r, half=half_times(t, c, s).tolist(), done=done, solver='radau')


def rerun_race(r):
    s = np.array(r['s'])
    t, c, done = ARMS['hierarchy'].run(s, r['delta'], stop_loss=1e-12, stiff=True, max_steps=2_000_000)
    half = half_times(t, c, s)
    tops, win = r['tops'], int(np.argmin(r['taus']))
    return dict(r, half=half.tolist(), obs=int(half[tops[1]] < half[tops[0]]), done=done, solver='radau',
                gap=half[tops[1 - win]] - half[tops[win]])


def rerun():
    """Unfinished runs only, stored apart from the original sweeps; model.merged overlays them."""
    out = dict(order=io.pmap(rerun_order, [r for r in load('order')['results'] if not r['done']]),
               race=io.pmap(rerun_race, [r for r in load('race')['results'] if not r['done']]))
    # independent_main never finished in the loss and overshoot arm tables either.
    s = np.array(S_MAIN)
    t, c, done = ARMS['independent'].run(s, .01, stiff=True, max_steps=2_000_000)
    out['independent_main'] = dict(half=half_times(t, c, s).tolist(), O=overshoot_of(c, s).tolist(), done=done, s=S_MAIN,
                                   t_L=times_to_loss(t, c, s))
    save('rerun', out)


# ------------------------------------------------------------ late-time signs, read much closer to convergence
def signs_one(r):
    """Same teacher and init as overshoot_one, with the sign window moved from L ~ 1e-10 down to L ~ 1e-20."""
    s = np.array(r['s'])
    t, c, done = ARMS['hierarchy'].run(s, .01, stop_loss=1e-21)
    obs = late_signs(c, s, 1e-21, 1e-19)
    th = np.array(r['sign_th'])
    return dict(seed=r['seed'], done=done, sign_obs=obs.tolist(), match=bool((obs == th).all() or (obs == -th).all()), gap=r['gap'])


def signs():
    save('signs', dict(results=io.pmap(signs_one, [r for r in load('overshoot')['results'] if not r['match']], chunksize=2)))


# ------------------------------------------------------------ overshoot per stage, windows matched to the adiabatic scan
def stages():
    """Each stage is only the time when c_j/s_j is in (0.005, 0.995), the adiabatic scan's range."""
    out = []
    for s_ in [S_MAIN, S_ALT]:
        s = np.array(s_)
        t, c, done = ARMS['hierarchy'].run(s, .001, stiff=True, max_steps=2_000_000, sub=20)
        order_ = np.argsort(half_times(t, c, s))
        Ks = K_matrix(s)
        for stage in range(1, 4):
            j = order_[stage]
            first = np.argmax(c[j] > .005 * s[j])
            last = first + np.argmax(c[j][first:] > .995 * s[j])
            learned = order_[:stage]
            measured = (c[:, first:last + 1] / s[:, None] - 1).max(1)
            out.append(dict(s=s_, done=done, learning=int(j), learned=learned.tolist(), K_learned=np.diag(Ks)[learned].tolist(),
                            K_learning=float(Ks[j, j]), measured=measured[learned].tolist(),
                            adiabatic=adiabatic_stage(s, order_, stage)[learned].tolist()))
    save('stages', dict(delta=.001, results=out))


if __name__ == '__main__':
    io.main(globals(), ['order', 'race', 'scaling', 'overshoot', 'loss', 'dense'], HERE)
