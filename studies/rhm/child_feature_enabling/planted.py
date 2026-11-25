"""Four-arm planted sweep (EXPERIMENT.md, section 1): tied or untied child tables, with (s=1) or without (s=0) the
easy root target, under population squared-loss gradient flow on the two-level four-class grammar.

Root classes are the signs (U, UV); the left subtree carries U and the right V. Predictions are b h_L for sU and
w h_L h_R for UV. All arms start at b = w = 0 with unit-norm child tables of alignment delta with the planted sign.

Two integrations of each arm, with the same solver and tolerances:
  reduced     the closed ODEs of notes/child-feature-enabling/{toy,untied}.tex in (r, b, w) or (r_L, r_R, b, w);
  full table  the 16 values of each child table plus the two readouts, with the velocity written directly as an
              average over the 256 equiprobable leaf strings. The table starts at h = delta H + sqrt(1-delta^2) N
              (N: seed-21 Gaussian Walsh coefficients, projected off H, unit norm); both untied copies start equal.
The full-table velocity is first compared with autograd of the 256-row loss at a generic point (nonzero readouts,
unequal tables). Observables: r_i = E[h_i H_i]/sqrt(E[h_i^2]), the trained hard fit D_h = 1 - E[(UV - w h_L h_R)^2]
and the frozen-child ceiling A_h = max_w D_h = r_L^2 r_R^2. Events are the first upward crossings of r_L = 1/2,
r_R = 1/2 and D_h = 1/2. Each trajectory runs to its last event or t = 4/delta^2 + 300, then TAIL time units further
(for the D_h, A_h overlay only; event times are unaffected).

  python -m studies.rhm.child_feature_enabling.planted
"""
import itertools
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from studies.rhm.child_feature_enabling import record, toy_check

HERE = Path(__file__).resolve().parent
DELTAS = (0.08, 0.04, 0.02, 0.01, 0.005)
RTOL, ATOL, TAIL, N_SAMPLES, TABLE_SEED = 2e-11, 2e-13, 15.0, 200, 21
EVENTS = ('left_half', 'right_half', 'hard_half')
COLUMNS = ('r_L', 'r_R', 'b', 'w', 'D_h', 'A_h')
PHI = (1 + np.sqrt(5)) / 2
LAMBDA_1, LAMBDA_2 = 1 / PHI, (np.sqrt(PHI ** 2 + 4 * PHI) - PHI) / 2


def strings():
    """The grammar by enumeration: per leaf pair its hidden sign H, the Walsh characters, and per string the two
    pair indices and the root targets (same construction as toy_check.enumeration_check)."""
    pairs = list(itertools.product(range(4), repeat=2))
    parent = [x ^ y for x, y in pairs]
    root = [2 * (a >> 1) + ((a ^ b) >> 1) for a, b in pairs]
    ids = np.array(list(itertools.product(range(16), repeat=2)))
    classes = np.array([root[4 * parent[i] + parent[j]] for i, j in ids])
    H = np.array([1.0 - 2 * (a >> 1) for a in parent])
    walsh = np.array([[(-1.0) ** ((i & j).bit_count()) for j in range(16)] for i in range(16)])
    easy, hard = 1.0 - 2 * (classes >> 1), 1.0 - 2 * (classes % 2)
    assert np.array_equal(easy, H[ids[:, 0]]) and np.array_equal(hard, H[ids[:, 0]] * H[ids[:, 1]])
    return H, walsh, ids, easy, hard


H, WALSH, IDS, EASY, HARD = strings()


# ------------------------------------------------------------------------------------------------ reduced flows

def reduced_rhs(tied, s):
    if tied:
        return lambda t, y: toy_check.reduced_flow(y, s)  # state (r, b, w)

    def flow(t, y):
        rl, rr, b, w = y
        Rl, Rr = 1 + b * b + w * w, 1 + w * w
        al, ar = rl * np.sqrt(Rl), rr * np.sqrt(Rr)
        return ((1 - rl * rl) * (s * b + w * ar) / np.sqrt(Rl), (1 - rr * rr) * w * al / np.sqrt(Rr),
                s * al - b * Rl, al * ar - w * Rl * Rr)
    return flow


def reduced_obs(tied):
    """y -> (r_L, r_R, b, w, D_h, A_h); y may carry trailing sample axes."""
    def obs(y):
        if tied:
            r, b, w = y
            R = 1 + b * b + 2 * w * w
            return np.stack([r, r, b, w, r ** 4 - (w * R - r * r) ** 2, r ** 4])
        rl, rr, b, w = y
        Rl, Rr = 1 + b * b + w * w, 1 + w * w
        return np.stack([rl, rr, b, w, (rl * rr) ** 2 - (w * np.sqrt(Rl * Rr) - rl * rr) ** 2, (rl * rr) ** 2])
    return obs


# --------------------------------------------------------------------------------------------- full-table flows

def split(y, tied):
    return (y[:16], y[:16], y[16], y[17]) if tied else (y[:16], y[16:32], y[32], y[33])


def table_rhs(tied, s):
    """Unit-rate gradient flow in coordinates orthonormal for <f, g> = E[f g] over the 16 leaf pairs, i.e.
    dh/dt = -16 dL/dh for the raw table entries, with 2L = E[(sU - b h_L)^2 + (UV - w h_L h_R)^2]."""
    def flow(t, y):
        hl, hr, b, w = split(y, tied)
        product = np.outer(hl, hr)
        resid = np.outer(H, H) - w * product  # hard residual on the 16 x 16 equiprobable (left, right) pairs
        easy = s * H - b * hl
        dl = b * easy + w * (resid @ hr) / 16
        dr = w * (hl @ resid) / 16
        tables = [dl + dr] if tied else [dl, dr]  # a tied table sums the gradients of its two occurrences
        return np.concatenate(tables + [[(easy * hl).mean(), (resid * product).mean()]])
    return flow


def table_obs(tied):
    def obs(y):
        hl, hr, b, w = split(y, tied)
        product = np.outer(hl, hr)
        target = np.outer(H, H)
        fit = 1 - ((target - w * product) ** 2).mean()
        ceiling = (target * product).mean() ** 2 / (product ** 2).mean()
        return np.array([(hl * H).mean() / np.sqrt((hl ** 2).mean()), (hr * H).mean() / np.sqrt((hr ** 2).mean()),
                         b, w, fit, ceiling])
    return obs


def initial_table(delta):
    g = torch.Generator().manual_seed(TABLE_SEED)
    noise = WALSH @ torch.randn(16, generator=g, dtype=torch.float64).numpy()
    noise -= (noise * H).mean() * H
    noise /= np.sqrt((noise ** 2).mean())
    return delta * H + np.sqrt(1 - delta ** 2) * noise


def gradient_check(delta=0.08):
    """Table and readout velocities against autograd of the 256-row loss in Walsh-coefficient coordinates, at a
    generic point: nonzero readouts and (for untied) unequal tables."""
    errors = {}
    g = torch.Generator().manual_seed(5)
    hl = initial_table(delta)
    hr_other = hl + 0.3 * WALSH @ torch.randn(16, generator=g, dtype=torch.float64).numpy() / 4
    il, ir = torch.tensor(IDS).T
    for tied in (True, False):
        for s in (0, 1):
            hr = hl if tied else hr_other
            b, w = 0.23, -0.17
            walsh = torch.tensor(WALSH)
            cl = (walsh.T @ torch.tensor(hl) / 16).requires_grad_()
            cr = cl if tied else (walsh.T @ torch.tensor(hr) / 16).requires_grad_()
            bt, wt = torch.tensor(b, dtype=torch.float64, requires_grad=True), torch.tensor(w, dtype=torch.float64, requires_grad=True)
            tl, tr = walsh @ cl, walsh @ cr
            loss = ((s * torch.tensor(EASY) - bt * tl[il]).square()
                    + (torch.tensor(HARD) - wt * tl[il] * tr[ir]).square()).mean() / 2
            params = (cl, bt, wt) if tied else (cl, cr, bt, wt)
            grads = torch.autograd.grad(loss, params)
            reference = np.concatenate([(walsh @ -x).numpy() for x in grads[:-2]]
                                       + [[-grads[-2].item(), -grads[-1].item()]])
            velocity = table_rhs(tied, s)(0.0, np.concatenate(([hl] if tied else [hl, hr]) + [[b, w]]))
            parts = dict(tables=slice(0, -2), easy_readout=slice(-2, -1), hard_readout=slice(-1, None))
            for name, sl in parts.items():
                errors[f"{'tied' if tied else 'untied'}_s{s}_{name}"] = float(np.abs(velocity[sl] - reference[sl]).max())
    assert max(errors.values()) < 1e-12, errors
    return dict(delta=delta, point='b=0.23, w=-0.17; untied right table = left + 0.3 x seed-5 Walsh noise / 4',
                tolerance=1e-12, max_abs_errors=errors)


# -------------------------------------------------------------------------------------------------- integration

def integrate(rhs, y0, obs, delta, rtol, atol):
    """Radau to the last of the three first upward crossings (or the cap), then TAIL further. Returns the event
    times (None = not reached by the cap), the final time, and a function evaluating the state at given times."""
    crossings = [lambda t, y, k=k: obs(y)[k] - 0.5 for k in (0, 1, 4)]
    for f in crossings:
        f.direction = 1

    def all_crossed(t, y):
        return min(f(t, y) for f in crossings)
    all_crossed.direction, all_crossed.terminal = 1, True
    cap = 4 / delta ** 2 + 300
    kw = dict(method='Radau', rtol=rtol, atol=atol, dense_output=True)
    first = solve_ivp(rhs, (0, cap), y0, events=(*crossings, all_crossed), **kw)
    assert first.success, first.message
    events = {name: (float(t[0]) if t.size else None) for name, t in zip(EVENTS, first.t_events)}
    t_end = float(first.t[-1])
    segments = [first]
    if first.status == 1:  # all three crossed before the cap
        tail = solve_ivp(rhs, (t_end, t_end + TAIL), first.y[:, -1], **kw)
        assert tail.success, tail.message
        segments.append(tail)

    def state(times):
        times = np.asarray(times, dtype=float)
        out = segments[0].sol(np.minimum(times, t_end))
        if len(segments) > 1 and (times > t_end).any():
            out[:, times > t_end] = segments[1].sol(times[times > t_end])
        return out
    return events, float(segments[-1].t[-1]), state, sum(s.nfev for s in segments)


def run_arm(tied, s, delta, rtol=RTOL, atol=ATOL, samples=True):
    y0 = (delta, 0.0, 0.0) if tied else (delta, delta, 0.0, 0.0)
    events, t_final, state, nfev = integrate(reduced_rhs(tied, s), y0, reduced_obs(tied), delta, rtol, atol)
    table = initial_table(delta)
    y0_table = np.concatenate(([table] if tied else [table, table]) + [[0.0, 0.0]])
    t_events, t_final_table, t_state, t_nfev = integrate(table_rhs(tied, s), y0_table, table_obs(tied), delta,
                                                         rtol, atol)
    out = dict(tied=tied, s=s, delta=delta, rtol=rtol, atol=atol, cap=4 / delta ** 2 + 300,
               events=events, events_full_table=t_events, t_final=t_final, nfev=nfev, nfev_full_table=t_nfev,
               censored=[k for k in EVENTS if events[k] is None])
    out['event_time_abs_diff_full_vs_reduced'] = {
        k: (None if events[k] is None or t_events[k] is None else abs(events[k] - t_events[k])) for k in EVENTS}
    if samples:
        times = np.geomspace(1e-2, t_final, N_SAMPLES)
        at_events = [events[k] for k in EVENTS if events[k] is not None]
        reduced = reduced_obs(tied)(state(times))
        full = np.stack([table_obs(tied)(y) for y in t_state(np.minimum(times, t_final_table)).T], axis=1)
        out.update(t=times.tolist(), **{c: row.tolist() for c, row in zip(COLUMNS, reduced)},
                   at_events={k: dict(zip(COLUMNS, reduced_obs(tied)(state([events[k]]))[:, 0].tolist()))
                              for k in EVENTS if events[k] is not None},
                   max_abs_diff_full_vs_reduced=dict(zip(COLUMNS, np.abs(full - reduced).max(1).tolist())))
        # D_h can never exceed the frozen-child ceiling.
        assert (reduced[4] <= reduced[5] + 1e-12).all() and len(at_events) == 3 - len(out['censored'])
    return out


def main():
    t0 = time.time()
    meta = record.provenance(__file__, toy_check.__file__)
    meta.update(method='scipy solve_ivp Radau, float64', rtol=RTOL, atol=ATOL, tail=TAIL, deltas=DELTAS,
                table_seed=TABLE_SEED, columns=COLUMNS, thresholds=dict(r_L=0.5, r_R=0.5, D_h=0.5),
                lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)
    check = gradient_check()
    print('gradient check passed:', max(check['max_abs_errors'].values()), flush=True)
    arms = []
    for tied in (True, False):
        for s in (1, 0):
            for delta in DELTAS:
                arm = run_arm(tied, s, delta)
                arms.append(arm)
                print(f"{'tied' if tied else 'untied':>6} s={s} delta={delta:<6} "
                      + '  '.join(f"{k} {arm['events'][k]:.4f}" if arm['events'][k] is not None else f'{k} censored'
                                  for k in EVENTS)
                      + f"  full-table diff {max(v for v in arm['event_time_abs_diff_full_vs_reduced'].values() if v is not None):.1e}",
                      flush=True)
    # Solver sensitivity at the smallest delta: both tolerances divided by four.
    sensitivity = []
    for tied in (True, False):
        for s in (1, 0):
            base = next(a for a in arms if a['tied'] == tied and a['s'] == s and a['delta'] == DELTAS[-1])
            tight = run_arm(tied, s, DELTAS[-1], RTOL / 4, ATOL / 4, samples=False)
            sensitivity.append(dict(tied=tied, s=s, delta=DELTAS[-1], rtol=RTOL / 4, atol=ATOL / 4,
                                    events=tight['events'], events_full_table=tight['events_full_table'],
                                    change={k: tight['events'][k] - base['events'][k] for k in EVENTS},
                                    change_full_table={k: tight['events_full_table'][k] - base['events_full_table'][k]
                                                       for k in EVENTS}))
    meta['seconds'] = time.time() - t0
    record.dump(HERE / 'results' / 'planted' / 'planted.json',
                dict(meta=meta, gradient_check=check, arms=arms, tolerance_sensitivity=sensitivity))


if __name__ == '__main__':
    main()
