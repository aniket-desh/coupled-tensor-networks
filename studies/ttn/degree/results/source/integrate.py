"""Two integrators for autonomous flows dy/dt = fun(y).

solve: scipy DOP853 with a fixed step cap and dense output, terminal on events. Used where exact event times
    matter or a dense interpolant is resampled (refine). The cap matters: without it DOP853 takes steps so long
    that its dense interpolant dominates the error.
capped: DOP853 (or Radau) stepped by hand with a cap on how far the state moves per step, so plateaus of length
    ~1/delta^depth cost almost nothing while transitions stay finely sampled.
"""
import numpy as np
from scipy.integrate import Radau, solve_ivp
from scipy.integrate._ivp.rk import DOP853


def solve(fun, y0, t_max, done=None, events=(), max_step=0.5, rtol=1e-12, atol=1e-14):
    """Integrate to t_max; stop where done(y) crosses zero. Further `events` (y -> scalar, with optional
    .terminal and .direction) follow done in sol.t_events."""
    evs = []
    for e in ([done] if done else []) + list(events):
        ev = (lambda f: lambda t, y: f(y))(e)
        ev.terminal = getattr(e, 'terminal', e is done)
        ev.direction = getattr(e, 'direction', 0)
        evs.append(ev)
    return solve_ivp(lambda t, y: fun(y), (0, t_max), np.asarray(y0, float), method='DOP853', rtol=rtol, atol=atol,
                     max_step=max_step, dense_output=True, events=evs or None)


def refine(sol, sub=20):
    """Time grid that keeps every adaptive step and subdivides it, so peaks between steps are resolved."""
    t = sol.t
    return np.unique(np.concatenate([np.linspace(t[i], t[i + 1], sub, endpoint=False)
                                     for i in range(len(t) - 1)] + [t[-1:]]))


def capped(fun, y0, stop, cap=0.05, t_max=1e15, sub=6, max_steps=400_000, scale=None, jac=None, rtol=None, atol=None):
    """Step until stop(y) is true. The step cap is cap/|f|_inf (suited to log coordinates), or with `scale` the
    relative cap cap*|y|/|f| for raw parameters. Each step is resampled at `sub` points of its dense output.
    Passing `jac` selects Radau: once a component has converged its fast contraction limits DOP853 to steps of
    about 0.2 for the whole wait until the next component, which can exhaust max_steps.
    Returns t, Y (state x times), and whether stop was reached."""
    y0 = np.array(y0, float)
    if jac is None:
        solver = DOP853(lambda t, y: fun(y), 0.0, y0, t_max, rtol=rtol or 1e-12, atol=atol or 1e-13)
    else:
        solver = Radau(lambda t, y: fun(y), 0.0, y0, t_max, rtol=rtol or 1e-10, atol=atol or 1e-11, jac=lambda t, y: jac(y))
    T, Y = [0.0], [y0]
    for _ in range(max_steps):
        if solver.status != 'running':
            break
        size = np.abs(solver.f).max() if scale is None else np.linalg.norm(solver.f) / max(np.linalg.norm(solver.y), 1e-300)
        solver.max_step = cap / max(size, 1e-300)
        solver.step()
        local = solver.dense_output()
        for tt in np.linspace(solver.t_old, solver.t, sub + 1)[1:]:
            T.append(tt)
            Y.append(local(tt))
        if stop(solver.y):
            break
    return np.array(T), np.array(Y).T, bool(stop(solver.y))
