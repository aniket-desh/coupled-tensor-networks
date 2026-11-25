"""Small algebra and numerical checks for the two-level four-class grammar.

From the repository root: python3 -m studies.rhm.child_feature_enabling.toy_check
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch


def enumeration_check(seed=21):
    """Differentiate a loss over all 256 leaf sequences in orthonormal coordinates."""
    pairs = list(itertools.product(range(4), repeat=2))
    parent = [x ^ y for x, y in pairs]
    assert np.bincount(parent).tolist() == [4] * 4
    root = [2 * (a >> 1) + ((a ^ b) >> 1) for a, b in pairs]
    assert np.bincount(root).tolist() == [4] * 4
    leaf_ids = list(itertools.product(range(16), repeat=2))
    classes = torch.tensor([root[4 * parent[i] + parent[j]] for i, j in leaf_ids])
    assert torch.bincount(classes).tolist() == [64] * 4
    idx_l, idx_r = torch.tensor(leaf_ids).T
    dtype = torch.float64
    easy = (1 - 2 * (classes >> 1)).to(dtype)
    hard = (1 - 2 * (classes % 2)).to(dtype)
    H = torch.tensor([1 - 2 * (a >> 1) for a in parent], dtype=dtype)
    walsh = torch.tensor([[(-1) ** ((i & j).bit_count()) for j in range(16)]
                          for i in range(16)], dtype=dtype)
    assert torch.equal(walsh.T @ walsh, 16 * torch.eye(16, dtype=dtype))
    generator = torch.Generator().manual_seed(seed)
    coeff = torch.randn(16, generator=generator, dtype=dtype) / 4
    coeff += 0.6 * (walsh.T @ H / 16)
    coeff.requires_grad_()
    b, w = torch.tensor([0.23, -0.17], dtype=dtype, requires_grad=True).unbind()
    h = walsh @ coeff
    R, a = h.square().mean(), (h * H).mean()
    left, product = h[idx_l], h[idx_l] * h[idx_r]
    best_w = torch.linalg.lstsq(product[:, None], hard).solution[0]
    oracle_fit = 1 - (hard - best_w * product).square().mean()
    errors = {'frozen_oracle': abs(float((oracle_fit - a**4 / R**2).detach()))}
    for s in (0, 1):
        loss = ((s * easy - b * left).square() + (hard - w * product).square()).mean() / 2
        grad = torch.autograd.grad(loss, (coeff, b, w), retain_graph=True)
        dh_dt = (s * b + 2 * w * a) * H - (b**2 + 2 * w**2 * R) * h
        velocities = (walsh.T @ dh_dt / 16, s * a - b * R, a**2 - w * R**2)
        errors[f's{s}_gradient_max_abs'] = max(float((g + v).abs().max().detach())
                                                for g, v in zip(grad, velocities))
        # Differentiate R - b^2 - 2w^2 in the direction of the enumerated gradient.
        rate = -2 * (coeff @ grad[0]) + 2 * b * grad[1] + 4 * w * grad[2]
        errors[f's{s}_balance_rate_abs'] = abs(float(rate.detach()))
    assert max(errors.values()) < 1e-12, errors
    return dict(seed=seed, leaf_sequences=len(classes), counts=torch.bincount(classes).tolist(),
                coordinate_system='unnormalized Walsh characters, uniform input inner product',
                dtype='float64', max_error_tolerance=1e-12, errors=errors)


def reduced_flow(state, s):
    """Unit-rate flow for normalized child alignment and the easy and hard weights."""
    r, b, w = state
    R = 1 + b*b + 2*w*w
    return ((1 - r*r) * (s*b / np.sqrt(R) + 2*w*r),
            s*r*np.sqrt(R) - b*R,
            R*r*r - w*R*R)


def learning_times(delta, s, q=0.5, p=0.5, *, rtol=2e-11, atol=2e-13):
    """Return first child-alignment and trained hard-variance threshold times."""
    if s not in (0, 1) or not (0 < delta < q < 1 and 0 < p < 1):
        raise ValueError('Expected s in {0,1}, 0 < delta < q < 1 and 0 < p < 1.')

    def child(t, state):
        return state[0] - q

    def hard(t, state):
        r, b, w = state
        R = 1 + b*b + 2*w*w
        return r**4 - (w*R - r*r)**2 - p

    child.direction = hard.direction = 1
    state = (delta, 0.0, 0.0)
    start = 0.0
    end = 12 + (0.3 / delta**2 if s == 0 else 2*np.log(1 / delta))
    times = [None, None]
    while any(value is None for value in times):
        sol = solve_ivp(lambda t, y: reduced_flow(y, s), (start, end), state,
                        method='Radau', rtol=rtol, atol=atol, events=(child, hard))
        if not sol.success:
            raise RuntimeError(sol.message)
        for i, events in enumerate(sol.t_events):
            if times[i] is None and events.size:
                times[i] = float(events[0])
        state = sol.y[:, -1]
        start, end = end, 2*end
    return dict(child=times[0], hard=times[1])


def untied_escape_check():
    """Check the second escape using raw signal/nuisance amplitudes on each branch."""
    def flow(t, state):
        a_l, n_l, a_r, n_r, b, w = state
        R_l, R_r = a_l**2 + n_l**2, a_r**2 + n_r**2
        decay_l, decay_r = b*b + w*w*R_r, w*w*R_l
        return (b + w*a_r - decay_l*a_l, -decay_l*n_l,
                w*a_l - decay_r*a_r, -decay_r*n_r,
                a_l - b*R_l, a_l*a_r - w*R_l*R_r)

    def left_half(t, state):
        a_l, n_l = state[:2]
        return a_l / np.hypot(a_l, n_l) - 0.5

    def half_fit(t, state):
        a_l, n_l, a_r, n_r, _, w = state
        return 2*w*a_l*a_r - w*w*(a_l*a_l+n_l*n_l)*(a_r*a_r+n_r*n_r) - 0.5

    half_fit.terminal = True
    deltas = [1e-3, 1e-4, 1e-5]
    times = []
    for d in deltas:
        nuisance = np.sqrt(1 - d*d)
        sol = solve_ivp(flow, (0, 100), (d, nuisance, d, nuisance, 0., 0.),
                        method='DOP853', rtol=2e-11, atol=2e-14, events=(left_half, half_fit))
        if not sol.success or any(len(t) != 1 for t in sol.t_events):
            raise RuntimeError(f'Expected both thresholds for delta={d}: {sol.message}')
        times.append([float(t[0]) for t in sol.t_events])
    slopes = np.diff(times, axis=0) / np.diff(np.log(1/np.asarray(deltas)))[:, None]
    phi = (1 + np.sqrt(5)) / 2
    lambda_2 = (np.sqrt(phi*phi + 4*phi) - phi) / 2
    predicted = np.array([phi, phi + 1/lambda_2])
    max_error = float(np.max(np.abs(slopes[-1] - predicted)))
    assert max_error < 0.001, (slopes, predicted)
    return dict(method='DOP853, original six signal/nuisance and readout coordinates',
                rtol=2e-11, atol=2e-14, s=1, q_left=0.5, p=0.5, deltas=deltas,
                columns=['left_time', 'hard_time'], times=times, finite_difference_slopes=slopes.tolist(),
                predicted_slopes=predicted.tolist(), final_max_slope_error=max_error,
                final_max_slope_tolerance=0.001)


def main():
    # Subtract the independently predicted pole before measuring a logarithmic slope.
    seeds = [0.04, 0.02, 0.01]
    settings = dict(q=0.5, p=0.5, rtol=2e-11, atol=2e-13)
    times = [learning_times(d, 0, **settings) for d in seeds]
    residual = [t['child'] - 1/(4*d*d) for d, t in zip(seeds, times)]
    slopes = np.diff(residual) / np.diff(np.log(1 / np.asarray(seeds)))
    assert abs(slopes[-1] - 2.5) < 0.01, slopes
    report = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__)),
        enumeration=enumeration_check(), untied_second_escape=untied_escape_check(),
        hard_only=dict(method='Radau', s=0, deltas=seeds, **settings, times=times,
                       child_times_minus_pole=residual, slopes_in_log_inverse_delta=slopes.tolist(),
                       theoretical_coefficient=2.5, final_slope_error=float(abs(slopes[-1] - 2.5)),
                       final_slope_tolerance=0.01))
    out = Path(__file__).with_name('results') / 'toy_checks.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
