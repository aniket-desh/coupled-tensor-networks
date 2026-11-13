"""Discrete optimizers, written as one update so any gradient function can drive them."""
import numpy as np


def step(theta, g, state, opt):
    """One update. opt: ('gd', eta), ('hb', eta, beta) or ('adam', lr, beta1, beta2). `state` is a dict the
    caller keeps between steps (velocity, moments, step count)."""
    if opt[0] == 'gd':
        return theta - opt[1] * g
    if opt[0] == 'hb':
        _, eta, beta = opt
        state['v'] = beta * state.get('v', 0.0) - eta * g
        return theta + state['v']
    _, lr, b1, b2 = opt
    state['n'] = state.get('n', 0) + 1
    state['m'] = b1 * state.get('m', 0.0) + (1 - b1) * g
    state['w'] = b2 * state.get('w', 0.0) + (1 - b2) * g * g
    m_hat = state['m'] / (1 - b1 ** state['n'])
    w_hat = state['w'] / (1 - b2 ** state['n'])
    return theta - lr * m_hat / (np.sqrt(w_hat) + 1e-8)
