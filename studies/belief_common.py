"""Probe and model helpers shared by the belief-geometry experiments (studies/rhm/rhm-*-belief-*, studies/hmm).

Probes map activations a (N, p) to a target b (N, D) whose rows are probability vectors.
  affine      bhat = W a + c, least squares (the Simplex probe; the same fit as rhm-belief-geometry/analyze.py).
  projective  bhat = q / sum(q), q = W a + c: the belief up to normalization. A tensor network's natural state is an
              unnormalized message (a chain's h_t = W(h_{t-1}, token) has the form of HMM filtering,
              eta_t ~ eta_{t-1} T^(x); a tree's merge has the form of the grammar's upward message), whose scale varies
              from input to input, so the belief is a ratio of linear functions of the state. The fit is nonlinear
              least squares by L-BFGS from the affine solution, on standardized features. Both probes are invariant to
              an invertible linear change of the activations (a bond gauge)."""
import numpy as np
import torch


def fit_predict(A_fit, B_fit, A_test, ridge=1e-8):
    """Affine least squares; returns bhat for A_test. Ridge is relative to the mean diagonal (rank guard only)."""
    mu_a, mu_b = A_fit.mean(0), B_fit.mean(0)
    X, Y = A_fit - mu_a, B_fit - mu_b
    G = X.T @ X
    G += ridge * G.diagonal().mean().clamp_min(1e-30) * torch.eye(len(G), dtype=G.dtype, device=G.device)
    return (A_test - mu_a) @ torch.linalg.solve(G, X.T @ Y) + mu_b


def score(b, bhat, mu):
    """R2 against the fit mean `mu`, and RMSE = sqrt(mean_rows ||b - bhat||^2)."""
    sse, sst = ((b - bhat) ** 2).sum().item(), ((b - mu) ** 2).sum().item()
    return dict(sse=sse, sst=sst, n=len(b), R2=1 - sse / sst if sst > 0 else float('nan'), rmse=float(np.sqrt(sse / len(b))))


def projective_fit_predict(A_fit, B_fit, A_test, iters=300, ridge=1e-8):
    """Projective probe; returns (bhat for A_test, final fit loss / affine fit loss). Features are projected on their
    principal axes and scaled to unit variance (directions with relative variance < 1e-12 dropped), which changes
    nothing in the model class and conditions the optimization. Starts from the affine solution, for which sum(q) = 1
    on the fit set, so the fit loss can only go down; if the optimizer leaves the finite region, the affine solution
    is returned (ratio 1)."""
    mu = A_fit.mean(0)
    X = A_fit - mu
    evals, evecs = torch.linalg.eigh(X.T @ X / len(X))
    keep = evals > 1e-12 * evals.max()
    P = evecs[:, keep] / evals[keep].sqrt()
    Z, Zt = X @ P, (A_test - mu) @ P
    c0 = B_fit.mean(0)
    G = Z.T @ Z
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype, device=G.device)
    W0 = torch.linalg.solve(G, Z.T @ (B_fit - c0)).contiguous()
    W, c = W0.clone().requires_grad_(True), c0.contiguous().clone().requires_grad_(True)

    def predict(Zx, W, c):
        q = Zx @ W + c
        return q / q.sum(1, keepdim=True)

    loss_of = lambda W, c: ((predict(Z, W, c) - B_fit) ** 2).sum(1).mean()
    base = loss_of(W0, c0).item()
    opt = torch.optim.LBFGS([W, c], max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=20,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = loss_of(W, c)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        final = loss_of(W, c).item()
        if not np.isfinite(final) or final > base:
            return predict(Zt, W0, c0), 1.0
        return predict(Zt, W, c), final / max(base, 1e-300)


def chain_states(model, x):
    """Causal read-out of a ctn.tree.ChainTN: states (B, T, r) with states[:, t] = h_t after the first t tokens (h_0 the
    learned start state), and logits (B, T, O) = U h_t, so logits[:, t] predicts x_t from x_<t, as in the transformer."""
    _, hs = model(x, acts=True)
    states = torch.stack([model.h0.expand(x.shape[0], -1)] + hs[:-1], 1)
    return states, states @ model.U.T
