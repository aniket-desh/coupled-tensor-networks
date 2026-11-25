"""Enumerate root productions and project root contrasts onto additive child functions.

From the repository root: python3 -m studies.rhm.child_feature_enabling.grammar_audit
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space, subspace_angles
from sympy.polys.matrices import DomainMatrix
from sympy import ZZ
import torch

from studies.rhm import common, grammar


def exact_rank(array):
    return DomainMatrix.from_list(array.tolist(), ZZ).rank()


def audit(rules, vocab):
    n, m, arity = rules.shape
    assert arity == 2
    pairs = rules.reshape(-1, 2)
    assert len(np.unique(pairs, axis=0)) == n * m
    child = np.eye(vocab, dtype=int)
    D = np.column_stack([child[pairs[:, 0]], child[pairs[:, 1]]])
    E = np.repeat(np.eye(n, dtype=int), m, axis=0)
    rank_d = exact_rank(D)
    rank_joint = exact_rank(np.column_stack([D, E]))
    rank_marginals = exact_rank(D.T @ E)
    pure = n - rank_marginals
    additive = n + rank_d - rank_joint - 1

    # E.T E = m I. The columns of Q give unit centered label functions on edges.
    Q = E @ null_space(np.ones((1, n))) / np.sqrt(m)
    fitted = D @ np.linalg.lstsq(D, Q, rcond=None)[0]
    fractions = np.linalg.eigvalsh(fitted.T @ fitted)
    # Independent subspace-angle calculation, including zero singular values
    # when the child design has smaller rank than the contrast space.
    principal = np.cos(subspace_angles(D, Q)) ** 2
    principal = np.sort(np.pad(principal, (0, n - 1 - len(principal))))
    np.testing.assert_allclose(fractions, principal, atol=1e-12)
    assert np.count_nonzero(abs(fractions) < 1e-12) == pure
    assert np.count_nonzero(abs(1 - fractions) < 1e-12) == additive
    return dict(n_edges=n * m, rank_D=rank_d, rank_DtE=rank_marginals,
                rank_joined=rank_joint, pure_interaction_dim=pure,
                centered_additive_dim=additive,
                additive_fraction_min=float(fractions[0]),
                additive_fraction_max=float(fractions[-1]),
                additive_fractions=fractions.tolist())


def root_contrasts(rules, vocab, tol=1e-10):
    """Most (c_e) and least (c_h) additive unit-variance root contrasts of one top rule table.

    rules: (n, m, 2) integer array of ordered child pairs. On the n*m equiprobable legal rows, E one-hot encodes the
    root label and X the two child symbols (concatenated one-hots); H has orthonormal columns spanning the centered
    label vectors and Q = E H / sqrt(m) is their orthonormal image on rows. The eigenvalues of Q^T P_X Q are the
    fractions of a contrast's variance that the best additive function f(u) + g(v) of the true children explains.
    c = sqrt(n) H v has mean 0 and mean square 1 under the uniform root law.

    Sign: the largest-magnitude entry is positive (lowest index on a tie). If an extreme eigenvalue is degenerate
    within tol, the contrast is instead the normalized projection onto its eigenspace of the first centered class
    indicator (in class order) whose projection has norm above tol.
    """
    n, m, _ = rules.shape
    pairs = rules.reshape(-1, 2)
    child = np.eye(vocab)
    X = np.column_stack([child[pairs[:, 0]], child[pairs[:, 1]]])
    E = np.repeat(np.eye(n), m, axis=0)
    H = null_space(np.ones((1, n)))
    Q = E @ H / np.sqrt(m)
    fitted = X @ np.linalg.lstsq(X, Q, rcond=None)[0]  # P_X Q
    values, vectors = np.linalg.eigh(fitted.T @ fitted)
    out = dict(additive_fractions=values.tolist(), degeneracy_tol=tol,
               sign_rule='largest-magnitude entry positive, lowest index on a tie')
    for name, extreme, gap in (('easy', values[-1], values[-1] - values[-2]),
                               ('hard', values[0], values[1] - values[0])):
        space = H @ vectors[:, np.abs(values - extreme) <= tol]  # orthonormal in label space
        if space.shape[1] == 1:
            c = np.sqrt(n) * space[:, 0]
            top = np.flatnonzero(np.abs(c) >= np.abs(c).max() - 1e-12)[0]
            c = c * np.sign(c[top])
        else:
            centered = np.eye(n) - 1 / n
            proj = next(p for p in (space @ (space.T @ e) for e in centered) if np.linalg.norm(p) > tol)
            c = np.sqrt(n) * proj / np.linalg.norm(proj)
        # The Rayleigh quotient of the returned vector must reproduce the eigenvalue it was selected for.
        edge = E @ c
        eta = float(np.sum((X @ np.linalg.lstsq(X, edge, rcond=None)[0]) ** 2) / np.sum(edge ** 2))
        assert abs(eta - extreme) < 1e-10 and abs(c.mean()) < 1e-12 and abs(np.mean(c ** 2) - 1) < 1e-12
        out[name] = dict(vector=c.tolist(), additive_fraction=eta, neighbor_gap=float(gap),
                         eigenspace_dim=int(space.shape[1]))
    return out


def main():
    # Uniform binary parity has zero marginal signal; a left-copy rule is additive.
    parity = np.array([[[0, 0], [1, 1]], [[0, 1], [1, 0]]])
    left_copy = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
    for rules, expected in [(parity, (1, 0)), (left_copy, (0, 1))]:
        ref = audit(rules, 2)
        assert (ref['pure_interaction_dim'], ref['centered_additive_dim']) == expected

    tables = []
    for seed in (0, 1, 2):
        top = common.rules(dict(common.GRAMMAR, seed=seed))[0].numpy()
        tables.append(dict(seed=seed, rules=top.tolist(), **audit(top, common.GRAMMAR['v'])))
    source_hashes = {mod.__name__: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                     for mod in (common, grammar)}
    result = dict(grammar=common.GRAMMAR, torch=str(torch.__version__),
                  source_sha256=source_hashes, tables=tables)
    path = Path(__file__).with_name('results') / 'grammar_audit.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
