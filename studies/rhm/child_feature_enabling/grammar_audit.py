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
