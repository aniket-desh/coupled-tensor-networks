"""Planted Boolean-product teachers for the TTN experiments, described by where outputs agree.

Symmetry that shapes these designs (checked in checks.py): for independent +-1 inputs E[phi phi^T] = I, and the leaf
init is isotropic, so permuting a leaf's basis (including the constant phi_0 = 1) is an exact symmetry of population
GD. Product degree is therefore not a property the dynamics see, and the constant is not special. What they see is,
for each pair of outputs, the set of leaves on which the two use the same basis index (their agreement set S), and
where S sits in the tree.

So every teacher here is a full product over the 8 patches (one basis index per leaf, never the constant):
  y1 = s1 * prod_j x^(j)_1                       (coordinate 1 everywhere)
  y2 = s2 * prod_{j in S} x^(j)_1 * prod_{j not in S} x^(j)_2
  y3 = s3 * prod_j x^(j)_3                       (shares no leaf with y1 or y2: the comparison output)
Patches are 0-indexed here (paper's x^(1) is patch 0) and coordinates too (x_1 is coordinate 0).
At the teacher, y1 and y2 share exactly the tree nodes whose leaves all lie in S (shared_cores counts them).
"""
from ctn.ttn import LEAVES_OF

# Agreement sets for y1, y2 (0-indexed leaves). 'quarter' and 'split2' have the same extent, as do 'half' and 'split4';
# only the first of each pair is a subtree.
SETS = {
    'none': (),
    'leaf': (0,),
    'quarter': (0, 1),
    'split2': (0, 2),
    'half': (0, 1, 2, 3),
    'split4': (0, 1, 4, 5),
}


def agreement(S, s=(1.0, 0.5, 0.75), third=True):
    S = SETS.get(S, S)
    comps = [dict(out=0, s=s[0], coords=tuple((j, 0) for j in range(8))),
             dict(out=1, s=s[1], coords=tuple((j, 0 if j in S else 1) for j in range(8)))]
    if third:
        comps.append(dict(out=2, s=s[2], coords=tuple((j, 2) for j in range(8))))
    return comps


def shared_cores(S):
    """Cores y1 and y2 share at the teacher: leaves in S, plus every internal node whose leaves all lie in S.
    The root never counts (each output has its own row of C)."""
    S = set(SETS.get(S, S))
    return len(S) + sum(set(leaves) <= S for leaves in LEAVES_OF.values())


def degree(D, s=1.0):
    """One output, a product of coordinate 1 over patches 0..D-1, with the constant on the rest. Under the isotropic
    init this is the same problem for every D (the symmetry above); with a constant-biased leaf init
    (TTN const_weight > 1/9) each active patch must switch its leaf away from the constant, so D is a real axis."""
    return [dict(out=0, s=s, coords=tuple((j, 0) for j in range(D)))]
