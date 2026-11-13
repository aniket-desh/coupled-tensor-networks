"""Shared contract for the language-process experiments (studies/lang/<experiment>/).

A run's cfg is a flat dict:
  process: {'kind': ..., **params, 'pseed': instance seed}   (studies/lang/processes.py)
  model: 'tree' | 'chain';  r;  tie: 'none' | 'level';  perm: list of 16 positions or None;  seed (init);
  init: 'identity' (default) | 'variance';  init_eps (identity init noise scale, default 0.1);  init_scale (variance
  init multiplier, default 1);  arm: a label;  task: always 'lang' (so studies.rhm.train.evaluate reports KL to Bayes).
Training reuses studies/rhm/train.py: Adam, batch 256, 100-update warmup then cosine to 0.1 x peak, online data.
Evaluation sets are fixed per process instance: N_DIAG examples (diagnostic, logged through training) and N_TEST
(final), sampled on the CPU from seeded generators, with exact Bayes posteriors.
Checkpoints: studies.rhm.common.save format, '<run dir>/ckpt_<step>.pt' holding {'cfg', 'step', 'state'}."""
import torch

from ctn.tree import ChainTN, TreeTN, identity_init
from studies.lang import processes as P

DIAG_SEED, TEST_SEED, PROBE_SEED = 21_000, 31_000, 41_000
N_DIAG, N_TEST = 20_000, 50_000


def process(cfg):
    return P.make(cfg['process'])


def build(cfg):
    p = process(cfg)
    if cfg['model'] == 'chain':
        model = ChainTN(16, p.V_in, cfg['r'], p.O, seed=cfg['seed'])
    else:
        model = TreeTN(16, p.V_in, cfg['r'], p.O, tie=cfg.get('tie', 'none'), perm=cfg.get('perm'),
                       scale=cfg.get('init_scale', 1.0), seed=cfg['seed'])
    if cfg.get('init', 'identity') == 'identity':
        identity_init(model, cfg.get('init_eps', 0.1))
    return model


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model = build(ck['cfg']).to(device)
    model.load_state_dict(ck['state'])
    return model, ck['cfg'], ck['step']


def _seed(base, spec):
    """A deterministic seed per process instance (Python's hash() is salted per process, so it can't be used)."""
    return base + sum(ord(c) * (i + 1) for i, c in enumerate(P.key(spec)))


def fixed_set(spec, n, base):
    """(x, y, p_Bayes, strata or None), CPU tensors, deterministic in (spec, base)."""
    p = P.make(spec)
    g = torch.Generator().manual_seed(_seed(base, spec))
    xs, ys = [], []
    for i in range(0, n, 8192):
        x, y = p.sample_xy(min(8192, n - i), g)
        xs.append(x), ys.append(y)
    x, y = torch.cat(xs), torch.cat(ys)
    return x, y, p.bayes(x).float(), p.strata(x)


def chain_width(spec, r_tree):
    """Chain width whose parameter count r + V r + r^3 + O r is closest to the untied r_tree tree's."""
    from ctn.tree import tree_params
    p = P.make(spec)
    target = tree_params(16, p.V_in, r_tree, p.O)
    return min(range(2, 8 * r_tree), key=lambda w: abs(w + p.V_in * w + w ** 3 + p.O * w - target)), target


def perm_split(seed, n=16):
    """A seeded leaf permutation with no adjacent pair (2k, 2k+1) kept as model siblings (as in rhm-controls)."""
    g = torch.Generator().manual_seed(500 + seed)
    while True:
        p = torch.randperm(n, generator=g).tolist()
        if all({p[2 * j], p[2 * j + 1]} != {2 * k, 2 * k + 1} for j in range(n // 2) for k in range(n // 2)):
            return p
