"""Shared contract for the RHM experiments: grammar configs, tasks, models and checkpoints.

Task 'root': observe all 16 tokens, predict the root symbol (n classes). The grammar is unambiguous, so the Bayes
predictor is exact (accuracy 1, loss 0).
Task 'last': observe tokens 0..14, predict token 15 (v classes). Leaf 15 always receives the mask token v (leaf
vocabulary v + 1), so the student keeps the same tree. The Bayes reference is `grammar.posterior_token`.

A run's config (cfg) is a flat dict:
  model: 'tree' | 'chain';  r;  tie: 'none' | 'level';  perm: list of 16 leaf indices or None;  seed (init);
  optional (added by the training code): init: 'variance' (default) | 'identity', init_eps (default 0.1); other extra
  keys (e.g. 'arm') are labels that build() ignores;
  task: 'root' | 'last';  grammar: dict(v, n, m, s, L, seed).
Checkpoints: torch.save({'cfg': cfg, 'step': step, 'state': model.state_dict()}, '<run dir>/ckpt_<step>.pt').
`load(path)` returns (model, cfg, step)."""
import torch

from ctn.tree import ChainTN, TreeTN, identity_init
from studies.rhm import grammar as G

GRAMMAR = dict(v=16, n=16, m=4, s=2, L=4)  # the main setting: binary, depth 4 (16 tokens), 16 symbols per level, m = 4
CHECKPOINTS = [0, 100, 300, 1000, 3000, 10000, 20000, 40000]


def rules(gcfg):
    return G.sample_rules(gcfg['v'], gcfg['n'], gcfg['m'], gcfg['s'], gcfg['L'], seed=gcfg['seed'])


def n_leaves(gcfg):
    return gcfg['s'] ** gcfg['L']


def vocab_in(cfg):
    return cfg['grammar']['v'] + (1 if cfg['task'] == 'last' else 0)


def n_out(cfg):
    return cfg['grammar']['n'] if cfg['task'] == 'root' else cfg['grammar']['v']


def task_batch(trees, cfg):
    """(x, y) for the task: x (B, 16) long inputs to the student, y (B,) long targets."""
    L, v = cfg['grammar']['L'], cfg['grammar']['v']
    tok = trees[L]
    if cfg['task'] == 'root':
        return tok, trees[0][:, 0]
    x = tok.clone()
    x[:, -1] = v
    return x, tok[:, -1]


def build(cfg):
    n, V, O = n_leaves(cfg['grammar']), vocab_in(cfg), n_out(cfg)
    if cfg['model'] == 'chain':
        model = ChainTN(n, V, cfg['r'], O, seed=cfg['seed'])
    else:
        model = TreeTN(n, V, cfg['r'], O, tie=cfg.get('tie', 'none'), perm=cfg.get('perm'), seed=cfg['seed'])
    if cfg.get('init', 'variance') == 'identity':
        identity_init(model, cfg.get('init_eps', 0.1))
    return model


def save(path, model, cfg, step):
    torch.save({'cfg': cfg, 'step': step, 'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model = build(ck['cfg']).to(device)
    model.load_state_dict(ck['state'])
    return model, ck['cfg'], ck['step']


# Fixed evaluation sets, generated on the CPU with a seeded generator so every machine gets the same trees.
# (Added by the training code; analysis should call diag_trees rather than re-sample.)
DIAG_SEED, TEST_SEED = 20_000, 30_000
N_DIAG, N_TEST = 20_000, 50_000


def diag_trees(gcfg, B=N_DIAG):
    """The fixed diagnostic set for a grammar (trees dict, CPU): every run on this grammar logs against it."""
    return G.sample_trees(B, rules(gcfg), seed=DIAG_SEED + gcfg['seed'])


def test_trees(gcfg, B=N_TEST):
    """The separate final test set for a grammar (trees dict, CPU), used only for end-of-run numbers."""
    return G.sample_trees(B, rules(gcfg), seed=TEST_SEED + gcfg['seed'])
