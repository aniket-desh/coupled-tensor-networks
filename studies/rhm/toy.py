"""Two hand-built 3-symbol RHMs at one prefix length, small enough to enumerate, with exact targets and exact trees.

Grammars (v = 3 symbols at every level, m = 2 productions, binary, depth L = 3, so 8 tokens; uniform root):
  sym   every parent produces the other two symbols in either order:  0 -> 12 | 21,  1 -> 02 | 20,  2 -> 01 | 10,
        at all three levels. The parent of a valid pair (b, c), b != c, is the missing symbol -(b + c) mod 3.
  mod   as sym at the two upper levels; at the bottom level  0 -> 12 | 21,  1 -> 02 | 22,  2 -> 01 | 11.
        A parent's FIRST child has the same distribution as in sym, so P(x_5 | half symbol) is unchanged.

Task (the whole population is one batch): the model sees x_1 .. x_4, the last four leaves carry the mask token 3, and
it predicts x_5. Write A for the symbol of the observed half (level 1, node 0). x_5 is reached from A by three
"other symbol" steps with transition K = (11^T - I) / 2, so P(x_5 = y | x_1..4) = (K^3)_{A y} = 1/4 if y = A else 3/8.
There are 384 sequences and 24 legal prefixes, each of probability 1/24. The population loss is the cross-entropy
against these exact targets, and the excess loss is the mean KL(Bayes || model) over the 24 prefixes, in nats.
A uniform predictor has excess loss UNIFORM_KL = 0.016417 nats.

Model: ctn.tree.TreeTN(n=8, V=4, r, O=3). Levels as there: level 0 = 8 leaf bonds, level 1 = 4 bonds (token pairs),
level 2 = 2 bonds (halves; node 1 is the masked half, a constant), root core C. Logits are multilinear, so for a fixed
context they are linear in each bond.
  grouping 'aligned': leaves in order, level-1 nodes 0, 1 see (x_1, x_2), (x_3, x_4).
  grouping 'crossed': perm [0, 2, 1, 3, ...], level-1 nodes 0, 1 see (x_1, x_3), (x_2, x_4).

Centering: softmax ignores a shift of the logits, so everything compared with the target is centered over the three
outputs. The centered Bayes logits are F(A) = log(2/3) (e_A - 1/3), three vectors that sum to zero and span a plane.
"""
import itertools
import math

import numpy as np
import torch

from ctn.tree import TreeTN, identity_init
from studies.rhm import grammar as G
from studies.rhm.train import lr_at

V, MASK, T_OBS, N_LEAVES = 3, 3, 4, 8
SYM = [[(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]
MODB = [[(1, 2), (2, 1)], [(0, 2), (2, 2)], [(0, 1), (1, 1)]]
GRAMMARS = ('sym', 'mod')
PERMS = {'aligned': list(range(8)), 'crossed': [0, 2, 1, 3, 4, 5, 6, 7]}
UNIFORM_KL = math.log(3) - (0.25 * math.log(4) + 0.75 * math.log(8 / 3))  # 0.016417 nats
F_NORM = abs(math.log(2 / 3)) * math.sqrt(2 / 3)   # |F(A)|, the size of one centered Bayes logit vector
F_DIST = abs(math.log(2 / 3)) * math.sqrt(2)       # |F(A) - F(A')|, the distance between two of them


def rules(name):
    """{level: (3, 2, 2) long} in studies.rhm.grammar's format."""
    bottom = MODB if name == 'mod' else SYM
    return {0: torch.tensor(SYM), 1: torch.tensor(SYM), 2: torch.tensor(bottom)}


def population(name):
    """The 24 legal prefixes with exact targets, by enumerating all 384 trees.
    Returns dict: x (24, 8) long model input (tokens then masks), p (24, 3) float64 Bayes P(x_5 | prefix),
    A (24,) symbol of the observed half, cL, cR (24,) symbols of the pairs (x_1 x_2), (x_3 x_4)."""
    R = rules(name)
    trees = [({0: (root,)}, 1 / 3) for root in range(3)]
    for l in range(3):
        nxt = []
        for tree, pr in trees:
            for ks in itertools.product(range(2), repeat=len(tree[l])):
                child = sum((R[l][a, k].tolist() for a, k in zip(tree[l], ks)), [])
                nxt.append(({**tree, l + 1: tuple(child)}, pr / 2 ** len(ks)))
        trees = nxt
    joint, lat = {}, {}
    for tree, pr in trees:
        pre = tree[3][:T_OBS]
        joint.setdefault(pre, np.zeros(3))[tree[3][T_OBS]] += pr
        lat.setdefault(pre, set()).add((tree[1][0], tree[2][0], tree[2][1]))
    assert all(len(s) == 1 for s in lat.values()), 'a prefix parses uniquely'
    keys = sorted(joint)
    x = torch.tensor([list(k) + [MASK] * (N_LEAVES - T_OBS) for k in keys])
    p = torch.tensor(np.array([joint[k] / joint[k].sum() for k in keys]))
    A, cL, cR = (torch.tensor([next(iter(lat[k]))[i] for k in keys]) for i in range(3))
    return dict(x=x, p=p, A=A, cL=cL, cR=cR, n_sequences=len(trees), prefix_prob=np.array([joint[k].sum() for k in keys]))


def center(logits):
    return logits - logits.mean(-1, keepdim=True)


def excess(logits, p):
    """Mean KL(p || softmax(logits)) over the rows, in nats (the excess population loss)."""
    return (p * (p.log() - torch.log_softmax(logits, -1))).sum(-1).mean()


# ------------------------------------------------------------------------------------------ rank lower bounds

def cut_bound(name, rowpos):
    """Lower bound on the width of a bond whose subtree holds the observed positions `rowpos` (0-based, within 0..3).

    For a fixed context the centered logits are linear in the bond, so the matrix with rows = values of x[rowpos] and
    columns = (values of the other observed positions, output y) has rank <= r wherever it is defined. Only legal
    prefixes have a target, so the bound is the largest rank of a submatrix all of whose entries are legal: for every
    set of column contexts, the rows legal with all of them. Exhaustive (at most 2^9 context sets)."""
    pop = population(name)
    Fc = {tuple(k[:T_OBS].tolist()): center(q.log()).numpy() for k, q in zip(pop['x'], pop['p'])}
    colpos = [i for i in range(T_OBS) if i not in rowpos]
    rv = sorted({tuple(k[i] for i in rowpos) for k in Fc})
    cv = sorted({tuple(k[i] for i in colpos) for k in Fc})

    def full(r, c):
        k = [0] * T_OBS
        for i, a in zip(rowpos, r):
            k[i] = a
        for i, a in zip(colpos, c):
            k[i] = a
        return tuple(k)

    best = 0
    for n in range(1, len(cv) + 1):
        for cs in itertools.combinations(cv, n):
            rs = [r for r in rv if all(full(r, c) in Fc for c in cs)]
            if rs:
                M = np.array([[Fc[full(r, c)][y] for c in cs for y in range(3)] for r in rs])
                best = max(best, int(np.linalg.matrix_rank(M, tol=1e-9)))
    return best


def tree_bound(name, grouping):
    """max over the tree's bonds on the observed side of cut_bound: a uniform-width tree needs r >= this."""
    perm = PERMS[grouping]
    cuts = [(perm[i],) for i in range(4)] + [tuple(sorted(perm[0:2])), tuple(sorted(perm[2:4])), (0, 1, 2, 3)]
    return max(cut_bound(name, c) for c in cuts)


# ------------------------------------------------------------------------------------------ models

def build(r, grouping='aligned', seed=0, init='variance', eps=0.1):
    model = TreeTN(N_LEAVES, V + 1, r, V, perm=PERMS[grouping], seed=seed).double()
    if init == 'identity':
        identity_init(model, eps)
    return model


def _readout(model, P):
    """Root core so that logits_o = P[o] . zL when the masked half's bond is e_0."""
    with torch.no_grad():
        model.C.zero_()
        model.C[:, :P.shape[1], 0] = P


def exact_complex(grouping='aligned'):
    """An exact width-2 tree for `sym` (either grouping). Symbol a is the complex number z_a = exp(2 pi i a / 3) as two
    real coordinates; every core is (z, w) -> conj(z w), bilinear over R, which sends a valid pair (b, c) to its
    parent -(b + c) mod 3. Masks are z = 1. The half's bond is z_A with A = x_1 + x_2 + x_3 + x_4 mod 3 for any
    grouping, and logits_o = (2/3) log(2/3) Re(conj(z_o) z_A) = F(A)_o."""
    model = build(2, grouping)
    th = 2 * math.pi * torch.arange(3, dtype=torch.float64) / 3
    z = torch.stack([th.cos(), th.sin()], 1)                        # (3, 2): z_a
    conj_mul = torch.zeros(2, 2, 2, dtype=torch.float64)            # W[a, b, c]: conj((b0 + i b1)(c0 + i c1))
    conj_mul[0, 0, 0], conj_mul[0, 1, 1], conj_mul[1, 0, 1], conj_mul[1, 1, 0] = 1, -1, -1, -1
    with torch.no_grad():
        model.A.zero_()
        model.A[:, :3] = z
        model.A[:, MASK, 0] = 1.0
        for W in model.W:
            W.copy_(conj_mul.expand_as(W))
    _readout(model, (2 / 3) * math.log(2 / 3) * z)
    return model


def exact_onehot(name, r=3):
    """An exact aligned tree for either grammar (width r >= 3, channels 3.. unused): the grammar's own message passing
    on point masses. A complete subtree's bond is the one-hot of its symbol, each core on the observed side is the
    production table (1 on a valid pair's parent, 0 on every forbidden pair), masked subtrees carry e_0, and
    logits = log P(x_5 | A)."""
    model = build(r, 'aligned')
    R = rules(name)
    with torch.no_grad():
        model.A.zero_()
        model.A[:, :3, :3] = torch.eye(3, dtype=torch.float64)
        model.A[:, MASK, 0] = 1.0
        for W in model.W:
            W.zero_()
        for (lvl, nodes), W in zip([(2, (0, 1)), (1, (0,))], model.W):   # W[0]: token pairs -> pair symbols; W[1]: -> A
            for k in range(W.shape[0]):
                if k in nodes:
                    for a in range(3):
                        for b, c in R[lvl][a].tolist():
                            W[k, a, b, c] = 1.0
                else:
                    W[k, 0, 0, 0] = 1.0                                  # masked subtree: constant e_0
    K = (torch.ones(3, 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64)) / 2
    _readout(model, torch.linalg.matrix_power(K, 3).log().T)
    return model


# ------------------------------------------------------------------------------------------ training

def train(model, pop, steps, lr, log_every=100, ckpt_at=()):
    """Full-batch Adam on the population loss, with train.lr_at's schedule (100 warm-up updates, cosine to 0.1 lr).
    Returns (log [(step, excess KL)], {step: state_dict} for steps in ckpt_at plus the last)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, p = pop['x'], pop['p']
    log, ckpts = [], {}
    for step in range(steps + 1):
        loss = excess(model(x), p)
        if step % log_every == 0 or step == steps:
            log.append((step, loss.item()))
        if step in ckpt_at or step == steps:
            ckpts[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == steps or not math.isfinite(loss.item()):
            break
        for g in opt.param_groups:
            g['lr'] = lr_at(step, lr, steps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return log, ckpts


def polish(model, pop, iters=1000):  # = POLISH_ITERS, defined below
    """L-BFGS (strong Wolfe) on the same population loss, from where Adam stopped. Adam's step floor leaves the excess
    loss near 1e-10; the theorems are about exact solutions, so the measures are read after this polish.
    Returns (excess KL after it, dict(n_iter, func_evals, capped)); capped means it stopped on the iteration or
    function-evaluation limit (iters and 1.25 iters) and not on a tolerance, so it may not have converged."""
    x, p = pop['x'], pop['p']
    max_eval = iters * 5 // 4
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=iters, max_eval=max_eval, history_size=50,
                            tolerance_grad=1e-14, tolerance_change=1e-18, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = excess(model(x), p)
        loss.backward()
        return loss

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = excess(model(x), p).item()
    if not math.isfinite(start):
        return start, dict(n_iter=0, func_evals=0, capped=False)
    opt.step(closure)
    after = excess(model(x), p).item()
    st = opt.state[opt._params[0]]
    info = dict(n_iter=st['n_iter'], func_evals=st['func_evals'], capped=st['n_iter'] >= iters or st['func_evals'] >= max_eval)
    if not math.isfinite(after):  # a failed line search: keep Adam's endpoint
        model.load_state_dict(before)
        after = start
    return after, info


def regauge(model, seed, cond=3.0):
    """Change the basis of every bond in place (leaf and level-1, level-2 bonds) by a random invertible matrix with
    condition number <= cond. The function computed is unchanged; gauge-invariant measures must not move."""
    g = torch.Generator().manual_seed(seed)
    r = model.r

    def gl():
        q1, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))
        q2, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))
        return q1 @ torch.diag(torch.linspace(1.0, cond, r, dtype=torch.float64)) @ q2

    with torch.no_grad():
        Gs = [gl() for _ in range(model.n)]                       # one per bond of the current level
        model.A.copy_(torch.einsum('nvb,nab->nva', model.A, torch.stack(Gs)))
        for W in model.W:
            inv = torch.stack([torch.linalg.inv(G) for G in Gs])
            out = [gl() for _ in range(W.shape[0])]
            W.copy_(torch.einsum('nea,nabc,nbf,ncg->nefg', torch.stack(out), W, inv[0::2], inv[1::2]))
            Gs = out
        model.C.copy_(torch.einsum('oab,af,bg->ofg', model.C, torch.linalg.inv(Gs[0]), torch.linalg.inv(Gs[1])))
    return model


# ------------------------------------------------------------------------------------------ runs

CKPT_AT = (0, 100, 300, 1000, 3000, 10000)
LR = {'variance': 1e-2, 'identity': 3e-3}   # per init, the best of LRS on pilot seeds 0-3 (sym aligned, r = 2)
LRS = (1e-3, 3e-3, 1e-2, 3e-2)
STEPS = 20000
EXACT, NOT_EXACT = 1e-10, 1e-5   # excess nats after the polish: below EXACT counts as the exact predictor
ADAM_EXACT = 1e-6                # a run Adam itself brought below this was found by Adam; the polish only refined it
POLISH_ITERS = 1000


def run_name(cfg):
    """A run at its init's default rate LR[init] has no rate in its name; other rates are marked."""
    lr = '' if cfg['lr'] == LR[cfg['init']] else f"_lr{cfg['lr']:g}"
    return f"{cfg['grammar']}_{cfg['grouping']}_r{cfg['r']}_{cfg['init']}{lr}_s{cfg['seed']}"


def run_one(job):
    """Train one tree (Adam then the L-BFGS polish) and save <out>/<run_name>.pt:
    dict(cfg, log [(step, excess)], ckpts {step: state_dict}, polished state_dict, kl_adam, kl_polished, polish).
    A run that raises writes <run_name>.err with the traceback and does not stop the grid."""
    import time
    import traceback
    from pathlib import Path
    cfg, out = job
    path = Path(out) / f'{run_name(cfg)}.pt'
    if path.exists():
        return run_name(cfg), torch.load(path)['kl_polished'], 0.0
    torch.set_num_threads(1)
    t0 = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pop = population(cfg['grammar'])
        model = build(cfg['r'], cfg['grouping'], cfg['seed'], cfg['init'])
        log, ckpts = train(model, pop, cfg['steps'], cfg['lr'], ckpt_at=CKPT_AT)
        kl_polished, info = polish(model, pop, POLISH_ITERS)
        tmp = path.with_suffix('.tmp')
        torch.save(dict(cfg=cfg, log=log, ckpts=ckpts, polished={k: v.detach().clone() for k, v in model.state_dict().items()},
                        kl_adam=log[-1][1], kl_polished=kl_polished, polish=info, seconds=time.time() - t0), tmp)
        tmp.rename(path)
    except Exception:
        path.with_suffix('.err').write_text(traceback.format_exc())
        return run_name(cfg), float('nan'), time.time() - t0
    return run_name(cfg), kl_polished, time.time() - t0


def provenance():
    """What produced a result file: the commit, whether tracked files differed from it, and the library versions."""
    import subprocess
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
                torch=torch.__version__, numpy=np.__version__)


def run_grid(cfgs, out, procs):
    """Every cfg through run_one, `procs` at a time; each run is saved as it ends, and finished runs are skipped.
    (multiprocessing.Pool does not notice a worker killed for memory: if no line is printed for 5 minutes, stop the
    command and rerun it with --procs 1.)"""
    import os
    from multiprocessing import Pool
    jobs = [(c, out) for c in cfgs]
    with Pool(procs) as pool:
        for i, (name, kl, sec) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f'{i}/{len(jobs)}  {name}  excess {kl:.2e}  {sec:.0f}s', flush=True)
            if os.environ.get('ORCH_PROGRESS'):
                Path = __import__('pathlib').Path
                Path(os.environ['ORCH_PROGRESS']).write_text(f'{i}/{len(jobs)} runs\n')


def load_model(run, which='polished'):
    """The tree of a saved run at `which`: 'polished' or an Adam step in run['ckpts']."""
    cfg = run['cfg']
    model = build(cfg['r'], cfg['grouping'])
    model.load_state_dict(run['polished'] if which == 'polished' else run['ckpts'][which])
    return model
