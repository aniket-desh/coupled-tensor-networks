"""Part 2 of belief-probe-controls: do the directions of largest variance of a model's state lay out the belief?

    PYTHONPATH=. python studies/hmm/belief-probe-controls/lowdim.py FAMILY RUN [RUN ...]
      FAMILY = rhm-transformer | rhm-tree | hmm ; RUN = a run name of that experiment (e.g. v8m4s2L4_s0, mess3_chain_s1)

The experiments' probes use every coordinate of the state (64 to 1,088 features) against at most a few hundred
distinct beliefs per position, so they show that an affine image of the belief is somewhere in the state, and their
scrambled-target control cannot separate "laid out as the belief simplex" from "some code that identifies the belief
state". The Simplex-style claim is stronger: the belief geometry is the dominant structure of the state. This script
tests that with probes that may only use the state's k directions of largest variance.

For an activation set a_t (N, p) at position t, on the experiment's sequences and held-out prefix split:
    Z_k = the scores of a_t on its top-k principal axes (centered and fit on the fit rows),
    R2_k(target) = the experiment's affine probe and score, from Z_k to the target,
for k on a grid up to p, and two targets: the belief b_t, and the experiment's scrambled target (an independent
N(0, 1) 8-vector per distinct belief, seed 500 + t).

Reverse direction (`rev`, model and initial network): the state's top principal scores that hold 90% of its variance,
predicted affinely FROM the belief on the same held-out prefixes. Its R2 is the share of the state's variance that is
a function of the belief: the forward probe can be perfect while the belief is a small part of the state, and it can
fail at small k for a state that is a stretched or duplicated image of the belief, which the reverse direction passes.

Reference (`oracle`): the same computation with the exact belief itself as the activations, a_t := b_t. It gives, for
each k, what a state whose principal axes are the belief's own would score: R2_k(belief) is then the rank-k ceiling
(Eckart-Young), and R2_k(scrambled) is how well ANY geometric code would fit the scrambled labels, which is far from 0
where the distinct beliefs are close to affinely independent. Controls probed identically: the initial network
(checkpoint 0) and the one-hot prefix tokens.

Activation sets and position groups (the experiments' own eligible groups):
    rhm-transformer   L4, group eL4            (runs/rhm-belief-geometry)
    rhm-tree          top = [h_L, h_R], group h1 (t <= T/2) and h2 (t > T/2)   (runs/rhm-ttn-belief-geometry)
    hmm               transformer L4 / chain state (group e); tree top (groups h1, h2)   (runs/hmm-belief-geometry)
The tree's top_bil set is left out: its 1,088 product features have no natural scale, so their principal axes are
not a property of the state. For the tensor networks the principal axes of a bond depend on its gauge (the affine probe
does not): the result describes the gauge the training left, which is the object the Simplex figures show.

Saved per run (results/lowdim/<family>/<run>.json): for every group, position, set (model, init, tokens, oracle),
target and k: SSE and SST on the held-out prefixes, so the summary can pool like the experiments (1 - sum SSE / sum SST)."""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RHM = HERE.parents[1] / 'rhm'
sys.path.insert(0, str(RHM / 'rhm-ttn-belief-geometry'))
sys.path.insert(0, str(RHM / 'rhm-belief-geometry'))
sys.path.insert(0, str(HERE.parent / 'hmm-belief-geometry'))
sys.path.insert(0, str(HERE))
import analyze as AZ  # noqa: E402
import beliefs as BL  # noqa: E402
import hmm_analyze as HA  # noqa: E402
import hmm_run as HR  # noqa: E402
import model as M  # noqa: E402
import ttn_analyze as TA  # noqa: E402
import ttn_run as TR  # noqa: E402
from factor_product import features as hmm_features  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.hmm import processes as P  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SMOKE_N = int(os.environ.get('SMOKE_N', 0))  # code-path test on fewer sequences: positions kept, split not checked
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
ROOT = Path(os.environ.get('RUNS_ROOT', 'runs'))  # where the experiments' run folders (checkpoints) are
FAMILIES = {'rhm-transformer': (ROOT / 'rhm-belief-geometry', RHM / 'rhm-belief-geometry' / 'results'),
            'rhm-tree': (ROOT / 'rhm-ttn-belief-geometry', RHM / 'rhm-ttn-belief-geometry' / 'results'),
            'hmm': (ROOT / 'hmm-belief-geometry', HERE.parent / 'hmm-belief-geometry' / 'results')}


def curve(A, targets, fi, ti):
    """{target: {k: (sse, sst)}} for the affine probe from the top-k principal scores of A (fit on rows fi, scored on
    rows ti); k runs over KS below the number of features, then the number of features itself (the full probe)."""
    mu = A[fi].mean(0)
    X = A - mu
    evals, evecs = torch.linalg.eigh(X[fi].T @ X[fi])
    Z = X @ evecs.flip(1)  # columns by decreasing variance
    p = int((evals > 1e-12 * evals.max()).sum())  # directions with variance
    out = {name: {} for name in targets}
    for k in [k for k in KS if k < p] + [p]:
        for name, Y in targets.items():
            sc = BC.score(Y[ti], BC.fit_predict(Z[fi, :k], Y[fi], Z[ti, :k]), Y[fi].mean(0))
            out[name][k] = (sc['sse'], sc['sst'])
    return out


def reverse(A, b, fi, ti, share=0.9):
    """(m, sse, sst): the top m principal scores of A (the fewest holding `share` of its variance on the fit rows),
    predicted by the affine probe from b; SSE and SST summed over those m scores on rows ti."""
    mu = A[fi].mean(0)
    X = A - mu
    evals, evecs = torch.linalg.eigh(X[fi].T @ X[fi])
    evals, V = evals.flip(0).clamp_min(0), evecs.flip(1)
    m = int((evals.cumsum(0) < share * evals.sum()).sum()) + 1
    Z = X @ V[:, :m]
    sc = BC.score(Z[ti], BC.fit_predict(b[fi], Z[fi], b[ti]), Z[fi].mean(0))
    return m, sc['sse'], sc['sst']


def check():
    """With the target itself as activations, R2_k is the Eckart-Young value 1 - (tail of the spectrum) / total, in
    sample; and a state that is a rotation of [the belief, weaker noise directions] reproduces it."""
    gen = torch.Generator().manual_seed(0)
    b = torch.rand(600, 4, generator=gen, dtype=torch.float64) @ torch.rand(4, 9, generator=gen, dtype=torch.float64)
    idx = torch.arange(600)
    lam = torch.linalg.eigvalsh((b - b.mean(0)).T @ (b - b.mean(0))).flip(0)
    c = curve(b, {'b': b}, idx, idx)['b']
    for k in (1, 2, 3):
        assert abs((1 - c[k][0] / c[k][1]) - (1 - lam[k:].sum() / lam.sum()).item()) < 1e-6, k
    Q = torch.linalg.qr(torch.randn(12, 12, generator=gen, dtype=torch.float64))[0]
    a = torch.cat([b, 1e-3 * torch.randn(600, 3, generator=gen, dtype=torch.float64)], 1) @ Q
    c2 = curve(a, {'b': b}, idx, idx)['b']
    assert all(abs(c2[k][0] - c[k][0]) < 1e-4 * c[k][1] for k in (1, 2, 3))
    # reverse: a state that is a stretched linear image of the belief is fully explained by it; pure noise is not
    stretch = b @ torch.randn(9, 7, generator=gen, dtype=torch.float64) * torch.tensor([30., 1, 1, 1, 1, 1, 1], dtype=torch.float64)
    m, sse, sst = reverse(stretch, b, idx, idx)
    assert sse < 1e-6 * sst, (m, sse, sst)
    m, sse, sst = reverse(torch.randn(600, 7, generator=gen, dtype=torch.float64), b, idx[:300], idx[300:])
    assert sse > 0.9 * sst


def split_of(x, t, sp):
    _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
    side = torch.rand(int(pid.max()) + 1, generator=torch.Generator().manual_seed(100 + t)) < 0.5
    fi, ti = torch.nonzero(side[pid])[:, 0], torch.nonzero(~side[pid])[:, 0]
    assert SMOKE_N or (len(fi), len(ti)) == (sp['n_fit'], sp['n_test']), t  # the experiment's split
    return fi.to(DEV), ti.to(DEV)


def load(family, name):
    """(x, V, belief_at(t) -> (N, D) on DEV, {group: (set, positions)}, at(ckpt) -> f(t, set) -> (N, p), ckpts)."""
    runs, res_dir = FAMILIES[family]
    run_dir = Path(runs) / name
    res, ana = json.load(open(run_dir / 'result.json')), json.load(open(res_dir / name / 'analysis.json'))
    cfg, n = res['cfg'], SMOKE_N or ana['n_ana']
    state = lambda ck: torch.load(run_dir / f'ckpt_{ck}.pt', map_location=DEV)['state']
    E = ana['eligible']
    if family == 'hmm':
        kind, V = cfg['model'], cfg['V']
        proc = P.get(cfg['process'])
        x = P.sample(proc, n, cfg['T'], g=torch.Generator().manual_seed(HA.ANA_SEED))
        eta = P.beliefs(proc, x)[0].to(DEV)
        groups = {'h1': ('top', E['h1']), 'h2': ('top', E['h2'])} if kind == 'tree' else {'e': (ana['primary'], E['e'])}

        def at(ck):
            model = HR.build(cfg).to(DEV)
            model.load_state_dict(state(ck))
            return hmm_features(model.eval(), kind, x.to(DEV), V)
        return x, V, lambda t: eta[:, t], groups, at, res['ckpts'], ana
    g = cfg['grammar']
    V = g['v']
    rules = G.sample_rules(V, g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(n, rules, seed=AZ.ANA_SEED + g['seed'])[g['L']]
    bel = BL.beliefs(x, rules, g)
    if family == 'rhm-transformer':
        groups = {'eL4': ('L4', E['eL4'])}

        def at(ck):
            model = M.Transformer(V, cfg['T'], d=cfg['d'], layers=cfg['layers'], heads=cfg['heads']).to(DEV)
            model.load_state_dict(state(ck))
            model.eval()
            with torch.no_grad():
                L4 = torch.cat([model(M.inputs(x[i:i + 4096].to(DEV), V), resid=True)[1][-1].float()
                                for i in range(0, len(x), 4096)])
            return lambda t, s: L4[:, t].double()
    else:
        groups = {'h1': ('top', E['h1']), 'h2': ('top', E['h2'])}

        def at(ck):
            model = TR.build(cfg).to(DEV)
            model.load_state_dict(state(ck))
            model.eval()
            return lambda t, s: TA.bond_sets(model, x.to(DEV), t, V, bilinear=False)[s].double()
    return x, V, lambda t: bel[t][0].to(DEV), groups, at, res['ckpts'], ana


def run(family, name):
    x, V, belief_at, groups, at, ckpts, ana = load(family, name)
    fs = {'model': at(ckpts[-1]), 'init': at(ckpts[0])}
    aset = {t: s for s, pos in groups.values() for t in pos}
    out, info = {}, {}
    for t in sorted(aset):  # one position at a time: a belief matrix can be 60,000 x 4,096
        b = belief_at(t)
        uniq, uid = torch.unique(b.round(decimals=9), dim=0, return_inverse=True)
        scr = torch.randn(len(uniq), 8, generator=torch.Generator().manual_seed(500 + t), dtype=torch.float64).to(DEV)[uid]
        tg, sp = {'belief': b, 'scr': scr}, split_of(x, t, ana['prefix_split'][str(t)])
        sv = torch.linalg.svdvals(uniq - uniq.mean(0))
        info[t] = dict(n_unique=len(uniq), rank=int((sv > 1e-9 * sv[0]).sum()) if len(uniq) > 1 and sv[0] > 0 else 0,
                       D=b.shape[1])
        oh = torch.nn.functional.one_hot(x[:, :t].to(DEV), V).flatten(1).double()
        out[t] = dict(oracle=curve(b, tg, *sp), tokens=curve(oh, tg, *sp))
        for label, f in fs.items():
            A = f(t, aset[t])
            out[t][label] = curve(A, tg, *sp)
            out[t][label]['rev'] = reverse(A, b, *sp)
            del A
        del b, scr, tg, oh
    # the experiment's own number for the full probe at each position, to check this script against it
    per = ana['metrics'][str(ckpts[-1])]['per_pos']
    saved = {t: per[f'{aset[t]}@prefix'][str(t)]['R2'] for t in aset}
    return dict(run=name, family=family, groups={g: dict(set=s, positions=p) for g, (s, p) in groups.items()},
                info=info, curves=out, saved_full=saved, excess=ana.get('final_excess'), smoke=bool(SMOKE_N))


def main():
    check()
    family, names = sys.argv[1], sys.argv[2:]
    out_dir = HERE / 'results' / ('lowdim_smoke' if SMOKE_N else 'lowdim') / family
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (out_dir / f'{name}.json').exists():
            continue
        d = run(family, name)
        json.dump(d, open(out_dir / f'{name}.json', 'w'))
        t = d['groups'][next(iter(d['groups']))]['positions']
        if t:
            c = d['curves'][t[len(t) // 2]]
            r2 = lambda q: f'{1 - q[0] / q[1]:.3f}' if q[1] > 0 else '-'
            ks = sorted(c['oracle']['belief'])
            k = ks[min(3, len(ks) - 1)]
            print(f'{family} {name}: t={t[len(t) // 2]} k={k} belief model {r2(c["model"]["belief"].get(k, (0, 0)))} '
                  f'init {r2(c["init"]["belief"].get(k, (0, 0)))} oracle {r2(c["oracle"]["belief"][k])}', flush=True)
        else:
            print(f'{family} {name}: no eligible position', flush=True)


if __name__ == '__main__':
    main()
