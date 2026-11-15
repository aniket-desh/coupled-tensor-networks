"""Part 1 of belief-probe-controls, on the saved checkpoints of hmm-belief-geometry: is the joint belief of the product
processes carried as its two factor marginals?

    PYTHONPATH=. python studies/hmm/belief-probe-controls/factor_product.py [RUN ...]   # default: all 24 runs
    PYTHONPATH=. python studies/hmm/belief-probe-controls/factor_product.py --table     # table from saved JSON

The experiment found that the tensor networks give the factor marginals b^A, b^B with R2 near 1 but the 9-state joint
belief only at 0.75-0.85 on `factored`. On `factored` the joint is exactly b^AB(a, b) = b^A(a) b^B(b), so a state that
holds [b^A, b^B] linearly has the joint only as a product of its coordinates, which an affine or projective probe
cannot form. This script builds the joint from the experiment's own marginal probes,
    bhat^AB = bhat^A (x) bhat^B     (state index s = a S_B + b, as in processes.py),
with no further fitted parameter, and scores it like the direct joint probe: same sequences (N_ANA, same seed), same
held-out prefix split, same positions (the experiment's eligible group: `e` for the transformer and the chain, `h1`,
t <= T/2, for the tree), R2 = 1 - SSE / SST about the fit mean, pooled over positions as 1 - sum SSE / sum SST.
For the tree, the second half (group `h2b`, t > T/2, the set top_bil = [h_L, h_R, h_L (x) h_R]) is computed too and
saved under the key prefix `h2b_`; the rule below uses the first half, as the experiment's table did.

Per run, final checkpoint and checkpoint 0 (initial network), affine and projective probes on the primary set
(transformer L4, chain state, tree top = [h_L, h_R]):
  direct      the joint probed directly (the experiment's number, recomputed here)
  product     bhat^A (x) bhat^B from the marginal probes
  tok_direct  affine probe on the one-hot prefix, joint target (the experiment's token readout)
  tok_product the product of the marginals read affinely from the one-hot prefix. The control that matters: the tokens
              alone give the marginals at 0.96-0.97, so some of what `product` gains comes from the product form itself.
  true_product  b^A (x) b^B from the exact marginals: 1 on `factored`; on `coupled` what a perfect factor code gives.
  floor       the direct joint probe on a state that is exactly [b^A, b^B] (the exact marginals as the activations),
              same split and positions: what `direct` scores for a state that holds the marginals and nothing of the
              joint (0.755 affine on `factored` in the reviewer's computation).
`coupled` only, where C = b^AB - b^A (x) b^B is not zero:
  c_share     SST(C) / SST(b^AB): the share of the joint's variance that the marginals cannot give.
  c_aff       affine probe of C on the primary set (R2 of C about its fit mean), and c_tok on the one-hot prefix.
  c_quad      affine probe of C on [a, a (x) a] (upper triangle) for a = the chain's state or the tree's left top bond,
              where the position is eligible (ELIG_FACTOR * (features + 1) distinct fit prefixes).
  product+c   bhat^A (x) bhat^B + Chat (c_aff), scored as the joint.

Rule on `factored`, per arm, all three seeds (affine probes). The saved marginal and direct scores of the experiment
already decide most of it (see EXPERIMENT.md), so this is a measurement with its reading fixed, not an open prediction:
  marginals exact    product >= 0.99, and product halves the unexplained variance of tok_product and of the initial
                     network's product (1 - R2 <= 0.5 (1 - control)). If not: "marginals not exact".
  then, with gap = direct - floor:
    pure factor code                  gap <= TOL (0.03): the state gives the joint no better than its marginals do
    marginals plus part of the joint  gap > TOL and direct < JOINT (0.95)
    joint readable                    direct >= JOINT
`coupled` is descriptive, and read only if c_share is at least 0.01.
All of these are functions of the exact next-token distribution on these processes (next-token ceiling 1), so a pass
describes how the prediction is laid out in the state, not information beyond the prediction.
"""
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
import analyze as AZ  # noqa: E402
import hmm_analyze as HA  # noqa: E402
import hmm_run as HR  # noqa: E402
import model as M  # noqa: E402
import ttn_analyze as TA  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.hmm import processes as P  # noqa: E402

RUNS, RES = Path(os.environ.get('RUNS_ROOT', 'runs')) / 'hmm-belief-geometry', HERE.parent / 'hmm-belief-geometry' / 'results'
OUT = HERE / 'results' / ('factor_product_smoke.json' if os.environ.get('SMOKE_N') else 'factor_product.json')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SMOKE_N = int(os.environ.get('SMOKE_N', 0))  # code-path test on fewer sequences: positions kept, split not checked
PROCS, ARMS, SEEDS = ('factored', 'coupled'), ('transformer', 'chain', 'tree', 'tree8'), (0, 1, 2)
R2_MIN, RATIO, C_MIN, TOL, JOINT = 0.99, 0.5, 0.01, 0.03, 0.95
score = BC.score


def outer(m, SA):
    """(N, S_A + S_B) marginals -> (N, S_A S_B) product, s = a S_B + b."""
    return (m[:, :SA, None] * m[:, None, SA:]).flatten(1)


def quad(a):
    i, j = torch.triu_indices(a.shape[1], a.shape[1])
    return torch.cat([a, a[:, i] * a[:, j]], 1)


@torch.no_grad()
def features(model, kind, x, V):
    """at(t, name) -> (N, p) float64 on x's device: activation set `name` at position t (transformer L4, chain state,
    tree top or top_bil)."""
    if kind == 'transformer':
        L4 = torch.cat([model(M.inputs(x[i:i + 4096], V), resid=True)[1][-1].float() for i in range(0, len(x), 4096)])
        return lambda t, name: L4[:, t].double()
    if kind == 'chain':
        st = torch.cat([BC.chain_states(model, x[i:i + 8192])[0] for i in range(0, len(x), 8192)])
        return lambda t, name: st[:, t].double()
    return lambda t, name: TA.bond_sets(model, x, t, V, bilinear=name == 'top_bil')[name].double()


def check(proc):
    """A state that holds the exact marginals gives them affinely, the product then gives the joint exactly, and the
    direct affine joint probe on that state does not (this also fixes the index order of the product)."""
    x = P.sample(proc, 4000, 8, g=torch.Generator().manual_seed(1))
    eta = P.beliefs(proc, x)[0][:, 6]
    m = torch.cat(P.marginals(proc, eta), -1)
    SA = proc['factors'][0]
    f, t = slice(0, 2000), slice(2000, None)
    prod = score(eta[t], outer(BC.fit_predict(m[f], m[f], m[t]), SA), eta[f].mean(0))['R2']
    direct = score(eta[t], BC.fit_predict(m[f], eta[f], m[t]), eta[f].mean(0))['R2']
    if proc['name'] == 'factored':
        assert prod > 1 - 1e-6 and direct < 0.99, (prod, direct)
    return prod, direct


def run(name):
    res = json.load(open(RUNS / name / 'result.json'))
    ana = json.load(open(RES / name / 'analysis.json'))
    cfg = res['cfg']
    kind, T, V, r = cfg['model'], cfg['T'], cfg['V'], cfg.get('r', 0)
    proc = P.get(cfg['process'])
    SA = proc['factors'][0]
    x = P.sample(proc, SMOKE_N or ana['n_ana'], T, g=torch.Generator().manual_seed(HA.ANA_SEED))
    eta = P.beliefs(proc, x)[0].to(DEV)
    # (key prefix, the experiment's position group, activation set)
    groups = [('', 'h1', 'top'), ('h2b_', 'h2b', 'top_bil')] if kind == 'tree' else [('', 'e', ana['primary'])]
    split, quad_ts = {}, []
    for t in sorted({t for _, grp, _ in groups for t in ana['eligible'][grp]}):
        _, pid = torch.unique(x[:, :t], dim=0, return_inverse=True)
        side = torch.rand(int(pid.max()) + 1, generator=torch.Generator().manual_seed(100 + t)) < 0.5
        split[t] = (torch.nonzero(side[pid])[:, 0].to(DEV), torch.nonzero(~side[pid])[:, 0].to(DEV), int(side.sum()))
        sp = ana['prefix_split'][str(t)]
        assert SMOKE_N or (len(split[t][0]), len(split[t][1])) == (sp['n_fit'], sp['n_test']), (name, t)  # the experiment's split
    x = x.to(DEV)
    acc = {}  # key -> [sse, sst] summed over positions

    def add(key, sc):
        a = acc.setdefault(key, [0.0, 0.0])
        a[0] += sc['sse']
        a[1] += sc['sst']

    for ck in (res['ckpts'][-1], res['ckpts'][0]):
        model = HR.build(cfg).to(DEV)
        model.load_state_dict(torch.load(RUNS / name / f'ckpt_{ck}.pt', map_location=DEV)['state'])
        model.eval()
        at = features(model, kind, x, V)
        for grp_pre, grp, aset in groups:
            pre = grp_pre + ('' if ck == res['ckpts'][-1] else 'init_')
            for t in ana['eligible'][grp]:
                fi, ti, n_fit_pre = split[t]
                A, b = at(t, aset), eta[:, t]
                m = torch.cat(P.marginals(proc, b), -1)
                C = b - outer(m, SA)
                mu, mu_c = b[fi].mean(0), C[fi].mean(0)
                for fam, fn in (('aff', lambda Af, Bf, At, blocks: BC.fit_predict(Af, Bf, At)), ('proj', HA.projective)):
                    add(f'{pre}direct_{fam}', score(b[ti], fn(A[fi], b[fi], A[ti], [(0, b.shape[1])]), mu))
                    mh = fn(A[fi], m[fi], A[ti], [(0, SA), (SA, m.shape[1])])
                    add(f'{pre}marg_{fam}', score(m[ti], mh, m[fi].mean(0)))
                    add(f'{pre}product_{fam}', score(b[ti], outer(mh, SA), mu))
                    if fam == 'aff':
                        ch = BC.fit_predict(A[fi], C[fi], A[ti])
                        add(f'{pre}c_aff', score(C[ti], ch, mu_c))
                        add(f'{pre}product+c', score(b[ti], outer(mh, SA) + ch, mu))
                # coordinate products of the state: the chain's state, or the tree's left top bond at t <= T/2 (h_R is
                # constant there); the tree's second-half set top_bil already holds the cross-half products
                core = A if kind == 'chain' else A[:, :r] if aset == 'top' else None
                if core is not None and n_fit_pre >= AZ.ELIG_FACTOR * (core.shape[1] * (core.shape[1] + 3) // 2 + 1):
                    Q = quad(core)
                    add(f'{pre}c_quad', score(C[ti], BC.fit_predict(Q[fi], C[fi], Q[ti]), mu_c))
                    add(f'{pre}c_aff@quad', score(C[ti], BC.fit_predict(A[fi], C[fi], A[ti]), mu_c))  # same positions
                    if pre == '':
                        quad_ts.append(t)
                if ck == res['ckpts'][-1]:
                    oh = torch.nn.functional.one_hot(x[:, :t], V).flatten(1).double()
                    add(f'{grp_pre}tok_direct', score(b[ti], BC.fit_predict(oh[fi], b[fi], oh[ti]), mu))
                    add(f'{grp_pre}tok_product', score(b[ti], outer(BC.fit_predict(oh[fi], m[fi], oh[ti]), SA), mu))
                    add(f'{grp_pre}true_product', score(b[ti], outer(m[ti], SA), mu))
                    add(f'{grp_pre}floor_aff', score(b[ti], BC.fit_predict(m[fi], b[fi], m[ti]), mu))
                    add(f'{grp_pre}floor_proj', score(b[ti], HA.projective(m[fi], b[fi], m[ti], [(0, b.shape[1])]), mu))
                    add(f'{grp_pre}c_tok', score(C[ti], BC.fit_predict(oh[fi], C[fi], oh[ti]), mu_c))
                    add(f'{grp_pre}c_sst', dict(sse=score(C[ti], C[ti], mu_c)['sst'], sst=score(b[ti], b[ti], mu)['sst']))
        del at
    out = {k: 1 - sse / sst if sst > 0 else None for k, (sse, sst) in acc.items() if not k.endswith('c_sst')}
    for grp_pre, grp, _ in groups:
        if ana['eligible'][grp]:
            out[grp_pre + 'c_share'] = acc[grp_pre + 'c_sst'][0] / acc[grp_pre + 'c_sst'][1]
    out.update(run=name, process=cfg['process'], arm=name.split('_')[1], quad_positions=quad_ts,
               positions={grp: ana['eligible'][grp] for _, grp, _ in groups})
    return out


def table(runs):
    f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
    print(f'{"run":26s} direct: aff   proj | marg: aff   proj | product: aff  proj | tok: direct product | init product(proj)')
    for q in sorted(runs.values(), key=lambda q: q['run']):
        print(f'{q["run"]:26s}        {f(q["direct_aff"])}{f(q["direct_proj"])} |      {f(q["marg_aff"])}{f(q["marg_proj"])} |'
              f'         {f(q["product_aff"])}{f(q["product_proj"])} |     {f(q["tok_direct"])} {f(q["tok_product"])} |'
              f'  {f(q["init_product_proj"])}')
    print(f'\n{"run":26s} c_share  true_product  c_aff  c_tok | quad positions: c_aff c_quad | product+c (joint)')
    for q in sorted(runs.values(), key=lambda q: q['run']):
        if q['process'] == 'coupled':
            print(f'{q["run"]:26s} {q["c_share"]:.4f}   {f(q["true_product"])}       {f(q["c_aff"])} {f(q["c_tok"])} |'
                  f'                {f(q.get("c_aff@quad"))} {f(q.get("c_quad"))} |  {f(q["product+c"])}')
    print(f'\n{"tree, second half (h2b)":26s} direct: aff   proj | marg: aff   proj | product: aff  proj | tok: direct product')
    for q in sorted(runs.values(), key=lambda q: q['run']):
        if 'h2b_direct_aff' in q:
            print(f'{q["run"]:26s}        {f(q["h2b_direct_aff"])}{f(q["h2b_direct_proj"])} |      {f(q["h2b_marg_aff"])}'
                  f'{f(q["h2b_marg_proj"])} |         {f(q["h2b_product_aff"])}{f(q["h2b_product_proj"])} |     '
                  f'{f(q["h2b_tok_direct"])} {f(q["h2b_tok_product"])}')
    # reproduction: the direct joint probe recomputed here against the experiment's saved pooled number
    saved = {q['run']: q for q in json.load(open(RES / 'summary.json'))['runs']}
    diffs = [abs(q[f'direct_{k}'] - saved[q['run']][v]) for q in runs.values() for k, v in (('aff', 'aff'), ('proj', 'proj'))
             if saved.get(q['run'], {}).get(v) is not None and not SMOKE_N]
    if diffs:
        print(f'\nreproduction of the experiment\'s direct joint probe: largest |difference| in R2 over {len(diffs)} numbers = '
              f'{max(diffs):.2e}' + ('  <-- above 0.01: CHECK' if max(diffs) > 0.01 else ''))
    beats = lambda p, *cs: all((1 - p) <= RATIO * (1 - c) for c in cs)
    verdicts = {}
    print(f'\nRule on `factored` (affine): {"run":22s} product  tok_product  init_product | direct  floor   gap | class')
    for arm in ARMS:
        qs = [runs.get(f'factored_{arm}_s{s}') for s in SEEDS]
        if None in qs:
            continue
        cls = []
        for q in qs:
            p, gap = q['product_aff'], q['direct_aff'] - q['floor_aff']
            c = ('marginals not exact' if not (p >= R2_MIN and beats(p, q['tok_product'], q['init_product_aff'])) else
                 'joint readable' if q['direct_aff'] >= JOINT else
                 'pure factor code' if gap <= TOL else 'marginals plus part of the joint')
            cls.append(c)
            print(f'{"":28s}{q["run"]:22s} {f(p)}   {f(q["tok_product"])}       {f(q["init_product_aff"])}    | {f(q["direct_aff"])} '
                  f'{f(q["floor_aff"])} {gap:+.3f} | {c}')
        verdicts[arm] = dict(per_seed=cls, verdict=cls[0] if len(set(cls)) == 1 else 'seeds disagree: ' + ', '.join(cls))
        print(f'  -> {arm}: {verdicts[arm]["verdict"]}')
    if diffs and max(diffs) > 0.01:
        raise SystemExit('reproduction difference above 0.01: do not interpret these numbers')
    return verdicts


def main():
    for p in PROCS:
        print(f'check {p}: exact-marginal state, product R2 {check(P.get(p))[0]:.6f}, direct affine {check(P.get(p))[1]:.4f}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.load(open(OUT))['runs'] if OUT.exists() else {}
    args = sys.argv[1:]
    if args != ['--table']:
        for name in args or [f'{p}_{a}_s{s}' for p in PROCS for a in ARMS for s in SEEDS]:
            if name in done and not args:
                continue
            done[name] = run(name)
            q = done[name]
            print(f'{name}: direct {q["direct_aff"]:.3f}/{q["direct_proj"]:.3f} product {q["product_aff"]:.3f}/'
                  f'{q["product_proj"]:.3f} tok_product {q["tok_product"]:.3f}', flush=True)
            json.dump(dict(runs=done), open(OUT, 'w'), indent=1)
    json.dump(dict(runs=done, verdicts=table(done)), open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
