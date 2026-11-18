"""Decoded beliefs at every saved checkpoint of one training run: the frames of the belief-geometry animation.

    PYTHONPATH=. python studies/hmm/belief-animation/anim_frames.py RUN_DIR OUT_DIR [N_ANA] [--every K]
    PYTHONPATH=. python studies/hmm/belief-animation/anim_frames.py --steps 30000     # print the checkpoint steps

RUN_DIR is a run folder of hmm_run.py, rhm-belief-geometry/run.py, ttn_run.py or arch_run.py (result.json and
ckpt_<step>.pt). At each checkpoint the probe that the static figures draw is refit from scratch on the fit half of
the analysis set and its prediction on the test half is saved, so a frame is exactly the corresponding panel of
hmm_plot.py or of the RHM geometry figures at that checkpoint. Nothing is interpolated between checkpoints.

Analysis set, split and probes are those of the original analyses (same seeds, N_ANA = 60000, sequence split):
  HMM (hmm_analyze.py)  target eta_t = P(state | x_<t), t = 1 .. T-1.
      transformer  affine probe on the last layer L4.
      chain        projective probe on the state h_t.
      tree, tree8  projective probe on the top bonds [h_L, h_R] for t <= T/2 and on [h_L, h_R, h_L (x) h_R] after it.
  RHM (analyze.py, ttn_analyze.py, arch_analyze.py)  target b_t = P(pending symbols | x_<t), every t with D_t <= 4096.
      transformer  affine probe on L4;   chain  affine probe on the state h_t;
      trees        affine probe on [h_A, h_B, h_A (x) h_B] (the 'bilinear' probe), aligned or permuted leaves.
      D_t is up to 4096 but b_t has few distinct values, so the probe is fit to the coordinates c = (b - mean) V_r on
      an orthonormal basis V_r of the span of the centered beliefs. Least squares commutes with a linear map of the
      target and the prediction stays in that span, so the fit, its R2 and its projection are those of the full b_t.

Saved to OUT_DIR/<run>.npz (rewritten every 10 checkpoints, so a stop loses at most 10):
  steps (F,), positions (P,), sse (F, P) and sst (P,) over the whole test half (R2 = 1 - sse / sst, pooled over
  positions as 1 - sum sse / sum sst), and for the first SAVE_N test rows
  HMM  true (P, SAVE_N, S) float32, pred (F, P, SAVE_N, S) float16 (clipped to [-8, 8]).
  RHM  true (P, SAVE_N, 3) and pred (F, P, SAVE_N, 3): coordinates on the top three principal axes of the true
       beliefs at that position; uid (P, SAVE_N) the id of each row's distinct belief; n_unique (P,).
and OUT_DIR/<run>.json: cfg, the test-loss history of the run, and its update time."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parents[1]
for p in ('hmm/hmm-belief-geometry', 'rhm/rhm-belief-architectures', 'rhm/rhm-ttn-belief-geometry', 'rhm/rhm-belief-geometry'):
    sys.path.insert(0, str(STUDIES / p))
import analyze as AZ  # noqa: E402
import arch_analyze as AA  # noqa: E402
import arch_run as AR  # noqa: E402
import beliefs as BL  # noqa: E402
import hmm_analyze as HA  # noqa: E402
import hmm_run as HR  # noqa: E402
import model as M  # noqa: E402
import ttn_analyze as TA  # noqa: E402
import ttn_run as TR  # noqa: E402
from studies import belief_common as BC  # noqa: E402
from studies.hmm import processes as P  # noqa: E402
from studies.rhm import grammar as G  # noqa: E402

N_ANA, SAVE_N, CLIP = 60000, 1000, 8.0
N_LOG, FIRST, BASE_STEPS = 120, 10, 30_000


def frame_steps(total):
    """0, then steps on a fixed log grid: FIRST * q^k with q set so that N_LOG points span FIRST .. BASE_STEPS. Runs
    of any length share the grid, so a 50,000-update run has the frames of a 30,000-update run plus a few more."""
    q = (BASE_STEPS / FIRST) ** (1 / (N_LOG - 1))
    out, k = {0, total}, 0
    while FIRST * q ** k <= total * (1 + 1e-9):
        out.add(int(round(FIRST * q ** k)))
        k += 1
    return sorted(s for s in out if s <= total)


def family(cfg):
    if 'process' in cfg:
        return 'hmm'
    return {'ttn': 'rhm-tree', 'tn-arch': 'rhm-arch'}.get(cfg.get('kind'), 'rhm-transformer')


def setup_hmm(cfg, n_ana, dev):
    """Targets and, per checkpoint, the prediction of the drawn probe (hmm_plot.stacked) on the test half."""
    kind, T, V, S = cfg['model'], cfg['T'], cfg['V'], cfg['S']
    proc = P.get(cfg['process'])
    x = P.sample(proc, n_ana, T, g=torch.Generator().manual_seed(HA.ANA_SEED))
    eta = P.beliefs(proc, x)[0].to(dev)
    xd, half = x.to(dev), n_ana // 2
    positions = list(range(1, T))
    targets = {t: eta[:, t] for t in positions}

    def predict(state):
        model = HR.build(cfg).to(dev)
        model.load_state_dict(state)
        model.eval()
        at = HA.hidden(model, kind, xd, V, bilinear=True)
        out = {}
        for t in positions:
            b, sets = targets[t], at(t)
            if kind == 'transformer':
                A = sets['L4'].double()
                out[t] = BC.fit_predict(A[:half], b[:half], A[half:])
            else:
                A = sets['state' if kind == 'chain' else ('top' if t <= T // 2 else 'top_bil')].double()
                out[t] = HA.projective(A[:half], b[:half], A[half:], [(0, S)])
        return out
    return positions, targets, predict, None


def setup_rhm(cfg, fam, n_ana, dev):
    g, T, v = cfg['grammar'], cfg['T'], cfg['V']
    rules = G.sample_rules(v, g['n'], g['m'], g['s'], g['L'], seed=g['seed'])
    x = G.sample_trees(n_ana, rules, seed=AZ.ANA_SEED + g['seed'])[g['L']]  # the sequences of the three RHM analyses
    bel = BL.beliefs(x, rules, g)
    positions = [t for t in sorted(bel) if t >= 1]
    xd, half = x.to(dev), n_ana // 2
    targets, extra = {}, dict(uid={}, n_unique={})
    for t in positions:
        b = bel.pop(t)[0].to(dev)
        uniq, uid, cnt = torch.unique(b.round(decimals=9), dim=0, return_inverse=True, return_counts=True)
        # Principal axes over the sampled rows (as analyze.py), from the distinct beliefs weighted by their counts:
        # sum_rows (b - mean)(b - mean)^T = sum_u count_u (u - mean)(u - mean)^T, so the axes are the same.
        _, sv, Vh = torch.linalg.svd(cnt.double().sqrt()[:, None] * (uniq - b.mean(0)), full_matrices=False)
        rank = min(len(Vh), max(3, int((sv > 1e-9 * sv[0]).sum())))
        targets[t] = (b - b.mean(0)) @ Vh[:rank].T
        extra['uid'][t], extra['n_unique'][t] = uid[half:][:SAVE_N].cpu().numpy(), int(uid.max()) + 1
        del b
    arm = cfg.get('arm', 'tree') if fam != 'rhm-transformer' else 'transformer'

    def predict(state):
        if fam == 'rhm-transformer':
            torch.manual_seed(g['seed'])
            model = M.Transformer(v, T, d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'])
        else:
            model = TR.build(cfg) if fam == 'rhm-tree' else AR.build(cfg)
        model = model.to(dev)
        model.load_state_dict(state)
        model.eval()
        if arm == 'transformer':
            acts = AZ.activations(model, xd, v)
            feat = lambda t: AZ.at(acts, 'L4', t, dev)
        elif arm == 'chain':
            st = AA.chain_all_states(model, xd)
            feat = lambda t: st[:, t].double()
        else:
            feat = lambda t: TA.bond_sets(model, xd, t, v, bilinear=True)['top_bil'].double()
        out = {}
        for t in positions:
            A, c = feat(t), targets[t]
            out[t] = BC.fit_predict(A[:half], c[:half], A[half:])
        return out
    return positions, targets, predict, extra


def main():
    args = sys.argv[1:]
    if args[0] == '--steps':
        print(','.join(map(str, frame_steps(int(args[1])))))
        return
    every = int(args[args.index('--every') + 1]) if '--every' in args else 1
    args = args[:args.index('--every')] if '--every' in args else args
    run_dir, out_dir = Path(args[0]), Path(args[1])
    n_ana = int(args[2]) if len(args) > 2 else N_ANA
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(run_dir / 'result.json'))
    cfg, fam = res['cfg'], family(res['cfg'])
    positions, targets, predict, extra = setup_hmm(cfg, n_ana, dev) if fam == 'hmm' else setup_rhm(cfg, fam, n_ana, dev)
    half = n_ana // 2
    # RHM: the top three principal coordinates (zero-padded where a position has fewer than three)
    keep = (lambda z: z) if fam == 'hmm' else (lambda z: torch.nn.functional.pad(z[:, :3], (0, 3 - min(3, z.shape[1]))))
    true = np.stack([keep(targets[t][half:][:SAVE_N]).cpu().numpy().astype(np.float32) for t in positions])
    sst = np.array([((targets[t][half:] - targets[t][:half].mean(0)) ** 2).sum().item() for t in positions])
    ckpts = res['ckpts']
    ckpts = sorted(set(ckpts[::every]) | {ckpts[0], ckpts[-1]})
    steps, preds, sses = [], [], []

    def write():
        out = dict(steps=np.array(steps), positions=np.array(positions), true=true, sst=sst, sse=np.array(sses),
                   pred=np.stack(preds))
        if extra:
            out.update(uid=np.stack([extra['uid'][t] for t in positions]),
                       n_unique=np.array([extra['n_unique'][t] for t in positions]))
        tmp = out_dir / f'{run_dir.name}.tmp.npz'
        np.savez_compressed(tmp, **out)
        tmp.replace(out_dir / f'{run_dir.name}.npz')

    for i, ck in enumerate(ckpts):
        state = torch.load(run_dir / f'ckpt_{ck}.pt', map_location=dev)['state']
        pk = predict(state)
        sses.append([((targets[t][half:] - pk[t]) ** 2).sum().item() for t in positions])
        preds.append(np.stack([keep(pk[t][:SAVE_N]).clamp(-CLIP, CLIP).cpu().numpy().astype(np.float16) for t in positions]))
        steps.append(ck)
        print(f'{run_dir.name} step {ck:6d} ({i + 1}/{len(ckpts)}) pooled R2 {1 - sum(sses[-1]) / sst.sum():.4f}', flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(ckpts):
            write()
    json.dump(dict(run=run_dir.name, family=fam, cfg=cfg, n_ana=n_ana, status=res.get('status', 'finished'),
                   bayes=res['bayes'], ms_per_update=res['ms_per_update'],
                   hist=[{k: h[k] for k in ('step', 'test', 'excess')} for h in res['hist']]),
              open(out_dir / f'{run_dir.name}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
