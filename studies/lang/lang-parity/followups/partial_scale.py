"""Follow-up to lang-parity (not part of its criteria): how large are the partial-parity directions in the bonds?

    PYTHONPATH=. python studies/lang/lang-parity/followups/partial_scale.py runs/lang-parity

The write-up found that in the runs that never learn (k = 8, 16 at small init), the 4-bit partial parities become
linearly decodable from the whitened level-2 bonds while the 8-bit ones stay at chance. Whitening removes scale, so
that probe cannot say whether the 4-bit direction is a large part of the bond or a direction of negligible size that
whitening inflates. This script measures the size, exactly and without a probe.

For node (l, j) with span S and K = the k parity bits, let chi(x) = (-1)^(sum of x_t, t in S and K), the partial parity
as a sign. With unbiased bits chi has mean 0 and is orthogonal to every other parity function, so
    share = || E[z chi] ||^2 / E|| z - E z ||^2
is the fraction of the bond's variance that lies along the partial parity (its squared Fourier coefficient over the
bond's variance; 1 for a bond that is the parity and nothing else). The expectation is exact: all 2^16 inputs are
enumerated, so there is no sampling floor. The share is taken in the model's own coordinates (it is not gauge
invariant, unlike the whitened probe: that is the point, it measures the size the next core actually receives).

Reported per run and checkpoint, as the median over the nodes that hold m = 2, 4 or 8 of the k bits (bias-0 main runs,
k in {4, 8, 16}). Reading it:
  stalled runs (never reach 0.9 x Bayes accuracy): share_4 at the start and the end of training;
  learned runs: share_4 at the last checkpoint before the run's discovery time t, and at the end.
If the stalled runs' final share_4 is as large as the learned runs' share just before they learn, the 4-bit parities
are in place and the stall is at the next merge; if it is orders of magnitude smaller, the decodability in the write-up
is whitening acting on a negligible direction.

Also read from the saved interpretation files (runs/lang-parity/<run>/interpret.json), final checkpoint, 4-bit nodes:
the change in KL to the counterfactual when the partial-parity subspace ('sub') or the whole bond ('whole') is patched
from a donor, kl_base - kl_target. In a run whose output is still uniform both are zero, whatever the bonds hold."""
import json
import math
import sys
from pathlib import Path

import torch

from studies.lang import analysis as A
from studies.lang import common as L
from studies.lang import processes as P
from studies.lang.interpret import acts, checkpoints

HERE = Path(__file__).resolve().parent
KS, EPS, SEEDS = (4, 8, 16), (0.01, 0.1, 0.3, 1.0), (0, 1, 2)


def shares(model, cfg, x):
    """{m: median over nodes holding m of the k bits of ||E[z chi]||^2 / E||z - E z||^2}, exact over the rows of x."""
    k = cfg['process']['k']
    _, a = acts(model, x)
    out = {}
    for (l, j), S in P.spans(cfg.get('perm')).items():
        K = [t for t in S if t < k]
        if len(K) < 2:
            continue
        chi = 1.0 - 2.0 * (x[:, K].sum(1) % 2).double()
        z = a[l][:, j]
        var = ((z - z.mean(0)) ** 2).sum(1).mean().item()
        out.setdefault(len(K), []).append(((z * chi[:, None]).mean(0) ** 2).sum().item() / var if var > 0 else float('nan'))
    return {m: sorted(v)[len(v) // 2] for m, v in out.items()}


def check(x):
    """A bond that is [parity of bits 0-3, an independent bit] has share 1/2; the sign function is exactly balanced."""
    chi = 1.0 - 2.0 * (x[:, :4].sum(1) % 2).double()
    z = torch.stack([chi, x[:, 9].double() * 2 - 1], 1)
    assert abs(chi.mean().item()) < 1e-15
    assert abs(((z * chi[:, None]).mean(0) ** 2).sum().item() / ((z - z.mean(0)) ** 2).sum(1).mean().item() - 0.5) < 1e-12


def main():
    out_dir = sys.argv[1]
    x = torch.tensor([[(i >> b) & 1 for b in range(P.N)] for i in range(2 ** P.N)])  # every input once
    check(x)
    runs, _ = A.load_runs(out_dir)
    res = {}
    for k in KS:
        for e in EPS:
            for s in SEEDS:
                name = f'main_k{k}_e{e:g}_b0_s{s}'
                r = runs.get(name)
                if r is None or not A.finished(r):
                    continue
                t = A.row(r)['t']
                traj = {}
                for step, path in checkpoints(Path(out_dir) / name):
                    model, cfg, _ = L.load(path)
                    traj[step] = shares(model.eval(), cfg, x)
                ip = A.load_interp(out_dir, name)
                patch = None
                if ip is not None:
                    sf = A.steps_of(ip)[-1]
                    nodes = [f'{l}_{j}' for (l, j), S in P.spans(cfg.get('perm')).items() if sum(q < k for q in S) == 4]
                    d = lambda node, kind: (A.get(ip, sf, node, 'par', 'patch', kind, 'kl_base'),
                                            A.get(ip, sf, node, 'par', 'patch', kind, 'kl_target'))
                    patch = {kind: A.med([b - c for b, c in (d(n, kind) for n in nodes) if None not in (b, c)])
                             for kind in ('sub', 'whole')}
                res[name] = dict(k=k, eps=e, seed=s, t=t if math.isfinite(t) else None, traj=traj, patch_kl_gain=patch)
                steps = sorted(traj)
                before = max([q for q in steps if t is not None and math.isfinite(t) and q < t], default=None)
                f = lambda step, m: '    -    ' if step is None or m not in traj[step] else f'{traj[step][m]:9.2e}'
                print(f'{name:22s} t={"stalled" if not math.isfinite(t) else int(t):>7} | 4-bit share: init {f(steps[0], 4)} '
                      f'before t {f(before, 4)} end {f(steps[-1], 4)} | 8-bit: init {f(steps[0], 8)} end {f(steps[-1], 8)} '
                      f'| patch KL gain {patch}', flush=True)
    json.dump(res, open(HERE / 'partial_scale.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
