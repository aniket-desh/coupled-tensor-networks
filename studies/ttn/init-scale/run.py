"""Init scale. Part A (no training, descriptive): per-layer activation second moments of the variance-controlled init
for r = 4, 8, 16, 32, and the output's second moment as every core is scaled by alpha.
Part B: exact population GD on teachers.agreement('quarter') at alpha = 1, 0.95, 0.9, 0.85, 0.8, the same 32 inits
in every group, plus alpha = 1 at half the step size (discretization control). Groups run in parallel on the GPU.
Usage: PYTHONPATH=. python studies/ttn/init-scale/run.py --out <dir> [--hours H] [--smoke] [--groups g1,g2] [--eta E]"""
import json
from pathlib import Path

import numpy as np
import torch

from ctn.ttn import TTN, sample
from studies.ttn import common, teachers

ETA = 0.05
SEEDS = 32
T_MAX = 1e5  # at alpha = 1; smaller alpha scales it by alpha^-13 (the homogeneity of the 15-core output)


def part_a(a):
    x, phi = sample(8 if a.smoke else 64, 512 if a.smoke else 8192, 8, seed=1, device=a.device, dtype=common.DTYPE)
    rows = []
    for r in ([4, 8] if a.smoke else [4, 8, 16, 32]):
        for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            net = TTN(R=phi.shape[0], r=r, d=8, O=common.O, alpha=alpha, seed=100 + r, device=a.device, dtype=common.DTYPE)
            with torch.no_grad():
                acts = net.forward(phi)
            m2 = {k: (v ** 2).mean(tuple(range(1, v.dim()))).cpu().numpy() for k, v in acts.items()}
            rows.append(dict(r=r, alpha=alpha, **{f'{k}_m2_mean': float(v.mean()) for k, v in m2.items()},
                             **{f'{k}_m2_q': np.percentile(v, [5, 50, 95]).tolist() for k, v in m2.items()}))
    json.dump(dict(rows=rows, device=common.gpu_note()), open(Path(a.out) / 'activations.json', 'w'), indent=1)


def group(a, sp):
    common.run_group(a, sp['name'], teachers.agreement('quarter'), R=2 if a.smoke else (a.seeds or SEEDS), r=8, alpha=sp['alpha'],
                     eta=sp['eta'], t_max=5.0 if a.smoke else (a.t_max or T_MAX) * sp['alpha'] ** -13, N=0,
                     N_probe=512 if a.smoke else 2 ** 14, seed=1000)


if __name__ == '__main__':
    a = common.args()
    if not a.groups or 'activations' in a.groups:
        part_a(a)
        print('part A done', flush=True)
    eta = a.eta or ETA
    specs = [dict(name=f'alpha{al}_eta{eta}', alpha=al, eta=eta) for al in [1.0, 0.95, 0.9, 0.85, 0.8]]
    specs.append(dict(name=f'alpha1.0_eta{eta / 2}', alpha=1.0, eta=eta / 2))
    print(common.run_parallel(a, specs[:2] if a.smoke else specs, group), flush=True)
