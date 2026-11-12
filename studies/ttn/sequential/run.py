"""Sequential transfer. Exact population GD throughout.
Phase 1, run once: output 1 alone (y1 = prod_j x^(j)_1, identical in every family) from 16 inits; output 2's loss is
masked, not given a zero target. A seed passes the phase-1 gate when |c1/s1 - 1| < 1e-2 at the end of phase 1.
Phase 2 branches every (family, s2) group from those saved states, four arms per seed:
  all          update every core
  freeze_q1    quarter q1 (A1, A2, W1) frozen
  freeze_left  the whole left subtree (q1, q2, W_L) frozen
  fresh        an independently initialized tree trained on output 2 alone (the no-transfer baseline)
Families are agreement sets for (y1, y2) (teachers.agreement, no y3): none, quarter (S = q1), half (S = left half),
with s2 = 0.5 and 1.0. Phase-2 groups run in parallel on the GPU.
Usage: PYTHONPATH=. python studies/ttn/sequential/run.py --out <dir> [--hours H] [--smoke] [--groups ...] [--eta E]
--groups may name 'phase1' and/or phase-2 groups; phase-2 groups alone reuse the saved phase1_params.pt."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ctn.ttn import TTN
from studies.ttn import common, teachers

ETA = 0.05
SEEDS = 16
T_MAX = 1e5
PHASE1_SHARE = 0.45  # of the training deadline; phase-1 escapes are heavy-tailed, and every phase-2 group needs them
ARMS = ['all', 'freeze_q1', 'freeze_left', 'fresh']
FAMILIES = ['none', 'quarter', 'half']


def comps_for(S, s2):
    return teachers.agreement(S, s=(1.0, s2), third=False)


def phase1(a):
    seeds = 2 if a.smoke else (a.seeds or SEEDS)
    deadline = a.deadline
    a.deadline = time.time() + PHASE1_SHARE * (deadline - time.time())
    _, log, _ = common.run_group(a, 'phase1', comps_for('none', 0.5), R=seeds, r=8, alpha=1.0, eta=a.eta or ETA,
                                 t_max=5.0 if a.smoke else (a.t_max or T_MAX), N=0, N_probe=512 if a.smoke else 2 ** 14, seed=4000,
                                 outputs=[0])
    a.deadline = deadline
    c1 = np.asarray(log['c'][-1])[:, 0]
    meta = json.load(open(Path(a.out) / 'phase1.json'))
    meta['gate_ok'] = (np.abs(c1 - 1.0) < 1e-2).tolist()
    json.dump(meta, open(Path(a.out) / 'phase1.json', 'w'), indent=1, default=float)
    print(f"phase 1 gate: {sum(meta['gate_ok'])}/{seeds} seeds learned output 1", flush=True)


def phase2(a, sp):
    seeds = 2 if a.smoke else (a.seeds or SEEDS)
    p1 = torch.load(Path(a.out) / 'phase1_params.pt')
    meta1 = json.load(open(Path(a.out) / 'phase1.json'))
    R = len(ARMS) * seeds  # run index = arm * seeds + seed
    net = TTN(R=R, r=8, d=8, O=common.O, alpha=1.0, seed=4500, device=a.device, dtype=common.DTYPE)  # 'fresh' keeps it
    with torch.no_grad():
        for p, q in zip(net.params, p1):
            for k in range(3):
                p[k * seeds:(k + 1) * seeds] = q.to(p)
    net.freeze(list(range(seeds, 2 * seeds)), ['q1'])
    net.freeze(list(range(2 * seeds, 3 * seeds)), ['q1', 'q2', 'L'])
    comps = comps_for(sp['S'], sp['s2'])
    active = common.active_mask(comps, R, a.device)
    active[3 * seeds:, 0] = 0.0  # the fresh tree is trained on output 2 alone
    t1 = meta1['t_end']
    common.run_group(a, sp['name'], comps, R=R, r=8, alpha=1.0, eta=a.eta or ETA, t_max=t1 + (5.0 if a.smoke else (a.t_max or T_MAX)),
                     N=0, N_probe=512 if a.smoke else 2 ** 14, seed=4000, net=net, t0=t1, active=active,
                     extra_meta=dict(arms=ARMS, seeds=seeds, run_index='arm * seeds + seed', S=sp['S'], s2=sp['s2'],
                                     phase1_t_end=t1, phase1_gate_ok=meta1['gate_ok']))


if __name__ == '__main__':
    a = common.args()
    specs = [dict(name=f'{S}_s2{s2}', S=S, s2=s2) for S in FAMILIES for s2 in (0.5, 1.0)]
    named = [g for g in a.groups.split(',') if g]
    if not named or 'phase1' in named:
        phase1(a)
    rest = [g for g in named if g != 'phase1']
    if not named or rest:
        a.groups = ','.join(rest)
        print(common.run_parallel(a, specs[:2] if a.smoke else specs, phase2), flush=True)
