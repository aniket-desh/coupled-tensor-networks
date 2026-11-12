"""Reproduce the failed check `alignment_at_teacher_min` (studies/ttn/checks.py, section 8) on the CPU, and test
the gelsd fix. The setup is the check's: an r = 4 TTN set to the agreement('quarter') teacher, 4096 samples, seed 12.
At the teacher every node has exactly-zero activation columns, so each least-squares fit in ctn.ttn.alignment is
rank-deficient. Prints the column norms per node, then the 3 x 6 alignment matrix (components x nodes) n times with
torch's CPU default driver (gelsy) and n times with lstsq forced to gelsd. The check expects every entry > 0.999 and
the unrelated-feature value < 0.02. Run it several times: the gelsy rows change from run to run.
Usage: PYTHONPATH=. python studies/ttn/degree/diagnose_alignment.py [n]"""
import sys

import torch
from ctn.ttn import NODES, TTN, alignment, node_activations, sample, teacher_cores
from studies.ttn import teachers

f64 = torch.float64
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
comps = teachers.agreement('quarter')
net = TTN(R=1, r=4, d=8, O=3, dtype=f64)
with torch.no_grad():
    for p, w in zip(net.params, teacher_cores(comps, 4, 8, 3)):
        p[0] = torch.tensor(w)
x, phi = sample(1, 4096, 8, seed=12, device='cpu', dtype=f64)
with torch.no_grad():
    acts = net.forward(phi)
Z = node_activations(acts)[0]
print('activation column norms per node:', {v: Z[:, i].norm(dim=0).round().tolist() for i, v in enumerate(NODES)})
other = [dict(out=0, s=1.0, coords=((0, 5), (1, 5)))]
lstsq = torch.linalg.lstsq
for drv in ('gelsy', 'gelsd'):
    torch.linalg.lstsq = (lambda A, B, **k: lstsq(A, B, driver='gelsd')) if drv == 'gelsd' else lstsq
    for i in range(n):
        A = alignment(acts, x, comps)[0]
        bad = [(f'comp {k + 1}', NODES[v], f'{A[k, v].item():.6g}') for k in range(3) for v in range(6) if abs(A[k, v] - 1) > 1e-6]
        print(f'{drv} call {i + 1}: min {A.min().item():.15g}, entries off 1 by > 1e-6: {bad},'
              f' unrelated {alignment(acts, x, other)[0, 0, 0].item():.3g}')
torch.linalg.lstsq = lstsq
