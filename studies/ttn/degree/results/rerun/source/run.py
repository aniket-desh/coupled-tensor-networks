"""Degree under a constant-biased leaf init. One output, a product over D patches (teachers.degree), D = 1, 2, 4, 6, 8,
trained by exact population GD from inits whose leaves put a fraction w of their variance on the constant:
w = 0.25 and 0.5 (biased: each active patch must switch its leaf away from the constant), and w = 1/9 (isotropic) at
D = 1 and 8 only, where the basis-permutation symmetry says degree cannot matter. The same 16 inits per w across D.
Groups run in parallel on the GPU.
Usage: PYTHONPATH=. python studies/ttn/degree/run.py --out <dir> [--hours H] [--smoke] [--groups ...] [--eta E]"""
from studies.ttn import common, teachers

ETA = 0.05
SEEDS = 16
T_MAX = 1e5


def group(a, sp):
    common.run_group(a, sp['name'], teachers.degree(sp['D']), R=2 if a.smoke else (a.seeds or SEEDS), r=8, alpha=1.0,
                     eta=a.eta or ETA, t_max=5.0 if a.smoke else (a.t_max or T_MAX), N=0, N_probe=512 if a.smoke else 2 ** 14,
                     seed=3000, net_kw=dict(const_weight=sp['w']), extra_meta=dict(D=sp['D'], w=sp['w']))


if __name__ == '__main__':
    a = common.args()
    specs = [dict(name=f'w{w:.3f}_D{D}', w=w, D=D) for w in (0.25, 0.5) for D in (1, 2, 4, 6, 8)]
    specs += [dict(name=f'w{1 / 9:.3f}_D{D}', w=1 / 9, D=D) for D in (1, 8)]
    print(common.run_parallel(a, specs[:2] if a.smoke else specs, group), flush=True)
