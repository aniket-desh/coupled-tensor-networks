"""Where and how much computation two outputs share. Exact population GD on y1, y2, y3 = teachers.agreement(S) for
agreement sets S = none, leaf, quarter, split2, half, split4 (y1 and y3 are identical in every family; only how much
of y2 agrees with y1 changes). The same 16 inits in every family, so runs pair by seed. The quarter family is rerun
on N = 2^17 sampled inputs per run as the finite-data replicate. Groups run in parallel on the GPU.
Usage: PYTHONPATH=. python studies/ttn/sharing-location/run.py --out <dir> [--hours H] [--smoke] [--groups ...] [--eta E]"""
from studies.ttn import common, teachers

ETA = 0.05
SEEDS = 16
T_MAX = 1e5


def group(a, sp):
    common.run_group(a, sp['name'], teachers.agreement(sp['S']), R=2 if a.smoke else (a.seeds or SEEDS), r=8, alpha=1.0,
                     eta=a.eta or ETA, t_max=5.0 if a.smoke else (a.t_max or T_MAX), N=sp['N'], N_probe=512 if a.smoke else 2 ** 14,
                     seed=2000, extra_meta=dict(S=sp['S'], shared_cores=teachers.shared_cores(sp['S'])))


if __name__ == '__main__':
    a = common.args()
    specs = [dict(name=S, S=S, N=0) for S in teachers.SETS]
    specs.append(dict(name='quarter_sampled', S='quarter', N=64 if a.smoke else 2 ** 17))
    print(common.run_parallel(a, [specs[2], specs[-1]] if a.smoke else specs, group), flush=True)
