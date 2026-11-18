"""Compare each rerun with the original run it repeats (same script, seed and settings; only the checkpoints differ).

    PYTHONPATH=. python studies/hmm/belief-animation/anim_compare.py FRAMES_DIR OUT_JSON

For every run in FRAMES_DIR: the final excess test loss against the original's, and the per-position R2 of the drawn
probe (sequence split, test half) at the first and the last checkpoint against the same numbers in the original
analysis.json. The probes, analysis sequences and split are the same, so differences come from the training trajectory
(a different GPU model changes float32 rounding, and training amplifies it) and, for the RHM, from fitting on the
span of the beliefs instead of all D_t coordinates (exact up to rounding).

A rerun `matches` its original if the final excess loss is within max(0.002 nats, 50 %) of the original's and the
pooled R2 over positions at the last checkpoint is within 0.02 (the pooled R2 here is over all analysed positions,
with the original's per-position R2 pooled with this run's SST weights)."""
import json
import sys
from pathlib import Path

import numpy as np

STUDIES = Path(__file__).resolve().parents[2]
ORIG = {'hmm': 'hmm/hmm-belief-geometry', 'rhm-transformer': 'rhm/rhm-belief-geometry',
        'rhm-arch': 'rhm/rhm-belief-architectures', 'rhm-tree': 'rhm/rhm-ttn-belief-geometry'}


def drawn_key(fam, cfg, t):
    if fam == 'hmm':
        kind = cfg['model']
        return dict(transformer='L4', chain='state^proj').get(kind) or ('top^proj' if t <= cfg['T'] // 2 else 'top_bil^proj')
    if fam == 'rhm-transformer':
        return 'L4'
    return 'state' if cfg.get('arm') == 'chain' else 'top_bil'


def main():
    frames_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    print(f"{'run':24s} {'frames':>6s} {'excess':>9s} {'orig':>9s} | {'R2 init':>8s} {'orig':>8s} | {'R2 final':>8s} {'orig':>8s} {'max |d| per pos':>16s}  match")
    for jf in sorted(frames_dir.glob('*.json')):
        meta = json.load(open(jf))
        z = np.load(frames_dir / f"{meta['run']}.npz")
        fam, cfg = meta['family'], meta['cfg']
        a = json.load(open(STUDIES / ORIG[fam] / 'results' / meta['run'] / 'analysis.json'))
        o_exc = a.get('final_excess', a['hist'][-1]['excess'])
        cks = sorted(a['metrics'], key=int)
        w, pos = z['sst'], [int(t) for t in z['positions']]
        res = dict(run=meta['run'], n_frames=len(z['steps']), last_step=int(z['steps'][-1]), excess=meta['hist'][-1]['excess'],
                   excess_orig=o_exc, ms_per_update=meta['ms_per_update'])
        for tag, f, ck in (('init', 0, cks[0]), ('final', -1, cks[-1])):
            mine = 1 - z['sse'][f] / w
            orig = np.array([a['metrics'][ck]['per_pos'][drawn_key(fam, cfg, t)][str(t)]['R2'] for t in pos])
            res[f'R2_{tag}'] = float(1 - z['sse'][f].sum() / w.sum())
            res[f'R2_{tag}_orig'] = float(1 - ((1 - orig) * w).sum() / w.sum())
            res[f'maxdiff_{tag}'] = float(np.abs(mine - orig).max())
        res['match'] = bool(abs(res['excess'] - o_exc) <= max(0.002, 0.5 * abs(o_exc)) and abs(res['R2_final'] - res['R2_final_orig']) <= 0.02
                            and int(cks[-1]) == res['last_step'])
        rows.append(res)
        print(f"{res['run']:24s} {res['n_frames']:6d} {res['excess']:9.5f} {o_exc:9.5f} | {res['R2_init']:8.4f} {res['R2_init_orig']:8.4f} | "
              f"{res['R2_final']:8.4f} {res['R2_final_orig']:8.4f} {res['maxdiff_final']:16.4f}  {'yes' if res['match'] else 'NO'}")
    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
