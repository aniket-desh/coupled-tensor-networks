"""One table from the norm-diagnostic replays (results/norms_root_g2_*.json, written by norms.py), to compare the
r = 64 root run on grammar 2 at peak LR 3e-3 with the same run at 1e-3, with AMSGrad, and with r = 16.

  python studies/rhm/rhm-dynamics/norms_summary.py        # prints and writes results/norms_summary.txt

Columns (all on the training batch of the logged update; the replays log every 10 updates):
  rel0, sat0   Adam's step on a level-1 fusion core at update 0 relative to the core's Frobenius norm, mean over the
               8 nodes, and the value lr_0 sqrt(r^3) / ||W|| it takes if every entry moves by the full lr_0
               (lr_0 = peak/100, the first warmup step)
  ||W||, ||A|| Frobenius norm of a level-1 core and of a leaf embedding (mean over nodes) at update 0 -> last update
  after update 1,000: the number of logged batches with loss > 0.1 and > 1, the largest batch loss, the median and
               the smallest logit norm, and the median relative step of a level-1 core on batches with loss < 0.01
               and on batches with loss > 0.5."""
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
mean = lambda v: sum(v) / len(v)


def row(path):
    r = json.loads(path.read_text())
    R, lr = r['cfg']['r'], r['lr']
    s, loss = r['step'], r['loss']
    rel = [mean([u / p for u, p in zip(us, ps)]) for us, ps in zip(r['upd']['L1'], r['param']['L1'])]
    W = [mean(p) for p in r['param']['L1']]
    A = [mean(p) for p in r['param']['leaf']]
    logit = [x[0] for x in r['act']['root']]
    late = [i for i, t in enumerate(s) if t > 1000]
    quiet = [rel[i] for i in late if loss[i] < 0.01]
    loud = [rel[i] for i in late if loss[i] > 0.5]
    name = f"r={R} lr {lr:g}" + (' amsgrad' if r.get('amsgrad') else '')
    return (f"{name:22s} {s[-1]:6d} {rel[0]:9.2e} {lr / 100 * R ** 1.5 / W[0]:9.2e} {W[0]:6.1f} -> {W[-1]:5.1f} "
            f"{A[0]:5.2f} -> {A[-1]:4.2f} {sum(loss[i] > 0.1 for i in late):5d} {sum(loss[i] > 1 for i in late):5d} "
            f"/ {len(late):4d} {max(loss[i] for i in late):7.2f} {median(logit[i] for i in late):7.1f} "
            f"{min(logit[i] for i in late):6.1f} {median(quiet) if quiet else float('nan'):9.1e} "
            f"{median(loud) if loud else float('nan'):9.1e}")


if __name__ == '__main__':
    head = (f"{'run (root, grammar 2)':22s} {'last':>6s} {'rel0':>9s} {'sat0':>9s} {'||W|| 0 -> last':>15s} "
            f"{'||A|| 0 -> last':>13s} {'>0.1':>5s} {'>1':>5s} {'of':>6s} {'max':>7s} {'|logit|':>7s} {'min':>6s} "
            f"{'rel quiet':>9s} {'rel loud':>9s}")
    lines = [head] + [row(p) for p in sorted((HERE / 'results').glob('norms_root_g2_*.json'))]
    print('\n'.join(lines))
    (HERE / 'results' / 'norms_summary.txt').write_text('\n'.join(lines) + '\n')
