"""Run studies.lang.interpret on every tree run of an experiment output folder, several at a time.

  python -m studies.lang.interpret_all <out> [--procs 4] [--hours 2] [--only substr,...]

Writes <out>/<run>/interpret.json (skips runs that already have one). Runs are taken in name order after sorting so
that seed 0 comes first; whatever is unfinished at the deadline is listed in <out>/interpret_status.json."""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--hours', type=float, default=2.0)
    ap.add_argument('--only', default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--first', default='', help='comma-separated substrings: within a seed, these runs go first')
    a = ap.parse_args()
    deadline = time.time() + 3600 * a.hours
    runs = []
    for res in Path(a.out).glob('*/result.json'):
        r = json.load(open(res))
        if r['cfg']['model'] != 'tree' or r['status'] not in ('finished', 'truncated'):
            continue
        if a.only and not any(s in r['name'] for s in a.only.split(',')):
            continue
        first = a.first and any(s in r['name'] for s in a.first.split(','))
        runs.append((r['cfg']['seed'], 0 if first else 1, r['name']))
    runs.sort()
    status = {}

    def one(item):
        name = item[-1]
        d = Path(a.out) / name
        if (d / 'interpret.json').exists():
            return name, 'done'
        left = deadline - time.time()
        if left < 60:
            return name, 'skipped (deadline)'
        try:
            p = subprocess.run([sys.executable, '-m', 'studies.lang.interpret', str(d), '--out', str(d / 'interpret.json'),
                                '--device', a.device], timeout=left, capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stderr[-2000:], flush=True)
            return name, 'done' if p.returncode == 0 else f'failed ({p.returncode})'
        except subprocess.TimeoutExpired:
            return name, 'timeout'

    with ThreadPoolExecutor(a.procs) as ex:
        for name, st in ex.map(one, runs):
            status[name] = st
            print(f'{name}: {st}', flush=True)
    json.dump(status, open(Path(a.out) / 'interpret_status.json', 'w'), indent=1)
    print(f"{sum(v == 'done' for v in status.values())}/{len(status)} interpreted")


if __name__ == '__main__':
    main()
