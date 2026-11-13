"""Running experiments and keeping their outputs: JSON results, source snapshots, a process pool, a CLI."""
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, default=float)


def load_json(path):
    return json.load(open(path))


def pmap(fn, items, chunksize=1):
    """Parallel map in order. Tasks seed their own RNGs, so results don't depend on the worker count."""
    with Pool(min(14, os.cpu_count())) as p:
        return p.map(fn, items, chunksize=chunksize)


def snapshot(results_dir, study_dir):
    """Copy the library and the study's scripts next to the results they produce."""
    dest = Path(results_dir) / 'source'
    for f in list((ROOT / 'ctn').glob('*.py')) + list(Path(study_dir).glob('*.py')):
        sub = dest / ('ctn' if f.parent.name == 'ctn' else '')
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, sub)


def main(namespace, default, study_dir):
    """`python -m studies.<study>.experiments [name ...]` runs the named functions (default: all of `default`)."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    snapshot(Path(study_dir) / 'results', study_dir)
    for name in sys.argv[1:] or default:
        namespace[name]()
        print('done', name, flush=True)
