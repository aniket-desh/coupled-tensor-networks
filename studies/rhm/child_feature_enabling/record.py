"""Provenance shared by the study's drivers: source revision, local diff, library versions, machine."""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(tensor):
    """Hash of a tensor's values (contiguous little-endian bytes of its own dtype)."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def provenance(*source_files):
    """Commit, a hash of the uncommitted diff of tracked files (None when clean), and hashes of the driver sources."""
    diff = _git('diff', 'HEAD')
    return dict(
        commit=_git('rev-parse', 'HEAD').strip(),
        uncommitted_diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        source_sha256={str(Path(f).resolve().relative_to(ROOT)): sha256(f) for f in source_files},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      torch=str(torch.__version__)),
        machine=dict(platform=platform.platform(), processor=platform.processor() or platform.machine(),
                     cpus=os.cpu_count(), torch_threads=torch.get_num_threads()),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def dump(path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=indent) + '\n')
    tmp.replace(path)
    print('wrote', path, flush=True)
