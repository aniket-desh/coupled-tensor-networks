"""TinyStories as raw UTF-8 bytes for next-byte prediction (V = 256), with splits made by story.

Source: a prefix of TinyStoriesV2-GPT4-train.txt fetched with an HTTP range request; the partial story after the last
<|endoftext|> is dropped. Each story is stripped of surrounding whitespace (inner newlines are kept), the stories are
shuffled with a seeded permutation, and whole stories are dealt in that order to test (until >= 5 MB), validation
(>= 5 MB) and train (>= 100 MB); MB = 10^6 bytes. Splits are therefore disjoint by story.

Story boundaries: every split file is SEP s_1 SEP s_2 SEP ... s_K SEP with SEP = 0x00, a byte that never occurs in the
text (checked). Windows are taken at every position and may cross a boundary, but the boundary is visible in the
context, and predicting SEP (end of story) is part of the task. A window is the CTX = 16 bytes before a target position
t, so the targets of a split are the positions t in [CTX, len).

Pilot set: the prefix of train.bin up to and including the last SEP at or before byte 10^7 (about 10 MB of whole
stories), saved as pilot.bin.

Files in data/tinystories/: {train,val,test,pilot}.bin (uint8, np.fromfile-able) and meta.json (source URL, byte range,
split seed, story counts, sizes, sha256 of each file).

Prepare (on the coordinator; the raw prefix is cached at --raw, outside data/ so it is not packed for workers):
  python -m studies.lm.data prepare --raw <path to cache the 400 MB prefix>
Verify sizes and hashes against meta.json:
  python -m studies.lm.data verify"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'
DELIM = b'<|endoftext|>'
DIR = Path('data/tinystories')
SEP = 0
CTX = 16
V = 256
SPLITS = ('train', 'val', 'test', 'pilot')


def download(raw, nbytes):
    """Fetch bytes [0, nbytes) of the source file to `raw` (skipped if it already has that size)."""
    raw = Path(raw)
    if raw.exists() and raw.stat().st_size == nbytes:
        return
    raw.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={'Range': f'bytes=0-{nbytes - 1}', 'User-Agent': 'python-urllib'})
    with urllib.request.urlopen(req) as r, open(raw, 'wb') as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    assert raw.stat().st_size == nbytes, f'expected {nbytes} bytes, got {raw.stat().st_size}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def pack(stories):
    """SEP s_1 SEP s_2 ... s_K SEP as one uint8 array."""
    return np.frombuffer(bytes([SEP]) + b''.join(s + bytes([SEP]) for s in stories), dtype=np.uint8)


def prepare(raw, out=DIR, nbytes=400_000_000, seed=0, sizes=(('test', 5_000_000), ('val', 5_000_000),
                                                              ('train', 100_000_000)), pilot_bytes=10_000_000):
    download(raw, nbytes)
    text = Path(raw).read_bytes()
    pieces = text.split(DELIM)
    del text
    n_partial = len(pieces[-1])
    stories = [s.strip() for s in pieces[:-1]]  # pieces[-1] follows the last delimiter: partial, dropped
    del pieces
    n_empty = sum(len(s) == 0 for s in stories)
    stories = [s for s in stories if s]
    assert not any(bytes([SEP]) in s for s in stories), 'SEP byte occurs inside a story'
    order = np.random.default_rng(seed).permutation(len(stories))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(source=URL, byte_range=[0, nbytes - 1], raw_sha256=sha256(raw), split_seed=seed, sep=SEP, ctx=CTX,
                stories_in_prefix=len(stories), empty_dropped=n_empty, partial_tail_bytes_dropped=n_partial,
                order='test, val, train dealt in permutation order; each split takes whole stories until it reaches '
                      'its target size', target_sizes=dict(sizes), files={})
    k = 0
    for name, target in sizes:
        chosen, size = [], 1
        while size < target:
            s = stories[order[k]]
            chosen.append(s)
            size += len(s) + 1
            k += 1
        arr = pack(chosen)
        arr.tofile(out / f'{name}.bin')
        meta['files'][name] = dict(bytes=int(arr.size), stories=len(chosen), targets=int(arr.size - CTX))
    meta['stories_unused'] = len(stories) - k
    train = np.fromfile(out / 'train.bin', dtype=np.uint8)
    seps = np.flatnonzero(train[:pilot_bytes + 1] == SEP)
    pilot = train[:seps[-1] + 1]
    pilot.tofile(out / 'pilot.bin')
    meta['files']['pilot'] = dict(bytes=int(pilot.size), stories=int(len(seps) - 1), targets=int(pilot.size - CTX),
                                  note=f'prefix of train.bin up to the last SEP at or before byte {pilot_bytes}')
    for name in SPLITS:
        meta['files'][name]['sha256'] = sha256(out / f'{name}.bin')
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    return meta


def load(split, root=DIR):
    """The split as a uint8 array."""
    return np.fromfile(Path(root) / f'{split}.bin', dtype=np.uint8)


def verify(root=DIR, hashes=True):
    """Sizes, story counts (number of SEPs - 1) and sha256 of every split against meta.json; returns a list of problems."""
    root = Path(root)
    meta = json.load(open(root / 'meta.json'))
    bad = []
    for name in SPLITS:
        f, m = root / f'{name}.bin', meta['files'][name]
        if not f.exists():
            bad.append(f'{f} missing')
            continue
        d = load(name, root)
        if d.size != m['bytes']:
            bad.append(f'{name}: {d.size} bytes, meta says {m["bytes"]}')
        if int((d == SEP).sum()) - 1 != m['stories'] or d[0] != SEP or d[-1] != SEP:
            bad.append(f'{name}: separator count or ends do not match {m["stories"]} stories')
        if hashes and sha256(f) != m['sha256']:
            bad.append(f'{name}: sha256 mismatch')
    return bad


def positions(n_bytes):
    """All target positions of a split of n_bytes bytes."""
    return np.arange(CTX, n_bytes, dtype=np.int64)


def subsample(n_bytes, n, seed):
    """A fixed sorted sample of n target positions (all of them if n >= the number of targets)."""
    if n >= n_bytes - CTX:
        return positions(n_bytes)
    return np.sort(np.random.default_rng(seed).choice(n_bytes - CTX, size=n, replace=False)).astype(np.int64) + CTX


def windows(data, t):
    """data: (N,) uint8 tensor; t: (B,) long target positions -> x (B, CTX) long, the bytes t-CTX .. t-1 (oldest first,
    so x[:, -1] is the byte just before the target), and y (B,) long."""
    import torch
    off = torch.arange(-CTX, 0, device=t.device)
    return data[t[:, None] + off].long(), data[t].long()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['prepare', 'verify'])
    p.add_argument('--raw', help='where to cache the raw prefix (outside data/)')
    p.add_argument('--bytes', type=int, default=400_000_000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.cmd == 'prepare':
        print(json.dumps(prepare(a.raw, nbytes=a.bytes, seed=a.seed), indent=1))
    else:
        problems = verify()
        print('\n'.join(problems) or 'data ok')
        raise SystemExit(1 if problems else 0)
