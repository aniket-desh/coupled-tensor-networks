"""Byte n-gram baselines: interpolated modified Kneser-Ney (Chen & Goodman 1998), orders 1..8, vectorised numpy.

Order m conditions on the m - 1 bytes before the target (order 1 is the unigram, order 8 sees 7 bytes).

Keys. The m-gram ending at position t is packed into a uint64 with the target in bits 0-7, the byte before it in bits
8-15, and so on: key_m(t) = sum_{j < m} d[t - j] << 8j. Its context (the m - 1 earlier bytes) is key >> 8, and its
lower-order suffix (drop the oldest byte) is key & (2^(8(m-1)) - 1).

Model of order M, for a target w after context h (length M - 1, h' = h without its oldest byte):
  p_m(w | h) = [max(a(hw) - D(a(hw)), 0) + gamma(h) p_{m-1}(w | h')] / A(h)   if A(h) > 0, else p_{m-1}(w | h'),
  A(h) = sum_w a(hw),  gamma(h) = D1 N1(h) + D2 N2(h) + D3 N3+(h),  p_0 = 1/256,
where a is the raw count at the top length m = M and the continuation count N1+(. g) (number of distinct bytes seen
before g) at every lower length, N_k(h) counts the w with a(hw) = k (k >= 3 pooled), and D(a) is D1, D2 or D3+ for
a = 1, 2, >= 3. Because sum_w D(a(hw)) = gamma(h), p_m sums to 1 over w. Discounts come from each length's
count-of-counts n_k (Chen & Goodman's closed form): Y = n1 / (n1 + 2 n2), D_k = k - (k + 1) Y n_{k+1} / n_k, clipped
to [0, k]. They are not tuned on validation. All models M = 1..8 share the continuation statistics, so one pass over
the evaluation positions scores all orders."""
import time

import numpy as np

U64 = np.uint64
MAX_ORDER = 8


def gram_keys(d, m, t):
    """key_m(t) for positions t (each t >= m - 1)."""
    k = np.zeros(len(t), U64)
    for j in range(m):
        k |= d[t - j].astype(U64) << U64(8 * j)
    return k


def count_sorted(k):
    """Unique values of k (sorted) and their counts."""
    s = np.sort(k)
    if s.size == 0:
        return s, np.zeros(0, np.int64)
    start = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    return s[start], np.diff(np.append(start, s.size))


class Level:
    """Statistics of one length m with one count type: m-gram keys g with counts a, and per context h = g >> 8 the
    total A and the numbers N1, N2, N3+ of continuations with count 1, 2, >= 3, plus the discounts."""

    def __init__(self, g, a):
        self.g, self.a = g, a.astype(np.int64)
        n = np.array([(self.a == k).sum() for k in (1, 2, 3, 4)], dtype=float)
        Y = n[0] / (n[0] + 2 * n[1]) if n[0] + n[1] > 0 else 0.5
        D = [k - (k + 1) * Y * n[k] / n[k - 1] if n[k - 1] > 0 else 0.5 * k for k in (1, 2, 3)]
        self.D = np.clip(np.nan_to_num(np.array(D), nan=0.5), 0.0, [1.0, 2.0, 3.0])
        h = g >> U64(8)  # sorted, because g is
        if h.size == 0:
            self.h, self.A, self.N = h, np.zeros(0, np.int64), np.zeros((3, 0), np.int64)
            return
        start = np.flatnonzero(np.concatenate(([True], h[1:] != h[:-1])))
        self.h = h[start]
        self.A = np.add.reduceat(self.a, start)
        self.N = np.stack([np.add.reduceat((self.a == k).astype(np.int64), start) for k in (1, 2)] +
                          [np.add.reduceat((self.a >= 3).astype(np.int64), start)])

    def step(self, q, p_lower):
        """p_m for query m-gram keys q, given p_{m-1} of the same targets under the shorter context."""
        if self.h.size == 0:
            return p_lower.copy()
        a = lookup(self.g, self.a, q)
        hq = q >> U64(8)
        i = np.minimum(np.searchsorted(self.h, hq), self.h.size - 1)
        hit = self.h[i] == hq
        A = np.where(hit, self.A[i], 0).astype(float)
        gamma = np.where(hit, self.D @ self.N[:, i], 0.0)
        disc = np.where(a > 0, self.D[np.clip(a, 1, 3) - 1], 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = (np.maximum(a - disc, 0.0) + gamma * p_lower) / A
        return np.where(A > 0, p, p_lower)

    def nbytes(self):
        return sum(x.nbytes for x in (self.g, self.a, self.h, self.A, self.N))


def lookup(keys, vals, q):
    if keys.size == 0:
        return np.zeros(q.size, np.int64)
    i = np.minimum(np.searchsorted(keys, q), keys.size - 1)
    return np.where(keys[i] == q, vals[i], 0)


class KneserNey:
    def __init__(self, max_order=MAX_ORDER, log=print):
        self.M, self.log = max_order, log

    def fit(self, d):
        """Count every m-gram of the uint8 array d (all positions t >= m - 1), m = 1 .. M."""
        t0 = time.time()
        self.raw, self.cont = [], []
        k = d.astype(U64)  # key_1 at t = 0 .. N-1
        prev_unique = None
        for m in range(1, self.M + 1):
            if m > 1:
                k = k[:-1]  # key_{m-1} at t - 1 for t = m-1 .. N-1
                k <<= U64(8)
                k |= d[m - 1:].astype(U64)
            g, c = count_sorted(k)
            self.raw.append(Level(g, c))
            if prev_unique is not None:  # continuation counts of length m - 1 from the unique m-grams
                suf = g & U64((1 << (8 * (m - 1))) - 1)
                gs, cs = count_sorted(suf)
                self.cont.append(Level(gs, cs))
            prev_unique = g
            self.log(f'  ngram length {m}: {g.size} distinct, {time.time() - t0:.0f}s', flush=True)
        del k
        self.n_train = int(d.size)
        return self

    def probs(self, d, t):
        """(M, len(t)) probabilities of the target d[t] under orders 1..M (each t >= M - 1)."""
        out = np.empty((self.M, len(t)))
        p_cont = np.full(len(t), 1.0 / 256)
        for m in range(1, self.M + 1):
            q = gram_keys(d, m, t)
            out[m - 1] = self.raw[m - 1].step(q, p_cont)
            if m < self.M:
                p_cont = self.cont[m - 1].step(q, p_cont)
        return out

    def dist(self, context):
        """(M, 256) distributions of the next byte after `context` (uint8 array, length >= M - 1, oldest first)."""
        d = np.concatenate([np.asarray(context, np.uint8), np.zeros(1, np.uint8)])
        d = np.tile(d, (256, 1))
        d[:, -1] = np.arange(256)
        flat = d.reshape(-1)
        t = np.arange(256) * d.shape[1] + d.shape[1] - 1
        return self.probs(flat, t).reshape(self.M, 256)

    def bits(self, d, t, chunk=1 << 20):
        """Mean -log2 p per order over target positions t, in chunks."""
        tot = np.zeros(self.M)
        for i in range(0, len(t), chunk):
            tot += -np.log2(self.probs(d, t[i:i + chunk])).sum(1)
        return tot / len(t)

    def summary(self):
        return dict(distinct=[L.g.size for L in self.raw], contexts=[L.h.size for L in self.raw],
                    discounts_raw=[L.D.tolist() for L in self.raw], discounts_cont=[L.D.tolist() for L in self.cont],
                    stats_mb=round(sum(L.nbytes() for L in self.raw + self.cont) / 2 ** 20, 1))
