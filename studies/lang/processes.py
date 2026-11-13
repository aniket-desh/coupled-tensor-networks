"""Synthetic token processes for the 16-leaf TreeTN (plan: notes/2026-11-13-language-processes/plan.md).

Every process produces inputs x (B, 16) and a target y with an exact Bayes predictor p*(y | x) that depends on the
tokens only, so it can be evaluated on any input, including spliced ones (patching). Training uses the generative
y (sample_xy); `checks.py` confirms that the empirical conditional frequencies of the generative y match bayes(x).

Conventions:
  leaves 0..15 (0-based); a model node covers a set S of leaf positions (perm-aware, see `spans`);
  eps: with probability eps the target is replaced by a uniform draw, so Bayes logits stay finite;
  reserved input tokens (mask, blank, trigger) take index V, so the input vocabulary is V + 1.

summary(x, S) returns the quantities the theory predicts a bond over S could carry, as {name: (kind, target, C)}:
  kind 'class': target (B,) long with C classes; kind 'reg': target (B, C) float64. Only quantities that are functions
  of x_S are returned, so each has a probe ceiling of 1 (class) or R^2 = 1 (reg).
resample_same(x, S, g) returns x' that agrees with x outside S and has the same primary summary over S but different
surface tokens inside S (or None where the process has no such move): the invariance test.
"""
import math

import torch

N = 16


def spans(perm=None, n=N):
    """{(l, k): list of leaf positions} for tree node k at level l (l = 0 leaves .. 3 the root's children), with the
    model's leaf order perm (x[:, perm] enters the tree, so model leaf m reads position perm[m])."""
    p = list(range(n)) if perm is None else list(perm)
    out = {}
    for l in range(4):
        w = 2 ** l
        for k in range(n // w):
            out[(l, k)] = sorted(p[k * w:(k + 1) * w])
    return out


def _noisy(y, eps, O, g):
    """Replace each target by a uniform draw with probability eps."""
    flip = torch.rand(y.shape, generator=g, device=y.device) < eps
    return torch.where(flip, torch.randint(0, O, y.shape, generator=g, device=y.device), y)


def _mix(q, eps):
    """Bayes posterior with target noise: (1 - eps) q + eps / O."""
    return (1 - eps) * q + eps / q.shape[1]


def _onehot(y, O):
    return torch.nn.functional.one_hot(y, O).double()


def _centered_log(p, floor=0.0):
    lp = torch.log(p.clamp_min(floor) if floor else p)
    return lp - lp.mean(1, keepdim=True)


class Process:
    name = 'base'
    primary = ()  # summaries in priority order; the first one present at a node is used for the invariance test

    def __init__(self, V, eps=0.05, pseed=0):
        self.V, self.eps, self.pseed = V, eps, pseed
        self.g0 = torch.Generator().manual_seed(1_000_003 * (pseed + 1) + sum(map(ord, self.name)))
        self.dev = 'cpu'

    # tables are drawn on the CPU from g0 and moved with to()
    def to(self, device):
        for k, v in list(vars(self).items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        self.dev = device
        return self

    V_in = property(lambda self: self.V)
    O = property(lambda self: self.V)

    def sample_xy(self, B, g):
        raise NotImplementedError

    def bayes(self, x):
        raise NotImplementedError

    def valid(self, x):
        return torch.ones(len(x), dtype=torch.bool, device=x.device)

    def strata(self, x):
        return None

    def summary(self, x, S):
        return {}

    def resample_same(self, x, S, g):
        return None

    def inv_mask(self, x, S):
        """Examples on which the invariance test is meaningful (None = all)."""
        return None

    def support(self, name, S):
        """Positions in S whose tokens determine summary `name` (the counterfactual for patching that summary splices
        only these). Default: all of S."""
        return list(S)

    def primary_of(self, summ):
        return next((n for n in self.primary if n in summ), None)

    def _uniform(self, B, g, V=None, n=N):
        return torch.randint(0, V or self.V, (B, n), generator=g, device=g.device)


# ------------------------------------------------------------------------------------- lang-source

class FixedSource(Process):
    """x iid uniform over V; y = sigma(x_j) with target noise eps (sigma a fixed random permutation). The whole
    predictive computation depends on one token, so lag/position changes only where the useful input sits."""
    name = 'fixed'
    primary = ('src',)

    def __init__(self, V=8, j=0, eps=0.05, pseed=0):
        super().__init__(V, eps, pseed)
        self.j = j
        self.sigma = torch.randperm(V, generator=self.g0)

    def sample_xy(self, B, g):
        x = self._uniform(B, g)
        return x, _noisy(self.sigma[x[:, self.j]], self.eps, self.O, g)

    def bayes(self, x):
        return _mix(_onehot(self.sigma[x[:, self.j]], self.O), self.eps)

    def summary(self, x, S):
        out = {}
        if self.j in S:
            out['src'] = ('class', x[:, self.j], self.V)
        d = [t for t in S if t != self.j]
        if d:  # a distractor in the same span: a selective bond should let it fade
            out['distractor'] = ('class', x[:, d[0]], self.V)
        return out

    def support(self, name, S):
        return [self.j] if name == 'src' else [t for t in S if t != self.j][:1]

    def resample_same(self, x, S, g):
        d = [t for t in S if t != self.j]
        if not d or self.j not in S:
            return None
        x2 = x.clone()
        x2[:, d] = torch.randint(0, self.V, (len(x), len(d)), generator=g, device=x.device)
        return x2


class TwoSource(Process):
    """x iid uniform over V; y = f(x_i, x_j) with target noise eps, f a fixed random V x V table (values uniform, so it
    is not additive and each source alone is only partly informative). The lowest common ancestor of leaves i and j
    is the first node where the two inputs can be combined."""
    name = 'two'
    primary = ('f', 'src_i', 'src_j')

    def __init__(self, V=8, i=0, j=1, eps=0.05, pseed=0):
        super().__init__(V, eps, pseed)
        self.i, self.j = i, j
        self.f = torch.randint(0, V, (V, V), generator=self.g0)

    def sample_xy(self, B, g):
        x = self._uniform(B, g)
        return x, _noisy(self.f[x[:, self.i], x[:, self.j]], self.eps, self.O, g)

    def bayes(self, x):
        return _mix(_onehot(self.f[x[:, self.i], x[:, self.j]], self.O), self.eps)

    def summary(self, x, S):
        out = {}
        if self.i in S:
            out['src_i'] = ('class', x[:, self.i], self.V)
        if self.j in S:
            out['src_j'] = ('class', x[:, self.j], self.V)
        if self.i in S and self.j in S:
            out['f'] = ('class', self.f[x[:, self.i], x[:, self.j]], self.V)
        d = [t for t in S if t not in (self.i, self.j)]
        if d:
            out['distractor'] = ('class', x[:, d[0]], self.V)
        return out

    def support(self, name, S):
        return dict(src_i=[self.i], src_j=[self.j], f=[self.i, self.j]).get(
            name, [t for t in S if t not in (self.i, self.j)][:1])

    def resample_same(self, x, S, g):
        d = [t for t in S if t not in (self.i, self.j)]
        if not d or not (self.i in S or self.j in S):
            return None
        x2 = x.clone()
        x2[:, d] = torch.randint(0, self.V, (len(x), len(d)), generator=g, device=x.device)
        return x2


# ------------------------------------------------------------------------------------ lang-evidence

class MaskedBigram(Process):
    """Stationary Markov chain x_0 ~ pi, x_{t+1} ~ P(. | x_t); leaf p is masked (token V) and y = x_p. Rows of P put
    Dirichlet(1) weight on m = 3 random successors, mixed with floor 0.02 / V so every entry is positive.
    p*(y | x) is proportional to P(y | x_{p-1}) P(x_{p+1} | y): forward evidence plus backward evidence in logit space."""
    name = 'mbigram'
    primary = ('comb', 'fwd', 'bwd')
    CLIP = 1e-3

    def __init__(self, V=16, p=5, m=3, floor=0.02, pseed=0):
        super().__init__(V, 0.0, pseed)
        self.p, self.m = p, m
        P = torch.zeros(V, V, dtype=torch.float64)
        for a in range(V):
            succ = torch.randperm(V, generator=self.g0)[:m]
            w = -torch.log(torch.rand(m, generator=self.g0, dtype=torch.float64))  # Dirichlet(1)
            P[a, succ] = w / w.sum()
        self.P = (1 - floor) * P + floor / V
        pi = torch.full((V,), 1.0 / V, dtype=torch.float64)
        for _ in range(2000):
            pi = pi @ self.P
        self.pi = pi / pi.sum()

    V_in = property(lambda self: self.V + 1)

    def _chain(self, B, g):
        x = torch.empty(B, N, dtype=torch.long, device=g.device)
        x[:, 0] = torch.multinomial(self.pi.expand(B, -1), 1, generator=g)[:, 0]
        for t in range(1, N):
            x[:, t] = torch.multinomial(self.P[x[:, t - 1]], 1, generator=g)[:, 0]
        return x

    def sample_xy(self, B, g):
        x = self._chain(B, g)
        y = x[:, self.p].clone()
        x[:, self.p] = self.V
        return x, y

    def _ends(self, x):
        p = self.p
        a = x[:, p - 1] if p > 0 else None
        b = x[:, p + 1] if p < N - 1 else None
        return a, b

    def bayes(self, x):
        a, b = self._ends(x)
        lw = torch.zeros(len(x), self.V, dtype=torch.float64, device=x.device)
        lw = lw + (torch.log(self.P[a]) if a is not None else torch.log(self.pi))
        if b is not None:
            lw = lw + torch.log(self.P[:, b].T)
        return torch.softmax(lw, 1)

    def summary(self, x, S):
        p, out = self.p, {}
        a, b = self._ends(x)
        has_a, has_b = (p - 1) in S, (p + 1) in S
        # evidence vectors are clipped at log(CLIP) before centering: the unclipped log tail (down to ~ -13 nats)
        # carries most of the variance but almost none of the prediction, so R^2 would be set by it
        if has_a:
            out['fwd'] = ('reg', _centered_log(self.P[a], self.CLIP), self.V)
        if has_b:
            out['bwd'] = ('reg', _centered_log(self.P[:, b].T, self.CLIP), self.V)
        if has_a and has_b:
            out['comb'] = ('reg', _centered_log(self.bayes(x), self.CLIP), self.V)
        return out

    def support(self, name, S):
        return dict(fwd=[self.p - 1], bwd=[self.p + 1], comb=[self.p - 1, self.p + 1])[name]

    def resample_same(self, x, S, g):
        keep = {self.p - 1, self.p, self.p + 1}
        d = [t for t in S if t not in keep]
        if not d or not (keep & set(S)) - {self.p}:
            return None
        x2 = x.clone()
        x2[:, d] = torch.randint(0, self.V, (len(x), len(d)), generator=g, device=x.device)
        return x2


class Trigger(Process):
    """Background x_0..x_14 iid uniform over V symbols; with probability 1/2 one position t in 0..14 holds the reserved
    trigger token V. x_15 is the current token b. y = c(b) if a trigger is present, else d(b) (fixed random maps with
    c(b) != d(b) for every b), with target noise eps. Each subtree only needs the flag 'a trigger occurs in me', and
    flags merge by OR."""
    name = 'trigger'
    primary = ('flag',)

    def __init__(self, V=16, eps=0.05, pseed=0):
        super().__init__(V, eps, pseed)
        self.c = torch.randperm(V, generator=self.g0)
        self.d = (self.c + 1 + torch.randint(0, V - 1, (V,), generator=self.g0)) % V  # never equal to c

    V_in = property(lambda self: self.V + 1)

    def sample_xy(self, B, g):
        x = self._uniform(B, g)
        on = torch.rand(B, generator=g, device=g.device) < 0.5
        t = torch.randint(0, N - 1, (B,), generator=g, device=g.device)
        x[on, t[on]] = self.V
        return x, _noisy(self._clean(x), self.eps, self.O, g)

    def _flag(self, x, S=range(N - 1)):
        S = [t for t in S if t < N - 1]
        return (x[:, S] == self.V).any(1) if S else torch.zeros(len(x), dtype=torch.bool, device=x.device)

    def _clean(self, x):
        b = x[:, N - 1]
        return torch.where(self._flag(x), self.c[b], self.d[b])

    def bayes(self, x):
        return _mix(_onehot(self._clean(x), self.O), self.eps)

    def valid(self, x):
        return (x[:, :N - 1] == self.V).sum(1) <= 1

    def strata(self, x):
        """0 = no trigger; otherwise 1 + level of the lowest common ancestor of the trigger and leaf 15 (1..4)."""
        has = self._flag(x)
        t = (x[:, :N - 1] == self.V).double().argmax(1)
        lvl = torch.zeros_like(t)
        for l in range(1, 5):
            lvl = torch.where((lvl == 0) & (t // 2 ** l == (N - 1) // 2 ** l), torch.full_like(t, l), lvl)
        return torch.where(has, lvl, torch.zeros_like(t))

    def summary(self, x, S):
        out = {}
        bg = [t for t in S if t < N - 1]
        if bg:
            out['flag'] = ('class', self._flag(x, bg).long(), 2)
            if len(bg) > 1:  # where in the span the trigger sits (len(bg) = none): should be discarded
                hit = (x[:, bg] == self.V)
                out['where'] = ('class', torch.where(hit.any(1), hit.double().argmax(1), torch.full_like(hit[:, 0], len(bg), dtype=torch.long)), len(bg) + 1)
        if N - 1 in S:
            out['cur'] = ('class', x[:, N - 1], self.V)
        return out

    def support(self, name, S):
        return [N - 1] if name == 'cur' else [t for t in S if t < N - 1]

    def inv_mask(self, x, S):
        """Only examples with a trigger in the span: moving it tests whether the bond discards where it sat."""
        return self._flag(x, S)

    def resample_same(self, x, S, g):
        """Move the trigger (if any) to another background position of S and redraw the other background tokens."""
        bg = [t for t in S if t < N - 1]
        if len(bg) < 2:
            return None
        x2 = x.clone()
        B = len(x)
        has = self._flag(x, bg)
        x2[:, bg] = torch.randint(0, self.V, (B, len(bg)), generator=g, device=x.device)
        pos = torch.tensor(bg, device=x.device)[torch.randint(0, len(bg), (B,), generator=g, device=x.device)]
        rows = torch.nonzero(has)[:, 0]
        x2[rows, pos[rows]] = self.V
        return x2


# -------------------------------------------------------------------------------------- lang-parity

class Parity(Process):
    """x_t iid over {0..q-1} (q = 2: P(1) = 1/2 + bias); y = (sum_{t in K} x_t) mod q with target noise eps, where K is
    the first k leaves (an aligned subtree when k is a power of 2). With bias 0 every proper subset of K is independent
    of y. A scalar bond suffices for q = 2 (h_A = prod_{t in A cap K} s_t); q bonds suffice for the mod-q sum."""
    name = 'parity'
    primary = ('par',)

    def __init__(self, k=4, q=2, bias=0.0, eps=0.05, pseed=0):
        super().__init__(q, eps, pseed)
        self.k, self.q, self.bias = k, q, bias

    def sample_xy(self, B, g):
        if self.q == 2:
            x = (torch.rand(B, N, generator=g, device=g.device) < 0.5 + self.bias).long()
        else:
            x = self._uniform(B, g)
        return x, _noisy(self._clean(x), self.eps, self.O, g)

    def _clean(self, x, S=None):
        K = [t for t in range(self.k) if S is None or t in S]
        return x[:, K].sum(1) % self.q

    def bayes(self, x):
        return _mix(_onehot(self._clean(x), self.O), self.eps)

    def summary(self, x, S):
        if not any(t < self.k for t in S):
            return {}
        return {'par': ('class', self._clean(x, S), self.q)}

    def support(self, name, S):
        return [t for t in S if t < self.k]

    def resample_same(self, x, S, g):
        """Redraw tokens of S outside K, and for |S cap K| >= 2 change two K tokens by +a and -a (mod q), keeping the
        partial sum."""
        K = [t for t in S if t < self.k]
        if not K:
            return None
        x2 = x.clone()
        B = len(x)
        out = [t for t in S if t >= self.k]
        if out:
            x2[:, out] = torch.randint(0, self.q, (B, len(out)), generator=g, device=x.device)
        if len(K) >= 2:
            perm = torch.argsort(torch.rand(B, len(K), generator=g, device=x.device), 1)[:, :2]
            idx = torch.tensor(K, device=x.device)[perm]
            a = torch.randint(1, self.q, (B,), generator=g, device=x.device)
            r = torch.arange(B, device=x.device)
            x2[r, idx[:, 0]] = (x2[r, idx[:, 0]] + a) % self.q
            x2[r, idx[:, 1]] = (x2[r, idx[:, 1]] - a) % self.q
        if not out and len(K) < 2:
            return None
        return x2


# ----------------------------------------------------------------------------------- lang-induction

class Retrieval(Process):
    """Unambiguous associative retrieval. Leaves (2m, 2m+1), m = 0..6, hold key-value pairs with 7 distinct keys
    (drawn without replacement from V) and uniform values; leaf 14 is the query, the key of a uniformly chosen pair;
    leaf 15 is a blank (token V). y = that pair's value, with target noise eps. A subtree that does not see the query
    must keep its pairs' associations for every possible query; strata = index of the target pair (0..6)."""
    name = 'retrieval'
    primary = ('dict', 'ret')
    P = 7

    def __init__(self, V=8, eps=0.05, pseed=0):
        assert V >= self.P
        super().__init__(V, eps, pseed)

    V_in = property(lambda self: self.V + 1)

    def sample_xy(self, B, g):
        keys = torch.argsort(torch.rand(B, self.V, generator=g, device=g.device), 1)[:, :self.P]
        vals = torch.randint(0, self.V, (B, self.P), generator=g, device=g.device)
        x = torch.full((B, N), self.V, dtype=torch.long, device=g.device)
        x[:, 0:2 * self.P:2], x[:, 1:2 * self.P:2] = keys, vals
        t = torch.randint(0, self.P, (B,), generator=g, device=g.device)
        r = torch.arange(B, device=g.device)
        x[:, 14] = keys[r, t]
        return x, _noisy(vals[r, t], self.eps, self.O, g)

    def _match(self, x):
        return x[:, 0:2 * self.P:2] == x[:, 14:15]  # (B, 7)

    def bayes(self, x):
        m = self._match(x).double()
        q = torch.zeros(len(x), self.V, dtype=torch.float64, device=x.device)
        q.scatter_add_(1, x[:, 1:2 * self.P:2], m)
        return _mix(q / q.sum(1, keepdim=True).clamp_min(1e-300), self.eps)

    def valid(self, x):
        keys = x[:, 0:2 * self.P:2]
        distinct = (keys[:, :, None] != keys[:, None, :]).sum((1, 2)) == self.P * (self.P - 1)
        inrange = (x[:, :2 * self.P] < self.V).all(1) & (x[:, 14] < self.V) & (x[:, 15] == self.V)
        return distinct & (self._match(x).sum(1) == 1) & inrange

    def strata(self, x):
        return self._match(x).double().argmax(1)

    def _pairs(self, S):
        return [m for m in range(self.P) if 2 * m in S and 2 * m + 1 in S]

    def summary(self, x, S):
        out = {}
        ms = self._pairs(S)
        if ms:
            D = torch.zeros(len(x), self.V * self.V, dtype=torch.float64, device=x.device)
            for m in ms:
                D[torch.arange(len(x)), x[:, 2 * m] * self.V + x[:, 2 * m + 1]] = 1.0
            out['dict'] = ('reg', D, self.V * self.V)
        if 14 in S:
            out['query'] = ('class', x[:, 14], self.V)
            ret = torch.full((len(x),), self.V, dtype=torch.long, device=x.device)
            for m in ms:
                ret = torch.where(x[:, 2 * m] == x[:, 14], x[:, 2 * m + 1], ret)
            out['ret'] = ('class', ret, self.V + 1)
        return out

    def support(self, name, S):
        pairs = [t for m in self._pairs(S) for t in (2 * m, 2 * m + 1)]
        return dict(dict=pairs, query=[14], ret=pairs + [14])[name]

    def resample_same(self, x, S, g):
        """Shuffle the complete pairs inside S among their slots: same dictionary, different arrangement."""
        ms = self._pairs(S)
        if len(ms) < 2:
            return None
        x2 = x.clone()
        B = len(x)
        perm = torch.argsort(torch.rand(B, len(ms), generator=g, device=x.device), 1)
        slots = torch.tensor(ms, device=x.device)
        src = slots[perm]
        r = torch.arange(B, device=x.device)[:, None]
        x2[r, 2 * slots[None, :]] = x[r, 2 * src]
        x2[r, 2 * slots[None, :] + 1] = x[r, 2 * src + 1]
        return x2


class MultiMatch(Process):
    """Induction with accidental repeats. x_0..x_13 iid uniform over V; a planted i ~ U{0..12}; the query x_14 = x_i;
    leaf 15 is a blank (token V); y = x_{i+1} with target noise eps. The posterior over i is uniform on the matches
    M = {j <= 12 : x_j = x_14}, so p*(y | x) = (1 - eps) |M|^-1 sum_{j in M} [x_{j+1} = y] + eps / V.
    strata = min(|M|, 3)."""
    name = 'multimatch'
    primary = ('dict', 'ret')

    def __init__(self, V=8, eps=0.05, pseed=0):
        super().__init__(V, eps, pseed)

    V_in = property(lambda self: self.V + 1)

    def sample_xy(self, B, g):
        x = self._uniform(B, g)
        x[:, 15] = self.V
        i = torch.randint(0, 13, (B,), generator=g, device=g.device)
        r = torch.arange(B, device=g.device)
        x[:, 14] = x[r, i]
        return x, _noisy(x[r, i + 1], self.eps, self.O, g)

    def _match(self, x):
        return x[:, :13] == x[:, 14:15]

    def bayes(self, x):
        m = self._match(x).double()
        q = torch.zeros(len(x), self.V, dtype=torch.float64, device=x.device)
        q.scatter_add_(1, x[:, 1:14].clamp_max(self.V - 1), m)
        return _mix(q / q.sum(1, keepdim=True).clamp_min(1e-300), self.eps)

    def valid(self, x):
        return self._match(x).any(1) & (x[:, :15] < self.V).all(1) & (x[:, 15] == self.V)

    def strata(self, x):
        return self._match(x).sum(1).clamp_max(3)

    def support(self, name, S):
        pos = sorted({t for j in range(13) if j in S and j + 1 in S for t in (j, j + 1)})
        return dict(dict=pos, query=[14], ret=pos + [14])[name]

    def summary(self, x, S):
        out = {}
        js = [j for j in range(13) if j in S and j + 1 in S]
        if js:
            D = torch.zeros(len(x), self.V * self.V, dtype=torch.float64, device=x.device)
            for j in js:
                D[torch.arange(len(x)), x[:, j] * self.V + x[:, j + 1]] += 1.0
            out['dict'] = ('reg', D, self.V * self.V)
        if 14 in S:
            out['query'] = ('class', x[:, 14], self.V)
            if js:
                R = torch.zeros(len(x), self.V, dtype=torch.float64, device=x.device)
                for j in js:
                    R[torch.arange(len(x)), x[:, j + 1]] += (x[:, j] == x[:, 14]).double()
                out['ret'] = ('reg', R, self.V)
        return out


KINDS = dict(fixed=FixedSource, two=TwoSource, mbigram=MaskedBigram, trigger=Trigger, parity=Parity,
             retrieval=Retrieval, multimatch=MultiMatch)


def make(spec):
    """spec: {'kind': name, **params} (params include pseed)."""
    spec = dict(spec)
    return KINDS[spec.pop('kind')](**spec)


def key(spec):
    return '_'.join(f'{k}{v}' for k, v in sorted(spec.items()) if k != 'kind').join((spec['kind'] + '_', ''))
