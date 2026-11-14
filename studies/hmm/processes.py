"""Hidden Markov processes with exact Bayesian beliefs (Shai et al. 2024, "Transformers represent belief state geometry
in their residual stream"), for the belief-geometry experiments.

A process is a dict(name, T, pi, V, S, factors):
  T (V, S, S) float64, T[x, s, s'] = P(emit x and move to s' | state s);  sum_x T[x] is row-stochastic.
  pi (S,) the start distribution: the stationary distribution of sum_x T[x], except for `coins3`.
  factors: (S_A, S_B) for a process on a product state space (state index s = a * S_B + b, token x = x_a * V_B + x_b).

Belief. eta_t = P(state at time t | x_<t), the state that emits x_t; eta_0 = pi and
    eta_{t+1} = eta_t T[x_t] / (eta_t T[x_t] 1),     P(x_t | x_<t) = eta_t T[x_t] 1.
The unnormalized recursion u_{t+1} = u_t T[x_t] is bilinear in (state, one-hot token), the form of a ChainTN update.

Processes
  mess3(x, a)  3 states, 3 tokens: move s -> s' with probability 1 - 2x if s' = s else x, then emit e with probability a
               if e = s' else (1 - a) / 2. (x, a) = (0.05, 0.85) is the process of Shai et al. (Fig. 5).
  rrxor        5 states S, A0, A1, B0, B1, 2 tokens: two fair random bits, then their XOR, repeated.
  z1r          3 states, 2 tokens: emit 0, emit 1, emit a fair random bit, repeated.
  coins3       3 states that never change, 2 tokens: state k emits 1 with probability (0.2, 0.5, 0.8)[k]; pi uniform.
               Non-ergodic: the belief is the posterior over which coin generated the sequence.
  factored     the product of independent mess3(0.05, 0.85) (factor A) and mess3(0.15, 0.6) (factor B): 9 states, 9 tokens.
  coupled      as factored, but factor B's kernel is mixed with a copy of factor A's current state a:
                   T[(xa, xb), (a, b), (a', b')] = T_A[xa, a, a'] * ((1 - eps) T_B[xb, b, b'] + eps 1[xb = a] 1[b' = a]),
               eps = 0.3. With probability eps, B emits token a and jumps to state a, so B's tokens are evidence about
               A's state and the joint belief is not the product of its marginals."""
import torch

F64 = torch.float64


def mess3(x=0.05, a=0.85):
    b, y = (1 - a) / 2, 1 - 2 * x
    T = torch.empty(3, 3, 3, dtype=F64)
    for e in range(3):
        for s in range(3):
            for s2 in range(3):
                T[e, s, s2] = (y if s2 == s else x) * (a if e == s2 else b)
    return T


def rrxor():
    S, A0, A1, B0, B1 = range(5)
    T = torch.zeros(2, 5, 5, dtype=F64)
    T[0, S, A0] = T[1, S, A1] = 0.5
    T[0, A0, B0] = T[1, A0, B1] = 0.5   # second bit; B_k: the XOR will be k
    T[0, A1, B1] = T[1, A1, B0] = 0.5
    T[0, B0, S] = T[1, B1, S] = 1.0
    return T


def z1r():
    T = torch.zeros(2, 3, 3, dtype=F64)
    T[0, 0, 1] = T[1, 1, 2] = 1.0
    T[0, 2, 0] = T[1, 2, 0] = 0.5
    return T


def coins(p=(0.2, 0.5, 0.8)):
    T = torch.zeros(2, len(p), len(p), dtype=F64)
    for k, pk in enumerate(p):
        T[1, k, k], T[0, k, k] = pk, 1 - pk
    return T


def product(TA, TB):
    """Independent factors: T[(xa, xb), (a, b), (a', b')] = TA[xa, a, a'] TB[xb, b, b']."""
    (VA, SA, _), (VB, SB, _) = TA.shape, TB.shape
    return torch.einsum('xac,ybd->xyabcd', TA, TB).reshape(VA * VB, SA * SB, SA * SB)


def coupled(TA, TB, eps):
    (VA, SA, _), (VB, SB, _) = TA.shape, TB.shape
    assert VB == SB == SA, 'the copy kernel needs token, state and source-state sets of the same size'
    K = torch.zeros(SA, VB, SB, SB, dtype=F64)  # K[a, xb, b, b'] = 1[xb = a] 1[b' = a]
    for a in range(SA):
        K[a, a, :, a] = 1.0
    TBa = (1 - eps) * TB[None] + eps * K  # (a, xb, b, b')
    return torch.einsum('xac,aybd->xyabcd', TA, TBa).reshape(VA * VB, SA * SB, SA * SB)


def stationary(T):
    """The left eigenvector of sum_x T[x] with eigenvalue 1, normalized (unique for the irreducible processes here)."""
    w, V = torch.linalg.eig(T.sum(0).T)
    v = V[:, (w - 1).abs().argmin()].real
    return v / v.sum()


NAMES = ('mess3', 'rrxor', 'z1r', 'coins3', 'factored', 'coupled')


def get(name):
    A, B = mess3(0.05, 0.85), mess3(0.15, 0.6)
    T = dict(mess3=lambda: A, rrxor=rrxor, z1r=z1r, coins3=coins, factored=lambda: product(A, B),
             coupled=lambda: coupled(A, B, 0.3))[name]()
    pi = torch.full((3,), 1 / 3, dtype=F64) if name == 'coins3' else stationary(T)
    return dict(name=name, T=T, pi=pi, V=T.shape[0], S=T.shape[1], factors=(3, 3) if name in ('factored', 'coupled') else None)


def sample(proc, B, n, g=None, device='cpu'):
    """x (B, n) long: start state ~ pi, then (token, next state) ~ T[:, s, :] at every step."""
    V, S = proc['V'], proc['S']
    P = proc['T'].permute(1, 0, 2).reshape(S, V * S).float().to(device)  # row s: joint over (x, s'), index x * S + s'
    s = torch.multinomial(proc['pi'].float().to(device), B, replacement=True, generator=g)
    x = torch.empty(B, n, dtype=torch.long, device=device)
    for t in range(n):
        j = torch.multinomial(P[s], 1, generator=g)[:, 0]
        x[:, t], s = j // S, j % S
    return x


def beliefs(proc, x):
    """x (B, n) long -> eta (B, n, S) with eta[:, t] = P(state emitting x_t | x_<t), and p (B, n, V) with
    p[:, t] = P(x_t = . | x_<t). Float64, on x's device."""
    T = proc['T'].to(x.device)
    eta = proc['pi'].to(x.device).expand(x.shape[0], -1)
    etas, ps = [], []
    for t in range(x.shape[1]):
        etas.append(eta)
        ps.append(torch.einsum('bs,xsj->bx', eta, T))
        joint = torch.einsum('bs,bsj->bj', eta, T[x[:, t]])
        eta = joint / joint.sum(1, keepdim=True)
    return torch.stack(etas, 1), torch.stack(ps, 1)


def bayes_loss(proc, x):
    """(n,) mean over sequences of -log P(x_t | x_<t), exact."""
    p = beliefs(proc, x)[1]
    return -p.gather(2, x[..., None])[..., 0].log().mean(0)


def context_loss(proc, x, m):
    """(n,) mean over sequences of -log q_m(x_t | the last min(t, m) tokens), where q_m is the process's exact conditional
    given only those tokens: the optimal predictor of Markov order m (an (m + 1)-gram model). Every process here has the
    state marginal pi at every time (a stationary start; coins3's states never change), so the window is filtered from
    pi. Its excess over `bayes_loss` on the same sequences is what a model that uses only the last m tokens leaves on
    the table."""
    out = []
    for t in range(x.shape[1]):
        p = beliefs(proc, x[:, max(0, t - m):t + 1])[1][:, -1]
        out.append(-p.gather(1, x[:, t:t + 1])[:, 0].log().mean())
    return torch.stack(out)


def marginals(proc, eta):
    """Factor marginals of a belief on a product state space: (..., S_A), (..., S_B)."""
    SA, SB = proc['factors']
    j = eta.reshape(*eta.shape[:-1], SA, SB)
    return j.sum(-1), j.sum(-2)
