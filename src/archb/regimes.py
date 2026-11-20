"""Sequence 4: regime-switching AR targets and the classical models.

Grid: 5 bits, 32 equal bins of [-4, 4], edge bins absorb the tails; circuits see Gray labels (archb.targets).
Truths are discrete processes with a hidden state, defined like P* in sequences 1-2: given the bin centre z of x_t
and the new hidden state h', x_{t+1} is the binned N(c_s z, sigma_s^2) of the regime s = s(h'), so every likelihood
is exact. Because the next value depends on the bin of x_t only (its centre), not on a position inside the bin, the
binned process is itself the 2-state (msar) or 2N-state (smsar) input-output HMM below: the position in the bin is not
an extra hidden variable, and the regime filter on the bins is the exact filter of the process the models see. Both are input-output HMMs T[x, h, x', h'] = P(x', h' | x, h) = Q[h, h'] K_{s(h')}[x, x'] (archb.iolin):

  msar      Markov-switching AR(1) (Hamilton): s in {calm, turbulent}, kept with prob. 0.98, c = 0.95 / 0.7,
            sigma = 0.3 / 1.0. h = s.
  smsar(N)  semi-Markov switching: same AR, regimes alternate, each lasts a duration uniform on {1..N};
            h = (s, j), j = steps left after the current one, index s N + j (2N states).

Data: 2,000 paths, BURN_IN steps discarded, 50 values kept (98,000 transitions); a held-out path of 200,000 steps
with its true regimes.

Classical models:
  family fits  the true family (MS-AR: c, sigma, stay prob. per regime; semi-Markov MS-AR: c, sigma and a duration
               law on {1..N} per regime, N known) by maximum likelihood on the paths (L-BFGS, JAX gradients).
  IOHMM(S)     input-output HMM by EM (batched forward-backward): 'full' P(x', s' | x, s) nonparametric,
               'fact' P(s' | s, x) P(x' | x, s') (the structure of the truths), 'tied' P(x', s' | s) (an HMM on x
               alone). Dirichlet pseudo-counts (MAP-EM) so that no held-out transition gets probability 0.
  Markov(k)    order-k chain on x from add-1/2 counts within the paths.
"""

import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jnorm
from scipy.optimize import minimize

from archb import iolin
from archb.targets import centres, edges, gray_labels

jax.config.update("jax_enable_x64", True)

N_BITS, A = 5, 32
LO, HI = -4.0, 4.0
E, Z = edges(LO, HI, A), centres(LO, HI, A)
LOB = gray_labels(N_BITS, 1)  # label of each bin
N_PATHS, N_KEEP, BURN_IN, N_TEST = 2000, 50, 200, 200000
DATA_DIR = Path(__file__).resolve().parents[2] / "runs" / "seq4" / "data"

MSAR = {"c": (0.95, 0.7), "sig": (0.3, 1.0), "stay": (0.98, 0.98)}


# ------------------------------------------------------------------------------- truths as input-output HMMs
def ar_kernels(c, sig):
    """(2, A, A): K_s[x, x'] = P(bin x' | centre of x) under N(c_s z, sigma_s^2), edge bins absorb the tails."""
    c, sig = jnp.asarray(c, jnp.float64), jnp.asarray(sig, jnp.float64)
    u = jnp.asarray(np.concatenate([[-1e4], E[1:-1], [1e4]]))  # tails absorbed; finite so that gradients exist
    cdf = jnorm.cdf((u[None, None, :] - c[:, None, None] * jnp.asarray(Z)[None, :, None]) / sig[:, None, None])
    return jnp.diff(cdf, axis=2)


def msar_T(c, sig, stay):
    stay = jnp.asarray(stay, jnp.float64)
    Q = jnp.stack([jnp.stack([stay[0], 1 - stay[0]]), jnp.stack([1 - stay[1], stay[1]])])
    return jnp.einsum("hk,kxy->xhyk", Q, ar_kernels(c, sig))


def smsar_T(c, sig, dur):
    """dur (2, N): duration law of each regime on {1..N}; entering regime s draws j' = duration - 1."""
    dur = jnp.asarray(dur, jnp.float64)
    N = dur.shape[1]
    Q = jnp.zeros((2 * N, 2 * N))
    for s in range(2):
        for j in range(1, N):
            Q = Q.at[s * N + j, s * N + j - 1].set(1.0)
        Q = Q.at[s * N, (1 - s) * N:(2 - s) * N].set(dur[1 - s])
    K = jnp.repeat(ar_kernels(c, sig), N, axis=0)  # kernel of each hidden state h' = (s, j)
    return jnp.einsum("hk,kxy->xhyk", Q, K)


def regime_of(kind, N=None):
    """Regime s of each hidden state."""
    return np.arange(2) if kind == "msar" else np.repeat(np.arange(2), N)


def truth_T(kind, N=None):
    if kind == "msar":
        return np.asarray(msar_T(**MSAR))
    return np.asarray(smsar_T(MSAR["c"], MSAR["sig"], np.full((2, N), 1.0 / N)))


# ------------------------------------------------------------------------------------------------ data
def _cum(T):
    A_, S = T.shape[0], T.shape[1]
    c = np.cumsum(np.asarray(T).reshape(A_, S, A_ * S), axis=2)
    c[..., -1] = 1.0
    return c


def simulate(T, n_paths, n_keep, burn_in, seed):
    """Paths (n_paths, n_keep) of x and their hidden states, started from the stationary joint law, burn-in dropped."""
    T = np.asarray(T, float)
    A_, S = T.shape[0], T.shape[1]
    rng = np.random.default_rng(seed)
    u = iolin.IOLin.from_iohmm(T).u.ravel()
    k = rng.choice(A_ * S, size=n_paths, p=np.clip(u, 0, None) / np.clip(u, 0, None).sum())
    x, h = k // S, k % S
    cum = _cum(T)
    X, H = np.empty((n_paths, n_keep), np.int64), np.empty((n_paths, n_keep), np.int64)
    for t in range(burn_in + n_keep):
        if t >= burn_in:
            X[:, t - burn_in], H[:, t - burn_in] = x, h
        j = (cum[x, h] < rng.random(n_paths)[:, None]).sum(1)
        x, h = j // S, j % S
    return X, H


def make_data(kind, N, seed):
    T = truth_T(kind, N)
    X, H = simulate(T, N_PATHS, N_KEEP, BURN_IN, seed)
    xt, ht = simulate(T, 1, N_TEST, BURN_IN, seed + 1)
    return {"X": X, "H": H, "test": xt[0], "test_h": ht[0]}


def load_data(name, kind, N, seed):
    p = DATA_DIR / f"{name}.npz"
    if not p.exists():  # written atomically: several workers may create the same (deterministic) file
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **make_data(kind, N, seed))
        os.replace(tmp, p)
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------- true families by MLE
def _family_T(kind, z, N):
    c, sig = z[0:2], jnp.exp(z[2:4])
    if kind == "msar":
        return msar_T(c, sig, jax.nn.sigmoid(z[4:6]))
    return smsar_T(c, sig, jax.nn.softmax(z[4:].reshape(2, N), axis=1))


def family_params(kind, z, N=None):
    z = np.asarray(z)
    out = {"c": z[0:2].tolist(), "sig": np.exp(z[2:4]).tolist()}
    if kind == "msar":
        out["stay"] = (1 / (1 + np.exp(-z[4:6]))).tolist()
    else:
        out["dur"] = np.asarray(jax.nn.softmax(jnp.asarray(z[4:]).reshape(2, N), axis=1)).tolist()
    return out


def fit_family(kind, X, N=None, starts=4, seed=0, maxiter=500):
    """Maximum likelihood of the MS-AR (kind 'msar') or semi-Markov MS-AR ('smsar', N known) family on the paths,
    from `starts` generic starting points (not the truth). Returns (T, params, train mean log-likelihood, info)."""
    Xj = jnp.asarray(X)

    def nll(z):
        T = _family_T(kind, z, N)
        S = T.shape[1]
        return -iolin.loglik(T.transpose(0, 2, 1, 3), jnp.ones(S), Xj)

    vg = jax.jit(jax.value_and_grad(nll))
    rng = np.random.default_rng(seed)
    best = None
    for k in range(starts):
        z0 = np.concatenate([[0.8, 0.4], np.log([0.5, 1.5]) + 0.2 * rng.normal(size=2)])
        z0[:2] += 0.1 * rng.normal(size=2)
        z0 = np.concatenate([z0, [3.0, 3.0] + rng.normal(size=2)] if kind == "msar" else [z0, rng.normal(size=2 * N)])
        out = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(jnp.asarray(z))), z0, jac=True,
                       method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9})
        if best is None or out.fun < best.fun:
            best = out
    T = np.asarray(_family_T(kind, jnp.asarray(best.x), N))
    return T, family_params(kind, best.x, N), -float(best.fun), {"n_params": len(best.x), "message": str(best.message)}


# ------------------------------------------------------------------------------- input-output HMM by EM
def n_params_iohmm(kind, S):
    return {"full": A * S * (A * S - 1), "fact": A * S * (S - 1) + A * S * (A - 1), "tied": S * (A * S - 1)}[kind]


@partial(jax.jit, static_argnames=("kind",))
def _em_step(T, q, X, kind, prior):
    """One EM step on all paths (initial hidden law q shared by the paths; x_0 not modelled). Returns (T_new, q_new,
    mean log-likelihood per transition under (T, q))."""
    P, S = X.shape[0], T.shape[1]
    xs, ys = X[:, :-1].T, X[:, 1:].T
    a0 = jnp.broadcast_to(q, (P, S))

    def fwd(a, xy):
        a = jnp.einsum("ps,psk->pk", a, T[xy[0], :, xy[1], :])
        c = a.sum(1)
        return a / c[:, None], (a / c[:, None], c)

    _, (alph, c) = jax.lax.scan(fwd, a0, (xs, ys))
    aprev = jnp.concatenate([a0[None], alph[:-1]])

    def bwd(carry, inp):
        b, N = carry
        x, y, ct, ap = inp
        Txy = T[x, :, y, :]
        N = N.at[x, :, y, :].add(ap[:, :, None] * Txy * b[:, None, :] / ct[:, None, None])
        return (jnp.einsum("psk,pk->ps", Txy, b) / ct[:, None], N), None

    (b0, N), _ = jax.lax.scan(bwd, (jnp.ones((P, S)), jnp.zeros_like(T)), (xs, ys, c, aprev), reverse=True)
    g0 = (a0 * b0).sum(0)
    if kind == "full":
        Nn = N + prior
        Tn = Nn / Nn.sum((2, 3), keepdims=True)
    elif kind == "fact":
        R = N.sum(2) + prior  # [x, s, s']
        Em = N.sum(1).transpose(0, 2, 1) + prior  # [x, s', x']
        R, Em = R / R.sum(2, keepdims=True), Em / Em.sum(2, keepdims=True)
        Tn = jnp.einsum("xsk,xky->xsyk", R, Em)
    else:  # tied
        Nt = N.sum(0) + prior
        Tn = jnp.broadcast_to(Nt / Nt.sum((1, 2), keepdims=True), T.shape)
    return Tn, g0 / g0.sum(), jnp.mean(jnp.log(c))


def _random_T(kind, S, rng):
    if kind == "full":
        return rng.dirichlet(np.ones(A * S), size=(A, S)).reshape(A, S, A, S)
    if kind == "fact":
        R = rng.dirichlet(np.ones(S), size=(A, S))
        Em = rng.dirichlet(np.ones(A), size=(A, S))
        return np.einsum("xsk,xky->xsyk", R, Em)
    H = rng.dirichlet(np.ones(A * S), size=S).reshape(S, A, S)
    return np.broadcast_to(H, (A, S, A, S)).copy()


PRIOR = {"full": lambda S: 0.5 / S, "fact": lambda S: 0.5, "tied": lambda S: 1e-3}


def iohmm_em(X, S, kind, seed, max_iter=1000, tol=1e-9):
    """EM from a random Dirichlet start; stops when the mean log-likelihood per transition improves by < tol."""
    rng = np.random.default_rng(seed)
    T, q, Xj = jnp.asarray(_random_T(kind, S, rng)), jnp.full(S, 1.0 / S), jnp.asarray(X)
    prior, curve = PRIOR[kind](S), []
    for it in range(max_iter):
        Tn, qn, ll = _em_step(T, q, Xj, kind, prior)
        curve.append(float(ll))
        if it and curve[-1] - curve[-2] < tol:
            break
        T, q = Tn, qn
    return np.asarray(T), np.asarray(curve)


# ------------------------------------------------------------------------------- Markov chains on paths
def markov_counts(X, k, alpha=0.5):
    """Order-k transition table from the paths, Pc[w, x'] with w the last k symbols in base A (oldest first)."""
    X = np.asarray(X)
    ctx = np.zeros(X[:, k:].shape, dtype=np.int64)
    for j in range(k):
        ctx = ctx * A + X[:, j:X.shape[1] - k + j]
    C = np.zeros((A**k, A)) + alpha
    np.add.at(C, (ctx.ravel(), X[:, k:].ravel()), 1.0)
    return C / C.sum(1, keepdims=True)


def markov_logprobs(tables, xs):
    """log P(x_t | x_<t) on one path; tables[j] is the order-j table (j = 0: marginal, shape (1, A)); the first k
    symbols use the lower orders."""
    xs, k = np.asarray(xs), len(tables) - 1
    out = np.empty(len(xs))
    for t in range(len(xs)):
        j = min(t, k)
        w = 0
        for y in xs[t - j:t]:
            w = w * A + int(y)
        out[t] = np.log(tables[j][w, xs[t]])
        if t >= k:
            break
    ctx = np.zeros(len(xs) - k, dtype=np.int64)
    for j in range(k):
        ctx = ctx * A + xs[j:len(xs) - k + j]
    out[k:] = np.log(tables[k][ctx, xs[k:]])
    return out


def markov_tables(X, k):
    return [markov_counts(X, j) for j in range(k + 1)]


def markov_path_loglik(tables, X):
    return float(np.mean([markov_logprobs(tables, x) for x in np.asarray(X)]))


def n_params_markov(k):
    return A**k * (A - 1)


# ------------------------------------------------------------------------------- regime decoding
@jax.jit
def _smooth(T, v, xs):
    def fwd(a, xy):
        a = a @ T[xy[0], :, xy[1], :]
        return a / a.sum(), a / a.sum()

    _, al = jax.lax.scan(fwd, v, (xs[:-1], xs[1:]))
    al = jnp.concatenate([v[None], al])

    def bwd(b, xy):
        b = T[xy[0], :, xy[1], :] @ b
        return b / b.sum(), b / b.sum()

    _, be = jax.lax.scan(bwd, jnp.ones_like(v), (xs[:-1], xs[1:]), reverse=True)
    be = jnp.concatenate([be, jnp.ones_like(v)[None]])
    g = al * be
    return g / g.sum(1, keepdims=True)


def decode_regimes(kind, N, xs):
    """MAP regime at every step of a path under the TRUE process (forward-backward smoothing of the hidden state)."""
    T = truth_T(kind, N)
    u = iolin.IOLin.from_iohmm(T).u
    g = np.asarray(_smooth(jnp.asarray(T), jnp.asarray(u[xs[0]] / u[xs[0]].sum()), jnp.asarray(xs)))
    s = regime_of(kind, N)
    return (g[:, s == 1].sum(1) > 0.5).astype(int)


def run_lengths(s):
    """Lengths of the complete runs of a 0/1 sequence (first and last runs dropped)."""
    cut = np.flatnonzero(np.diff(s)) + 1
    return np.diff(cut)
