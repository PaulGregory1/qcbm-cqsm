"""Sequence 4: input-output linear representations, the common form of every truth and model.

The observed value x_t (A symbols) is fed back; an internal row vector a (dimension d) carries the hidden part:

    P(x_1 .. x_T | x_0, a_0) = a_0 M[x_0, x_1] M[x_1, x_2] ... M[x_{T-1}, x_T] f

  input-output HMM, T[x, s, x', s'] = P(x', s' | x, s):     M[x, x'] = T[x, :, x', :], f = 1         (d = S)
  conditional QSM (archb.cqsm), Kraus K[x, x', c] (D x D): M[x, x'] = sum_c (K (x) conj K)^T, f = vec(I) (d = D^2)
  Markov kernel P on x (Architecture B, tabular counts):   M[x, x'] = P[x, x'] (1 x 1)                (d = 1)
  order-2 Markov chain: internal state = the previous symbol (d = A).

Joint stationary law u (A, d): u_{x'} = sum_x u_x M[x, x'], sum_x u_x f = 1; every path starts from it
(P(x_0) = u_{x_0} f, a_0 = u_{x_0} / P(x_0)). It is computed as the Abel limit of the chain started from the uniform
law on x with an uninformative internal state, eps sum_k (1 - eps)^k u0 G^k (one linear solve, eps = 1e-10). This equals
the fixed point when it is unique and stays defined when it is not (a memory that no gate touches).

JAX functions (stationary, path_loglik, logprobs) serve training and evaluation; class IOLin adds the exact metrics
(averaged one-step kernel, acf, kurtosis, law of the h-step sum and its VaR / ES), filtering and sampling.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

EPS = 1e-10
FLOOR = 1e-300


def big(M):
    """(A, A, d, d) -> (A d, A d) transition matrix of the joint chain, G[(x, i), (x', j)] = M[x, x', i, j]."""
    A, _, d, _ = M.shape
    return M.transpose(0, 2, 1, 3).reshape(A * d, A * d)


def stationary(M, f, eps=EPS):
    A, _, d, _ = M.shape
    G = big(M)
    u0 = jnp.tile(f / (f @ f), A).astype(G.dtype) / A
    u = jnp.linalg.solve((jnp.eye(A * d, dtype=G.dtype) - (1 - eps) * G).T, eps * u0).reshape(A, d)
    return u / jnp.real(jnp.sum(u @ f))


def path_loglik(M, f, u, X):
    """Mean log-likelihood per symbol of the paths X (P, T + 1), first symbol included, from the joint law u."""
    a = u[X[:, 0]]
    p = jnp.maximum(jnp.real(a @ f), FLOOR)
    tot, a = jnp.sum(jnp.log(p)), a / p[:, None]

    def step(a, xy):
        a = jnp.einsum("pi,pij->pj", a, M[xy[0], xy[1]])
        p = jnp.maximum(jnp.real(a @ f), FLOOR)
        return a / p[:, None], jnp.sum(jnp.log(p))

    _, lp = jax.lax.scan(step, a, (X[:, :-1].T, X[:, 1:].T))
    return (tot + jnp.sum(lp)) / X.size


def loglik(M, f, X):
    return path_loglik(M, f, stationary(M, f), X)


@jax.jit
def _scan_path(M, f, u, xs):
    a = u[xs[0]]
    p = jnp.real(a @ f)

    def step(a, xy):
        a = a @ M[xy[0], xy[1]]
        p = jnp.real(a @ f)
        a = a / p
        return a, (jnp.log(jnp.maximum(p, FLOOR)), a)

    _, (lp, states) = jax.lax.scan(step, a / p, (xs[:-1], xs[1:]))
    return jnp.concatenate([jnp.log(p)[None], lp]), jnp.concatenate([(a / p)[None], states])


def logprobs(M, f, u, xs):
    """log P(x_t | x_<t) for every t of one path (t = 0: log P(x_0) under u)."""
    return _scan_path(M, f, u, xs)[0]


class IOLin:
    """Exact metrics of one representation (numpy side). M (A, A, d, d), f (d,)."""

    def __init__(self, M, f, u=None):
        self.M, self.f = np.asarray(M), np.asarray(f)
        self.A, _, self.d, _ = self.M.shape
        self.u = np.asarray(stationary(jnp.asarray(self.M), jnp.asarray(self.f))) if u is None else np.asarray(u)

    @classmethod
    def from_iohmm(cls, T):
        T = np.asarray(T, float)
        return cls(T.transpose(0, 2, 1, 3), np.ones(T.shape[1]))

    @classmethod
    def from_kernel(cls, P):
        return cls(np.asarray(P, float)[:, :, None, None], np.ones(1))

    @classmethod
    def from_markov2(cls, P2):
        """P2[(w, x), x'] = P(x_{t+1} = x' | x_{t-1} = w, x_t = x); internal state = the previous symbol."""
        A = P2.shape[1]
        P2 = np.asarray(P2, float).reshape(A, A, A)  # [w, x, x']
        M = np.zeros((A, A, A, A))
        for x in range(A):
            M[x, :, :, x] = P2[:, x, :].T  # M[x, x'][w, x] = P2[w, x, x']
        return cls(M, np.ones(A))

    def G(self):
        return np.asarray(big(jnp.asarray(self.M)))

    def marginal(self):
        return np.real(self.u @ self.f)

    def kernel(self):
        """One-step kernel averaged over the hidden part under the stationary law, P[x, x'] = P(x_{t+1} | x_t)."""
        J = np.real(np.einsum("xi,xyij,j->xy", self.u, self.M, self.f))
        return J / J.sum(1, keepdims=True)

    def logprobs(self, xs):
        return np.asarray(logprobs(jnp.asarray(self.M), jnp.asarray(self.f), jnp.asarray(self.u), jnp.asarray(xs)))

    def filtered(self, xs):
        """Normalized internal vector after each observation x_t (row t: given x_0 .. x_t)."""
        return np.asarray(_scan_path(jnp.asarray(self.M), jnp.asarray(self.f), jnp.asarray(self.u), jnp.asarray(xs))[1])

    def path_loglik(self, X):
        return float(path_loglik(jnp.asarray(self.M), jnp.asarray(self.f), jnp.asarray(self.u), jnp.asarray(X)))

    def acf(self, g, lags):
        """corr(g(x_t), g(x_{t+k})), k = 1..lags, exact under the stationary joint law."""
        G, A, d = self.G(), self.A, self.d
        gw = np.repeat(np.asarray(g, float), d)
        U, F = self.u.ravel(), np.tile(self.f, A)
        pi = self.marginal()
        mu, m2 = pi @ g, pi @ np.asarray(g) ** 2
        r, h, out = U * gw, gw * F, []
        for _ in range(lags):
            r = r @ G
            out.append(np.real(r @ h))
        return (np.asarray(out) - mu**2) / (m2 - mu**2)

    def sum_law(self, h):
        """Law of the bin-index sum x_{t+1} + .. + x_{t+h} from the stationary joint law (support 0 .. (A-1) h)."""
        A, d, G = self.A, self.d, self.G()
        L = (A - 1) * h + 1
        W = np.zeros((L, A, d), dtype=self.u.dtype)
        W[np.arange(A), np.arange(A)] = self.u
        for _ in range(h - 1):
            V = (W.reshape(L, A * d) @ G).reshape(L, A, d)
            W = np.zeros_like(W)
            for y in range(A):
                W[y:, y] = V[:L - y, y]
        return np.clip(np.real(W @ self.f).sum(1), 0, None)

    def var_es(self, z, h, alphas=(0.01, 0.05)):
        """VaR and expected shortfall (positive = loss) of the h-step sum of values; z = the A bin centres (equal
        width). Exact discrete law: VaR = - alpha-quantile, ES = - E[S | lower alpha tail] with the atom split."""
        p = self.sum_law(h)
        w = z[1] - z[0]
        s = h * z[0] + w * np.arange(len(p))
        F = np.cumsum(p)
        out = {}
        for a in alphas:
            k = int(np.searchsorted(F, a))
            below = F[k - 1] if k else 0.0
            es = (np.sum(p[:k] * s[:k]) + (a - below) * s[k]) / a
            out[f"var{round(100 * a)}"], out[f"es{round(100 * a)}"] = float(-s[k]), float(-es)
        return out

    def sample(self, n, seed):
        """One path of n symbols from the stationary joint law (filtering sampler, exact in law)."""
        rng = np.random.default_rng(seed)
        pi = self.marginal()
        x = int(rng.choice(self.A, p=pi / pi.sum()))
        a, xs = self.u[x] / pi[x], np.empty(n, dtype=np.int64)
        xs[0] = x
        for t in range(1, n):
            B = np.einsum("i,yij->yj", a, self.M[x])
            p = np.clip(np.real(B @ self.f), 0, None)
            x = int(rng.choice(self.A, p=p / p.sum()))
            a, xs[t] = B[x] / p[x], x
        return xs
