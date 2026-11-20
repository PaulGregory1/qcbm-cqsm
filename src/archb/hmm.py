"""Sequence 3: hidden-state processes, classical sequence models and exact metrics.

Every model is a linear representation of a stationary process over symbols 0..A-1: a row vector v, one
matrix E[x] per symbol and a column f, with

    P(x_1 .. x_L) = v E[x_1] ... E[x_L] f.

  HMM (edge-emitting): T[s, x, s'] = P(x_t = x, s_{t+1} = s' | s_t = s); v = stationary law of sum_x T[:, x, :],
      E[x] = T[:, x, :], f = 1. A state-emitting HMM (emission B[s, x], transition A[s, s']) is the special case
      T = B[:, :, None] A[:, None, :], so the edge-emitting form with S states contains it.
  QSM: v = vec(rho_ss), E[x] = (K_x (x) conj K_x)^T, f = vec(I)            (archb.qseq.linear)
  Markov chain of order k: class Markov below (state = last k symbols), same interface.

Truth processes (exact word probabilities):
  renewal(N)   binary; the number of 0s between consecutive 1s is uniform on {0..N-1}. Count-down HMM with N states:
               state j = 0s left before the next 1; j > 0 emits 0 and goes to j-1; j = 0 emits 1 and draws j' uniform.
  regime()     hidden regime low/high, kept with probability `stay` per step; x_t = sigma_s xi_t, xi ~ N(0, 1),
               binned on 8 equal bins of [-3 sigma_high, 3 sigma_high] (edge bins absorb the tails).
Classical fits: baum_welch (edge-emitting HMM, EM, JAX scan for forward-backward), markov_fit (order k, add-alpha
counts).

Sequence 5 additions: flw_qubit (a qubit process with no finite HMM), sample_linear, hankel_rank, renewal-type
(hazard) HMMs as starting points, and exact_fit (an HMM fitted to the exact law of the words of length L).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------------------- linear representations
class Linear:
    """P(x_1..x_L) = v E[x_1] ... E[x_L] f (dense matrices, real or complex)."""

    def __init__(self, v, E, f):
        self.v, self.E, self.f = np.asarray(v), [np.asarray(e) for e in E], np.asarray(f)
        self.A = len(self.E)

    def word_probs(self, L):
        """[P_1, ..., P_L], P_l of shape (A^l,), words in lexicographic order (x_1 most significant)."""
        R, out = self.v[None, :], []
        for _ in range(L):
            R = np.stack([R @ e for e in self.E], axis=1).reshape(-1, len(self.v))
            out.append(np.real(R @ self.f))
        return out

    def prob(self, word):
        r = self.v
        for x in word:
            r = r @ self.E[x]
        return float(np.real(r @ self.f))

    def acf(self, g, lags):
        """Autocorrelation of g(x_t) under the stationary law, lags 1..lags (exact)."""
        Eg = sum(gi * e for gi, e in zip(g, self.E))
        Eg2 = sum(gi**2 * e for gi, e in zip(g, self.E))
        Es = sum(self.E)
        mu, m2 = np.real(self.v @ Eg @ self.f), np.real(self.v @ Eg2 @ self.f)
        r, h, out = self.v @ Eg, Eg @ self.f, []
        for _ in range(lags):
            out.append(np.real(r @ h))
            r = r @ Es
        return (np.asarray(out) - mu**2) / (m2 - mu**2)


def hmm_linear(T):
    T = np.asarray(T, float)
    return Linear(stationary(T.sum(1)), [T[:, x, :] for x in range(T.shape[1])], np.ones(T.shape[0]))


def stationary(P):
    """Stationary row vector of a row-stochastic matrix (eigenvector of eigenvalue 1)."""
    w, V = np.linalg.eig(P.T)
    p = np.real(V[:, np.argmin(np.abs(w - 1))])
    return p / p.sum()


def sample(T, n, seed):
    """Sample x_1..x_n from an edge-emitting HMM started from its stationary law."""
    T = np.asarray(T, float)
    S, A, _ = T.shape
    rng = np.random.default_rng(seed)
    cum = np.cumsum(T.reshape(S, A * S), axis=1)
    cum[:, -1] = 1.0
    s = int(rng.choice(S, p=stationary(T.sum(1))))
    u, xs = rng.random(n), np.empty(n, dtype=np.int64)
    for t in range(n):
        j = int(np.searchsorted(cum[s], u[t], side="right"))
        xs[t], s = divmod(j, S)
    return xs


# ------------------------------------------------------------------------------- truth processes
def renewal(N):
    T = np.zeros((N, 2, N))
    T[0, 1, :] = 1.0 / N
    for j in range(1, N):
        T[j, 0, j - 1] = 1.0
    return T


REGIME = {"stay": 0.98, "sig": (0.5, 1.5), "n_bins": 8}


def regime_bins(sig_hi=REGIME["sig"][1], n=REGIME["n_bins"]):
    e = np.linspace(-3 * sig_hi, 3 * sig_hi, n + 1)
    return e, 0.5 * (e[:-1] + e[1:])


def regime(stay=REGIME["stay"], sig=REGIME["sig"], n_bins=REGIME["n_bins"]):
    e, _ = regime_bins(sig[1], n_bins)
    u = e.copy()
    u[0], u[-1] = -np.inf, np.inf
    B = np.array([np.diff(norm.cdf(u / s)) for s in sig])  # (2, n_bins)
    Atr = np.array([[stay, 1 - stay], [1 - stay, stay]])
    return B[:, :, None] * Atr[:, None, :]


# ------------------------------------------------------------------------------- Markov chains of order k
class Markov:
    """Order-k Markov chain on the symbols: Pc[w, x] = P(x | last k symbols = w), w in base A, oldest symbol most
    significant. Stationary block law pi_k; word probabilities are those of the stationary chain."""

    def __init__(self, Pc, k, A):
        self.Pc, self.k, self.A = np.asarray(Pc, float), k, A
        self.pi = self._stationary()

    def _shift(self):  # index of the block (w_2..w_k, x) for every (w, x)
        w = np.arange(self.A**self.k)
        return (w % self.A ** (self.k - 1))[:, None] * self.A + np.arange(self.A)[None, :]

    def _stationary(self, tol=1e-15, max_iter=200000):
        p, sh = np.full(self.A**self.k, self.A**-self.k), self._shift()
        for _ in range(max_iter):
            q = np.bincount(sh.ravel(), (p[:, None] * self.Pc).ravel(), minlength=len(p))
            if np.abs(q - p).sum() < tol:
                return q / q.sum()
            p = q
        return p / p.sum()

    def word_probs(self, L):
        A, k, out = self.A, self.k, []
        for l in range(1, L + 1):
            if l <= k:
                out.append(self.pi.reshape(A**l, A ** (k - l)).sum(1))
            else:
                prev = out[-1]
                out.append((prev[:, None] * self.Pc[np.arange(len(prev)) % A**k]).ravel())
        return out

    def prob(self, word):
        word = list(word)
        k, A = self.k, self.A
        if len(word) <= k:
            idx = 0
            for x in word:
                idx = idx * A + x
            return float(self.word_probs(len(word))[-1][idx])
        idx = 0
        for x in word[:k]:
            idx = idx * A + x
        p = self.pi[idx]
        for x in word[k:]:
            p *= self.Pc[idx, x]
            idx = (idx % A ** (k - 1)) * A + x
        return float(p)

    def acf(self, g, lags):
        g = np.asarray(g, float)
        sh = self._shift()
        last = np.arange(self.A**self.k) % self.A
        gw = g[last]
        mu, m2 = self.pi @ gw, self.pi @ gw**2
        h, out = gw.copy(), []
        for _ in range(lags):
            h = (self.Pc * h[sh]).sum(1)  # h <- M h, M the block transition matrix
            out.append(self.pi @ (gw * h))
        return (np.asarray(out) - mu**2) / (m2 - mu**2)


def markov_fit(xs, k, A, alpha=0.5):
    """Order-k chain from add-alpha smoothed transition counts of the training string."""
    xs = np.asarray(xs)
    ctx = np.zeros(len(xs) - k, dtype=np.int64)
    for j in range(k):
        ctx = ctx * A + xs[j:len(xs) - k + j]
    C = np.zeros((A**k, A)) + alpha
    np.add.at(C, (ctx, xs[k:]), 1.0)
    return Markov(C / C.sum(1, keepdims=True), k, A)


def markov_logprobs(m, xs):
    """log P(x_t | x_{<t}) for every t under the stationary chain (first k symbols from the block law pi_k)."""
    xs, k, A = np.asarray(xs), m.k, m.A
    head = [np.log(m.prob(xs[:j + 1]) / m.prob(xs[:j])) if j else np.log(m.prob(xs[:1])) for j in range(k)]
    ctx = np.zeros(len(xs) - k, dtype=np.int64)
    for j in range(k):
        ctx = ctx * A + xs[j:len(xs) - k + j]
    return np.concatenate([head, np.log(m.Pc[ctx, xs[k:]])])


# ------------------------------------------------------------------------------- Baum-Welch (edge-emitting HMM)
@partial(jax.jit, static_argnames=("A",))
def em_step(T, pi0, xs, A, prior=0.0):
    """One EM step on one string. Returns (T_new, pi0_new, mean log-likelihood per symbol under (T, pi0)).
    prior: pseudo-count added to every expected transition count (0 = plain maximum likelihood EM)."""

    def fwd(a, x):
        a = a @ T[:, x, :]
        c = a.sum()
        return a / c, (a / c, c)

    _, (alph, c) = jax.lax.scan(fwd, pi0, xs)

    def bwd(b, xc):
        x, ct = xc
        return T[:, x, :] @ b / ct, b

    b0, beta = jax.lax.scan(bwd, jnp.ones_like(pi0), (xs, c), reverse=True)
    aprev = jnp.concatenate([pi0[None, :], alph[:-1]], axis=0)
    xi = aprev[:, :, None] * T[:, xs, :].transpose(1, 0, 2) * beta[:, None, :] / c[:, None, None]
    N = jnp.einsum("tx,tij->ixj", jax.nn.one_hot(xs, A), xi) + prior
    g0 = pi0 * b0
    return N / N.sum((1, 2), keepdims=True), g0 / g0.sum(), jnp.mean(jnp.log(c))


def baum_welch(xs, S, A, seed, max_iter=3000, tol=1e-10, prior=1e-3, T0=None, max_seconds=None):
    """EM from a random Dirichlet start (or from T0); stops when the per-symbol log-likelihood improves by < tol, or
    after max_seconds (capped). Returns dict(T, loglik (per symbol, of T), curve, iters, capped)."""
    import time

    t0, capped = time.time(), False
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.dirichlet(np.ones(A * S), size=S).reshape(S, A, S) if T0 is None else T0)
    pi0, xs = jnp.full(S, 1.0 / S), jnp.asarray(xs)
    curve = []
    for it in range(max_iter):
        Tn, pin, ll = em_step(T, pi0, xs, A, prior)
        curve.append(float(ll))
        if it > 0 and curve[-1] - curve[-2] < tol:
            break
        if max_seconds is not None and time.time() - t0 > max_seconds:
            capped = True
            break
        T, pi0 = Tn, pin
    return {"T": np.asarray(T), "pi0": np.asarray(pi0), "loglik": curve[-1], "curve": np.asarray(curve),
            "iters": len(curve), "capped": capped}


def lump(T, groups):
    """Edge-emitting HMM on groups of states (state i -> groups[i]), each group weighted by the stationary law:
    T_l[a, x, b] = sum_{i in a} pi_i / pi_a sum_{j in b} T[i, x, j]."""
    groups = np.asarray(groups)
    pi, G = stationary(T.sum(1)), np.eye(groups.max() + 1)[groups]  # G[i, a]
    W = (pi[:, None] * G) / (pi @ G)[None, :]
    return np.einsum("ia,ixj,jb->axb", W, T, G)


def hmm_loglik(T, xs):
    """Mean log-likelihood per symbol, started from the stationary law."""
    return float(np.mean(hmm_logprobs(T, xs)))


@jax.jit
def _hmm_lp(T, v, xs):
    def fwd(a, x):
        a = a @ T[:, x, :]
        c = a.sum()
        return a / c, jnp.log(c)

    return jax.lax.scan(fwd, v, xs)[1]


def hmm_logprobs(T, xs):
    """log P(x_t | x_{<t}) for every t, filtered from the stationary law."""
    T = np.asarray(T, float)
    return np.asarray(_hmm_lp(jnp.asarray(T), jnp.asarray(stationary(T.sum(1))), jnp.asarray(xs)))


# ------------------------------------------------------------------------------- block likelihood (for gradients)
def blocks(xs, A, max_words=4096):
    """Cut xs into blocks of B symbols (A^B <= max_words): block ids (word index, first symbol most significant),
    the leftover tail and B. Filtering block by block with precomputed products of the B per-symbol operators gives
    the same likelihood as symbol by symbol with B times fewer sequential steps."""
    xs, B = np.asarray(xs), 1
    while A ** (B + 1) <= max_words:
        B += 1
    nb = len(xs) // B
    ids = np.zeros(nb, dtype=np.int64)
    for j in range(B):
        ids = ids * A + xs[j:nb * B:B]
    return {"ids": jnp.asarray(ids), "tail": tuple(int(x) for x in xs[nb * B:]), "B": B, "n": len(xs)}


def hmm_block_loglik(T, data):
    """Mean log-likelihood per symbol of an edge-emitting HMM T (S, A, S), from its stationary law (differentiable)."""
    E = jnp.transpose(T, (1, 0, 2))
    tab = E
    for _ in range(data["B"] - 1):
        tab = jnp.einsum("wij,xjk->wxik", tab, E).reshape(-1, E.shape[1], E.shape[2])
    P = T.sum(1)
    M = (P.T - jnp.eye(P.shape[0])).at[0].set(1.0)
    v = jnp.linalg.solve(M, jnp.zeros(P.shape[0]).at[0].set(1.0))

    def step(a, Mw):
        a = a @ Mw
        c = a.sum()
        return a / c, jnp.log(c)

    a, lp = jax.lax.scan(step, v, tab[data["ids"]])
    tot = jnp.sum(lp)
    for x in data["tail"]:
        a = a @ E[x]
        tot, a = tot + jnp.log(a.sum()), a / a.sum()
    return tot / data["n"]


def _unpack(z, S, A):
    return jax.nn.softmax(z.reshape(S, A * S), axis=1).reshape(S, A, S)


def hmm_polish(T0, xs, lbfgs_iters=3000, max_seconds=None):
    """L-BFGS-B (scipy) on the exact log-likelihood after EM, T = softmax of logits over (x, s') per row s, keeping the
    best point evaluated (T0 included); stops after max_seconds (capped).
    Returns (T, dict(final_loss = - mean log-likelihood, loss curve, lbfgs_iters, lbfgs_message, capped))."""
    from archb.train import lbfgs_capped

    S, A, _ = T0.shape
    data = blocks(xs, A)
    vg = jax.jit(jax.value_and_grad(lambda z: -hmm_block_loglik(_unpack(z, S, A), data)))
    z0 = np.log(np.maximum(T0, 1e-12)).ravel()
    out = lbfgs_capped(lambda z: vg(jnp.asarray(z)), z0, lbfgs_iters, max_seconds)
    if not np.isfinite(out["fun"]):  # capped before the first evaluation
        out["fun"] = float(vg(jnp.asarray(z0))[0])
    return np.asarray(_unpack(jnp.asarray(out["x"]), S, A)), {
        "final_loss": out["fun"], "loss": out["curve"], "lbfgs_iters": out["nit"], "lbfgs_message": out["message"],
        "capped": out["capped"]}


def seg_kl_rate(lp_true, lp_model, n_seg=20):
    """Held-out KL-rate estimate (1/n) sum_t [log P(x_t|x_<t) - log Q(x_t|x_<t)] on a string drawn from P, with the
    standard error over n_seg equal segments."""
    d = (np.asarray(lp_true) - np.asarray(lp_model))
    n = len(d) // n_seg * n_seg
    seg = d[:n].reshape(n_seg, -1).mean(1)
    return float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(n_seg))


# ------------------------------------------------------------------------------- exact metrics
def kl_curve(P, Q):
    """P, Q: lists of word-probability arrays for L = 1..Lmax. Returns (KL_L, D_L = KL_L / L, d_L) where
    d_L = KL_L - KL_{L-1} = E_P KL(P(x_L | x_{1:L-1}) || Q(x_L | x_{1:L-1})) (chain rule)."""
    kl = []
    for p, q in zip(P, Q):
        m = p > 0
        kl.append(float(np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300))))))
    kl = np.asarray(kl)
    return kl, kl / np.arange(1, len(kl) + 1), np.diff(np.concatenate([[0.0], kl]))


def entropy_curve(P):
    """Block entropies H_L and conditional entropies h_L = H_L - H_{L-1} (nats)."""
    H = np.array([float(-np.sum(p[p > 0] * np.log(p[p > 0]))) for p in P])
    return H, np.diff(np.concatenate([[0.0], H]))


def gap_dist(model, nmax):
    """Renewal gap law: P(n 0s between two 1s) = P(1 0^n 1) / P(1), n = 0..nmax."""
    p1 = model.prob([1])
    return np.array([model.prob([1] + [0] * n + [1]) / p1 for n in range(nmax + 1)])


def n_params_hmm(S, A):
    return S * (S * A - 1)


def n_params_markov(k, A):
    return A**k * (A - 1)


# ------------------------------------------------------------------------------- sequence 5
def flw_qubit(lam=0.8, alpha=1.0, xi=None):
    """Kraus operators (2, 2, 2) of the qubit reduction (p = 1) of Theorem 5 in Fanizza, Lumbreras, Winter,
    arXiv:2209.11225. Symbols: b = 0, a = 1.
      A = e^{-rX} e^{i alpha Z / 2} e^{rX},  tanh 2r = (1 - lam) / |1 - lam e^{i alpha}|,   K_b = sqrt(lam) A;
      I - K_b^dag K_b = sum_j mu_j |v_j><v_j| has rank 1 (mu clipped at 0) and K_a = sqrt(mu) |xi><v|, xi = |+>.
    a resets the memory to xi (a renewal process); the b map has eigenvalues lam, lam, lam e^{+-i alpha}: with alpha / pi
    irrational no finite nonnegative realization exists (Perron-Frobenius), so no HMM of any size is exact."""
    from scipy.linalg import expm

    X, Z = np.array([[0, 1], [1, 0]], complex), np.diag([1.0, -1.0]).astype(complex)
    r = 0.5 * np.arctanh((1 - lam) / abs(1 - lam * np.exp(1j * alpha)))
    Kb = np.sqrt(lam) * expm(-r * X) @ expm(0.5j * alpha * Z) @ expm(r * X)
    mu, V = np.linalg.eigh(np.eye(2) - Kb.conj().T @ Kb)
    xi = np.array([1.0, 1.0]) / np.sqrt(2) if xi is None else np.asarray(xi)
    j = int(np.argmax(mu))
    Ka = np.sqrt(max(mu[j], 0.0)) * np.outer(xi, V[:, j].conj())
    return np.stack([Kb, Ka]).astype(complex)


def sample_linear(lin, n, seed):
    """Sample x_1..x_n from any (positive) linear representation by filtering from its stationary vector."""
    rng = np.random.default_rng(seed)
    E = np.stack(lin.E)
    u, xs = rng.random(n), np.empty(n, dtype=np.int64)
    r = lin.v
    for t in range(n):
        R = np.einsum("i,xij->xj", r, E)
        p = np.clip(np.real(R @ lin.f), 0, None)
        c = np.cumsum(p)
        x = min(int(np.searchsorted(c, u[t] * c[-1], side="right")), lin.A - 1)
        xs[t], r = x, R[x] / p[x]
    return xs


def hankel_rank(lin, L=12, tol=1e-10, gap=False, n_gap=64):
    """Numerical rank of a Hankel matrix of the process and its singular values (normalized by the largest).
    gap=False: H[u, w] = P(uw) over all words u, w of length 0..L/2 (the rank of the process if L/2 is long enough).
    gap=True (renewal processes, binary, 1 = event): H[i, j] = g(i + j), g = gap law (gap_dist), n_gap x n_gap; the
    rank of a renewal process equals the rank of its gap law Hankel matrix."""
    if gap:
        g = gap_dist(lin, 2 * n_gap - 2)
        H = g[np.arange(n_gap)[:, None] + np.arange(n_gap)[None, :]]
    else:
        E, k = np.stack(lin.E), L // 2
        R, F = [lin.v[None, :]], [lin.f[:, None]]
        for _ in range(k):  # prefix states v E[u] and suffix vectors E[w] f, words of length 1..k
            R.append(np.einsum("ui,xij->uxj", R[-1], E).reshape(-1, len(lin.v)))
            F.append(np.einsum("xij,jw->ixw", E, F[-1]).reshape(len(lin.v), -1))
        H = np.real(np.concatenate(R) @ np.concatenate(F, axis=1))
    sv = np.linalg.svd(H, compute_uv=False)
    sv = sv / sv[0]
    return int((sv > tol).sum()), sv


def hazard_chain(g, S):
    """Renewal-type HMM with S states (binary, 1 = event) from a gap law g (g[n] = P(n 0s between two 1s)): state k =
    0s since the last 1 (k < S - 1) emits 1 with the hazard h_k = g_k / P(gap >= k) and returns to 0, else emits 0 and
    goes to k + 1; state S - 1 holds the tail with the geometric hazard of matching mean. Exact for gap laws supported
    on 0..S-2 plus a geometric tail; the 'count-down' starting point of the HMM fits."""
    g = np.asarray(g, float) / np.sum(g)
    surv = np.concatenate([[1.0], 1 - np.cumsum(g)])[:-1]
    h = np.clip(g / np.maximum(surv, 1e-300), 0.0, 1.0)
    tail = g[S - 1:]
    ht = 1.0 / (1.0 + np.arange(len(tail)) @ tail / max(tail.sum(), 1e-300))
    T = np.zeros((S, 2, S))
    for k in range(S):
        hk = h[k] if k < S - 1 else ht
        T[k, 1, 0], T[k, 0, min(k + 1, S - 1)] = hk, 1 - hk
    return T


def abel_stationary(P, eps=1e-10):
    """Stationary row vector as the Abel limit eps sum_k (1 - eps)^k u0 P^k from the uniform law u0 (as archb.iolin):
    nonnegative and defined for reducible chains (a mixture over the closed classes), equal to the stationary law
    when it is unique. Differentiable (JAX)."""
    n = P.shape[0]
    v = jnp.linalg.solve((jnp.eye(n) - (1 - eps) * P).T, jnp.full(n, eps / n))
    return v / v.sum()


def hmm_linear_abel(T):
    T = np.asarray(T, float)
    return Linear(np.asarray(abel_stationary(jnp.asarray(T.sum(1)))), [T[:, x, :] for x in range(T.shape[1])],
                  np.ones(T.shape[0]))


def hmm_logprobs_abel(T, xs):
    """hmm_logprobs from the Abel stationary law (the same law as hmm_linear_abel)."""
    T = jnp.asarray(T, jnp.float64)
    return np.asarray(_hmm_lp(T, abel_stationary(T.sum(1)), jnp.asarray(xs)))


def word_law(T, L):
    """Probabilities of all A^L words (lexicographic, first symbol most significant) under the HMM T started from its
    Abel stationary law (differentiable)."""
    R = abel_stationary(T.sum(1))[None, :]
    for _ in range(L):
        R = jnp.einsum("wi,ixj->wxj", R, T).reshape(-1, T.shape[0])
    return R.sum(1)


def exact_objective(P_L, L):
    """D_L(T) = KL(P_L || Q_L(T)) / L for the fixed word law P_L (numpy, A^L words)."""
    P_L = np.asarray(P_L, float)
    m = P_L > 0
    ent = float(np.sum(P_L[m] * np.log(P_L[m])))
    Pj, idx = jnp.asarray(P_L[m]), jnp.asarray(np.flatnonzero(m))

    def D(T):
        return (ent - jnp.sum(Pj * jnp.log(jnp.maximum(word_law(T, L)[idx], 1e-300)))) / L

    return D


def exact_fit(P_L, L, T0, maxiter=20000, max_seconds=None):
    """HMM fitted to the exact law of the words of length L (no sample): L-BFGS-B on D_L(T), T = softmax of logits
    over (x, s') per row, from T0. Returns (T, dict(D_L, D_L_init, lbfgs_iters, lbfgs_message, capped, loss curve))."""
    from archb.train import lbfgs_capped

    S, A, _ = T0.shape
    D = exact_objective(P_L, L)
    vg = jax.jit(jax.value_and_grad(lambda z: D(_unpack(z, S, A))))
    z0 = np.log(np.maximum(T0, 1e-12)).ravel()
    f0 = float(vg(jnp.asarray(z0))[0])
    out = lbfgs_capped(lambda z: vg(jnp.asarray(z)), z0, maxiter, max_seconds)
    z = out["x"] if out["fun"] <= f0 else z0
    return np.asarray(_unpack(jnp.asarray(z), S, A)), {
        "D_L": min(out["fun"], f0), "D_L_init": f0, "lbfgs_iters": out["nit"], "lbfgs_message": out["message"],
        "capped": out["capped"], "loss": out["curve"]}
