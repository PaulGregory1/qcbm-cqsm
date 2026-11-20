"""Memory diagnostics of the regime-switching models (msar, smsar), on fits that already exist (no retraining of the
circuits). Every model is an archb.iolin representation (row vector a, M[x, x'], f).

Predictive information (evaluation on one held-out path xs, targets t >= t0, natural log):
  p_* (y | H_t)          the truth filtered from the start of the path (IOLin.logprobs)
  p_k (y | x_{t-k..t})   the truth restarted k steps back from its stationary joint law given x_{t-k} (window_logprobs);
                         k = 0 is the oracle memoryless predictor p_0 = the truth's averaged kernel, whose mixture
                         weights P(S_{t+1} = s | X_t = x) depend on x
  G_k = mean log p_* / p_k  (G_0 = I(Y; past | X_t)),  G_theta = mean log p_* / q_theta,  R_theta = 1 - G_theta / G_0
  reset-and-replay: the same window predictor for a model (the model restarted k steps back).

Frozen interventions on a trained C-QSM (archb.cqsm transfer matrices, real coordinates of rho), applied after each
step, before the next input:
  dephasing    rho -> (1 - lam) rho + lam Delta_U(rho), Delta_U(rho) = sum_j P_j rho P_j, P_j = U|j><j|U^dag
               (U = I: archb.cqsm.dephase); U can be optimized on the training paths with theta frozen
  reset        rho -> Tr(rho) rho_reset, rho_reset = the stationary mean memory state (history independent)
Exact classical reduction of a dephased memory: T[x, y][i, j] = sum_c |<j|K[x, y, c]|i>|^2, an input-output HMM whose
path law equals that of the circuit with gamma = 1 (classical_from_kraus).

Regime posterior probes: q*_t = P(S_t = turbulent | x_0..t) from the truth's filter; least-squares probes of
q*_t on features of (x_t) or (x_t, memory) fitted on the first half of the held-out path, scored on the second half.
Commutators of memory states of histories that end at the same x: kappa = ||[rho, sigma]||_F / (sqrt 2 ||rho - I/D||_F
||sigma - I/D||_F) in [0, 1] (Boettcher-Wenzel bound; for a qubit, the sine of the angle between Bloch vectors).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from archb import cqsm, iolin

jax.config.update("jax_enable_x64", True)

FLOOR = 1e-300


# ------------------------------------------------------------------------------------- predictive information
def window_logprobs(lin, xs, k, t0, chunk_elems=4_000_000):
    """log q(x_t | x_{t-1-k} .. x_{t-1}) for every target t in t0 .. len(xs) - 1, the model restarted from its
    stationary joint law given x_{t-1-k} (t0 >= k + 1)."""
    xs = np.asarray(xs)
    M, f, u = lin.M, lin.f, lin.u
    n, d = len(xs), lin.d
    idx_all = np.arange(t0, n)
    out = np.empty(len(idx_all))
    step = max(1, chunk_elems // (d * d))
    for c0 in range(0, len(idx_all), step):
        idx = idx_all[c0:c0 + step]
        a = u[xs[idx - 1 - k]]
        a = a / np.real(a @ f)[:, None]
        for j in range(k, 0, -1):
            s = idx - 1 - j
            a = np.einsum("pi,pij->pj", a, M[xs[s], xs[s + 1]])
            a = a / np.real(a @ f)[:, None]
        p = np.real(np.einsum("pi,pij,j->p", a, M[xs[idx - 1], xs[idx]], f))
        out[c0:c0 + len(idx)] = np.log(np.maximum(p, FLOOR))
    return out


def seg_mean(diff, n_seg=20):
    """Mean of a per-step difference and its standard error over n_seg equal segments."""
    d = np.asarray(diff)
    n = len(d) // n_seg * n_seg
    seg = d[:n].reshape(n_seg, -1).mean(1)
    return float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(n_seg))


# ------------------------------------------------------------------------------------- frozen interventions
def deph_matrix(U, D):
    """Real-coordinate matrix of Delta_U acting on the right of a row vector (D^2 x D^2)."""
    C = jnp.asarray(cqsm.herm_basis(D))
    Ci = jnp.asarray(np.linalg.inv(cqsm.herm_basis(D)))
    S = 0
    for j in range(D):
        P = jnp.outer(U[:, j], jnp.conj(U[:, j]))
        S = S + jnp.kron(P.T, P)  # row-major vec(A rho B) = vec(rho) kron(A^T, B)
    return jnp.real(Ci @ S @ C)


def unitary(h, D):
    """U = expm(i H), H Hermitian from D^2 real numbers (diagonal, then Re / Im of the upper entries)."""
    H = jnp.zeros((D, D), complex).at[jnp.diag_indices(D)].set(h[:D])
    iu = np.triu_indices(D, 1)
    off = h[D::2][: len(iu[0])] + 1j * h[D + 1::2][: len(iu[0])]
    H = H.at[iu].set(off)
    H = H + jnp.conj(jnp.triu(H, 1)).T
    return expm(1j * H)


def dephased(M, D, lam, U=None):
    """Transfer matrices followed by rho -> (1 - lam) rho + lam Delta_U(rho)."""
    U = jnp.eye(D, dtype=complex) if U is None else U
    return M @ ((1 - lam) * jnp.eye(D * D) + lam * deph_matrix(U, D))


def reset(M, f, r):
    """Transfer matrices followed by rho -> Tr(rho) rho_reset (r: real coordinates of rho_reset, r . f = 1)."""
    return M @ np.outer(f, r)


def optimize_basis(M, f, D, X, n_starts=4, seed=0, maxiter=300):
    """The dephasing basis U that maximizes the training log-likelihood with the circuit frozen (full dephasing).
    -> (U, train mean log-likelihood)."""
    M, f, Xj = jnp.asarray(M), jnp.asarray(f), jnp.asarray(X)
    nll = jax.jit(lambda h: -iolin.loglik(dephased(M, D, 1.0, unitary(h, D)), f, Xj))
    vg = jax.jit(jax.value_and_grad(lambda h: -iolin.loglik(dephased(M, D, 1.0, unitary(h, D)), f, Xj)))
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        h0 = np.zeros(D * D) if s == 0 else rng.normal(0, 1.0, D * D)
        r = minimize(lambda h: tuple(np.asarray(v, float) for v in vg(jnp.asarray(h))), h0, jac=True,
                     method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or r.fun < best.fun:
            best = r
    return np.asarray(unitary(jnp.asarray(best.x), D)), float(-nll(jnp.asarray(best.x)))


def classical_from_kraus(K):
    """Input-output HMM of the fully dephased memory: M[x, y][i, j] = sum_c |K[x, y, c, j, i]|^2 (K[..., i, j] =
    <i|K|j>), f = 1. Rows sum to one over (y, j)."""
    K = np.asarray(K)
    return iolin.IOLin(np.einsum("xycji->xyij", np.abs(K) ** 2), np.ones(K.shape[-1]))


def word_logprobs(lin, L):
    """log P(x_0 .. x_{L-1}) of every word of length L under the stationary joint law (A^L words, lexicographic)."""
    A = lin.A
    a, last = lin.u, np.arange(A)  # words of length 1; word w ends with symbol last[w]
    for _ in range(L - 1):
        a = np.einsum("wi,wyij->wyj", a, lin.M[last]).reshape(-1, lin.d)  # word w y -> index w A + y
        last = np.tile(np.arange(A), len(last))
    return np.log(np.maximum(np.real(a @ lin.f), FLOOR))


# ------------------------------------------------------------------------------------- posterior probes
def regime_posterior(truth, xs, regime):
    """q*_t = P(regime of the state that emitted x_t = 1 | x_0..t) from the truth's exact filter."""
    a = truth.filtered(xs)
    return np.real(a[:, regime == 1].sum(1))


def features(xs, states=None, A=32, interact=False):
    """[one-hot x_t, memory coordinates, their pairwise products] (real; complex states split); interact adds the
    products of the one-hot x_t with the memory coordinates (a readout of the regime that depends on x_t, which a
    trained C-QSM memory can need)."""
    X1 = np.eye(A)[np.asarray(xs)]
    F = [X1]
    if states is not None:
        S = np.asarray(states)
        S = np.concatenate([S.real, S.imag], 1) if np.iscomplexobj(S) else S
        S = S[:, np.std(S, 0) > 1e-12]
        iu = np.triu_indices(S.shape[1])
        F += [S, (S[:, :, None] * S[:, None, :])[:, iu[0], iu[1]]]
        if interact:
            F.append((X1[:, :, None] * S[:, None, :]).reshape(len(S), -1))
    return np.concatenate(F, 1)


def fit_probe(F, q, ridge=1e-8, clip=1e-4):
    """Least-squares probe of the soft label q on the features (exact when q is linear in them, as for the filter of
    a classical 2-state memory), clipped to [clip, 1 - clip]. -> predict(F)."""
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z = np.concatenate([(F - mu) / sd, np.ones((len(F), 1))], 1)
    w = np.linalg.solve(Z.T @ Z + ridge * len(Z) * np.eye(Z.shape[1]), Z.T @ q)
    return lambda G: np.clip(np.concatenate([(G - mu) / sd, np.ones((len(G), 1))], 1) @ w, clip, 1 - clip)


def posterior_scores(qhat, qstar, s, n_bins=10):
    """MSE to the true posterior, Brier score and balanced accuracy against the realized regime, mean excess
    KL(Bern(q*) || Bern(qhat)), reliability table (mean qhat, realized frequency, count per bin of qhat)."""
    e = 1e-12
    qh, qs = np.clip(qhat, e, 1 - e), np.clip(qstar, e, 1 - e)
    kl = qs * np.log(qs / qh) + (1 - qs) * np.log((1 - qs) / (1 - qh))
    pred = qhat > 0.5
    bacc = 0.5 * (np.mean(pred[s == 1]) + np.mean(~pred[s == 0]))
    b = np.minimum((qhat * n_bins).astype(int), n_bins - 1)
    rel = [[float(qhat[b == i].mean()), float(s[b == i].mean()), int((b == i).sum())] for i in range(n_bins)
           if (b == i).any()]
    return {"mse": float(np.mean((qhat - qstar) ** 2)), "brier": float(np.mean((qhat - s) ** 2)),
            "kl": float(np.mean(kl)), "bacc": float(bacc), "reliability": rel}


# ------------------------------------------------------------------------------------- memory geometry
def rhos(states, m):
    Ci = np.linalg.inv(cqsm.herm_basis(2**m))
    D = 2**m
    return (np.asarray(states) @ Ci).reshape(-1, D, D)


def commutators(R, xs, n_pairs=20000, seed=0):
    """kappa over random pairs of times with the same x_t, and the mean spread ||rho - mean_x rho||_F."""
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs)
    D = R.shape[1]
    R0 = R - np.eye(D) / D
    kap, spread = [], []
    for x in np.unique(xs):
        ix = np.flatnonzero(xs == x)
        if len(ix) < 50:
            continue
        spread.append(np.linalg.norm(R[ix] - R[ix].mean(0), axis=(1, 2)).mean() * len(ix))
        n = max(1, int(n_pairs * len(ix) / len(xs)))
        i, j = rng.choice(ix, n), rng.choice(ix, n)
        Cm = R[i] @ R[j] - R[j] @ R[i]
        den = np.sqrt(2) * np.linalg.norm(R0[i], axis=(1, 2)) * np.linalg.norm(R0[j], axis=(1, 2))
        ok = den > 1e-9
        kap += list(np.linalg.norm(Cm, axis=(1, 2))[ok] / den[ok])
    kap = np.asarray(kap)
    return {"kappa_mean": float(kap.mean()), "kappa_median": float(np.median(kap)),
            "kappa_q90": float(np.quantile(kap, 0.9)), "spread": float(np.sum(spread) / len(xs))}


def predictive(lin, states, xs):
    """Predictive laws q(. | x_t, a_t) of x_{t+1} for every t (rows sum to 1)."""
    xs, states = np.asarray(xs), np.asarray(states)
    P = np.concatenate([np.real(np.einsum("ti,tyij,j->ty", states[c:c + 2000], lin.M[xs[c:c + 2000]], lin.f))
                        for c in range(0, len(xs), 2000)])
    return np.clip(P, 0, None) / np.clip(P, 0, None).sum(1, keepdims=True)


def memory_rhos(states, m=None):
    """Memory states for holevo: a C-QSM's density matrices (m qubits, real coordinates of rho), else classical
    beliefs a (rows normalized to sum 1; they stand for diag(a), kept as vectors)."""
    if m is not None:
        return rhos(states, m)
    S = np.clip(np.real(np.asarray(states)), 0, None)
    return S / S.sum(1, keepdims=True)


def entropy(R):
    """Von Neumann entropies (nats) of a batch of density matrices, or Shannon entropies of a batch of beliefs."""
    if R.ndim == 2:
        return -np.sum(np.where(R > 0, R * np.log(np.where(R > 0, R, 1)), 0), axis=1)
    lam = np.clip(np.linalg.eigvalsh(0.5 * (R + np.conj(np.swapaxes(R, 1, 2)))), 0, None)
    return -np.sum(np.where(lam > 0, lam * np.log(np.where(lam > 0, lam, 1)), 0), axis=1)


def holevo(R, xs):
    """chi_{H:M|X} = sum_x p(x) [S(mean of rho_t over t with x_t = x) - mean S(rho_t)]: history information held by the
    memory beyond the current value (<= ln D), on the empirical ensemble of the path."""
    xs = np.asarray(xs)
    H = entropy(R)
    chi = 0.0
    for x in np.unique(xs):
        ix = np.flatnonzero(xs == x)
        chi += len(ix) * (entropy(R[ix].mean(0)[None])[0] - H[ix].mean())
    return float(chi / len(xs))


def predictive_info(P, xs, n_seg=20, seed=0):
    """Predictive information encoded by a memory, from its predictive laws P[t] = q(. | x_t, m_t) of x_{t+1}.
    The memory-averaged law qbar(. | x) is estimated on the first half, the scores on the second half:
      model   I(Y : M | X) under the model's own law: mean KL(P_t || qbar(. | x_t))
      true    mean log P_t(y_t) / qbar(y_t | x_t) with y_t the realized x_{t+1} (the information the memory adds to
              the model's own memoryless average, scored on the true data; the truth's own memory gives G_0)
      null    'true' with the laws P_t shuffled among the steps of the same x_t (bias control: about 0 or below)
    'true' and 'null' come with their SE over n_seg segments."""
    xs = np.asarray(xs)
    P, x, y = np.asarray(P)[:-1], xs[:-1], xs[1:]
    h = len(x) // 2
    A = P.shape[1]
    Q = np.full((A, A), 1.0 / A)
    for v in np.unique(x[:h]):
        Q[v] = P[:h][x[:h] == v].mean(0)
    Q = np.clip(Q, 1e-300, None)
    P2, x2, y2 = np.clip(P[h:], 1e-300, None), x[h:], y[h:]
    kl = np.sum(P2 * (np.log(P2) - np.log(Q[x2])), 1)
    r = np.arange(len(x2))
    tr = np.log(P2[r, y2]) - np.log(Q[x2, y2])
    rng = np.random.default_rng(seed)
    perm = np.arange(len(x2))
    for v in np.unique(x2):
        ix = np.flatnonzero(x2 == v)
        perm[ix] = rng.permutation(ix)
    nu = np.log(P2[perm, y2]) - np.log(Q[x2, y2])
    return {"model": float(kl.mean()), "true": seg_mean(tr, n_seg), "null": seg_mean(nu, n_seg)}


def swap_effect(model, Sm, truth, St, xs, qstar, lo=0.2, hi=0.8, n_pairs=5000, seed=0):
    """Memory-state swap between histories that end at the same x with contrasting true posteriors (q* < lo and
    q* > hi): mean TV between the two predictive laws, for the model (filtered states Sm) and the truth (St), and
    their mean gap."""
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs)
    I, J = [], []
    for x in np.unique(xs):
        a = np.flatnonzero((xs == x) & (qstar < lo))
        b = np.flatnonzero((xs == x) & (qstar > hi))
        if len(a) and len(b):
            n = max(1, int(n_pairs * min(len(a), len(b)) / len(xs) * 4))
            I += list(rng.choice(a, n))
            J += list(rng.choice(b, n))
    I, J = np.asarray(I), np.asarray(J)
    tm = 0.5 * np.abs(predictive(model, Sm[I], xs[I]) - predictive(model, Sm[J], xs[J])).sum(1)
    tt = 0.5 * np.abs(predictive(truth, St[I], xs[I]) - predictive(truth, St[J], xs[J])).sum(1)
    return {"n_pairs": int(len(I)), "tv_model": float(tm.mean()), "tv_truth": float(tt.mean()),
            "abs_gap": float(np.abs(tm - tt).mean())}
