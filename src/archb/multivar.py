"""Sequence 9: several correlated variables on one register (the setting of Coyle et al. 2021, with the conditional
kernels and regime memories of sequences 4-6).

State: d coordinates of k bits each, n = d k bits, K = 2^n joint bins. Joint bin index i = sum_c i_c 2^(k (d - 1 - c))
(row-major, coordinate 0 most significant, as var2d of sequence 2); the circuits see the Gray code of each coordinate,
concatenated MSB first (archb.cqsm.bin_labels). Every kernel and transfer matrix is in joint BIN order.

A spec is a dict: n, d, k, values (d, 2^k) bin values per coordinate, edges (d, 2^k + 1) with the outer edges at +-BIG.

Synthetic truths (exact likelihoods, input-output HMMs as archb.regimes): Markov-switching VAR(1) on 2 coordinates,
x' = c_s x + sigma_s L(rho_s) eps, regimes kept with probability 0.98; given the bin centres z of x_t and the new
regime, x_{t+1} is the binned bivariate normal.
  msvar   calm c = 0.95, sigma = 0.3, rho = 0.2; turbulent c = 0.7, sigma = 1.0, rho = 0.8 (correlation rises in crises)
  msvarc  the same with rho = 0.5 in both regimes (control: the regime changes the volatility only)
Grid: 4 bits per coordinate, 16 equal bins of [-4, 4] each.

Models (all as archb.iolin representations):
  kernels   circuits 'qa' (C-QSM register with m = 0, 2 ancillas, cross gates between the coordinate registers),
            'q0' (no ancilla, cross gates), 'qn' (2 ancillas, no cross gate: ablation); conditional RBM 'rbm<H>' on the
            Gray bits (n + n^2 + H + 2 n H parameters, exact normalization over the K outputs); mixture 'mix<K>' of K
            bivariate normals (d = 2) with weights softmax(a + b.v), means c + D v, log sd f + g |v|, correlation
            0.99 tanh(r + q mean|v|), v = the values of the current bin (15 K - 3 parameters)
  switch    S regimes over S kernels, P(s' | s, x) = softmax over s' of W[s, 0, s'] + sum_q W[s, q + 1, s'] u_q(x) with
            u the Gray bits of x (bit-linear: the input enters as the C-QSM memory sees it); S (S - 1) (n + 1) free
            transition parameters
  C-QSM     archb.cqsm with memory m = 1, 2 on the 'qa' circuit
  IOHMM     archb.regimes EM ('fact', 'tied' = HMM on the joint bins) on K symbols
  MS-VAR    the truth family on the bins (binned inputs: the bin values), per-regime c, sd per coordinate, rho, stay
Continuous references (real data): GARCH(1,1)-t margins (archb.families) joined by a Gaussian copula with constant
(CCC), DCC(1,1) or identity (independent) correlation, scored on the bins by integrating over each cell (Gauss-Legendre;
for d > 2 nested over the first d - 2 coordinates).
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jnorm
from scipy.optimize import minimize
from scipy.stats import norm

from archb import cqsm, iolin
from archb import regimes as R

jax.config.update("jax_enable_x64", True)

BIG = 1e4
GL_TRUTH, GL_MODEL = 24, 8
L = 3
TRUTH = {"msvar": {"c": (0.95, 0.7), "sig": (0.3, 1.0), "rho": (0.2, 0.8), "stay": (0.98, 0.98)},
         "msvarc": {"c": (0.95, 0.7), "sig": (0.3, 1.0), "rho": (0.5, 0.5), "stay": (0.98, 0.98)}}


# ------------------------------------------------------------------------------------------------ grids
def coords(d, k):
    """(K, d) coordinate bin indices of every joint bin."""
    i = np.arange(2 ** (d * k))
    return np.stack([(i >> (k * (d - 1 - c))) & (2**k - 1) for c in range(d)], axis=1)


def make_spec(values, edges):
    values, edges = np.asarray(values, float), np.asarray(edges, float).copy()
    d, nb = values.shape
    edges[:, 0], edges[:, -1] = -BIG, BIG
    k = int(np.log2(nb))
    return {"n": d * k, "d": d, "k": k, "values": values, "edges": edges}


def uniform_spec(d=2, k=4, lo=-4.0, hi=4.0):
    e = np.linspace(lo, hi, 2**k + 1)
    return make_spec(np.tile(0.5 * (e[:-1] + e[1:]), (d, 1)), np.tile(e, (d, 1)))


def joint_values(spec):
    """(K, d) values of every joint bin."""
    C = coords(spec["d"], spec["k"])
    return np.stack([spec["values"][c][C[:, c]] for c in range(spec["d"])], axis=1)


def gray_bits(spec):
    """(K, n) Gray bits (0 / 1, MSB first) of every joint bin, as the circuit sees them."""
    lab = cqsm.bin_labels(spec["n"], spec["d"])
    return ((lab[:, None] >> (spec["n"] - 1 - np.arange(spec["n"]))[None, :]) & 1).astype(np.float64)


# ------------------------------------------------------------------------------------------------ bivariate cells
def bvn_cells(m1, m2, s1, s2, rho, e1, e2, nodes=GL_MODEL):
    """P[..., j1, j2] of a bivariate normal (means m, sds s, correlation rho; any common batch shape) over the cells of
    edges e1 (..., n1 + 1), e2 (..., n2 + 1). Gauss-Legendre in x1 on each bin (on [m1 - 10 s1, m1 + 10 s1] where the
    bin is wider), times the conditional normal CDF of x2; rows renormalized (quadrature error)."""
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    m1, m2, s1, s2, rho = (jnp.asarray(x)[..., None, None] for x in (m1, m2, s1, s2, rho))
    e1, e2 = jnp.asarray(e1), jnp.asarray(e2)
    a = jnp.maximum(e1[..., :-1, None], m1 - 10 * s1)
    b = jnp.maximum(jnp.minimum(e1[..., 1:, None], m1 + 10 * s1), a)  # (..., n1, 1)
    x1 = 0.5 * (b - a) * xg + 0.5 * (b + a)  # (..., n1, Q)
    w = 0.5 * (b - a) * wg * jnorm.pdf(x1, m1, s1)
    cm = m2 + rho * s2 / s1 * (x1 - m1)
    cs = s2 * jnp.sqrt(1 - rho**2)
    cdf = jnorm.cdf((e2[..., None, None, :] - cm[..., None]) / cs[..., None])  # (..., n1, Q, n2 + 1)
    P = jnp.einsum("...nq,...nqj->...nj", w, jnp.diff(cdf, axis=-1))
    P = jnp.maximum(P, 0.0)
    return P / jnp.sum(P, axis=(-2, -1), keepdims=True)


def gauss_cells(Rm, H, nodes=12, chunk=256):
    """Cell probabilities (T, n^d) of N(0, R_t) over the per-day edges H (T, d, n + 1) (normal-score units, outer edges
    +-inf allowed). d = 2: bvn_cells; d > 2: tensor Gauss-Legendre over the first d - 2 coordinates (nodes per bin,
    truncated to [-6, 6]) times the bivariate cells of the last two given them. numpy in, numpy out."""
    Rm, H = np.asarray(Rm, float), np.clip(np.asarray(H, float), -BIG, BIG)
    T, d, nb1 = H.shape
    nb = nb1 - 1
    if d == 2:
        out = [np.asarray(bvn_cells(0.0, 0.0, 1.0, 1.0, Rm[t:t + chunk, 0, 1], H[t:t + chunk, 0], H[t:t + chunk, 1],
                                    nodes)).reshape(-1, nb * nb) for t in range(0, T, chunk)]
        return np.concatenate(out)
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    o, ii = np.arange(d - 2), np.array([d - 2, d - 1])
    out = np.empty((T, nb**d))
    for t in range(T):
        Ro, Rio, Ri = Rm[t][np.ix_(o, o)], Rm[t][np.ix_(ii, o)], Rm[t][np.ix_(ii, ii)]
        a = np.maximum(H[t, o, :-1], -6.0)
        b = np.maximum(np.minimum(H[t, o, 1:], 6.0), a)  # (d - 2, nb)
        x = 0.5 * (b - a)[..., None] * xg + 0.5 * (b + a)[..., None]  # (d - 2, nb, Q)
        w = 0.5 * (b - a)[..., None] * wg
        grids = np.meshgrid(*[x[c].ravel() for c in range(d - 2)], indexing="ij")
        wgrid = np.prod(np.meshgrid(*[w[c].ravel() for c in range(d - 2)], indexing="ij"), axis=0).ravel()
        X = np.stack([g.ravel() for g in grids], axis=1)  # (G, d - 2)
        Ri_o = np.linalg.solve(Ro, Rio.T).T  # R_io R_oo^-1
        dens = np.exp(-0.5 * np.einsum("gi,ij,gj->g", X, np.linalg.inv(Ro), X)) / np.sqrt(
            (2 * np.pi) ** (d - 2) * np.linalg.det(Ro))
        mu = X @ Ri_o.T  # (G, 2)
        C = Ri - Ri_o @ Rio.T
        sd = np.sqrt(np.diag(C))
        r = C[0, 1] / (sd[0] * sd[1])
        P2 = np.asarray(bvn_cells(mu[:, 0], mu[:, 1], sd[0], sd[1], r, H[t, ii[0]], H[t, ii[1]], nodes))  # (G, nb, nb)
        W = (wgrid * dens).reshape([nb, nodes] * (d - 2))
        P2 = P2.reshape([nb, nodes] * (d - 2) + [nb, nb])
        axes = tuple(2 * c + 1 for c in range(d - 2))
        P = np.sum(W.reshape(W.shape + (1, 1)) * P2, axis=axes)  # (nb,) * d
        P = np.maximum(P, 0.0)
        out[t] = (P / P.sum()).ravel()
    return out


# ------------------------------------------------------------------------------------------------ synthetic truths
def ar_kernels(c, sig, rho, spec, nodes=GL_TRUTH):
    """(S, K, K): per regime, the binned bivariate normal with mean c_s z (z = joint bin values), sd sig_s per coordinate
    (sig (S,) or (S, 2)) and correlation rho_s."""
    Z = jnp.asarray(joint_values(spec))
    c, rho = jnp.asarray(c, jnp.float64), jnp.asarray(rho, jnp.float64)
    sig = jnp.broadcast_to(jnp.asarray(sig, jnp.float64).reshape(len(c), -1), (len(c), 2))
    e = jnp.asarray(spec["edges"])
    P = bvn_cells(c[:, None] * Z[None, :, 0], c[:, None] * Z[None, :, 1], sig[:, :1], sig[:, 1:], rho[:, None],
                  e[0], e[1], nodes)  # (S, K, n1, n2)
    return P.reshape(len(c), Z.shape[0], -1)


def msvar_T(c, sig, rho, stay, spec, nodes=GL_TRUTH):
    """Input-output HMM T[x, h, x', h'] = Q[h, h'] K_h'[x, x'] of the 2-regime MS-VAR."""
    stay = jnp.asarray(stay, jnp.float64)
    Q = jnp.stack([jnp.stack([stay[0], 1 - stay[0]]), jnp.stack([1 - stay[1], stay[1]])])
    return jnp.einsum("hk,kxy->xhyk", Q, ar_kernels(c, sig, rho, spec, nodes))


def truth_T(kind, spec=None):
    p = TRUTH[kind]
    return np.asarray(msvar_T(p["c"], p["sig"], p["rho"], p["stay"], spec or uniform_spec()))


def make_data(kind, seed, n_paths=R.N_PATHS, n_keep=R.N_KEEP, n_test=100_000):
    T = truth_T(kind)
    X, H = R.simulate(T, n_paths, n_keep, R.BURN_IN, seed)
    xt, ht = R.simulate(T, 1, n_test, R.BURN_IN, seed + 1)
    return {"X": X, "H": H, "test": xt[0], "test_h": ht[0]}


def _msvar_unpack(z):
    return z[0:2], jnp.exp(z[2:6]).reshape(2, 2), 0.99 * jnp.tanh(z[6:8]), jax.nn.sigmoid(z[8:10])


def fit_msvar(X, spec, starts=4, seed=0, maxiter=1000, val=None):
    """Maximum binned likelihood of the MS-VAR family (10 parameters: per regime c, rho, stay and the sd of each
    coordinate) on the paths X, from generic starts. Regime 1 = the larger mean sd. Returns (T, params, train ll, info);
    val(T) -> score picks among the starts when given (real data), else the training likelihood does."""
    Xj = jnp.asarray(X)

    def T_of(z):
        c, sig, rho, stay = _msvar_unpack(z)
        return msvar_T(c, sig, rho, stay, spec)

    def nll(z):
        return -iolin.loglik(T_of(z).transpose(0, 2, 1, 3), jnp.ones(2), Xj)

    vg = jax.jit(jax.value_and_grad(nll))
    rng, best = np.random.default_rng(seed), None
    for k in range(starts):
        z0 = np.concatenate([[0.5, 0.3] + 0.1 * rng.normal(size=2), np.log([0.5, 0.5, 1.2, 1.2]) + 0.2 * rng.normal(size=4),
                             [0.2, 0.6] + 0.1 * rng.normal(size=2), [3.0, 3.0] + rng.normal(size=2)])
        out = minimize(lambda z: tuple(np.asarray(v, float) for v in vg(jnp.asarray(z))), z0, jac=True,
                       method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9})
        sc = -out.fun if val is None else val(np.asarray(T_of(jnp.asarray(out.x))))
        if best is None or sc > best[0]:
            best = (sc, out)
    z = np.asarray(best[1].x)
    c, sig, rho, stay = (np.asarray(a) for a in _msvar_unpack(jnp.asarray(z)))
    if sig[0].mean() > sig[1].mean():
        c, sig, rho, stay = c[::-1], sig[::-1], rho[::-1], stay[::-1]
    T = np.asarray(msvar_T(c, sig, rho, stay, spec))
    params = {"c": c.tolist(), "sig": sig.tolist(), "rho": rho.tolist(), "stay": stay.tolist()}
    return T, params, -float(best[1].fun), {"n_params": 10, "message": str(best[1].message)}


# ------------------------------------------------------------------------------------------------ kernels
CIRCUITS = {"qa": (2, True), "q0": (0, True), "qn": (2, False)}  # kind -> (ancillas, cross gates)


def parse(kind):
    if kind in CIRCUITS:
        return kind, 0
    m = re.fullmatch(r"(rbm|mix)(\d+)", kind)
    if m is None:
        raise ValueError(f"unknown kernel {kind}")
    return m.group(1), int(m.group(2))


def is_circuit(kind):
    return kind in CIRCUITS


def n_kernel(kind, spec):
    fam, h = parse(kind)
    n = spec["n"]
    if fam in CIRCUITS:
        return cqsm.n_params(0, CIRCUITS[fam][0], L, n, spec["d"], CIRCUITS[fam][1])
    return n + n * n + h + 2 * n * h if fam == "rbm" else 15 * h - 3


def _rbm(theta, H, U):
    n = U.shape[1]
    b, A_ = theta[:n], theta[n:n + n * n].reshape(n, n)
    o = n + n * n
    c, B, W = theta[o:o + H], theta[o + H:o + H + H * n].reshape(H, n), theta[o + H + H * n:].reshape(n, H)
    vis = (b[None, :] + U @ A_.T) @ U.T
    hid = (c[None, :] + U @ B.T)[:, None, :] + (U @ W)[None, :, :]  # (x, x', H)
    return jax.nn.softmax(vis + jnp.sum(jax.nn.softplus(hid), axis=2), axis=1)


def _mix_unpack(theta, K):
    s = [K - 1, 2 * (K - 1), 2 * K, 4 * K, 2 * K, 2 * K, K, K]
    parts, o = [], 0
    for n in s:
        parts.append(theta[o:o + n])
        o += n
    a, b, c, D, f, g, r, q = parts
    return (jnp.concatenate([jnp.zeros(1), a]), jnp.concatenate([jnp.zeros((1, 2)), b.reshape(K - 1, 2)]),
            c.reshape(K, 2), D.reshape(K, 2, 2), f.reshape(K, 2), g.reshape(K, 2), r, q)


def _mix(theta, K, V, e):
    a, b, c, D, f, g, r, q = _mix_unpack(theta, K)
    pi = jax.nn.softmax(a[None, :] + V @ b.T, axis=1)  # (x, K)
    mu = c[None] + jnp.einsum("kij,xj->xki", D, V)  # (x, K, 2)
    sig = jnp.exp(f[None] + g[None] * jnp.abs(V)[:, None, :])
    rho = 0.99 * jnp.tanh(r[None] + q[None] * jnp.abs(V).mean(1)[:, None])
    P = bvn_cells(mu[..., 0], mu[..., 1], sig[..., 0], sig[..., 1], rho, e[0], e[1])  # (x, K, n1, n2)
    return jnp.einsum("xk,xkij->xij", pi, P).reshape(V.shape[0], -1)


def kernel(theta, kind, spec):
    """Q (K, K) in joint BIN order."""
    fam, h = parse(kind)
    theta = jnp.asarray(theta)
    if fam in CIRCUITS:
        na, cr = CIRCUITS[fam]
        return cqsm.transfer(theta, 0, na, L, spec["n"], 0.0, spec["d"], cr)[:, :, 0, 0]
    if fam == "rbm":
        return _rbm(theta, h, jnp.asarray(gray_bits(spec)))
    return _mix(theta, h, jnp.asarray(joint_values(spec)), jnp.asarray(spec["edges"]))


def init_kernel(kind, seed, spec, b_train, noise=0.2, start="copy"):
    """Circuits: start 'copy' = archb.cqsm init (copy gates, ancillas in |+>), for persistent synthetic paths; 'marginal'
    = every angle N(0, 0.02^2), ancillas in |+>, the final R_y layer at the training frequency of each output bit (near a
    product approximation of the i.i.d. law, as the real-data circuits of sequence 4). RBM near the product of the
    training bit marginals; mixture near the i.i.d. law of the training values (equal weights, means at training
    quantiles, training sd, no dependence on v, rho = 0)."""
    fam, h = parse(kind)
    rng = np.random.default_rng(seed)
    n = spec["n"]
    if fam in CIRCUITS:
        na, cr = CIRCUITS[fam]
        if start == "copy":
            return cqsm.init_theta(seed, 0, na, L, n, noise, spec["d"], cr)
        gates = cqsm.gate_list(0, na, L, n, spec["d"], cr)
        th = 0.02 * rng.normal(size=len(gates))
        th[[j for j, g in enumerate(gates) if g[3] == "anc_plus"]] += np.pi / 2
        p1 = np.clip(gray_bits(spec)[np.asarray(b_train)].mean(0), 1e-3, 1 - 1e-3)
        th[-n:] = 2 * np.arcsin(np.sqrt(p1))
        return th
    if fam == "rbm":
        p1 = np.clip(gray_bits(spec)[np.asarray(b_train)].mean(0), 1e-3, 1 - 1e-3)
        return np.concatenate([np.log(p1 / (1 - p1)), 0.01 * rng.normal(size=n * n), np.zeros(h),
                               0.1 * rng.normal(size=2 * n * h)])
    V = joint_values(spec)[np.asarray(b_train)]
    qs = (np.arange(h) + 0.5) / h
    c = np.stack([np.quantile(V[:, 0], rng.permutation(qs)), np.quantile(V[:, 1], rng.permutation(qs))], 1)
    f = np.tile(np.log(np.maximum(V.std(0), 1e-3)), (h, 1)) + 0.05 * rng.normal(size=(h, 2))
    return np.concatenate([0.01 * rng.normal(size=3 * (h - 1)), (c + 0.05 * rng.normal(size=(h, 2))).ravel(),
                           0.01 * rng.normal(size=4 * h), f.ravel(), 0.01 * rng.normal(size=2 * h),
                           0.01 * rng.normal(size=2 * h)])


# ------------------------------------------------------------------------------------------------ switch
def sw_n_theta(S, kind, spec):
    return S * (spec["n"] + 1) * S + S * n_kernel(kind, spec)


def sw_n_params(S, kind, spec):
    return S * (S - 1) * (spec["n"] + 1) + S * n_kernel(kind, spec)


def switch_M(theta, S, kind, spec):
    """Transfer matrices M (K, K, S, S), M[x, x'][s, s'] = P(s' | s, x) Q_s'[x, x']."""
    n, nk = spec["n"], n_kernel(kind, spec)
    theta = jnp.asarray(theta)
    W = theta[:S * (n + 1) * S].reshape(S, n + 1, S)
    U = jnp.asarray(gray_bits(spec))
    logits = W[None, :, 0, :] + jnp.einsum("xq,sqt->xst", U, W[:, 1:, :])  # (x, s, s')
    P = jax.nn.softmax(logits, axis=2)
    o = S * (n + 1) * S
    Q = jnp.stack([kernel(theta[o + j * nk:o + (j + 1) * nk], kind, spec) for j in range(S)])  # (s', x, x')
    return jnp.einsum("xst,txy->xyst", P, Q)


def init_switch(theta_kernel, S, seed, kind, spec, noise=0.1, stay=3.0):
    rng = np.random.default_rng(seed)
    n = spec["n"]
    W = 0.01 * rng.normal(size=(S, n + 1, S))
    W[:, 0, :] += stay * np.eye(S)
    kern = [np.asarray(theta_kernel) + noise * rng.normal(size=len(theta_kernel)) for _ in range(S)]
    return np.concatenate([W.ravel(), *kern])


# ------------------------------------------------------------------------------------------------ models as IOLin
def model_M(theta, model, spec):
    """(M, f) of a model id: 'k-<kind>', 'sw-<kind>-S<S>', 'cq<m>' (C-QSM, memory m, on the 'qa' circuit)."""
    if model.startswith("cq"):
        m = int(model[2:])
        return cqsm.transfer(theta, m, 2, L, spec["n"], 0.0, spec["d"], True), cqsm.f_vec(m)
    p = model.split("-")
    if p[0] == "k":
        return kernel(theta, p[1], spec)[:, :, None, None], jnp.ones(1)
    S = int(p[2][1:])
    return switch_M(theta, S, p[1], spec), jnp.ones(S)


def model_n_params(model, spec):
    if model.startswith("cq"):
        return cqsm.n_params(int(model[2:]), 2, L, spec["n"], spec["d"], True)
    p = model.split("-")
    return n_kernel(p[1], spec) if p[0] == "k" else sw_n_params(int(p[2][1:]), p[1], spec)


def cq_embed(theta_b, seed, m, spec):
    return cqsm.embed(theta_b, seed, m, 2, L, spec["n"], 0.1, spec["d"], True)


# ------------------------------------------------------------------------------------------------ IOHMM, Markov
def iohmm_random(kind, S, A, rng):
    if kind == "fact":
        Rr = rng.dirichlet(np.ones(S), size=(A, S))
        Em = rng.dirichlet(np.ones(A), size=(A, S))
        return np.einsum("xsk,xky->xsyk", Rr, Em)
    H = rng.dirichlet(np.ones(A * S), size=S).reshape(S, A, S)
    return np.broadcast_to(H, (A, S, A, S)).copy()


def iohmm_n_params(kind, S, A):
    return {"fact": A * S * (S - 1) + A * S * (A - 1), "tied": S * (A * S - 1)}[kind]


def iohmm_fit(X, S, kind, seed, A, max_iter=1000, tol=1e-9, val=None, check=10, max_seconds=None):
    """EM (archb.regimes._em_step) on A symbols from a random Dirichlet start; with val(T) -> score, the iterate with
    the best validation score (checked every `check` iterations and at the end) is kept. Returns (T, info)."""
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    T, q, Xj = jnp.asarray(iohmm_random(kind, S, A, rng)), jnp.full(S, 1.0 / S), jnp.asarray(X)
    prior, curve, best = R.PRIOR[kind](S), [], None
    for it in range(max_iter):
        Tn, qn, ll = R._em_step(T, q, Xj, kind, prior)
        curve.append(float(ll))
        if val is not None and it % check == 0:
            s = val(np.asarray(T))
            if best is None or s > best[0]:
                best = (s, np.asarray(T), it)
        if (it and curve[-1] - curve[-2] < tol) or (max_seconds and time.time() - t0 > max_seconds):
            break
        T, q = Tn, qn
    if val is not None:
        s = val(np.asarray(T))
        if s > best[0]:
            best = (s, np.asarray(T), len(curve))
        return best[1], {"em_iters": len(curve), "stop_iter": best[2], "train_ll_em": curve[-1]}
    return np.asarray(T), {"em_iters": len(curve), "train_ll_em": curve[-1]}


def markov_kernel(X, A, alpha=0.5, order=1):
    """Order 1: add-alpha transition counts within the paths; order 0: the add-alpha marginal (i.i.d.)."""
    X = np.asarray(X)
    if order == 0:
        p = np.bincount(X.ravel(), minlength=A) + alpha
        return np.tile(p / p.sum(), (A, 1))
    C = np.zeros((A, A)) + alpha
    np.add.at(C, (X[:, :-1].ravel(), X[:, 1:].ravel()), 1.0)
    return C / C.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ copula references
def dcc_R(Nz, a, b, Qbar):
    """Correlations R_t (T, d, d) of DCC(1, 1) on normal scores Nz (T, d): Q_t = (1 - a - b) Qbar + a n n' + b Q_{t-1},
    Q_0 = Qbar; R_t predicts day t from days < t."""
    T, d = Nz.shape
    out, Q = np.empty((T, d, d)), Qbar.copy()
    for t in range(T):
        s = 1 / np.sqrt(np.diag(Q))
        out[t] = Q * s[:, None] * s[None, :]
        Q = (1 - a - b) * Qbar + a * np.outer(Nz[t], Nz[t]) + b * Q
    return out


def copula_ll(Nz, Rt):
    """Mean Gaussian-copula log-density of normal scores Nz under correlations Rt."""
    sign, logdet = np.linalg.slogdet(Rt)
    quad = np.einsum("ti,tij,tj->t", Nz, np.linalg.inv(Rt), Nz) - np.sum(Nz**2, 1)
    return float(np.mean(-0.5 * logdet - 0.5 * quad))


def fit_dcc(Nz):
    """(a, b) maximizing the copula likelihood on Nz; a + b = 0.999 sigmoid(p0) split by sigmoid(p1)."""
    Qbar = np.corrcoef(Nz.T)

    def ab(p):
        pers = 0.999 / (1 + np.exp(-p[0]))
        sh = 1 / (1 + np.exp(-p[1]))
        return pers * sh, pers * (1 - sh)

    out = minimize(lambda p: -copula_ll(Nz, dcc_R(Nz, *ab(p), Qbar)), np.array([3.0, -3.0]), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 400})
    a, b = ab(out.x)
    return a, b, Qbar


# ------------------------------------------------------------------------------------------------ predictive statistics
def marginals(P, spec):
    """Per-coordinate predictive laws: list of d arrays (T, 2^k)."""
    d, nb = spec["d"], 2 ** spec["k"]
    Pr = P.reshape((len(P),) + (nb,) * d)
    return [Pr.sum(axis=tuple(1 + j for j in range(d) if j != c)) for c in range(d)]


def pred_corr(P, spec, a=0, b=1):
    """Predicted correlation of the bin values of coordinates a and b on each row (scale-free per coordinate)."""
    V = joint_values(spec)
    m = P @ V
    cov = P @ (V[:, a] * V[:, b]) - m[:, a] * m[:, b]
    va, vb = P @ V[:, a] ** 2 - m[:, a] ** 2, P @ V[:, b] ** 2 - m[:, b] ** 2
    return cov / np.sqrt(np.maximum(va * vb, 1e-300))


# ------------------------------------------------------------------------------------------------ real data
PAIRS = {"eurgbp": ("eurusd", "gbpusd"), "eurchf": ("eurusd", "usdchf"), "eurjpy": ("eurusd", "usdjpy"),
         "eurndq": ("eurusd", "nasdaq"), "fx4": ("eurusd", "gbpusd", "usdchf", "usdjpy")}


def real_dataset(name):
    """'<pair>-<grid>': daily log-returns of the series of PAIRS[pair] on their common quote days (FRED quotes as
    published: USD per EUR / GBP, CHF / JPY per USD), chronological 70/10/20 split, z = standardized on train per
    coordinate. 8 bits in all: 4 per coordinate for pairs, 2 for fx4. Grid 'q': per-coordinate quantile edges of the
    training z; 'v': per-coordinate EWMA scale s_t (lambda 0.94, from past bins only, as archb.realdata) and equal bins
    of [-4, 4] for y = z / s. Bin values = mean training y in each bin."""
    from archb import realdata as D

    base, grid = name.rsplit("-", 1)
    series = PAIRS[base]
    d = len(series)
    k = 8 // d
    nb = 2**k
    loaded = [D.load_prices(s) for s in series]
    common = loaded[0][0]
    for dt, _ in loaded[1:]:
        common = np.intersect1d(common, dt)
    prices = np.stack([p[np.searchsorted(dt, common)] for dt, p in loaded], axis=1)
    r = np.diff(np.log(prices), axis=0)
    n = len(r)
    a, b = D.split_points(n)
    mu, sd = r[:a].mean(0), r[:a].std(0)
    z = (r - mu) / sd
    if grid == "q":
        E = np.stack([D.quantile_edges(z[:a, c], nb) for c in range(d)])
        scale, bc = np.ones((n, d)), np.stack([D.discretize(z[:, c], E[c]) for c in range(d)], axis=1)
    else:
        e = D.uniform_edges(-4.0, 4.0, nb)
        E, cen = np.tile(e, (d, 1)), D.grid_centres(-4.0, 4.0, nb)
        scale, bc, st = np.empty((n, d)), np.empty((n, d), np.int64), np.ones(d)
        for t in range(n):
            scale[t] = st
            bc[t] = [D.discretize(z[t, c] / st[c], e) for c in range(d)]
            st = D.next_scale(st, bc[t], D.LAM, cen)
    y = z / scale
    values = np.stack([D.bin_values(y[:a, c], bc[:a, c], E[c]) for c in range(d)])
    joint = sum(bc[:, c] << (k * (d - 1 - c)) for c in range(d))
    return {"name": name, "series": series, "grid": grid, "d": d, "k": k, "dates": common[1:], "r": r, "z": z,
            "scale": scale, "bc": bc, "b": joint, "edges": E, "values": values, "spec": make_spec(values, E),
            "n_train": a, "n_val": b - a, "n_test": n - b, "mu": mu, "sd": sd,
            "corr_train": np.corrcoef(z[:a].T).tolist(), "lam": D.LAM if grid == "v" else 1.0}


def _margin(ds, c):
    """Univariate view of coordinate c for archb.families.fit_garch."""
    return {"z": ds["z"][:, c], "n_train": ds["n_train"], "scale": ds["scale"][:, c], "edges": ds["edges"][c],
            "values": ds["values"][c], "b": ds["bc"][:, c]}


def copula_refs(ds, starts=3):
    """GARCH(1,1)-t margins (continuous likelihood on train) joined by a Gaussian copula: 'ind' (identity), 'ccc'
    (training correlation of the normal scores), 'dcc' (DCC(1,1) fitted on the training scores). Returns
    {name: predictive matrix (n, K)} and the fitted parameters."""
    from archb import families as F

    n, d, a = len(ds["z"]), ds["d"], ds["n_train"]
    H, Nz, info = np.empty((n, d, ds["edges"].shape[1])), np.empty((n, d)), {"garch": []}
    for c in range(d):
        g, _ = F.fit_garch(_margin(ds, c), "t", "continuous", starts=starts)
        sdv = np.sqrt(g.sigma2(_margin(ds, c)))
        ed = ds["scale"][:, c, None] * ds["edges"][c][None, :]
        H[:, c] = norm.ppf(np.clip(g.cdf((ed - g.p["mu"]) / sdv[:, None]), 1e-16, 1 - 1e-16))
        H[:, c, 0], H[:, c, -1] = -np.inf, np.inf
        Nz[:, c] = norm.ppf(np.clip(g.cdf((ds["z"][:, c] - g.p["mu"]) / sdv), 1e-16, 1 - 1e-16))
        info["garch"].append(g.p)
    a_, b_, Qbar = fit_dcc(Nz[:a])
    Rd = dcc_R(Nz, a_, b_, Qbar)
    info.update(dcc_a=a_, dcc_b=b_, R_train=Qbar.tolist())
    Rs = {"ind": np.broadcast_to(np.eye(d), (n, d, d)), "ccc": np.broadcast_to(Qbar, (n, d, d)), "dcc": Rd}
    return {k: gauss_cells(Rm, H, 12) for k, Rm in Rs.items()}, {**info, "R_dcc": Rd}
