"""Sequence 2 targets: harder Markov kernels than the OU.

  arch    x' = c x + sqrt(a + b x^2) xi                           1D, 5 bits on [-4, 4]
          the row's spread depends on the input, not only its centre.
  merton  x' = c x + m + s xi + J (mu_J + s_J eta),  J ~ Bernoulli(p), m = -p mu_J (compensated)
                                                                  1D, 5 bits on [-4, 4]
          bimodal rows: no jump / downward jump.
  var2d   x' = A x + e,  e ~ N(0, Q)                              2D, 2 x 4 bits on [-L, L]^2
          rows are joint laws of two coupled, correlated variables.

Conventions as in archb.ou: kernels are row-stochastic K x K matrices in BIN space, bins of a 2D
state are flattened row-major (i = 16 i1 + i2), edge bins absorb the tails. The circuit sees a
bit label per bin: Gray code per coordinate, coordinates concatenated MSB first
(label = gray(i1) << 4 | gray(i2)). Exact kernels integrate the transition density over each bin
(closed form in 1D, Gauss-Legendre in x1 times a conditional normal CDF in x2 for var2d).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from archb.ou import binary_to_gray

DATA_DIR = Path(__file__).resolve().parents[2] / "runs" / "seq2" / "data"
N_TRAJ, N_STEPS, BURN_IN = 2000, 50, 200
GL_NODES = 24


# ------------------------------------------------------------------------------- grid helpers
def edges(lo, hi, n):
    return np.linspace(lo, hi, n + 1)


def centres(lo, hi, n):
    e = edges(lo, hi, n)
    return 0.5 * (e[:-1] + e[1:])


def bin_probs_gauss(mean, sd, e):
    """P(bin j) for N(mean, sd^2), mean/sd shape (R,), edges e (n+1,), edge bins absorb the tails."""
    u = e.copy()
    u[0], u[-1] = -np.inf, np.inf
    return np.diff(norm.cdf((u[None, :] - mean[:, None]) / sd[:, None]), axis=1)


def to_bins(x, lo, hi, n):
    """Bin index per value, out-of-range values clipped to the edge bins; returns (idx, clipped fraction)."""
    w = (hi - lo) / n
    idx = np.floor((np.asarray(x) - lo) / w).astype(int)
    clipped = float(np.mean((idx < 0) | (idx >= n)))
    return np.clip(idx, 0, n - 1), clipped


def gray_labels(n_bits_per_dim, d):
    """label_of_bin for the flattened state: Gray code per coordinate, concatenated MSB first."""
    n = 2**n_bits_per_dim
    g = binary_to_gray(np.arange(n))
    if d == 1:
        return g
    return (g[:, None] << n_bits_per_dim | g[None, :]).ravel()


# ------------------------------------------------------------------------------------ targets
@dataclass
class Target:
    name: str
    d: int
    bits: int  # per coordinate
    lo: float
    hi: float
    params: dict
    seed: int
    fit_x0: list = field(default_factory=list)  # starting point of the parametric fit
    family: str = ""  # process family (defaults to name); variants share a family and differ in grid or labels
    interleave: bool = False  # 2D only: interleave the two coordinates' Gray bits (g1_0 g2_0 g1_1 g2_1 ...)

    def __post_init__(self):
        if not self.family:
            self.family = self.name

    @property
    def n_per_dim(self):
        return 2**self.bits

    @property
    def n_bits(self):
        return self.d * self.bits

    @property
    def K(self):
        return 2**self.n_bits

    @property
    def z(self):
        return centres(self.lo, self.hi, self.n_per_dim)

    @property
    def e(self):
        return edges(self.lo, self.hi, self.n_per_dim)

    def state_values(self):
        """(K, d) decoded value of each flattened state."""
        z = self.z
        if self.d == 1:
            return z[:, None]
        Z1, Z2 = np.meshgrid(z, z, indexing="ij")
        return np.stack([Z1.ravel(), Z2.ravel()], axis=1)

    def label_of_bin(self):
        lob = gray_labels(self.bits, self.d)
        if self.d == 2 and self.interleave:
            b = self.bits
            g1, g2 = lob >> b, lob & (2**b - 1)
            out = np.zeros_like(lob)
            for k in range(b):  # bit k of each coordinate, MSB first (k = 0)
                s1, s2 = (g1 >> (b - 1 - k)) & 1, (g2 >> (b - 1 - k)) & 1
                out |= (s1 << (2 * b - 1 - 2 * k)) | (s2 << (2 * b - 2 - 2 * k))
            return out
        return lob

    # family kernel: subclasses below via `kernel_from`
    def exact_kernel(self):
        return kernel_from(self.family, self.params, self)

    def step(self, x, rng):
        return step_from(self.family, self.params, x, rng)


ARCH = Target("arch", 1, 5, -4.0, 4.0, {"c": 0.8, "a": 0.09, "b": 0.3}, seed=20261019,
              fit_x0=[0.7, 0.1, 0.2])
MERTON = Target("merton", 1, 5, -4.0, 4.0, {"c": 0.9, "s": 0.3, "p": 0.1, "mu_j": -1.0, "s_j": 0.3},
                seed=20261020, fit_x0=[0.85, 0.25, 0.05, -0.8, 0.2])
VAR2D = Target("var2d", 2, 4, -3.5, 3.5, {"A": [[0.8, 0.0], [0.3, 0.7]], "sd": [0.5, 0.5], "rho": 0.5},
               seed=20261021, fit_x0=[0.7, 0.0, 0.2, 0.6, 0.4, 0.4, 0.3])
TARGETS = {t.name: t for t in (ARCH, MERTON, VAR2D)}
# Sequence 2 follow-ups: the same VAR(1) on 2 x 3 bits, and both grids with the two
# coordinates' bits interleaved, so the nearest-neighbour output chain links bit k of x1 to bit k of x2.
# Same process and data seed as var2d: the 3-bit datasets bin the same trajectories more coarsely.
VAR2D3 = Target("var2d3", 2, 3, -3.5, 3.5, VAR2D.params, seed=VAR2D.seed, fit_x0=VAR2D.fit_x0, family="var2d")
VAR2D3I = Target("var2d3i", 2, 3, -3.5, 3.5, VAR2D.params, seed=VAR2D.seed, fit_x0=VAR2D.fit_x0, family="var2d",
                 interleave=True)
VAR2DI = Target("var2di", 2, 4, -3.5, 3.5, VAR2D.params, seed=VAR2D.seed, fit_x0=VAR2D.fit_x0, family="var2d",
                interleave=True)
FOLLOW_UPS = {t.name: t for t in (VAR2D3, VAR2D3I, VAR2DI)}


def kernel_from(name, p, T):
    z, e = T.z, T.e
    if name == "arch":
        sd = np.sqrt(p["a"] + p["b"] * z**2)
        return bin_probs_gauss(p["c"] * z, sd, e)
    if name == "merton":
        m = -p["p"] * p["mu_j"]
        base = p["c"] * z + m
        k0 = bin_probs_gauss(base, np.full_like(z, p["s"]), e)
        k1 = bin_probs_gauss(base + p["mu_j"], np.full_like(z, np.hypot(p["s"], p["s_j"])), e)
        return (1 - p["p"]) * k0 + p["p"] * k1
    if name == "var2d":
        return var2d_kernel(np.asarray(p["A"], float), np.asarray(p["sd"], float), float(p["rho"]), T)
    raise ValueError(name)


def var2d_kernel(A, sd, rho, T):
    """P[(i1,i2), (j1,j2)] = P(x'_1 in bin j1, x'_2 in bin j2 | x = (z_i1, z_i2)), Gaussian with mean A x and
    covariance Q (sd, rho). Integral over x'_1 in each bin by Gauss-Legendre (GL_NODES nodes on the bin,
    on [mean - 10 sd, edge] for the two edge bins), times the conditional normal CDF of x'_2 over each bin."""
    X = T.state_values()  # (K, 2)
    M = X @ A.T  # (K, 2)
    s1, s2 = sd
    cs = s2 * np.sqrt(1 - rho**2)  # conditional sd of x2 given x1
    e = T.e
    n = T.n_per_dim
    xg, wg = np.polynomial.legendre.leggauss(GL_NODES)
    u2 = e.copy()
    u2[0], u2[-1] = -np.inf, np.inf
    P = np.zeros((T.K, n, n))
    m1, m2 = M[:, 0][:, None], M[:, 1][:, None]  # (K, 1)
    for j1 in range(n):
        a = np.full((T.K, 1), e[j1]) if j1 > 0 else m1 - 10 * s1
        b = np.full((T.K, 1), e[j1 + 1]) if j1 < n - 1 else m1 + 10 * s1
        b = np.maximum(b, a)
        x1 = 0.5 * (b - a) * xg[None, :] + 0.5 * (b + a)  # (K, nodes)
        w = 0.5 * (b - a) * wg[None, :] * norm.pdf(x1, m1, s1)
        cm = m2 + rho * s2 / s1 * (x1 - m1)
        cdf = norm.cdf((u2[None, None, :] - cm[:, :, None]) / cs)  # (K, nodes, n+1)
        P[:, j1, :] = np.einsum("kq,kqj->kj", w, np.diff(cdf, axis=2))
    P = P.reshape(T.K, T.K)
    return P / P.sum(axis=1, keepdims=True)  # quadrature error is ~1e-12; renormalize rows


def step_from(name, p, x, rng):
    if name == "arch":
        return p["c"] * x + np.sqrt(p["a"] + p["b"] * x**2) * rng.standard_normal(x.shape)
    if name == "merton":
        m = -p["p"] * p["mu_j"]
        jump = rng.random(x.shape) < p["p"]
        return (p["c"] * x + m + p["s"] * rng.standard_normal(x.shape)
                + jump * (p["mu_j"] + p["s_j"] * rng.standard_normal(x.shape)))
    if name == "var2d":
        A = np.asarray(p["A"], float)
        s1, s2 = p["sd"]
        L = np.array([[s1, 0.0], [p["rho"] * s2, s2 * np.sqrt(1 - p["rho"] ** 2)]])
        return x @ A.T + rng.standard_normal(x.shape) @ L.T
    raise ValueError(name)


# --------------------------------------------------------------------------------------- data
def make_dataset(T: Target, n_traj=N_TRAJ, n_steps=N_STEPS):
    """Trajectories started at 0, BURN_IN steps discarded, then n_steps kept; returns pair counts N (K x K, bin
    space), pi_hat (row marginal), clipped fraction, n_transitions."""
    rng = np.random.default_rng(T.seed)
    x = np.zeros((n_traj, T.d))
    for _ in range(BURN_IN):
        x = T.step(x, rng)
    xs = np.empty((n_steps, n_traj, T.d))
    for t in range(n_steps):
        xs[t] = x
        x = T.step(x, rng)
    idx, clips = [], []
    for k in range(T.d):
        b, c = to_bins(xs[..., k], T.lo, T.hi, T.n_per_dim)
        idx.append(b)
        clips.append(c)
    flat = idx[0] if T.d == 1 else idx[0] * T.n_per_dim + idx[1]  # (n_steps, n_traj)
    N = np.zeros((T.K, T.K), dtype=np.int64)
    np.add.at(N, (flat[:-1].ravel(), flat[1:].ravel()), 1)
    return {"N": N, "pi_hat": N.sum(axis=1) / N.sum(), "clip_frac": float(np.mean(clips)),
            "n_transitions": int(N.sum())}


def load_dataset(T: Target):
    path = DATA_DIR / f"{T.name}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **make_dataset(T))
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------------- references
def _unpack(name, x):
    if name == "arch":
        return {"c": x[0], "a": abs(x[1]) + 1e-6, "b": abs(x[2])}
    if name == "merton":
        return {"c": x[0], "s": abs(x[1]) + 1e-6, "p": 1 / (1 + np.exp(-x[2])), "mu_j": x[3], "s_j": abs(x[4]) + 1e-6}
    if name == "var2d":
        return {"A": [[x[0], x[1]], [x[2], x[3]]], "sd": [abs(x[4]) + 1e-6, abs(x[5]) + 1e-6], "rho": np.tanh(x[6])}
    raise ValueError(name)


def fit_parametric(T: Target, data):
    """Maximum likelihood of the target's own parametric family on the binned pair counts (the strong classical
    reference: the true model class, fitted to the same data the circuits see)."""
    N = np.asarray(data["N"], float)
    x0 = np.array(T.fit_x0, float)
    if T.family == "merton":
        x0[2] = np.log(x0[2] / (1 - x0[2]))
    if T.family == "var2d":
        x0[6] = np.arctanh(x0[6])

    def nll(x):
        K = kernel_from(T.family, _unpack(T.family, x), T)
        return -np.sum(N * np.log(np.maximum(K, 1e-300))) / N.sum()

    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10})
    return _unpack(T.family, res.x), float(res.fun)


def reference_kernels(T: Target, data):
    N = np.asarray(data["N"], float)
    pi_hat = np.asarray(data["pi_hat"], float)
    fitted, _ = fit_parametric(T, data)
    M = N + 0.1
    return {"exact": T.exact_kernel(), "parametric": kernel_from(T.family, fitted, T),
            "tabular": M / M.sum(axis=1, keepdims=True),
            "iid": np.tile(pi_hat / pi_hat.sum(), (T.K, 1))}, fitted


# ------------------------------------------------------------------------------------ metrics
def stationary(P):
    vals, vecs = np.linalg.eig(np.asarray(P).T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = np.clip(v / v.sum(), 0, None)
    return v / v.sum()


def _tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


ACF_LAGS = np.arange(1, 21)
TIGHT = {"weighted_tv": 0.05, "stationary_tv": 0.02, "lag1": 0.02, "cond_mean_rmse": 0.05, "cond_sd_rel": 0.10}


def metrics(P, T: Target, P_star=None):
    """Kernel metrics against P* (all exact functions of the kernels):
    weighted_tv, tail_tv (rows with a coordinate beyond 2 stationary sd of P*), stationary_tv,
    per coordinate k: lag1_k / lag1_exact_k / acf_err_k (lags 1-20, chain of P under its own stationary law),
    cond_mean_rmse_k (pi*-weighted RMS error of E[x'_k | x], in stationary sd of P*),
    cond_sd_rel_k (pi*-weighted mean relative error of sd[x'_k | x]),
    and for 2D: corr_err (stationary correlation of x_1, x_2, model minus exact)."""
    P_star = T.exact_kernel() if P_star is None else P_star
    P = np.asarray(P, float)
    pi_s, pi_m = stationary(P_star), stationary(P)
    X = T.state_values()
    rtv = _tv(P, P_star)
    out = {"weighted_tv": float(pi_s @ rtv), "stationary_tv": float(_tv(pi_m, pi_s))}
    mean_s = pi_s @ X
    sd_s = np.sqrt(pi_s @ X**2 - mean_s**2)
    tail = np.any(np.abs(X - mean_s) > 2 * sd_s, axis=1)
    out["tail_tv"] = float(rtv[tail].mean())
    Pk_m, Pk_s = np.eye(T.K), np.eye(T.K)
    acf_m, acf_s = [], []
    for _ in ACF_LAGS:
        Pk_m, Pk_s = Pk_m @ P, Pk_s @ P_star
        acf_m.append(_acf_from(Pk_m, pi_m, X))
        acf_s.append(_acf_from(Pk_s, pi_s, X))
    acf_m, acf_s = np.array(acf_m), np.array(acf_s)  # (lags, d)
    for k in range(T.d):
        out[f"lag1_{k}"] = float(acf_m[0, k])
        out[f"lag1_exact_{k}"] = float(acf_s[0, k])
        out[f"acf_err_{k}"] = float(np.max(np.abs(acf_m[:, k] - acf_s[:, k])))
        cm_m, cm_s = P @ X[:, k], P_star @ X[:, k]
        sd_m = np.sqrt(np.clip(P @ X[:, k] ** 2 - cm_m**2, 0, None))
        sd_e = np.sqrt(np.clip(P_star @ X[:, k] ** 2 - cm_s**2, 0, None))
        out[f"cond_mean_rmse_{k}"] = float(np.sqrt(pi_s @ (cm_m - cm_s) ** 2) / sd_s[k])
        out[f"cond_sd_rel_{k}"] = float(pi_s @ (np.abs(sd_m - sd_e) / sd_e))
    if T.d == 2:
        out["corr_err"] = float(_corr(pi_m, X) - _corr(pi_s, X))
    return out


def _acf_from(Pk, pi, X):
    mean = pi @ X
    var = pi @ X**2 - mean**2
    return (pi @ (X * (Pk @ X)) - mean**2) / var


def _corr(pi, X):
    m = pi @ X
    c = pi @ (X[:, 0] * X[:, 1]) - m[0] * m[1]
    v = pi @ X**2 - m**2
    return c / np.sqrt(v[0] * v[1])


def tight_pass(m, d):
    ok = m["weighted_tv"] <= TIGHT["weighted_tv"] and m["stationary_tv"] <= TIGHT["stationary_tv"]
    for k in range(d):
        ok &= abs(m[f"lag1_{k}"] - m[f"lag1_exact_{k}"]) <= TIGHT["lag1"]
        ok &= m[f"cond_mean_rmse_{k}"] <= TIGHT["cond_mean_rmse"] and m[f"cond_sd_rel_{k}"] <= TIGHT["cond_sd_rel"]
    return bool(ok)
