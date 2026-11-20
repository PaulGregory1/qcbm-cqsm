"""OU process, 5-bit grid and encodings, exact discretized kernel, fixed dataset.

Conventions: kernel rows/columns are BIN indices (decoded order);
P[i, j] = P(x_{t+1} in bin j | x_t in bin i). Encodings only change the bit string
the circuit sees for a bin: label_of_bin[i].
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

N_BITS = 5
N_BINS = 2**N_BITS  # 32
X_MIN, X_MAX = -3.0, 3.0
WIDTH = (X_MAX - X_MIN) / N_BINS  # 0.1875
EDGES = np.linspace(X_MIN, X_MAX, N_BINS + 1)  # u, 33 values
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])  # z, 32 values

N_TRAJ, N_STEPS = 2000, 50
DATA_SEED = 20261013
DATA_PATH = Path(__file__).resolve().parents[2] / "runs" / "data.npz"
ENCODINGS = ("binary", "gray")


# ----------------------------------------------------------------- parameters
@dataclass(frozen=True)
class OUParams:
    kappa: float = 1.0
    mu: float = 0.0
    sigma: float = 1.0
    dt: float = 0.1

    @property
    def c(self) -> float:
        return float(np.exp(-self.kappa * self.dt))

    @property
    def m(self) -> float:
        return self.mu * (1.0 - self.c)

    @property
    def s(self) -> float:
        return float(np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * self.dt))))

    @property
    def stat_sd(self) -> float:
        return self.sigma / np.sqrt(2 * self.kappa)


OU = OUParams()


# ------------------------------------------------------------------ encodings
def value_to_bin(x):
    """Map values to bin indices 0..31, clipping out-of-range values to the edge bins.

    Returns (bins, clipped_fraction)."""
    x = np.asarray(x, dtype=float)
    clipped = (x < X_MIN) | (x > X_MAX)
    bins = np.clip(np.floor((x - X_MIN) / WIDTH).astype(int), 0, N_BINS - 1)
    return bins, float(clipped.mean()) if x.size else 0.0


def int_to_bits(i, n_bits: int = N_BITS) -> np.ndarray:
    """Integer(s) -> bit array(s), MSB first; shape (..., n_bits)."""
    i = np.asarray(i, dtype=int)
    shifts = np.arange(n_bits - 1, -1, -1)
    return ((i[..., None] >> shifts) & 1).astype(np.int8)


def bits_to_int(bits) -> np.ndarray:
    """Bit array(s), MSB first, shape (..., n_bits) -> integer(s)."""
    bits = np.asarray(bits, dtype=int)
    weights = 1 << np.arange(bits.shape[-1] - 1, -1, -1)
    return bits @ weights


def binary_to_gray(b):
    """g1 = b1, gk = bk xor b(k-1): as integers, g = b ^ (b >> 1)."""
    b = np.asarray(b, dtype=int)
    return b ^ (b >> 1)


def gray_to_binary(g):
    g = np.asarray(g, dtype=int)
    b = g.copy()
    shift = g >> 1
    while np.any(shift):
        b ^= shift
        shift >>= 1
    return b


def encoding_perm(name: str):
    """Return (label_of_bin, bin_of_label) for encoding 'binary' or 'gray'.

    label_of_bin[i] is the bit-string integer the circuit sees for bin i;
    bin_of_label is its inverse permutation."""
    idx = np.arange(N_BINS)
    if name == "binary":
        label_of_bin = idx.copy()
    elif name == "gray":
        label_of_bin = binary_to_gray(idx)
    else:
        raise ValueError(f"unknown encoding {name!r}; expected one of {ENCODINGS}")
    bin_of_label = np.empty(N_BINS, dtype=int)
    bin_of_label[label_of_bin] = idx
    return label_of_bin, bin_of_label


# ------------------------------------------------------------------- kernels
def gaussian_bin_kernel(c: float, m: float, s: float) -> np.ndarray:
    """K[i, j] = Phi((u_{j+1} - c z_i - m)/s) - Phi((u_j - c z_i - m)/s), edge bins absorb tails.

    The discretization shared by P* and the fitted Gaussian AR(1) reference."""
    u = EDGES.copy()
    u[0], u[-1] = -np.inf, np.inf
    mean = c * CENTRES + m
    cdf = norm.cdf((u[None, :] - mean[:, None]) / s)
    return np.diff(cdf, axis=1)


def exact_kernel(p: OUParams = OU) -> np.ndarray:
    """P* = gaussian_bin_kernel(c, m, s) with the true OU one-step parameters."""
    return gaussian_bin_kernel(p.c, p.m, p.s)


def stationary(P: np.ndarray) -> np.ndarray:
    """Leading left eigenvector of a row-stochastic matrix, normalized to sum 1."""
    vals, vecs = np.linalg.eig(np.asarray(P).T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = v / v.sum()
    return np.clip(v, 0.0, None) / np.clip(v, 0.0, None).sum()


def counts_to_kernel(N: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Row-normalize (N + alpha); rows with no mass become uniform."""
    M = np.asarray(N, dtype=float) + alpha
    rs = M.sum(axis=1, keepdims=True)
    return np.where(rs > 0, M / np.where(rs > 0, rs, 1.0), 1.0 / M.shape[1])


# ---------------------------------------------------------------------- data
def simulate(p: OUParams = OU, n_traj: int = N_TRAJ, n_steps: int = N_STEPS, seed: int = DATA_SEED):
    """Trajectories x[n, t], t = 0..n_steps-1, x_0 from the stationary law, exact recursion."""
    rng = np.random.default_rng(seed)
    x = np.empty((n_traj, n_steps))
    x[:, 0] = p.mu + p.stat_sd * rng.standard_normal(n_traj)
    for t in range(1, n_steps):
        x[:, t] = p.c * x[:, t - 1] + p.m + p.s * rng.standard_normal(n_traj)
    return x


def make_dataset(p: OUParams = OU, n_traj: int = N_TRAJ, n_steps: int = N_STEPS, seed: int = DATA_SEED):
    """Return dict with N (32x32 pair counts, bin indices), pi_hat (empirical marginal of x_t
    over source states), clip_frac (over all simulated values), n_transitions."""
    x = simulate(p, n_traj, n_steps, seed)
    bins, clip_frac = value_to_bin(x)
    src, dst = bins[:, :-1].ravel(), bins[:, 1:].ravel()
    N = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    np.add.at(N, (src, dst), 1)
    pi_hat = N.sum(axis=1) / N.sum()
    return {"N": N, "pi_hat": pi_hat, "clip_frac": clip_frac, "n_transitions": int(N.sum()), "seed": seed}


def save_dataset(data: dict, path=DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return path


def load_dataset(path=DATA_PATH) -> dict:
    """Load runs/data.npz; generate and save it first if missing."""
    path = Path(path)
    if not path.exists():
        save_dataset(make_dataset(), path)
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    d["clip_frac"] = float(d["clip_frac"])
    d["n_transitions"] = int(d["n_transitions"])
    d["seed"] = int(d["seed"])
    return d


if __name__ == "__main__":
    data = make_dataset()
    path = save_dataset(data)
    P = exact_kernel()
    pi_star = stationary(P)
    Pe = counts_to_kernel(data["N"])
    row_tv = 0.5 * np.abs(Pe - P).sum(axis=1)
    print(f"c={OU.c:.4f} s={OU.s:.4f} stat_sd={OU.stat_sd:.4f} width={WIDTH}")
    print(f"saved {path}: transitions={data['n_transitions']} clip_frac={data['clip_frac']:.2e}")
    print(f"TV(pi_hat, pi*)={0.5 * np.abs(data['pi_hat'] - pi_star).sum():.4f}")
    print(f"weighted TV(empirical, P*)={np.sum(pi_star * row_tv):.4f}")
