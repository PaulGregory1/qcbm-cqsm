"""Sequence 6: compact classical kernels Q[x, x'] = P(bin x' | bin x) on a 32-bin grid, the classical counterparts of the
Architecture B circuits inside the classical-memory switch (archb.switch).

    mlp<H>  softmax over x' of W2 tanh(w1 v_x + b1) + b2: one input (the value v_x of bin x), H tanh units, 32 logits
            with the logit of bin 0 fixed at 0 (the softmax gauge): 2H + 31H + 31 = 33H + 31 free parameters
    mix<K>  sum_k pi_k(v) [Phi((e_{x'+1} - mu_k(v)) / sigma_k(v)) - Phi((e_{x'} - mu_k(v)) / sigma_k(v))], v = v_x, with
            pi = softmax(a_k + b_k v) (a_0 = b_0 = 0, the gauge), mu_k = c_k + d_k v, log sigma_k = f_k + g_k |v|:
            6K - 2 free parameters; the edge bins absorb the tails
    rbm<H>  conditional RBM on the Gray bits, as the circuit sees them: input bits u of bin x, output bits y of bin x'
            (5 each, MSB first, values 0 / 1), H binary hidden units,
                P(y | u) ∝ exp((b + A u) . y) prod_j (1 + exp(c_j + (B u)_j + (W^T y)_j)),
            normalized exactly over the 32 outputs (hidden units summed analytically): 5 + 25 + H + 5H + 5H = 11H + 30
            free parameters (b, A, c, B, W)

Sizes matched to the circuits (free parameters): Architecture B anc2 215 -> mlp6 229, mlp5 196, mix36 214, rbm17 217;
B0 119 -> mlp3 130, mix20 118, rbm8 118. A one-component mix with d = c_s, f = log sigma_s, g = 0 is the binned AR(1) kernel of a regime of the
sequence 4 truths (tested), so on msar / smsar the mixture is a well-specified kernel family, the MLP a neutral one.

A grid is (values (32,), edges (33,)) in the same units (z on the synthetic targets, training sd on real data); the outer
edges are set to +-1e4 (edge bins absorb the tails, finite so that gradients exist).
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jnorm

from archb.targets import gray_labels

jax.config.update("jax_enable_x64", True)

A = 32
BIG = 1e4
BITS = ((gray_labels(5, 1)[:, None] >> (4 - np.arange(5))[None, :]) & 1).astype(np.float64)  # (32 bins, 5), MSB first


def parse(kind):
    m = re.fullmatch(r"(mlp|mix|rbm)(\d+)", kind)
    if m is None:
        raise ValueError(f"not a classical kernel: {kind}")
    return m.group(1), int(m.group(2))


def is_classical(kind):
    return re.fullmatch(r"(mlp|mix|rbm)\d+", kind) is not None


def n_params(kind):
    fam, k = parse(kind)
    return {"mlp": 33 * k + 31, "mix": 6 * k - 2, "rbm": 11 * k + 30}[fam]


def make_grid(values, edges):
    e = np.asarray(edges, dtype=np.float64).copy()
    e[0], e[-1] = -BIG, BIG
    return np.asarray(values, dtype=np.float64), e


def _mlp(theta, H, v):
    w1, b1 = theta[:H], theta[H:2 * H]
    W2 = theta[2 * H:2 * H + H * (A - 1)].reshape(H, A - 1)
    b2 = theta[2 * H + H * (A - 1):]
    h = jnp.tanh(v[:, None] * w1[None, :] + b1[None, :])  # (32, H)
    logits = jnp.concatenate([jnp.zeros((A, 1)), h @ W2 + b2[None, :]], axis=1)
    return jax.nn.softmax(logits, axis=1)


def _mix(theta, K, v, e):
    a = jnp.concatenate([jnp.zeros(1), theta[:K - 1]])
    b = jnp.concatenate([jnp.zeros(1), theta[K - 1:2 * K - 2]])
    c, d, f, g = (theta[2 * K - 2 + j * K: 2 * K - 2 + (j + 1) * K] for j in range(4))
    pi = jax.nn.softmax(a[None, :] + b[None, :] * v[:, None], axis=1)  # (32, K)
    mu = c[None, :] + d[None, :] * v[:, None]
    sig = jnp.exp(f[None, :] + g[None, :] * jnp.abs(v)[:, None])
    cdf = jnorm.cdf((e[None, None, :] - mu[:, :, None]) / sig[:, :, None])  # (32, K, 33)
    return jnp.einsum("xk,xky->xy", pi, jnp.diff(cdf, axis=2))


def _rbm(theta, H):
    b, A_ = theta[:5], theta[5:30].reshape(5, 5)
    c, B = theta[30:30 + H], theta[30 + H:30 + 6 * H].reshape(H, 5)
    W = theta[30 + 6 * H:].reshape(5, H)
    U = jnp.asarray(BITS)  # input bits of every bin x, and output bits of every bin x' (same table)
    vis = (b[None, :] + U @ A_.T) @ U.T  # (x, x'): (b + A u_x) . y_x'
    hid = (c[None, :] + U @ B.T)[:, None, :] + (U @ W)[None, :, :]  # (x, x', H)
    logits = vis + jnp.sum(jax.nn.softplus(hid), axis=2)
    return jax.nn.softmax(logits, axis=1)


def kernel(theta, kind, grid=None):
    """Q (32, 32), rows = input bin x, columns = output bin x', in BIN order (the RBM ignores the grid)."""
    fam, k = parse(kind)
    theta = jnp.asarray(theta)
    if fam == "rbm":
        return _rbm(theta, k)
    v, e = jnp.asarray(grid[0]), jnp.asarray(grid[1])
    return _mlp(theta, k, v) if fam == "mlp" else _mix(theta, k, v, e)


def init_theta(kind, seed, grid, b_train):
    """Start near the i.i.d. law of the training bins: MLP with small random weights and output biases at the log
    training frequencies (add-1/2); mixture with equal weights, means at the training quantiles of the bin values, the
    training sd of the values, no dependence on v (plus small noise); RBM near the product of the bit marginals."""
    fam, k = parse(kind)
    rng = np.random.default_rng(seed)
    if fam == "rbm":  # visible biases at the training frequency of each output bit, small random couplings
        p1 = np.clip(BITS[np.asarray(b_train)].mean(0), 1e-3, 1 - 1e-3)
        return np.concatenate([np.log(p1 / (1 - p1)), 0.01 * rng.normal(size=25), np.zeros(k),
                               0.1 * rng.normal(size=5 * k), 0.1 * rng.normal(size=5 * k)])
    v = np.asarray(grid[0])[np.asarray(b_train)]
    if fam == "mlp":
        freq = np.bincount(b_train, minlength=A) + 0.5
        lf = np.log(freq / freq.sum())
        w1, b1 = rng.normal(0, 0.5, k), rng.normal(0, 0.5, k)
        W2 = rng.normal(0, 0.05, (k, A - 1))
        return np.concatenate([w1, b1, W2.ravel(), lf[1:] - lf[0]])
    q = np.quantile(v, (np.arange(k) + 0.5) / k)
    a, b = 0.01 * rng.normal(size=k - 1), 0.01 * rng.normal(size=k - 1)
    c = q + 0.05 * rng.normal(size=k)
    d = 0.01 * rng.normal(size=k)
    f = np.full(k, np.log(max(v.std(), 1e-3))) + 0.05 * rng.normal(size=k)
    g = 0.01 * rng.normal(size=k)
    return np.concatenate([a, b, c, d, f, g])
