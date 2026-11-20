import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from archb.circuit import (
    LAYER_SIZE, N_LAYERS, N_OUT, N_PARAMS, idx_chain, idx_final, idx_inout, idx_local, init_theta, kernel,
    label_kernel, unpack,
)
from archb.ou import N_BINS, encoding_perm


def test_n_params_and_layout():
    assert N_PARAMS == 119
    idx = [idx_local(l, r) for l in range(3) for r in range(5)]
    idx += [idx_inout(l, q, r) for l in range(3) for q in range(5) for r in range(5)]
    idx += [idx_chain(l, r, d) for l in range(3) for r in range(4) for d in range(2)]
    idx += [idx_final(r) for r in range(5)]
    assert sorted(idx) == list(range(N_PARAMS))
    theta = np.arange(N_PARAMS, dtype=float)
    p = unpack(theta)
    assert p["inout"][1, 2, 3] == idx_inout(1, 2, 3)
    assert p["chain"][2, 3, 1] == idx_chain(2, 3, 1)
    assert p["local"][0, 4] == idx_local(0, 4) and p["final"][2] == idx_final(2)


def test_rows_sum_to_one():
    for seed in range(3):
        Q = np.asarray(label_kernel(init_theta(seed, noise=1.0)))
        assert Q.shape == (32, 32) and np.all(Q >= 0)
        assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("encoding", ["binary", "gray"])
def test_copy_init_is_identity(encoding):
    theta = init_theta(0, noise=0.0)
    assert np.abs(np.asarray(label_kernel(theta)) - np.eye(N_BINS)).max() < 1e-14
    assert np.abs(np.asarray(kernel(theta, encoding)) - np.eye(N_BINS)).max() < 1e-14


def test_copy_init_with_noise_near_identity():
    diags = [np.diag(np.asarray(label_kernel(init_theta(s)))).mean() for s in range(5)]
    print(f"\nmean diagonal at copy init, noise 0.2 (seeds 0-4): {np.round(diags, 3)}")
    assert min(diags) > 0.5


def test_bin_space_permutation():
    theta = init_theta(7)
    Q = np.asarray(label_kernel(theta))
    lob, bol = encoding_perm("gray")
    P = np.asarray(kernel(theta, "gray"))
    assert np.allclose(P[3, 5], Q[lob[3], lob[5]])
    assert np.allclose(P[np.ix_(bol, bol)], Q)
    assert np.allclose(np.asarray(kernel(theta, "binary")), Q)


# ------------------------------------------------- brute-force 10-qubit reference
def _ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])


def _op(k, g, n=10):
    """Gate g on qubit k of an n-qubit register (qubit 0 = most significant)."""
    return np.kron(np.kron(np.eye(2**k), g), np.eye(2 ** (n - k - 1)))


def _cop(c, t, g):
    p0, p1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    return _op(c, p0) + _op(c, p1) @ _op(t, g)


def _full_unitary(theta):
    """Qubits 0..4 = input register, 5..9 = output register; every gate applied explicitly."""
    out = lambda r: 5 + r  # noqa: E731
    gates = []
    for l in range(N_LAYERS):
        gates += [_op(out(r), _ry(theta[idx_local(l, r)])) for r in range(N_OUT)]
        gates += [_cop(q, out(r), _ry(theta[idx_inout(l, q, r)])) for q in range(5) for r in range(5)]
        for r in range(4):
            gates.append(_cop(out(r), out(r + 1), _ry(theta[idx_chain(l, r, 0)])))
            gates.append(_cop(out(r + 1), out(r), _ry(theta[idx_chain(l, r, 1)])))
    gates += [_op(out(r), _ry(theta[idx_final(r)])) for r in range(N_OUT)]
    U = np.eye(1024)
    for g in gates:
        U = g @ U
    return U


def test_brute_force_10_qubit_unitary():
    theta = np.asarray(jax.random.uniform(jax.random.PRNGKey(3), (N_PARAMS,), minval=-np.pi, maxval=np.pi))
    U = _full_unitary(theta)
    assert np.allclose(U.T @ U, np.eye(1024), atol=1e-12)
    P_ref = np.empty((N_BINS, N_BINS))
    for b in range(N_BINS):
        amp = U[:, b * 32].reshape(32, 32)  # [c, b'] = <c, b'|U|b, 0>
        P_ref[b] = (amp**2).sum(axis=0)
    # input register untouched: all weight on c = b
    assert np.allclose([(U[b * 32 : b * 32 + 32, b * 32] ** 2).sum() for b in range(N_BINS)], 1.0)
    assert np.abs(np.asarray(label_kernel(theta)) - P_ref).max() < 1e-10


def test_all_zero_theta_stationary_output():
    Q = np.asarray(label_kernel(jnp.zeros(N_PARAMS)))
    assert np.allclose(Q[:, 0], 1.0) and np.allclose(Q[:, 1:], 0.0)


# ------------------------------------------------------------------- gradients
def _loss(theta, W):
    return jnp.sum(kernel(theta, "binary") * W)


def test_grad_finite_nonzero_at_copy_init():
    W = jnp.asarray(np.random.default_rng(0).standard_normal((N_BINS, N_BINS)))
    g = jax.grad(_loss)(init_theta(0), W)
    assert np.all(np.isfinite(g)) and np.abs(g).max() > 1e-3
    # finite-difference check on a few angles
    theta = init_theta(0)
    for k in (idx_local(0, 2), idx_inout(0, 1, 1), idx_chain(1, 2, 0), idx_final(4)):
        e = jnp.zeros(N_PARAMS).at[k].set(1e-6)
        fd = (_loss(theta + e, W) - _loss(theta - e, W)) / 2e-6
        assert abs(fd - g[k]) < 1e-6


def test_noiseless_copy_init_is_a_critical_point():
    """Every P_ij is 0 or 1 there (its min or max), so every smooth loss has zero gradient."""
    W = jnp.asarray(np.random.default_rng(0).standard_normal((N_BINS, N_BINS)))
    g = jax.grad(_loss)(init_theta(0, noise=0.0), W)
    assert np.abs(g).max() < 1e-12


def test_timings():
    theta = init_theta(0)
    W = jnp.ones((N_BINS, N_BINS))
    fwd = jax.jit(lambda t: kernel(t, "gray"))
    vg = jax.jit(jax.value_and_grad(lambda t: jnp.sum(kernel(t, "gray") * W)))
    fwd(theta).block_until_ready()
    jax.block_until_ready(vg(theta))
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        fwd(theta).block_until_ready()
    t1 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(vg(theta))
    t2 = time.perf_counter()
    print(f"\njitted forward {1e3 * (t1 - t0) / n:.3f} ms, value_and_grad {1e3 * (t2 - t1) / n:.3f} ms")


def test_extra_layer_with_zero_angles_is_unchanged():
    """n_layers is read from the vector length; an appended all-zero layer acts as the identity."""
    th3 = init_theta(3, 0.2)
    th4 = init_theta(3, 0.2, n_layers=4)
    assert th4.shape == (4 * LAYER_SIZE + 5,)
    padded = jnp.concatenate([th3[: 3 * LAYER_SIZE], jnp.zeros(LAYER_SIZE), th3[3 * LAYER_SIZE :]])
    np.testing.assert_allclose(label_kernel(padded), label_kernel(th3), atol=1e-12)
