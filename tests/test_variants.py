"""Sequence 1 variants: the general simulator against archb.circuit and against a brute-force
dense-unitary simulation; parameter counts; stochasticity; init pitfalls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from archb import circuit, variants as V
from archb.ou import N_BINS

jax.config.update("jax_enable_x64", True)

EXPECTED_PARAMS = {"B0": 119, "B0-L4": 157, "B0-L5": 195, "inout": 194, "allpairs": 155, "quad": 269,
                   "anc1": 167, "anc2": 215}


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def dense_kernel(theta, variant, n_layers, n_bits=5):
    """Brute force: build every gate as a 2^n x 2^n matrix, apply to |b, 0>, trace all but outputs."""
    gates, n_anc = V.gate_list(variant, n_layers, n_bits)
    n = 2 * n_bits + n_anc
    D = 2**n
    idx = np.arange(D)
    bit = lambda q: (idx >> (n - 1 - q)) & 1  # noqa: E731
    U = np.eye(D)
    for k, g in enumerate(gates):
        qubits = g[1:-1]
        ctrl, tgt = qubits[:-1], qubits[-1]
        R = _ry(float(theta[k]))
        G = np.zeros((D, D))
        on = np.ones(D, bool)
        for c in ctrl:
            on &= bit(c) == 1
        for i in range(D):
            if not on[i]:
                G[i, i] = 1.0
                continue
            b = bit(tgt)[i]
            j = i ^ (1 << (n - 1 - tgt))  # partner with the target bit flipped
            G[i, i] += R[b, b]
            G[j, i] += R[1 - b, b]
        U = G @ U
    nb = 2**n_bits
    Q = np.zeros((nb, nb))
    for b in range(nb):
        psi = U[:, b << (n_bits + n_anc)]
        out = (idx >> n_anc) & (nb - 1)
        np.add.at(Q[b], out, psi**2)
    return Q


@pytest.mark.parametrize("name", list(V.LINEUP))
def test_param_counts(name):
    variant, L = V.LINEUP[name]
    assert V.n_params(variant, L) == EXPECTED_PARAMS[name]


@pytest.mark.parametrize("L", [3, 4])
def test_b0_matches_spec_circuit(L):
    theta = circuit.init_theta(3, noise=0.7, n_layers=L)
    np.testing.assert_allclose(V.label_kernel(theta, "B0", L), circuit.label_kernel(theta), atol=1e-12)


@pytest.mark.parametrize("variant", ["inout", "allpairs", "quad", "anc1"])
def test_matches_dense_bruteforce(variant):
    theta = V.init_theta(5, 0.8, variant, 1)  # one layer keeps the dense check fast
    np.testing.assert_allclose(V.label_kernel(theta, variant, 1), dense_kernel(np.asarray(theta), variant, 1),
                               atol=1e-12)


@pytest.mark.parametrize("name", list(V.LINEUP))
def test_rows_stochastic(name):
    variant, L = V.LINEUP[name]
    Q = np.asarray(V.label_kernel(V.init_theta(1, 1.0, variant, L), variant, L))
    assert Q.min() >= 0
    np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-12)


def test_ancilla_starts_in_plus_and_gates_have_gradient():
    """Pitfall 2: with the ancilla in |+>, the ancilla -> output gates get a non-zero gradient at init."""
    variant, L = "anc1", 3
    gates, _ = V.gate_list(variant, L)
    theta = V.init_theta(0, 0.0, variant, L)  # noise 0: copy gates + ancilla pi/2 only
    plus = [k for k, g in enumerate(gates) if g[-1] == "anc_plus"]
    assert np.allclose(np.asarray(theta)[plus], np.pi / 2)
    a_to_out = [k for k, g in enumerate(gates) if g[0] == "cry" and g[1] == 10 and g[2] >= 5]
    f = lambda th: jnp.sum(V.label_kernel(th, variant, L) * jnp.arange(32.0)[None, :])  # noqa: E731
    g = np.asarray(jax.grad(f)(theta + 0.3 * (jnp.arange(theta.shape[0]) % 2)))
    assert np.abs(g[a_to_out]).max() > 1e-3


def test_copy_init_is_identity_without_noise():
    for name in ("B0", "inout", "quad", "anc2"):
        variant, L = V.LINEUP[name]
        Q = np.asarray(V.label_kernel(V.init_theta(0, 0.0, variant, L), variant, L))
        np.testing.assert_allclose(Q, np.eye(N_BINS), atol=1e-12)


@pytest.mark.parametrize("variant", ["B0", "inout", "anc2"])
def test_other_register_sizes_match_dense(variant):
    """The n_bits code path (sequence 2 uses 8): checked on 3 bits, 2 layers, against the dense brute force."""
    theta = V.init_theta(2, 0.8, variant, 2, 3)
    np.testing.assert_allclose(V.label_kernel(theta, variant, 2, 3), dense_kernel(np.asarray(theta), variant, 2, 3),
                               atol=1e-12)


def test_anc2x_matches_dense():
    """Cross-coordinate entanglers (sequence 2 follow-up) on a 2 x 2-bit register, 1 layer."""
    theta = V.init_theta(4, 0.8, "anc2x", 1, 4)
    np.testing.assert_allclose(V.label_kernel(theta, "anc2x", 1, 4), dense_kernel(np.asarray(theta), "anc2x", 1, 4),
                               atol=1e-12)
