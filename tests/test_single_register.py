import numpy as np
import pytest

from archb import single_register as sr
from archb import train as tr
from archb.ou import N_BINS, encoding_perm, stationary


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _op(k, g, n=5):
    """Gate g on qubit k (MSB first) of an n-qubit register."""
    out = np.eye(1)
    for j in range(n):
        out = np.kron(out, g if j == k else np.eye(2))
    return out


def _cop(c, t, g):
    P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    return _op(c, P0) + _op(c, P1) @ _op(t, g)


def _full_unitary(theta):
    """U(theta) as an explicit 32 x 32 matrix, gate by gate (independent of the simulator)."""
    U = np.eye(N_BINS)
    L = (len(theta) - 5) // 13
    for l in range(L):
        for r in range(5):
            U = _op(r, _ry(theta[13 * l + r])) @ U
        for r in range(4):
            U = _cop(r, r + 1, _ry(theta[13 * l + 5 + 2 * r])) @ U
            U = _cop(r + 1, r, _ry(theta[13 * l + 5 + 2 * r + 1])) @ U
    for r in range(5):
        U = _op(r, _ry(theta[13 * L + r])) @ U
    return U


def test_n_params():
    assert sr.N_PARAMS == 44 and sr.init_theta(0).shape == (44,)


def test_matches_explicit_unitary():
    theta = np.asarray(sr.init_theta(3, noise=1.0))
    U = _full_unitary(theta)
    assert np.allclose(U.T @ U, np.eye(N_BINS), atol=1e-12)
    # Q[b, b'] = |<b'|U|b>|^2 = U[b', b]^2
    assert np.abs(np.asarray(sr.label_kernel(theta)) - (U.T) ** 2).max() < 1e-12


@pytest.mark.parametrize("encoding", ["binary", "gray"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_doubly_stochastic(encoding, seed):
    P = np.asarray(sr.kernel(sr.init_theta(seed, noise=1.0), encoding))
    assert P.shape == (N_BINS, N_BINS) and np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(P.sum(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("encoding", ["binary", "gray"])
def test_stationary_law_is_uniform(encoding):
    for seed in range(3):
        P = np.asarray(sr.kernel(sr.init_theta(seed, noise=1.0), encoding))
        assert np.abs(stationary(P) - 1.0 / N_BINS).max() < 1e-10


def test_zero_theta_is_identity():
    for enc in ("binary", "gray"):
        assert np.abs(np.asarray(sr.kernel(np.zeros(44), enc)) - np.eye(N_BINS)).max() < 1e-14


def test_bin_space_permutation():
    theta = sr.init_theta(1, noise=1.0)
    Q = np.asarray(sr.label_kernel(theta))
    lob, _ = encoding_perm("gray")
    P = np.asarray(sr.kernel(theta, "gray"))
    assert P[5, 9] == Q[lob[5], lob[9]]


def test_train_hook_runs_and_stays_doubly_stochastic():
    res = tr.train("gray", seed=0, steps=20, log_every=0, model="single")
    assert res["theta"].shape == (44,) and res["settings"]["model"] == "single"
    assert np.all(np.isfinite(res["loss"])) and res["final_loss"] < res["init_loss"]
    P = res["kernel"]
    assert np.allclose(P.sum(axis=0), 1.0, atol=1e-12) and np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
