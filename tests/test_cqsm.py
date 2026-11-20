"""Sequence 4, step 0: the conditional QSM (archb.cqsm) and the input-output representation (archb.iolin)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from archb import cqsm, iolin, qseq, variants
from archb.targets import gray_labels

jax.config.update("jax_enable_x64", True)

SHAPES = [(1, 0), (1, 2), (2, 0), (2, 1)]


@pytest.mark.parametrize("m,a", SHAPES)
def test_kraus_complete(m, a):
    K = np.asarray(cqsm.kraus(jnp.asarray(cqsm.init_theta(1, m, a, noise=0.7)), m, a))
    assert K.shape == (32, 32, 2**a, 2**m, 2**m)
    assert np.abs(np.einsum("xycij,xycik->xjk", K.conj(), K) - np.eye(2**m)).max() < 1e-12


@pytest.mark.parametrize("m,a", SHAPES)
def test_words_normalized_and_stationary(m, a):
    lin = cqsm.io(cqsm.init_theta(2, m, a, n_bits=2, noise=0.8), m, a, n_bits=2)
    pi = lin.marginal()
    assert abs(pi.sum() - 1) < 1e-12 and pi.min() > 0
    for x in range(4):  # the joint law is stationary
        assert np.abs(sum(lin.u[w] @ lin.M[w, x] for w in range(4)) - lin.u[x]).max() < 1e-8
    P3 = np.array([np.real(lin.u[w[0]] @ lin.M[w[0], w[1]] @ lin.M[w[1], w[2]] @ lin.f)
                   for w in itertools.product(range(4), repeat=3)])
    assert abs(P3.sum() - 1) < 1e-12 and P3.min() > -1e-14
    assert np.abs(P3.reshape(16, 4).sum(1) - P3.reshape(4, 16).sum(0)).max() < 1e-8  # shift invariance


@pytest.mark.parametrize("a,variant", [(0, "B0"), (1, "anc1"), (2, "anc2")])
@pytest.mark.parametrize("m", [1, 2])
def test_reduces_to_architecture_b(m, a, variant):
    thb = np.asarray(variants.init_theta(3, 0.4, variant, 3))
    th = cqsm.embed(thb, 0, m, a)
    th[cqsm.mem_mask(m, a)] = 0.0
    K = np.asarray(cqsm.kraus(jnp.asarray(th), m, a))
    eye = np.einsum("xyc,ij->xycij", K[..., 0, 0], np.eye(2**m))
    assert np.abs(K - eye).max() < 1e-13  # K = amplitude x identity on the memory
    P = np.asarray(variants.kernel(thb, gray_labels(5, 1), variant, 3))
    assert np.abs((np.abs(K[..., 0, 0]) ** 2).sum(2) - P).max() < 1e-12
    xs = iolin.IOLin.from_kernel(P).sample(3000, 0)
    lp_m = cqsm.io(th, m, a).logprobs(xs)
    lp_b = iolin.IOLin.from_kernel(P).logprobs(xs)
    assert np.abs(lp_m[1:] - lp_b[1:]).max() < 1e-12 and abs(lp_m[0] - lp_b[0]) < 1e-6


@pytest.mark.parametrize("m", [1, 2])
def test_reduces_to_qsm(m):
    th = cqsm.init_theta(4, m, 0, noise=0.8)
    th[cqsm.input_mask(m, 0)] = 0.0
    K = np.asarray(cqsm.kraus(jnp.asarray(th), m, 0))
    assert np.abs(K - K[:1]).max() < 1e-14  # the channel ignores x
    lin = cqsm.io(th, m, 0)
    xs = lin.sample(4000, 1)
    lp_q = np.asarray(qseq.logprobs(jnp.asarray(K[0, :, 0]), jnp.asarray(xs)))
    assert np.abs(lin.logprobs(xs) - lp_q).max() < 1e-7
    assert np.abs(np.asarray(cqsm.qsm_transfer(K[0, :, 0])) - lin.M).max() < 1e-14


def _dense_kraus(theta, m, a, L, n):
    """Gate by gate on the full register (inputs included), dense; K[x, x', c, i, j] in label order."""
    nq = 2 * n + m + a
    gates = cqsm.gate_list(m, a, L, n)

    def op(U, t):
        return np.kron(np.kron(np.eye(2**t), U), np.eye(2 ** (nq - t - 1)))

    def ry(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])

    def rz(t):
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

    V = np.eye(2**nq, dtype=complex)
    P1 = np.diag([0.0, 1.0])
    for th, (kind, c, t, _) in zip(theta, gates):
        U = ry(th) if kind == "ry" else rz(th)
        G = op(U, t) if c is None else op(np.diag([1.0, 0.0]), c) + op(P1, c) @ op(U, t)
        V = G @ V
    D, A, C = 2**m, 2**n, 2**a
    K = np.zeros((A, A, C, D, D), dtype=complex)
    for x in range(A):
        for j in range(D):
            col = V[:, ((x * A) * D + j) * C]  # |x>_in |0>_out |j>_mem |0>_anc
            amp = col.reshape(A, A, D, C)
            assert np.abs(np.delete(amp, x, axis=0)).max() < 1e-13  # inputs are never changed
            K[x, :, :, :, j] = amp[x].transpose(0, 2, 1)
    return K


@pytest.mark.parametrize("m,a,L", [(1, 1, 2), (2, 0, 2), (2, 1, 1)])
def test_matches_dense_brute_force(m, a, L):
    th = cqsm.init_theta(6, m, a, L, n_bits=2, noise=1.0)
    K = np.asarray(cqsm.label_kraus(jnp.asarray(th), m, a, L, 2))
    assert np.abs(K - _dense_kraus(th, m, a, L, 2)).max() < 1e-12


def test_monte_carlo_trajectories():
    """Sampled trajectories with explicit measurement and collapse of a pure memory state (dense Kraus operators,
    an independent code path) reproduce the exact law of words (x_0, x_1, x_2)."""
    m, a, L, n = 1, 1, 2, 2
    th = cqsm.init_theta(7, m, a, L, n_bits=n, noise=1.0)
    lob = gray_labels(n, 1)
    K = _dense_kraus(th, m, a, L, n)[lob][:, lob]
    lin = cqsm.io(th, m, a, L, n)
    exact = np.array([np.real(lin.u[w[0]] @ lin.M[w[0], w[1]] @ lin.M[w[1], w[2]] @ lin.f)
                      for w in itertools.product(range(4), repeat=3)])
    rng, Ntr, D = np.random.default_rng(0), 400000, 2
    pi = lin.marginal()
    x = rng.choice(4, size=Ntr, p=pi / pi.sum())
    psi = np.zeros((Ntr, D), complex)
    for v in range(4):  # pure states drawn from the eigen-decomposition of the memory state given x_0
        sel = np.flatnonzero(x == v)
        w, U = np.linalg.eigh(cqsm.rho_of(lin.u[v], m) / pi[v])
        k = rng.choice(D, size=len(sel), p=np.clip(w, 0, None) / np.clip(w, 0, None).sum())
        psi[sel] = U[:, k].T
    words = x.copy()
    for _ in range(2):
        amp = np.einsum("nycij,nj->nyci", K[x], psi).reshape(Ntr, -1, D)  # outcome (x', c)
        p = (np.abs(amp) ** 2).sum(2)
        j = (np.cumsum(p, 1) < rng.random(Ntr)[:, None]).sum(1)
        psi = amp[np.arange(Ntr), j] / np.sqrt(p[np.arange(Ntr), j])[:, None]
        x = j // 2**a
        words = words * 4 + x
    emp = np.bincount(words, minlength=64) / Ntr
    assert np.abs(emp - exact).max() < 5 * np.sqrt(exact.max() / Ntr)


def test_staged_init_has_memory_gradient():
    """At the stage 2 start the memory gates have non-zero gradients (memory in |+>-like states)."""
    m, a = 1, 0
    thb = np.asarray(variants.init_theta(3, 0.2, "B0", 3))
    th = cqsm.embed(thb, 0, m, a, noise=0.0)
    xs = iolin.IOLin.from_kernel(np.asarray(variants.kernel(thb, gray_labels(5, 1), "B0", 3))).sample(400, 2)
    g = jax.grad(lambda t: cqsm.loglik(t, jnp.asarray(xs.reshape(8, 50)), m, a))(jnp.asarray(th))
    assert np.abs(np.asarray(g)[cqsm.mem_mask(m, a)]).max() > 1e-4


def test_real_coordinates_exact():
    """The real transfer matrices are a similarity transform of the complex superoperators: same word laws."""
    m, a = 2, 1
    K = np.asarray(cqsm.kraus(jnp.asarray(cqsm.init_theta(8, m, a, n_bits=2, noise=0.9)), m, a, n_bits=2))
    Mc = np.einsum("xycij,xyckl->xyjlik", K, K.conj()).reshape(4, 4, 16, 16)
    lc, lr = iolin.IOLin(Mc, np.eye(4).ravel()), iolin.IOLin(np.asarray(cqsm.transfer_from_kraus(K)), cqsm.f_vec(m))
    assert lr.M.dtype == np.float64 and np.abs(lc.marginal() - lr.marginal()).max() < 1e-10
    xs = lc.sample(500, 3)
    assert np.abs(lc.logprobs(xs) - lr.logprobs(xs)).max() < 1e-10
    rho = cqsm.rho_of(lr.u[0], m)
    assert np.abs(rho - rho.conj().T).max() < 1e-14 and np.abs(rho - lc.u[0].reshape(4, 4)).max() < 1e-8
