"""Sequence 6: compact classical kernels (archb.ckernels) and the classical switch built from them."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from archb import ckernels, iolin, regimes as R, switch

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
GRID = ckernels.make_grid(R.Z, R.E)


@pytest.mark.parametrize("kind,n", [("mlp6", 229), ("mlp5", 196), ("mlp3", 130), ("mix36", 214), ("mix20", 118),
                                    ("rbm17", 217), ("rbm8", 118)])
def test_param_counts_and_rows(kind, n):
    assert ckernels.n_params(kind) == n == switch.n_kernel(kind)
    b = np.random.default_rng(0).integers(0, 32, 500)
    th = ckernels.init_theta(kind, 1, GRID, b)
    assert len(th) == n
    th = th + 0.3 * np.random.default_rng(2).normal(size=n)
    Q = np.asarray(ckernels.kernel(th, kind, GRID))
    assert Q.shape == (32, 32) and Q.min() >= 0 and np.abs(Q.sum(1) - 1).max() < 1e-12


def test_rbm_against_brute_force():
    """P(y | u) of the conditional RBM against a sum over every hidden configuration, on the Gray bits of the bins."""
    import itertools

    from archb.targets import gray_labels

    H = 3
    th = np.random.default_rng(4).normal(size=ckernels.n_params(f"rbm{H}"))
    Q = np.asarray(ckernels.kernel(th, f"rbm{H}"))
    b, A_ = th[:5], th[5:30].reshape(5, 5)
    c, B, W = th[30:30 + H], th[30 + H:30 + 6 * H].reshape(H, 5), th[30 + 6 * H:].reshape(5, H)
    bits = lambda i: np.array([(gray_labels(5, 1)[i] >> (4 - r)) & 1 for r in range(5)], float)
    for x in (0, 7, 19, 31):
        u = bits(x)
        w = np.array([sum(np.exp((b + A_ @ u) @ bits(y) + (c + B @ u) @ np.array(h) + bits(y) @ W @ np.array(h))
                          for h in itertools.product((0, 1), repeat=H)) for y in range(32)])
        assert np.abs(Q[x] - w / w.sum()).max() < 1e-12


def test_init_is_near_iid():
    b = np.random.default_rng(0).integers(0, 32, 5000)
    th = ckernels.init_theta("mlp6", 0, GRID, b)
    th[:12] = 0.0  # no hidden activity: rows equal the softmax of the output biases
    f = (np.bincount(b, minlength=32) + 0.5) / (5000 + 16)
    assert np.abs(np.asarray(ckernels.kernel(th, "mlp6", GRID)) - f[None, :]).max() < 1e-12


def test_one_component_mixture_is_the_ar_kernel():
    """mix1 with mu = c v, log sigma = log s is the binned AR(1) kernel of a regime of the sequence 4 truths."""
    for c, s in zip(R.MSAR["c"], R.MSAR["sig"]):
        th = np.array([0.0, c, np.log(s), 0.0])  # K = 1: no weight parameters; c_0, d_0, f_0, g_0
        Q = np.asarray(ckernels.kernel(th, "mix1", GRID))
        ref = np.asarray(R.ar_kernels([c], [s]))[0]
        assert np.abs(Q - ref).max() < 1e-12


def test_classical_switch():
    X = np.random.default_rng(1).integers(0, 32, size=(4, 30))
    b = X.ravel()
    k0 = ckernels.init_theta("mix20", 0, GRID, b)
    th1 = switch.init_switch(k0, 1, 0, "mix20", noise=0.0)
    Q = np.asarray(ckernels.kernel(k0, "mix20", GRID))
    M1 = np.asarray(switch.switch_M(th1, 1, "mix20", grid=GRID))
    assert np.abs(M1[:, :, 0, 0] - Q).max() < 1e-12
    # S = 1 is the Markov chain with that kernel, first value from its stationary law
    w, V = np.linalg.eig(Q.T)
    pi = np.real(V[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    ref = np.mean([np.log(pi[x[0]]) + np.log(Q[x[:-1], x[1:]]).sum() for x in X]) / X.shape[1]
    lin = iolin.IOLin(M1, np.ones(1))
    assert abs(lin.path_loglik(X) - ref) < 1e-10
    th2 = switch.init_switch(k0, 2, 3, "mix20")
    M2 = np.asarray(switch.switch_M(th2, 2, "mix20", grid=GRID))
    assert np.abs(M2.sum(axis=(1, 3)) - 1).max() < 1e-12
    g = jax.grad(lambda t: switch.loglik(t, jnp.asarray(X), 2, "mix20", grid=GRID))(jnp.asarray(th2))
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0


def test_seq6_runner_jobs():
    spec = importlib.util.spec_from_file_location("run_seq6", ROOT / "scripts" / "run_seq6.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    jobs = rs.jobs_all()
    names = {(j.part, j.dataset, j.model) for j in jobs}
    assert ("synth", "msar", "sw-mlp6-S2") in names and ("real", "nasdaq-v", "sw-anc2-S2") in names
    assert all(j.stage in (1, 2) for j in jobs)
    # every stage-2 classical switch has its stage-1 kernel fits in the table
    for j in jobs:
        if j.stage == 2 and ckernels.is_classical(j.model.split("-")[1]):
            assert (j.part, j.dataset, "k-" + j.model.split("-")[1]) in names
