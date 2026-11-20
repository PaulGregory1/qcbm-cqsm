import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from archb import sinkhorn as sk
from archb import train as tr
from archb.circuit import N_PARAMS, init_theta
from archb.ou import CENTRES

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def measures():
    pi_hat, v = tr.data_measures()
    return pi_hat, v


def random_measure(rng, zero_frac=0.0):
    a = rng.dirichlet(np.ones(1024)).reshape(32, 32)
    a[rng.random((32, 32)) < zero_frac] = 0.0
    return jnp.asarray(a / a.sum())


def test_separable_softmin_matches_dense():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(32, 32))
    h[rng.random((32, 32)) < 0.5] = -np.inf
    X = np.stack(np.meshgrid(CENTRES, CENTRES, indexing="ij"), -1).reshape(-1, 2)
    C = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    dense = -sk.EPS * logsumexp(h.reshape(-1)[None, :] - C / sk.EPS, axis=1)
    assert np.max(np.abs(dense - np.asarray(sk.softmin(jnp.asarray(h), sk.EPS)).ravel())) < 1e-12


def test_divergence_of_measure_with_itself_is_zero(measures):
    _, v = measures
    w = random_measure(np.random.default_rng(1))
    assert abs(sk.divergence(v, v)) < 1e-9
    assert abs(sk.divergence(w, w)) < 1e-9


def test_divergence_nonnegative_on_random_pairs(measures):
    _, v = measures
    rng = np.random.default_rng(2)
    for k in range(4):
        a = random_measure(rng, zero_frac=0.3 * (k % 2))
        b = random_measure(rng) if k < 2 else v
        assert sk.divergence(a, b) > -1e-9


def test_envelope_gradient_matches_full_autodiff(measures):
    pi_hat, v = measures
    theta = jax.random.uniform(jax.random.PRNGKey(7), (N_PARAMS,), minval=-np.pi, maxval=np.pi)

    def w_of(th):
        return tr.model_measure(th, "binary", pi_hat)

    w = w_of(theta)
    f, g, _ = sk.converge_cross(w, v)
    p, _ = sk.converge_self(w)
    pv, _ = sk.converge_self(v)
    ot_vv = sk.ot_self_value(v, pv)
    env = jax.grad(lambda th: sk.envelope_objective(w_of(th), v, f, g, p, ot_vv))(theta)
    full = jax.jit(jax.grad(lambda th: sk.divergence_unrolled(w_of(th), v, n_iter=300)))(theta)
    cos = float(env @ full / (jnp.linalg.norm(env) * jnp.linalg.norm(full)))
    assert cos > 0.99
    assert abs(float(jnp.linalg.norm(full) / jnp.linalg.norm(env)) - 1.0) < 0.02
    # the envelope objective's value is the converged divergence
    value = float(sk.envelope_objective(w, v, f, g, p, ot_vv))
    assert abs(value - sk.divergence(w, v)) < 1e-9


def test_zero_weights_give_finite_values_and_gradients(measures):
    _, v = measures
    assert int((v == 0).sum()) > 0
    rng = np.random.default_rng(3)
    a = random_measure(rng, zero_frac=0.5)
    val, (ga, gb) = jax.value_and_grad(lambda a, b: sk.divergence_unrolled(a, b, n_iter=50), argnums=(0, 1))(a, v)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(ga))) and np.all(np.isfinite(np.asarray(gb)))
    f, g, _ = sk.converge_cross(a, v)
    p, _ = sk.converge_self(a)
    for x in (f, g, p):
        assert np.all(np.isfinite(np.asarray(x)))
    assert np.isfinite(sk.divergence(a, v))


def test_short_training_decreases_loss_and_saves(tmp_path):
    res = tr.train("gray", seed=0, steps=30, log_every=0)
    assert res["steps"] == 30 and np.all(np.isfinite(res["loss"]))
    assert res["loss"][-1] < 0.5 * res["loss"][0]
    assert res["final_loss"] < res["init_loss"]
    np.testing.assert_allclose(res["kernel"].sum(axis=1), 1.0, atol=1e-12)
    out = tr.save_run(res, tmp_path)
    for name in ("theta.npy", "kernel.npy", "loss.npy", "metrics.json"):
        assert (out / name).exists()
    info = json.loads((out / "metrics.json").read_text())
    assert np.isfinite(info["metrics"]["weighted_tv"])
    assert np.asarray(init_theta(0)).shape == np.load(out / "theta.npy").shape
