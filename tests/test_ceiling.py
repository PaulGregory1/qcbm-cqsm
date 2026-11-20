"""Step 0 and maximum-likelihood objectives (archb.ceiling): zero exactly at their minimizers, positive elsewhere."""

import jax.numpy as jnp
import numpy as np

from archb import ceiling
from archb.ou import counts_to_kernel, exact_kernel, load_dataset


def test_kl_objective_zero_at_exact_kernel():
    P_star = exact_kernel()
    loss = ceiling.kl_objective(lambda th: jnp.asarray(P_star))
    assert abs(float(loss(None))) < 1e-12
    uniform = ceiling.kl_objective(lambda th: jnp.full((32, 32), 1 / 32))
    assert float(uniform(None)) > 0.5


def test_data_kl_zero_at_empirical_kernel():
    data = load_dataset()
    P_emp = counts_to_kernel(data["N"])
    loss = ceiling.data_kl_objective(lambda th: jnp.asarray(P_emp), data)
    assert abs(float(loss(None))) < 1e-12
    loss_star = ceiling.data_kl_objective(lambda th: jnp.asarray(exact_kernel()), data)
    assert float(loss_star(None)) > 0


def test_fit_recovers_a_reachable_kernel():
    """A 2-parameter family that contains P*'s neighbour: the fit must drive the KL near 0."""
    P_star = exact_kernel()
    U = np.full((32, 32), 1 / 32)

    def kfn(th):
        a = jnp.clip(jnp.sin(th[0]) ** 2, 0, 1)
        return a * jnp.asarray(P_star) + (1 - a) * jnp.asarray(U)

    res = ceiling.fit(kfn, jnp.asarray([0.3, 0.0]), adam_steps=500, lbfgs_iters=200)
    assert res["final_loss"] < 1e-6
