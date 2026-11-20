"""Sequence 1, step 0: expressivity ceiling.

Fit a circuit directly to the exact kernel P*, with no data and no Sinkhorn:

    L(theta) = sum_i pi*(i) KL( P*(.|i) || P_theta(.|i) )

by autodiff. A fit, not a training protocol: the goal is the best value the circuit can reach.
Phase 1: Adam, ADAM_STEPS steps, lr 0.03 decayed geometrically to 0.003, keeping the best iterate
(checked every CHUNK steps; Adam at this rate can jump out of a good basin). Phase 2: L-BFGS-B
(scipy) from that iterate, up to LBFGS_ITERS iterations, on the same deterministic objective.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from archb.ou import exact_kernel, stationary

jax.config.update("jax_enable_x64", True)

ADAM_STEPS = 2000
LBFGS_ITERS = 20000
CHUNK = 250
LR0, LR1 = 0.03, 0.003
KL_FLOOR = 1e-300
B1, B2, AEPS = 0.9, 0.999, 1e-8


def kl_objective(kernel_fn, P_star=None):
    P_star = exact_kernel() if P_star is None else P_star
    pi = jnp.asarray(stationary(P_star))
    Ps = jnp.asarray(P_star)
    ent = float(np.sum(stationary(P_star)[:, None] * np.where(P_star > 0, P_star * np.log(np.where(P_star > 0, P_star, 1)), 0)))

    def loss(theta):
        P = kernel_fn(theta)
        cross = jnp.sum(pi[:, None] * Ps * jnp.log(jnp.maximum(P, KL_FLOOR)))
        return ent - cross

    return loss


def data_kl_objective(kernel_fn, data):
    """Maximum likelihood on the observed transitions (a sequence 1 diagnostic, not the spec loss):
    KL(v || pi_hat x P_theta) = sum_ij v_ij log(v_ij / (pi_hat_i P_theta(j|i))), v = N / sum N.
    Same data as the Sinkhorn training; only the objective changes."""
    N = np.asarray(data["N"], dtype=float)
    v = N / N.sum()
    pi_hat = v.sum(axis=1)
    const = float(np.sum(np.where(v > 0, v * np.log(np.where(v > 0, v, 1) / np.where(pi_hat[:, None] > 0, pi_hat[:, None], 1)), 0)))
    vj = jnp.asarray(v)

    def loss(theta):
        return const - jnp.sum(vj * jnp.log(jnp.maximum(kernel_fn(theta), KL_FLOOR)))

    return loss


def fit(kernel_fn, theta0, adam_steps=ADAM_STEPS, lbfgs_iters=LBFGS_ITERS, log=None, loss_fn=None, lr0=LR0, lr1=LR1):
    """Two phases: Adam (keeps the best iterate; lr lr0 -> lr1), then L-BFGS from that iterate.
    Returns dict(theta, loss curve (Adam steps then one value per L-BFGS iteration), steps,
    adam_loss, final_loss, lbfgs_message, wall_time)."""
    t0 = time.time()
    loss_fn = kl_objective(kernel_fn) if loss_fn is None else loss_fn
    res = adam(kernel_fn, theta0, adam_steps, log, loss_fn, lr0, lr1)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    curve, last = [], [None]

    def fun(x):
        v, g = vg(jnp.asarray(x))
        last[0] = float(v)
        return last[0], np.asarray(g, dtype=np.float64)

    out = minimize(fun, res["theta"], jac=True, method="L-BFGS-B",
                   callback=lambda x: curve.append(last[0]),
                   options={"maxiter": lbfgs_iters, "maxfun": 2 * lbfgs_iters, "ftol": 1e-13, "gtol": 1e-9})
    theta = out.x if out.fun <= res["final_loss"] else res["theta"]
    final = min(float(out.fun), res["final_loss"])
    return {"theta": np.asarray(theta), "loss": np.concatenate([res["loss"], np.asarray(curve)]),
            "steps": res["steps"] + int(out.nit), "adam_loss": res["final_loss"], "final_loss": final,
            "lbfgs_message": str(out.message), "lbfgs_iters": int(out.nit), "wall_time": time.time() - t0}


def adam(kernel_fn, theta0, steps=ADAM_STEPS, log=None, loss_fn=None, lr0=LR0, lr1=LR1):
    """Adam on loss_fn (default kl_objective), lr decayed geometrically lr0 -> lr1 (lr1 = lr0: constant);
    returns the best iterate seen (the loss can jump at lr 0.03)."""
    loss_fn = kl_objective(kernel_fn) if loss_fn is None else loss_fn
    vg = jax.value_and_grad(loss_fn)
    decay = (lr1 / lr0) ** (1.0 / steps)

    @jax.jit
    def chunk(state):
        def body(s, _):
            th, m, v, t = s
            val, g = vg(th)
            t = t + 1
            m = B1 * m + (1 - B1) * g
            v = B2 * v + (1 - B2) * g**2
            lr = lr0 * decay ** (t - 1)
            th = th - lr * (m / (1 - B1**t)) / (jnp.sqrt(v / (1 - B2**t)) + AEPS)
            return (th, m, v, t), val
        return jax.lax.scan(body, state, None, length=CHUNK)

    t0 = time.time()
    z = jnp.zeros_like(theta0)
    state = (jnp.asarray(theta0, jnp.float64), z, z, jnp.zeros((), jnp.float64))
    losses = []
    best_theta, best = np.asarray(theta0), float(loss_fn(jnp.asarray(theta0)))
    while len(losses) < steps:
        state, vals = chunk(state)
        losses.extend(np.asarray(vals).tolist())
        if not np.isfinite(losses[-1]):
            raise FloatingPointError("non-finite loss")
        cur = float(loss_fn(state[0]))
        if cur < best:
            best, best_theta = cur, np.asarray(state[0])
        if log:
            log(len(losses), losses[-1])
    return {"theta": best_theta, "loss": np.asarray(losses), "steps": len(losses),
            "final_loss": best, "wall_time": time.time() - t0}
