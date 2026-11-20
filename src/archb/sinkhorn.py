"""Debiased Sinkhorn divergence between weighted measures on the 32 x 32 grid (JAX, float64).

Support: the 1,024 points x_ij = (z_i, z_j) in R^2, z = decoded bin centres, flattened
row-major (k = 32 i + j), so a measure is a (32, 32) array of non-negative weights summing
to 1. Cost C(x, y) = |x - y|^2 (squared Euclidean distance on decoded values).

Convention (Feydy et al. 2019, geomloss). With the soft-min
    softmin_eps(h)_x = -eps log sum_y exp(h_y - C(x, y) / eps),
the dual potentials (f, g) of OT_eps(a, b) solve
    f = softmin_eps(log b + g / eps),     g = softmin_eps(log a + f / eps)
and at that fixed point
    OT_eps(a, b) = <a, f> + <b, g>,
the value of the entropic problem  min_pi <pi, C> + eps KL(pi | a x b)  (marginals a, b).
The debiased divergence is
    S_eps(a, b) = OT_eps(a, b) - 1/2 OT_eps(a, a) - 1/2 OT_eps(b, b),
with OT_eps(a, a) = 2 <a, p> for the symmetric potential p = softmin_eps(log a + p / eps).
Its gradient in a (mass-preserving directions) is f_ab - p_aa (envelope theorem).

Solvers: the cross term uses alternating updates (f from g, then g from f); the self term
uses the averaged symmetric update p <- (p + softmin(log a + p / eps)) / 2, which does not
oscillate. Both have the fixed point above.

Separable cost: C = (z_i - z_i')^2 + (z_j - z_j')^2, so the 1,024 x 1,024 log-sum-exp is
done exactly as two nested 32-point log-sum-exps (32^3 terms each instead of 1,024^2).

Zero weights: log 0 = -inf is kept as -inf (masked, no NaN in values or gradients); the
log-sum-exp below returns -inf for an all -inf slice with zero gradient. Potentials stay
finite everywhere because the opposite measure always has positive mass.
"""

import jax
import jax.numpy as jnp
import numpy as np

from archb.ou import CENTRES

jax.config.update("jax_enable_x64", True)

Z = jnp.asarray(CENTRES)
EPS = 0.01
COST_1D = (Z[:, None] - Z[None, :]) ** 2  # (32, 32): (z_i - z_i')^2
DIAMETER_SQ = float(2 * (CENTRES[-1] - CENTRES[0]) ** 2)  # max cost on the grid, ~66


def use_grid(z):
    """Switch the support to decoded centres z (sequence 2 targets on another range). Module-level state,
    read when the solvers are first traced: call once per process, before any solve. The default is the OU grid."""
    global Z, COST_1D, DIAMETER_SQ
    Z = jnp.asarray(z)
    COST_1D = (Z[:, None] - Z[None, :]) ** 2
    DIAMETER_SQ = float(2 * (float(z[-1]) - float(z[0])) ** 2)


def safe_log(a):
    """log a with log 0 = -inf, finite (zero) gradient at a = 0."""
    pos = a > 0
    return jnp.where(pos, jnp.log(jnp.where(pos, a, 1.0)), -jnp.inf)


def _lse(x, axis):
    """log-sum-exp along `axis`; -inf on all -inf slices, never NaN in the gradient."""
    m = jax.lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    s = jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True)
    pos = s > 0
    out = jnp.where(pos, jnp.log(jnp.where(pos, s, 1.0)) + m, -jnp.inf)
    return jnp.squeeze(out, axis=axis)


def softmin(h, eps):
    """softmin_eps(h)[i, j] = -eps log sum_{i', j'} exp(h[i', j'] - C((i, j), (i', j')) / eps)."""
    K = -COST_1D / eps
    inner = _lse(h[:, None, :] + K[None, :, :], axis=2)  # [i', j] = lse_j' h[i', j'] + K[j, j']
    return -eps * _lse(K[:, :, None] + inner[None, :, :], axis=1)  # [i, j] = lse_i' K[i, i'] + inner[i', j]


def cross_step(la, lb, f, g, eps):
    f = softmin(lb + g / eps, eps)
    g = softmin(la + f / eps, eps)
    return f, g


def self_step(la, p, eps):
    return 0.5 * (p + softmin(la + p / eps, eps))


def cross_iter(la, lb, f, g, eps, n_iter):
    """n_iter alternating updates from (f, g); differentiable (fori_loop, static count)."""
    return jax.lax.fori_loop(0, n_iter, lambda _, fg: cross_step(la, lb, *fg, eps), (f, g))


def self_iter(la, p, eps, n_iter):
    return jax.lax.fori_loop(0, n_iter, lambda _, p: self_step(la, p, eps), p)


def eps_schedule(eps=EPS, ratio=0.5):
    """Geometric eps-scaling from the squared diameter down to eps (excluded)."""
    out, e = [], DIAMETER_SQ
    while e > eps:
        out.append(e)
        e *= ratio
    return out


def cold_cross(la, lb, eps=EPS, n_iter=300, per_scale=3):
    """Cold start: eps-scaling (per_scale updates per scale), then n_iter updates at eps.
    Fully differentiable with a fixed number of iterations (used for the autodiff check)."""
    f = g = jnp.zeros_like(la)
    for e in eps_schedule(eps):
        f, g = cross_iter(la, lb, f, g, e, per_scale)
    return cross_iter(la, lb, f, g, eps, n_iter)


def cold_self(la, eps=EPS, n_iter=300, per_scale=3):
    p = jnp.zeros_like(la)
    for e in eps_schedule(eps):
        p = self_iter(la, p, e, per_scale)
    return self_iter(la, p, eps, n_iter)


@jax.jit
def _converge_cross(la, lb, f, g, eps, tol, max_iter):
    def cond(s):
        return (s[2] > tol) & (s[3] < max_iter)

    def body(s):
        f, g, _, n = s
        f2, g2 = cross_step(la, lb, f, g, eps)
        err = jnp.maximum(jnp.max(jnp.abs(f2 - f)), jnp.max(jnp.abs(g2 - g)))
        return f2, g2, err, n + 1

    f, g, err, n = jax.lax.while_loop(cond, body, (f, g, jnp.inf, 0))
    return f, g, err, n


@jax.jit
def _converge_self(la, p, eps, tol, max_iter):
    def cond(s):
        return (s[1] > tol) & (s[2] < max_iter)

    def body(s):
        p, _, n = s
        p2 = self_step(la, p, eps)
        return p2, jnp.max(jnp.abs(p2 - p)), n + 1

    p, err, n = jax.lax.while_loop(cond, body, (p, jnp.inf, 0))
    return p, err, n


def converge_cross(a, b, eps=EPS, f=None, g=None, tol=1e-12, max_iter=100_000):
    """Potentials (f, g) of OT_eps(a, b) iterated to max |change| < tol. Without a warm start,
    eps-scaling first. Returns (f, g, n_iter)."""
    la, lb = safe_log(a), safe_log(b)
    if f is None:
        f, g = cold_cross(la, lb, eps, n_iter=0)
    f, g, _, n = _converge_cross(la, lb, f, g, eps, tol, max_iter)
    return f, g, int(n)


def converge_self(a, eps=EPS, p=None, tol=1e-12, max_iter=100_000):
    la = safe_log(a)
    if p is None:
        p = cold_self(la, eps, n_iter=0)
    p, _, n = _converge_self(la, p, eps, tol, max_iter)
    return p, int(n)


def ot_value(a, b, f, g):
    return jnp.sum(a * f) + jnp.sum(b * g)


def ot_self_value(a, p):
    return 2.0 * jnp.sum(a * p)


def divergence(a, b, eps=EPS, tol=1e-12):
    """S_eps(a, b) with all three terms converged (numpy-facing; not for gradients)."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    f, g, _ = converge_cross(a, b, eps, tol=tol)
    pa, _ = converge_self(a, eps, tol=tol)
    pb, _ = converge_self(b, eps, tol=tol)
    return float(ot_value(a, b, f, g) - 0.5 * ot_self_value(a, pa) - 0.5 * ot_self_value(b, pb))


def divergence_unrolled(a, b, eps=EPS, n_iter=300):
    """S_eps(a, b) with fixed-length cold-start solvers, differentiable end to end in a and b."""
    la, lb = safe_log(a), safe_log(b)
    f, g = cold_cross(la, lb, eps, n_iter)
    pa, pb = cold_self(la, eps, n_iter), cold_self(lb, eps, n_iter)
    return ot_value(a, b, f, g) - 0.5 * ot_self_value(a, pa) - 0.5 * ot_self_value(b, pb)


def envelope_objective(a, b, f, g, p, ot_bb):
    """Dual value of S_eps(a, b) with potentials held fixed: its value is the loss estimate and
    its gradient in a is f - p (envelope theorem). ot_bb is the constant OT_eps(b, b)."""
    f, g, p = jax.lax.stop_gradient((f, g, p))
    return jnp.sum(a * f) + jnp.sum(b * g) - jnp.sum(a * p) - 0.5 * ot_bb
