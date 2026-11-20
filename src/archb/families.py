"""Sequence 4, target D: the parametric families used on real returns, each scored on the 32-bin grid of
archb.realdata by integrating its predictive density over the bins.

Bins of day t are the intervals s_t [e_j, e_{j+1}) of z (s_t = 1 except on the volatility-normalized grid 'v', see
archb.realdata); bin values in z units are s_t v_j, and every simulator carries s along its own simulated bins.

GARCH(1,1) on standardized returns z: z_t = mu + sigma_t eps_t, sigma^2_{t+1} = omega + alpha (u_t - mu)^2 + beta
  sigma^2_t (GJR: + gamma (u_t - mu)^2 1{u_t < mu}), eps ~ N(0, 1) ('norm') or a unit-variance Student-t ('t');
  sigma^2_0 = the variance of the training inputs. Inputs u: the returns themselves ('continuous inputs', fitted by
  the continuous likelihood) or the bin values of the observed bins ('binned inputs', fitted by the binned likelihood:
  the information set of the circuits). numpy / scipy, L-BFGS-B with finite-difference gradients (4-6 parameters).
MS-AR(1), 2 regimes: z_t = mu_s + c_s z_{t-1} + sigma_s eps, s Markov with stay probabilities p_s (8 parameters).
  Continuous inputs: Hamilton filter on z, fitted by the continuous likelihood (JAX gradients), Kim smoother; binned
  inputs: the input-output HMM T[x, s, x', s'] = Q[s, s'] P(bin x' | N(mu_s' + c_s' v_x, sigma_s'^2)) fitted by the
  binned likelihood (archb.iolin), on the bin values v of the grid (on 'v': an MS-AR of the normalized series).
  Regime 1 = the larger sigma.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jnorm
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm, t as student

from archb import iolin
from archb.realdata import discretize, next_scale, row_edges

jax.config.update("jax_enable_x64", True)

BIG = 1e4  # finite stand-in for the infinite outer edges (gradients)


# ------------------------------------------------------------------------------------------------ GARCH
def garch_params(th, dist, gjr=False):
    """GARCH: persistence alpha + beta = 0.9999 sigmoid(th2) split by sigmoid(th3). GJR: alpha + gamma / 2 + beta =
    0.9999 sigmoid(th2) split by softmax(th3, th4, 0); the innovation parameter follows."""
    pers = 0.9999 / (1 + np.exp(-th[2]))
    p = {"mu": float(th[0]), "omega": float(np.exp(th[1]))}
    if gjr:
        w = np.exp([th[3], th[4], 0.0])
        w /= w.sum()
        p.update(alpha=float(pers * w[0]), gamma=float(2 * pers * w[1]), beta=float(pers * w[2]))
    else:
        share = 1 / (1 + np.exp(-th[3]))
        p.update(alpha=float(pers * share), beta=float(pers * (1 - share)))
    if dist == "t":
        p["nu"] = float(2.05 + np.exp(th[5 if gjr else 4]))
    return p


def std_dist(dist, nu=None):
    """(cdf, logpdf, ppf, sampler) of the unit-variance innovation."""
    if dist == "norm":
        return norm.cdf, norm.logpdf, norm.ppf, lambda rng, size: rng.standard_normal(size)
    k = np.sqrt(nu / (nu - 2))
    return (lambda x: student.cdf(x * k, nu), lambda x: student.logpdf(x * k, nu) + np.log(k),
            lambda a: student.ppf(a, nu) / k, lambda rng, size: rng.standard_t(nu, size) / k)


def garch_news(p, u):
    """(alpha + gamma 1{u < mu}) (u - mu)^2, the shock term of the variance recursion."""
    e = np.asarray(u) - p["mu"]
    return (p["alpha"] + p.get("gamma", 0.0) * (e < 0)) * e**2


def garch_sigma2(p, u, s0):
    """Predictive variance of every step: sigma2[t] = Var(z_t | u_<t), sigma2[0] = s0."""
    s = lfilter([1.0], [1.0, -p["beta"]], p["omega"] + garch_news(p, u)[:-1], zi=[p["beta"] * s0])[0]
    return np.concatenate([[s0], s])


def bin_probs(loc, scale, edges, cdf):
    """P[t, j] = cdf((e_{j+1} - loc_t) / scale_t) - cdf((e_j - loc_t) / scale_t) for one location-scale law per row;
    edges (A + 1,) shared or (n, A + 1) per row."""
    C = cdf((np.asarray(edges) - np.asarray(loc)[..., None]) / np.asarray(scale)[..., None])
    return np.maximum(np.diff(C, axis=-1), 1e-300)


class Garch:
    def __init__(self, th, dist, inputs, s0, gjr=False):
        self.th, self.dist, self.inputs, self.s0, self.gjr = np.asarray(th), dist, inputs, s0, gjr
        self.p = garch_params(self.th, dist, gjr)
        self.cdf, self.logpdf, self.ppf, self.draw = std_dist(dist, self.p.get("nu"))

    def n_params(self):
        return len(self.th)

    def u(self, ds):
        return ds["z"] if self.inputs == "continuous" else ds["scale"] * ds["values"][ds["b"]]

    def sigma2(self, ds):
        return garch_sigma2(self.p, self.u(ds), self.s0)

    def predictive(self, ds):
        return bin_probs(np.full(len(ds["z"]), self.p["mu"]), np.sqrt(self.sigma2(ds)), row_edges(ds), self.cdf)

    def _step(self, ds, s2, sg, n, rng):
        """One simulated day: (bin, value in z units, next variance, next grid scale)."""
        p = self.p
        z = p["mu"] + np.sqrt(s2) * self.draw(rng, n)
        b = discretize(z / sg, ds["edges"])
        v = sg * ds["values"][b]
        s2 = p["omega"] + garch_news(p, z if self.inputs == "continuous" else v) + p["beta"] * s2
        return b, v, s2, next_scale(sg, b, ds["lam"], ds["centres"])

    def mc_sums(self, ds, t, h, n, rng):
        """Sums of the bin values (z units) of days t+1 .. t+h given everything up to t (index t of the series)."""
        s2, sg, tot = np.full(n, self.sigma2(ds)[t + 1]), np.full(n, ds["scale"][t + 1]), np.zeros(n)
        for _ in range(h):
            _, v, s2, sg = self._step(ds, s2, sg, n, rng)
            tot += v
        return tot

    def sample_values(self, ds, n_chains, n, rng, burn=1000):
        """Simulated paths of bin values in z units (grid scale from 1)."""
        s2, sg, out = np.full(n_chains, self.s0), np.ones(n_chains), np.empty((n_chains, n))
        for k in range(burn + n):
            _, v, s2, sg = self._step(ds, s2, sg, n_chains, rng)
            if k >= burn:
                out[:, k - burn] = v
        return out


def fit_garch(ds, dist, inputs, starts=3, seed=0, gjr=False):
    """Maximum likelihood on the training period: continuous density (inputs 'continuous') or bin probabilities
    (inputs 'binned', on the day's bin intervals). Returns (Garch, mean log-likelihood per step of the fit objective)."""
    a = ds["n_train"]
    u = (ds["z"] if inputs == "continuous" else ds["scale"] * ds["values"][ds["b"]])[:a]
    z, b, e = ds["z"][:a], ds["b"][:a], row_edges(ds)[:a]
    s0 = float(np.var(u))
    i = np.arange(a)

    def nll(th):
        p = garch_params(th, dist, gjr)
        cdf, logpdf, _, _ = std_dist(dist, p.get("nu"))
        sd = np.sqrt(garch_sigma2(p, u, s0))
        if inputs == "continuous":
            return -np.mean(logpdf((z - p["mu"]) / sd) - np.log(sd))
        lo, hi = cdf((e[i, b] - p["mu"]) / sd), cdf((e[i, b + 1] - p["mu"]) / sd)
        return -np.mean(np.log(np.maximum(hi - lo, 1e-300)))

    rng, best = np.random.default_rng(seed), None
    base = [0.0, np.log(0.02), 3.5] + ([-2.5, -2.5, 2.5] if gjr else [-2.5]) + ([np.log(6.0)] if dist == "t" else [])
    for k in range(starts):
        th0 = np.array(base) + (0.3 * rng.normal(size=len(base)) if k else 0.0)
        out = minimize(nll, th0, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8})
        if best is None or out.fun < best.fun:
            best = out
    return Garch(best.x, dist, inputs, s0, gjr), -float(best.fun)


# ------------------------------------------------------------------------------------------------ MS-AR
def msar_params(th):
    th = np.asarray(th)
    return {"mu": th[0:2].tolist(), "c": th[2:4].tolist(), "sig": np.exp(th[4:6]).tolist(),
            "stay": (1 / (1 + np.exp(-th[6:8]))).tolist()}


def _q(stay):
    return jnp.stack([jnp.stack([stay[0], 1 - stay[0]]), jnp.stack([1 - stay[1], stay[1]])])


def _unpack(th):
    return th[0:2], th[2:4], jnp.exp(th[4:6]), _q(jax.nn.sigmoid(th[6:8]))


def _stat2(Q):
    a, b = 1 - Q[0, 0], 1 - Q[1, 1]
    return jnp.stack([b, a]) / (a + b)


@jax.jit
def hamilton(th, z):
    """Hamilton filter on z (z_0 conditioned on). Returns (log f(z_t | z_<t), predicted xi_{t|t-1}, filtered xi_{t|t})
    for t = 1 .. n-1."""
    mu, c, sig, Q = _unpack(th)

    def step(xi, zz):
        pred = xi @ Q
        dens = jnp.exp(jnorm.logpdf(zz[1], mu + c * zz[0], sig))
        lik = pred @ dens
        post = pred * dens / lik
        return post, (jnp.log(lik), pred, post)

    _, (ll, pred, post) = jax.lax.scan(step, _stat2(Q), jnp.stack([z[:-1], z[1:]], axis=1))
    return ll, pred, post


def msar_T(th, edges, values):
    """IOHMM T[x, s, x', s'] of the binned-input MS-AR (outer edges at +-BIG so that gradients exist)."""
    mu, c, sig, Q = _unpack(th)
    e = jnp.asarray(np.clip(edges, -BIG, BIG))
    m = mu[:, None] + c[:, None] * jnp.asarray(values)[None, :]  # [s, x]
    K = jnp.diff(jnorm.cdf((e[None, None, :] - m[:, :, None]) / sig[:, None, None]), axis=2)  # [s', x, x']
    return jnp.einsum("hk,kxy->xhyk", Q, K)


def _sort_regimes(th):
    """Relabel so that regime 1 has the larger sigma."""
    th = np.asarray(th, float).copy()
    if th[4] > th[5]:
        th = th[[1, 0, 3, 2, 5, 4, 7, 6]]
    return th


class MSAR:
    def __init__(self, th, inputs, ds=None):
        self.th, self.inputs = _sort_regimes(th), inputs
        self.p = msar_params(self.th)
        self.lin = iolin.IOLin.from_iohmm(np.asarray(msar_T(jnp.asarray(self.th), ds["edges"], ds["values"]))) \
            if inputs == "binned" else None

    def n_params(self):
        return 8

    def filter(self, ds):
        ll, pred, post = hamilton(jnp.asarray(self.th), jnp.asarray(ds["z"]))
        return np.asarray(ll), np.asarray(pred), np.asarray(post)

    def predictive(self, ds):
        if self.lin is not None:
            from archb.realdata import predictive
            return predictive(self.lin, ds["b"])
        _, pred, _ = self.filter(ds)
        mu, c, sig = (np.asarray(self.p[k]) for k in ("mu", "c", "sig"))
        z = ds["z"]
        P = np.full((len(z), len(ds["edges"]) - 1), 1.0 / (len(ds["edges"]) - 1))
        E = row_edges(ds)[1:]
        P[1:] = sum(pred[:, s:s + 1] * bin_probs(mu[s] + c[s] * z[:-1], np.full(len(z) - 1, sig[s]), E, norm.cdf)
                    for s in range(2))
        return P

    def smoothed(self, ds):
        """P(s_t = turbulent | whole series), t = 1 .. n-1 (Kim smoother)."""
        _, pred, post = self.filter(ds)
        Q = np.asarray(_q(jnp.asarray(self.p["stay"])))
        sm = post.copy()
        for t in range(len(post) - 2, -1, -1):
            sm[t] = post[t] * (Q @ (sm[t + 1] / pred[t + 1]))
        return np.concatenate([[np.nan], sm[:, 1] / sm.sum(1)])

    def mc_sums(self, ds, t, h, n, rng):
        """Sums of the bin values of z_{t+1} .. z_{t+h} given everything up to t (filter computed once per series)."""
        from archb.realdata import mc_sums
        if getattr(self, "_cache", (None,))[0] is not ds:
            self._cache = (ds, self.lin.filtered(ds["b"]) if self.lin is not None else self.filter(ds)[2])
        if self.lin is not None:
            return mc_sums(self.lin, self._cache[1][t], ds["b"][t], ds["values"], h, n, rng,
                           (ds["scale"][t + 1], ds["lam"], ds["centres"]))
        mu, c, sig, stay = (np.asarray(self.p[k]) for k in ("mu", "c", "sig", "stay"))
        s = (rng.random(n) < self._cache[1][t - 1, 1]).astype(int)
        z, sg, tot = np.full(n, ds["z"][t]), np.full(n, ds["scale"][t + 1]), np.zeros(n)
        for _ in range(h):
            s = np.where(rng.random(n) < stay[s], s, 1 - s)
            z = mu[s] + c[s] * z + sig[s] * rng.standard_normal(n)
            b = discretize(z / sg, ds["edges"])
            tot += sg * ds["values"][b]
            sg = next_scale(sg, b, ds["lam"], ds["centres"])
        return tot

    def sample_values(self, ds, n_chains, n, rng, burn=1000):
        """Simulated paths of bin values in z units of the continuous-input MS-AR (grid scale from 1)."""
        mu, c, sig, stay = (np.asarray(self.p[k]) for k in ("mu", "c", "sig", "stay"))
        s, z, sg = rng.integers(0, 2, n_chains), np.zeros(n_chains), np.ones(n_chains)
        out = np.empty((n_chains, n))
        for k in range(burn + n):
            s = np.where(rng.random(n_chains) < stay[s], s, 1 - s)
            z = mu[s] + c[s] * z + sig[s] * rng.standard_normal(n_chains)
            b = discretize(z / sg, ds["edges"])
            if k >= burn:
                out[:, k - burn] = sg * ds["values"][b]
            sg = next_scale(sg, b, ds["lam"], ds["centres"])
        return out


def fit_msar(ds, inputs, starts=6, seed=0):
    """Maximum likelihood of the MS-AR(1) on the training period (continuous density or binned IOHMM likelihood).
    Returns (MSAR, mean log-likelihood per step of the fit objective)."""
    a = ds["n_train"]
    if inputs == "continuous":
        zj = jnp.asarray(ds["z"][:a])
        nll = lambda th: -jnp.mean(hamilton(th, zj)[0])
    else:
        X = jnp.asarray(ds["b"][:a])[None]
        nll = lambda th: -iolin.loglik(msar_T(th, ds["edges"], ds["values"]).transpose(0, 2, 1, 3), jnp.ones(2), X)
    vg = jax.jit(jax.value_and_grad(nll))
    rng, best = np.random.default_rng(seed), None
    for k in range(starts):
        th0 = np.concatenate([0.05 * rng.normal(size=2), 0.05 * rng.normal(size=2),
                              np.log([0.6, 1.5]) + 0.2 * rng.normal(size=2), [4.0, 3.0] + 0.5 * rng.normal(size=2)])
        out = minimize(lambda th: tuple(np.asarray(v, float) for v in vg(jnp.asarray(th))), th0, jac=True,
                       method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-9})
        if np.isfinite(out.fun) and (best is None or out.fun < best.fun):
            best = out
    return MSAR(best.x, inputs, ds), -float(best.fun)
