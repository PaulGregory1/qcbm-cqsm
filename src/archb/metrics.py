"""The six metrics of the spec for a 32x32 bin-space kernel P against P*, and the go/no-go checks.

All metrics are exact functions of the kernels (no sampling). Values are the decoded bin
centres z. pi* = stationary(P*) weights everything that needs a state weighting.

Go/no-go (spec, "Go/no-go criteria"): GO when, for one encoding, at least MIN_SEEDS_PASSING
seeds pass the four per-seed checks AND the across-seed spread check passes.
Across-seed spread is defined as (max - min) / mean of the weighted TV over ALL seeds of the
encoding (not only the passing ones), so one diverged seed fails it; it is the only check
that cannot be evaluated per seed.
"""

import numpy as np

from .ou import CENTRES, OU, stationary

# ---- go/no-go thresholds (first guesses, spec) ----
STAT_TV_MAX = 0.05          # stationary TV < 0.05
LAG1_TOL = 0.02             # |lag-1 acf - exact lag-1 acf| <= 0.02
SLOPE_TOL = 0.03            # |conditional-mean slope - c| <= 0.03
WTV_IID_RATIO = 0.5         # weighted TV <= 0.5 * weighted TV of the i.i.d. oracle
SEED_SPREAD_MAX = 0.2       # (max - min) / mean of weighted TV across seeds < 0.2
MIN_SEEDS_PASSING = 2       # of 3

TAIL_Z = 2 * OU.stat_sd     # 1.414: tail rows are |z_i| > TAIL_Z (16 of 32 bins)
ACF_LAGS = np.arange(1, 21)


def _tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


def row_tv(P, P_star):
    """Per-row TV, shape (32,)."""
    return _tv(P, P_star)


def acf(P, lags=ACF_LAGS, z=CENTRES):
    """Autocorrelation of the stationary chain of P on values z at the given lags:
    corr(x_t, x_{t+k}) = (pi . (z * P^k z) - mean^2) / var, pi = stationary(P)."""
    P = np.asarray(P, dtype=float)
    pi = stationary(P)
    mean = pi @ z
    var = pi @ z**2 - mean**2
    lags = np.atleast_1d(lags)
    out = np.empty(len(lags))
    for n, k in enumerate(lags):
        out[n] = (pi @ (z * (np.linalg.matrix_power(P, int(k)) @ z)) - mean**2) / var
    return out


def cond_mean_sd(P, z=CENTRES):
    """(E[x_{t+1} | x_t = z_i], sd[x_{t+1} | x_t = z_i]) per row, each shape (32,)."""
    P = np.asarray(P, dtype=float)
    m = P @ z
    return m, np.sqrt(np.clip(P @ z**2 - m**2, 0.0, None))


def cond_mean_slope(P, weights, z=CENTRES):
    """Slope of E[x_{t+1} | x_t = z_i] against z_i by weighted least squares.

    Weighting: pi* (the exact stationary law), the same weights for every model, so the slope
    reflects the states the process actually visits and is comparable across kernels; the
    almost-never-visited edge rows (where edge-bin absorption bends the conditional mean)
    count for little."""
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    y = cond_mean_sd(P, z)[0]
    zb, yb = w @ z, w @ y
    return float((w * (z - zb) * (y - yb)).sum() / (w * (z - zb) ** 2).sum())


def all_metrics(P, P_star) -> dict:
    """The six spec metrics (plus the exact lag-1 value for comparison) of P against P_star."""
    pi_star = stationary(P_star)
    rtv = row_tv(P, P_star)
    acf_p, acf_s = acf(P), acf(P_star)
    return {
        "weighted_tv": float(pi_star @ rtv),
        "tail_tv": float(rtv[np.abs(CENTRES) > TAIL_Z].mean()),
        "stationary_tv": float(_tv(stationary(P), pi_star)),
        "lag1_acf": float(acf_p[0]),
        "lag1_acf_exact": float(acf_s[0]),
        "acf_err": float(np.max(np.abs(acf_p - acf_s))),
        "slope": cond_mean_slope(P, pi_star),
    }


def go_no_go(metrics_per_seed, iid_metrics, c=OU.c) -> dict:
    """Spec go/no-go for ONE encoding.

    metrics_per_seed: list of all_metrics dicts (one per seed); iid_metrics: all_metrics of the
    i.i.d. oracle. Returns per-seed check results, the across-seed spread and the verdict:
    go = (#seeds passing all four per-seed checks >= MIN_SEEDS_PASSING) and spread check."""
    wtv_max = WTV_IID_RATIO * iid_metrics["weighted_tv"]
    seeds = []
    for m in metrics_per_seed:
        checks = {
            "stationary_tv": m["stationary_tv"] < STAT_TV_MAX,
            "lag1_acf": abs(m["lag1_acf"] - m["lag1_acf_exact"]) <= LAG1_TOL,
            "slope": abs(m["slope"] - c) <= SLOPE_TOL,
            "weighted_tv_vs_iid": m["weighted_tv"] <= wtv_max,
        }
        seeds.append({"checks": checks, "pass": all(checks.values())})
    wtv = np.array([m["weighted_tv"] for m in metrics_per_seed])
    spread = float((wtv.max() - wtv.min()) / wtv.mean())
    n_pass = sum(s["pass"] for s in seeds)
    spread_ok = spread < SEED_SPREAD_MAX
    return {
        "seeds": seeds,
        "n_seeds_passing": n_pass,
        "weighted_tv_spread": spread,
        "spread_pass": spread_ok,
        "go": bool(n_pass >= MIN_SEEDS_PASSING and spread_ok),
    }
