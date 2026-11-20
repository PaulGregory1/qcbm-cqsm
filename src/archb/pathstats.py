"""Generated trajectories and regime switches of the input-output models (archb.iolin), against the truth.

Free generation: independent paths from the stationary joint law (sample_paths, batched filtering sampler, exact in
law), then path statistics on the bin values z: marginal quantiles and CDF distance, autocorrelations of x, x^2 and
|x|, exceedances of thresholds u fixed from the truth (rate, cluster length), waiting time until the next exceedance
(Kaplan-Meier, censored at the path end). Each model-to-truth discrepancy is set against the truth-to-truth
discrepancy of independent truth samples of the same size.

Conditional forecasts (exact, no sampling): from the filtered state at an origin t of the held-out path, the law of
x_{t+h} (forecast_laws); scored by TV to the truth's law, log score and central-interval coverage of the realized value.

Switches: excess log-loss e_t aligned on the true switches; a detector on a regime score q_t (declare a new regime
after r consecutive steps beyond eta) gives the delay, the missed switches and the false alarms per 1,000 steps.
"""

import numpy as np


# ------------------------------------------------------------------------------------- free generation
def sample_paths(lin, n_paths, n_steps, seed):
    """(n_paths, n_steps) symbols, each path from the stationary joint law."""
    rng = np.random.default_rng(seed)
    A, M, f = lin.A, lin.M, lin.f
    pi = np.clip(lin.marginal(), 0, None)
    x = rng.choice(A, size=n_paths, p=pi / pi.sum())
    a = lin.u[x] / np.real(lin.u[x] @ f)[:, None]
    out = np.empty((n_paths, n_steps), np.int64)
    out[:, 0] = x
    ar = np.arange(n_paths)
    for t in range(1, n_steps):
        B = np.einsum("pi,pyij->pyj", a, M[x])
        p = np.clip(np.real(B @ f), 0, None)
        c = np.cumsum(p / p.sum(1, keepdims=True), 1)
        x = np.minimum((c < rng.random(n_paths)[:, None]).sum(1), A - 1)
        a = B[ar, x] / p[ar, x][:, None]
        out[:, t] = x
    return out


def acf(V, lags):
    """Pooled autocorrelation of the paths V (P, T) at lags 1..lags (global mean and variance)."""
    mu, var = V.mean(), V.var()
    W = V - mu
    return np.array([np.mean(W[:, :-k] * W[:, k:]) for k in range(1, lags + 1)]) / var


def runs_above(E):
    """Lengths of the runs of True in each row of E (complete runs only)."""
    out = []
    for e in E:
        d = np.diff(np.concatenate([[0], e.astype(int), [0]]))
        s, t = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        keep = (s > 0) & (t < len(e))
        out += list((t - s)[keep])
    return np.asarray(out)


def waiting_times(E, hmax):
    """Kaplan-Meier survival S(j) = P(no exceedance in t+1..t+j | no exceedance at t), j = 1..hmax, pooled over every
    start t with E[t] False; starts whose path ends first are censored."""
    events = np.zeros(hmax + 2)
    at_risk = np.zeros(hmax + 2)
    for e in E:
        T = len(e)
        nxt = np.full(T, -1)
        nxt_ix = -1
        for t in range(T - 1, -1, -1):  # index of the next exceedance after t
            nxt[t] = nxt_ix
            if e[t]:
                nxt_ix = t
        st = np.flatnonzero(~e)
        w = np.where(nxt[st] >= 0, nxt[st] - st, -1)
        cens = np.minimum(T - 1 - st, hmax + 1)
        obs = (w > 0) & (w <= hmax)
        np.add.at(events, w[obs], 1)
        dur = np.where(obs, w, cens)
        cnt = np.bincount(np.minimum(dur, hmax + 1), minlength=hmax + 2)
        at_risk += cnt[::-1].cumsum()[::-1]
    S = np.cumprod(1 - events[1:hmax + 1] / np.maximum(at_risk[1:hmax + 1], 1))
    return S


def path_stats(X, z, thresholds, lags=50, hmax=200):
    """Statistics of the symbol paths X (P, T) on the bin values z."""
    V = z[X]
    out = {"quantiles": np.quantile(V, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist(), "var": float(V.var()),
           "cdf": np.cumsum(np.bincount(X.ravel(), minlength=len(z)) / X.size).tolist(),
           "acf_x": acf(V, lags).tolist(), "acf_x2": acf(V**2, lags).tolist(), "acf_abs": acf(np.abs(V), lags).tolist()}
    for k, u in thresholds.items():
        E = np.abs(V) > u
        cl = runs_above(E)
        out[f"exc_{k}"] = {"rate": float(E.mean()), "cluster_mean": float(cl.mean()) if len(cl) else 0.0,
                           "n_clusters": int(len(cl)), "survival": waiting_times(E, hmax).tolist()}
    return out


def discrepancies(s, ref, thresholds):
    """Scalar distances between two path_stats dicts."""
    d = {"cdf_ks": float(np.max(np.abs(np.asarray(s["cdf"]) - ref["cdf"]))),
         "var_rel": float(abs(s["var"] / ref["var"] - 1)),
         "acf_x_max": float(np.max(np.abs(np.asarray(s["acf_x"]) - ref["acf_x"]))),
         "acf_x2_max": float(np.max(np.abs(np.asarray(s["acf_x2"]) - ref["acf_x2"]))),
         "acf_abs_max": float(np.max(np.abs(np.asarray(s["acf_abs"]) - ref["acf_abs"])))}
    for k in thresholds:
        a, b = s[f"exc_{k}"], ref[f"exc_{k}"]
        d[f"exc_{k}_rate_rel"] = float(abs(a["rate"] / b["rate"] - 1))
        d[f"exc_{k}_cluster"] = float(abs(a["cluster_mean"] - b["cluster_mean"]))
        d[f"exc_{k}_survival_sup"] = float(np.max(np.abs(np.asarray(a["survival"]) - b["survival"])))
    return d


# ------------------------------------------------------------------------------------- conditional forecasts
def forecast_laws(lin, states, xs, horizons):
    """Laws of x_{t+h} given the filtered states a_t and inputs x_t (origins in rows), h in horizons:
    -> dict h -> (n, A)."""
    A, d = lin.A, lin.d
    G = lin.G()
    W = np.einsum("ti,tyij->tyj", states, lin.M[np.asarray(xs)]).reshape(len(xs), A * d)  # joint law of x_{t+1}
    out, h = {}, 1
    for H in sorted(horizons):
        while h < H:
            W = W @ G
            h += 1
        P = np.clip(np.real(W.reshape(len(xs), A, d) @ lin.f), 0, None)
        out[H] = P / P.sum(1, keepdims=True)
    return out


def forecast_scores(Pm, Pt, y, level=0.9):
    """Mean TV to the truth's law, log scores of the realized y (model, truth), coverage of the model's central
    interval (in bins) at the given level."""
    n = len(y)
    tv = 0.5 * np.abs(Pm - Pt).sum(1)
    ls_m = np.log(np.maximum(Pm[np.arange(n), y], 1e-300))
    ls_t = np.log(np.maximum(Pt[np.arange(n), y], 1e-300))
    C = np.cumsum(Pm, 1)
    lo = (C < (1 - level) / 2).sum(1)
    hi = (C < 1 - (1 - level) / 2).sum(1)
    return {"tv": float(tv.mean()), "tv_se": float(tv.std(ddof=1) / np.sqrt(n)), "logscore": float(ls_m.mean()),
            "logscore_truth": float(ls_t.mean()), "logscore_gap_se": float((ls_t - ls_m).std(ddof=1) / np.sqrt(n)),
            "coverage": float(np.mean((y >= lo) & (y <= hi)))}


# ------------------------------------------------------------------------------------- switches
def switch_times(s):
    """Indices tau with s[tau - 1] != s[tau] and the new regime."""
    tau = np.flatnonzero(np.diff(s) != 0) + 1
    return tau, s[tau]


def aligned(e, tau, lo=-20, hi=50, isolated_from=None):
    """Mean of e over windows [tau + lo, tau + hi] (rows: switches), optionally only switches with no other switch in
    the window. -> (mean curve, n windows)."""
    tau = tau[(tau + lo >= 0) & (tau + hi < len(e))]
    if isolated_from is not None:
        allsw = isolated_from
        keep = [not np.any((allsw > t + lo) & (allsw <= t + hi) & (allsw != t)) for t in tau]
        tau = tau[np.asarray(keep, bool)] if len(tau) else tau
    if not len(tau):
        return np.full(hi - lo + 1, np.nan), 0
    return np.mean([e[t + lo:t + hi + 1] for t in tau], 0), int(len(tau))


def declared(q, eta, r):
    """Declared regime: switch to 1 after r consecutive steps with q > eta, back to 0 after r steps with q < 1 - eta."""
    out = np.zeros(len(q), np.int8)
    cur, run = 0, 0
    for t, v in enumerate(q):
        beyond = v > eta if cur == 0 else v < 1 - eta
        run = run + 1 if beyond else 0
        if run >= r:
            cur, run = 1 - cur, 0
        out[t] = cur
    return out


def detection(q, s, eta, r, max_delay=50):
    """Delay (steps from the true switch until the declared regime equals the new one, before the next true switch
    and within max_delay; delay_cens also counts each missed switch as max_delay), missed fraction and false alarms
    per 1,000 steps (declared changes into a regime that is not the true one at that step)."""
    D = declared(q, eta, r)
    tau, new = switch_times(s)
    nxt = np.append(tau[1:], len(s))
    delays, missed = [], 0
    for t, n_, e in zip(tau, new, nxt):
        hit = np.flatnonzero(D[t:min(e, t + max_delay + 1)] == n_)
        if len(hit):
            delays.append(hit[0])
        else:
            missed += 1
    ch = np.flatnonzero(np.diff(D) != 0) + 1
    fa = int(np.sum(D[ch] != s[ch]))
    cens = delays + [max_delay] * missed  # a missed switch counts as max_delay: a detector stuck in one regime
    return {"eta": eta, "r": r, "delay_mean": float(np.mean(delays)) if delays else np.nan,  # misses half of them
            "delay_cens": float(np.mean(cens)) if cens else np.nan,
            "delay_median": float(np.median(delays)) if delays else np.nan, "missed": missed / max(len(tau), 1),
            "fa_per_1000": 1000 * fa / len(s), "n_switches": int(len(tau))}
