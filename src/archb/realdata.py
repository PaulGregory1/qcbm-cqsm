"""Sequence 4, target D: real daily log-returns on the 5-bit grid, and the scores every
model gets on it.

Data: FRED CSVs in data/raw/ (observation_date, value; empty value = no quote that day, dropped), from START on.
Returns r_t = log p_t - log p_{t-1} between consecutive quotes. Chronological split: train = first 70% of the
returns, validation = next 10%, test = the rest. z = (r - mean_train) / sd_train.

Grids (32 bins, bin index order = value order, Gray labels for the circuits as in sequence 4): 'u' = 32 equal bins of
[-4, 4], edge bins absorb the tails; 'q' = edges at the 1/32 .. 31/32 quantiles of the training z; 'v' = the uniform
grid on y_t = z_t / s_t, s_t an EWMA volatility (lambda = 0.94) computed from PAST BINS only, s_{t+1}^2 = lam s_t^2 +
(1 - lam) (s_t c_{b_t})^2 with c the bin centres, s_0 = 1: the bin of day t is the interval s_t [e_j, e_{j+1}) of z,
and every model is scored on these same intervals (same information set: s_t is a function of b_<t). On 'u' and 'q',
s_t = 1 (lam = 1). Edges are stored with -inf / +inf at the ends. Bin values v = mean of the training y in each bin
(midpoint if a bin is empty); in z units the value of bin j on day t is s_t v_j.

Every model is reduced to its predictive matrix over the whole series, P[t, j] = P(b_t = j | b_<t (or z_<t)), row 0
unused; scores are read from P on the test rows:
  log-score    log P[t, b_t], with a moving-block bootstrap standard error;
  1-day VaR    randomized PIT u_t = F_t(b_t - 1) + V_t P[t, b_t] (exactly uniform for a correct discrete law), hit =
               u_t < alpha; Kupiec and Christoffersen tests; ES statistic Z = mean(h_t L_t / (alpha ES_t)) - 1 with
               h_t = E[hit_t | b_t], L = - z_t, the realized return (0 for a correct model, > 0: tail losses
               underestimated). p-values: twice the median over the draws of the uniforms (a valid p-value for a
               randomized test, Meinshausen, Meier and Buehlmann 2009), capped at 1.
  10-day       the same on the sums of bin values over non-overlapping 10-day windows, from Monte Carlo samples of
               each model given its state before the window; ES statistic on the realized sums of z.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from archb import iolin

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SERIES = {"eurusd": "DEXUSEU", "nasdaq": "NASDAQCOM", "gbpusd": "DEXUSUK", "usdchf": "DEXSZUS", "usdjpy": "DEXJPUS"}
START = "1999-01-04"
FRACS = (0.7, 0.1)
A = 32
LO, HI = -4.0, 4.0
LAM = 0.94


# ------------------------------------------------------------------------------------------------ data
def load_prices(name, start=START):
    """(dates, prices) of a series, missing quotes dropped."""
    with open(RAW / f"{SERIES[name]}.csv") as fh:
        rows = list(csv.reader(fh))[1:]
    rows = [(d, v) for d, v in rows if d >= start and v.strip() not in ("", ".")]
    return np.array([d for d, _ in rows]), np.array([float(v) for _, v in rows])


def split_points(n, fracs=FRACS):
    """End indices (exclusive) of train and validation: train [0, a), validation [a, b), test [b, n)."""
    a = int(round(fracs[0] * n))
    return a, a + int(round(fracs[1] * n))


def uniform_edges(lo=LO, hi=HI, n=A):
    e = np.linspace(lo, hi, n + 1)
    e[0], e[-1] = -np.inf, np.inf
    return e


def quantile_edges(z_train, n=A):
    e = np.quantile(z_train, np.arange(1, n) / n)
    return np.concatenate([[-np.inf], e, [np.inf]])


def discretize(z, e):
    return np.searchsorted(e[1:-1], z, side="right")


def bin_values(z, b, e):
    """Mean of z in each bin; midpoint of the finite edges (or one width beyond the last edge) if a bin is empty."""
    n = len(e) - 1
    fin = e[1:-1]
    w = fin[1] - fin[0]
    mid = np.concatenate([[fin[0] - w / 2], 0.5 * (fin[:-1] + fin[1:]), [fin[-1] + w / 2]])
    cnt = np.bincount(b, minlength=n)
    s = np.bincount(b, weights=z, minlength=n)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), mid)


def grid_centres(lo=LO, hi=HI, n=A):
    e = np.linspace(lo, hi, n + 1)
    return 0.5 * (e[:-1] + e[1:])


def next_scale(s, b, lam, c):
    """EWMA step of the grid scale from the observed bin: s_{t+1} = sqrt(lam s_t^2 + (1 - lam) (s_t c_{b_t})^2)."""
    return np.sqrt(lam * s**2 + (1 - lam) * (s * c[b]) ** 2)


def ewma_bins(z, e, lam=LAM, s0=1.0):
    """Scales s_t and bins b_t of the volatility-normalized grid (s_t uses b_<t only)."""
    c = grid_centres()
    s, b = np.empty(len(z)), np.empty(len(z), dtype=np.int64)
    st = s0
    for t in range(len(z)):
        s[t], b[t] = st, discretize(z[t] / st, e)
        st = next_scale(st, b[t], lam, c)
    return s, b


def make_dataset(series, grid):
    dates, p = load_prices(series)
    r = np.diff(np.log(p))
    n = len(r)
    a, b = split_points(n)
    mu, sd = r[:a].mean(), r[:a].std()
    z = (r - mu) / sd
    e = quantile_edges(z[:a]) if grid == "q" else uniform_edges()
    if grid == "v":
        scale, bins = ewma_bins(z, e)
        lam = LAM
    else:
        scale, bins, lam = np.ones(n), discretize(z, e), 1.0
    y = z / scale
    sl = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}
    clip = {k: float(np.mean((y[s] < LO) | (y[s] > HI))) for k, s in sl.items()}
    edge = {k: float(np.mean((bins[s] == 0) | (bins[s] == A - 1))) for k, s in sl.items()}
    return {"series": series, "grid": grid, "dates": dates[1:], "r": r, "z": z, "b": bins, "edges": e,
            "values": bin_values(y[:a], bins[:a], e), "scale": scale, "lam": lam, "centres": grid_centres(),
            "n_train": a, "n_val": b - a, "n_test": n - b, "mu": mu, "sd": sd, "clip_frac": clip, "edge_frac": edge,
            "occupancy_train": np.bincount(bins[:a], minlength=A).tolist()}


def row_edges(ds):
    """Edges of every day's bins in z units, (n, A + 1)."""
    return ds["scale"][:, None] * ds["edges"][None, :]


def row_values(ds):
    """Bin values of every day in z units, (n, A)."""
    return ds["scale"][:, None] * ds["values"][None, :]


# ------------------------------------------------------------------------------------------------ scores
def log_scores(P, b):
    return np.log(np.maximum(P[np.arange(len(b)), b], 1e-300))


def block_bootstrap(x, block=50, n_boot=2000, seed=0):
    """Standard error of mean(x) by the moving-block bootstrap (x: 1D, or 2D with the series on axis 0)."""
    x = np.asarray(x)
    n = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(1).std(ddof=1, axis=0)


def rand_pit(P, b, u):
    F = np.cumsum(P, axis=1)
    i = np.arange(len(b))
    return F[i, b] - P[i, b] + u * P[i, b]


def kupiec(h, alpha):
    """p-value of the unconditional coverage LR test (1 dof)."""
    T, n1 = len(h), int(np.sum(h))
    ll0 = (T - n1) * np.log(1 - alpha) + n1 * np.log(alpha)
    ph = n1 / T
    ll1 = (T - n1) * np.log(1 - ph) + n1 * np.log(ph) if 0 < n1 < T else 0.0
    return float(chi2.sf(-2 * (ll0 - ll1), 1)), float(-2 * (ll0 - ll1))


def christoffersen(h, alpha):
    """p-values of the independence (1 dof) and conditional coverage (2 dof) LR tests."""
    h = np.asarray(h, int)
    a, c = h[:-1], h[1:]
    n = {(i, j): int(np.sum((a == i) & (c == j))) for i in (0, 1) for j in (0, 1)}
    xl = lambda k, p: k * np.log(p) if k else 0.0
    p01 = n[0, 1] / max(n[0, 0] + n[0, 1], 1)
    p11 = n[1, 1] / max(n[1, 0] + n[1, 1], 1)
    p1 = (n[0, 1] + n[1, 1]) / max(sum(n.values()), 1)
    l1 = xl(n[0, 0], 1 - p01) + xl(n[0, 1], p01) + xl(n[1, 0], 1 - p11) + xl(n[1, 1], p11)
    l0 = xl(n[0, 0] + n[1, 0], 1 - p1) + xl(n[0, 1] + n[1, 1], p1)
    lr_ind = max(-2 * (l0 - l1), 0.0)
    lr_uc = kupiec(h, alpha)[1]
    return float(chi2.sf(lr_ind, 1)), float(chi2.sf(lr_uc + lr_ind, 2))


def var_es(P, values, alpha):
    """VaR and ES (positive = loss) of each row's discrete law on the bin values (ascending; one vector, or one row per
    day), atom split at the alpha-quantile. Returns (VaR, ES, expected-hit weights W) with W[t, j] = P(hit | b_t = j)."""
    values = np.broadcast_to(values, P.shape)
    F = np.cumsum(P, axis=1)
    k = np.minimum((F < alpha).sum(1), P.shape[1] - 1)
    i = np.arange(len(P))
    below = np.where(k > 0, F[i, np.maximum(k - 1, 0)], 0.0)
    W = (np.arange(P.shape[1])[None, :] < k[:, None]).astype(float)
    W[i, k] = np.clip((alpha - below) / np.maximum(P[i, k], 1e-300), 0, 1)
    es = -(P * W * values).sum(1) / alpha
    return -values[i, k], es, W


def pval(ps):
    """Valid p-value of a randomized test from its draws: twice the median, capped at 1."""
    return float(min(1.0, 2 * np.median(ps)))


def backtest(P, b, values, alpha, loss=None, n_draws=200, seed=0):
    """1-day backtest at level alpha on rows P (n, A) and realized bins b; loss = realized losses (default: minus the
    realized bin values). Randomized hits, p-values from n_draws draws of the uniforms (same seed for every model)."""
    var, es, W = var_es(P, values, alpha)
    i = np.arange(len(b))
    hexp = W[i, b]
    loss = -np.broadcast_to(values, P.shape)[i, b] if loss is None else loss
    rng = np.random.default_rng(seed)
    pk, pi, pc = [], [], []
    for _ in range(n_draws):
        h = rand_pit(P, b, rng.random(len(b))) < alpha
        pk.append(kupiec(h, alpha)[0])
        ind, cc = christoffersen(h, alpha)
        pi.append(ind)
        pc.append(cc)
    z = float(np.mean(hexp * loss / (alpha * es)) - 1)
    return {"alpha": alpha, "n": len(b), "hit_rate": float(hexp.mean()), "kupiec_p": pval(pk), "ind_p": pval(pi),
            "cc_p": pval(pc), "es_z": z, "mean_var": float(var.mean()),
            "mean_es": float(es.mean())}


def backtest_samples(S, s_real, alpha, loss=None, n_draws=200, seed=0):
    """Backtest of h-day sums: S (W, N) Monte Carlo sums per window, s_real (W,) realized sums of bin values (hits);
    loss = realized losses for the ES statistic (default - s_real)."""
    loss = -s_real if loss is None else loss
    lo = (S < s_real[:, None]).mean(1)
    eq = (S == s_real[:, None]).mean(1)
    srt = np.sort(S, axis=1)
    k = max(int(np.floor(alpha * S.shape[1])), 1)
    var, es = -srt[:, k - 1], -srt[:, :k].mean(1)
    hexp = np.where(eq > 0, np.clip((alpha - lo) / np.maximum(eq, 1e-300), 0, 1), (lo < alpha).astype(float))
    rng = np.random.default_rng(seed)
    pk, pc = [], []
    for _ in range(n_draws):
        h = lo + rng.random(len(lo)) * eq < alpha
        pk.append(kupiec(h, alpha)[0])
        pc.append(christoffersen(h, alpha)[1])
    return {"alpha": alpha, "n": len(s_real), "hit_rate": float(hexp.mean()), "kupiec_p": pval(pk),
            "cc_p": pval(pc), "es_z": float(np.mean(hexp * loss / (alpha * es)) - 1),
            "mean_var": float(var.mean()), "mean_es": float(es.mean())}


def continuous_backtest(z, loc, scale, ppf, cdf, alpha):
    """Usual backtest on the raw (standardized) returns for a location-scale predictive law: hit = z < VaR."""
    q = loc + scale * ppf(alpha)
    h = z < q
    ind, cc = christoffersen(h, alpha)
    return {"alpha": alpha, "n": len(z), "hit_rate": float(h.mean()), "kupiec_p": kupiec(h, alpha)[0],
            "ind_p": ind, "cc_p": cc}


# ------------------------------------------------------------------------------------------------ linear models
def predictive(lin, xs):
    """P[t, j] = P(x_t = j | x_<t) of a linear representation along the path xs (row 0: the marginal)."""
    S = lin.filtered(xs)
    P = np.empty((len(xs), lin.A))
    P[0] = lin.marginal()
    for x in np.unique(xs[:-1]):
        idx = np.flatnonzero(xs[:-1] == x)
        P[idx + 1] = np.real(np.einsum("ti,yij,j->ty", S[idx], lin.M[x], lin.f))
    return np.clip(P, 0, None) / np.clip(P, 0, None).sum(1, keepdims=True)


def mc_sums(lin, state, x, values, h, n, rng, scale=(1.0, 1.0, None)):
    """n Monte Carlo sums of the next h bin values (in z units) given the normalized internal state after observing x;
    scale = (s of the first simulated day, lam, bin centres): the grid scale follows the simulated bins."""
    a = np.tile(np.asarray(state), (n, 1))
    xs = np.full(n, x)
    tot = np.zeros(n)
    s, lam, cen = np.full(n, scale[0]), scale[1], scale[2]
    for _ in range(h):
        B = np.empty((n, lin.A, lin.d), dtype=a.dtype)
        for v in np.unique(xs):
            idx = np.flatnonzero(xs == v)
            B[idx] = np.einsum("ni,yij->nyj", a[idx], lin.M[v])
        p = np.clip(np.real(B @ lin.f), 0, None)
        c = np.cumsum(p / p.sum(1, keepdims=True), axis=1)
        y = np.minimum((c < rng.random(n)[:, None]).sum(1), lin.A - 1)
        a = B[np.arange(n), y] / p[np.arange(n), y][:, None]
        xs, tot = y, tot + s * values[y]
        if lam < 1:
            s = next_scale(s, y, lam, cen)
    return tot


def sample_lin(lin, n_chains, n, rng, burn=500):
    """n_chains independent paths of n symbols of a linear representation (after a burn-in from its stationary law)."""
    pi = lin.marginal()
    x = np.minimum((np.cumsum(pi / pi.sum()) < rng.random(n_chains)[:, None]).sum(1), lin.A - 1)
    a = lin.u[x] / pi[x][:, None]
    out = np.empty((n_chains, n), dtype=np.int64)
    for k in range(burn + n):
        B = np.empty((n_chains, lin.A, lin.d), dtype=a.dtype)
        for v in np.unique(x):
            idx = np.flatnonzero(x == v)
            B[idx] = np.einsum("ni,yij->nyj", a[idx], lin.M[v])
        p = np.clip(np.real(B @ lin.f), 0, None)
        c = np.cumsum(p / p.sum(1, keepdims=True), axis=1)
        x = np.minimum((c < rng.random(n_chains)[:, None]).sum(1), lin.A - 1)
        a = B[np.arange(n_chains), x] / p[np.arange(n_chains), x][:, None]
        if k >= burn:
            out[:, k - burn] = x
    return out


def path_values(ds, B):
    """Values in z units of simulated bin paths B (chains, n): the grid scale follows each path from s = 1."""
    V = ds["values"][B].astype(float)
    if ds["lam"] < 1:
        s = np.ones(len(B))
        for k in range(B.shape[1]):
            V[:, k] *= s
            s = next_scale(s, B[:, k], ds["lam"], ds["centres"])
    return V


def stationary_facts(lin, values, lags=50):
    """Exact acf of |x| (lags 1..lags) and kurtosis of the stationary process of a linear representation."""
    pi = lin.marginal()
    mu = pi @ values
    kurt = float(pi @ (values - mu) ** 4 / (pi @ (values - mu) ** 2) ** 2)
    return lin.acf(np.abs(values), lags), kurt


def sample_acf(x, lags=50):
    x = np.asarray(x, float) - np.mean(x)
    v = x @ x
    return np.array([x[:-k] @ x[k:] / v for k in range(1, lags + 1)])


def kurtosis(x):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.mean(x**4) / np.mean(x**2) ** 2)


def realized_vol(r, window=21):
    """Trailing realized volatility sqrt(mean r^2 over the last `window` days, today included)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(r) ** 2)])
    k = np.arange(1, len(r) + 1)
    lo = np.maximum(k - window, 0)
    return np.sqrt((c[k] - c[lo]) / (k - lo))


def iolin_of_markov(b, k):
    """Order-k Markov chain (k = 0, 1, 2) on one path from add-1/2 counts, as a linear representation."""
    from archb import regimes as R

    tab = R.markov_counts(np.asarray(b)[None], k)
    if k == 0:
        return iolin.IOLin.from_kernel(np.repeat(tab, A, axis=0)), tab
    return (iolin.IOLin.from_kernel(tab) if k == 1 else iolin.IOLin.from_markov2(tab)), tab
