"""Target D (real data): data, grids, scores, backtests (archb.realdata) and the parametric families
(archb.families)."""

import numpy as np
import pytest
from scipy.stats import kstest

from archb import families as F, iolin, realdata as D


@pytest.fixture(scope="module")
def eur():
    return D.make_dataset("eurusd", "u")


def _rand_laws(n, A, seed):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(A) * 0.7, size=n)
    b = (np.cumsum(P, 1) < rng.random(n)[:, None]).sum(1)
    return P, np.minimum(b, A - 1)


# ------------------------------------------------------------------------------------------------ data and grids
@pytest.mark.parametrize("series", ["eurusd", "nasdaq"])
def test_prices_clean_and_ordered(series):
    d, p = D.load_prices(series)
    assert d[0] >= D.START and np.all(d[1:] > d[:-1]) and np.all(p > 0) and len(p) > 6000


def test_split_is_chronological():
    a, b = D.split_points(6968)
    assert (a, b - a, 6968 - b) == (4878, 697, 1393)


@pytest.mark.parametrize("grid", ["u", "q"])
def test_dataset_standardized_on_train_only(grid):
    ds = D.make_dataset("nasdaq", grid)
    a = ds["n_train"]
    assert abs(ds["z"][:a].mean()) < 1e-12 and abs(ds["z"][:a].std() - 1) < 1e-12
    assert ds["n_train"] + ds["n_val"] + ds["n_test"] == len(ds["z"]) == len(ds["dates"])
    assert ds["b"].min() >= 0 and ds["b"].max() <= D.A - 1
    e, v = ds["edges"], ds["values"]
    assert np.all(np.diff(v) > 0) and np.all((v >= np.clip(e[:-1], -50, 50)) & (v <= np.clip(e[1:], -50, 50)))
    assert np.all((ds["z"] >= e[ds["b"]]) & (ds["z"] < e[ds["b"] + 1]))


def test_uniform_grid_edges_and_clipping():
    e = D.uniform_edges()
    assert e[0] == -np.inf and e[-1] == np.inf and np.allclose(np.diff(e[1:-1]), 0.25)
    z = np.array([-9.0, -4.0, -3.99, 0.0, 3.99, 4.0, 9.0])
    assert D.discretize(z, e).tolist() == [0, 0, 0, 16, 31, 31, 31]


def test_vol_grid_uses_past_bins_only():
    ds = D.make_dataset("eurusd", "v")
    z, s, e = ds["z"], ds["scale"], ds["edges"]
    E = D.row_edges(ds)
    t = np.arange(len(z))
    assert np.all((z >= E[t, ds["b"]]) & (z < E[t, ds["b"] + 1]))
    s2, b2 = D.ewma_bins(np.where(t == 3000, 50.0, z), e)  # a shock on day 3000 moves only later scales
    assert np.array_equal(s2[:3001], s[:3001]) and s2[3001] > s[3001]
    ref = np.sqrt(D.LAM * s[:-1] ** 2 + (1 - D.LAM) * (s[:-1] * D.grid_centres()[ds["b"][:-1]]) ** 2)
    assert np.allclose(s[1:], ref) and s[0] == 1.0
    a = ds["n_train"]
    assert abs(np.std(z[a:] / s[a:]) - 1) < 0.25  # normalized test returns are back near unit scale


def test_quantile_grid_balanced_on_train():
    ds = D.make_dataset("nasdaq", "q")
    occ = np.asarray(ds["occupancy_train"])
    assert occ.min() > 0.8 * ds["n_train"] / D.A and occ.max() < 1.2 * ds["n_train"] / D.A


# ------------------------------------------------------------------------------------------------ scores
def test_block_bootstrap_iid_se():
    x = np.random.default_rng(0).normal(size=4000)
    assert abs(D.block_bootstrap(x, block=20) * np.sqrt(4000) - 1) < 0.1


def test_randomized_pit_uniform_under_model():
    P, b = _rand_laws(20000, 8, 1)
    u = D.rand_pit(P, b, np.random.default_rng(2).random(len(b)))
    assert kstest(u, "uniform").pvalue > 0.01


def test_var_es_against_brute_force():
    P, _ = _rand_laws(50, 12, 3)
    v = np.sort(np.random.default_rng(4).normal(size=12))
    for alpha in (0.01, 0.05, 0.3):
        var, es, W = D.var_es(P, v, alpha)
        assert np.allclose((P * W).sum(1), alpha)
        for t in range(len(P)):  # ES = mean of the lowest alpha mass, from a fine split of every atom
            q = np.repeat(v, np.round(P[t] * 1e6).astype(int))[: int(round(alpha * 1e6))]
            assert abs(es[t] + q.mean()) < 1e-3 and var[t] == -v[np.searchsorted(np.cumsum(P[t]), alpha)]


def test_randomized_pvalue():
    assert D.pval([0.2, 0.3, 0.4]) == 0.6 and D.pval([0.6, 0.9]) == 1.0


def test_coverage_tests_known_values():
    h = np.zeros(1000, bool)
    h[::20] = True  # 50 hits = 5%, evenly spread
    assert D.kupiec(h, 0.05)[0] > 0.999
    assert D.kupiec(h, 0.01)[0] < 1e-6
    hc = np.zeros(1000, bool)
    hc[500:550] = True  # same count, clustered
    assert D.christoffersen(hc, 0.05)[0] < 1e-6 and D.christoffersen(h, 0.05)[0] > 0.01


def test_backtest_calibrated_model():
    P, b = _rand_laws(20000, 16, 5)
    v = np.linspace(-3, 3, 16)
    for alpha in (0.01, 0.05):
        r = D.backtest(P, b, v, alpha, n_draws=20)
        assert abs(r["hit_rate"] - alpha) < 3 * np.sqrt(alpha / 20000) and abs(r["es_z"]) < 0.15
        assert r["kupiec_p"] > 0.01 and r["cc_p"] > 0.01


def test_backtest_samples_calibrated():
    rng = np.random.default_rng(6)
    S, s = rng.normal(size=(3000, 2000)), rng.normal(size=3000)
    r = D.backtest_samples(S, s, 0.05, n_draws=10)
    assert abs(r["hit_rate"] - 0.05) < 0.015 and abs(r["mean_var"] - 1.645) < 0.05 and abs(r["es_z"]) < 0.2


def test_predictive_matches_logprobs_and_mc():
    rng = np.random.default_rng(7)
    T = rng.dirichlet(np.ones(32 * 2), size=(32, 2)).reshape(32, 2, 32, 2)
    lin = iolin.IOLin.from_iohmm(T)
    xs = lin.sample(500, 8)
    P = D.predictive(lin, xs)
    assert np.allclose(P.sum(1), 1) and np.allclose(D.log_scores(P, xs)[1:], lin.logprobs(xs)[1:])
    v = np.linspace(-2, 2, 32)  # 2-step sum after the last symbol: Monte Carlo against the exact law
    st = lin.filtered(xs)[-1]
    p1 = st @ np.einsum("yij,j->iy", lin.M[xs[-1]], lin.f)
    exact = sum(np.real(np.einsum("i,ij,yjk,k->y", st, lin.M[xs[-1], y], lin.M[y], lin.f)) @ (v[y] + v)
                for y in range(32))
    mc = D.mc_sums(lin, st, xs[-1], v, 2, 40000, np.random.default_rng(9))
    assert np.isclose(p1.sum(), 1) and abs(mc.mean() - exact) < 4 * mc.std() / np.sqrt(len(mc))


def test_markov_orders_as_linear():
    b = np.random.default_rng(10).integers(0, 32, 3000)
    for k in (0, 1, 2):
        lin, tab = D.iolin_of_markov(b, k)
        P = D.predictive(lin, b)
        expect = {0: lambda: tab[[0] * 2998], 1: lambda: tab[b[1:-1]], 2: lambda: tab[b[:-2] * 32 + b[1:-1]]}[k]()
        assert np.allclose(P[2:], expect)


def test_realized_vol():
    r = np.arange(1.0, 31.0)
    rv = D.realized_vol(r, 21)
    assert np.isclose(rv[0], 1.0) and np.isclose(rv[25], np.sqrt(np.mean(r[5:26] ** 2)))


# ------------------------------------------------------------------------------------------------ families
def _garch_path(p, n, seed, nu=None):
    rng = np.random.default_rng(seed)
    s2, z = p["omega"] / (1 - p["alpha"] - p["beta"]), np.empty(n)
    for t in range(n):
        eps = rng.standard_t(nu) / np.sqrt(nu / (nu - 2)) if nu else rng.standard_normal()
        z[t] = p["mu"] + np.sqrt(s2) * eps
        s2 = p["omega"] + p["alpha"] * (z[t] - p["mu"]) ** 2 + p["beta"] * s2
    return z


def _fake_ds(z, grid="u"):
    a = int(0.7 * len(z))
    mu, sd = z[:a].mean(), z[:a].std()
    zz = (z - mu) / sd
    e = D.uniform_edges() if grid == "u" else D.quantile_edges(zz[:a])
    b = D.discretize(zz, e)
    return {"z": zz, "b": b, "edges": e, "values": D.bin_values(zz[:a], b[:a], e), "n_train": a,
            "scale": np.ones(len(z)), "lam": 1.0, "centres": D.grid_centres()}


def test_garch_sigma2_recursion():
    p = {"mu": 0.1, "omega": 0.05, "alpha": 0.08, "beta": 0.9}
    u = np.random.default_rng(11).normal(size=50)
    s, ref = F.garch_sigma2(p, u, 1.3), [1.3]
    for t in range(49):
        ref.append(p["omega"] + p["alpha"] * (u[t] - p["mu"]) ** 2 + p["beta"] * ref[-1])
    assert np.allclose(s, ref)


def test_gjr_recursion_and_fit():
    p = {"mu": 0.0, "omega": 0.05, "alpha": 0.02, "gamma": 0.1, "beta": 0.9}
    u = np.array([1.0, -1.0, 2.0])
    assert np.allclose(F.garch_sigma2(p, u, 1.0), [1.0, 0.05 + 0.02 + 0.9, 0.05 + 0.12 + 0.9 * 0.97])
    th = np.array([0.0, np.log(0.05), 2.0, 0.3, -0.4, np.log(5.0)])
    q = F.garch_params(th, "t", gjr=True)
    assert np.isclose(q["alpha"] + q["gamma"] / 2 + q["beta"], 0.9999 / (1 + np.exp(-2.0))) and q["gamma"] > 0


def test_mc_sums_constant_scale_unchanged():
    rng = np.random.default_rng(15)
    lin = iolin.IOLin.from_kernel(rng.dirichlet(np.ones(32), size=32))
    v = np.linspace(-2, 2, 32)
    a = D.mc_sums(lin, np.ones(1), 3, v, 5, 1000, np.random.default_rng(0))
    b = D.mc_sums(lin, np.ones(1), 3, v, 5, 1000, np.random.default_rng(0), (2.0, 1.0, D.grid_centres()))
    assert np.allclose(b, 2 * a)
    c = D.mc_sums(lin, np.ones(1), 3, v, 5, 1000, np.random.default_rng(0), (2.0, 0.94, D.grid_centres()))
    assert np.allclose(c[:0], 0) and np.isfinite(c).all() and not np.allclose(c, b)


def test_garch_t_recovers_parameters():
    true = {"mu": 0.0, "omega": 0.02, "alpha": 0.06, "beta": 0.92}
    ds = _fake_ds(_garch_path(true, 8000, 12, nu=6.0))
    g, _ = F.fit_garch(ds, "t", "continuous")
    assert abs(g.p["alpha"] - 0.06) < 0.025 and abs(g.p["beta"] - 0.92) < 0.03 and 4 < g.p["nu"] < 9
    P = g.predictive(ds)
    assert np.allclose(P.sum(1), 1) and P.shape == (8000, 32)
    gb, _ = F.fit_garch(ds, "t", "binned")
    assert abs(gb.p["alpha"] + gb.p["beta"] - 0.98) < 0.03


def test_msar_continuous_filter_and_recovery():
    rng = np.random.default_rng(13)
    n, s, z = 8000, 0, [0.0]
    for _ in range(n - 1):
        s = s if rng.random() < (0.99, 0.97)[s] else 1 - s
        z.append((0.0, 0.0)[s] + (0.1, -0.1)[s] * z[-1] + (0.5, 1.5)[s] * rng.standard_normal())
    ds = _fake_ds(np.array(z))
    m, ll = F.fit_msar(ds, "continuous", starts=3)
    sig = np.array(m.p["sig"]) * 1.0
    assert sig[1] / sig[0] > 2.4 and sig[1] / sig[0] < 3.6 and abs(m.p["stay"][0] - 0.99) < 0.01
    a = ds["n_train"]  # filter log-likelihood against a plain numpy Hamilton filter
    mu, c, sg, st = (np.asarray(m.p[k]) for k in ("mu", "c", "sig", "stay"))
    Q = np.array([[st[0], 1 - st[0]], [1 - st[1], st[1]]])
    xi, tot = np.array([1 - st[1], 1 - st[0]]) / (2 - st.sum()), 0.0
    zt = ds["z"][:a]
    for t in range(1, a):
        pr = xi @ Q
        d = np.exp(-0.5 * ((zt[t] - mu - c * zt[t - 1]) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
        tot += np.log(pr @ d)
        xi = pr * d / (pr @ d)
    assert np.isclose(tot / (a - 1), ll, atol=1e-6)
    P, sm = m.predictive(ds), m.smoothed(ds)
    assert np.allclose(P.sum(1), 1) and np.nanmin(sm) >= 0 and np.nanmax(sm) <= 1


def test_msar_binned_is_an_iohmm():
    ds = _fake_ds(np.random.default_rng(14).standard_t(5, size=3000), grid="q")
    m, ll = F.fit_msar(ds, "binned", starts=2)
    T = np.asarray(F.msar_T(np.asarray(m.th), ds["edges"], ds["values"]))
    assert np.allclose(T.sum((2, 3)), 1)
    P = m.predictive(ds)
    a = ds["n_train"]
    assert np.isclose(D.log_scores(P, ds["b"])[:a].mean(), ll, atol=1e-8)
