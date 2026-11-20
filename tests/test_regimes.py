"""Sequence 4, step 1: regime-switching targets, exact metrics and the classical models (archb.regimes, archb.iolin)."""

import jax.numpy as jnp
import numpy as np
import pytest

from archb import iolin, regimes as R


@pytest.mark.parametrize("kind,N", [("msar", None), ("smsar", 3)])
def test_truth_rows_and_monte_carlo(kind, N):
    T = R.truth_T(kind, N)
    assert np.abs(T.sum((2, 3)) - 1).max() < 1e-12
    lin = iolin.IOLin.from_iohmm(T)
    X, H = R.simulate(T, 4000, 60, 50, 3)
    J = np.zeros((32, 32))
    np.add.at(J, (X[:, :-1].ravel(), X[:, 1:].ravel()), 1.0)
    J /= J.sum()
    Jx = lin.marginal()[:, None] * lin.kernel()
    assert np.abs(J - Jx).max() < 5 * np.sqrt(Jx.max() / X[:, 1:].size)
    s = R.regime_of(kind, N)[H]
    if kind == "smsar":  # regime durations uniform on 1..N (complete runs only)
        runs = []
        for row in s:
            cut = np.flatnonzero(np.diff(row)) + 1
            runs += np.diff(cut).tolist()
        cnt = np.bincount(runs, minlength=N + 2)[1:]
        assert cnt[N:].sum() == 0 and np.abs(cnt[:N] / cnt.sum() - 1 / N).max() < 0.02
    else:
        stay = np.mean(s[:, 1:] == s[:, :-1])
        assert abs(stay - 0.98) < 0.003


def test_exact_metrics_against_simulation():
    T = R.truth_T("msar")
    lin = iolin.IOLin.from_iohmm(T)
    X, _ = R.simulate(T, 20000, 30, 100, 5)
    v = R.Z[X]
    g = np.abs(v)
    emp = [np.corrcoef(g[:, 0], g[:, k])[0, 1] for k in (1, 5, 20)]
    acf = lin.acf(np.abs(R.Z), 20)
    assert np.abs(np.array(emp) - acf[[0, 4, 19]]).max() < 0.03
    p = lin.sum_law(10)
    s10 = X[:, 1:11].sum(1)
    assert abs(p.sum() - 1) < 1e-10
    emp_cdf = np.array([np.mean(s10 <= k) for k in range(len(p))])
    assert np.abs(np.cumsum(p) - emp_cdf).max() < 0.015
    ve = lin.var_es(R.Z, 10)
    assert ve["es1"] >= ve["var1"] >= ve["var5"] > 0 and ve["es5"] >= ve["var5"]


def test_filtering_matches_direct_forward():
    T = R.truth_T("smsar", 4)
    lin = iolin.IOLin.from_iohmm(T)
    xs = lin.sample(2000, 1)
    a = lin.u[xs[0]] / lin.u[xs[0]].sum()
    lp = [np.log(lin.u[xs[0]].sum())]
    for x, y in zip(xs[:-1], xs[1:]):  # belief over h, then P(x', h' | x, h)
        b = a @ T[x, :, y, :]
        lp.append(np.log(b.sum()))
        a = b / b.sum()
    assert np.abs(lin.logprobs(xs) - np.array(lp)).max() < 1e-10


def test_iohmm_em():
    X, _ = R.simulate(R.truth_T("msar"), 300, 50, 50, 7)
    for kind in ("full", "fact", "tied"):
        T, curve = R.iohmm_em(X, 2, kind, seed=0, max_iter=60)
        assert np.abs(T.sum((2, 3)) - 1).max() < 1e-12 and T.min() > 0
        assert curve[-1] > curve[0]
        if kind == "tied":  # plain HMM on x: one table for every x; EM (tiny prior) is monotone
            assert np.abs(T - T[:1]).max() < 1e-15 and np.diff(curve).min() > -1e-8
        if kind == "fact":  # P(s' | s, x) P(x' | x, s')
            Rk = T.sum(2)
            assert np.abs(T - Rk[:, :, None, :] * (T / Rk[:, :, None, :])[:, :1]).max() < 1e-12


def test_markov_tables_and_order2_representation():
    X, _ = R.simulate(R.truth_T("msar"), 400, 50, 50, 9)
    tabs = R.markov_tables(X, 2)
    lin2 = iolin.IOLin.from_markov2(tabs[2])
    xs = lin2.sample(500, 0)
    assert np.abs(lin2.logprobs(xs)[2:] - R.markov_logprobs(tabs, xs)[2:]).max() < 1e-10
    lin1 = iolin.IOLin.from_kernel(tabs[1])
    assert np.abs(lin1.logprobs(xs)[1:] - R.markov_logprobs(tabs[:2], xs)[1:]).max() < 1e-10


def test_family_fit_recovers_msar():
    X, _ = R.simulate(R.truth_T("msar"), 2000, 50, 200, 11)
    T, p, ll, _ = R.fit_family("msar", X, starts=2)
    assert np.abs(np.array(p["c"]) - R.MSAR["c"]).max() < 0.03
    assert np.abs(np.array(p["sig"]) - R.MSAR["sig"]).max() < 0.05
    assert np.abs(np.array(p["stay"]) - 0.98).max() < 0.01
    assert ll >= iolin.IOLin.from_iohmm(R.truth_T("msar")).path_loglik(X) - 1e-9


def test_regime_decoding():
    """The truth's smoother recovers the regimes of a path of the truth far above chance; run lengths are exact."""
    xs, hs = R.simulate(R.truth_T("smsar", 8), 1, 20000, 50, 13)
    s = R.regime_of("smsar", 8)[hs[0]]
    d = R.decode_regimes("smsar", 8, xs[0])
    assert np.mean(d == s) > 0.8
    assert R.run_lengths(np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])).tolist() == [3, 1, 2]
