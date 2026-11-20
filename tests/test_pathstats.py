"""Trajectory statistics and switch detection (archb.pathstats)."""

import numpy as np

from archb import cqsm, iolin, memdiag, pathstats as PS, regimes as R


def test_sample_paths_law():
    lin = cqsm.io(cqsm.init_theta(2, 1, 1, n_bits=2, noise=0.8), 1, 1, n_bits=2)
    X = PS.sample_paths(lin, 4000, 3, 0)
    emp = np.bincount(X[:, 0] * 16 + X[:, 1] * 4 + X[:, 2], minlength=64) / len(X)
    exact = np.exp(memdiag.word_logprobs(lin, 3))
    assert 0.5 * np.abs(emp - exact).sum() < 0.05


def test_waiting_times_geometric():
    p = 0.1
    E = np.random.default_rng(0).random((200, 3000)) < p
    S = PS.waiting_times(E, 20)
    assert np.abs(S - (1 - p) ** np.arange(1, 21)).max() < 0.01


def test_runs_above():
    assert PS.runs_above(np.array([[0, 1, 1, 0, 1, 0, 1]], bool)).tolist() == [2, 1]  # the last run is incomplete


def test_forecast_laws_limits():
    lin = iolin.IOLin.from_iohmm(R.truth_T("msar"))
    xs = lin.sample(300, 4)
    S = lin.filtered(xs)
    F = PS.forecast_laws(lin, S[100:110], xs[100:110], [1, 400])
    assert np.abs(F[1] - memdiag.predictive(lin, S[100:110], xs[100:110])).max() < 1e-12
    assert np.abs(F[400] - lin.marginal()[None]).max() < 1e-6


def test_detection_exact_and_late():
    s = np.array([0] * 50 + [1] * 50 + [0] * 50)
    q = s.astype(float)
    r = PS.detection(q, s, 0.9, 1)
    assert r["delay_mean"] == 0 and r["missed"] == 0 and r["fa_per_1000"] == 0
    r3 = PS.detection(q, s, 0.9, 3)
    assert r3["delay_mean"] == 2
    noisy = q.copy()
    noisy[10] = 1.0  # a single spike in the calm regime
    assert PS.detection(noisy, s, 0.9, 1)["fa_per_1000"] > 0 and PS.detection(noisy, s, 0.9, 2)["fa_per_1000"] == 0
