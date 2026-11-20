import numpy as np
import pytest

from archb.metrics import (
    SEED_SPREAD_MAX, STAT_TV_MAX, acf, all_metrics, cond_mean_sd, cond_mean_slope, go_no_go,
)
from archb.ou import CENTRES, N_BINS, OU, exact_kernel, gaussian_bin_kernel, load_dataset, stationary
from archb.references import REFERENCE_NAMES, fit_ar1, reference_kernels


@pytest.fixture(scope="module")
def refs():
    return reference_kernels(load_dataset())


def test_gaussian_bin_kernel_is_exact_kernel():
    assert np.array_equal(gaussian_bin_kernel(OU.c, OU.m, OU.s), exact_kernel())


def test_references_row_stochastic(refs):
    assert set(refs) == set(REFERENCE_NAMES)
    for P in refs.values():
        assert P.shape == (N_BINS, N_BINS)
        assert np.all(P >= 0)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_exact_scores_zero(refs):
    P = refs["exact"]
    m = all_metrics(P, P)
    assert m["weighted_tv"] == m["tail_tv"] == m["stationary_tv"] == m["acf_err"] == 0.0
    assert m["lag1_acf"] == m["lag1_acf_exact"]
    # discretized chain is close to the continuous OU: acf(k) ~ c^k
    assert np.allclose(acf(P, [1, 5, 20]), OU.c ** np.array([1, 5, 20]), atol=2e-3)


def test_iid_oracle_acf_is_zero(refs):
    # Every row equals pi_hat, so the chain's own stationary law is pi_hat and P^k z is the
    # constant pi_hat.z: the acf is zero exactly (floating point), whatever pi_hat vs pi* is.
    assert np.max(np.abs(acf(refs["iid"]))) < 1e-12
    m = all_metrics(refs["iid"], refs["exact"])
    assert abs(m["slope"]) < 1e-12
    assert m["acf_err"] == pytest.approx(m["lag1_acf_exact"], abs=1e-12)
    # its stationary TV is exactly TV(pi_hat, pi*)
    pi_hat = refs["iid"][0]
    assert m["stationary_tv"] == pytest.approx(0.5 * np.abs(pi_hat - stationary(refs["exact"])).sum())


def test_identity_kernel_slope_one():
    I = np.eye(N_BINS)
    assert cond_mean_slope(I, np.ones(N_BINS)) == pytest.approx(1.0)
    m, sd = cond_mean_sd(I)
    assert np.allclose(m, CENTRES) and np.allclose(sd, 0.0)


def test_slope_of_linear_conditional_mean():
    # a kernel with E[x'|z_i] = a z_i exactly: mix the mirrored bin with the identity
    a = 0.4
    P = (1 + a) / 2 * np.eye(N_BINS) + (1 - a) / 2 * np.eye(N_BINS)[::-1]
    assert cond_mean_slope(P, stationary(exact_kernel())) == pytest.approx(a)


def test_ar1_fit_recovers_c():
    c, m, s = fit_ar1(load_dataset()["N"])
    assert c == pytest.approx(OU.c, abs=0.01)
    assert abs(m) < 0.01
    assert s == pytest.approx(OU.s, abs=0.015)  # inflated by within-bin quantization


def test_tabular_rows_sum_to_one(refs):
    assert np.allclose(refs["tabular"].sum(axis=1), 1.0)
    assert np.all(refs["tabular"] > 0)


def _m(wtv, stat_tv=0.01, lag1=0.90, slope=0.90):
    return {"weighted_tv": wtv, "stationary_tv": stat_tv, "lag1_acf": lag1,
            "lag1_acf_exact": OU.c, "slope": slope}


def test_go_no_go_logic():
    iid = {"weighted_tv": 0.5}
    good = [_m(0.10), _m(0.105), _m(0.11)]
    r = go_no_go(good, iid)
    assert r["go"] and r["n_seeds_passing"] == 3 and r["spread_pass"]

    # one seed fails a per-seed check, spread still fine: 2 of 3 is enough
    r = go_no_go([_m(0.10), _m(0.105, stat_tv=STAT_TV_MAX + 0.01), _m(0.11)], iid)
    assert r["go"] and r["n_seeds_passing"] == 2
    assert r["seeds"][1]["checks"]["stationary_tv"] is False

    # two seeds fail (lag-1 off, slope off): no go
    r = go_no_go([_m(0.10), _m(0.105, lag1=0.87), _m(0.11, slope=0.86)], iid)
    assert not r["go"] and r["n_seeds_passing"] == 1

    # weighted TV above half the iid oracle's
    r = go_no_go([_m(0.26), _m(0.27), _m(0.28)], iid)
    assert not r["go"] and not any(s["checks"]["weighted_tv_vs_iid"] for s in r["seeds"])

    # all per-seed checks pass but the spread is too large: no go
    r = go_no_go([_m(0.08), _m(0.10), _m(0.12)], iid)
    assert r["n_seeds_passing"] == 3 and r["weighted_tv_spread"] == pytest.approx(0.4)
    assert r["weighted_tv_spread"] > SEED_SPREAD_MAX and not r["go"]
