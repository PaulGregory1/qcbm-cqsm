"""Sequence 2 targets: exact kernels against Monte Carlo, labels, metrics."""

import numpy as np
import pytest

from archb import targets as TG


@pytest.mark.parametrize("name", list(TG.TARGETS))
def test_exact_kernel_rows(name):
    P = TG.TARGETS[name].exact_kernel()
    assert P.min() >= 0
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("name", list(TG.TARGETS))
def test_exact_kernel_matches_monte_carlo(name):
    """Simulate one step from a few start states (exact bin centres) and compare bin frequencies to the kernel rows."""
    T = TG.TARGETS[name]
    P = T.exact_kernel()
    X = T.state_values()
    rng = np.random.default_rng(0)
    rows = np.linspace(0, T.K - 1, 6).astype(int)
    for r in rows:
        x = np.repeat(X[r][None, :], 200_000, axis=0)
        y = T.step(x, rng)
        idx = [TG.to_bins(y[:, k], T.lo, T.hi, T.n_per_dim)[0] for k in range(T.d)]
        flat = idx[0] if T.d == 1 else idx[0] * T.n_per_dim + idx[1]
        freq = np.bincount(flat, minlength=T.K) / len(flat)
        assert 0.5 * np.abs(freq - P[r]).sum() < 0.01


def test_gray_labels_are_a_permutation_with_unit_steps():
    lob = TG.gray_labels(4, 2)
    assert sorted(lob) == list(range(256))
    g = TG.gray_labels(5, 1)
    assert all(bin(int(a) ^ int(b)).count("1") == 1 for a, b in zip(g[:-1], g[1:]))


@pytest.mark.parametrize("name", list(TG.TARGETS))
def test_metrics_zero_at_exact(name):
    T = TG.TARGETS[name]
    m = TG.metrics(T.exact_kernel(), T)
    assert m["weighted_tv"] == 0 and m["stationary_tv"] < 1e-9
    assert all(m[f"cond_mean_rmse_{k}"] < 1e-12 and m[f"cond_sd_rel_{k}"] < 1e-12 for k in range(T.d))
    assert TG.tight_pass(m, T.d)


def test_follow_up_targets():
    """var2d3 / var2d3i share the process and data; interleaving only relabels bins."""
    a, b = TG.VAR2D3, TG.VAR2D3I
    np.testing.assert_allclose(a.exact_kernel(), b.exact_kernel())
    assert sorted(b.label_of_bin()) == list(range(64)) and sorted(TG.VAR2DI.label_of_bin()) == list(range(256))
    g = TG.binary_to_gray(np.arange(8))
    assert b.label_of_bin()[2 * 8 + 5] == int("".join(x + y for x, y in zip(f"{g[2]:03b}", f"{g[5]:03b}")), 2)
