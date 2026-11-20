import numpy as np
import pytest

from archb.ou import (
    CENTRES, EDGES, N_BINS, OU, WIDTH, bits_to_int, binary_to_gray, counts_to_kernel,
    encoding_perm, exact_kernel, gray_to_binary, int_to_bits, load_dataset, make_dataset,
    stationary, value_to_bin,
)


def test_derived_constants():
    assert OU.c == pytest.approx(0.905, abs=1e-3)
    assert OU.s == pytest.approx(0.301, abs=1e-3)
    assert OU.stat_sd == pytest.approx(0.707, abs=1e-3)
    assert OU.m == 0.0


def test_grid():
    assert len(EDGES) == 33 and len(CENTRES) == 32
    assert WIDTH == pytest.approx(0.1875)
    assert EDGES[0] == -3.0 and EDGES[-1] == 3.0
    assert np.allclose(np.diff(EDGES), WIDTH)


def test_value_to_bin_clipping():
    bins, frac = value_to_bin(np.array([-5.0, -3.0, 0.0, 2.99, 3.0, 7.0]))
    assert list(bins) == [0, 0, 16, 31, 31, 31]
    assert frac == pytest.approx(2 / 6)
    assert np.array_equal(value_to_bin(CENTRES)[0], np.arange(N_BINS))


def test_bits_round_trip_msb_first():
    assert list(int_to_bits(1)) == [0, 0, 0, 0, 1]
    assert list(int_to_bits(16)) == [1, 0, 0, 0, 0]
    idx = np.arange(N_BINS)
    assert np.array_equal(bits_to_int(int_to_bits(idx)), idx)


def test_gray_definition_and_round_trip():
    idx = np.arange(N_BINS)
    b, g = int_to_bits(idx), int_to_bits(binary_to_gray(idx))
    assert np.array_equal(g[:, 0], b[:, 0])
    assert np.array_equal(g[:, 1:], b[:, 1:] ^ b[:, :-1])
    assert np.array_equal(gray_to_binary(binary_to_gray(idx)), idx)


def test_gray_single_bit_adjacency():
    lab, inv = encoding_perm("gray")
    flips = [bin(int(lab[i]) ^ int(lab[i + 1])).count("1") for i in range(N_BINS - 1)]
    assert flips == [1] * (N_BINS - 1)
    assert np.array_equal(inv[lab], np.arange(N_BINS))
    assert sorted(lab) == list(range(N_BINS))
    blab, binv = encoding_perm("binary")
    assert np.array_equal(blab, np.arange(N_BINS)) and np.array_equal(binv, np.arange(N_BINS))
    assert bin(int(blab[15]) ^ int(blab[16])).count("1") == 5  # 01111 -> 10000
    with pytest.raises(ValueError):
        encoding_perm("onehot")


def test_exact_kernel_row_stochastic():
    P = exact_kernel()
    assert P.shape == (32, 32) and np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(P, P[::-1, ::-1])  # mu = 0: symmetric under x -> -x


def test_stationary():
    P = exact_kernel()
    pi = stationary(P)
    assert pi.sum() == pytest.approx(1.0) and np.all(pi >= 0)
    assert np.allclose(pi @ P, pi, atol=1e-12)
    # close to the continuous stationary law N(0, 0.5) discretized
    from scipy.stats import norm
    u = EDGES.copy(); u[0], u[-1] = -np.inf, np.inf
    pi_cont = np.diff(norm.cdf(u / OU.stat_sd))
    assert 0.5 * np.abs(pi - pi_cont).sum() < 0.02


def test_dataset_counts_and_fit():
    d = make_dataset()
    assert d["n_transitions"] == 98_000 == d["N"].sum()
    assert np.allclose(d["pi_hat"], d["N"].sum(axis=1) / 98_000)
    print(f"clipped fraction = {d['clip_frac']:.2e}")
    assert d["clip_frac"] < 1e-3
    P, Pe = exact_kernel(), counts_to_kernel(d["N"])
    bulk = np.abs(CENTRES) < 1.0
    row_tv = 0.5 * np.abs(Pe - P).sum(axis=1)
    assert row_tv[bulk].max() < 0.04
    assert 0.5 * np.abs(d["pi_hat"] - stationary(P)).sum() < 0.03
    # deterministic: same seed gives the same data
    assert np.array_equal(make_dataset()["N"], d["N"])


def test_saved_dataset_matches(tmp_path):
    d = load_dataset(tmp_path / "data.npz")  # generates and saves
    assert np.array_equal(d["N"], make_dataset()["N"])
    assert np.array_equal(load_dataset(tmp_path / "data.npz")["N"], d["N"])
