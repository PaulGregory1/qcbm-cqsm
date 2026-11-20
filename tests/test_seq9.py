import itertools

import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal, norm

from archb import cqsm, iolin, targets, variants
from archb import multivar as MV


def test_labels_and_coords():
    assert (cqsm.bin_labels(5, 1) == targets.gray_labels(5, 1)).all()
    assert (cqsm.bin_labels(8, 2) == targets.gray_labels(4, 2)).all()
    C = MV.coords(4, 2)
    assert C.shape == (256, 4) and (sum(C[:, c] << (2 * (3 - c)) for c in range(4)) == np.arange(256)).all()
    lab = cqsm.bin_labels(8, 4)
    assert sorted(lab) == list(range(256))
    g = targets.gray_labels(2, 1)
    assert lab[0b01_10_11_00] == (g[1] << 6 | g[2] << 4 | g[3] << 2 | g[0])


def test_cqsm_2d_is_anc2x():
    th = variants.init_theta(0, 0.3, "anc2x", 1, 8)
    Q1 = np.asarray(variants.label_kernel(jnp.asarray(th), "anc2x", 1, 8))
    K = np.asarray(cqsm.label_kraus(jnp.asarray(th), 0, 2, 1, 8, 2, True))
    assert np.abs(Q1 - (np.abs(K) ** 2).sum((2, 3, 4))).max() < 1e-12
    assert cqsm.n_params(0, 2, 3, 8, 2) == variants.n_params("anc2x", 3, 8) == 440
    assert cqsm.n_params(0, 2, 3, 8, 2, False) == variants.n_params("anc2", 3, 8)


def test_bvn_cells():
    e = np.linspace(-3, 3, 7)
    e[0], e[-1] = -MV.BIG, MV.BIG
    P = np.asarray(MV.bvn_cells(0.3, -0.2, 0.8, 1.1, 0.0, e, e, 24))
    p1 = np.diff(norm.cdf(e, 0.3, 0.8))
    p2 = np.diff(norm.cdf(e, -0.2, 1.1))
    assert abs(P.sum() - 1) < 1e-12 and np.abs(P - np.outer(p1, p2)).max() < 1e-9
    P = np.asarray(MV.bvn_cells(0.3, -0.2, 0.8, 1.1, 0.6, e, e, 24))
    mvn = multivariate_normal([0.3, -0.2], [[0.64, 0.6 * 0.88], [0.6 * 0.88, 1.21]])
    cell = mvn.cdf([e[3], e[4]]) - mvn.cdf([e[2], e[4]]) - mvn.cdf([e[3], e[3]]) + mvn.cdf([e[2], e[3]])
    assert abs(P[2, 3] - cell) < 1e-5


def test_truth_and_mixture():
    spec = MV.uniform_spec()
    T = MV.truth_T("msvar", spec)
    assert T.shape == (256, 2, 256, 2) and np.abs(T.sum((2, 3)) - 1).max() < 1e-12
    p = MV.TRUTH["msvar"]
    Kt = np.asarray(MV.ar_kernels(p["c"], p["sig"], p["rho"], spec, MV.GL_MODEL))
    for s in (0, 1):  # a one-component mixture is the regime's binned AR kernel
        th = np.concatenate([np.zeros(2), np.diag([p["c"][s]] * 2).ravel(), np.log([p["sig"][s]] * 2), np.zeros(2),
                             [np.arctanh(p["rho"][s] / 0.99)], [0.0]])
        assert len(th) == MV.n_kernel("mix1", spec)
        assert np.abs(np.asarray(MV.kernel(th, "mix1", spec)) - Kt[s]).max() < 1e-10
    th = MV.init_kernel("mix5", 0, spec, np.arange(256))
    Q = np.asarray(MV.kernel(th, "mix5", spec))
    assert len(th) == MV.n_kernel("mix5", spec) and np.abs(Q.sum(1) - 1).max() < 1e-12
    lin = iolin.IOLin.from_iohmm(T)
    P = np.stack([lin.kernel()[i] for i in range(256)])
    c = MV.pred_corr(P, spec)
    assert np.all(np.abs(c) < 1) and c.mean() > 0.2


def test_rbm_brute_force():
    spec = MV.uniform_spec()
    H, n = 2, 8
    rng = np.random.default_rng(1)
    th = rng.normal(size=MV.n_kernel(f"rbm{H}", spec))
    Q = np.asarray(MV.kernel(th, f"rbm{H}", spec))
    U = MV.gray_bits(spec)
    b, A = th[:n], th[n:n + n * n].reshape(n, n)
    o = n + n * n
    c, B, W = th[o:o + H], th[o + H:o + H + H * n].reshape(H, n), th[o + H + H * n:].reshape(n, H)
    x = 37
    w = np.array([sum(np.exp((b + A @ U[x]) @ U[y] + hh @ (c + B @ U[x] + W.T @ U[y]))
                      for hh in map(np.array, itertools.product([0, 1], repeat=H))) for y in range(256)])
    assert np.abs(Q[x] - w / w.sum()).max() < 1e-12


def test_switch_and_cqsm_rows():
    spec = MV.uniform_spec(2, 2)
    th = MV.init_switch(MV.init_kernel("rbm3", 0, spec, np.arange(16)), 2, 0, "rbm3", spec)
    assert len(th) == MV.sw_n_theta(2, "rbm3", spec)
    M, f = MV.model_M(jnp.asarray(th), "sw-rbm3-S2", spec)
    assert np.abs(np.asarray(M).sum((1, 3)) - 1).max() < 1e-12
    M, f = MV.model_M(jnp.asarray(MV.cq_embed(MV.init_kernel("qa", 0, spec, np.arange(16)), 0, 1, spec)), "cq1", spec)
    lin = iolin.IOLin(np.asarray(M), np.asarray(f))
    assert abs(lin.marginal().sum() - 1) < 1e-10 and np.abs(lin.kernel().sum(1) - 1).max() < 1e-10


def test_gauss_cells_and_copula():
    rng = np.random.default_rng(2)
    e = np.array([-np.inf, -0.7, 0.0, 0.6, np.inf])
    H = np.broadcast_to(e, (2, 4, 5)).copy()
    P = MV.gauss_cells(np.broadcast_to(np.eye(4), (2, 4, 4)), H, 12)
    p1 = np.diff(norm.cdf(e))
    prod = np.einsum("a,b,c,d->abcd", p1, p1, p1, p1).ravel()
    assert np.abs(P[0] / prod - 1).max() < 1e-5
    A = rng.normal(size=(4, 4))
    S = A @ A.T + np.eye(4)
    Rm = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    P = MV.gauss_cells(Rm[None], H[:1], 10)[0].reshape(4, 4, 4, 4)
    P01 = np.asarray(MV.bvn_cells(0.0, 0.0, 1.0, 1.0, Rm[0, 1], np.clip(e, -1e4, 1e4), np.clip(e, -1e4, 1e4), 24))
    assert abs(P.sum() - 1) < 1e-12 and np.abs(P.sum((2, 3)) - P01).max() < 2e-4
    Nz = rng.normal(size=(500, 2))
    assert abs(MV.copula_ll(Nz, np.broadcast_to(np.eye(2), (500, 2, 2)))) < 1e-12
    Qb = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert np.allclose(MV.dcc_R(Nz, 0.0, 0.0, Qb), Qb)


def test_real_dataset():
    ds = MV.real_dataset("eurchf-q")
    a = ds["n_train"]
    assert ds["d"] == 2 and ds["k"] == 4 and ds["edges"].shape == (2, 17)
    assert (ds["b"] == ds["bc"][:, 0] * 16 + ds["bc"][:, 1]).all()
    assert np.allclose(ds["z"][:a].mean(0), 0) and np.allclose(ds["z"][:a].std(0), 1)
    assert ds["corr_train"][0][1] < -0.5
    occ = np.bincount(ds["bc"][:a, 0], minlength=16)
    assert occ.min() > 0.9 * a / 16
    fx = MV.real_dataset("fx4-v")
    assert fx["d"] == 4 and fx["values"].shape == (4, 4) and fx["b"].max() < 256
