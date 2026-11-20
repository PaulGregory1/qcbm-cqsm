import itertools

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
from scipy.stats import t as student

from archb import iolin
from archb import tails as TL


def test_grids_and_labels():
    for enc in "ug":
        for n, d in ((6, 1), (8, 2)):
            G = TL.grid(enc, n, d)
            assert sorted(G["lab"]) == list(range(G["K"]))
            assert np.all(np.diff(G["edges"]) > 0) and np.all(np.diff(G["values"]) > 0)
    G = TL.grid("g")
    i = np.arange(64)
    assert (G["lab"][32:] == 32 + np.arange(32)).all() and (G["lab"][:32] == 31 - np.arange(32)).all()


def test_geometric_law_is_a_product():
    """On the log-magnitude encoding a law geometric in the magnitude index is a product law over the bits."""
    G = TL.grid("g")
    j = np.where(np.arange(64) >= 32, np.arange(64) - 32, 31 - np.arange(64))
    p = 0.8 ** j
    p /= p.sum()
    U = G["bits"]
    marg = np.array([p @ U[:, q] for q in range(6)])
    prod = np.prod(np.where(U == 1, marg, 1 - marg), axis=1)
    assert np.abs(prod - p).max() < 1e-12


def test_t_cells():
    G = TL.grid("g")
    for mu, s, nu in ((0.3, 0.7, 3.0), (-2.0, 0.05, 1.5), (1.0, 3.0, 30.0)):
        P = np.asarray(TL.t_cells(mu, s, nu, G["edges_m"], 16))
        ex = np.diff(student.cdf((G["edges"] - mu) / s, nu))
        assert np.abs(P - ex).max() < 1e-6


def test_truths():
    for t, d in TL.LEVEL1.items():
        for enc in "ug":
            p = TL.truth_static(t, TL.grid(enc, 6 if d == 1 else 8, d))
            assert abs(p.sum() - 1) < 1e-10 and p.min() >= 0
    G = TL.grid("u", 8, 2)
    P = TL.truth_static("tcop", G).reshape(16, 16)
    assert np.abs(P.sum(1) - np.diff(student.cdf(G["edges"], 3))).max() < 1e-6  # t margins
    assert np.abs(P - P.T).max() < 1e-9
    for t in TL.LEVEL2:
        assert np.abs(TL.truth_kernel(t, TL.grid("g")).sum(1) - 1).max() < 1e-12
    for t in TL.LEVEL3:
        T = TL.truth_T(t, TL.grid("u"))
        assert T.shape == (64, 2, 64, 2) and np.abs(T.sum((2, 3)) - 1).max() < 1e-12
    p = TL.CRASH
    G = TL.grid("u")
    z = np.concatenate([p["mu"], p["c"], np.log(p["sig"]), [np.log(1e8)] * 2,
                        np.log(np.array([1 - p["go"], p["stay"]]) / np.array([p["go"], 1 - p["stay"]]))])
    assert np.abs(np.asarray(TL.ms_family_T(jnp.asarray(z), G)) - TL.truth_T("crash", G)).max() < 1e-5


def test_families():
    for n, d in ((6, 1), (8, 2)):
        G = TL.grid("g", n, d)
        kinds = ["qB0L2", "qA2L1", "rbm4", "mix4", "mps2", "nmps2"] + (["mixt3"] if d == 1 else [])
        for k in kinds:
            th = TL.init(k, 0, G, np.arange(G["K"]), False)
            assert len(th) == TL.n_params(k, G, False)
            p = np.asarray(TL.static_probs(th, k, G))
            assert abs(p.sum() - 1) < 1e-10 and p.min() >= 0
    G = TL.grid("u")
    for k in ("qB0L1", "qA2L2", "rbm3", "mix4", "mixt3"):
        th = TL.init(k, 1, G, np.arange(64), True)
        assert len(th) == TL.n_params(k, G, True)
        assert np.abs(np.asarray(TL.cond_probs(th, k, G)).sum(1) - 1).max() < 1e-10
    th = TL.init_switch(TL.init("rbm3", 0, G, np.arange(64), True), 2, 0, G)
    assert np.abs(np.asarray(TL.switch_M(th, 2, "rbm3", G)).sum((1, 3)) - 1).max() < 1e-10
    from archb import cqsm
    M, f = TL.cqsm_M(cqsm.embed(TL.init("qA2L3", 0, G, np.arange(64), True), 0, 1, 2, 3, 6), 1, G)
    lin = iolin.IOLin(np.asarray(M), np.asarray(f))
    assert np.abs(lin.kernel().sum(1) - 1).max() < 1e-10


def test_static_circuit_and_mps_brute_force():
    G = TL.grid("u", 4, 1)
    rng = np.random.default_rng(3)
    th = rng.normal(size=TL.n_params("mps2", G, False))
    shapes = TL._mps_shapes(2, 4)
    A, o = [], 0
    for sh in shapes:
        A.append(th[o:o + int(np.prod(sh))].reshape(sh))
        o += int(np.prod(sh))
    psi = np.array([(A[0][y[0]] @ A[1][y[1]] @ A[2][y[2]] @ A[3][y[3]])[0, 0]
                    for y in itertools.product([0, 1], repeat=4)])
    p = np.asarray(TL.static_probs(th, "mps2", G))
    assert np.abs(p - (psi**2 / (psi**2).sum())[G["lab"]]).max() < 1e-12
    # the static circuit is the law of input 0 of the conditional circuit with every input-controlled angle at 0
    G = TL.grid("g")
    na, gates, free = TL._circ("qA2", 2, G)
    full = np.zeros(len(gates))
    full[free] = rng.normal(size=int(free.sum()))
    Q = np.asarray(TL.cond_probs(full, "qA2L2", G))
    ps = np.asarray(TL.static_probs(full[free], "qA2L2", G))
    assert np.abs(Q - ps[None, :]).max() < 1e-12


def test_tail_scores():
    p = np.diff(norm.cdf(TL.grid("u")["edges"]))
    A = TL.tail_mask(p, 0.01)
    assert p[A].sum() <= 0.02 + 1e-12 and A[0] and A[-1]
    assert abs(TL.censored_kl(p, p, A)) < 1e-14 and TL.kl(p, p) == 0
    q = p.copy()
    q[~A] *= 1.01
    q /= q.sum()
    assert TL.censored_kl(p, q, A) > 0
