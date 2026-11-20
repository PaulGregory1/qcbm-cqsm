"""Memory diagnostics (archb.memdiag): window predictors, frozen interventions, exact classical reduction, probes."""

import jax
import jax.numpy as jnp
import numpy as np

from archb import cqsm, iolin, memdiag, regimes as R

jax.config.update("jax_enable_x64", True)


def _cqsm(m=1, a=1, seed=3):
    return cqsm.init_theta(seed, m, a, n_bits=2, noise=0.9), m, a


def test_classical_reduction_matches_dephased_circuit():
    th, m, a = _cqsm()
    K = np.asarray(cqsm.kraus(jnp.asarray(th), m, a, n_bits=2))
    cl = memdiag.classical_from_kraus(K)
    assert cl.M.min() >= 0 and np.abs(cl.M.sum((1, 3)) - 1).max() < 1e-12
    deph = cqsm.io(th, m, a, n_bits=2, gamma=1.0)
    assert np.abs(memdiag.word_logprobs(cl, 5) - memdiag.word_logprobs(deph, 5)).max() < 1e-9
    xs = deph.sample(3000, 0)
    assert np.abs(cl.logprobs(xs) - deph.logprobs(xs)).max() < 1e-9


def test_word_logprobs_normalized():
    lin = cqsm.io(*_cqsm(m=2, a=0), n_bits=2)
    assert abs(np.exp(memdiag.word_logprobs(lin, 4)).sum() - 1) < 1e-12


def test_deph_matrix_identity_basis_and_projector():
    for m in (1, 2):
        D = 2**m
        M = cqsm.transfer(jnp.asarray(cqsm.init_theta(1, m, 0, n_bits=2, noise=0.7)), m, 0, n_bits=2)
        for lam in (0.0, 0.3, 1.0):
            assert np.abs(memdiag.dephased(M, D, lam) - cqsm.dephase(M, D, lam)).max() < 1e-12
        U = memdiag.unitary(jnp.asarray(np.random.default_rng(m).normal(size=D * D)), D)
        assert np.abs(U @ U.conj().T - np.eye(D)).max() < 1e-12
        P = np.asarray(memdiag.deph_matrix(U, D))
        f = np.asarray(cqsm.f_vec(m))
        assert np.abs(P @ P - P).max() < 1e-12 and np.abs(P @ f - f).max() < 1e-12  # projector, keeps the trace
        lin = iolin.IOLin(np.asarray(memdiag.dephased(M, D, 1.0, U)), f)
        assert abs(np.exp(memdiag.word_logprobs(lin, 3)).sum() - 1) < 1e-12


def test_optimized_basis_not_worse_than_computational():
    th, m, a = _cqsm()
    M, f = cqsm.transfer(jnp.asarray(th), m, a, n_bits=2), cqsm.f_vec(m)
    X = np.stack([cqsm.io(th, m, a, n_bits=2).sample(41, s) for s in range(40)])
    ll0 = float(iolin.loglik(memdiag.dephased(M, 2, 1.0), f, jnp.asarray(X)))
    U, ll = memdiag.optimize_basis(M, f, 2, X, n_starts=2, maxiter=100)
    assert ll >= ll0 - 1e-9


def test_window_predictor_limits():
    lin = iolin.IOLin.from_iohmm(R.truth_T("msar"))
    xs = lin.sample(400, 1)
    full = lin.logprobs(xs)
    k0 = memdiag.window_logprobs(lin, xs, 0, 1)
    assert np.abs(k0 - np.log(lin.kernel()[xs[:-1], xs[1:]])).max() < 1e-10  # k = 0: averaged kernel = p_0
    k = 30
    w = memdiag.window_logprobs(lin, xs, k, k + 1)  # first target: the window starts at x_0 like the full filter
    assert abs(w[0] - full[k + 1]) < 1e-10
    small = memdiag.window_logprobs(lin, xs, k, k + 1, chunk_elems=37)
    assert np.abs(small - w).max() < 1e-12


def test_reset_removes_memory():
    th, m, a = _cqsm()
    lin = cqsm.io(th, m, a, n_bits=2)
    r = lin.u.sum(0)
    rl = iolin.IOLin(memdiag.reset(lin.M, lin.f, r / (r @ lin.f)), lin.f)
    xs = lin.sample(500, 2)
    assert np.abs(memdiag.window_logprobs(rl, xs, 0, 6) - memdiag.window_logprobs(rl, xs, 5, 6)).max() < 1e-10


def test_commutator_is_sine_for_a_qubit():
    def rho(r):
        return 0.5 * (np.eye(2) + r[0] * np.array([[0, 1], [1, 0]]) + r[2] * np.diag([1, -1]))

    a, b = np.array([0.6, 0, 0]), np.array([0.3, 0, 0.3])
    R_ = np.stack([rho(a), rho(b)] * 100)
    c = memdiag.commutators(R_, np.zeros(200, int), n_pairs=2000)
    k = c["kappa_mean"]
    assert 0 < k < 1  # pairs mix equal (kappa 0) and different (sin 45 deg) states
    Cm = rho(a) @ rho(b) - rho(b) @ rho(a)
    kab = np.linalg.norm(Cm) / (np.sqrt(2) * np.linalg.norm(rho(a) - np.eye(2) / 2) * np.linalg.norm(rho(b) - np.eye(2) / 2))
    assert abs(kab - np.sin(np.pi / 4)) < 1e-12


def test_posterior_tools():
    lin = iolin.IOLin.from_iohmm(R.truth_T("msar"))
    xs = lin.sample(4000, 3)
    q = memdiag.regime_posterior(lin, xs, R.regime_of("msar"))
    assert q.min() >= -1e-12 and q.max() <= 1 + 1e-12
    s = (np.random.default_rng(0).random(len(q)) < q).astype(int)
    sc = memdiag.posterior_scores(q, q, s)
    assert sc["mse"] == 0 and abs(sc["kl"]) < 1e-12
    pr = memdiag.fit_probe(memdiag.features(xs, lin.filtered(xs)), q)
    assert np.mean((pr(memdiag.features(xs, lin.filtered(xs))) - q) ** 2) < 1e-3


def test_holevo_and_predictive_info():
    # two orthogonal pure states, equally likely at one x: chi = ln 2; one state: 0; classical beliefs as diag
    R0 = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], complex)
    xs = np.zeros(1000, int)
    R = R0[np.arange(1000) % 2]
    assert abs(memdiag.holevo(R, xs) - np.log(2)) < 1e-12
    assert abs(memdiag.holevo(R0[np.zeros(1000, int)], xs)) < 1e-12
    Rc = memdiag.memory_rhos(np.tile([[0.5, 0.5]], (10, 1)))
    assert abs(memdiag.holevo(Rc, np.zeros(10, int))) < 1e-12
    # chi of a C-QSM's filtered states is within [0, ln D]
    th, m, a = _cqsm()
    lin = cqsm.io(th, m, a, n_bits=2)
    xs = lin.sample(4000, 1)
    chi = memdiag.holevo(memdiag.memory_rhos(lin.filtered(xs), m), xs)
    assert -1e-12 <= chi <= np.log(2) + 1e-12
    # predictive information: a memory that ignores its state gives 0; the null stays near 0
    P = memdiag.predictive(lin, lin.filtered(xs), xs)
    pi = memdiag.predictive_info(P, xs)
    assert pi["model"] >= 0 and abs(pi["null"][0]) < 4 * pi["null"][1] + 1e-3
    flat = np.tile(P.mean(0), (len(P), 1))
    assert abs(memdiag.predictive_info(flat, xs)["model"]) < 1e-3
