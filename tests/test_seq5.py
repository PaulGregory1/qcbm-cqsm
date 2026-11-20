"""Sequence 5: stochastic / dephased QSM, the FLW qubit process, Hankel ranks, exact-law HMM fits, the classical switch,
wall-clock caps and the runner's resumability."""

import importlib.util
import itertools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from archb import cqsm, hmm, qseq, switch, train

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]


def _rand_hmm(S, A, seed):
    return np.random.default_rng(seed).dirichlet(np.ones(A * S), size=S).reshape(S, A, S)


# ------------------------------------------------------------------------------- 1-2: superoperator filter
@pytest.mark.parametrize("m", [1, 2])
def test_r1_superops_equal_kraus_loglik(m):
    th = qseq.init_theta(3, "iso", m, 1)
    K = qseq.iso_isometry_r(th, m, 1, 1)
    assert np.abs(np.asarray(K[:, 0] - qseq.kraus(th, "iso", m, 1))).max() < 1e-14
    xs = hmm.sample(hmm.renewal(5), 3001, 1)
    ref = float(qseq.loglik(K[:, 0], xs))
    data = hmm.blocks(xs, 2)
    sup = float(qseq.sup_block_loglik(qseq.superops(K), qseq.trace_vec(2**m), data))
    assert abs(sup - ref) < 1e-10
    lp = qseq.sup_logprobs(qseq.superops(K), qseq.trace_vec(2**m), jnp.asarray(xs))
    assert abs(float(jnp.mean(lp)) - ref) < 1e-10


@pytest.mark.parametrize("m,r,real", [(1, 3, False), (2, 16, False), (1, 2, True)])
def test_iso_r_complete_and_words_normalized(m, r, real):
    th = np.random.default_rng(0).normal(size=qseq.n_params_r(m, 1, r, real))
    K = np.asarray(qseq.iso_isometry_r(th, m, 1, r, real))
    assert K.shape == (2, r, 2**m, 2**m)
    assert np.abs(np.einsum("xcij,xcik->jk", K.conj(), K) - np.eye(2**m)).max() < 1e-12
    for g in (0.0, 0.3, 1.0):
        P = qseq.sup_linear(qseq.superops(K, g), qseq.trace_vec(2**m)).word_probs(6)
        assert abs(P[-1].sum() - 1) < 1e-12 and P[-1].min() > -1e-14
        assert np.abs(P[-1].reshape(32, 2).sum(1) - P[-2]).max() < 1e-12


@pytest.mark.parametrize("S", [2, 4])
def test_hmm_embedding_dephased(S):
    """K_{x,(s,s')} = sqrt(T[s, x, s']) |s'><s| (r = S^2) and K_{x,c}[s', s] = delta_cs sqrt(T[s, x, s']) (r = S, each
    operator reads one diagonal entry) with gamma = 1 give the HMM's filter."""
    T = _rand_hmm(S, 2, S)
    xs = hmm.sample(T, 2000, 0)
    ref = hmm.hmm_loglik(T, xs)
    K2 = np.zeros((2, S * S, S, S))
    K1 = np.zeros((2, S, S, S))
    for x, s, t in itertools.product(range(2), range(S), range(S)):
        K2[x, s * S + t, t, s] = np.sqrt(T[s, x, t])
        K1[x, s, t, s] = np.sqrt(T[s, x, t])
    f, data = qseq.trace_vec(S), hmm.blocks(xs, 2)
    for K in (K2, K1):
        assert abs(float(qseq.sup_block_loglik(qseq.superops(jnp.asarray(K), 1.0), f, data)) - ref) < 1e-12


# ------------------------------------------------------------------------------- 3: FLW
def test_flw_instrument_and_spectrum():
    lam, al = 0.8, 1.0
    K = hmm.flw_qubit(lam, al)
    assert np.abs(sum(k.conj().T @ k for k in K) - np.eye(2)).max() < 1e-12
    assert np.linalg.eigvalsh(sum(k.conj().T @ k for k in K)).max() <= 1 + 1e-12
    assert np.linalg.matrix_rank(K[1], tol=1e-10) == 1  # a: Kraus rank 1, resets to |+>
    ev = np.sort_complex(np.linalg.eigvals(np.kron(K[0], K[0].conj())))
    target = np.sort_complex(np.array([lam, lam, lam * np.exp(1j * al), lam * np.exp(-1j * al)]))
    assert np.abs(ev - target).max() < 1e-10
    # P(b^n | a) = lam^n (c0 + 2 Re c e^{i n alpha}): the oscillating part is present
    xi = np.ones(2) / np.sqrt(2)
    rho, Sn = np.outer(xi, xi), []
    for n in range(31):
        Sn.append(np.trace(rho).real)
        rho = K[0] @ rho @ K[0].conj().T
    n = np.arange(31)
    B = np.stack([np.ones(31), np.cos(n * al), np.sin(n * al)], axis=1)
    c, *_ = np.linalg.lstsq(B, np.asarray(Sn) / lam**n, rcond=None)
    assert np.abs(B @ c - np.asarray(Sn) / lam**n).max() < 1e-9
    assert 0.5 * np.hypot(c[1], c[2]) > 1e-3


def test_flw_sampler_matches_word_probs():
    lin = qseq.linear(hmm.flw_qubit())
    xs = hmm.sample_linear(lin, 200000, 7)
    P3 = lin.word_probs(3)[-1]
    idx = xs[:-2] * 4 + xs[1:-1] * 2 + xs[2:]
    emp = np.bincount(idx, minlength=8) / len(idx)
    se = np.sqrt(P3 * (1 - P3) / len(idx)) * 4  # loose: neighbouring windows are dependent
    assert np.all(np.abs(emp - P3) < 4 * se + 1e-4)


# ------------------------------------------------------------------------------- 4: Hankel ranks
def test_hankel_ranks():
    assert hmm.hankel_rank(hmm.hmm_linear(_rand_hmm(3, 2, 11)), L=8)[0] == 3
    assert hmm.hankel_rank(hmm.hmm_linear(hmm.renewal(5)), gap=True)[0] == 5
    assert hmm.hankel_rank(hmm.hmm_linear(hmm.renewal(5)), L=12)[0] == 5
    flw = qseq.linear(hmm.flw_qubit())
    assert hmm.hankel_rank(flw, gap=True)[0] == 3
    assert hmm.hankel_rank(flw, L=12)[0] == 3


# ------------------------------------------------------------------------------- 5: switch, C-QSM dephasing
def test_switch_one_state_is_architecture_b():
    X = np.random.default_rng(1).integers(0, 32, size=(5, 12))
    thk = cqsm.init_theta(2, 0, 2, noise=0.3)
    th = switch.init_switch(thk, 1, 3, "anc2", noise=0.0)
    assert abs(float(switch.loglik(th, X, 1, "anc2")) - float(cqsm.loglik(thk, X, 0, 2))) < 1e-12
    th2 = switch.init_switch(thk, 2, 3, "anc2")
    M = np.asarray(switch.switch_M(th2, 2, "anc2"))
    assert np.abs(M.sum(axis=(1, 3)) - 1).max() < 1e-12  # sum over x', s' = 1


def test_cqsm_dephasing():
    th = cqsm.init_theta(4, 1, 0, n_bits=2, noise=0.6)
    a = np.asarray(cqsm.transfer_from_kraus(cqsm.kraus(th, 1, 0, n_bits=2)))
    assert np.array_equal(np.asarray(cqsm.transfer(th, 1, 0, n_bits=2)), a)
    assert np.array_equal(np.asarray(cqsm.transfer(th, 1, 0, n_bits=2, gamma=0.0)), a)
    # memory angles 0: the memory is untouched, gamma = 1 changes nothing, the model is Architecture B
    X = np.random.default_rng(2).integers(0, 4, size=(10, 8))
    thb = cqsm.init_theta(5, 0, 1, n_bits=2, noise=0.5)
    th1 = np.zeros(cqsm.n_params(1, 1, n_bits=2))
    th1[~cqsm.mem_mask(1, 1, n_bits=2)] = thb
    ref = float(cqsm.loglik(thb, X, 0, 1, n_bits=2))
    for g in (0.0, 1.0):
        assert abs(float(cqsm.loglik(th1, X, 1, 1, n_bits=2, gamma=g)) - ref) < 1e-12


# ------------------------------------------------------------------------------- 6: caps
def test_capped_fits_return_finite():
    xs = hmm.sample(hmm.renewal(8), 50000, 3)
    em = hmm.baum_welch(xs, 16, 2, seed=0, max_iter=10**6, tol=0.0, max_seconds=2)
    assert em["capped"] and np.isfinite(em["loglik"])
    t0 = time.time()
    T, r = hmm.hmm_polish(em["T"], xs, lbfgs_iters=10**6, max_seconds=2)
    assert r["capped"] and np.isfinite(r["final_loss"]) and time.time() - t0 < 30
    assert np.abs(T.sum((1, 2)) - 1).max() < 1e-12
    data = hmm.blocks(xs, 2)
    loss = lambda t: -qseq.qsm_r_loglik(t, 1, 1, 2, 0.0, data)
    res = train.fit_capped(loss, np.random.default_rng(0).normal(size=qseq.n_params_r(1, 1, 2)), 0, 10**6, 0.05)
    assert res["capped"] and np.isfinite(res["final_loss"])


def test_runner_skips_completed(tmp_path):
    spec = importlib.util.spec_from_file_location("run_seq5", ROOT / "scripts" / "run_seq5.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    rs.configure(tmp_path, smoke=True)
    job = rs.Job(1, "e2b", "renewal16-s0", "markov-2", 0, 60)
    rs.prepare_data(["renewal16-s0"])
    assert rs.run_job(job)[1] == "ran"
    assert (tmp_path / "e2b" / "renewal16-s0" / "markov-2" / "0" / "result.json").exists()
    assert np.load(tmp_path / "e2b" / "renewal16-s0" / "markov-2" / "0" / "lp_test.npy").dtype == np.float32
    assert rs.run_job(job)[1] == "done"


# ------------------------------------------------------------------------------- 7: exact-law objective
def test_exact_objective_equals_kl_curve():
    P = hmm.hmm_linear(_rand_hmm(4, 2, 1)).word_probs(8)
    T = _rand_hmm(3, 2, 2)
    _, D, _ = hmm.kl_curve(P, hmm.hmm_linear(T).word_probs(8))
    assert abs(float(hmm.exact_objective(P[-1], 8)(jnp.asarray(T))) - D[-1]) < 1e-12
    T2, info = hmm.exact_fit(P[-1], 8, T, maxiter=50)
    assert info["D_L"] < info["D_L_init"] and not info["capped"]


def test_hazard_chain_exact_for_short_renewal():
    g = hmm.gap_dist(hmm.hmm_linear(hmm.renewal(5)), 40)
    Tc = hmm.hazard_chain(g, 6)
    assert np.abs(hmm.gap_dist(hmm.hmm_linear(Tc), 40) - g).max() < 1e-12


def test_word_law_valid_for_reducible_chains():
    T = _rand_hmm(3, 2, 4)
    T[2] = 0.0
    T[2, 0, 2] = 1.0  # state 2 absorbing, emits 0
    for L in (1, 6):
        Q = np.asarray(hmm.word_law(jnp.asarray(T), L))
        assert abs(Q.sum() - 1) < 1e-12 and Q.min() >= 0
    lin = hmm.hmm_linear_abel(T)
    assert lin.v.min() >= 0 and abs(lin.v.sum() - 1) < 1e-12
