"""Sequence 3: quantum sequence model, truth processes, classical baselines and exact metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from archb import hmm, qseq

jax.config.update("jax_enable_x64", True)

SHAPES = [("iso", 1, 1, None), ("iso", 2, 1, None), ("iso", 2, 3, None), ("circuit", 1, 1, 2), ("circuit", 2, 3, 3)]


@pytest.mark.parametrize("kind,m,b,L", SHAPES)
def test_kraus_complete_and_words_normalized(kind, m, b, L):
    K = np.asarray(qseq.kraus(qseq.init_theta(3, kind, m, b, L), kind, m, b, L))
    D = 2**m
    assert K.shape == (2**b, D, D)
    assert np.abs(np.einsum("xji,xjk->ik", K.conj(), K) - np.eye(D)).max() < 1e-12
    rho = np.asarray(qseq.rho_ss(jnp.asarray(K)))
    assert abs(np.trace(rho) - 1) < 1e-12 and np.linalg.eigvalsh(rho).min() > -1e-12
    assert np.abs(sum(k @ rho @ k.conj().T for k in K) - rho).max() < 1e-12
    P = qseq.linear(K).word_probs(3 if b == 3 else 6)
    for l, p in enumerate(P):
        assert abs(p.sum() - 1) < 1e-12 and p.min() > -1e-14
        if l:  # marginals: summing out the last or the first symbol gives the shorter word law (stationarity)
            A = 2**b
            assert np.abs(p.reshape(-1, A).sum(1) - P[l - 1]).max() < 1e-12
            assert np.abs(p.reshape(A, -1).sum(0) - P[l - 1]).max() < 1e-12


def test_circuit_matches_gate_by_gate_statevector():
    m, b, L = 2, 1, 2
    n = m + b
    th = qseq.init_theta(5, "circuit", m, b, L)
    X = np.asarray(qseq.circuit_isometry(th, m, b, L))

    def one(state, U, q):  # apply a 2x2 gate to qubit q (0 = most significant)
        s = np.moveaxis(state.reshape([2] * n), q, 0)
        s = np.tensordot(U, s, axes=(1, 0))
        return np.moveaxis(s, 0, q).reshape(-1)

    def cnot(state, c, t):
        s = state.reshape([2] * n).copy()
        idx = [slice(None)] * n
        idx[c] = 1
        sub = s[tuple(idx)]
        tt = t - (1 if t > c else 0)
        s[tuple(idx)] = np.flip(sub, axis=tt)
        return s.reshape(-1)

    def rz(a):
        return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])

    def ry(a):
        return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])

    t = th.reshape(L + 1, n, 3)
    for s in range(2**m):
        psi = np.zeros(2**n, complex)
        psi[s] = 1.0  # output qubits |0>, memory |s>
        for l in range(L + 1):
            for q in range(n):
                a, b_, c = t[l, q]
                psi = one(psi, rz(c) @ ry(b_) @ rz(a), q)
            if l < L:
                for c_, t_ in [(0, 1), (1, 2), (2, 0)]:
                    psi = cnot(psi, c_, t_)
        assert np.abs(psi - X[:, s]).max() < 1e-12


def test_iso_parametrization_reaches_any_isometry():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    W, _ = np.linalg.qr(Z)
    th = np.concatenate([W.real.ravel(), W.imag.ravel()])
    assert np.abs(np.asarray(qseq.iso_isometry(th, 2, 1)) - W).max() < 1e-12


def test_qmachine_represents_markov_chain_exactly():
    P = np.array([[0.9, 0.1], [0.35, 0.65]])
    K = qseq.qmachine_markov(P)
    assert np.abs(np.einsum("xji,xjk->ik", K.conj(), K) - np.eye(2)).max() < 1e-12
    T = np.zeros((2, 2, 2))
    for s in range(2):
        for x in range(2):
            T[s, x, x] = P[s, x]
    for p, q in zip(hmm.hmm_linear(T).word_probs(8), qseq.linear(K).word_probs(8)):
        assert np.abs(p - q).max() < 1e-12


def test_qsm_fit_recovers_period_two_process():
    """m = 1 QSM trained on 0101... reaches the exact optimum: log 2 for the first symbol (the stationary state does
    not know the phase), 0 for every later one."""
    from archb import ceiling

    xs = np.arange(400) % 2
    data = hmm.blocks(xs, 2)
    loss = lambda t: -qseq.block_loglik(qseq.kraus(t, "iso", 1, 1), data)
    r = ceiling.fit(None, qseq.init_theta(0, "iso", 1, 1), adam_steps=250, lbfgs_iters=500, loss_fn=loss)
    assert abs(r["final_loss"] - np.log(2) / len(xs)) < 1e-7


@pytest.mark.parametrize("kind,m,b,L", [("iso", 2, 1, None), ("circuit", 1, 3, 2)])
def test_block_loglik_equals_symbol_filter(kind, m, b, L):
    xs = np.random.default_rng(1).integers(0, 2**b, 1003)
    K = qseq.kraus(qseq.init_theta(1, kind, m, b, L), kind, m, b, L)
    ref = float(qseq.loglik(K, xs))
    assert abs(float(qseq.block_loglik(K, hmm.blocks(xs, 2**b))) - ref) < 1e-12
    assert abs(float(jnp.mean(qseq.logprobs(K, jnp.asarray(xs)))) - ref) < 1e-12
    # and the filter likelihood equals the exact word probability
    short = xs[:9] if b == 1 else xs[:4]
    assert abs(float(qseq.loglik(K, short)) * len(short) - np.log(qseq.linear(np.asarray(K)).prob(short))) < 1e-10


def test_renewal_matches_monte_carlo():
    N = 5
    T = hmm.renewal(N)
    lin = hmm.hmm_linear(T)
    gaps = hmm.gap_dist(lin, 2 * N)
    assert np.abs(gaps[:N] - 1 / N).max() < 1e-12 and np.abs(gaps[N:]).max() < 1e-12
    xs = hmm.sample(T, 200000, 7)
    ones = np.flatnonzero(xs == 1)
    emp = np.bincount(np.diff(ones) - 1, minlength=N) / (len(ones) - 1)
    assert np.abs(emp - 1 / N).max() < 0.01
    P4 = lin.word_probs(4)[-1]
    w = xs[:-3] * 8 + xs[1:-2] * 4 + xs[2:-1] * 2 + xs[3:]
    assert np.abs(np.bincount(w, minlength=16) / len(w) - P4).max() < 0.005


def test_regime_matches_direct_simulation():
    T = hmm.regime()
    e, _ = hmm.regime_bins()
    rng = np.random.default_rng(3)
    n, s = 400000, 0
    stay = rng.random(n) < hmm.REGIME["stay"]
    reg = np.empty(n, dtype=int)
    for t in range(n):
        reg[t] = s
        s = s if stay[t] else 1 - s
    x = np.array(hmm.REGIME["sig"])[reg] * rng.normal(size=n)
    xb = np.clip(np.searchsorted(e, x, side="right") - 1, 0, 7)
    lin = hmm.hmm_linear(T)
    P2 = lin.word_probs(2)
    assert np.abs(np.bincount(xb, minlength=8) / n - P2[0]).max() < 0.005
    assert np.abs(np.bincount(xb[:-1] * 8 + xb[1:], minlength=64) / (n - 1) - P2[1]).max() < 0.004
    _, c = hmm.regime_bins()
    a = np.abs(c)[xb]
    emp = [np.corrcoef(a[:-k], a[k:])[0, 1] for k in (1, 5, 20)]
    assert np.abs(np.array(emp) - lin.acf(np.abs(c), 20)[[0, 4, 19]]).max() < 0.02


def test_baum_welch_increases_likelihood_and_hmm_block_loglik():
    T = hmm.renewal(4)
    xs = hmm.sample(T, 3000, 2)
    r = hmm.baum_welch(xs, 3, 2, seed=0, max_iter=60, prior=0.0)
    assert np.all(np.diff(r["curve"]) > -1e-12) and r["curve"][-1] > r["curve"][0]
    Tf = r["T"]
    assert abs(Tf.sum((1, 2)) - 1).max() < 1e-12
    ref = hmm.hmm_loglik(Tf, xs)
    assert abs(float(hmm.hmm_block_loglik(jnp.asarray(Tf), hmm.blocks(xs, 2))) - ref) < 1e-12
    assert abs(np.sum(hmm.hmm_logprobs(Tf, xs[:7])) - np.log(hmm.hmm_linear(Tf).prob(xs[:7]))) < 1e-10


def test_markov_chain_word_probs_and_likelihood():
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 3, 5000)
    for k in (1, 2):
        m = hmm.markov_fit(xs, k, 3)
        P = m.word_probs(5)
        for l, p in enumerate(P):
            assert abs(p.sum() - 1) < 1e-12
            if l:
                assert np.abs(p.reshape(-1, 3).sum(1) - P[l - 1]).max() < 1e-12
                assert np.abs(p.reshape(3, -1).sum(0) - P[l - 1]).max() < 1e-12
        lp = hmm.markov_logprobs(m, xs[:12])
        assert abs(lp.sum() - np.log(m.prob(xs[:12]))) < 1e-10
    # order-1 chain = edge-emitting HMM whose state is the last symbol: same words and acf
    m = hmm.markov_fit(xs, 1, 3)
    T = np.zeros((3, 3, 3))
    for s in range(3):
        for x in range(3):
            T[s, x, x] = m.Pc[s, x]
    lin = hmm.hmm_linear(T)
    for p, q in zip(m.word_probs(4), lin.word_probs(4)):
        assert np.abs(p - q).max() < 1e-12
    g = np.array([0.3, 1.0, 2.0])
    assert np.abs(m.acf(g, 5) - lin.acf(g, 5)).max() < 1e-10


def test_kl_curve_chain_rule_and_held_out_rate():
    P = hmm.hmm_linear(hmm.renewal(3)).word_probs(6)
    Q = hmm.markov_fit(hmm.sample(hmm.renewal(3), 5000, 0), 1, 2).word_probs(6)
    kl, D, d = hmm.kl_curve(P, P)
    assert np.abs(kl).max() < 1e-12
    kl, D, d = hmm.kl_curve(P, Q)
    assert np.all(d > -1e-12) and abs(d.sum() - kl[-1]) < 1e-12
    lp = hmm.hmm_logprobs(hmm.renewal(3), hmm.sample(hmm.renewal(3), 2000, 1))
    assert hmm.seg_kl_rate(lp, lp) == (0.0, 0.0)


def test_lump_identity_and_stochastic():
    T = hmm.regime()
    assert np.abs(hmm.lump(T, [0, 1]) - T).max() < 1e-14
    L = hmm.lump(hmm.renewal(6), [0, 1, 2, 2, 2, 2])
    assert L.shape == (3, 2, 3) and np.abs(L.sum((1, 2)) - 1).max() < 1e-12
    # lumping keeps the one-symbol law (the stationary law is preserved)
    assert abs(hmm.hmm_linear(L).prob([1]) - hmm.hmm_linear(hmm.renewal(6)).prob([1])) < 1e-12
