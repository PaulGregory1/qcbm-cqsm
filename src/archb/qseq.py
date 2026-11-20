"""Sequence 3: quantum sequence model (QSM) with a persistent memory register, after X. Wang, C. Yang,
C. A. Somasundaram, J. Thompson, M. Gu, arXiv:2603.24069 (2026).

Memory S: m qubits (D = 2^m), never measured. At each step one shared unitary V(theta) acts on (output, S); the output
register (b qubits, A = 2^b symbols) starts in |0>, is measured in the computational basis and discarded:

    K_x = (<x|_out (x) I_S) V (|0>_out (x) I_S),   sum_x K_x^dag K_x = I        (one D x D matrix per symbol x)

Qubit order: output qubits first (most significant), then memory, so K_x[s', s] = V[x D + s', s].
Two parametrizations:
  circuit  L layers of (R_z R_y R_z on every qubit, then a CNOT ring 0->1->..->n-1->0, a single CNOT for n = 2),
           then a final rotation layer: 3 n (L + 1) angles, n = b + m.
  iso      the general model of memory dimension D: the stacked Kraus operators W = M R^{-1} with M (A D x D) complex
           and R^dag R = M^dag M (Cholesky), so W^dag W = I. M = W reaches any isometry W, and every isometry is
           V (|0> (x) I) for some unitary V: the best any circuit with m memory qubits can do. 2 A D^2 real numbers.
Labels: the circuit's bit string is a label; `perm[x]` is the label of symbol x (Gray code for binned values), and
kraus() returns the operators in symbol order.

Stationary memory state rho_ss: fixed point of Phi(rho) = sum_x K_x rho K_x^dag (linear solve with Tr rho = 1).
Log-likelihood: forward filter rho <- K_x rho K_x^dag / p_t from rho_ss, log L = sum_t log p_t (jax.lax.scan).
"""

import jax
import jax.numpy as jnp
import numpy as np

from archb.hmm import Linear

jax.config.update("jax_enable_x64", True)


def n_params(kind, m, b, n_layers=None):
    n, D, A = m + b, 2**m, 2**b
    return 3 * n * (n_layers + 1) if kind == "circuit" else 2 * A * D * D


def n_effective(m, b):
    """Real dimension of the set of QSM processes reachable with memory D, an upper bound: isometries C^D -> C^{AD}
    (2 A D^2 - D^2) minus the memory gauge K_x -> U K_x U^dag (D^2 - 1) and a phase per K_x (A)."""
    D, A = 2**m, 2**b
    return 2 * A * D * D - 2 * D * D + 1 - A


def _rot(a, b_, c):
    """R_z(c) R_y(b) R_z(a) (R_z(a) applied first)."""
    rz = lambda t: jnp.array([[jnp.exp(-0.5j * t), 0], [0, jnp.exp(0.5j * t)]])
    ry = jnp.array([[jnp.cos(b_ / 2), -jnp.sin(b_ / 2)], [jnp.sin(b_ / 2), jnp.cos(b_ / 2)]])
    return rz(c) @ ry @ rz(a)


def _cnot_ring(n):
    """Permutation matrix of CNOT(0,1) .. CNOT(n-2,n-1) then CNOT(n-1,0) (n > 2); qubit 0 = most significant."""
    def cnot(c, t):
        idx = np.arange(2**n)
        cbit = (idx >> (n - 1 - c)) & 1
        P = np.zeros((2**n, 2**n))
        P[idx ^ (cbit << (n - 1 - t)), idx] = 1.0
        return P

    pairs = [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    M = np.eye(2**n)
    for c, t in pairs:
        M = cnot(c, t) @ M
    return M


def circuit_isometry(theta, m, b, n_layers):
    """V (|0>_out (x) I_S): the first D columns of V, shape (2^n, D)."""
    n, D = m + b, 2**m
    th = jnp.asarray(theta).reshape(n_layers + 1, n, 3)
    C = jnp.asarray(_cnot_ring(n), dtype=jnp.complex128)
    X = jnp.eye(2**n, D, dtype=jnp.complex128)
    for l in range(n_layers + 1):
        R = _rot(*th[l, 0])
        for q in range(1, n):
            R = jnp.kron(R, _rot(*th[l, q]))
        X = R @ X
        if l < n_layers:
            X = C @ X
    return X


def iso_isometry(theta, m, b):
    D, A = 2**m, 2**b
    t = jnp.asarray(theta).reshape(2, A * D, D)
    M = t[0] + 1j * t[1]
    Lc = jnp.linalg.cholesky(M.conj().T @ M)  # M^dag M = L L^dag, R = L^dag
    return jax.scipy.linalg.solve_triangular(Lc.conj(), M.T, lower=True).T  # M (L^dag)^{-1}


def kraus(theta, kind, m, b, n_layers=None, perm=None):
    """(A, D, D) Kraus operators in symbol order."""
    D, A = 2**m, 2**b
    W = circuit_isometry(theta, m, b, n_layers) if kind == "circuit" else iso_isometry(theta, m, b)
    K = W.reshape(A, D, D)
    return K if perm is None else K[jnp.asarray(perm)]


def init_theta(seed, kind, m, b, n_layers=None):
    rng = np.random.default_rng(seed)
    if kind == "circuit":
        return rng.uniform(0, 2 * np.pi, n_params(kind, m, b, n_layers))
    return rng.normal(size=n_params(kind, m, b))


def rho_ss(K):
    """Fixed point of sum_x K_x rho K_x^dag with Tr rho = 1 (row-major vec: vec(K rho K^dag) = (K (x) conj K) vec rho)."""
    D = K.shape[1]
    Sop = jnp.einsum("xij,xkl->ikjl", K, K.conj()).reshape(D * D, D * D)
    M = (Sop - jnp.eye(D * D)).at[0].set(jnp.eye(D).ravel())
    r = jnp.linalg.solve(M, jnp.zeros(D * D).at[0].set(1.0)).reshape(D, D)
    return 0.5 * (r + r.conj().T)


def _filter(K, rho, ops):
    def step(r, k):
        r = k @ r @ k.conj().T
        p = jnp.maximum(jnp.real(jnp.trace(r)), 1e-300)
        return r / p, jnp.log(p)

    return jax.lax.scan(step, rho, ops)


def loglik(K, xs):
    """Mean log-likelihood per symbol of the string xs, filtered symbol by symbol from rho_ss (reference form)."""
    return jnp.mean(_filter(K, rho_ss(K), K[jnp.asarray(xs)])[1])


@jax.jit
def logprobs(K, xs):
    """log P(x_t | x_<t) for every t, filtered from rho_ss."""
    return _filter(K, rho_ss(K), K[xs])[1]


def block_loglik(K, data):
    """Same value as loglik, filtered block by block (data = archb.hmm.blocks(xs, A)): the Kraus operator of a block
    x_1..x_B is K_{x_B} ... K_{x_1}, precomputed for all A^B words."""
    tab = K
    for _ in range(data["B"] - 1):
        tab = jnp.einsum("xij,wjk->wxik", K, tab).reshape(-1, K.shape[1], K.shape[2])
    r, lp = _filter(K, rho_ss(K), tab[data["ids"]])
    tot = jnp.sum(lp)
    for x in data["tail"]:
        r = K[x] @ r @ K[x].conj().T
        p = jnp.real(jnp.trace(r))
        tot, r = tot + jnp.log(p), r / p
    return tot / data["n"]


def linear(K):
    """Exact linear representation (archb.hmm.Linear) of the QSM process for numpy metrics."""
    K = np.asarray(K)
    D = K.shape[1]
    r = np.asarray(rho_ss(jnp.asarray(K)))
    E = [np.kron(k, k.conj()).T for k in K]
    return Linear(r.ravel(), E, np.eye(D).ravel())


def qmachine_markov(P):
    """Exact QSM with one memory qubit for a binary first-order Markov chain P[s, x] (s = last symbol): memory
    states |sigma_s> with <sigma_0|sigma_1> = sum_x sqrt(P[0, x] P[1, x]), K_x = |sigma_x><phi_x| with
    <phi_x|sigma_s> = sqrt(P[s, x]). A test of the formalism, not used for training."""
    c = float(np.sum(np.sqrt(P[0] * P[1])))
    Sg = np.array([[1.0, c], [0.0, np.sqrt(1 - c * c)]])  # columns: |sigma_0>, |sigma_1>
    Sinv = np.linalg.inv(Sg)
    return np.stack([np.outer(Sg[:, x], np.sqrt(P[:, x]) @ Sinv) for x in range(2)]).astype(complex)


# ------------------------------------------------------------------------------- sequence 5: Kraus rank r, dephasing
# A stochastic QSM: r Kraus operators per symbol (r - 1 traced ancilla outcomes), K[x, c] (A, r, D, D), and an optional
# dephasing of the memory after every step, rho -> (1 - gamma) rho + gamma diag(rho). The per-symbol transfer matrix
# sum_c (K (x) conj K)^T is written in the real coordinates of Hermitian matrices (archb.cqsm.herm_basis); dephasing
# scales the off-diagonal coordinates by 1 - gamma. gamma = 1 with r = D^2 is the class of HMMs with D states; the
# classes are nested in gamma (dephasing composes) and in r.
def iso_isometry_r(theta, m, b, r, real=False):
    """K (A, r, D, D): the stacked operators W = M R^-1 (archb.qseq.iso_isometry) with A r D rows; real=True: M real
    (A r D^2 parameters), else complex (2 A r D^2). r = 1 gives the same operators as kraus(theta, 'iso', m, b)."""
    D, A = 2**m, 2**b
    n = A * r * D
    t = jnp.asarray(theta)
    M = t.reshape(n, D) if real else t.reshape(2, n, D)[0] + 1j * t.reshape(2, n, D)[1]
    Lc = jnp.linalg.cholesky(M.conj().T @ M)
    W = jax.scipy.linalg.solve_triangular(Lc.conj(), M.T, lower=True).T
    return W.reshape(A, r, D, D)


def n_params_r(m, b, r, real=False):
    return (1 if real else 2) * 2**b * r * 4**m


def superops(K, gamma=0.0):
    """Real transfer matrices (A, D^2, D^2) of the Kraus table K (A, r, D, D), memory dephased by gamma after each step."""
    from archb.cqsm import dephase, herm_basis

    K = jnp.asarray(K)
    A, D = K.shape[0], K.shape[-1]
    M = jnp.einsum("xcij,xckl->xjlik", K, K.conj()).reshape(A, D * D, D * D)
    C = herm_basis(D)
    M = jnp.real(jnp.einsum("ab,xbc,cd->xad", jnp.asarray(np.linalg.inv(C)), M, jnp.asarray(C)))
    return dephase(M, D, gamma)


def trace_vec(D):
    """vec(I) in the real coordinates: f with a @ f = Tr rho."""
    from archb.cqsm import herm_basis

    return jnp.asarray(np.real(np.linalg.inv(herm_basis(D)) @ np.eye(D).ravel()))


def sup_stationary(Ms, f):
    """Stationary row vector a (a sum_x Ms[x] = a, a f = 1)."""
    P = Ms.sum(0)
    G = (P.T - jnp.eye(P.shape[0])).at[0].set(f)
    return jnp.linalg.solve(G, jnp.zeros(P.shape[0]).at[0].set(1.0))


def sup_block_loglik(Ms, f, data):
    """Mean log-likelihood per symbol from the stationary state, block by block (data = archb.hmm.blocks)."""
    tab = Ms
    for _ in range(data["B"] - 1):
        tab = jnp.einsum("wij,xjk->wxik", tab, Ms).reshape(-1, Ms.shape[1], Ms.shape[2])

    def step(a, Mw):
        a = a @ Mw
        p = jnp.maximum(a @ f, 1e-300)
        return a / p, jnp.log(p)

    a, lp = jax.lax.scan(step, sup_stationary(Ms, f), tab[data["ids"]])
    tot = jnp.sum(lp)
    for x in data["tail"]:
        a = a @ Ms[x]
        p = a @ f
        tot, a = tot + jnp.log(p), a / p
    return tot / data["n"]


@jax.jit
def sup_logprobs(Ms, f, xs):
    """log P(x_t | x_<t) for every t, filtered from the stationary state."""
    def step(a, x):
        a = a @ Ms[x]
        p = jnp.maximum(a @ f, 1e-300)
        return a / p, jnp.log(p)

    return jax.lax.scan(step, sup_stationary(Ms, f), xs)[1]


def qsm_r_loglik(theta, m, b, r, gamma, data, real=False):
    """Training objective of the stochastic QSM: the Kraus filter for r = 1, gamma = 0 (cheaper), else the superoperator
    filter; both give the same value."""
    K = iso_isometry_r(theta, m, b, r, real)
    if r == 1 and not gamma:
        return block_loglik(K[:, 0], data)
    return sup_block_loglik(superops(K, gamma), trace_vec(2**m), data)


def sup_linear(Ms, f):
    """archb.hmm.Linear of the superoperator representation (real), for the exact metrics."""
    Ms, f = jnp.asarray(Ms), jnp.asarray(f)
    return Linear(np.asarray(sup_stationary(Ms, f)), list(np.asarray(Ms)), np.asarray(f))
