"""Sequence 5 (A1): classical-memory switch between Architecture B kernels, the classical ablation of the C-QSM.

A hidden state s in {0..S-1} replaces the quantum memory. Given the input x_t and the state s_t, the new state s' is
drawn from a table P(s' | s, x), then x_{t+1} from the Architecture B kernel of that state, Q_{s'}[x, x']:

    M[x, x'][s, s'] = P(s' | s, x) Q_{s'}[x, x'],     f = 1                         (archb.iolin, d = S)

P: softmax over s' of logits (S, 2^n, S). Q_{s'}: one circuit per state (archb.cqsm with m = 0: 'anc2' = two traced
ancillas, 'B0' = none), kernels in BIN order. S = 1 is Architecture B. With S = 2 the filter is as large as a C-QSM with
one memory qubit holds classically (equal memory: the diagonal of a qubit); S = 4 matches its filter size (D^2 = 4).

Init: logits = stay on the diagonal plus N(0, 0.01^2), each kernel = the stage 1 Architecture B fit + N(0, noise^2)
(a different perturbation per state, so the states can separate).

Sequence 6: kind may also be a compact classical kernel of archb.ckernels ('mlp<H>', 'mix<K>'), which needs the grid
(bin values and edges); the switch is then entirely classical.
"""

import jax
import jax.numpy as jnp
import numpy as np

from archb import ckernels, cqsm, iolin

jax.config.update("jax_enable_x64", True)

N_ANC = {"anc2": 2, "B0": 0}
L = 3


def n_kernel(kind, n_layers=L, n_bits=5):
    if ckernels.is_classical(kind):
        return ckernels.n_params(kind)
    return cqsm.n_params(0, N_ANC[kind], n_layers, n_bits)


def kernel(theta_k, kind, n_layers=L, n_bits=5, grid=None):
    """One state's kernel Q[x, x'] in BIN order."""
    if ckernels.is_classical(kind):
        return ckernels.kernel(theta_k, kind, grid)
    return cqsm.transfer(theta_k, 0, N_ANC[kind], n_layers, n_bits)[:, :, 0, 0]


def n_theta(S, kind, n_layers=L, n_bits=5):
    """Length of the parameter vector (S logits per row, one of them a gauge)."""
    return S * 2**n_bits * S + S * n_kernel(kind, n_layers, n_bits)


def n_params(S, kind, n_layers=L, n_bits=5):
    """Free parameters: S (S - 1) 2^n transition probabilities and S kernels."""
    return S * 2**n_bits * (S - 1) + S * n_kernel(kind, n_layers, n_bits)


def switch_M(theta, S, kind, n_layers=L, n_bits=5, grid=None):
    """Transfer matrices M (2^n, 2^n, S, S)."""
    nb, nk = 2**n_bits, n_kernel(kind, n_layers, n_bits)
    theta = jnp.asarray(theta)
    P = jax.nn.softmax(theta[:S * nb * S].reshape(S, nb, S), axis=2)  # P[s, x, s']
    off = S * nb * S
    Q = jnp.stack([kernel(theta[off + k * nk: off + (k + 1) * nk], kind, n_layers, n_bits, grid)
                   for k in range(S)])  # Q[s', x, x']
    return jnp.einsum("sxt,txy->xyst", P, Q)


def init_switch(theta_kernel, S, seed, kind, noise=0.1, stay=3.0, n_bits=5):
    rng = np.random.default_rng(seed)
    nb = 2**n_bits
    logits = stay * np.broadcast_to(np.eye(S)[:, None, :], (S, nb, S)) + 0.01 * rng.normal(size=(S, nb, S))
    kern = [np.asarray(theta_kernel) + noise * rng.normal(size=len(theta_kernel)) for _ in range(S)]
    th = np.concatenate([logits.ravel(), *kern])
    assert len(th) == n_theta(S, kind, n_bits=n_bits)
    return th


def loglik(theta, X, S, kind, n_layers=L, n_bits=5, grid=None):
    return iolin.loglik(switch_M(theta, S, kind, n_layers, n_bits, grid), jnp.ones(S), X)


def io(theta, S, kind, n_layers=L, n_bits=5, grid=None):
    return iolin.IOLin(np.asarray(switch_M(jnp.asarray(theta), S, kind, n_layers, n_bits, grid)), np.ones(S))
