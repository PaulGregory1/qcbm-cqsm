"""Sequence 4: conditional QCBM with a quantum memory (C-QSM).

Register: input (n bits, classical controls only, holds x_t), output (n qubits, starts in |0>, measured to give x_{t+1},
then reset), memory (m qubits, never measured, carried over), ancillas (traced at every step, as in sequence 1).
For each input x the circuit V_x(theta) gives Kraus operators on the memory, one per (output, ancilla) outcome:

    K[x, x', c] = (<x', c|_{out, anc} (x) I_mem) V_x(theta) (|0, 0>_{out, anc} (x) I_mem),   sum_{x', c} K^dag K = I

and the transfer matrices of archb.iolin, M[x, x'] = sum_c (K (x) conj K)^T, f = vec(I_D) (D = 2^m), written in real
coordinates of Hermitian matrices (diagonal, then Re and Im of each upper entry): the map rho -> K rho K^dag keeps rho
Hermitian, so the similarity C^-1 M C is real (exact; a quarter of the complex arithmetic). rho_of() maps back.

Gates (one angle each; register qubits: input q -> q, output r -> n + r, memory k -> 2n + k, ancilla a -> 2n + m + a).
Layers l = 0..L-1, then a final R_y layer on the outputs:
  Architecture B layer, as archb.variants: R_y on outputs (n), input q -> output r controlled R_y (n^2, q outer),
      output chain r <-> r+1 (2(n-1)); then per ancilla the anc block of archb.variants (R_y, input -> a, a -> output,
      output -> a).
  memory block, per memory qubit k: R_z R_y R_z (complex amplitudes), input q -> k controlled R_y (n), k -> output r
      controlled R_y (n), output r -> k controlled R_y (n); for m = 2 also mem 0 -> mem 1 and mem 1 -> mem 0.
With every memory angle at 0 the memory is untouched and the model is Architecture B (B0 for 0 ancillas, anc1 / anc2),
gate for gate; with every input-controlled angle at 0 the channel does not depend on x: a QSM (sequence 3).
Bins and labels as in sequences 1-2: Gray labels, kernels and Kraus tables in BIN order.

Dephasing (sequence 5): gamma > 0 follows each step by rho -> (1 - gamma) rho + gamma diag(rho) on the memory (dephase);
gamma = 1 leaves a classical memory driven by the same circuit.

Init: spec angles N(0, noise^2), copy gates (layer 0, input q -> output q) = pi, layer-0 R_y on each ancilla and on
each memory qubit + pi/2 (so their controlled gates have gradients). Staged training starts the memory block from a
fitted Architecture B (embed) with small random memory angles.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from archb import iolin
from archb.targets import gray_labels
from archb.variants import _crot_flat, _input_bits, _rot_flat

jax.config.update("jax_enable_x64", True)


def gate_list(m, n_anc, n_layers=3, n_bits=5, d=1, cross=True):
    """[(kind, ctrl or None, target, tag)] in application order. tag: 'copy', 'anc_plus', 'mem_plus' or None.
    Memory gates are those touching a memory qubit (mem_mask). d > 1 coordinates of n_bits / d bits each (sequence 9):
    with cross, output bit r of coordinate a <-> output bit r of coordinate b for every pair a < b, after the chain
    (for d = 2 the anc2x gates of archb.variants)."""
    n = n_bits
    h = n // d
    IN, OUT = list(range(n)), [n + r for r in range(n)]
    MEM, ANC = [2 * n + k for k in range(m)], [2 * n + m + a for a in range(n_anc)]
    g = []
    for l in range(n_layers):
        g += [("ry", None, OUT[r], None) for r in range(n)]
        for q in range(n):
            g += [("ry", IN[q], OUT[r], "copy" if (l == 0 and q == r) else None) for r in range(n)]
        for r in range(n - 1):
            g += [("ry", OUT[r], OUT[r + 1], None), ("ry", OUT[r + 1], OUT[r], None)]
        if d > 1 and cross:
            for ca in range(d):
                for cb in range(ca + 1, d):
                    for r in range(h):
                        g += [("ry", OUT[ca * h + r], OUT[cb * h + r], None), ("ry", OUT[cb * h + r], OUT[ca * h + r], None)]
        for a in ANC:
            g.append(("ry", None, a, "anc_plus" if l == 0 else None))
            g += [("ry", IN[q], a, None) for q in range(n)]
            g += [("ry", a, OUT[r], None) for r in range(n)]
            g += [("ry", OUT[r], a, None) for r in range(n)]
        for k in MEM:
            g += [("rz", None, k, None), ("ry", None, k, "mem_plus" if l == 0 else None), ("rz", None, k, None)]
            g += [("ry", IN[q], k, None) for q in range(n)]
            g += [("ry", k, OUT[r], None) for r in range(n)]
            g += [("ry", OUT[r], k, None) for r in range(n)]
        if m == 2:
            g += [("ry", MEM[0], MEM[1], None), ("ry", MEM[1], MEM[0], None)]
    g += [("ry", None, OUT[r], None) for r in range(n)]
    return g


def n_params(m, n_anc, n_layers=3, n_bits=5, d=1, cross=True):
    return len(gate_list(m, n_anc, n_layers, n_bits, d, cross))


def mem_mask(m, n_anc, n_layers=3, n_bits=5, d=1, cross=True):
    lo = 2 * n_bits
    return np.array([(c is not None and lo <= c < lo + m) or lo <= t < lo + m
                     for _, c, t, _ in gate_list(m, n_anc, n_layers, n_bits, d, cross)])


def input_mask(m, n_anc, n_layers=3, n_bits=5, d=1, cross=True):
    return np.array([c is not None and c < n_bits for _, c, t, _ in gate_list(m, n_anc, n_layers, n_bits, d, cross)])


def _rz_flat(psi, n_q, t, ang):
    B = psi.shape[0]
    x = psi.reshape(B, 2**t, 2, 2 ** (n_q - t - 1))
    ph = jnp.exp(-0.5j * ang)[:, None, None]
    return jnp.stack([ph * x[:, :, 0, :], jnp.conj(ph) * x[:, :, 1, :]], axis=2).reshape(B, 2**n_q)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
def label_kraus(theta, m, n_anc, n_layers=3, n_bits=5, d=1, cross=True):
    """K[x, x', c, i, j] in LABEL order, shape (2^n, 2^n, 2^n_anc, D, D): memory |j> -> |i>, outcome (x', c).
    Simulated state: (input label, memory column j) x (outputs, memory, ancillas); input-controlled R_y on the same
    qubit are folded into one per-input rotation (exact, as archb.variants)."""
    n, D, nb = n_bits, 2**m, 2**n_bits
    n_q = n + m + n_anc
    gates = gate_list(m, n_anc, n_layers, n_bits, d, cross)
    pos = lambda q: q - n  # output r -> r, memory k -> n + k, ancilla -> n + m + a
    bits = np.repeat(_input_bits(n), D, axis=0)  # (nb D, n)
    psi0 = np.zeros((nb * D, 2**n_q), dtype=complex if m else float)
    psi0[np.arange(nb * D), np.tile(np.arange(D), nb) << n_anc] = 1.0
    psi, theta = jnp.asarray(psi0), jnp.asarray(theta, jnp.float64)
    pending = {}

    def flush(q):
        nonlocal psi
        if q in pending:
            psi = _rot_flat(psi, n_q, pos(q), pending.pop(q))

    for k, (kind, c, t, _) in enumerate(gates):
        if kind == "rz":
            flush(t)
            psi = _rz_flat(psi, n_q, pos(t), jnp.full(nb * D, theta[k]))
        elif c is None or c < n:
            mask = jnp.asarray(bits[:, c]) if c is not None else jnp.ones(nb * D)
            pending[t] = pending.get(t, 0.0) + theta[k] * mask
        else:
            flush(c)
            flush(t)
            psi = _crot_flat(psi, n_q, pos(c), pos(t), jnp.full(nb * D, theta[k]))
    for q in list(pending):
        flush(q)
    return psi.reshape(nb, D, nb, D, 2**n_anc).transpose(0, 2, 4, 3, 1)


def bin_labels(n_bits, d=1):
    """Label of each bin: Gray code per coordinate (n_bits / d bits each), concatenated MSB first; bins row-major."""
    h = n_bits // d
    g = gray_labels(h, 1)
    i = np.arange(2**n_bits)
    return sum(g[(i >> (h * (d - 1 - c))) & (2**h - 1)] << (h * (d - 1 - c)) for c in range(d))


def kraus(theta, m, n_anc, n_layers=3, n_bits=5, d=1, cross=True):
    """Kraus table in BIN order (Gray labels)."""
    lob = jnp.asarray(bin_labels(n_bits, d))
    return label_kraus(theta, m, n_anc, n_layers, n_bits, d, cross)[lob][:, lob]


def herm_basis(D):
    """C (D^2 x D^2): the row vec(rho) times C gives the real coordinates of a Hermitian rho: rho_ii, then
    Re rho_ij and Im rho_ij for i < j."""
    C, k = np.zeros((D * D, D * D), complex), D
    C[np.arange(D) * (D + 1), np.arange(D)] = 1.0
    for i in range(D):
        for j in range(i + 1, D):
            C[i * D + j, k], C[j * D + i, k] = 0.5, 0.5
            C[i * D + j, k + 1], C[j * D + i, k + 1] = -0.5j, 0.5j
            k += 2
    return C


def transfer_from_kraus(K):
    """Real transfer matrices C^-1 (sum_c (K (x) conj K)^T) C of a Kraus table K[x, x', c, i, j]."""
    nb, D = K.shape[0], K.shape[-1]
    M = jnp.einsum("xycij,xyckl->xyjlik", K, K.conj()).reshape(nb, nb, D * D, D * D)
    C = herm_basis(D)
    return jnp.real(jnp.einsum("ab,xybc,cd->xyad", jnp.asarray(np.linalg.inv(C)), M, jnp.asarray(C)))


def dephase(M, D, gamma):
    """Transfer matrices (..., D^2, D^2) followed by the dephasing rho -> (1 - gamma) rho + gamma diag(rho) of the memory
    (sequence 5): the off-diagonal real coordinates scaled by 1 - gamma. gamma = 0 returns M itself."""
    if not gamma:
        return M
    return M * jnp.asarray(np.concatenate([np.ones(D), np.full(D * D - D, 1.0 - gamma)]))


def transfer(theta, m, n_anc, n_layers=3, n_bits=5, gamma=0.0, d=1, cross=True):
    return dephase(transfer_from_kraus(kraus(theta, m, n_anc, n_layers, n_bits, d, cross)), 2**m, gamma)


def f_vec(m):
    """vec(I) in the real coordinates (the trace)."""
    D = 2**m
    return jnp.asarray(np.real(np.linalg.inv(herm_basis(D)) @ np.eye(D).ravel()))


def rho_of(a, m):
    """Density matrix (D x D) of an internal vector in real coordinates."""
    D = 2**m
    return (np.asarray(a) @ np.linalg.inv(herm_basis(D))).reshape(D, D)


def loglik(theta, X, m, n_anc, n_layers=3, n_bits=5, gamma=0.0, d=1, cross=True):
    return iolin.loglik(transfer(theta, m, n_anc, n_layers, n_bits, gamma, d, cross), f_vec(m), X)


def io(theta, m, n_anc, n_layers=3, n_bits=5, gamma=0.0, d=1, cross=True):
    return iolin.IOLin(np.asarray(transfer(jnp.asarray(theta), m, n_anc, n_layers, n_bits, gamma, d, cross)),
                       np.asarray(f_vec(m)))


def init_theta(seed, m, n_anc, n_layers=3, n_bits=5, noise=0.2, d=1, cross=True):
    gates = gate_list(m, n_anc, n_layers, n_bits, d, cross)
    th = noise * np.random.default_rng(seed).normal(size=len(gates))
    for k, (_, _, _, tag) in enumerate(gates):
        th[k] = np.pi if tag == "copy" else th[k] + (np.pi / 2 if tag in ("anc_plus", "mem_plus") else 0.0)
    return th


def embed(theta_b, seed, m, n_anc, n_layers=3, n_bits=5, noise=0.1, d=1, cross=True):
    """Stage 2 start: the Architecture B angles (m = 0 vector) in place, memory angles N(0, noise^2), the layer-0
    memory R_y + pi/2."""
    mask = mem_mask(m, n_anc, n_layers, n_bits, d, cross)
    gates = gate_list(m, n_anc, n_layers, n_bits, d, cross)
    th = np.zeros(len(gates))
    th[~mask] = theta_b
    th[mask] = noise * np.random.default_rng(seed).normal(size=int(mask.sum()))
    th[[k for k, g in enumerate(gates) if g[3] == "mem_plus"]] += np.pi / 2
    return th


def qsm_transfer(Ksym, n_bits=5):
    """A sequence 3 QSM (Kraus operators Ksym[x'] in symbol = bin order) as transfer matrices independent of x."""
    Ksym = jnp.asarray(Ksym)
    nb = 2**n_bits
    return transfer_from_kraus(jnp.broadcast_to(Ksym[None, :, None], (nb,) + Ksym.shape[:1] + (1,) + Ksym.shape[1:]))
