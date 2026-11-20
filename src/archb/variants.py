"""Sequence 1 circuit variants: a general full-register simulator.

Register: 5 input qubits (q = 0..4, MSB first), 5 output qubits (r = 0..4, MSB first), then
n_anc ancillas. For every input label b the circuit starts in |b>|00000>|0..0>; all 32 inputs
are simulated at once as a real statevector of shape (32, 2, ..., 2). The kernel is the spec's
general formula with every non-output qubit traced:

    Q[b, b'] = sum_c |<c, b'| U(theta) |b, 0>|^2      (c over input and ancilla qubits)

Unlike `archb.circuit`, nothing assumes the input qubits stay in a basis state, so the same code
covers gates that target input qubits (variant "inout") and ancillas.

Gates (all real): ("ry", t), ("cry", c, t), ("ccry", c1, c2, t), with register-wide qubit
indices: input q -> q, output r -> 5 + r, ancilla a -> 10 + a. Each gate owns one angle.

Variants (layers l = 0..L-1, then a final R_y layer on the outputs):
  B0      spec layer: local R_y on outputs (5), input q -> output r controlled R_y (25, q outer),
          output chain r <-> r+1 (8). L = 3, 4, 5 -> 119, 157, 195 angles (identical to
          archb.circuit for any L; tested).
  inout   B0 layer + output r -> input q controlled R_y (25, r outer), after the chain, so the
          next layer's input-controlled gates see the rotated inputs. 63/layer, 194 for L = 3.
  allpairs  B0 with the chain replaced by all ordered output pairs (20). 50/layer, 155.
  quad    B0 layer + doubly-controlled R_y from input pairs (q < q') to each output r (50),
          after the input -> output block. 88/layer, 269.
  anc2x   (sequence 2 follow-up, 2D registers) anc2 + controlled R_y between output bit r of x1 and output bit r
          of x2 (both directions), after the chain: 2 x (n_bits / 2) more angles per layer.
  anc1/anc2 B0 layer + per ancilla a: R_y on a, input q -> a (5), a -> output r (5),
          output r -> a (5), after the chain. 54/layer (167) and 70/layer (215).

Initialization (spec, plus the known pitfalls): every angle N(0, noise^2), except the copy
gates (layer 0, input q -> output q) = pi, and the layer-0 R_y on each ancilla = pi/2 (+ noise),
so ancillas start in |+> and their controlled gates have non-zero gradients.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from archb.ou import N_BITS, encoding_perm

jax.config.update("jax_enable_x64", True)

N_IN = N_OUT = N_BITS  # 5 (sequence 1); every function takes n_bits for sequence 2 (8 bits for var2d)
VARIANTS = ("B0", "inout", "allpairs", "quad", "anc1", "anc2", "anc2x")


def gate_list(variant: str, n_layers: int = 3, n_bits: int = N_BITS):
    """[(gate, qubits..., tag)] in application order, one angle per gate, and n_anc.
    tag marks gates with a special init: 'copy' (pi) or 'anc_plus' (pi/2).
    Register: input q -> q, output r -> n_bits + r, ancilla a -> 2 n_bits + a."""
    n_anc = {"anc1": 1, "anc2": 2, "anc2x": 2}.get(variant, 0)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    N_IN = N_OUT = n_bits
    IN = list(range(n_bits))
    OUT = [n_bits + r for r in range(n_bits)]

    def _anc(a):
        return 2 * n_bits + a

    gates = []
    for l in range(n_layers):
        gates += [("ry", OUT[r], None) for r in range(N_OUT)]
        for q in range(N_IN):
            for r in range(N_OUT):
                gates.append(("cry", IN[q], OUT[r], "copy" if (l == 0 and q == r) else None))
        if variant == "quad":
            for q in range(N_IN):
                for q2 in range(q + 1, N_IN):
                    gates += [("ccry", IN[q], IN[q2], OUT[r], None) for r in range(N_OUT)]
        if variant == "allpairs":
            for r in range(N_OUT):
                for r2 in range(r + 1, N_OUT):
                    gates += [("cry", OUT[r], OUT[r2], None), ("cry", OUT[r2], OUT[r], None)]
        else:
            for r in range(N_OUT - 1):
                gates += [("cry", OUT[r], OUT[r + 1], None), ("cry", OUT[r + 1], OUT[r], None)]
        if variant == "anc2x":  # 2D registers: output bit r of x1 <-> output bit r of x2, both directions
            h = n_bits // 2
            for r in range(h):
                gates += [("cry", OUT[r], OUT[r + h], None), ("cry", OUT[r + h], OUT[r], None)]
        if variant == "inout":
            for r in range(N_OUT):
                gates += [("cry", OUT[r], IN[q], None) for q in range(N_IN)]
        for a in range(n_anc):
            A = _anc(a)
            gates.append(("ry", A, "anc_plus" if l == 0 else None))
            gates += [("cry", IN[q], A, None) for q in range(N_IN)]
            gates += [("cry", A, OUT[r], None) for r in range(N_OUT)]
            gates += [("cry", OUT[r], A, None) for r in range(N_OUT)]
    gates += [("ry", OUT[r], None) for r in range(N_OUT)]
    return gates, n_anc


def n_params(variant: str, n_layers: int = 3, n_bits: int = N_BITS) -> int:
    return len(gate_list(variant, n_layers, n_bits)[0])


def _apply(x, axis, ctrl_axes, ang):
    """R_y on `axis` of x (shape (B, 2, ..., 2)), restricted to the slice where every axis in
    ctrl_axes is 1. ang has shape (B,) (per-input angle). Slices and stacks only: no transpose,
    and the controlled branch never computes the untouched half."""
    if ctrl_axes:
        c0 = ctrl_axes[0]
        rest = [a - (a > c0) for a in ctrl_axes[1:]]
        x0 = jnp.take(x, 0, axis=c0)
        x1 = _apply(jnp.take(x, 1, axis=c0), axis - (axis > c0), rest, ang)
        return jnp.stack([x0, x1], axis=c0)
    shape = (-1,) + (1,) * (x.ndim - 2)
    c, s = jnp.cos(ang / 2).reshape(shape), jnp.sin(ang / 2).reshape(shape)
    a0, a1 = jnp.take(x, 0, axis=axis), jnp.take(x, 1, axis=axis)
    return jnp.stack([c * a0 - s * a1, s * a0 + c * a1], axis=axis)


def _input_bits(n_bits):
    i = np.arange(2**n_bits)
    return ((i[:, None] >> (n_bits - 1 - np.arange(n_bits))[None, :]) & 1).astype(np.float64)  # MSB first


def _rot_flat(psi, n_q, t, ang):
    """R_y(ang) on qubit t (0 = most significant) of a flat (B, 2^n_q) state; ang shape (B,).
    A contiguous reshape exposes the target bit (no transpose)."""
    B = psi.shape[0]
    x = psi.reshape(B, 2**t, 2, 2 ** (n_q - t - 1))
    c, s = jnp.cos(ang / 2)[:, None, None], jnp.sin(ang / 2)[:, None, None]
    a0, a1 = x[:, :, 0, :], x[:, :, 1, :]
    return jnp.stack([c * a0 - s * a1, s * a0 + c * a1], axis=2).reshape(B, 2**n_q)


def _crot_flat(psi, n_q, c_q, t, ang):
    """R_y(ang) on qubit t where qubit c_q is 1: reshape to expose both bits, rotate the c_q = 1 half only."""
    B = psi.shape[0]
    lo, hi = min(c_q, t), max(c_q, t)
    x = psi.reshape(B, 2**lo, 2, 2 ** (hi - lo - 1), 2, 2 ** (n_q - hi - 1))
    ca, ta = (2, 4) if c_q < t else (4, 2)  # axes of control and target bits in x
    x0, x1 = jnp.take(x, 0, axis=ca), jnp.take(x, 1, axis=ca)  # target axis index shifts if it was after ca
    tt = ta if ta < ca else ta - 1
    c, s = jnp.cos(ang / 2).reshape(-1, 1, 1, 1), jnp.sin(ang / 2).reshape(-1, 1, 1, 1)
    a0, a1 = jnp.take(x1, 0, axis=tt), jnp.take(x1, 1, axis=tt)
    x1 = jnp.stack([c * a0 - s * a1, s * a0 + c * a1], axis=tt)
    return jnp.stack([x0, x1], axis=ca).reshape(B, 2**n_q)


@partial(jax.jit, static_argnums=(1, 2, 3))
def label_kernel(theta, variant: str = "B0", n_layers: int = 3, n_bits: int = N_BITS):
    """Q[b, b'] in LABEL space, shape (2^n_bits, 2^n_bits), rows sum to 1.

    Input qubits that no gate targets stay in |b> for the whole circuit: they are not simulated,
    and a gate they control fires with angle theta * prod(b_q) per input (exact, R_y(0) = I).
    Gates whose controls are all such classical inputs are folded: consecutive R_y on the same qubit
    add their (per-input) angles, so they are applied as one rotation just before the next gate that
    touches that qubit (exact: R_y rotations about the same axis commute and add). Every other gate is
    applied on a flat (inputs, 2^n_simulated) state."""
    gates, n_anc = gate_list(variant, n_layers, n_bits)
    nb = 2**n_bits
    INPUT_BITS = _input_bits(n_bits)
    targeted = {g[-2] for g in gates}
    quantum = [q for q in range(2 * n_bits + n_anc) if q >= n_bits or q in targeted]
    pos = {q: i for i, q in enumerate(quantum)}  # register qubit -> bit position in the flat state
    n_q = len(quantum)
    psi0 = np.zeros((nb, 2**n_q))
    for b in range(nb):
        idx = 0
        for q in quantum:
            idx = 2 * idx + (int(INPUT_BITS[b, q]) if q < n_bits else 0)
        psi0[b, idx] = 1.0
    psi = jnp.asarray(psi0)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    pending = {}  # simulated qubit -> accumulated per-input angle (B,)

    def flush(q):
        nonlocal psi
        if q in pending:
            psi = _rot_flat(psi, n_q, pos[q], pending.pop(q))

    for k, g in enumerate(gates):
        qubits = g[1:-1]
        ctrl, tgt = qubits[:-1], qubits[-1]
        classical = [q for q in ctrl if q not in pos]
        qctrl = [q for q in ctrl if q in pos]
        mask = jnp.asarray(INPUT_BITS[:, classical].prod(axis=1)) if classical else jnp.ones(nb)
        if not qctrl:
            pending[tgt] = pending.get(tgt, 0.0) + theta[k] * mask
            continue
        if len(qctrl) != 1:
            raise NotImplementedError("at most one simulated control per gate")
        flush(qctrl[0])
        flush(tgt)
        psi = _crot_flat(psi, n_q, pos[qctrl[0]], pos[tgt], theta[k] * mask)
    for q in list(pending):
        flush(q)
    prob = (psi**2).reshape((nb,) + (2,) * n_q)
    keep = [1 + pos[n_bits + r] for r in range(n_bits)]
    traced = tuple(a for a in range(1, n_q + 1) if a not in keep)
    return prob.sum(axis=traced).reshape(nb, nb)


def kernel(theta, encoding, variant: str = "B0", n_layers: int = 3, n_bits: int = N_BITS):
    """P[i, j] in BIN space (decoded order). `encoding`: 'binary' / 'gray' (5-bit OU grid) or a label_of_bin
    array (sequence 2 targets)."""
    lob = encoding_perm(encoding)[0] if isinstance(encoding, str) else np.asarray(encoding)
    return label_kernel(theta, variant, n_layers, n_bits)[np.ix_(lob, lob)]


def init_theta(seed=0, noise: float = 0.2, variant: str = "B0", n_layers: int = 3, n_bits: int = N_BITS):
    gates, _ = gate_list(variant, n_layers, n_bits)
    key = jax.random.PRNGKey(int(seed))
    theta = noise * jax.random.normal(key, (len(gates),), dtype=jnp.float64)
    tags = np.array([g[-1] for g in gates], dtype=object)
    theta = theta.at[np.flatnonzero(tags == "copy")].set(jnp.pi)
    plus = np.flatnonzero(tags == "anc_plus")
    return theta.at[plus].add(jnp.pi / 2)


class Model:
    """Adapter for archb.train.MODELS: kernel(theta, encoding) and init_theta(seed, noise, n_layers)."""

    def __init__(self, variant: str, n_layers: int = 3):
        self.variant, self.n_layers = variant, n_layers
        self.n_params = n_params(variant, n_layers)

    def kernel(self, theta, encoding):
        return kernel(theta, encoding, self.variant, self.n_layers)

    def init_theta(self, seed=0, noise=0.2, n_layers=None):
        return init_theta(seed, noise, self.variant, self.n_layers)


# name -> (variant, n_layers), the sequence 1 line-up
LINEUP = {
    "B0": ("B0", 3), "B0-L4": ("B0", 4), "B0-L5": ("B0", 5),
    "inout": ("inout", 3), "allpairs": ("allpairs", 3), "quad": ("quad", 3),
    "anc1": ("anc1", 3), "anc2": ("anc2", 3),
}
