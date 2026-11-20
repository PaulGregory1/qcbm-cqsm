"""Conditional QCBM simulator (JAX, float64): 5 input qubits (controls only) + 5 output qubits.

Because the input qubits are never rotated, the 10-qubit circuit acting on |b, 00000> is
exactly a 5-qubit circuit U_b(theta) on the output register whose classically-controlled
gates fire iff the corresponding input bit b_q is 1. All 32 inputs are simulated at once as
a real statevector of shape (32, 2, 2, 2, 2, 2): axis 0 = input label b, axes 1..5 = output
qubits r = 0..4 (MSB first), so flattening the last five axes gives the output label b'.

Gate: R_y(t) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]] (real).

Gate order within one layer (layers l = 0, 1, 2), then a final R_y layer:
  (a) local       R_y(theta) on output r, for r = 0..4
  (b) input->out  controlled R_y, control input bit q, target output r, q outer (0..4), r inner (0..4)
  (c) out chain   for r = 0..3: control r -> target r+1, then control r+1 -> target r
  final           R_y(theta) on output r, for r = 0..4

Angle-vector layout (119 = 3 * 38 + 5), offset of layer l is 38 * l:
  +0  .. +4    local[l, r]                      index 38l + r
  +5  .. +29   inout[l, q, r]                   index 38l + 5 + 5q + r
  +30 .. +37   chain[l, r, d], r = 0..3         index 38l + 30 + 2r + d
               d = 0: control r -> target r+1;  d = 1: control r+1 -> target r
  114 .. 118   final[r]

Implementation note: blocks (a) and (b) only touch output qubit r through R_y gates whose
controls are classical, so they compose into one per-input rotation R_y(local[l,r] +
sum_q b_q inout[l,q,r]) on each r (R_y angles add). This is exact; the brute-force test
in tests/test_circuit.py applies the 30 gates one by one on the full 10-qubit register.
"""

import jax
import jax.numpy as jnp
import numpy as np

from archb.ou import N_BINS, N_BITS, encoding_perm, int_to_bits

jax.config.update("jax_enable_x64", True)

N_IN = N_OUT = N_BITS  # 5
N_LAYERS = 3
N_LOCAL = N_OUT  # 5
N_INOUT = N_IN * N_OUT  # 25
N_CHAIN = 2 * (N_OUT - 1)  # 8
LAYER_SIZE = N_LOCAL + N_INOUT + N_CHAIN  # 38
N_PARAMS = N_LAYERS * LAYER_SIZE + N_OUT  # 119

INPUT_BITS = int_to_bits(np.arange(N_BINS)).astype(np.float64)  # (32, 5), row b = bits of label b


# ---------------------------------------------------------------- index helpers
def idx_local(l: int, r: int) -> int:
    return LAYER_SIZE * l + r


def idx_inout(l: int, q: int, r: int) -> int:
    return LAYER_SIZE * l + N_LOCAL + N_OUT * q + r


def idx_chain(l: int, r: int, d: int) -> int:
    """d = 0: control r -> target r+1; d = 1: control r+1 -> target r (r = 0..3)."""
    return LAYER_SIZE * l + N_LOCAL + N_INOUT + 2 * r + d


def idx_final(r: int) -> int:
    return N_LAYERS * LAYER_SIZE + r


def unpack(theta):
    """Split the 119-vector into named blocks: local (3,5), inout (3,5,5)[l,q,r],
    chain (3,4,2)[l,r,d], final (5,). The layer count L is read from the length
    (38 L + 5), so a vector built with init_theta(..., n_layers=4) works unchanged."""
    n_layers = (theta.shape[0] - N_OUT) // LAYER_SIZE
    layers = theta[: n_layers * LAYER_SIZE].reshape(n_layers, LAYER_SIZE)
    return {
        "local": layers[:, :N_LOCAL],
        "inout": layers[:, N_LOCAL : N_LOCAL + N_INOUT].reshape(n_layers, N_IN, N_OUT),
        "chain": layers[:, N_LOCAL + N_INOUT :].reshape(n_layers, N_OUT - 1, 2),
        "final": theta[n_layers * LAYER_SIZE :],
    }


# ------------------------------------------------------------------ simulation
def _ry(psi, r: int, angle):
    """R_y(angle) on output qubit r; angle scalar or per-input (32,)."""
    half = jnp.broadcast_to(jnp.asarray(angle) / 2, (psi.shape[0],)).reshape(-1, 1, 1, 1, 1)
    c, s = jnp.cos(half), jnp.sin(half)  # (32, 1, 1, 1, 1), broadcasts over the other 4 qubits
    a = jnp.moveaxis(psi, r + 1, -1)
    a0, a1 = a[..., 0], a[..., 1]
    out = jnp.stack([c * a0 - s * a1, s * a0 + c * a1], axis=-1)
    return jnp.moveaxis(out, -1, r + 1)


def _cry(psi, ctrl: int, tgt: int, angle):
    """Controlled R_y between output qubits: rotate tgt on the ctrl = 1 subspace only."""
    rotated = _ry(psi, tgt, angle)
    mask = np.zeros((1,) + (2,) * N_OUT, dtype=bool)
    mask[(slice(None),) + tuple(1 if k == ctrl else slice(None) for k in range(N_OUT))] = True
    return jnp.where(mask, rotated, psi)


def output_states(theta):
    """Real output statevectors, shape (32, 2,2,2,2,2): psi[b] = U_b(theta)|00000>."""
    p = unpack(jnp.asarray(theta, dtype=jnp.float64))
    bits = jnp.asarray(INPUT_BITS)
    psi = jnp.zeros((N_BINS,) + (2,) * N_OUT, dtype=jnp.float64).at[(slice(None),) + (0,) * N_OUT].set(1.0)
    for l in range(p["local"].shape[0]):
        ang = p["local"][l][None, :] + bits @ p["inout"][l]  # (32, 5): blocks (a)+(b)
        for r in range(N_OUT):
            psi = _ry(psi, r, ang[:, r])
        for r in range(N_OUT - 1):  # block (c)
            psi = _cry(psi, r, r + 1, p["chain"][l, r, 0])
            psi = _cry(psi, r + 1, r, p["chain"][l, r, 1])
    for r in range(N_OUT):
        psi = _ry(psi, r, p["final"][r])
    return psi


@jax.jit
def label_kernel(theta):
    """Q[b, b'] = |<b'|U_b(theta)|0>|^2 in LABEL space, shape (32, 32), rows sum to 1."""
    return output_states(theta).reshape(N_BINS, N_BINS) ** 2


def kernel(theta, encoding: str):
    """P[i, j] = Q[label_of_bin[i], label_of_bin[j]] in BIN space (decoded order).
    `encoding` is 'binary' or 'gray'; traceable inside jit with encoding static."""
    lob, _ = encoding_perm(encoding)
    return label_kernel(theta)[np.ix_(lob, lob)]


def init_theta(key_or_seed=0, noise: float = 0.2, n_layers: int = N_LAYERS):
    """All angles N(0, noise^2), except layer-0 inout[q, q] = pi (copy x_t to the output).
    n_layers != 3 is a deviation from the spec, used only by the step 5 diagnostic."""
    key = jax.random.PRNGKey(key_or_seed) if isinstance(key_or_seed, (int, np.integer)) else key_or_seed
    theta = noise * jax.random.normal(key, (n_layers * LAYER_SIZE + N_OUT,), dtype=jnp.float64)
    copy_idx = np.array([idx_inout(0, q, q) for q in range(N_IN)])
    return theta.at[copy_idx].set(jnp.pi)
