"""Single-register 5-qubit model (spec, "Out of scope and effort", the optional extra).

    P_theta(b' | b) = |<b'| U(theta) |b>|^2

The input state |b> is the circuit's initial state; there is no separate output register, so
the kernel is the entrywise squared modulus of one 32 x 32 unitary. Such a kernel is
unistochastic, hence doubly stochastic (rows AND columns sum to 1), and its stationary law is
uniform (1/32 per bin) whenever the chain is irreducible, whatever theta. This is the model
the input register of Architecture B is there to avoid.

Ansatz: real gates only, the same blocks as the output side of `archb.circuit` without the
input -> output controls. Per layer (l = 0, 1, 2), then a final R_y layer:
  (a) local      R_y(theta) on qubit r, r = 0..4                                 5 angles
  (c) chain      for r = 0..3: control r -> target r+1, then control r+1 -> target r   8 angles
  final          R_y(theta) on qubit r                                            5 angles
Angle-vector layout (44 = 3 * 13 + 5), offset of layer l is 13 l:
  +0 .. +4   local[l, r]           index 13l + r
  +5 .. +12  chain[l, r, d]        index 13l + 5 + 2r + d  (d = 0: r -> r+1; d = 1: r+1 -> r)
  39 .. 43   final[r]

Initialization: all angles N(0, noise^2). theta = 0 gives U = I and P = I, the natural
near-identity start (no copy gates needed); like the copy start of the conditional QCBM it is
an exact critical point, so the noise is required.
"""

import jax
import jax.numpy as jnp
import numpy as np

from archb.circuit import _cry, _ry
from archb.ou import N_BINS, N_BITS, encoding_perm

jax.config.update("jax_enable_x64", True)

N_QUBITS = N_BITS  # 5
N_LAYERS = 3
N_LOCAL = N_QUBITS  # 5
N_CHAIN = 2 * (N_QUBITS - 1)  # 8
LAYER_SIZE = N_LOCAL + N_CHAIN  # 13
N_PARAMS = N_LAYERS * LAYER_SIZE + N_QUBITS  # 44


def unpack(theta):
    """Split the angle vector into local (L,5), chain (L,4,2)[l,r,d] and final (5,); L from the length."""
    n_layers = (theta.shape[0] - N_QUBITS) // LAYER_SIZE
    layers = theta[: n_layers * LAYER_SIZE].reshape(n_layers, LAYER_SIZE)
    return {"local": layers[:, :N_LOCAL], "chain": layers[:, N_LOCAL:].reshape(n_layers, N_QUBITS - 1, 2),
            "final": theta[n_layers * LAYER_SIZE:]}


def unitary_columns(theta):
    """psi[b] = U(theta)|b> for all 32 basis inputs, shape (32, 2,2,2,2,2) (real)."""
    p = unpack(jnp.asarray(theta, dtype=jnp.float64))
    psi = jnp.eye(N_BINS, dtype=jnp.float64).reshape((N_BINS,) + (2,) * N_QUBITS)
    for l in range(p["local"].shape[0]):
        for r in range(N_QUBITS):
            psi = _ry(psi, r, p["local"][l, r])
        for r in range(N_QUBITS - 1):
            psi = _cry(psi, r, r + 1, p["chain"][l, r, 0])
            psi = _cry(psi, r + 1, r, p["chain"][l, r, 1])
    for r in range(N_QUBITS):
        psi = _ry(psi, r, p["final"][r])
    return psi


@jax.jit
def label_kernel(theta):
    """Q[b, b'] = |<b'|U(theta)|b>|^2 in LABEL space, shape (32, 32); doubly stochastic."""
    return unitary_columns(theta).reshape(N_BINS, N_BINS) ** 2


def kernel(theta, encoding: str):
    """P[i, j] = Q[label_of_bin[i], label_of_bin[j]] in BIN space (a permutation of Q, so still
    doubly stochastic). `encoding` is 'binary' or 'gray'."""
    lob, _ = encoding_perm(encoding)
    return label_kernel(theta)[np.ix_(lob, lob)]


def init_theta(key_or_seed=0, noise: float = 0.2, n_layers: int = N_LAYERS):
    """All angles N(0, noise^2) (theta = 0 is the identity kernel)."""
    key = jax.random.PRNGKey(key_or_seed) if isinstance(key_or_seed, (int, np.integer)) else key_or_seed
    return noise * jax.random.normal(key, (n_layers * LAYER_SIZE + N_QUBITS,), dtype=jnp.float64)
