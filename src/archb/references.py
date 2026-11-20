"""The four untrained reference kernels of the spec, all 32x32 row-stochastic in bin space.

- exact:   P*, the closed-form discretized OU kernel;
- ar1:     Gaussian AR(1) fitted by weighted least squares on bin centres, discretized like P*;
- tabular: (N_ij + 0.1) normalized per row;
- iid:     every row equals the empirical marginal pi_hat.
"""

import numpy as np

from .ou import CENTRES, counts_to_kernel, exact_kernel, gaussian_bin_kernel

TABULAR_ALPHA = 0.1
REFERENCE_NAMES = ("exact", "ar1", "tabular", "iid")


def fit_ar1(N: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of z_{t+1} = c z_t + m + s xi on bin centres, pair (i, j) weighted by N_ij.

    c, m: weighted regression of z_j on z_i; s: square root of the weighted mean squared residual
    (divided by the total count, no degrees-of-freedom correction). Using centres instead of the
    raw values adds the within-bin quantization noise (~width^2/12) to both sides, so s comes out
    slightly above the true 0.301 and c slightly below 0.905."""
    w = np.asarray(N, dtype=float)
    W = w.sum()
    zi, zj = CENTRES[:, None], CENTRES[None, :]
    mx = (w * zi).sum() / W
    my = (w * zj).sum() / W
    c = (w * (zi - mx) * (zj - my)).sum() / (w * (zi - mx) ** 2).sum()
    m = my - c * mx
    s = np.sqrt((w * (zj - c * zi - m) ** 2).sum() / W)
    return float(c), float(m), float(s)


def reference_kernels(data: dict) -> dict[str, np.ndarray]:
    """{'exact', 'ar1', 'tabular', 'iid'} -> 32x32 kernel, from the dataset dict of ou.load_dataset."""
    N, pi_hat = data["N"], np.asarray(data["pi_hat"], dtype=float)
    return {
        "exact": exact_kernel(),
        "ar1": gaussian_bin_kernel(*fit_ar1(N)),
        "tabular": counts_to_kernel(N, TABULAR_ALPHA),
        "iid": np.tile(pi_hat / pi_hat.sum(), (len(pi_hat), 1)),
    }
