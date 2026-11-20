"""Sequence 1 check: does the Sinkhorn divergence rank kernels like the weighted TV against P*?

For each circuit and encoding, evaluate the converged Sinkhorn divergence S_eps(pi_hat x P, v) (the spec loss,
no training) at three kernels already on disk: the best step 0 fit to P*, the best maximum-likelihood fit to the
data, and the end of the Sinkhorn warm start; plus the exact kernel P* and the two references.
Writes runs/seq1/sinkhorn_rank.json.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from archb import sinkhorn as sk
from archb.metrics import all_metrics
from archb.ou import exact_kernel, load_dataset
from archb.references import reference_kernels

ROOT = Path(__file__).resolve().parents[1]
SEQ1 = ROOT / "runs" / "seq1"
EPS = 0.01


def s_eps(P, pi_hat, v, ot_vv):
    w = pi_hat[:, None] * jnp.asarray(P)
    f, g, _ = sk.converge_cross(w, v, EPS, tol=1e-11)
    p, _ = sk.converge_self(w, EPS, tol=1e-11)
    return float(sk.ot_value(w, v, f, g) - 0.5 * sk.ot_self_value(w, p) - 0.5 * ot_vv)


def main():
    data = load_dataset()
    N = np.asarray(data["N"], float)
    v = jnp.asarray(N / N.sum())
    pi_hat = jnp.asarray(data["pi_hat"], dtype=jnp.float64)
    pv, _ = sk.converge_self(v, EPS, tol=1e-11)
    ot_vv = sk.ot_self_value(v, pv)
    P_star = exact_kernel()
    S = json.loads((SEQ1 / "summary.json").read_text())
    out = {"references": {}, "circuits": {}}
    for name, K in reference_kernels(data).items():
        if name != "iid":
            out["references"][name] = {"sinkhorn": s_eps(K, pi_hat, v, ot_vv),
                                        "weighted_tv": all_metrics(K, P_star)["weighted_tv"]}
    for c, CE in S["circuits"].items():
        for enc in ("binary", "gray"):
            E = CE.get(enc, {})
            row = {}
            for mode in ("ceiling", "mle", "warm"):
                if mode not in E:
                    continue
                best = E[mode]["best"]["id"]
                r = next(x for x in E[mode]["runs"] if x["id"] == best)
                K = np.load(ROOT / r["dir"] / "kernel.npy")
                row[mode] = {"sinkhorn": s_eps(K, pi_hat, v, ot_vv), "weighted_tv": r["weighted_tv"]}
            out["circuits"].setdefault(c, {})[enc] = row
            print(c, enc, {m: (round(x["sinkhorn"], 5), round(x["weighted_tv"], 4)) for m, x in row.items()}, flush=True)
    print(out["references"])
    (SEQ1 / "sinkhorn_rank.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
