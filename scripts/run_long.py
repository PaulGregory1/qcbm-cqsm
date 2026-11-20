"""Resumable long fit (sequence 2 follow-ups, depth on var2d): the same objective and optimizer as
scripts/run_seq2.py (fit to P* or maximum likelihood; Adam then L-BFGS-B), cut into segments so a fit can span
several jobs on a machine with a wall-time limit.

State in <run dir>/ckpt/: theta.npy (current iterate), loss.npy (curve so far), state.json (phase, L-BFGS
iterations done, converged flag). Each call runs Adam once (if not done), then L-BFGS segments of SEGMENT
iterations from the saved iterate (the L-BFGS memory restarts at each segment boundary), saving after each segment,
and returns when the time budget is spent (resume by calling again), when TOTAL L-BFGS iterations are done, or when a
segment ends by convergence rather than by its iteration limit. At SNAPSHOT iterations (the cap used for the 3-layer
var2d runs) the metrics are also written to snapshot_<n>.json, so deeper circuits can be compared at equal budget.

Usage: python scripts/run_long.py <target> <ceiling|mle> <circuit> [--id 0] [--total 10000] [--budget-min 105]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_seq2 as R  # noqa: E402
from archb import ceiling, targets as TG, variants as V  # noqa: E402

ADAM_STEPS, SEGMENT, SNAPSHOT = 500, 1000, 6000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("mode", choices=["ceiling", "mle"])
    ap.add_argument("circuit")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--budget-min", type=float, default=112,
                    help="stop before starting a segment that would end after this many minutes (estimated from the "
                         "previous segment's duration)")
    a = ap.parse_args(argv)
    t0 = time.time()
    T = R.ALL_TARGETS[a.target]
    variant, n_layers = R.CIRCUITS[a.circuit]
    d = R.OUT / T.name / a.mode / a.circuit / str(a.id)
    if (d / "metrics.json").exists():
        print("done already:", d)
        return 0
    ck = d / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    M = R.Model(variant, n_layers, T)
    loss_fn = (ceiling.kl_objective(M.kernel, T.exact_kernel()) if a.mode == "ceiling"
               else ceiling.data_kl_objective(M.kernel, TG.load_dataset(T)))
    st = json.loads((ck / "state.json").read_text()) if (ck / "state.json").exists() else \
        {"adam_done": False, "lbfgs_iters": 0, "converged": False, "wall_time": 0.0, "segments": []}
    base = {"target": T.name, "mode": a.mode, "circuit": a.circuit, "id": a.id, "n_layers": n_layers,
            "n_params": V.n_params(variant, n_layers, T.n_bits),
            "n_qubits": 2 * T.n_bits + {"anc1": 1, "anc2": 2, "anc2x": 2}.get(variant, 0),
            "fit": {"adam_steps": ADAM_STEPS, "lbfgs_iters": a.total, "segment": SEGMENT}}

    def checkpoint(theta, curve):
        np.save(ck / "theta.npy", np.asarray(theta))
        np.save(ck / "loss.npy", np.asarray(curve))
        st["wall_time"] += time.time() - t_seg
        (ck / "state.json").write_text(json.dumps(st, indent=2) + "\n")

    t_seg = time.time()
    if not st["adam_done"]:
        res = ceiling.adam(M.kernel, M.init_theta(a.id, R.NOISE), ADAM_STEPS, loss_fn=loss_fn)
        st["adam_done"], st["adam_objective"] = True, res["final_loss"]
        checkpoint(res["theta"], res["loss"])
        print(f"adam done: {res['final_loss']:.6f}", flush=True)
    theta = np.load(ck / "theta.npy")
    curve = list(np.load(ck / "loss.npy"))
    vg = jax.jit(jax.value_and_grad(loss_fn))
    last = [None]

    def fun(x):
        v, g = vg(jnp.asarray(x))
        last[0] = float(v)
        return last[0], np.asarray(g, dtype=np.float64)

    last_seg = 0.0
    while not st["converged"] and st["lbfgs_iters"] < a.total:
        if time.time() - t0 + 1.15 * last_seg > a.budget_min * 60:
            print(f"time budget spent at {st['lbfgs_iters']} L-BFGS iterations; resume to continue", flush=True)
            return 0
        t_seg = time.time()
        n = min(SEGMENT, a.total - st["lbfgs_iters"])
        seg = []
        out = minimize(fun, theta, jac=True, method="L-BFGS-B", callback=lambda x: seg.append(last[0]),
                       options={"maxiter": n, "maxfun": 2 * n, "ftol": 1e-13, "gtol": 1e-9})
        theta = out.x
        curve += seg
        st["lbfgs_iters"] += int(out.nit)
        st["converged"] = "ITERATIONS REACHED LIMIT" not in str(out.message) and "TOTAL NO. OF F" not in str(out.message)
        st["segments"].append({"iters": int(out.nit), "objective": float(out.fun), "message": str(out.message)})
        checkpoint(theta, curve)
        last_seg = time.time() - t_seg
        print(f"L-BFGS {st['lbfgs_iters']}: {out.fun:.6f} ({out.message})", flush=True)
        if st["lbfgs_iters"] >= SNAPSHOT and not (d / f"snapshot_{SNAPSHOT}.json").exists() and \
                st["lbfgs_iters"] - int(out.nit) < SNAPSHOT:
            K = np.asarray(M.kernel(theta))
            snap = {**base, "lbfgs_iters": st["lbfgs_iters"], "objective": float(out.fun),
                    "metrics": TG.metrics(K, T)}
            (d / f"snapshot_{SNAPSHOT}.json").write_text(json.dumps(snap, indent=2) + "\n")
    K = np.asarray(M.kernel(theta))
    info = {**base, "objective": float(loss_fn(jnp.asarray(theta))), "adam_objective": st["adam_objective"],
            "steps": ADAM_STEPS + st["lbfgs_iters"], "lbfgs_iters": st["lbfgs_iters"],
            "lbfgs_message": st["segments"][-1]["message"] if st["segments"] else "", "wall_time": st["wall_time"],
            "converged": st["converged"]}
    R.save(d, T, theta, K, np.asarray(curve), info)
    m = info["metrics"]
    print(f"final: WTV {m['weighted_tv']:.4f} corr {m.get('corr_err', 0):+.3f} after {st['lbfgs_iters']} iterations",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
