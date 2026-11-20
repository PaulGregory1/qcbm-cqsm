"""Sequence 4 diagnostic: continue the L-BFGS of a finished C-QSM fit far beyond the campaign cap, to tell training
from circuit limits. Starts from runs/seq4/<dataset>/<model>/<init>/params.npz (default: the init with the best
training likelihood), runs L-BFGS-B in segments of SEG iterations (checkpoint after each, resumable), stops when a
segment improves the mean log-likelihood by less than 1e-7 or at --iters. Result: runs/seq4/<dataset>/<model>-long/<init>/
with the same metrics as the campaign (scripts/run_seq4.evaluate).

Usage: python scripts/seq4_long.py --dataset msar --model cqsm-m1-anc2 [--init k] [--iters 20000]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seq4 as RS  # noqa: E402

from archb import cqsm  # noqa: E402

SEG = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msar")
    ap.add_argument("--model", default="cqsm-m1-anc2")
    ap.add_argument("--init", type=int, default=None)
    ap.add_argument("--iters", type=int, default=20000)
    a = ap.parse_args()
    src = RS.OUT / a.dataset / a.model
    if a.init is None:
        a.init = max(RS.STAGE2, key=lambda i: json.loads((src / str(i) / "result.json").read_text())["train_ll"])
    out = RS.OUT / a.dataset / f"{a.model}-long" / str(a.init)
    out.mkdir(parents=True, exist_ok=True)
    m, anc = int(a.model.split("-")[1][1:]), (2 if a.model.endswith("anc2") else 0)
    X = RS.load(a.dataset)["X"]
    Xj = jnp.asarray(X)
    vg = jax.jit(jax.value_and_grad(lambda t: -cqsm.loglik(t, Xj, m, anc, RS.L)))
    ck = out / "checkpoint.npz"
    if ck.exists():
        z = np.load(ck)
        th, done, curve = z["theta"], int(z["done"]), list(z["curve"])
    else:
        th, done, curve = np.load(src / str(a.init) / "params.npz")["theta"], 0, []
        curve.append(float(vg(jnp.asarray(th))[0]))
    t0 = time.time()
    while done < a.iters:
        last = [None]

        def fun(x):
            v, g = vg(jnp.asarray(x))
            last[0] = float(v)
            return last[0], np.asarray(g, np.float64)

        r = minimize(fun, th, jac=True, method="L-BFGS-B", callback=lambda x: curve.append(last[0]),
                     options={"maxiter": SEG, "maxfun": 2 * SEG, "ftol": 1e-15, "gtol": 1e-10})
        gain = curve[-r.nit - 1] - r.fun if r.nit else 0.0
        th, done = r.x, done + max(int(r.nit), 1)
        np.savez(ck, theta=th, done=done, curve=np.asarray(curve))
        print(f"{done} iterations: train ll {-r.fun:.6f} (segment gain {gain:.2e}) [{time.time() - t0:.0f} s]",
              flush=True)
        if gain < 1e-7:
            break
    lin = cqsm.io(th, m, anc, RS.L)
    res = {"dataset": a.dataset, **RS.DATASETS[a.dataset], "model": f"{a.model}-long", "init": a.init,
           "memory_dim": 2**m, "n_ancillas": anc, "n_layers": RS.L, "n_params": cqsm.n_params(m, anc, RS.L),
           "lbfgs_iters_extra": done, "from_ll": -curve[0]}
    res.update(RS.evaluate(a.dataset, lin, lin.logprobs(RS.load(a.dataset)["test"]), -curve[-1], X))
    res["wall_time"] = time.time() - t0
    np.savez(out / "params.npz", theta=th)
    np.save(out / "loss.npy", np.asarray(curve))
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"held-out KL rate {res['heldout_kl_rate']:.3e} (was "
          f"{json.loads((src / str(a.init) / 'result.json').read_text())['heldout_kl_rate']:.3e})")


if __name__ == "__main__":
    main()
