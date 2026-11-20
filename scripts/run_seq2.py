"""Sequence 2: harder Markov targets (archb.targets), circuits B0 / anc1 / anc2, Gray code.

Modes, per target and circuit (same settings as sequence 1 unless stated):
  ceiling  fit to the exact kernel P* (archb.ceiling.kl_objective: Adam then L-BFGS), inits 0..4, noise 0.2
  mle      maximum likelihood on the target's data (archb.ceiling.data_kl_objective, Adam then L-BFGS), seeds 0..2
  train    Sinkhorn training (archb.train.train, 100 Sinkhorn iterations, lr 0.03, eps 0.01, cap 1500 steps,
           stop rule on a 25-step moving average), seeds 0..2. 1D targets only.
Sequence 2 follow-ups: var2d3 / var2d3i (2 x 3 bits, plain / interleaved bits, full
protocol without Sinkhorn), var2di (2 x 4 bits interleaved, reduced protocol), circuits anc2x, anc2-L4, anc2-L5.
var2d (2 x 4 bits, 16 qubits + ancillas, ~1-2.5 s per gradient here) uses a reduced protocol: 1 ceiling init and
1 likelihood seed per circuit, Adam 500 steps then L-BFGS capped at 6,000 iterations (one fit stays under
the 2-hour job limit of the machine used), no Sinkhorn.

-> runs/seq2/<target>/<mode>/<circuit>/<k>/{theta,kernel,loss}.npy + metrics.json (archb.targets.metrics).
Resumable: existing runs are skipped.

Usage: python scripts/run_seq2.py <target> <mode> [--circuits B0 anc1 anc2] [--ids 0 1 2]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from archb import ceiling, sinkhorn as sk, targets as TG, train as TR, variants as V  # noqa: E402

OUT = ROOT / "runs" / "seq2"
CIRCUITS = {"B0": ("B0", 3), "anc1": ("anc1", 3), "anc2": ("anc2", 3),
            # follow-ups: depth on merton, cross-coordinate entanglers on var2d
            "anc2-L4": ("anc2", 4), "anc2-L5": ("anc2", 5), "anc2x": ("anc2x", 3),
            "anc2x-L4": ("anc2x", 4), "anc2x-L5": ("anc2x", 5)}
DEFAULT_CIRCUITS = ("B0", "anc1", "anc2")
ALL_TARGETS = {**TG.TARGETS, **TG.FOLLOW_UPS}
NOISE = 0.2
SINKHORN = {"steps": 1500, "lr": 0.03, "eps": 0.01, "sinkhorn_iters": 100, "stop_smooth": 25}
FULL = {"ceiling": (0, 1, 2, 3, 4), "mle": (0, 1, 2), "train": (0, 1, 2)}
REDUCED = {"ceiling": (0,), "mle": (0,)}  # var2d
FIT_FULL = {"adam_steps": ceiling.ADAM_STEPS, "lbfgs_iters": ceiling.LBFGS_ITERS}
FIT_REDUCED = {"adam_steps": 500, "lbfgs_iters": 6000}


class Model:
    """archb.train adapter: the target fixes the bit labels, `encoding` is ignored."""

    def __init__(self, variant, n_layers, T):
        self.variant, self.n_layers, self.T, self.lob = variant, n_layers, T, T.label_of_bin()

    def kernel(self, theta, encoding=None):
        return V.kernel(theta, self.lob, self.variant, self.n_layers, self.T.n_bits)

    def init_theta(self, seed=0, noise=NOISE, n_layers=None):
        return V.init_theta(seed, noise, self.variant, self.n_layers, self.T.n_bits)


def save(d: Path, T, theta, K, loss, info):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "theta.npy", np.asarray(theta))
    np.save(d / "kernel.npy", np.asarray(K))
    np.save(d / "loss.npy", np.asarray(loss))
    info["metrics"] = TG.metrics(np.asarray(K), T)
    info["tight_pass"] = TG.tight_pass(info["metrics"], T.d)
    (d / "metrics.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def run(T, mode, circuit, k):
    d = OUT / T.name / mode / circuit / str(k)
    if (d / "metrics.json").exists():
        return json.loads((d / "metrics.json").read_text())
    variant, n_layers = CIRCUITS[circuit]
    M = Model(variant, n_layers, T)
    base = {"target": T.name, "mode": mode, "circuit": circuit, "id": k,
            "n_params": V.n_params(variant, n_layers, T.n_bits), "n_layers": n_layers,
            "n_qubits": 2 * T.n_bits + {"anc1": 1, "anc2": 2, "anc2x": 2}.get(variant, 0)}
    fit_opts = FIT_REDUCED if T.K > 64 else FIT_FULL
    if mode in ("ceiling", "mle"):
        theta0 = M.init_theta(k, NOISE)
        loss_fn = (ceiling.kl_objective(M.kernel, T.exact_kernel()) if mode == "ceiling"
                   else ceiling.data_kl_objective(M.kernel, TG.load_dataset(T)))
        res = ceiling.fit(M.kernel, theta0, loss_fn=loss_fn, **fit_opts)
        info = {**base, "objective": res["final_loss"], "adam_objective": res["adam_loss"], "steps": res["steps"],
                "lbfgs_iters": res["lbfgs_iters"], "lbfgs_message": res["lbfgs_message"], "wall_time": res["wall_time"],
                "fit": fit_opts}
        return save(d, T, res["theta"], M.kernel(res["theta"]), res["loss"], info)
    if mode == "train":
        if T.d != 1:
            raise ValueError("Sinkhorn training is 1D only in sequence 2")
        sk.use_grid(T.z)
        TR.MODELS[f"seq2_{circuit}"] = M
        res = TR.train("gray", k, steps=SINKHORN["steps"], lr=SINKHORN["lr"], eps=SINKHORN["eps"], noise=NOISE,
                       log_every=0, data=TG.load_dataset(T), sinkhorn_iters=SINKHORN["sinkhorn_iters"],
                       model=f"seq2_{circuit}", stop_smooth=SINKHORN["stop_smooth"])
        info = {**base, "objective": res["final_loss"], "init_loss": res["init_loss"], "steps": res["steps"],
                "wall_time": res["wall_time"], "settings": SINKHORN}
        return save(d, T, res["theta"], res["kernel"], res["loss"], info)
    raise ValueError(mode)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=list(ALL_TARGETS))
    ap.add_argument("mode", choices=["ceiling", "mle", "train"])
    ap.add_argument("--circuits", nargs="+", default=list(DEFAULT_CIRCUITS))
    ap.add_argument("--ids", nargs="+", type=int)
    a = ap.parse_args(argv)
    T = ALL_TARGETS[a.target]
    ids = a.ids if a.ids is not None else (REDUCED if T.K > 64 else FULL)[a.mode]
    for c in a.circuits:
        for k in ids:
            t0 = time.time()
            info = run(T, a.mode, c, k)
            m = info["metrics"]
            print(f"{T.name:6s} {a.mode:7s} {c:5s} {k}  WTV {m['weighted_tv']:.4f}  statTV {m['stationary_tv']:.4f}  "
                  f"tailTV {m['tail_tv']:.3f}  sd_rel {m['cond_sd_rel_0']:.3f}  tight {info['tight_pass']}  "
                  f"steps {info['steps']}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
