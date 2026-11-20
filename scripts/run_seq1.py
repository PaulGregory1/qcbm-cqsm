"""Sequence 1: step 0 ceilings and trained runs for the circuit line-up.

  ceiling  fit each circuit directly to P* (archb.ceiling: Adam then L-BFGS), 10 inits x 2 encodings:
           inits 0-4 copy init + noise 0.2, inits 5-9 copy init + noise 0.5.
           -> runs/seq1/ceiling/<circuit>/<encoding>/init<k>/
  train    Sinkhorn training (archb.train), converged-solver settings (100 Sinkhorn iterations,
           lr 0.03, eps 0.01, noise 0.2), cap 1500 steps, stop rule on a 25-step moving average,
           seeds 0-2 x 2 encodings.
           -> runs/seq1/train/<circuit>/<encoding>/seed<k>/
  mle      diagnostic (not in the plan): fit to the DATA by maximum likelihood (archb.ceiling.
           data_kl_objective, Adam then L-BFGS), from the same inits as train (seeds 0-2,
           noise 0.2). Same data as train, only the objective changes: separates the Sinkhorn
           objective from the circuit and the optimizer.
           -> runs/seq1/mle/<circuit>/<encoding>/seed<k>/
  mle_adam same objective as mle but optimized like train: Adam only, constant lr 0.03,
           1500 steps (best iterate kept). Separates the objective from the optimizer.
           -> runs/seq1/mle_adam/<circuit>/<encoding>/seed<k>/
  warm     same as train but starting from the best ceiling fit of that circuit and encoding
           (seed = 0 only; runs where the plan's reading calls for it).
           -> runs/seq1/warm/<circuit>/<encoding>/seed0/

Every run writes theta.npy, kernel.npy, loss.npy and metrics.json (six spec metrics + the
boundary-flux diagnostic of run_all.py). Runs already on disk are skipped (resumable).

Usage: python scripts/run_seq1.py ceiling --circuits B0 [B0-L4 ...]
       python scripts/run_seq1.py train   --circuits ...
       python scripts/run_seq1.py warm    --circuits ...
       python scripts/run_seq1.py mle     --circuits ...   (and mle_adam)
       (ceiling also takes --inits 0 1 2 ...)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from archb import ceiling, train as T, variants as V  # noqa: E402
from archb.metrics import all_metrics  # noqa: E402
from archb.ou import ENCODINGS, exact_kernel, load_dataset  # noqa: E402
from run_all import carry_diagnostic  # noqa: E402

OUT = ROOT / "runs" / "seq1"
INITS = tuple(range(10))
SEEDS = (0, 1, 2)
TRAIN = {"steps": 1500, "lr": 0.03, "eps": 0.01, "noise": 0.2, "sinkhorn_iters": 100, "stop_smooth": 25}


def init_noise(k: int) -> float:
    return 0.2 if k < 5 else 0.5


def score(P) -> dict:
    P_star = exact_kernel()
    return {**all_metrics(P, P_star), **carry_diagnostic(P, P_star)}


def save(d: Path, theta, kernel, loss, info: dict):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "theta.npy", np.asarray(theta))
    np.save(d / "kernel.npy", np.asarray(kernel))
    np.save(d / "loss.npy", np.asarray(loss))
    info["metrics"] = score(np.asarray(kernel))
    (d / "metrics.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def run_ceiling(name: str, enc: str, k: int):
    d = OUT / "ceiling" / name / enc / f"init{k}"
    if (d / "metrics.json").exists():
        return json.loads((d / "metrics.json").read_text())
    variant, L = V.LINEUP[name]
    theta0 = V.init_theta(k, init_noise(k), variant, L)
    res = ceiling.fit(lambda th: V.kernel(th, enc, variant, L), theta0)
    P = np.asarray(V.kernel(res["theta"], enc, variant, L))
    info = {"circuit": name, "variant": variant, "n_layers": L, "n_params": V.n_params(variant, L),
            "encoding": enc, "init": k, "noise": init_noise(k), "steps": res["steps"],
            "final_kl": res["final_loss"], "wall_time": res["wall_time"]}
    info.update(adam_kl=res["adam_loss"], lbfgs_iters=res["lbfgs_iters"], lbfgs_message=res["lbfgs_message"])
    return save(d, res["theta"], P, res["loss"], info)


def run_mle(name: str, enc: str, seed: int, mode: str = "mle"):
    d = OUT / mode / name / enc / f"seed{seed}"
    if (d / "metrics.json").exists():
        return json.loads((d / "metrics.json").read_text())
    variant, L = V.LINEUP[name]
    kfn = lambda th: V.kernel(th, enc, variant, L)  # noqa: E731
    theta0 = V.init_theta(seed, TRAIN["noise"], variant, L)
    obj = ceiling.data_kl_objective(kfn, load_dataset())
    if mode == "mle":
        res = ceiling.fit(kfn, theta0, loss_fn=obj)
    else:  # mle_adam
        res = ceiling.adam(kfn, theta0, TRAIN["steps"], loss_fn=obj, lr0=TRAIN["lr"], lr1=TRAIN["lr"])
        res.update(adam_loss=res["final_loss"], lbfgs_iters=0)
    P = np.asarray(kfn(res["theta"]))
    info = {"circuit": name, "variant": variant, "n_layers": L, "n_params": V.n_params(variant, L),
            "encoding": enc, "seed": seed, "mode": mode, "steps": res["steps"], "final_data_kl": res["final_loss"],
            "adam_data_kl": res["adam_loss"], "lbfgs_iters": res["lbfgs_iters"], "wall_time": res["wall_time"]}
    return save(d, res["theta"], P, res["loss"], info)


def best_ceiling_theta(name: str, enc: str):
    d = OUT / "ceiling" / name / enc
    runs = [(json.loads((p / "metrics.json").read_text())["final_kl"], p) for p in d.glob("init*")]
    return np.load(min(runs)[1] / "theta.npy")


def run_train(name: str, enc: str, seed: int, mode: str = "train"):
    d = OUT / mode / name / enc / f"seed{seed}"
    if (d / "metrics.json").exists():
        return json.loads((d / "metrics.json").read_text())
    variant, L = V.LINEUP[name]
    model = V.Model(variant, L)
    T.MODELS[name] = model
    theta0 = best_ceiling_theta(name, enc) if mode == "warm" else None
    res = T.train(enc, seed, steps=TRAIN["steps"], lr=TRAIN["lr"], eps=TRAIN["eps"], noise=TRAIN["noise"],
                  log_every=0, sinkhorn_iters=TRAIN["sinkhorn_iters"], model=name,
                  stop_smooth=TRAIN["stop_smooth"], theta0=theta0)
    info = {"circuit": name, "variant": variant, "n_layers": L, "n_params": model.n_params, "encoding": enc,
            "seed": seed, "mode": mode, "steps": res["steps"], "init_loss": res["init_loss"],
            "final_loss": res["final_loss"], "wall_time": res["wall_time"], "settings": TRAIN}
    return save(d, res["theta"], res["kernel"], res["loss"], info)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ceiling", "train", "warm", "mle", "mle_adam"])
    ap.add_argument("--inits", nargs="+", type=int, default=list(INITS))
    ap.add_argument("--circuits", nargs="+", default=list(V.LINEUP))
    ap.add_argument("--encodings", nargs="+", default=list(ENCODINGS))
    a = ap.parse_args(argv)
    for name in a.circuits:
        for enc in a.encodings:
            items = {"ceiling": a.inits, "train": SEEDS, "mle": SEEDS, "mle_adam": SEEDS, "warm": (0,)}[a.what]
            for k in items:
                t0 = time.time()
                if a.what == "ceiling":
                    info = run_ceiling(name, enc, k)
                elif a.what in ("mle", "mle_adam"):
                    info = run_mle(name, enc, k, a.what)
                else:
                    info = run_train(name, enc, k, a.what)
                m = info["metrics"]
                print(f"{a.what:7s} {name:8s} {enc:6s} {k}  WTV {m['weighted_tv']:.4f}  statTV "
                      f"{m['stationary_tv']:.4f}  flux5 {m['flux_ratio_depth5']:.3f}  flux4 "
                      f"{m['flux_ratio_depth4']:.3f}  steps {info['steps']}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
