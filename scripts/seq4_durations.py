"""Sequence 4, smsar: law of regime durations implied by each model (plan metric 3). A model has no regime variable,
so a path of N_SAMPLE steps is sampled from the selected fit (best training likelihood) and its regimes are decoded
by the TRUE process's smoother (MAP regime per step); the run lengths are compared with those decoded the same way
from a path of the truth: total variation between the two run-length laws (lengths 1..3N, longer runs pooled) and
mean decoded duration. -> runs/seq4/<dataset>/durations.json

Usage: python scripts/seq4_durations.py [--datasets smsar4 smsar8 smsar16]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seq4 as RS  # noqa: E402

from archb import cqsm, iolin, qseq, regimes as R  # noqa: E402

N_SAMPLE = 100000
MODELS = ["archb-anc2", "cqsm-m1", "cqsm-m2", "cqsm-m1-anc2", "cqsm-m2-anc2", "qsm-m1", "qsm-m2", "iohmm-full-2",
          "iohmm-full-4", "iohmm-full-16", "iohmm-fact-2", "iohmm-fact-4", "iohmm-fact-16", "family-msar",
          "family-smsar", "truth"]


def rebuild(ds, mo):
    """IOLin of the init with the best training likelihood."""
    d = RS.OUT / ds / mo
    best = max(d.glob("*/result.json"), key=lambda p: json.loads(p.read_text())["train_ll"]).parent
    z = np.load(best / "params.npz")
    p = mo.split("-")
    if p[0] in ("archb", "cqsm"):
        m = 0 if p[0] == "archb" else int(p[1][1:])
        return cqsm.io(z["theta"], m, 2 if mo.endswith("anc2") else 0, RS.L)
    if p[0] == "qsm":
        m = int(p[1][1:])
        M = cqsm.qsm_transfer(qseq.kraus(jnp.asarray(z["theta"]), "iso", m, 5))
        return iolin.IOLin(np.asarray(M), np.asarray(cqsm.f_vec(m)))
    if p[0] == "truth":
        s = RS.DATASETS[ds]
        return iolin.IOLin.from_iohmm(R.truth_T(s["kind"], s["N"]))
    return iolin.IOLin.from_iohmm(z["T"])


def law(runs, cap):
    h = np.bincount(np.minimum(runs, cap), minlength=cap + 1)[1:]
    return h / h.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["smsar4", "smsar8", "smsar16"])
    a = ap.parse_args()
    for ds in a.datasets:
        s = RS.DATASETS[ds]
        cap = 3 * s["N"]
        out, ref = {}, None
        for mo in MODELS:
            if not (RS.OUT / ds / mo).exists():
                continue
            xs = rebuild(ds, mo).sample(N_SAMPLE, 1234)
            runs = R.run_lengths(R.decode_regimes(s["kind"], s["N"], xs))
            out[mo] = {"law": law(runs, cap).tolist(), "mean": float(runs.mean()), "n_runs": int(len(runs))}
        ref = np.asarray(out["truth"]["law"])
        for mo in out:
            out[mo]["tv_to_truth"] = float(0.5 * np.abs(np.asarray(out[mo]["law"]) - ref).sum())
            print(ds, mo, f"mean {out[mo]['mean']:.2f}  TV {out[mo]['tv_to_truth']:.3f}", flush=True)
        (RS.OUT / ds / "durations.json").write_text(json.dumps({"n_sample": N_SAMPLE, "cap": cap, "models": out},
                                                               indent=1) + "\n")


if __name__ == "__main__":
    main()
