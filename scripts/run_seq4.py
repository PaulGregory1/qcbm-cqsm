"""Sequence 4: conditional QCBM with a quantum memory (C-QSM) against Architecture B, the QSM of
sequence 3, input-output HMMs, Markov chains and the true families, on regime-switching AR targets.
Modules: archb.cqsm (C-QSM), archb.iolin (common representation, exact metrics), archb.regimes (targets, classical
models).

Datasets (runs/seq4/data/<name>.npz: 2,000 training paths of 50 values, a held-out path of 200,000 steps + regimes):
  msar        target A, Markov-switching AR(1)
  smsar<N>    target B, semi-Markov switching, regime durations uniform on {1..N}, N = 4, 8, 16
Models (ids = inits):
  archb-B0, archb-anc2   Architecture B (sequences 1-2, 3 layers), maximum likelihood, inits 0..2 (= stage 1)
  cqsm-m<m>[-anc2]       C-QSM, m memory qubits, 3 layers, staged: from the best stage 1 fit (archb-B0 / archb-anc2)
                         with the memory released (small random angles, memory R_y + pi/2), stage 2 seeds 0..4
  qsm-m<m>               sequence 3 QSM (general isometry, 32 symbols), no input, inits 0..4
  iohmm-<full|fact>-<S>  input-output HMM by EM, S = 2, 4, 16 (and S = 2N, fact, on smsar), inits 0..4
  hmm-<S>                HMM on x alone (tied IOHMM), S = 2, 4, inits 0..4
  markov-<k>             order-k chain on x, add-1/2 counts, k = 1..3 (k = 1: the tabular Markov kernel)
  family-msar, family-smsar   true family by maximum likelihood (MS-AR also on smsar: Markov switching, misspecified)
  truth                  the generating process
Training of the circuits: exact mean log-likelihood of the paths (first value from the model's joint stationary law),
Adam then L-BFGS-B (archb.ceiling.fit): stage 1 and QSM Adam 500 steps lr 0.03 -> 0.003, L-BFGS cap 2,000; stage 2 Adam
300 steps lr 0.003 -> 0.001 (a warm start), L-BFGS cap 3,000.
Metrics: held-out KL rate (exact filtering on both sides, standard error over 20 segments); the one-step kernel
averaged over the hidden state against the truth's (archb.targets.metrics); exact acf of |x| (lags 1-50), kurtosis,
10-step VaR / ES at 1% and 5%; linear readout of the true regime from the filtered internal state (fit on the first
half of the held-out path, accuracy on the second).

-> runs/seq4/<dataset>/<model>/<id>/{result.json, params.npz, loss.npy}. Resumable; lock files with the pid keep
workers off the same run; stage 2 runs wait for their stage 1 inits.

Usage: python scripts/run_seq4.py [--datasets ...] [--models ...] [--max-minutes M] [--list]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from archb import ceiling, cqsm, hmm, iolin, qseq, regimes as R, targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "seq4"
L = 3
STAGE1, STAGE2, QSM_INITS, EM_INITS = range(3), range(5), range(5), range(5)
ACF_LAGS, HORIZON = 50, 10
GRID = targets.Target("seq4", 1, 5, R.LO, R.HI, {}, seed=0)

DATASETS = {"msar": {"kind": "msar", "N": None, "seed": 700}}
for _N, _s in ((4, 710), (8, 720), (16, 730)):
    DATASETS[f"smsar{_N}"] = {"kind": "smsar", "N": _N, "seed": _s}


def models(spec):
    out = [(f"archb-{v}", i) for v in ("B0", "anc2") for i in STAGE1]
    out += [(f"cqsm-m{m}{s}", i) for s in ("", "-anc2") for m in (1, 2) for i in STAGE2]
    out += [(f"qsm-m{m}", i) for m in (1, 2) for i in QSM_INITS]
    sizes = [2, 4, 16]
    out += [(f"iohmm-{k}-{S}", i) for k in ("full", "fact") for S in sizes for i in EM_INITS]
    if spec["kind"] == "smsar" and 2 * spec["N"] not in sizes:
        out += [(f"iohmm-fact-{2 * spec['N']}", i) for i in EM_INITS]
    out += [(f"hmm-{S}", i) for S in (2, 4) for i in EM_INITS]
    out += [(f"markov-{k}", 0) for k in (1, 2, 3)] + [("family-msar", 0)]
    out += [("family-smsar", 0)] if spec["kind"] == "smsar" else []
    return out + [("truth", 0)]


def load(name):
    s = DATASETS[name]
    return R.load_data(name, s["kind"], s["N"], s["seed"])


_REF = {}


def reference(name):
    """Truth: representation, held-out log-probabilities and exact statistics (computed once per process)."""
    if name not in _REF:
        s, d = DATASETS[name], load(name)
        lin = iolin.IOLin.from_iohmm(R.truth_T(s["kind"], s["N"]))
        _REF[name] = {"lin": lin, "lp": lin.logprobs(d["test"]), "P": lin.kernel(),
                      "acf": lin.acf(np.abs(R.Z), ACF_LAGS), "kurt": kurtosis(lin), "var": lin.var_es(R.Z, HORIZON)}
    return _REF[name]


def kurtosis(lin):
    pi, z = lin.marginal(), R.Z
    mu = pi @ z
    return float(pi @ (z - mu) ** 4 / (pi @ (z - mu) ** 2) ** 2)


def readout(states, regimes):
    """Accuracy of a linear least-squares readout of the regime from the filtered internal state (real and imaginary
    parts), fitted on the first half of the held-out path, scored on the second; the majority rate for reference."""
    F = np.asarray(states)
    F = np.concatenate([F.real, F.imag], axis=1) if np.iscomplexobj(F) else F
    F = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    h = len(F) // 2
    w = np.linalg.lstsq(F[:h], regimes[:h].astype(float), rcond=None)[0]
    acc = float(np.mean((F[h:] @ w > 0.5) == regimes[h:]))
    return acc, float(max(regimes[h:].mean(), 1 - regimes[h:].mean()))


def evaluate(name, lin, lp, train_ll, X, exact=True):
    s, ref, d = DATASETS[name], reference(name), load(name)
    rate, se = hmm.seg_kl_rate(ref["lp"], lp)
    out = {"heldout_kl_rate": rate, "heldout_se": se, "train_ll": train_ll,
           "truth_train_ll": ref["lin"].path_loglik(X), "truth_test_ll": float(np.mean(ref["lp"]))}
    if not exact:
        return out
    out["kernel"] = targets.metrics(lin.kernel(), GRID, ref["P"])
    acf = lin.acf(np.abs(R.Z), ACF_LAGS)
    out["acf_abs"], out["acf_abs_err_max"] = acf.tolist(), float(np.abs(acf - ref["acf"]).max())
    out["acf_abs_err_mean"] = float(np.abs(acf - ref["acf"]).mean())
    out["kurtosis"], out["truth_kurtosis"] = kurtosis(lin), ref["kurt"]
    out["var_es"], out["truth_var_es"] = lin.var_es(R.Z, HORIZON), ref["var"]
    if lin.d > 1:
        reg = R.regime_of(s["kind"], s["N"])[d["test_h"]]
        out["readout_acc"], out["readout_majority"] = readout(lin.filtered(d["test"]), reg)
    return out


# ------------------------------------------------------------------------------------------------ fitters
def _result_dir(name, model, i):
    return OUT / name / model / str(i)


def best_stage1(name, model):
    """Best training likelihood over the stage 1 inits, or None while one is missing."""
    best = None
    for i in STAGE1:
        p = _result_dir(name, model, i) / "result.json"
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        if best is None or r["train_ll"] > best[0]:
            best = (r["train_ll"], i)
    return np.load(_result_dir(name, model, best[1]) / "params.npz")["theta"], best[1]


def fit_circuit(name, model, i, X):
    Xj = jnp.asarray(X)
    if model.startswith("archb"):
        m, a = 0, (2 if model.endswith("anc2") else 0)
        th0, extra = cqsm.init_theta(i, m, a, L), {}
        kw = {"adam_steps": 500, "lbfgs_iters": 2000}
    else:
        m, a = int(model.split("-")[1][1:]), (2 if model.endswith("anc2") else 0)
        st = best_stage1(name, "archb-anc2" if a else "archb-B0")
        th0, extra = cqsm.embed(st[0], i, m, a, L), {"stage1_init": st[1]}
        kw = {"adam_steps": 300, "lbfgs_iters": 3000, "lr0": 0.003, "lr1": 0.001}
    loss = lambda t: -cqsm.loglik(t, Xj, m, a, L)
    r = ceiling.fit(None, jnp.asarray(th0), loss_fn=loss, **kw)
    lin = cqsm.io(r["theta"], m, a, L)
    info = {"memory_dim": 2**m, "n_ancillas": a, "n_layers": L, "n_params": cqsm.n_params(m, a, L),
            "adam_ll": -r["adam_loss"], "init_ll": float(-loss(jnp.asarray(th0))), "lbfgs_iters": r["lbfgs_iters"],
            "lbfgs_message": r["lbfgs_message"], **extra}
    if m:
        info["mem_angle_rms"] = float(np.sqrt(np.mean(r["theta"][cqsm.mem_mask(m, a, L)] ** 2)))
    return lin, -r["final_loss"], info, {"theta": r["theta"]}, r["loss"]


def fit_qsm(name, model, i, X):
    m, Xj = int(model[-1]), jnp.asarray(X)
    f = cqsm.f_vec(m)
    M = lambda t: cqsm.qsm_transfer(qseq.kraus(t, "iso", m, 5))
    loss = lambda t: -iolin.loglik(M(t), f, Xj)
    r = ceiling.fit(None, jnp.asarray(qseq.init_theta(i, "iso", m, 5)), adam_steps=500, lbfgs_iters=2000, loss_fn=loss)
    lin = iolin.IOLin(np.asarray(M(jnp.asarray(r["theta"]))), np.asarray(f))
    info = {"memory_dim": 2**m, "n_params": qseq.n_params("iso", m, 5), "n_effective": qseq.n_effective(m, 5),
            "adam_ll": -r["adam_loss"], "lbfgs_iters": r["lbfgs_iters"], "lbfgs_message": r["lbfgs_message"]}
    return lin, -r["final_loss"], info, {"theta": r["theta"]}, r["loss"]


def fit_em(name, model, i, X):
    parts = model.split("-")
    kind, S = ("tied", int(parts[1])) if parts[0] == "hmm" else (parts[1], int(parts[2]))
    T, curve = R.iohmm_em(X, S, kind, seed=i)
    lin = iolin.IOLin.from_iohmm(T)
    info = {"memory_dim": S, "n_params": R.n_params_iohmm(kind, S), "em_iters": len(curve), "em_ll": float(curve[-1])}
    return lin, lin.path_loglik(X), info, {"T": T}, -curve


def fit_markov(name, model, i, X):
    k = int(model[-1])
    tabs = R.markov_tables(X, k)
    lin = None if k == 3 else iolin.IOLin.from_kernel(tabs[1]) if k == 1 else iolin.IOLin.from_markov2(tabs[2])
    info = {"memory_dim": R.A ** (k - 1), "n_params": R.n_params_markov(k)}
    return lin, R.markov_path_loglik(tabs, X), info, {"Pc": tabs[k]}, np.zeros(0), \
        R.markov_logprobs(tabs, load(name)["test"])


def fit_family(name, model, i, X):
    s, kind = DATASETS[name], model.split("-")[1]
    N = s["N"] if kind == "smsar" else None
    T, p, ll, info = R.fit_family(kind, X, N)
    return iolin.IOLin.from_iohmm(T), ll, {"memory_dim": T.shape[1], **info, "params": p}, {"T": T}, np.zeros(0)


def fit_truth(name, model, i, X):
    lin = reference(name)["lin"]
    return lin, lin.path_loglik(X), {"memory_dim": lin.d, "n_params": None}, {}, np.zeros(0)


def fitter(model):
    if model.startswith(("archb", "cqsm")):
        return fit_circuit
    return {"qsm": fit_qsm, "iohmm": fit_em, "hmm": fit_em, "markov": fit_markov, "family": fit_family,
            "truth": fit_truth}[model.split("-")[0]]


def run(name, model, i):
    d = _result_dir(name, model, i)
    if (d / "result.json").exists():
        return "done"
    if model.startswith("cqsm") and best_stage1(name, "archb-anc2" if model.endswith("anc2") else "archb-B0") is None:
        return "waiting"
    d.mkdir(parents=True, exist_ok=True)
    lock = d / "lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0)
            return "locked"
        except (ValueError, ProcessLookupError):
            lock.write_text(str(os.getpid()))  # stale lock of a killed worker
    t0 = time.time()
    X = load(name)["X"]
    out = fitter(model)(name, model, i, X)
    lin, train_ll, info, params, loss = out[:5]
    lp = out[5] if len(out) > 5 else lin.logprobs(load(name)["test"])
    res = {"dataset": name, **DATASETS[name], "model": model, "init": i, **info}
    res.update(evaluate(name, lin, lp, train_ll, X, exact=lin is not None))
    res["wall_time"] = time.time() - t0
    np.savez(d / "params.npz", **params)
    np.save(d / "loss.npy", np.asarray(loss))
    (d / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    lock.unlink(missing_ok=True)
    return "ran"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["msar"])
    ap.add_argument("--models", nargs="*", default=None, help="model name prefixes (default: all)")
    ap.add_argument("--max-minutes", type=float, default=600.0, help="start no new run after this many minutes")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(n, mo, i) for n in a.datasets for mo, i in models(DATASETS[n])
            if a.models is None or any(mo.startswith(p) for p in a.models)]
    if a.list:
        todo = [j for j in jobs if not (_result_dir(*j) / "result.json").exists()]
        print(f"{len(jobs)} runs, {len(todo)} to do")
        for j in todo:
            print(*j)
        return
    while True:
        waiting = 0
        for n, mo, i in jobs:
            if (time.time() - t0) / 60 > a.max_minutes:
                print("time budget reached", flush=True)
                sys.exit(3)
            t1 = time.time()
            st = run(n, mo, i)
            waiting += st in ("waiting", "locked")
            if st == "ran":
                r = json.loads((_result_dir(n, mo, i) / "result.json").read_text())
                print(f"{n} {mo} {i}: train ll {r['train_ll']:.5f} (truth {r['truth_train_ll']:.5f}) held-out KL rate "
                      f"{r['heldout_kl_rate']:.3e} ± {r['heldout_se']:.1e} [{time.time() - t1:.0f} s]", flush=True)
        if not waiting:
            break
        time.sleep(20)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
