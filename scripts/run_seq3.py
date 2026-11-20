"""Sequence 3: quantum sequence model (QSM) with a memory register vs classical HMMs and
Markov chains on non-Markov processes. Modules: archb.qseq (QSM), archb.hmm (processes, baselines, exact metrics).

Datasets (runs/seq3/data/<name>.npz: train string, held-out string of 201,600 symbols from the same process):
  renewalN              uniform renewal process, N = 3..8, T = 50,000
  renewal5-T5000-s<k>   N = 5, T = 5,000, data seeds k = 0..2
  renewal5-T500-s<k>    N = 5, T = 500, data seeds k = 0..2
  regime                two hidden regimes (sd 0.5 / 1.5, stay 0.98), 8 bins (3 bits, Gray labels), T = 50,000
Models (ids = random inits):
  qsm-iso-m<m>    general QSM with m memory qubits (every isometry; archb.qseq "iso"), inits 0..5; m = 3 (D = 8) only
                  on renewal N = 6..8 at T = 50,000, inits 0..3
  qsm-circ-m<m>   circuit QSM (R_z R_y R_z + CNOT ring), layers chosen so that angles >= effective dimension, inits 0..3
  hmm-<S>         edge-emitting HMM with S states: Baum-Welch (<= 500 EM iterations) then L-BFGS polish, inits 0..5;
                  renewal at T = 50,000: init 100 = the true countdown chain with states >= S-1 lumped (an
                  oracle-informed start, S < N only), and inits 6..19 for N = 3, 5, 8
  markov-<k>      order-k Markov chain on the symbols, add-1/2 counts, k = 1..4
  truth           the exact process (reference)
QSM training: exact mean log-likelihood of the training string (forward filter from rho_ss), Adam 500 steps
(lr 0.03 -> 0.003) then L-BFGS (cap 2,000 iterations), archb.ceiling.fit.
Metrics (exact against the truth): KL(P_L || Q_L) for L = 1..12 (binary) / 1..5 (8 symbols), D_L = KL_L / L,
conditional d_L = KL_L - KL_{L-1}; held-out KL-rate estimate (1/n) sum_t log P(x_t|x_<t) - log Q(x_t|x_<t) on the
held-out string with its standard error over 20 segments; renewal gap law (n = 0..3N); regime: acf of |x| (bin
centres), lags 1..30.

-> runs/seq3/<dataset>/<model>/<id>/{result.json, params.npz, loss.npy}. Resumable: finished runs are skipped,
a lock file (with the pid) keeps two workers off the same run.

Usage: python scripts/run_seq3.py [--datasets ...] [--models ...] [--max-minutes M] [--list]
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

from archb import ceiling, hmm, qseq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "seq3"
N_TEST = 201600
ADAM, LBFGS = 500, 2000
EM_ITERS = 500
ACF_LAGS = 30
QSM_INITS, CIRC_INITS, HMM_INITS, M3_INITS = range(6), range(4), range(6), range(4)
HMM_EXTRA, LUMP = range(6, 20), 100  # renewal at T = 50,000: 20 random inits (N in EXTRA_NS) + the lumped-truth init
EXTRA_NS = (3, 5, 8)  # compute budget: the extra random inits only where they were run first
# circuit layers per (m, b): 3 n (L + 1) angles >= qseq.n_effective(m, b)
CIRC_LAYERS = {(1, 1): 2, (2, 1): 4, (1, 3): 5, (2, 3): 15}


def datasets():
    d = {}
    for N in range(3, 9):
        d[f"renewal{N}"] = {"kind": "renewal", "N": N, "T": 50000, "seed": 100 + N, "test_seed": 200 + N}
    for T, base in ((5000, 300), (500, 400)):
        for s in range(3):
            d[f"renewal5-T{T}-s{s}"] = {"kind": "renewal", "N": 5, "T": T, "seed": base + s, "test_seed": 205}
    d["regime"] = {"kind": "regime", "T": 50000, "seed": 500, "test_seed": 600}
    return d


DATASETS = datasets()
ORDER = ["regime", "renewal5", "renewal8", "renewal3", *[f"renewal5-T{T}-s{s}" for T in (5000, 500) for s in range(3)],
         "renewal4", "renewal6", "renewal7"]


def truth(spec):
    return hmm.renewal(spec["N"]) if spec["kind"] == "renewal" else hmm.regime()


def alphabet(spec):
    return 2 if spec["kind"] == "renewal" else 8


def lmax(spec):
    return 12 if alphabet(spec) == 2 else 5


def models(spec):
    out = []
    for m in (1, 2):
        out += [(f"qsm-iso-m{m}", i) for i in QSM_INITS]
    for m in (1, 2):
        out += [(f"qsm-circ-m{m}", i) for i in CIRC_INITS]
    states = sorted({2, 4, spec["N"]}) if spec["kind"] == "renewal" else [2, 4]
    for S in states:
        out += [(f"hmm-{S}", i) for i in HMM_INITS]
        if spec["kind"] == "renewal" and spec["T"] == 50000:  # strengthen the classical baseline where it matters
            out += [(f"hmm-{S}", i) for i in (HMM_EXTRA if spec["N"] in EXTRA_NS else ())]
            out += [(f"hmm-{S}", LUMP)] if S < spec["N"] else []
    out += [(f"markov-{k}", 0) for k in range(1, 5)] + [("truth", 0)]
    if spec["kind"] == "renewal" and spec["N"] >= 6 and spec["T"] == 50000:  # D = 8, against the HMM with N states
        out += [("qsm-iso-m3", i) for i in M3_INITS]
    return out


def load_data(name):
    spec = DATASETS[name]
    p = OUT / "data" / f"{name}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        T = truth(spec)
        np.savez(p, train=hmm.sample(T, spec["T"], spec["seed"]), test=hmm.sample(T, N_TEST, spec["test_seed"]))
    z = np.load(p)
    return z["train"], z["test"]


_CACHE = {}


def reference(name):
    """Truth word laws, held-out log-probabilities and long-range statistics, computed once per process."""
    if name not in _CACHE:
        spec = DATASETS[name]
        T = truth(spec)
        lin = hmm.hmm_linear(T)
        _, test = load_data(name)
        _CACHE[name] = {"P": lin.word_probs(lmax(spec)), "lp_test": hmm.hmm_logprobs(T, test)}
    return _CACHE[name]


def perm(spec):
    A = alphabet(spec)
    return np.arange(A) ^ (np.arange(A) >> 1)  # Gray label of each symbol (identity for A = 2)


def evaluate(name, model, lp_test):
    spec, ref = DATASETS[name], reference(name)
    kl, D, d = hmm.kl_curve(ref["P"], model.word_probs(lmax(spec)))
    rate, se = hmm.seg_kl_rate(ref["lp_test"], lp_test)
    out = {"kl": kl.tolist(), "D": D.tolist(), "d": d.tolist(), "heldout_kl_rate": rate, "heldout_se": se}
    if spec["kind"] == "renewal":
        out["gap"] = hmm.gap_dist(model, 3 * spec["N"]).tolist()
    else:
        _, c = hmm.regime_bins()
        out["acf_abs"] = model.acf(np.abs(c), ACF_LAGS).tolist()
    return out


def fit_qsm(name, model, i, train, test):
    spec = DATASETS[name]
    kind = "iso" if "iso" in model else "circuit"
    m, b = int(model[-1]), {2: 1, 8: 3}[alphabet(spec)]
    L = CIRC_LAYERS[(m, b)] if kind == "circuit" else None
    pm = perm(spec)
    data = hmm.blocks(train, 2**b)
    loss = lambda t: -qseq.block_loglik(qseq.kraus(t, kind, m, b, L, pm), data)
    r = ceiling.fit(None, qseq.init_theta(i, kind, m, b, L), adam_steps=ADAM, lbfgs_iters=LBFGS, loss_fn=loss)
    K = np.asarray(qseq.kraus(jnp.asarray(r["theta"]), kind, m, b, L, pm))
    lp = np.asarray(qseq.logprobs(jnp.asarray(K), jnp.asarray(test)))
    info = {"memory_dim": 2**m, "n_params": qseq.n_params(kind, m, b, L), "n_effective": qseq.n_effective(m, b),
            "n_layers": L, "train_ll": -r["final_loss"], "adam_ll": -r["adam_loss"], "lbfgs_iters": r["lbfgs_iters"],
            "lbfgs_message": r["lbfgs_message"]}
    return qseq.linear(K), lp, info, {"theta": r["theta"], "K": K}, r["loss"]


def fit_hmm(name, model, i, train, test):
    A, S = alphabet(DATASETS[name]), int(model.split("-")[1])
    T0 = None
    if i == LUMP:  # countdown states 0..S-2 kept, states >= S-1 merged, weighted by the stationary law
        Tt = truth(DATASETS[name])
        T0 = 0.95 * hmm.lump(Tt, np.minimum(np.arange(Tt.shape[0]), S - 1)) + 0.05 / (A * S)
    em = hmm.baum_welch(train, S, A, seed=i, max_iter=EM_ITERS, T0=T0)
    T, r = hmm.hmm_polish(em["T"], train, lbfgs_iters=LBFGS)
    info = {"memory_dim": S, "n_params": hmm.n_params_hmm(S, A), "n_effective": hmm.n_params_hmm(S, A),
            "train_ll": -r["final_loss"], "em_ll": em["loglik"], "em_iters": em["iters"],
            "lbfgs_iters": r["lbfgs_iters"], "lbfgs_message": r["lbfgs_message"]}
    return hmm.hmm_linear(T), hmm.hmm_logprobs(T, test), info, {"T": T, "T_em": em["T"]}, \
        np.concatenate([-em["curve"], r["loss"]])


def fit_markov(name, model, i, train, test):
    A, k = alphabet(DATASETS[name]), int(model.split("-")[1])
    mk = hmm.markov_fit(train, k, A)
    info = {"memory_dim": A**k, "n_params": hmm.n_params_markov(k, A), "n_effective": hmm.n_params_markov(k, A),
            "train_ll": float(np.mean(hmm.markov_logprobs(mk, train)))}
    return mk, hmm.markov_logprobs(mk, test), info, {"Pc": mk.Pc}, np.zeros(0)


def fit_truth(name, model, i, train, test):
    T = truth(DATASETS[name])
    S, A = T.shape[0], T.shape[1]
    info = {"memory_dim": S, "n_params": None, "n_effective": None, "train_ll": hmm.hmm_loglik(T, train)}
    return hmm.hmm_linear(T), hmm.hmm_logprobs(T, test), info, {"T": T}, np.zeros(0)


def run(name, model, i):
    d = OUT / name / model / str(i)
    if (d / "result.json").exists():
        return "done"
    d.mkdir(parents=True, exist_ok=True)
    lock = d / "lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        try:
            pid = int(lock.read_text() or 0)
            os.kill(pid, 0)
            return "locked"
        except (ValueError, ProcessLookupError):
            lock.write_text(str(os.getpid()))  # stale lock of a killed worker
    t0 = time.time()
    train, test = load_data(name)
    fitter = fit_qsm if model.startswith("qsm") else fit_hmm if model.startswith("hmm") else \
        fit_markov if model.startswith("markov") else fit_truth
    mdl, lp, info, params, loss = fitter(name, model, i, train, test)
    res = {"dataset": name, **DATASETS[name], "model": model, "init": i, **info,
           "truth_train_ll": hmm.hmm_loglik(truth(DATASETS[name]), train)}
    res.update(evaluate(name, mdl, lp))
    res["wall_time"] = time.time() - t0
    np.savez(d / "params.npz", **params)
    np.save(d / "loss.npy", np.asarray(loss))
    (d / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    lock.unlink(missing_ok=True)
    return "ran"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ORDER)
    ap.add_argument("--models", nargs="*", default=None, help="model name prefixes (default: all)")
    ap.add_argument("--max-minutes", type=float, default=110.0, help="start no new run after this many minutes")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(n, mo, i) for n in a.datasets for mo, i in models(DATASETS[n])
            if a.models is None or any(mo.startswith(p) for p in a.models)]
    if a.list:
        todo = [j for j in jobs if not (OUT / j[0] / j[1] / str(j[2]) / "result.json").exists()]
        print(f"{len(jobs)} runs, {len(todo)} to do")
        for j in todo:
            print(*j)
        return
    for n, mo, i in jobs:
        if (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached", flush=True)
            sys.exit(3)
        t1 = time.time()
        st = run(n, mo, i)
        if st == "ran":
            r = json.loads((OUT / n / mo / str(i) / "result.json").read_text())
            print(f"{n} {mo} {i}: train ll {r['train_ll']:.5f} (truth {r['truth_train_ll']:.5f}) held-out KL rate "
                  f"{r['heldout_kl_rate']:.2e} d_L {r['d'][-1]:.2e} [{time.time() - t1:.0f} s]", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
