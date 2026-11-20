"""Sequence 7: consolidate the one positive quantum-memory result of sequence 5 (uniform renewal N = 32: QSM_4 beat
HMM_16 at equal filter size by about 20%, two seeds) on the compute server. Same models, fits and metrics as
scripts/run_seq5.py (imported, not copied); only the datasets, the job table and the budgets change.

What sequence 5 left open, and the experiment that answers it here:
  core     replication of the decisive pair: renewal N = 32, five independent training strings (seeds 0-4; seeds 0, 1
           are the sequence 5 strings), QSM D = 4 (6 inits) against HMM S = 16 (equal filter size, 6 inits + the
           countdown 'lump' init) and S = 32 (the true size: in sequence 5 its single fit stopped on its budget, so
           "HMM_32 beats QSM_8" was non-decisive), the truth.                                              tier 1
  memory   memory-size curve: N = 16, 32, 64, training seeds 0-2, QSM D = 2, 4, 8, 16 (m = 1-4 memory qubits) against
           HMM S = 2, 4, 8, 16, 32 (+ 64 for N = 64), Markov orders 1-8, the truth. Equal memory: QSM_D vs HMM_D;
           equal filter size: QSM_D vs HMM_{D^2}.                                                         tier 2
  datasize renewal N = 32 with T = 2e5 training symbols (4x sequence 5), seeds 0-2: QSM D = 4, 8, 16, HMM S = 16, 32,
           truth. Sequence 5 saw QSM_8 overfit at T = 5e4; this says whether that was data or model.   tier 3
  es       HMM_32 overfits in the core fits (training likelihood above the truth's, held-out KL worse than HMM_16), so
           HMM S = 16 and 32 again with early stopping: Baum-Welch on the first 80% of the training string, the iterate
           with the best likelihood on the last 20% kept (checked every 20 iterations, stop after 10 checks without
           gain), no L-BFGS polish; selected among inits by that validation likelihood. Core strings (6 inits) and
           the T = 2e5 strings (HMM_32, 4 inits).                                                        tier 4
  curve    data-size curve of renewal N = 32: T = 2.5e4, 1e5, 4e5 (seeds 0-4, new strings) and two more T = 2e5 strings
           (seeds 3, 4), each with QSM D = 4 (4 inits), D = 8 (3), HMM S = 16 (3 + lump), HMM_32 with early stopping
           (4), the truth; QSM_8 on the T = 5e4 strings 3, 4. With the core and datasize fits this gives 5 strings at
           every T in {2.5e4, 5e4, 1e5, 2e5, 4e5}: where the true-size HMM overtakes QSM_4, and whether QSM_4 stays
           ahead of HMM_16 (equal filter size) as T grows. Caps x1 / 2 / 3 / 6 for T <= 5e4 / 1e5 / 2e5 / 4e5. tier 5
Budgets: L-BFGS up to 20,000 iterations (sequence 5: 3,000) and Baum-Welch up to 3,000 (500), with wall-clock caps
sized so that fits end on convergence, not on the budget; every result records capped and the optimizer message.
The held-out string (201,600 symbols, seed 999 per N) is shared by all training seeds of one N, as in sequence 5.

-> runs/seq7/<exp>/<dataset>/<model>/<init>/{result.json, params.npz, loss.npy, lp_test.npy}; data in runs/seq7/data/.
Resumable (finished fits skipped, lock files with the pid), so the server job can be stopped and restarted.

Usage: python scripts/run_seq7.py [--exps ...] [--tiers ...] [--datasets ...] [--models ...] [--inits ...]
                                  [--workers 32] [--max-minutes M] [--out DIR] [--list]
       python scripts/run_seq7.py --smoke     (one fit per type, 20 s caps, -> runs/seq7_smoke/)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

import run_seq5 as S5  # noqa: E402
from archb import hmm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Job = S5.Job

# Every module-level change below also runs in the worker processes (spawn re-imports this file as __mp_main__ before
# the first task), so the sequence 5 fit functions see the new datasets and budgets.
S5.CFG["out"] = ROOT / "runs" / "seq7"
S5.LBFGS, S5.EM_ITERS = 20000, 3000

DATASETS = {}
for _N, _seeds, _base in ((32, range(5), 1200), (16, range(3), 1100), (64, range(3), 1600)):
    for _s in _seeds:  # base as sequence 5 (1200 + s for N = 32, 1100 + s for N = 16): its strings are seeds 0, 1 / 0
        DATASETS[f"renewal{_N}-s{_s}"] = {"kind": "renewal", "N": _N, "T": 50000, "seed": _base + _s}
for _s in range(5):  # seeds 0-2: tier 3; 3, 4: tier 5
    DATASETS[f"renewal32-T200k-s{_s}"] = {"kind": "renewal", "N": 32, "T": 200000, "seed": 1700 + _s}
CURVE_T = {25000: ("T25k", 1800), 100000: ("T100k", 1900), 400000: ("T400k", 2000)}  # tag, seed base
for _T, (_tag, _base) in CURVE_T.items():
    for _s in range(5):
        DATASETS[f"renewal32-{_tag}-s{_s}"] = {"kind": "renewal", "N": 32, "T": _T, "seed": _base + _s}
S5.DATASETS.update(DATASETS)

INITS = ["0", "1", "2", "3", "4", "5", "lump"]


def _init(i):
    return i if i == "lump" else int(i)


# Wall-clock caps (seconds) for T = 5e4; 3x for T = 2e5. Measured on 1 core (x86, T = 5e4): one loss + gradient
# evaluation takes 0.015 / 0.026 / 0.038 / 0.20 s for QSM D = 2 / 4 / 8 / 16, one Baum-Welch iteration 0.3 / 0.6 / 1.8 s
# for HMM S = 16 / 32 / 64. The QSM caps leave room for Adam + 20,000 L-BFGS iterations; for HMM S >= 32 the L-BFGS
# polish after EM can still reach the cap (EM gets at most half of it). A capped fit keeps its best point and says so.
QSM_CAP = {2: 1200, 4: 2400, 8: 5400, 16: 10800}
HMM_CAP = {2: 600, 4: 600, 8: 1200, 16: 2400, 32: 7200, 64: 14400}


def qsm(tier, exp, ds, D, n, scale=1):
    return [Job(tier, exp, ds, f"qsm-D{D}-r1", i, scale * QSM_CAP[D]) for i in range(n)]


def hmm_(tier, exp, ds, S, inits, scale=1):
    """The countdown 'lump' init merges the true chain's states into S groups; with S >= N it is the truth itself (an
    oracle start), so it is used only for S < N."""
    N = DATASETS[ds]["N"]
    return [Job(tier, exp, ds, f"hmm-{S}", _init(i), scale * HMM_CAP[S]) for i in inits if i != "lump" or S < N]


def jobs_all():
    J = []
    for s in range(5):  # core
        ds = f"renewal32-s{s}"
        J += qsm(1, "core", ds, 4, 6) + hmm_(1, "core", ds, 16, INITS) + hmm_(1, "core", ds, 32, INITS[:3] + ["lump"])
        J += [Job(1, "core", ds, "truth", 0, 10)]
    for N in (16, 32, 64):  # memory
        for s in range(3):
            ds = f"renewal{N}-s{s}"
            J += qsm(2, "memory", ds, 2, 4) + qsm(2, "memory", ds, 8, 4) + qsm(2, "memory", ds, 16, 3)
            J += qsm(2, "memory", ds, 4, 6) if N != 32 else []  # N = 32: the core fits
            for S in (2, 4, 8):
                J += hmm_(2, "memory", ds, S, INITS[:4] + ["lump"])
            J += hmm_(2, "memory", ds, 16, INITS) if N != 32 else []
            J += hmm_(2, "memory", ds, 32, INITS[:3] + ["lump"]) if N != 32 else []
            J += hmm_(2, "memory", ds, 64, INITS[:3] + ["lump"]) if N == 64 else []
            J += [Job(2, "memory", ds, f"markov-{k}", 0, 60) for k in range(1, 9)]
            J += [Job(2, "memory", ds, "truth", 0, 10)] if N != 32 else []
    for s in range(3):  # datasize
        ds = f"renewal32-T200k-s{s}"
        J += qsm(3, "datasize", ds, 4, 4, 3) + qsm(3, "datasize", ds, 8, 4, 3) + qsm(3, "datasize", ds, 16, 2, 3)
        J += hmm_(3, "datasize", ds, 16, INITS[:4] + ["lump"], 3) + hmm_(3, "datasize", ds, 32, INITS[:3] + ["lump"], 3)
        J += [Job(3, "datasize", ds, "truth", 0, 10)]
    for s in range(5):  # es
        ds = f"renewal32-s{s}"
        J += [Job(4, "es", ds, f"hmm-{S}-es", i, HMM_CAP[S]) for S in (16, 32) for i in range(6)]
    for s in range(3):
        J += [Job(4, "es", f"renewal32-T200k-s{s}", "hmm-32-es", i, 3 * HMM_CAP[32]) for i in range(4)]
    for s in range(5):  # curve
        for ds in [f"renewal32-{tag}-s{s}" for tag, _ in CURVE_T.values()] + ([f"renewal32-T200k-s{s}"] if s >= 3 else []):
            k = curve_scale(DATASETS[ds]["T"])
            J += qsm(5, "curve", ds, 4, 4, k) + qsm(5, "curve", ds, 8, 3, k)
            J += hmm_(5, "curve", ds, 16, INITS[:3] + ["lump"], k)
            J += [Job(5, "curve", ds, "hmm-32-es", i, k * HMM_CAP[32]) for i in range(4)]
            J += [Job(5, "curve", ds, "truth", 0, 10)]
        if s >= 3:  # T = 5e4: QSM_8 ran on seeds 0-2 only (memory tier)
            J += qsm(5, "curve", f"renewal32-s{s}", 8, 3)
    return J


def curve_scale(T):
    """Wall-clock cap multiplier of the curve fits (the cost of a fit grows linearly with T)."""
    return {25000: 1, 50000: 1, 100000: 2, 200000: 3, 400000: 6}[T]


ES_CHUNK, ES_PATIENCE = 20, 10


def fit_hmm_es(job, xs, test):
    """Baum-Welch with early stopping on the last 20% of the training string (see 'es' above)."""
    S, t0 = int(job.model.split("-")[1]), time.time()
    n_val = len(xs) // 5
    tr, va = xs[:-n_val], xs[-n_val:]
    T, best, curve, val_curve, since, iters, converged = None, None, [], [], 0, 0, False
    while iters < S5.EM_ITERS and time.time() - t0 < job.cap and since < ES_PATIENCE and not converged:
        em = hmm.baum_welch(tr, S, 2, seed=job.init, max_iter=ES_CHUNK, T0=T)
        T, iters, converged = em["T"], iters + em["iters"], em["iters"] < ES_CHUNK
        curve += list(em["curve"])
        v = float(np.mean(hmm.hmm_logprobs_abel(T, va)))
        val_curve.append(v)
        if best is None or v > best[0]:
            best, since = (v, T, iters), 0
        else:
            since += 1
    val_ll, T, it_best = best
    info = {"S": S, "filter_dim": S, "n_params": hmm.n_params_hmm(S, 2), "val_ll": val_ll, "em_iters": iters,
            "em_iters_best": it_best, "em_converged": converged, "n_train_fit": len(tr), "n_val": n_val,
            "capped": time.time() - t0 >= job.cap, "selection": "validation"}
    train_ll = float(np.mean(hmm.hmm_logprobs_abel(T, xs)))
    return hmm.hmm_linear_abel(T), hmm.hmm_logprobs_abel(T, test), train_ll, info, {"T": T}, \
        np.concatenate([-np.asarray(curve), -np.asarray(val_curve)])


_fit_hmm = S5.fit_hmm


def fit_hmm(job, xs, test):
    return fit_hmm_es(job, xs, test) if job.model.endswith("-es") else _fit_hmm(job, xs, test)


S5.fit_hmm = fit_hmm  # run_seq5.fit looks it up at call time, also in the spawned workers


def result(job):
    """Path of a job's result; core fits on N = 32 seeds 0-2 also serve the memory curve (read by the summary)."""
    return S5.result_dir(job) / "result.json"


def ordered(jobs):  # tier first, then the longest cap first so the big fits do not end the run alone
    return [j for _, j in sorted(enumerate(jobs), key=lambda t: (t[1].tier, -t[1].cap, t[0]))]


def worker_init(out, smoke):
    S5.configure(out, smoke)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None)
    ap.add_argument("--tiers", nargs="*", type=int, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="model name prefixes")
    ap.add_argument("--inits", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output directory (default runs/seq7)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 6)))
    ap.add_argument("--max-minutes", type=float, default=1e9, help="start no new fit after this many minutes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one fit per type, caps <= 20 s, no Adam, -> runs/seq7_smoke/")
    a = ap.parse_args()
    if a.smoke or a.out:
        S5.configure(Path(a.out) if a.out else ROOT / "runs" / "seq7_smoke", smoke=a.smoke)
    jobs = [j for j in ordered(jobs_all()) if (a.exps is None or j.exp in a.exps)
            and (a.tiers is None or j.tier in a.tiers) and (a.datasets is None or j.dataset in a.datasets)
            and (a.models is None or any(j.model.startswith(p) for p in a.models))
            and (a.inits is None or str(j.init) in a.inits)]
    if a.smoke:
        jobs = S5.smoke_selection(jobs)
    if a.list:
        S5.list_jobs(jobs)
        return
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes
    pending = [j for j in jobs if not result(j).exists()]
    print(f"{datetime.now():%H:%M:%S} {len(jobs)} fits, {len(pending)} to do, {a.workers} workers, out {S5.CFG['out']}",
          flush=True)
    S5.prepare_data(sorted({j.dataset for j in pending}))
    running, n_done, n_err = {}, 0, 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=worker_init, initargs=(S5.CFG["out"], S5.CFG["smoke"]),
                  maxtasksperchild=8) as pool:
        while pending or running:
            if time.time() > deadline and not running:
                print(f"time budget reached: {len(pending)} fits not started", flush=True)
                break
            while pending and len(running) < a.workers and time.time() <= deadline:
                j = pending.pop(0)
                running[j] = (pool.apply_async(S5.run_job, (j,)), time.time())
            for j in [j for j, (r, _) in running.items() if r.ready()]:
                r, _ = running.pop(j)
                try:
                    _, st, wt, msg = r.get()
                except Exception as e:  # noqa: BLE001
                    st, wt, msg = "crash", 0.0, repr(e)
                n_done += st == "ran"
                n_err += st in ("error", "crash")
                print(f"{datetime.now():%H:%M:%S} [{st}] {j.exp} {j.dataset} {j.model} {j.init}: {msg} [{wt:.0f} s]",
                      flush=True)
            for j in [j for j, (_, ts) in running.items() if time.time() - ts > 2 * j.cap + 900]:
                running.pop(j)
                print(f"{datetime.now():%H:%M:%S} [lost] {j}: no answer after {2 * j.cap + 900} s", flush=True)
            time.sleep(1)
    print(f"{datetime.now():%H:%M:%S} {n_done} fits ran, {n_err} errors, {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
