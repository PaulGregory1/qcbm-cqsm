"""Sequence 8: controls of the C-QSM memory on the regime-switching AR targets, for the compute server. Same circuits,
data generator and metrics as sequences 4-5; the job table, the budgets and two new controls change.

  deph    retrained dephasing curve: C-QSM m = 1 + 2 anc trained with its memory dephased by gamma in
          {0, .1, .25, .5, .75, 1} after every step (gamma = 0 the coherent model, 1 a classical memory driven by the same
          circuit), m = 2 + 2 anc with gamma in {0, .5, 1}; msar and smsar16 (sequence 4 data and stage 1 fits),
          5 inits each.                                                                                    tier 1
  basis   basis-optimized dephasing: C-QSM m = 1, 2 + 2 anc with the memory fully dephased in a basis U trained jointly
          with the circuit (Delta_U, archb.memdiag; U = expm(i H), D^2 more parameters); msar and smsar16, 5 inits. tier 1
  rep     replication across datasets: msar (tier 2) and smsar16 (tier 3), the sequence 4 dataset and 4 new ones (new
          training paths and held-out path), each with: Architecture B anc2 stage 1 (3 inits), C-QSM m = 1, 2 + 2 anc
          coherent and dephased (gamma = 1), the classical switch of anc2 kernels with S = 2, 4 states, the factored IOHMM
          S = 2, 4 (5 inits each), the MS-AR family fit and the truth. This gives the accuracy-versus-memory curve
          (D = 2: m = 1, dephased m = 1, S = 2; D = 4: m = 2, dephased m = 2, S = 4) with dataset and init variability.
          Tier 4: 15 more inits (5-19) of the C-QSM m = 1, coherent and dephased, on the 5 smsar16 datasets. With 5
          inits the coherent model was ahead on 4/5 datasets, but best-of-5 varied by up to 5e-3 between runs of the
          same model, more than the coherent / dephased gap; best of 20 says whether the gap is optimization noise.
Budgets: circuits Adam (stage 1: 500 steps, lr 0.03 -> 0.003; stage 2: 300 steps, 0.003 -> 0.001) then L-BFGS up to
20,000 iterations (sequences 4-5: 2,000 / 3,000, which most fits reached), EM up to 2,000 iterations, with wall-clock caps;
every result records capped and the optimizer message. Stage 2 fits start from the best stage 1 fit of their dataset
(training likelihood) and wait for it.

-> runs/seq8/<exp>/<dataset>/<model>/<init>/{result.json, params.npz, loss.npy, lp_test.npy}; new data in runs/seq8/data/
(regenerated from the seeds). Resumable (finished fits skipped, lock files with the pid).

Usage: python scripts/run_seq8.py [--exps ...] [--tiers ...] [--datasets ...] [--models ...] [--inits ...]
                                  [--workers 32] [--max-minutes M] [--out DIR] [--list]
       python scripts/run_seq8.py --smoke     (one fit per type, 20 s caps, -> runs/seq8_smoke/)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import run_seq5 as S5  # noqa: E402
from archb import cqsm, hmm, iolin, memdiag as MD, regimes as R, switch, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEQ4 = ROOT / "runs" / "seq4"
Job = S5.Job
L, N_ANC = 3, 2
LBFGS, EM_ITERS = 20000, 2000
CAP = {"archb": 1800, "m1": 3600, "m2": 7200, "S2": 3600, "S4": 5400, "iohmm": 1800, "family": 900, "truth": 60}
S5.CFG["out"] = ROOT / "runs" / "seq8"

# name -> (kind, N, seed); 'msar' and 'smsar16' are the sequence 4 datasets (read from runs/seq4/data/)
DATASETS = {"msar": ("msar", None, 700), "smsar16": ("smsar", 16, 730)}
for _d in range(1, 5):
    DATASETS[f"msar-d{_d}"] = ("msar", None, 800 + 10 * _d)
    DATASETS[f"smsar16-d{_d}"] = ("smsar", 16, 900 + 10 * _d)
GAMMAS_M1, GAMMAS_M2 = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0), (0.0, 0.5, 1.0)
MORE_INITS = range(5, 20)  # tier 4: inits 5-19 of the C-QSM m = 1 coherent / dephased on the smsar16 datasets


# ------------------------------------------------------------------------------------------------ data and truth
def data_path(name):
    p = SEQ4 / "data" / f"{name}.npz"
    return p if p.exists() else S5.CFG["out"] / "data" / f"{name}.npz"


def prepare_data(names):
    for name in names:
        p = data_path(name)
        if not p.exists():
            kind, N, seed = DATASETS[name]
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **R.make_data(kind, N, seed))
            os.replace(tmp, p)


def load(name):
    with np.load(data_path(name)) as z:
        return {k: z[k] for k in z.files}


_REF = {}


def reference(name):
    if name not in _REF:
        kind, N, _ = DATASETS[name]
        d, lin = load(name), iolin.IOLin.from_iohmm(R.truth_T(kind, N))
        _REF[name] = {"lin": lin, "lp": lin.logprobs(d["test"]), "train_ll": lin.path_loglik(d["X"])}
    return _REF[name]


# ------------------------------------------------------------------------------------------------ job table
def _cqsm_cap(m):
    return CAP[f"m{m}"]


def jobs_all():
    J = []
    for ds in ("msar", "smsar16"):
        J += [Job(1, "deph", ds, f"cqsm-m1-anc2-g{g:g}", i, CAP["m1"]) for g in GAMMAS_M1 for i in range(5)]
        J += [Job(1, "deph", ds, f"cqsm-m2-anc2-g{g:g}", i, CAP["m2"]) for g in GAMMAS_M2 for i in range(5)]
        J += [Job(1, "basis", ds, f"cqsm-m{m}-anc2-u", i, _cqsm_cap(m)) for m in (1, 2) for i in range(5)]
    for tier, base in ((2, "msar"), (3, "smsar16")):
        for ds in [base] + [f"{base}-d{d}" for d in range(1, 5)]:
            J += [Job(tier, "rep", ds, "archb-anc2", i, CAP["archb"]) for i in range(3)]
            for m in (1, 2):
                J += [Job(tier, "rep", ds, f"cqsm-m{m}-anc2-g{g:g}", i, _cqsm_cap(m)) for g in (0.0, 1.0)
                      for i in range(5)]
            J += [Job(tier, "rep", ds, f"sw-anc2-S{S}", i, CAP[f"S{S}"]) for S in (2, 4) for i in range(5)]
            J += [Job(tier, "rep", ds, f"iohmm-fact-{S}", i, CAP["iohmm"]) for S in (2, 4) for i in range(5)]
            J += [Job(tier, "rep", ds, "family-msar", 0, CAP["family"]), Job(tier, "rep", ds, "truth", 0, CAP["truth"])]
    for ds in ["smsar16"] + [f"smsar16-d{d}" for d in range(1, 5)]:  # more inits (tier 4)
        J += [Job(4, "rep", ds, f"cqsm-m1-anc2-g{g:g}", i, CAP["m1"]) for g in (0.0, 1.0) for i in MORE_INITS]
    return J


def stage1_dir(job):
    """Architecture B anc2 fits that start the stage 2 circuits: sequence 4 for deph / basis, this run for rep."""
    base = SEQ4 if job.exp in ("deph", "basis") else S5.CFG["out"] / "rep"
    return base / job.dataset / "archb-anc2"


def deps(job):
    if job.model.startswith(("cqsm", "sw")):
        return [stage1_dir(job) / str(i) / "result.json" for i in range(3)]
    return []


def ordered(jobs):  # tier first, stage 1 before the rest, then the longest cap first
    return [j for _, j in sorted(enumerate(jobs), key=lambda t: (t[1].tier, bool(deps(t[1])), -t[1].cap, t[0]))]


# ------------------------------------------------------------------------------------------------ fits
def _adam(n):
    return 0 if S5.CFG["smoke"] else n


def _circuit(th0, loss, cap, stage1):
    if stage1:
        return train.fit_capped(loss, jnp.asarray(th0), _adam(500), LBFGS, cap)
    return train.fit_capped(loss, jnp.asarray(th0), _adam(300), LBFGS, cap, 0.003, 0.001)


def _opt_info(res):
    return {"adam_ll": -res["adam_loss"], "lbfgs_iters": res["lbfgs_iters"], "lbfgs_message": res["lbfgs_message"],
            "capped": res["capped"]}


def fit_archb(job, d):
    th0, Xj = cqsm.init_theta(job.init, 0, N_ANC, L), jnp.asarray(d["X"])
    loss = lambda t: -cqsm.loglik(t, Xj, 0, N_ANC, L)
    res = _circuit(th0, loss, job.cap, True)
    info = {"filter_dim": 1, "n_params": cqsm.n_params(0, N_ANC, L), **_opt_info(res)}
    return cqsm.io(res["theta"], 0, N_ANC, L), -res["final_loss"], info, {"theta": res["theta"]}, res["loss"]


def fit_cqsm(job, d):
    """cqsm-m<m>-anc2-g<gamma> (dephasing gamma in the computational basis) or cqsm-m<m>-anc2-u (full dephasing in
    a basis trained with the circuit)."""
    m = int(re.match(r"cqsm-m(\d)", job.model).group(1))
    D, f, Xj = 2**m, cqsm.f_vec(m), jnp.asarray(d["X"])
    prm, i1 = S5.best_result(stage1_dir(job))
    th0 = cqsm.embed(prm["theta"], job.init, m, N_ANC, L)
    nth = len(th0)
    if job.model.endswith("-u"):
        th0 = np.concatenate([th0, 0.3 * np.random.default_rng(100 + job.init).normal(size=D * D)])
        M = lambda t: MD.dephased(cqsm.transfer(t[:nth], m, N_ANC, L), D, 1.0, MD.unitary(t[nth:], D))
        g = None
    else:
        g = float(job.model.split("-g")[1])
        M = lambda t: cqsm.transfer(t, m, N_ANC, L, gamma=g)
    loss = lambda t: -iolin.loglik(M(t), f, Xj)
    res = _circuit(th0, loss, job.cap, False)
    th = res["theta"]
    info = {"D": D, "n_ancillas": N_ANC, "gamma": g, "basis": "trained" if g is None else "computational",
            "filter_dim": D * D, "n_params": len(th), "stage1_init": i1, "init_ll": float(-loss(jnp.asarray(th0))),
            "mem_angle_rms": float(np.sqrt(np.mean(th[:nth][cqsm.mem_mask(m, N_ANC, L)] ** 2))), **_opt_info(res)}
    lin = iolin.IOLin(np.asarray(M(jnp.asarray(th))), np.asarray(f))
    return lin, -res["final_loss"], info, {"theta": th}, res["loss"]


def fit_switch(job, d):
    S = int(job.model.split("-S")[1])
    prm, i1 = S5.best_result(stage1_dir(job))
    th0, Xj = switch.init_switch(prm["theta"], S, job.init, "anc2"), jnp.asarray(d["X"])
    loss = lambda t: -switch.loglik(t, Xj, S, "anc2")
    res = _circuit(th0, loss, job.cap, False)
    info = {"S": S, "kernel": "anc2", "filter_dim": S, "n_params": switch.n_params(S, "anc2"), "stage1_init": i1,
            "init_ll": float(-loss(jnp.asarray(th0))), **_opt_info(res)}
    return switch.io(res["theta"], S, "anc2"), -res["final_loss"], info, {"theta": res["theta"]}, res["loss"]


def fit_iohmm(job, d):
    S = int(job.model.split("-")[2])
    T, curve = R.iohmm_em(d["X"], S, "fact", seed=job.init, max_iter=20 if S5.CFG["smoke"] else EM_ITERS)
    lin = iolin.IOLin.from_iohmm(T)
    info = {"S": S, "filter_dim": S, "n_params": R.n_params_iohmm("fact", S), "em_iters": len(curve),
            "capped": len(curve) >= EM_ITERS}
    return lin, lin.path_loglik(d["X"]), info, {"T": T}, -np.asarray(curve)


def fit_family(job, d):
    T, p, ll, info = R.fit_family("msar", d["X"])
    return iolin.IOLin.from_iohmm(T), ll, {"filter_dim": 2, **info, "params": p}, {"T": T}, np.zeros(0)


def fit_truth(job, d):
    lin = reference(job.dataset)["lin"]
    return lin, lin.path_loglik(d["X"]), {"filter_dim": lin.d}, {}, np.zeros(0)


def heldout_logprobs(r, d):
    """Per-step held-out log-probabilities of the stored fit in d (result dict r), recomputed from params.npz by the
    code path of fit(); scripts/seq8_summary.py calls it when lp_test.npy (written by the fit, not tracked) is missing."""
    model = r["model"]
    prm = dict(np.load(d / "params.npz")) if (d / "params.npz").exists() else {}
    kind = model.split("-")[0]
    if kind == "archb":
        lin = cqsm.io(prm["theta"], 0, N_ANC, L)
    elif kind == "cqsm":
        m = int(re.match(r"cqsm-m(\d)", model).group(1))
        D, th = 2**m, jnp.asarray(prm["theta"])
        if model.endswith("-u"):
            nth = len(th) - D * D
            M = MD.dephased(cqsm.transfer(th[:nth], m, N_ANC, L), D, 1.0, MD.unitary(th[nth:], D))
        else:
            M = cqsm.transfer(th, m, N_ANC, L, gamma=float(model.split("-g")[1]))
        lin = iolin.IOLin(np.asarray(M), np.asarray(cqsm.f_vec(m)))
    elif kind == "sw":
        lin = switch.io(prm["theta"], int(model.split("-S")[1]), "anc2")
    elif kind in ("iohmm", "family"):
        lin = iolin.IOLin.from_iohmm(prm["T"])
    else:
        lin = reference(r["dataset"])["lin"]
    prepare_data([r["dataset"]])
    return np.asarray(lin.logprobs(load(r["dataset"])["test"]))


def fit(job):
    """-> (result dict, params, held-out log-probabilities, loss curve); called by run_seq5.run_job."""
    d = load(job.dataset)
    f = {"archb": fit_archb, "cqsm": fit_cqsm, "sw": fit_switch, "iohmm": fit_iohmm, "family": fit_family,
         "truth": fit_truth}[job.model.split("-")[0]]
    lin, train_ll, info, params, loss = f(job, d)
    ref = reference(job.dataset)
    lp = lin.logprobs(d["test"])
    rate, se = hmm.seg_kl_rate(ref["lp"], lp)
    kind, N, seed = DATASETS[job.dataset]
    res = {"exp": job.exp, "dataset": job.dataset, "kind": kind, "N": N, "seed": seed, "model": job.model,
           "init": job.init, "cap_seconds": job.cap, **info, "train_ll": train_ll, "truth_train_ll": ref["train_ll"],
           "truth_test_ll": float(np.mean(ref["lp"])), "heldout_kl_rate": rate, "heldout_se": se}
    return res, params, lp, loss


S5.fit = fit  # run_seq5.run_job calls fit(job); also set in the spawned workers, which re-import this file


# ------------------------------------------------------------------------------------------------ main
def worker_init(out, smoke):
    S5.configure(out, smoke)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None)
    ap.add_argument("--tiers", nargs="*", type=int, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="model name prefixes")
    ap.add_argument("--inits", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output directory (default runs/seq8)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 6)))
    ap.add_argument("--max-minutes", type=float, default=1e9, help="start no new fit after this many minutes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one fit per type, caps <= 20 s, no Adam, -> runs/seq8_smoke/")
    a = ap.parse_args()
    if a.smoke or a.out:
        S5.configure(Path(a.out) if a.out else ROOT / "runs" / "seq8_smoke", smoke=a.smoke)
    jobs = [j for j in ordered(jobs_all()) if (a.exps is None or j.exp in a.exps)
            and (a.tiers is None or j.tier in a.tiers) and (a.datasets is None or j.dataset in a.datasets)
            and (a.models is None or any(j.model.startswith(p) for p in a.models))
            and (a.inits is None or str(j.init) in a.inits)]
    if a.smoke:  # one job per (experiment, model type), rep on msar only, plus the stage 1 fits they wait for
        seen, sel = set(), []
        for j in jobs:
            key = (j.exp, re.sub(r"\d+(\.\d+)?", "#", j.model))
            if key not in seen and (j.exp != "rep" or j.dataset == "msar"):
                seen.add(key)
                sel.append(j)
        need = {(j.exp, j.dataset) for j in sel if deps(j) and j.exp == "rep"}
        jobs = sel + [j for j in jobs if j.model == "archb-anc2" and (j.exp, j.dataset) in need and j not in sel]
    if a.list:
        S5.list_jobs(jobs)
        return
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes
    pending = [j for j in jobs if not S5.result_dir(j).joinpath("result.json").exists()]
    print(f"{datetime.now():%H:%M:%S} {len(jobs)} fits, {len(pending)} to do, {a.workers} workers, out {S5.CFG['out']}",
          flush=True)
    prepare_data(sorted({j.dataset for j in pending}))
    running, n_done, n_err, failed = {}, 0, 0, set()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=worker_init, initargs=(S5.CFG["out"], S5.CFG["smoke"]),
                  maxtasksperchild=8) as pool:
        while pending or running:
            if time.time() > deadline and not running:
                print(f"time budget reached: {len(pending)} fits not started", flush=True)
                break
            for j in [j for j in pending if any(str(p.parent) in failed for p in deps(j))]:
                pending.remove(j)  # its stage 1 failed
                n_err += 1
                print(f"{datetime.now():%H:%M:%S} [skipped] {j}: a stage 1 fit failed", flush=True)
            while len(running) < a.workers and time.time() <= deadline:
                ready = [j for j in pending if all(p.exists() for p in deps(j))]
                if not ready:
                    break
                j = ready[0]
                pending.remove(j)
                running[j] = (pool.apply_async(S5.run_job, (j,)), time.time())
            for j in [j for j, (r, _) in running.items() if r.ready()]:
                r, _ = running.pop(j)
                try:
                    _, st, wt, msg = r.get()
                except Exception as e:  # noqa: BLE001
                    st, wt, msg = "crash", 0.0, repr(e)
                n_done += st == "ran"
                if st in ("error", "crash"):
                    n_err += 1
                    failed.add(str(S5.result_dir(j)))
                print(f"{datetime.now():%H:%M:%S} [{st}] {j.exp} {j.dataset} {j.model} {j.init}: {msg} [{wt:.0f} s]",
                      flush=True)
            if pending and not running and not any(all(p.exists() for p in deps(j)) for j in pending):
                print(f"{len(pending)} fits wait for stage 1 fits that are not in this selection", flush=True)
                break
            for j in [j for j, (_, ts) in running.items() if time.time() - ts > 2 * j.cap + 900]:
                running.pop(j)
                print(f"{datetime.now():%H:%M:%S} [lost] {j}: no answer after {2 * j.cap + 900} s", flush=True)
            time.sleep(1)
    print(f"{datetime.now():%H:%M:%S} {n_done} fits ran, {n_err} errors, {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
