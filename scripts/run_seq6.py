"""Sequence 6: Architecture B (QCBM) kernels with a classical regime memory, against the same switch with compact classical
kernels at equal parameter count (archb.switch, archb.ckernels).

Parts and models (ids = inits / seeds):
  synth   msar, smsar16 (sequence 4 data; protocol of sequence 5 A1, whose QCBM-kernel switches are reused from runs/seq5/a1)
          k-<kind>         one classical kernel (Markov chain on the bins), inits 0..2: Adam 500 (lr 0.03 -> 0.003) then
                           L-BFGS 2,000 (sequence 4 stage 1)
          sw-<kind>-S<S>   classical switch, S = 2 (seeds 0..4), S = 4 on smsar16 for mlp6 / mix36 / rbm17 (seeds 0..2), from the
                           stage 1 kernel with the best training likelihood: Adam 300 (lr 0.003 -> 0.001) then L-BFGS 3,000
                           (sequence 4 stage 2, as A1)
          metric: held-out KL rate on the sequence 4 held-out path (200,000 steps) with the 20-segment SE
  real    eurusd / nasdaq x grids u, q, v (step 13 data and protocol: scripts/run_seq4_real.py)
          sw-anc2-S2, sw-anc2-S4, sw-B0-S2   QCBM-kernel switch, seeds 0..4, from the step 13 Architecture B stage 1 fit
                           with the best training likelihood
          k-<kind>         classical kernel, inits 0..2, as the step 13 Architecture B stage 1 (Adam 500 lr 0.01 -> 0.001,
                           L-BFGS 4,000)
          sw-<kind>-S<S>   classical switch, S = 2 (all kinds), S = 4 (mlp6, mix36, rbm17), seeds 0..4, as the step 13 C-QSM
                           stage 2
          every fit with early stopping on the validation log-likelihood; the init with the best validation score is the fit
Classical kinds (archb.ckernels, free parameters): matched to the anc2 kernel (215): mlp6 229, mlp5 196, mix36 214,
rbm17 217; matched to the B0 kernel (119): mlp3 130, mix20 118, rbm8 118. Inputs of the MLP and mixture kernels: the value
of the current bin (synthetic: bin centre; real data: mean training return in the bin, in the grid's units); the
conditional RBM sees the Gray bits of the current and next bins, as the circuits do.

Stages:
  fit    phase 1 (classical stage 1 and the real-data QCBM switches), then phase 2 (classical switches), in a process
         pool; resumable -> runs/seq6/<part>/<dataset>/<model>/<init>/{result.json, params.npz[, lp_test.npy]}
  eval   synth: runs/seq6/synth/summary.json; real: per dataset, the selected init of every model scored as in step 13
         (test log-score, block-bootstrap SE, paired differences to the step 13 references, 1-day and 10-day VaR / ES,
         memory readout) -> runs/seq6/real/<ds>/eval.json, eval_arrays.npz
  all    fit, eval, then scripts/seq6_summary.py

Usage: python scripts/run_seq6.py [fit|eval|all|list] [--parts synth real] [--workers W] [--smoke]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import namedtuple  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from archb import ckernels, hmm, iolin, realdata as D, regimes as R, switch, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = {"out": ROOT / "runs" / "seq6", "smoke": False}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S5, RR = _load("run_seq5"), _load("run_seq4_real")
SYNTH = ["msar", "smsar16"]
REAL = RR.DATASETS + RR.V_DATASETS
T1, T2 = ["mlp6", "mlp5", "mix36", "rbm17"], ["mlp3", "mix20", "rbm8"]
S4_KINDS = ("mlp6", "mix36", "rbm17")
QCBM = [("anc2", 2), ("anc2", 4), ("B0", 2)]
GRID_SYN = ckernels.make_grid(R.Z, R.E)
ALPHAS, HORIZON, N_MC, BLOCK = RR.ALPHAS, RR.HORIZON, RR.N_MC, RR.BLOCK
REFS = ["garch-t-cont", "garch-t-bin", "garch-gjr-cont", "msar-cont", "msar-bin", "markov-0"]
Job = namedtuple("Job", "part stage dataset model init cap")


# ------------------------------------------------------------------------------------------------ job table
def jobs_all():
    J = []
    for ds in SYNTH:
        J += [Job("synth", 1, ds, f"k-{k}", i, 900) for k in T1 + T2 for i in range(3)]
        J += [Job("synth", 2, ds, f"sw-{k}-S2", i, 1800) for k in T1 + T2 for i in range(5)]
        if ds == "smsar16":
            J += [Job("synth", 2, ds, f"sw-{k}-S4", i, 1800) for k in S4_KINDS for i in range(3)]
    for ds in REAL:
        J += [Job("real", 1, ds, f"sw-{k}-S{S}", i, None) for k, S in QCBM for i in range(5)]
        J += [Job("real", 1, ds, f"k-{k}", i, None) for k in T1 + T2 for i in range(3)]
        J += [Job("real", 2, ds, f"sw-{k}-S2", i, None) for k in T1 + T2 for i in range(5)]
        J += [Job("real", 2, ds, f"sw-{k}-S4", i, None) for k in S4_KINDS for i in range(5)]
    return J


def result_dir(job):
    return CFG["out"] / job.part / job.dataset / job.model / str(job.init)


def _iters(n):
    return max(2, n // 100) if CFG["smoke"] else n


def _parse(model):
    """('k', kind, 1) or ('sw', kind, S)."""
    p = model.split("-")
    return (p[0], p[1], 1) if p[0] == "k" else (p[0], p[1], int(p[2][1:]))


def best_stage1(job, key):
    """(theta, init) of the stage 1 classical kernel with the best training objective."""
    best = None
    for p in sorted((CFG["out"] / job.part / job.dataset / f"k-{_parse(job.model)[1]}").glob("*/result.json")):
        r = json.loads(p.read_text())
        if best is None or r[key] > best[0]:
            best = (r[key], p.parent)
    with np.load(best[1] / "params.npz") as z:
        return z["theta"], best[1].name


def kernel_M(kind, grid):
    return lambda t: ckernels.kernel(t, kind, grid)[:, :, None, None]


# ------------------------------------------------------------------------------------------------ synthetic fits
def fit_synth(job):
    d = S5.load_seq4(job.dataset)
    X = jnp.asarray(d["X"])
    typ, kind, S = _parse(job.model)
    if typ == "k":
        M = kernel_M(kind, GRID_SYN)
        th0 = ckernels.init_theta(kind, job.init, GRID_SYN, np.asarray(d["X"]).ravel())
        loss = lambda t: -iolin.loglik(M(t), jnp.ones(1), X)
        res = train.fit_capped(loss, jnp.asarray(th0), _iters(500), _iters(2000), job.cap, 0.03, 0.003)
        lin, extra = iolin.IOLin(np.asarray(M(jnp.asarray(res["theta"]))), np.ones(1)), {}
    else:
        k0, i1 = best_stage1(job, "train_ll")
        th0 = switch.init_switch(k0, S, job.init, kind)
        loss = lambda t: -switch.loglik(t, X, S, kind, grid=GRID_SYN)
        res = train.fit_capped(loss, jnp.asarray(th0), _iters(300), _iters(3000), job.cap, 0.003, 0.001)
        lin, extra = switch.io(res["theta"], S, kind, grid=GRID_SYN), {"stage1_init": i1}
    info = {"S": S, "kernel": kind, "filter_dim": S, "n_params": switch.n_params(S, kind) if typ == "sw" else
            ckernels.n_params(kind), "n_kernel": ckernels.n_params(kind), "init_ll": float(-loss(jnp.asarray(th0))),
            "adam_ll": -res["adam_loss"], "lbfgs_iters": res["lbfgs_iters"], "lbfgs_message": res["lbfgs_message"],
            "capped": res["capped"], **extra}
    lp, ev = S5.eval_seq4(job.dataset, lin, -res["final_loss"], d)
    return {**info, **ev}, {"theta": res["theta"]}, lp


# ------------------------------------------------------------------------------------------------ real-data fits
def fit_real(job):
    ds = RR.dataset(job.dataset)
    grid = ckernels.make_grid(ds["values"], ds["edges"])
    X = jnp.asarray(ds["b"][:ds["n_train"]])[None]
    score = RR.val_scorer(ds)
    typ, kind, S = _parse(job.model)
    if typ == "k":
        M = kernel_M(kind, grid)
        th0 = ckernels.init_theta(kind, job.init, grid, ds["b"][:ds["n_train"]])
        th, info = RR.fit_es(lambda t: -iolin.loglik(M(t), jnp.ones(1), X), lambda t: score(M(jnp.asarray(t)),
                             np.ones(1)), th0, _iters(500), _iters(4000), 0.01, 0.001)
        lin = iolin.IOLin(np.asarray(M(jnp.asarray(th))), np.ones(1))
        info.update(n_params=ckernels.n_params(kind))
    else:
        if ckernels.is_classical(kind):
            k0, i1 = best_stage1(job, "fit_ll")
        else:
            k0, i1 = RR.best_stage1(job.dataset, f"archb-{kind}")
        th0 = switch.init_switch(k0, S, job.init, kind)
        g = grid if ckernels.is_classical(kind) else None
        th, info = RR.fit_es(lambda t: -switch.loglik(t, X, S, kind, grid=g),
                             lambda t: score(switch.switch_M(jnp.asarray(t), S, kind, grid=g), np.ones(S)),
                             th0, _iters(300), _iters(3000), 0.003, 0.001)
        lin = switch.io(th, S, kind, grid=g)
        info.update(n_params=switch.n_params(S, kind), stage1_init=i1)
    info.update(S=S, kernel=kind, filter_dim=S, n_kernel=switch.n_kernel(kind), memory_dim=S)
    P = D.predictive(lin, ds["b"])
    lp = D.log_scores(P, ds["b"])
    a, v = ds["n_train"], ds["n_val"]
    fit_ll = float(lin.path_loglik(np.asarray(X)))
    return {**info, "fit_ll": fit_ll, "train_ll": float(lp[2:a].mean()), "val_ll": float(lp[a:a + v].mean()),
            "test_ll": float(lp[a + v:].mean())}, {"theta": th}, None


# ------------------------------------------------------------------------------------------------ one job
def run_job(job):
    d = result_dir(job)
    if (d / "result.json").exists():
        return job, "done", 0.0, ""
    d.mkdir(parents=True, exist_ok=True)
    t0, status, msg = time.time(), "error", ""
    log = CFG["out"] / "logs" / f"{job.part}__{job.dataset}__{job.model}__{job.init}.log"
    with S5._log_to(log):
        try:
            res, params, lp = (fit_synth if job.part == "synth" else fit_real)(job)
            res.update(part=job.part, dataset=job.dataset, model=job.model, init=job.init, stage=job.stage,
                       wall_time=time.time() - t0)
            np.savez(d / "params.npz", **params)
            if lp is not None:
                np.save(d / "lp_test.npy", np.asarray(lp, dtype=np.float32))
            tmp = d / "result.json.tmp"
            tmp.write_text(json.dumps(res, indent=1, default=float) + "\n")
            os.replace(tmp, d / "result.json")
            status = "ran"
            msg = (f"held-out KL {res['heldout_kl_rate']:.3e} ± {res['heldout_se']:.1e}" if job.part == "synth" else
                   f"train {res['train_ll']:.4f} val {res['val_ll']:.4f} test {res['test_ll']:.4f}")
            print(msg, flush=True)
        except Exception:
            traceback.print_exc()
            msg = traceback.format_exc().strip().splitlines()[-1]
    return job, status, time.time() - t0, msg


def _init_worker(out, smoke):
    CFG["out"], CFG["smoke"] = Path(out), smoke


def fit_stage(parts, workers):
    jobs = [j for j in jobs_all() if j.part in parts]
    if CFG["smoke"]:
        jobs = [j for j in jobs if j.dataset in ("msar", "eurusd-u", "nasdaq-v") and j.init in (0, 1)
                and (j.stage == 1 or j.init == 0)]
    for phase in (1, 2):
        todo = [j for j in jobs if j.stage == phase and not (result_dir(j) / "result.json").exists()]
        todo.sort(key=lambda j: (-(j.model.startswith("sw-anc2-S4")), -(j.model.startswith("sw-anc2")), j.part))
        print(f"phase {phase}: {len(todo)} fits", flush=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers, _init_worker, (str(CFG["out"]), CFG["smoke"])) as pool:
            for k, (job, st, sec, msg) in enumerate(pool.imap_unordered(run_job, todo), 1):
                print(f"[{k}/{len(todo)} {(time.time() - t0) / 60:.0f} min] {job.part} {job.dataset} {job.model} "
                      f"{job.init}: {st} {sec:.0f} s {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ eval
def _results(part, ds):
    allr = {}
    for p in sorted((CFG["out"] / part / ds).glob("*/*/result.json")):
        r = json.loads(p.read_text())
        allr.setdefault(r["model"], []).append(r)
    return allr


def seg_diff(lp_a, lp_b, n_seg=20):
    """Mean of lp_a - lp_b per step with the n_seg-segment SE (> 0: a better)."""
    dl = np.asarray(lp_a, float) - np.asarray(lp_b, float)
    n = len(dl) // n_seg * n_seg
    seg = dl[:n].reshape(n_seg, -1).mean(1)
    return float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(n_seg))


def lp_test(src, ds, mo):
    """Per-step held-out log-probabilities of the synthetic fit in src: lp_test.npy (written by the fit, not tracked),
    else recomputed from params.npz by the code path of the fit."""
    p = src / "lp_test.npy"
    if not p.exists():
        if src.parents[2].name == "a1":  # QCBM-kernel switches of sequence 5
            lp = S5.heldout_logprobs(json.loads((src / "result.json").read_text()), src)
        else:
            typ, kind, S = _parse(mo)
            with np.load(src / "params.npz") as z:
                th = z["theta"]
            lin = (iolin.IOLin(np.asarray(kernel_M(kind, GRID_SYN)(jnp.asarray(th))), np.ones(1)) if typ == "k"
                   else switch.io(th, S, kind, grid=GRID_SYN))
            lp = lin.logprobs(S5.load_seq4(ds)["test"])
        np.save(p, np.asarray(lp, dtype=np.float32))
    return np.load(p)


def eval_synth():
    out = {}
    a1 = ROOT / "runs" / "seq5" / "a1"
    for ds in SYNTH:
        rows, lps = {}, {}
        allr = _results("synth", ds)
        for mo in sorted(p.name for p in (a1 / ds).iterdir()):  # QCBM-kernel switches of sequence 5
            allr[mo] = [json.loads(p.read_text()) for p in sorted((a1 / ds / mo).glob("*/result.json"))]
        for mo, rs in allr.items():
            r = max(rs, key=lambda x: x["train_ll"])
            src = (a1 if mo in ("sw-anc2-S2", "sw-anc2-S4", "sw-B0-S2") else CFG["out"] / "synth") / ds / mo / str(r["init"])
            lps[mo] = lp_test(src, ds, mo)
            kl = [x["heldout_kl_rate"] for x in rs]
            rows[mo] = {"init": r["init"], "n_inits": len(rs), "train_ll": r["train_ll"], "truth_train_ll":
                        r["truth_train_ll"], "heldout_kl_rate": r["heldout_kl_rate"], "heldout_se": r["heldout_se"],
                        "kl_inits_min": min(kl), "kl_inits_max": max(kl), "n_params": r.get("n_params"),
                        "n_kernel": r.get("n_kernel", switch.n_kernel(mo.split("-")[1]) if mo.startswith("sw") else None),
                        "S": r.get("S"), "kernel": r.get("kernel"), "it_limit": r.get("lbfgs_iters", 0) >= 3000,
                        "capped": r.get("capped")}
        for mo in rows:  # paired differences to the QCBM switches (> 0: the QCBM switch is better)
            for ref in ("sw-anc2-S2", "sw-B0-S2", "sw-anc2-S4"):
                if ref in lps and mo != ref:
                    rows[mo][f"vs_{ref}"] = seg_diff(lps[ref], lps[mo])
        out[ds] = rows
    (CFG["out"] / "synth").mkdir(parents=True, exist_ok=True)
    (CFG["out"] / "synth" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


def eval_real(name):
    ds = RR.dataset(name)
    grid = ckernels.make_grid(ds["values"], ds["edges"])
    allr = _results("real", name)
    a, v, n = ds["n_train"], ds["n_val"], len(ds["b"])
    s = a + v
    b, z = ds["b"], ds["z"]
    V = D.row_values(ds)
    vb = V[np.arange(n), b]
    starts = np.arange(s - 1, n - HORIZON, HORIZON)
    s_real = np.array([vb[t + 1:t + 1 + HORIZON].sum() for t in starts])
    z_real = np.array([z[t + 1:t + 1 + HORIZON].sum() for t in starts])
    ref_ev = json.loads((RR.OUT / name / "eval.json").read_text())
    ref_arr = np.load(RR.OUT / name / "eval_arrays.npz")
    cq = ref_ev["cqsm_chosen"]
    refs = {mo: ref_arr[f"lp/{mo}"] for mo in REFS + [cq, "cqsm-mix"]}
    sel = {mo: max(rs, key=lambda x: x["val_ll"]) for mo, rs in allr.items()}
    fam = {"qcbm": [m for m in sel if m.startswith("sw-") and not ckernels.is_classical(m.split("-")[1])],
           "classical": [m for m in sel if m.startswith("sw-") and ckernels.is_classical(m.split("-")[1])]}
    chosen = {f: max(ms, key=lambda m: sel[m]["val_ll"]) for f, ms in fam.items() if ms}
    rv = np.log(D.realized_vol(ds["r"], 21))
    ev, arrays = {}, {}
    for mo, r in sel.items():
        t0 = time.time()
        _, kind, S = _parse(mo)
        with np.load(CFG["out"] / "real" / name / mo / str(r["init"]) / "params.npz") as zf:
            th = zf["theta"]
        if mo.startswith("k-"):
            lin = iolin.IOLin(np.asarray(ckernels.kernel(th, kind, grid))[:, :, None, None], np.ones(1))
        else:
            lin = switch.io(th, S, kind, grid=grid if ckernels.is_classical(kind) else None)
        M = RR.Lin(lin)
        P = M.predictive(ds)
        lp = D.log_scores(P, b)
        arrays[f"lp/{mo}"] = lp
        tl = [x["test_ll"] for x in allr[mo]]
        e = {"init": r["init"], "n_inits": len(tl), "test_ll": float(lp[s:].mean()),
             "test_se": float(D.block_bootstrap(lp[s:], BLOCK)), "val_ll": float(lp[a:s].mean()),
             "train_ll": float(lp[2:a].mean()), "test_ll_inits_min": float(min(tl)), "test_ll_inits_max": float(max(tl)),
             "n_params": r["n_params"], "n_kernel": r.get("n_kernel"), "S": S, "kernel": kind,
             "stop_iter": r.get("stop_iter"), "lbfgs_iters": r.get("lbfgs_iters")}
        for ref, lr in refs.items():
            dl = lp[s:] - lr[s:]
            e[f"diff/{ref}"] = [float(dl.mean()), float(D.block_bootstrap(dl, BLOCK))]
        if mo.startswith("sw-"):
            e["var1"] = {f"{al}": D.backtest(P[s:], b[s:], V[s:], al, loss=-z[s:]) for al in ALPHAS}
            Ssum = np.stack([M.mc_sums(ds, t, HORIZON, N_MC, np.random.default_rng(10_000 + k))
                             for k, t in enumerate(starts)])
            e["var10"] = {f"{al}": D.backtest_samples(Ssum, s_real, al, loss=-z_real) for al in ALPHAS}
            if S > 1:
                feats = M.states(ds)
                _, c_rv, c_rv_tr = RR.readout(feats, rv, a, s)
                e["readout_rv"] = {"test": c_rv, "train": c_rv_tr}
        e["eval_time"] = time.time() - t0
        ev[mo] = e
        print(f"{name} {mo}: test {e['test_ll']:.4f} ± {e['test_se']:.4f} vs GARCH-t {e['diff/garch-t-cont'][0]:+.4f} "
              f"[{e['eval_time']:.0f} s]", flush=True)
    pairs = {}
    for f1, f2 in (("qcbm", "classical"),):
        if f1 in chosen and f2 in chosen:
            dl = arrays[f"lp/{chosen[f1]}"][s:] - arrays[f"lp/{chosen[f2]}"][s:]
            pairs[f"{f1}-{f2}"] = [float(dl.mean()), float(D.block_bootstrap(dl, BLOCK))]
    for q, cl in (("sw-anc2-S2", ["sw-mlp6-S2", "sw-mlp5-S2", "sw-mix36-S2", "sw-rbm17-S2"]),
                  ("sw-B0-S2", ["sw-mlp3-S2", "sw-mix20-S2", "sw-rbm8-S2"]),
                  ("sw-anc2-S4", ["sw-mlp6-S4", "sw-mix36-S4", "sw-rbm17-S4"])):
        for c in cl:
            if f"lp/{q}" in arrays and f"lp/{c}" in arrays:
                dl = arrays[f"lp/{q}"][s:] - arrays[f"lp/{c}"][s:]
                pairs[f"{q}|{c}"] = [float(dl.mean()), float(D.block_bootstrap(dl, BLOCK))]
    out = {"dataset": name, "n_test": n - s, "chosen": chosen, "pairs": pairs, "models": ev, "cqsm_chosen": cq,
           "refs": {mo: {"test_ll": float(refs[mo][s:].mean()), "var1": ref_ev["models"].get(mo, {}).get("var1"),
                         "var10": ref_ev["models"].get(mo, {}).get("var10")} for mo in refs}}
    (CFG["out"] / "real" / name / "eval.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    np.savez_compressed(CFG["out"] / "real" / name / "eval_arrays.npz", **arrays)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="fit", choices=["fit", "eval", "all", "list"])
    ap.add_argument("--parts", nargs="*", default=["synth", "real"])
    ap.add_argument("--datasets", nargs="*", default=None, help="real datasets to evaluate (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="few iterations, output to runs/seq6_smoke")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    CFG["smoke"] = a.smoke
    CFG["out"] = Path(a.out) if a.out else (ROOT / "runs" / ("seq6_smoke" if a.smoke else "seq6"))
    if a.stage == "list":
        J = [j for j in jobs_all() if j.part in a.parts]
        done = sum((result_dir(j) / "result.json").exists() for j in J)
        print(f"{len(J)} jobs, {done} done")
        return
    if a.stage in ("fit", "all"):
        fit_stage(a.parts, a.workers)
    if a.stage in ("eval", "all"):
        if "synth" in a.parts:
            eval_synth()
        if "real" in a.parts:
            names = a.datasets or REAL
            me = [sys.executable, __file__, "eval", "--parts", "real", "--out", str(CFG["out"])]
            if a.stage == "eval" and len(names) == 1:
                eval_real(names[0])
            else:
                procs = [subprocess.Popen(me + ["--datasets", nm]) for nm in names]
                if any(p.wait() for p in procs):
                    sys.exit("an eval worker failed")
    if a.stage == "all" and not a.smoke:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "seq6_summary.py")], check=True)


if __name__ == "__main__":
    main()
