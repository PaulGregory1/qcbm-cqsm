"""Sequence 9: several correlated variables on 8 bits (archb.multivar), circuits against classical models.

Datasets
  synth  msvar (regime-dependent correlation 0.2 / 0.8), msvarc (control: correlation 0.5 in both regimes); 2 x 4 bits,
         2,000 paths x 50 steps for training, a held-out path of 100,000 steps with its regimes
  real   pairs of daily FRED returns on 2 x 4 bits, grids q (quantile) and v (EWMA-normalized): eurgbp, eurchf, eurjpy
         (q, v), eurndq (q: EUR/USD with the NASDAQ, weak correlation); fx4-q (EUR, GBP, CHF, JPY on 4 x 2 bits, the
         Coyle et al. setting)
Models (ids; inits)
  stage 1  k-qa, k-q0 (circuit kernels, cross gates; k-qn without them on synth), k-rbm<H>, k-mix<K> (sizes matched to
           qa and q0; no mixture for fx4) 3 inits; iohmm-fact-S2, hmm-S4 (EM) 3 inits; markov1, iid; msvar (family, 4
           starts; d = 2)
  stage 2  sw-<kind>-S2 for qa and the classical kinds matched to qa, 3 seeds, from the stage 1 kernel with the best
           training likelihood; cq1 (3 seeds), cq2 (2 seeds on msvar, 1 on msvarc and grid q) from the best k-qa (archb.cqsm.embed)
Circuit kernels start from the copy init on synth, near the i.i.d. law on real data (sequence 4).
Protocol: synth as sequence 6 (Adam then L-BFGS on the training likelihood, init chosen by training likelihood);
real as sequence 6 (early stopping on the validation log-score, checked every 25 L-BFGS iterations, init chosen on
validation). Every fit has a wall-clock cap (CAPS); a capped fit keeps its best point.
Outputs: runs/seq9/<part>/<dataset>/<model>/<init>/{result.json, params.npz}; eval -> runs/seq9/synth/summary.json,
runs/seq9/real/<dataset>/eval.json; worker logs in runs/seq9/logs/.

Usage: python scripts/run_seq9.py [fit|eval|all|list|bench] [--parts synth real] [--models PAT ..] [--skip PAT ..]
       [--workers W] [--smoke]
  --models / --skip (fnmatch patterns on model ids) split the fits between processes: cq2 needs a GPU (61 s per
  gradient on one CPU core, mostly the 4,096 x 4,096 stationary solve).
  bench times one gradient of the --models given (default k-qa, cq1, cq2) on the current JAX device.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import fnmatch  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import namedtuple  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from archb import ceiling, iolin, train  # noqa: E402
from archb import multivar as MV  # noqa: E402
from archb import realdata as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = {"out": ROOT / "runs" / "seq9", "smoke": False}
SYNTH = ["msvar", "msvarc"]
REAL = ["eurgbp-q", "eurgbp-v", "eurchf-q", "eurchf-v", "eurjpy-q", "eurjpy-v", "eurndq-q", "fx4-q"]
DATA_SEED = {"msvar": 20261102, "msvarc": 20261103}
CAPS = {"k": 1800, "em": 1800, "msvar": 3600, "sw": 10800, "cq1": 10800, "cq2": 9000}
CHECK, BLOCK, N_STAT, ALPHAS = 25, 50, 20000, (0.01, 0.05)
Job = namedtuple("Job", "part stage dataset model init cap")
_CACHE = {}


class _Deadline(Exception):
    pass


# ------------------------------------------------------------------------------------------------ data
def synth_data(name):
    if name not in _CACHE:
        p = CFG["out"] / "data" / f"{name}.npz"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **MV.make_data(name, DATA_SEED[name]))
        with np.load(p) as z:
            _CACHE[name] = {k: z[k] for k in z.files}
    return _CACHE[name]


def real_data(name):
    if name not in _CACHE:
        _CACHE[name] = MV.real_dataset(name)
    return _CACHE[name]


def spec_of(part, name):
    return MV.uniform_spec() if part == "synth" else real_data(name)["spec"]


def matched(spec):
    """Classical kernels sized to each circuit kernel: {circuit: [rbm<H>, mix<K>]} (no mixture for d > 2)."""
    out = {}
    for circ in ("qa", "q0"):
        nc = MV.n_kernel(circ, spec)
        H = min(range(1, 80), key=lambda h: abs(MV.n_kernel(f"rbm{h}", spec) - nc))
        out[circ] = [f"rbm{H}"] + ([f"mix{min(range(2, 80), key=lambda K: abs(15 * K - 3 - nc))}"]
                                   if spec["d"] == 2 else [])
    return out


# ------------------------------------------------------------------------------------------------ job table
def jobs_all():
    J = []
    for part, names in (("synth", SYNTH), ("real", REAL)):
        for ds in names:
            spec = spec_of(part, ds)
            mt = matched(spec)
            ks = ["qa", "q0"] + (["qn"] if part == "synth" else []) + mt["qa"] + mt["q0"]
            J += [Job(part, 1, ds, f"k-{k}", i, CAPS["k"]) for k in ks for i in range(3)]
            J += [Job(part, 1, ds, m, i, CAPS["em"]) for m in ("iohmm-fact-S2", "hmm-S4") for i in range(3)]
            J += [Job(part, 1, ds, m, 0, CAPS["em"]) for m in ("markov1", "iid")]
            if spec["d"] == 2:
                J.append(Job(part, 1, ds, "msvar", 0, CAPS["msvar"]))
            J += [Job(part, 2, ds, f"sw-{k}-S2", i, CAPS["sw"]) for k in ["qa"] + mt["qa"] for i in range(3)]
            J += [Job(part, 2, ds, "cq1", i, CAPS["cq1"]) for i in range(3)]
            if part == "synth" or ds.endswith("-q"):  # GPU only; 2 seeds on msvar, 1 elsewhere
                J += [Job(part, 2, ds, "cq2", i, CAPS["cq2"]) for i in range(2 if ds == "msvar" else 1)]
    return J


def result_dir(job):
    return CFG["out"] / job.part / job.dataset / job.model / str(job.init)


def _iters(n):
    return max(2, n // 100) if CFG["smoke"] else n


def _cap(c):
    return 120 if CFG["smoke"] else c


# ------------------------------------------------------------------------------------------------ fitting
def val_scorer(ds):
    """Mean validation log-score of (M, f), filtering from the start of the series."""
    a, v = ds["n_train"], ds["n_val"]
    xs = jnp.asarray(ds["b"][:a + v])

    def score(M, f):
        M, f = jnp.asarray(M), jnp.asarray(f)
        return float(jnp.mean(iolin.logprobs(M, f, iolin.stationary(M, f), xs)[a:]))

    return score


def fit_es_capped(loss, val, th0, adam_steps, lbfgs_iters, lr0, lr1, cap):
    """Adam, then L-BFGS-B keeping the iterate with the best validation score (after Adam, every CHECK iterations and at
    the end), with a wall-clock cap on the L-BFGS part."""
    t0 = time.time()
    res = ceiling.adam(None, jnp.asarray(th0), adam_steps, None, loss, lr0, lr1)
    vg = jax.jit(jax.value_and_grad(loss))
    best = {"val": val(res["theta"]), "theta": np.asarray(res["theta"]), "iter": 0}
    it, curve = [0], []

    def fun(x):
        if time.time() - t0 > cap:
            raise _Deadline
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    def cb(x):
        it[0] += 1
        if it[0] % CHECK == 0:
            s = val(x)
            curve.append((it[0], s))
            if s > best["val"]:
                best.update(val=s, theta=np.asarray(x).copy(), iter=it[0])

    capped, msg = False, ""
    try:
        out = minimize(fun, np.asarray(res["theta"]), jac=True, method="L-BFGS-B", callback=cb,
                       options={"maxiter": lbfgs_iters, "maxfun": 2 * lbfgs_iters, "ftol": 1e-13, "gtol": 1e-9})
        msg = str(out.message)
        s_last = val(out.x)
        if s_last > best["val"]:
            best.update(val=s_last, theta=np.asarray(out.x), iter=int(out.nit))
    except _Deadline:
        capped, msg = True, "wall-clock cap"
    return best["theta"], {"stop_iter": best["iter"], "lbfgs_iters": it[0], "lbfgs_message": msg, "capped": capped,
                           "adam_ll": -float(res["final_loss"]), "val_curve": curve, "best_val": best["val"]}


def best_stage1(job, model):
    """theta and init of the stage 1 fit `model` with the best training objective."""
    best = None
    for p in sorted((CFG["out"] / job.part / job.dataset / model).glob("*/result.json")):
        r = json.loads(p.read_text())
        if best is None or r["fit_ll"] > best[0]:
            best = (r["fit_ll"], p.parent)
    if best is None:
        raise RuntimeError(f"no stage 1 fit {model} for {job.dataset}")
    with np.load(best[1] / "params.npz") as z:
        return z["theta"], int(best[1].name)


def train_paths(job):
    if job.part == "synth":
        return synth_data(job.dataset)["X"]
    ds = real_data(job.dataset)
    return ds["b"][:ds["n_train"]][None]


def compact(T, kind):
    """Factors of an EM table (git-sized): 'tied' H[s, x', s'] (the same for every x); 'fact' R[x, s, s'] and
    Em[x, s', x'] with T[x, s, x', s'] = R[x, s, s'] Em[x, s', x']."""
    if kind == "tied":
        return {"H": T[0]}
    Rr = T.sum(2)
    return {"R": Rr, "Em": (T[:, 0] / Rr[:, 0, None, :]).transpose(0, 2, 1)}


def lin_of(model, params, spec):
    if "theta" in params:
        M, f = MV.model_M(jnp.asarray(params["theta"]), model, spec)
        return iolin.IOLin(np.asarray(M), np.asarray(f))
    if "H" in params:
        H = params["H"]
        return iolin.IOLin.from_iohmm(np.broadcast_to(H, (H.shape[1],) + H.shape))
    if "Em" in params:
        return iolin.IOLin.from_iohmm(np.einsum("xsk,xky->xsyk", params["R"], params["Em"]))
    if "stay" in params:
        return iolin.IOLin.from_iohmm(np.asarray(MV.msvar_T(params["c"], params["sig"], params["rho"], params["stay"],
                                                            spec)))
    if "T" in params:
        return iolin.IOLin.from_iohmm(params["T"])
    if "p" in params:
        return iolin.IOLin.from_kernel(np.tile(params["p"], (len(params["p"]), 1)))
    return iolin.IOLin.from_kernel(params["P"])


def fit_job(job):
    spec = spec_of(job.part, job.dataset)
    X = train_paths(job)
    Xj = jnp.asarray(X)
    K = 2 ** spec["n"]
    real = job.part == "real"
    score = val_scorer(real_data(job.dataset)) if real else None
    m, info = job.model, {}
    if m.startswith(("k-", "sw-", "cq")):
        if m.startswith("k-"):
            th0 = MV.init_kernel(m[2:], job.init, spec, X.ravel(), start="marginal" if real else "copy")
            steps, lr = (500, _iters(4000 if real else 2000)), ((0.01, 0.001) if real else (0.03, 0.003))
        else:
            if m.startswith("sw-"):
                kind = m.split("-")[1]
                k0, i1 = best_stage1(job, f"k-{kind}")
                th0 = MV.init_switch(k0, 2, job.init, kind, spec)
            else:
                k0, i1 = best_stage1(job, "k-qa")
                th0 = MV.cq_embed(k0, job.init, int(m[2:]), spec)
            steps, lr = (300, _iters(3000)), (0.003, 0.001)
            info["stage1_init"] = i1

        def loss(t):
            M, f = MV.model_M(t, m, spec)
            return -iolin.loglik(M, f, Xj)

        info["init_ll"] = float(-loss(jnp.asarray(th0)))
        if real:
            th, inf2 = fit_es_capped(loss, lambda t: score(*MV.model_M(jnp.asarray(t), m, spec)), th0,
                                     _iters(steps[0]), steps[1], *lr, _cap(job.cap))
        else:
            res = train.fit_capped(loss, jnp.asarray(th0), _iters(steps[0]), steps[1], _cap(job.cap), *lr)
            th = res["theta"]
            inf2 = {k: res[k] for k in ("adam_loss", "lbfgs_iters", "lbfgs_message", "capped")}
        info.update(inf2)
        params = {"theta": np.asarray(th)}
    elif m.startswith(("iohmm", "hmm")):
        S = int(m.split("-")[-1][1:])
        kind = "fact" if m.startswith("iohmm") else "tied"
        val = (lambda T: score(T.transpose(0, 2, 1, 3), np.ones(S))) if real else None
        T, info = MV.iohmm_fit(X, S, kind, job.init, K, max_iter=_iters(1000), val=val, max_seconds=_cap(job.cap))
        params = compact(T, kind)
        info["n_params"] = MV.iohmm_n_params(kind, S, K)
    elif m in ("markov1", "iid"):
        P = MV.markov_kernel(X, K, order=1 if m == "markov1" else 0)
        params = {"P": P} if m == "markov1" else {"p": P[0]}
        info["n_params"] = K * (K - 1) if m == "markov1" else K - 1
    elif m == "msvar":
        val = (lambda T: score(T.transpose(0, 2, 1, 3), np.ones(2))) if real else None
        T, fam, _, inf2 = MV.fit_msvar(X, spec, starts=2 if CFG["smoke"] else 4, seed=job.init,
                                       maxiter=_iters(1000), val=val)
        params, info = {k: np.asarray(v) for k, v in fam.items()}, {**inf2, "family": fam}
    else:
        raise ValueError(m)
    if "n_params" not in info:
        info["n_params"] = MV.model_n_params(m, spec)
    lin = lin_of(m, params, spec)
    out = {**info, "fit_ll": float(lin.path_loglik(X))}
    if real:
        ds = real_data(job.dataset)
        lp = D.log_scores(D.predictive(lin, ds["b"]), ds["b"])
        a, v = ds["n_train"], ds["n_val"]
        out.update(train_ll=float(lp[1:a].mean()), val_ll=float(lp[a:a + v].mean()), test_ll=float(lp[a + v:].mean()))
    else:
        out["train_ll"] = out["fit_ll"]
    return out, params


@contextlib.contextmanager
def _log_to(path):
    """Send this process's stdout and stderr (file descriptors included) to path while the fit runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush(), sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a") as fh:
        os.dup2(fh.fileno(), 1), os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            os.close(saved[0]), os.close(saved[1])


def run_job(job):
    d = result_dir(job)
    if (d / "result.json").exists():
        return job, "done", 0.0, ""
    d.mkdir(parents=True, exist_ok=True)
    t0, status, msg = time.time(), "error", ""
    with _log_to(CFG["out"] / "logs" / f"{job.part}__{job.dataset}__{job.model}__{job.init}.log"):
        print(f"{datetime.now():%H:%M:%S} start {job} pid {os.getpid()} device {jax.devices()[0]}", flush=True)
        try:
            res, params = fit_job(job)
            res.update(part=job.part, dataset=job.dataset, model=job.model, init=job.init, stage=job.stage,
                       wall_time=time.time() - t0, device=str(jax.devices()[0].platform))
            np.savez_compressed(d / "params.npz", **params)
            tmp = d / "result.json.tmp"
            tmp.write_text(json.dumps(res, indent=1, default=float) + "\n")
            os.replace(tmp, d / "result.json")
            status = "ran"
            msg = f"fit {res['fit_ll']:.4f}" + (f" val {res['val_ll']:.4f} test {res['test_ll']:.4f}"
                                                if "val_ll" in res else "")
            print(msg, flush=True)
        except Exception:
            traceback.print_exc()
            msg = traceback.format_exc().strip().splitlines()[-1]
    return job, status, time.time() - t0, msg


def _init_worker(out, smoke):
    CFG["out"], CFG["smoke"] = Path(out), smoke


def _match(model, pats):
    return any(fnmatch.fnmatchcase(model, p) for p in pats)


def select_jobs(parts, models=("*",), skip=()):
    jobs = [j for j in jobs_all() if j.part in parts and _match(j.model, models) and not _match(j.model, skip)]
    if CFG["smoke"]:
        jobs = [j for j in jobs if j.dataset in ("msvar", "eurgbp-v", "fx4-q") and j.init == 0]
    return jobs


def fit_stage(parts, models, skip, workers):
    for name in SYNTH if "synth" in parts else []:
        synth_data(name)  # written once before the workers start
    jobs = select_jobs(parts, models, skip)
    for phase in (1, 2):
        todo = [j for j in jobs if j.stage == phase and not (result_dir(j) / "result.json").exists()]
        todo.sort(key=lambda j: -j.cap)  # longest first
        print(f"{datetime.now():%F %T} phase {phase}: {len(todo)} fits ({jax.devices()[0].platform})", flush=True)
        if not todo:
            continue
        t0 = time.time()
        with mp.get_context("spawn").Pool(min(workers, len(todo)), _init_worker,
                                          (str(CFG["out"]), CFG["smoke"])) as pool:
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


def _params(part, ds, model, init):
    with np.load(CFG["out"] / part / ds / model / str(init) / "params.npz") as z:
        return {k: z[k] for k in z.files}


def seg_diff(dl, n_seg=20):
    n = len(dl) // n_seg * n_seg
    seg = np.asarray(dl[:n], float).reshape(n_seg, -1).mean(1)
    return [float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(n_seg))]


def dep_gain(P, b, spec):
    """Per-step log P(joint bin) - sum_c log P(coordinate bin), each from the same predictive law."""
    C = MV.coords(spec["d"], spec["k"])
    lj = np.log(np.maximum(P[np.arange(len(b)), b], 1e-300))
    lm = sum(np.log(np.maximum(Pc[np.arange(len(b)), C[b, c]], 1e-300)) for c, Pc in enumerate(MV.marginals(P, spec)))
    return lj - lm


PAIRS_Q = (("k-qa", "k-{rq}"), ("k-qa", "k-{mq}"), ("k-q0", "k-{r0}"), ("k-q0", "k-{m0}"), ("k-qa", "k-qn"),
           ("sw-qa-S2", "sw-{rq}-S2"), ("sw-qa-S2", "sw-{mq}-S2"), ("cq1", "sw-qa-S2"), ("cq1", "sw-{rq}-S2"),
           ("cq1", "sw-{mq}-S2"), ("cq1", "iohmm-fact-S2"), ("cq2", "cq1"), ("cq1", "k-qa"), ("sw-qa-S2", "k-qa"))


def model_pairs(spec):
    mt = matched(spec)
    sub = {"rq": mt["qa"][0], "r0": mt["q0"][0], "mq": mt["qa"][-1], "m0": mt["q0"][-1]}
    return [(a.format(**sub), b.format(**sub)) for a, b in PAIRS_Q]


def eval_synth():
    out = {}
    spec = MV.uniform_spec()
    for ds in SYNTH:
        d = synth_data(ds)
        xs, hs = d["test"], d["test_h"]
        truth = iolin.IOLin.from_iohmm(MV.truth_T(ds))
        lp_t = truth.logprobs(xs)
        Pt = D.predictive(truth, xs[:N_STAT])
        ct = MV.pred_corr(Pt, spec)
        rows, lps = {}, {}
        tr = {"heldout_ll": float(lp_t.mean()), "corr_by_regime": [float(ct[hs[:N_STAT] == h].mean()) for h in (0, 1)],
              "dep_gain": float(dep_gain(Pt, xs[:N_STAT], spec).mean()),
              "train_ll": float(truth.path_loglik(d["X"]))}
        for mo, rs in _results("synth", ds).items():
            r = max(rs, key=lambda x: x["train_ll"])
            lin = lin_of(mo, _params("synth", ds, mo, r["init"]), spec)
            lp = lin.logprobs(xs)
            lps[mo] = lp
            P = D.predictive(lin, xs[:N_STAT])
            c = MV.pred_corr(P, spec)
            kl = seg_diff(lp_t - lp)
            rows[mo] = {"init": r["init"], "n_inits": len(rs), "n_params": r["n_params"], "train_ll": r["train_ll"],
                        "heldout_kl_rate": kl[0], "heldout_se": kl[1], "capped": r.get("capped"),
                        "lbfgs_iters": r.get("lbfgs_iters"), "train_ll_inits": sorted(x["train_ll"] for x in rs),
                        "corr_by_regime": [float(c[hs[:N_STAT] == h].mean()) for h in (0, 1)],
                        "corr_rmse": float(np.sqrt(np.mean((c - ct) ** 2))),
                        "dep_gain": float(dep_gain(P, xs[:N_STAT], spec).mean())}
            print(f"{ds} {mo}: KL {kl[0]:.4f} ± {kl[1]:.4f}", flush=True)
        pairs = {f"{a}|{b}": seg_diff(lps[a] - lps[b]) for a, b in model_pairs(spec) if a in lps and b in lps}
        out[ds] = {"truth": tr, "models": rows, "pairs": pairs}
    (CFG["out"] / "synth").mkdir(parents=True, exist_ok=True)
    (CFG["out"] / "synth" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


def tail_mask(ds):
    """Joint lower-tail cells of coordinates 0 and 1: per coordinate the lowest bins with training frequency <= 10% (at
    least one); coordinate 1 uses its upper tail when its training correlation with coordinate 0 is negative."""
    spec = ds["spec"]
    C = MV.coords(spec["d"], spec["k"])
    nb, a, masks = 2 ** spec["k"], ds["n_train"], []
    for c in (0, 1):
        f = np.bincount(ds["bc"][:a, c], minlength=nb) / a
        if c == 1 and ds["corr_train"][0][1] < 0:
            f = f[::-1]
        k = max(1, int(np.sum(np.cumsum(f) <= 0.10)))
        bins = np.arange(k) if not (c == 1 and ds["corr_train"][0][1] < 0) else nb - 1 - np.arange(k)
        masks.append(np.isin(C[:, c], bins))
    return masks[0] & masks[1]


def stats_real(P, ds, rc, dcc_corr):
    """Every real-data score of one predictive matrix P (n, K)."""
    spec, b, z = ds["spec"], ds["b"], ds["z"]
    a, v, n = ds["n_train"], ds["n_val"], len(b)
    s = a + v
    lp = D.log_scores(P, b)
    C = MV.coords(spec["d"], spec["k"])
    mar = MV.marginals(P, spec)
    e = {"test_ll": float(lp[s:].mean()), "test_se": float(D.block_bootstrap(lp[s:], BLOCK)),
         "val_ll": float(lp[a:s].mean()), "train_ll": float(lp[1:a].mean()),
         "margin_test_ll": [float(np.log(np.maximum(Pc[np.arange(n), C[b, c]], 1e-300))[s:].mean())
                            for c, Pc in enumerate(mar)],
         "dep_gain": float(dep_gain(P[s:], b[s:], spec).mean())}
    pc = MV.pred_corr(P, spec)
    ok = np.isfinite(rc)
    tt = np.arange(n) >= s
    e["corr"] = {"mean_test": float(pc[tt].mean()), "vs_realized_corr": float(np.corrcoef(pc[tt & ok], rc[tt & ok])[0, 1]),
                 "vs_realized_mae": float(np.mean(np.abs(pc[tt & ok] - rc[tt & ok]))),
                 "vs_dcc_mae": float(np.mean(np.abs(pc[tt] - dcc_corr[tt])))}
    mask = tail_mask(ds)
    p = np.clip(P[s:] @ mask, 1e-12, 1 - 1e-12)
    o = mask[b[s:]].astype(float)
    e["tail"] = {"pred": float(p.mean()), "obs": float(o.mean()),
                 "log_score": float(np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))}
    Vz = np.mean([ds["scale"][:, c, None] * ds["values"][c][C[None, :, c]] for c in range(spec["d"])], axis=0)
    order = np.argsort(Vz, axis=1, kind="stable")
    Ps, Vs = np.take_along_axis(P, order, 1), np.take_along_axis(Vz, order, 1)
    rank = np.argsort(order, axis=1)[np.arange(n), b]
    e["portfolio"] = {f"{al}": D.backtest(Ps[s:], rank[s:], Vs[s:], al, loss=-z[s:].mean(1)) for al in ALPHAS}
    return e, lp


def eval_real(name):
    ds = real_data(name)
    spec, z, n = ds["spec"], ds["z"], len(ds["b"])
    a, v = ds["n_train"], ds["n_val"]
    s = a + v
    t0 = time.time()
    refs, info = MV.copula_refs(ds)
    dcc_corr = info.pop("R_dcc")[:, 0, 1]
    win = 21
    rc = np.full(n, np.nan)
    for t in range(n - win):
        rc[t] = np.corrcoef(z[t:t + win, 0], z[t:t + win, 1])[0, 1]
    print(f"{name}: references {time.time() - t0:.0f} s", flush=True)
    ev, lps = {}, {}
    for mo, P in refs.items():
        ev[f"ref-{mo}"], lps[f"ref-{mo}"] = stats_real(P, ds, rc, dcc_corr)
    allr = _results("real", name)
    for mo, rs in allr.items():
        r = max(rs, key=lambda x: x["val_ll"])
        lin = lin_of(mo, _params("real", name, mo, r["init"]), spec)
        P = D.predictive(lin, ds["b"])
        e, lp = stats_real(P, ds, rc, dcc_corr)
        tl = [x["test_ll"] for x in rs]
        e.update(init=r["init"], n_inits=len(rs), n_params=r["n_params"], stop_iter=r.get("stop_iter"),
                 lbfgs_iters=r.get("lbfgs_iters"), capped=r.get("capped"), test_ll_inits=[min(tl), max(tl)])
        ev[mo], lps[mo] = e, lp
        print(f"{name} {mo}: test {e['test_ll']:.4f} ± {e['test_se']:.4f}", flush=True)
    for mo, e in ev.items():
        for ref in ("ref-dcc", "ref-ind"):
            dl = lps[mo][s:] - lps[ref][s:]
            e[f"diff/{ref}"] = [float(dl.mean()), float(D.block_bootstrap(dl, BLOCK))]
    pairs = {}
    for x, y in model_pairs(spec):
        if x in lps and y in lps:
            dl = lps[x][s:] - lps[y][s:]
            pairs[f"{x}|{y}"] = [float(dl.mean()), float(D.block_bootstrap(dl, BLOCK))]
    out = {"dataset": name, "series": ds["series"], "grid": ds["grid"], "n": n, "n_test": n - s,
           "dates": [str(ds["dates"][0]), str(ds["dates"][s]), str(ds["dates"][-1])], "corr_train": ds["corr_train"],
           "refs_info": info, "models": ev, "pairs": pairs}
    (CFG["out"] / "real" / name).mkdir(parents=True, exist_ok=True)
    (CFG["out"] / "real" / name / "eval.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


# ------------------------------------------------------------------------------------------------ bench
def bench(path, models):
    """Seconds per value-and-gradient of the models (k-qa, cq1, cq2) on the current device (synthetic-sized data)."""
    spec = MV.uniform_spec()
    X = jnp.asarray(np.random.default_rng(0).integers(0, 256, (2000, 50)))
    out = {"device": jax.devices()[0].platform}
    for model in models:
        th = MV.init_kernel("qa", 0, spec, np.asarray(X).ravel())
        if model.startswith("cq"):
            th = MV.cq_embed(th, 0, int(model[2:]), spec)
        f = jax.jit(jax.value_and_grad(lambda t, m=model: -iolin.loglik(*MV.model_M(t, m, spec), X)))
        t0 = time.time()
        jax.block_until_ready(f(jnp.asarray(th)))
        comp, t0 = time.time() - t0, time.time()
        for k in range(3):
            jax.block_until_ready(f(jnp.asarray(th) + 1e-3 * (k + 1)))
        out[model] = (time.time() - t0) / 3
        out[f"{model}_compile"] = comp
        print(f"{out['device']} {model}: {out[model]:.3f} s per gradient (compile {comp:.0f} s)", flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="fit", choices=["fit", "eval", "all", "list", "bench"])
    ap.add_argument("--parts", nargs="*", default=["synth", "real"])
    ap.add_argument("--models", nargs="*", default=None, help="fnmatch patterns of the model ids to fit (default all)")
    ap.add_argument("--skip", nargs="*", default=[], help="fnmatch patterns of model ids not to fit")
    ap.add_argument("--datasets", nargs="*", default=None, help="real datasets to evaluate (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="few iterations, 2-minute caps, output to runs/seq9_smoke")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None, help="bench: write the timings here")
    a = ap.parse_args()
    CFG["smoke"] = a.smoke
    CFG["out"] = Path(a.out) if a.out else (ROOT / "runs" / ("seq9_smoke" if a.smoke else "seq9"))
    if a.smoke:
        SYNTH[:], REAL[:] = ["msvar"], ["eurgbp-v", "fx4-q"]
    if a.stage == "bench":
        bench(a.json, a.models or ["k-qa", "cq1", "cq2"])
        return
    models = a.models or ["*"]
    if a.stage == "list":
        J = select_jobs(a.parts, models, a.skip)
        done = sum((result_dir(j) / "result.json").exists() for j in J)
        print(f"{len(J)} jobs, {done} done, sum of caps {sum(j.cap for j in J) / 3600:.0f} h")
        for mo in sorted({j.model for j in J}):
            print(f"  {mo}: {sum(j.model == mo for j in J)}")
        return
    if a.stage in ("fit", "all"):
        fit_stage(a.parts, models, a.skip, a.workers)
    if a.stage in ("eval", "all"):
        if "synth" in a.parts:
            eval_synth()
        if "real" in a.parts:
            names = a.datasets or REAL
            if a.stage == "eval" and len(names) == 1:
                eval_real(names[0])
            else:
                me = [sys.executable, __file__, "eval", "--parts", "real", "--out", str(CFG["out"])]
                me += ["--smoke"] if a.smoke else []
                procs = [subprocess.Popen(me + ["--datasets", nm]) for nm in names]
                if any(p.wait() for p in procs):
                    sys.exit("an eval worker failed")
    if a.stage == "all" and not a.smoke:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "seq9_summary.py")], check=True)


if __name__ == "__main__":
    main()
