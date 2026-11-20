"""Sequence 4, target D: the C-QSM against classical models on real daily log-returns.
Modules: archb.realdata (data, grids, scores, backtests), archb.families (GARCH, MS-AR), archb.cqsm / iolin / regimes
(circuits, linear representations, IOHMM by EM, Markov chains), as in sequence 4.

Datasets <series>-<grid>: series eurusd (FRED DEXUSEU), nasdaq (FRED NASDAQCOM), 1999-01-04 to the download date;
grid u (32 equal bins of [-4, 4] training sd), q (training quantiles) or v (the uniform grid on returns divided by
an EWMA volatility of past bins). Split 70 / 10 / 20 % chronologically.
Models (ids = inits):
  archb-B0, archb-anc2   Architecture B, 3 layers, inits 0..2 (= stage 1), started near the i.i.d. law (marginal_init)
  cqsm-m<m>[-anc2]       C-QSM, staged from the best stage 1 fit by training likelihood, seeds 0..4
  qsm-m<m>               sequence 3 QSM (general isometry), inits 0..4
  iohmm-<full|fact>-<S>  input-output HMM by EM, S = 2, 4, inits 0..4;  hmm-<S>: HMM on x alone (tied), S = 2, 4
  markov-<k>             order-k chain on the bins, k = 0 (i.i.d.), 1, 2, add-1/2 counts
  garch-<norm|t|gjr>-<cont|bin>  GARCH(1,1) (gjr: GJR-GARCH-t), continuous inputs (continuous MLE) or binned inputs
                         (binned MLE)
  msar-<cont|bin>        MS-AR(1), 2 regimes, same two forms
Training: maximum likelihood on the training period (one path, first value from the stationary law); circuits Adam then
L-BFGS-B as in sequence 4 (stage 1: Adam 500 steps lr 0.01 -> 0.001, L-BFGS cap 4,000, from marginal_init instead of
the copy start), EM with the sequence 4 pseudo-counts; early stopping on the validation log-likelihood
(checked every 50 L-BFGS / 10 EM iterations, best iterate kept). Every fit is scored on the bins of the whole series
(log P(b_t | past)). Stage 2 of the C-QSM starts from the stage 1 init with the best training objective; in the
evaluation, the init with the best VALIDATION log-likelihood is the model's fit, and variants are chosen on validation.

Stages:
  fit      fit every (dataset, model, init) -> runs/seq4_real/<ds>/<model>/<init>/{result.json, params.npz};
           resumable, lock files as in run_seq4.py, so several workers can run at once
  eval     per dataset, for the selected init of every model (and the 5-seed mixture of the chosen C-QSM): test
           log-score with block-bootstrap SE, paired differences to GARCH-t (continuous and binned inputs), per-year
           differences, 1-day and 10-day VaR / ES backtests, stylized facts, memory readouts
           -> runs/seq4_real/<ds>/eval.json, eval_arrays.npz
  all      fit with --workers processes, eval (one process per dataset), then scripts/seq4_real_report.py

Usage: python scripts/run_seq4_real.py [fit|eval|all] [--datasets ...] [--models ...] [--workers W]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from archb import ceiling, cqsm, families as F, iolin, qseq, realdata as D, regimes as R, targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "seq4_real"
L = 3
STAGE1, STAGE2, QSM_INITS, EM_INITS = range(3), range(5), range(5), range(5)
CHECK_LBFGS, CHECK_EM = 50, 10
ALPHAS, HORIZON, N_MC, ACF_LAGS, BLOCK = (0.01, 0.05), 10, 5000, 50, 50
REF, REF_BIN = "garch-t-cont", "garch-t-bin"
LAM_FLOOR = 0.94
DATASETS = [f"{s}-{g}" for s in ("eurusd", "nasdaq") for g in ("u", "q")]
V_DATASETS = [f"{s}-v" for s in ("eurusd", "nasdaq")]  # volatility-normalized grid

MODELS = ([(f"archb-{v}", i) for v in ("B0", "anc2") for i in STAGE1]
          + [(f"cqsm-m{m}{s}", i) for s in ("", "-anc2") for m in (1, 2) for i in STAGE2]
          + [(f"qsm-m{m}", i) for m in (1, 2) for i in QSM_INITS]
          + [(f"iohmm-{k}-{S}", i) for k in ("full", "fact") for S in (2, 4) for i in EM_INITS]
          + [(f"hmm-{S}", i) for S in (2, 4) for i in EM_INITS]
          + [(f"markov-{k}", 0) for k in (0, 1, 2)]
          + [(f"garch-{d}-{u}", 0) for d in ("norm", "t", "gjr") for u in ("cont", "bin")]
          + [(f"msar-{u}", 0) for u in ("cont", "bin")])

_DS = {}


def dataset(name):
    if name not in _DS:
        s, g = name.split("-")
        _DS[name] = D.make_dataset(s, g)
    return _DS[name]


def _dir(name, model, i):
    return OUT / name / model / str(i)


# ------------------------------------------------------------------------------------------------ fitting
def val_scorer(ds):
    """Mean validation log-score of a linear representation (M, f), filtering from the start of the series."""
    a, v = ds["n_train"], ds["n_val"]
    xs = jnp.asarray(ds["b"][:a + v])

    def score(M, f):
        M, f = jnp.asarray(M), jnp.asarray(f)
        return float(jnp.mean(iolin.logprobs(M, f, iolin.stationary(M, f), xs)[a:]))

    return score


def fit_es(loss, val, th0, adam_steps, lbfgs_iters, lr0=ceiling.LR0, lr1=ceiling.LR1):
    """Adam then L-BFGS-B (as archb.ceiling.fit), keeping the iterate with the best validation score (checked after Adam
    and every CHECK_LBFGS L-BFGS iterations)."""
    res = ceiling.adam(None, jnp.asarray(th0), adam_steps, None, loss, lr0, lr1)
    vg = jax.jit(jax.value_and_grad(loss))
    best = {"val": val(res["theta"]), "theta": np.asarray(res["theta"]), "iter": 0}
    it, curve = [0], []

    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    def cb(x):
        it[0] += 1
        if it[0] % CHECK_LBFGS == 0:
            s = val(x)
            curve.append((it[0], s))
            if s > best["val"]:
                best.update(val=s, theta=np.asarray(x).copy(), iter=it[0])

    out = minimize(fun, np.asarray(res["theta"]), jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": lbfgs_iters, "maxfun": 2 * lbfgs_iters, "ftol": 1e-13, "gtol": 1e-9})
    s_last = val(out.x)
    if s_last > best["val"]:
        best.update(val=s_last, theta=np.asarray(out.x), iter=int(out.nit))
    return best["theta"], {"stop_iter": best["iter"], "lbfgs_iters": int(out.nit), "lbfgs_message": str(out.message),
                           "fit_ll_last": -float(out.fun), "val_ll_last": s_last, "adam_ll": -float(res["final_loss"]),
                           "val_curve": curve}


def best_stage1(name, model):
    best = None
    for i in STAGE1:
        p = _dir(name, model, i) / "result.json"
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        if best is None or r["fit_ll"] > best[0]:
            best = (r["fit_ll"], i)
    return np.load(_dir(name, model, best[1]) / "params.npz")["theta"], best[1]


def circuit_spec(model):
    if model.startswith("archb"):
        return 0, (2 if model.endswith("anc2") else 0)
    return int(model.split("-")[1][1:]), (2 if model.endswith("anc2") else 0)


def marginal_init(seed, n_anc, b_train, noise=0.02):
    """Stage 1 start for near-i.i.d. returns: every angle N(0, noise^2) (no copy gates: an identity kernel is far from
    returns), the layer-0 ancilla R_y + pi/2, and the final R_y layer setting each output bit to its training frequency
    (bit r of the Gray label, MSB first), so the start is close to a product approximation of the i.i.d. law."""
    gates = cqsm.gate_list(0, n_anc, L)
    th = noise * np.random.default_rng(seed).normal(size=len(gates))
    th[[k for k, g in enumerate(gates) if g[3] == "anc_plus"]] += np.pi / 2
    lab = targets.gray_labels(5, 1)[b_train]
    p1 = np.array([np.mean((lab >> (4 - r)) & 1) for r in range(5)])
    th[-5:] = 2 * np.arcsin(np.sqrt(p1))
    return th


def fit_circuit(name, model, i, ds):
    m, a = circuit_spec(model)
    X = jnp.asarray(ds["b"][:ds["n_train"]])[None]
    if m == 0:
        th0, extra = marginal_init(i, a, ds["b"][:ds["n_train"]]), {}
        kw = {"adam_steps": 500, "lbfgs_iters": 4000, "lr0": 0.01, "lr1": 0.001}
    else:
        st = best_stage1(name, "archb-anc2" if a else "archb-B0")
        th0, extra = cqsm.embed(st[0], i, m, a, L), {"stage1_init": st[1]}
        kw = {"adam_steps": 300, "lbfgs_iters": 3000, "lr0": 0.003, "lr1": 0.001}
    score, f = val_scorer(ds), cqsm.f_vec(m)
    th, info = fit_es(lambda t: -cqsm.loglik(t, X, m, a, L), lambda t: score(cqsm.transfer(jnp.asarray(t), m, a, L), f),
                      th0, **kw)
    info.update(memory_dim=2**m, n_ancillas=a, n_params=cqsm.n_params(m, a, L), **extra)
    return th, float(cqsm.loglik(jnp.asarray(th), X, m, a, L)), info, {"theta": th}


def fit_qsm(name, model, i, ds):
    m = int(model[-1])
    X, f = jnp.asarray(ds["b"][:ds["n_train"]])[None], cqsm.f_vec(m)
    Mf = lambda t: cqsm.qsm_transfer(qseq.kraus(jnp.asarray(t), "iso", m, 5))
    score = val_scorer(ds)
    th, info = fit_es(lambda t: -iolin.loglik(Mf(t), f, X), lambda t: score(Mf(t), f),
                      qseq.init_theta(i, "iso", m, 5), adam_steps=500, lbfgs_iters=2000)
    info.update(memory_dim=2**m, n_params=qseq.n_params("iso", m, 5), n_effective=qseq.n_effective(m, 5))
    return th, float(iolin.loglik(Mf(th), f, X)), info, {"theta": th}


def fit_em(name, model, i, ds):
    parts = model.split("-")
    kind, S = ("tied", int(parts[1])) if parts[0] == "hmm" else (parts[1], int(parts[2]))
    X = jnp.asarray(ds["b"][:ds["n_train"]])[None]
    rng = np.random.default_rng(i)
    T, q, prior = jnp.asarray(R._random_T(kind, S, rng)), jnp.full(S, 1.0 / S), R.PRIOR[kind](S)
    score, one = val_scorer(ds), np.ones(S)
    best, curve, prev = None, [], None
    for it in range(1000):
        Tn, qn, ll = R._em_step(T, q, X, kind, prior)
        if it and it % CHECK_EM == 0:  # never keep the random start
            s = score(np.asarray(T).transpose(0, 2, 1, 3), one)
            curve.append((it, s))
            if best is None or s > best[0]:
                best = (s, np.asarray(T), it)
        if prev is not None and float(ll) - prev < 1e-9:
            break
        prev, T, q = float(ll), Tn, qn
    s = score(np.asarray(T).transpose(0, 2, 1, 3), one)
    if s > best[0]:
        best = (s, np.asarray(T), it)
    lin = iolin.IOLin.from_iohmm(best[1])
    info = {"memory_dim": S, "n_params": R.n_params_iohmm(kind, S), "em_iters": it, "stop_iter": best[2],
            "val_curve": curve}
    return best[1], lin.path_loglik(np.asarray(X)), info, {"T": best[1]}


def fit_markov(name, model, i, ds):
    k = int(model[-1])
    lin, tab = D.iolin_of_markov(ds["b"][:ds["n_train"]], k)
    ll = float(np.mean(D.log_scores(D.predictive(lin, ds["b"][:ds["n_train"]]), ds["b"][:ds["n_train"]])[k:]))
    return tab, ll, {"memory_dim": 32 ** max(k - 1, 0), "n_params": 32**k * 31}, {"Pc": tab}


def fit_family(name, model, i, ds):
    parts = model.split("-")
    inputs = {"cont": "continuous", "bin": "binned"}[parts[-1]]
    if parts[0] == "garch":
        g, ll = F.fit_garch(ds, "t" if parts[1] == "gjr" else parts[1], inputs, gjr=parts[1] == "gjr")
        return g, ll, {"memory_dim": 1, "n_params": g.n_params(), "params": g.p}, {"th": g.th, "s0": g.s0}
    m, ll = F.fit_msar(ds, inputs)
    return m, ll, {"memory_dim": 2, "n_params": 8, "params": m.p}, {"th": m.th}


def fitter(model):
    if model.startswith(("archb", "cqsm")):
        return fit_circuit
    return {"qsm": fit_qsm, "iohmm": fit_em, "hmm": fit_em, "markov": fit_markov, "garch": fit_family,
            "msar": fit_family}[model.split("-")[0]]


# ------------------------------------------------------------------------------------------------ models from runs
def path_facts(V, cap=1e3):
    """acf of |x| and kurtosis from simulated value paths (chains, n). On the volatility-normalized grid a model that
    keeps emitting edge bins makes the scale recursion explode along its own path: such chains (non-finite or beyond
    cap) are dropped and their fraction is returned as a third value."""
    keep = np.isfinite(V).all(1) & (np.abs(np.nan_to_num(V, nan=np.inf)).max(1) < cap)
    if not keep.any():
        return np.full(ACF_LAGS, np.nan), float("nan"), 1.0
    V = V[keep]
    return (np.mean([D.sample_acf(np.abs(v), ACF_LAGS) for v in V], axis=0), D.kurtosis(V.ravel()),
            float(1 - keep.mean()))


class Lin:
    """A linear representation (archb.iolin) with the evaluation interface."""

    def __init__(self, lin):
        self.lin, self._st = lin, None

    def predictive(self, ds):
        return D.predictive(self.lin, ds["b"])

    def mc_sums(self, ds, t, h, n, rng):
        if self._st is None or self._st[0] is not ds:
            self._st = (ds, self.lin.filtered(ds["b"]))
        return D.mc_sums(self.lin, self._st[1][t], ds["b"][t], ds["values"], h, n, rng,
                         (ds["scale"][t + 1], ds["lam"], ds["centres"]))

    def facts(self, ds):
        """Exact on fixed grids; on the volatility-normalized grid, simulated paths with the scale they imply."""
        if ds["lam"] == 1:
            acf, k = D.stationary_facts(self.lin, ds["values"], ACF_LAGS)
            return np.asarray(acf), k
        with np.errstate(over="ignore", invalid="ignore"):
            V = D.path_values(ds, D.sample_lin(self.lin, 20, 10000, np.random.default_rng(0)))
        return path_facts(V)

    def states(self, ds):
        return self.lin.filtered(ds["b"])


class Sim:
    """A continuous-state family (GARCH, continuous-input MS-AR): stylized facts from simulated paths."""

    def __init__(self, m):
        self.m = m

    def predictive(self, ds):
        return self.m.predictive(ds)

    def mc_sums(self, ds, t, h, n, rng):
        return self.m.mc_sums(ds, t, h, n, rng)

    def facts(self, ds):
        with np.errstate(over="ignore", invalid="ignore"):
            V = self.m.sample_values(ds, 20, 10000, np.random.default_rng(0))
        return path_facts(V)


def build(name, model, i):
    ds, p = dataset(name), np.load(_dir(name, model, i) / "params.npz")
    kind = model.split("-")[0]
    if kind in ("archb", "cqsm"):
        m, a = circuit_spec(model)
        return Lin(cqsm.io(p["theta"], m, a, L))
    if kind == "qsm":
        m = int(model[-1])
        return Lin(iolin.IOLin(np.asarray(cqsm.qsm_transfer(qseq.kraus(jnp.asarray(p["theta"]), "iso", m, 5))),
                               np.asarray(cqsm.f_vec(m))))
    if kind in ("iohmm", "hmm"):
        return Lin(iolin.IOLin.from_iohmm(p["T"]))
    if kind == "markov":
        return Lin(D.iolin_of_markov(ds["b"][:ds["n_train"]], int(model[-1]))[0])
    inputs = {"cont": "continuous", "bin": "binned"}[model.split("-")[-1]]
    if kind == "garch":
        d = model.split("-")[1]
        return Sim(F.Garch(p["th"], "t" if d == "gjr" else d, inputs, float(p["s0"]), gjr=d == "gjr"))
    msar = F.MSAR(p["th"], inputs, ds)
    return Lin(msar.lin) if inputs == "binned" else Sim(msar)


# ------------------------------------------------------------------------------------------------ fit stage
def run(name, model, i):
    d = _dir(name, model, i)
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
            lock.write_text(str(os.getpid()))
    t0 = time.time()
    ds = dataset(name)
    _, fit_ll, info, params = fitter(model)(name, model, i, ds)
    np.savez(d / "params.npz", **params)
    P = build(name, model, i).predictive(ds)
    lp = D.log_scores(P, ds["b"])
    a, v = ds["n_train"], ds["n_val"]
    res = {"dataset": name, "model": model, "init": i, "fit_ll": fit_ll, "train_ll": float(lp[2:a].mean()),
           "val_ll": float(lp[a:a + v].mean()), "test_ll": float(lp[a + v:].mean()), **info,
           "wall_time": time.time() - t0}
    (d / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    lock.unlink(missing_ok=True)
    return "ran"


def fit_stage(names, prefixes, max_minutes):
    t0 = time.time()
    jobs = [(n, mo, i) for n in names for mo, i in MODELS if prefixes is None or any(mo.startswith(p) for p in prefixes)]
    while True:
        waiting = 0
        for n, mo, i in jobs:
            if (time.time() - t0) / 60 > max_minutes:
                print("time budget reached", flush=True)
                sys.exit(3)
            t1 = time.time()
            st = run(n, mo, i)
            waiting += st in ("waiting", "locked")
            if st == "ran":
                r = json.loads((_dir(n, mo, i) / "result.json").read_text())
                print(f"{n} {mo} {i}: train {r['train_ll']:.4f} val {r['val_ll']:.4f} test {r['test_ll']:.4f} "
                      f"[{time.time() - t1:.0f} s]", flush=True)
        if not waiting:
            break
        time.sleep(20)


# ------------------------------------------------------------------------------------------------ eval stage
def selected(name):
    """{model: result of the init with the best VALIDATION log-likelihood} (the criterion of the early stopping; the
    training objective would pick the init that stopped latest, i.e. the most overfit), and {model: all results}."""
    out, allr = {}, {}
    for p in sorted((OUT / name).glob("*/*/result.json")):
        r = json.loads(p.read_text())
        allr.setdefault(r["model"], []).append(r)
        if r["model"] not in out or r["val_ll"] > out[r["model"]]["val_ll"]:
            out[r["model"]] = r
    return out, allr


class Mix:
    """Equal-weight mixture of the predictive laws of several fits (the seeds of one C-QSM variant)."""

    def __init__(self, parts):
        self.parts = parts

    def predictive(self, ds):
        return np.mean([m.predictive(ds) for m in self.parts], axis=0)

    def mc_sums(self, ds, t, h, n, rng):
        k = len(self.parts)
        return np.concatenate([m.mc_sums(ds, t, h, n // k, rng) for m in self.parts])


def readout(F_, y, a, s):
    """Least-squares linear readout of y from the features F_ (rows), fitted on rows [21, a), correlation on [s, n)."""
    X = np.concatenate([F_, np.ones((len(F_), 1))], axis=1)
    ok = np.isfinite(y)
    tr = np.arange(len(y))[(np.arange(len(y)) >= 21) & (np.arange(len(y)) < a) & ok]
    te = np.arange(len(y))[(np.arange(len(y)) >= s) & ok]
    w = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    return X @ w, float(np.corrcoef(X[te] @ w, y[te])[0, 1]), float(np.corrcoef(X[tr] @ w, y[tr])[0, 1])


def ewma_log_sd(v, lam=LAM_FLOOR):
    """log of the EWMA sd of the observed values after each day (a zero-parameter volatility filter)."""
    out, s2 = np.empty(len(v)), float(np.mean(v**2))
    for t, x in enumerate(v):
        s2 = lam * s2 + (1 - lam) * x**2
        out[t] = 0.5 * np.log(s2)
    return out


def evaluate(name):
    ds = dataset(name)
    sel, allr = selected(name)
    a, v, n = ds["n_train"], ds["n_val"], len(ds["b"])
    s = a + v
    b, z = ds["b"], ds["z"]
    V = D.row_values(ds)  # bin values of every day, z units
    vb = V[np.arange(n), b]
    starts = np.arange(s - 1, n - HORIZON, HORIZON)
    s_real = np.array([vb[t + 1:t + 1 + HORIZON].sum() for t in starts])
    z_real = np.array([z[t + 1:t + 1 + HORIZON].sum() for t in starts])
    emp_acf, emp_kurt = D.sample_acf(np.abs(vb), ACF_LAGS), D.kurtosis(vb)
    ev, arrays, models = {}, {}, {}
    cq = max([m for m in sel if m.startswith("cqsm")], key=lambda m: sel[m]["val_ll"])
    todo = [(mo, r) for mo, r in sel.items()] + [("cqsm-mix", {"init": None, "n_params": None, "memory_dim": None,
                                                                "of": cq})]
    for mo, r in todo:
        t0 = time.time()
        if mo == "cqsm-mix":
            M = models[mo] = Mix([build(name, cq, x["init"]) for x in allr[cq]])
        else:
            M = models[mo] = build(name, mo, r["init"])
        P = M.predictive(ds)
        lp = D.log_scores(P, b)
        arrays[f"lp/{mo}"] = lp
        e = {"init": r["init"], "test_ll": float(lp[s:].mean()), "test_se": float(D.block_bootstrap(lp[s:], BLOCK)),
             "val_ll": float(lp[a:s].mean()), "train_ll": float(lp[2:a].mean()), "n_params": r["n_params"],
             "memory_dim": r["memory_dim"], "filter_size": M.lin.d if isinstance(M, Lin) else None}
        if mo == "cqsm-mix":
            e.update(of=cq, n_inits=len(allr[cq]))
        else:
            tl = [x["test_ll"] for x in allr[mo]]
            e.update(n_inits=len(tl), test_ll_inits_min=float(min(tl)), test_ll_inits_max=float(max(tl)),
                     init_by_fit=max(allr[mo], key=lambda x: x["fit_ll"])["init"])
        e["var1"] = {f"{al}": D.backtest(P[s:], b[s:], V[s:], al, loss=-z[s:]) for al in ALPHAS}
        S = np.stack([M.mc_sums(ds, t, HORIZON, N_MC, np.random.default_rng(10_000 + k)) for k, t in enumerate(starts)])
        e["var10"] = {f"{al}": D.backtest_samples(S, s_real, al, loss=-z_real) for al in ALPHAS}
        if hasattr(M, "facts"):
            acf, kurt, *rest = M.facts(ds)
            e["exploding_paths"] = rest[0] if rest else 0.0
            arrays[f"acf/{mo}"] = acf
            e["acf_abs_err_max"] = float(np.abs(acf - emp_acf).max())
            e["acf_abs_err_mean"] = float(np.abs(acf - emp_acf).mean())
            e["kurtosis"] = float(kurt)
        e["eval_time"] = time.time() - t0
        ev[mo] = e
        print(f"{name} {mo}: test {e['test_ll']:.4f} ± {e['test_se']:.4f} [{e['eval_time']:.0f} s]", flush=True)
    for mo in ev:  # paired differences to the references (continuous- and binned-input GARCH-t)
        for key, ref in (("diff_ref", REF), ("diff_refbin", REF_BIN)):
            dlt = arrays[f"lp/{mo}"][s:] - arrays[f"lp/{ref}"][s:]
            ev[mo][key], ev[mo][key + "_se"] = float(dlt.mean()), float(D.block_bootstrap(dlt, BLOCK))
            ev[mo][key + "_se_blocks"] = {str(k): float(D.block_bootstrap(dlt, k)) for k in (20, 100)}
    # per calendar year, whole series: volatility and paired differences to the reference
    years = np.array([d[:4] for d in ds["dates"]])
    split = np.where(np.arange(n) < a, "train", np.where(np.arange(n) < s, "val", "test"))
    keys = [cq, "cqsm-mix", REF_BIN, "garch-gjr-cont", "msar-cont", "markov-0",
            max([m for m in sel if m.startswith("archb")], key=lambda m: sel[m]["val_ll"])]
    yearly = []
    for y in np.unique(years):
        ix = np.flatnonzero((years == y) & (np.arange(n) >= 2))
        yearly.append({"year": y, "n": len(ix), "split": "/".join(sorted(set(split[ix]))), "sd": float(z[ix].std()),
                       "diff": {mo: float(np.mean(arrays[f"lp/{mo}"][ix] - arrays[f"lp/{REF}"][ix])) for mo in keys}})
    # usual continuous backtests of the continuous-input families on the raw returns
    cont = {}
    for mo in [m for m in ev if m.startswith("garch") and m.endswith("cont")]:
        g = models[mo].m
        sd = np.sqrt(g.sigma2(ds))[s:]
        cont[mo] = {f"{al}": D.continuous_backtest(z[s:], g.p["mu"], sd, g.ppf, g.cdf, al) for al in ALPHAS}
    # memory readouts: the C-QSM chosen on validation, against realized volatility and the MS-AR smoothed regime
    rv = np.log(D.realized_vol(ds["r"], 21))
    sm = models["msar-cont"].m.smoothed(ds)
    arrays["rv21"], arrays["msar_smoothed"] = rv, sm
    mem = {}
    pick = lambda pref: max([m for m in sel if m.startswith(pref)], key=lambda m: sel[m]["val_ll"])
    for key, mo in (("cqsm", cq), ("iohmm", pick("iohmm")), ("msar-bin", "msar-bin"), ("archb", pick("archb")),
                    ("ewma", None)):
        if key == "archb":
            feats = np.eye(32)[b]  # a Markov kernel: the readout can only use the current bin
        elif key == "ewma":
            feats = ewma_log_sd(vb)[:, None]  # zero-parameter floor: EWMA of the observed bin values
        else:
            St = models[mo].states(ds)
            feats = np.concatenate([St.real, St.imag], axis=1) if np.iscomplexobj(St) else St
        y_rv, c_rv, c_rv_tr = readout(feats, rv, a, s)
        y_sm, c_sm, c_sm_tr = readout(feats, sm, a, s)
        mem[key] = {"model": mo or "ewma", "n_features": feats.shape[1], "corr_rv_test": c_rv, "corr_rv_train": c_rv_tr,
                    "corr_msar_test": c_sm, "corr_msar_train": c_sm_tr}
        arrays[f"mem_rv/{key}"], arrays[f"mem_msar/{key}"] = y_rv, y_sm
    g = models[REF].m
    ls = np.log(np.sqrt(g.sigma2(ds)))[1:]  # sigma_{t+1}, known after day t
    y, c, ctr = readout(ls[:, None], rv[:-1], a, s)
    mem["garch-t"] = {"model": REF, "n_features": 1, "corr_rv_test": c, "corr_rv_train": ctr,
                      "corr_msar_test": readout(ls[:, None], sm[:-1], a, s)[1]}
    arrays["mem_rv/garch-t"] = np.concatenate([y, [np.nan]])
    out = {"dataset": name, "series": ds["series"], "grid": ds["grid"], "dates": [ds["dates"][0], ds["dates"][a],
           ds["dates"][s], ds["dates"][-1]], "n_train": a, "n_val": v, "n_test": n - s, "mu": ds["mu"], "sd": ds["sd"],
           "clip_frac": ds["clip_frac"], "edge_frac": ds["edge_frac"], "occupancy_train": ds["occupancy_train"],
           "edges": [float(x) for x in ds["edges"][1:-1]], "values": ds["values"].tolist(), "n_windows": len(starts),
           "lam": ds["lam"], "empirical": {"kurtosis": emp_kurt, "acf_abs": emp_acf.tolist(),
                                           "test_kurtosis": D.kurtosis(vb[s:]), "test_sd": float(z[s:].std()),
                                           "val_sd": float(z[a:s].std())},
           "models": ev, "continuous_var1": cont, "memory": mem, "yearly": yearly, "ref": REF, "ref_bin": REF_BIN,
           "cqsm_chosen": cq}
    arrays["emp_acf"] = emp_acf
    (OUT / name / "eval.json").write_text(json.dumps(out, indent=1) + "\n")
    np.savez_compressed(OUT / name / "eval_arrays.npz", **arrays)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="fit", choices=["fit", "eval", "all"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", nargs="*", default=None, help="model name prefixes (default: all)")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    ap.add_argument("--max-minutes", type=float, default=1200.0)
    a = ap.parse_args()
    if a.stage == "fit":
        fit_stage(a.datasets, a.models, a.max_minutes)
    elif a.stage == "eval":
        for n in a.datasets:
            evaluate(n)
    else:
        me = [sys.executable, __file__]
        extra = ["--datasets", *a.datasets] + (["--models", *a.models] if a.models else [])
        procs = [subprocess.Popen(me + ["fit", *extra]) for _ in range(a.workers)]
        if any(p.wait() for p in procs):
            sys.exit("a fit worker failed")
        procs = [subprocess.Popen(me + ["eval", "--datasets", n]) for n in a.datasets]
        if any(p.wait() for p in procs):
            sys.exit("an eval worker failed")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "seq4_real_report.py")], check=True)


if __name__ == "__main__":
    main()
