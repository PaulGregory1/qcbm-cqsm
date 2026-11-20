"""Sequence 10: parsimony on rare events (archb.tails). For each target, encoding, training size and data seed, every model
family on a ladder of sizes, scored on exact tail metrics.

Levels and datasets (<target>-<enc>-n<size>-s<seed>; encodings u = uniform + Gray, g = sign + log-magnitude, binary)
  1  i.i.d. laws: t3, jump, gauss (6 bits), tcop, gcop (2 x 4 bits); n = 1,000 / 10,000 / 100,000 samples; seeds 0, 1
  2  Markov kernels: art, arch (6 bits); n = 100 / 400 / 2,000 paths of 50 steps (burn-in 200); seeds 0, 1
  3  hidden regimes: crash, msart (6 bits); n = 400 / 2,000 paths of 50 steps; seeds 0, 1; held-out path of 100,000 steps
Models (2 inits each, the one with the best training likelihood is kept)
  level 1  circuits qB0L1, 2, 4, 8 and qA2L1, 2, 4 (static Born machines), rbm 2-32, mix 4-32, mixt 2-16 (1D), mps / nmps
           bond 2, 3, 4, 6 (Born-rule and nonnegative controls), oracle (mixt1 on t3, mix2 on jump, mix1 on gauss;
           bivariate t / Gaussian copula with t margins on tcop / gcop), tab (add-1/2 counts)
  level 2  circuits qB0L1-4, qA2L1-3 (init 0 copy, init 1 near i.i.d.), rbm 2-16, mix 4-32, mixt 2-16, oracle (mixt1 on
           art: AR-t with location-scale dependence; ARCH family on arch), tab
  level 3  stage 1 kernels k-qA2L3, k-rbm16, k-mix32, k-mixt16; stage 2 cq1, cq2 (C-QSM from the best k-qA2L3),
           sw-<kind>-S2 (bit-linear switch from the best stage 1 kernel), iohmm-fact-S2, oracle (switching AR-t family,
           10 parameters), markov1
Scores (exact, from the truth; written in result.json)
  1  KL(p || q); per alpha in (1e-2, 1e-3): censored KL on the two-sided alpha tails of p, exceedance ratio q(A) / p(A),
     lower-tail ratio; 2D: censored KL on the cells where either coordinate is in its lower alpha tail, joint lower-tail
     ratio q(J) / p(J) (alpha in 0.05, 0.01)
  2  the same per input row, weighted by the truth's stationary law (tails of each row), and on the model's stationary law
  3  held-out KL rate (20-segment SE) and censored KL rate on the truth's stationary alpha tails (alpha 1e-2, 1e-3) along
     the held-out path, mean exceedance ratio and mean |log ratio| of the predicted tail probabilities
Outputs: runs/seq10/L<level>/<dataset>/<model>/<init>/{result.json, params.npz}; logs in runs/seq10/logs/.

Usage: python scripts/run_seq10.py [fit|list|bench] [--levels 1 2 3] [--models PAT ..] [--skip PAT ..] [--workers W]
       [--smoke]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import fnmatch  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zlib  # noqa: E402
from collections import namedtuple  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402
from scipy.stats import t as student  # noqa: E402

from archb import iolin, train  # noqa: E402
from archb import multivar as MV  # noqa: E402
from archb import realdata as D  # noqa: E402
from archb import regimes as R  # noqa: E402
from archb import tails as TL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = {"out": ROOT / "runs" / "seq10", "smoke": False}
SIZES = {1: (1000, 10000, 100000), 2: (100, 400, 2000), 3: (400, 2000)}
SEEDS = (0, 1)
ENCS = ("u", "g")
CAPS = {1: 600, 2: 1200, "k": 1200, "sw": 5400, "cq": 7200, "em": 1800, "fam": 3600}
N_TEST, N_STAT = 100_000, 50_000
Job = namedtuple("Job", "level stage dataset model init cap")
_CACHE = {}


class _Deadline(Exception):
    pass


# ------------------------------------------------------------------------------------------------ datasets
def ds_name(target, enc, size, seed):
    return f"{target}-{enc}-n{size}-s{seed}"


def parse_ds(name):
    t, enc, n, s = name.split("-")
    return t, enc, int(n[1:]), int(s[1:])


def grid_of(target):
    d = TL.LEVEL1.get(target, 1)
    return lambda enc: TL.grid(enc, 6 if d == 1 else 8, d)


def _seed(*a):
    return zlib.crc32("|".join(map(str, a)).encode())


def _paths(P, n_paths, seed, n_keep=50, burn=200):
    """Paths of a Markov kernel on the bins started from its stationary law."""
    rng = np.random.default_rng(seed)
    pi = iolin.IOLin.from_kernel(P).marginal()
    x = rng.choice(len(P), size=n_paths, p=np.clip(pi, 0, None) / np.clip(pi, 0, None).sum())
    cum = np.cumsum(P, 1)
    cum[:, -1] = 1.0
    X = np.empty((n_paths, n_keep), np.int64)
    for t in range(burn + n_keep):
        if t >= burn:
            X[:, t - burn] = x
        x = (cum[x] < rng.random(n_paths)[:, None]).sum(1)
    return X


def data(level, name):
    """Training data and the truth of a dataset: level 1 counts, level 2 paths and counts, level 3 paths and the held-out
    path (shared by the sizes of a seed)."""
    key = (level, name)
    if key in _CACHE:
        return _CACHE[key]
    t, enc, n, s = parse_ds(name)
    G = grid_of(t)(enc)
    p = CFG["out"] / "data" / f"L{level}-{name}.npz"
    if level == 1:
        truth = TL.truth_static(t, G)
        if not p.exists():
            b = np.random.default_rng(_seed(t, enc, s)).choice(G["K"], size=SIZES[1][-1], p=truth / truth.sum())[:n]
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, b=b)
        b = np.load(p)["b"]
        out = {"G": G, "truth": truth, "b": b, "counts": np.bincount(b, minlength=G["K"]).astype(float)}
    elif level == 2:
        P = TL.truth_kernel(t, G)
        if not p.exists():
            X = _paths(P, SIZES[2][-1], _seed(t, enc, s))[:n]
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, X=X)
        X = np.load(p)["X"]
        N = np.zeros((G["K"], G["K"]))
        np.add.at(N, (X[:, :-1].ravel(), X[:, 1:].ravel()), 1.0)
        out = {"G": G, "truth": P, "X": X, "N": N, "b": X.ravel()}
    else:
        T = TL.truth_T(t, G)
        if not p.exists():
            X, _ = R.simulate(T, SIZES[3][-1], 50, R.BURN_IN, _seed(t, enc, s))
            xt, ht = R.simulate(T, 1, N_TEST, R.BURN_IN, _seed(t, enc, s, "test"))
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, X=X[:n], test=xt[0], test_h=ht[0])
        z = np.load(p)
        X = z["X"]
        N = np.zeros((G["K"], G["K"]))
        np.add.at(N, (X[:, :-1].ravel(), X[:, 1:].ravel()), 1.0)
        out = {"G": G, "truth": T, "X": X, "N": N, "b": X.ravel(), "test": z["test"], "test_h": z["test_h"]}
    _CACHE[key] = out
    return out


# ------------------------------------------------------------------------------------------------ job table
L1_MODELS = (["qB0L1", "qB0L2", "qB0L4", "qB0L8", "qA2L1", "qA2L2", "qA2L4"] + [f"rbm{h}" for h in (2, 4, 8, 16, 32)] +
             [f"mix{k}" for k in (4, 8, 16, 32)] + [f"mps{c}" for c in (2, 3, 4, 6)] + [f"nmps{c}" for c in (2, 3, 4, 6)])
L1_MIXT = [f"mixt{k}" for k in (2, 4, 8, 16)]
L2_MODELS = (["qB0L1", "qB0L2", "qB0L3", "qB0L4", "qA2L1", "qA2L2", "qA2L3"] + [f"rbm{h}" for h in (2, 4, 8, 16)] +
             [f"mix{k}" for k in (4, 8, 16, 32)] + [f"mixt{k}" for k in (2, 4, 8, 16)])
L3_KERNELS = ["qA2L3", "rbm16", "mix32", "mixt16"]


def jobs_all():
    J = []
    for level, targets in ((1, list(TL.LEVEL1)), (2, list(TL.LEVEL2)), (3, list(TL.LEVEL3))):
        for t in targets:
            for enc in ENCS:
                for n in SIZES[level]:
                    for s in SEEDS:
                        ds = ds_name(t, enc, n, s)
                        if level == 1:
                            ms = L1_MODELS + (L1_MIXT if TL.LEVEL1[t] == 1 else [])
                            J += [Job(1, 1, ds, m, i, CAPS[1]) for m in ms for i in range(2)]
                            J += [Job(1, 1, ds, m, 0, CAPS[1]) for m in ("oracle", "tab")]
                        elif level == 2:
                            J += [Job(2, 1, ds, m, i, CAPS[2]) for m in L2_MODELS for i in range(2)]
                            J += [Job(2, 1, ds, m, 0, CAPS[2]) for m in ("oracle", "tab")]
                        else:
                            J += [Job(3, 1, ds, f"k-{k}", i, CAPS["k"]) for k in L3_KERNELS for i in range(2)]
                            J += [Job(3, 1, ds, "iohmm-fact-S2", i, CAPS["em"]) for i in range(2)]
                            J += [Job(3, 1, ds, m, 0, CAPS["fam"]) for m in ("oracle", "markov1")]
                            J += [Job(3, 2, ds, f"sw-{k}-S2", i, CAPS["sw"]) for k in L3_KERNELS for i in range(2)]
                            J += [Job(3, 2, ds, f"cq{m}", i, CAPS["cq"]) for m in (1, 2) for i in range(2)]
    return J


def result_dir(job):
    return CFG["out"] / f"L{job.level}" / job.dataset / job.model / str(job.init)


def _it(n):
    return max(2, n // 100) if CFG["smoke"] else n


def _cap(c):
    return 90 if CFG["smoke"] else c


# ------------------------------------------------------------------------------------------------ scores
ALPHAS = (1e-2, 1e-3)


def scores_static(p, q, G):
    out = {"kl": TL.kl(p, q)}
    if G["d"] == 1:
        for a in ALPHAS:
            A, Lo = TL.tail_mask(p, a), TL.tail_mask(p, a, "lower")
            out[f"{a:g}"] = {"ckl": TL.censored_kl(p, q, A), "ratio": float(q[A].sum() / p[A].sum()),
                             "ratio_lower": float(q[Lo].sum() / p[Lo].sum()), "mass": float(p[A].sum())}
        return out
    B, C = G["B"], G["coords"]
    P2 = p.reshape(B, B)
    for a in (0.05, 0.01):
        lo = [TL.tail_mask(P2.sum(1 - c), a, "lower") for c in (0, 1)]
        U = lo[0][C[:, 0]] | lo[1][C[:, 1]]
        J = lo[0][C[:, 0]] & lo[1][C[:, 1]]
        out[f"{a:g}"] = {"ckl": TL.censored_kl(p, q, U), "joint_ratio": float(q[J].sum() / p[J].sum()),
                         "joint_mass": float(p[J].sum()), "ratio": float(q[U].sum() / p[U].sum())}
    return out


def scores_kernel(P, Q):
    pi = iolin.IOLin.from_kernel(P).marginal()
    piq = iolin.IOLin.from_kernel(Q).marginal()
    out = {"kl": float(sum(pi[x] * TL.kl(P[x], Q[x]) for x in range(len(P))))}
    for a in ALPHAS:
        ck, num, den = 0.0, 0.0, 0.0
        for x in range(len(P)):
            A = TL.tail_mask(P[x], a)
            ck += pi[x] * TL.censored_kl(P[x], Q[x], A)
            num, den = num + pi[x] * Q[x][A].sum(), den + pi[x] * P[x][A].sum()
        As = TL.tail_mask(pi, a)
        out[f"{a:g}"] = {"ckl": float(ck), "ratio": float(num / den), "stat_ckl": TL.censored_kl(pi, piq, As),
                         "stat_ratio": float(piq[As].sum() / pi[As].sum())}
    return out


def seg(dl, n_seg=20):
    n = len(dl) // n_seg * n_seg
    s = np.asarray(dl[:n], float).reshape(n_seg, -1).mean(1)
    return [float(s.mean()), float(s.std(ddof=1) / np.sqrt(n_seg))]


def scores_path(lin, d):
    truth = iolin.IOLin.from_iohmm(d["truth"])
    xs = d["test"]
    out = {"kl_rate": seg(truth.logprobs(xs) - lin.logprobs(xs))}
    xs = xs[:N_STAT]
    Pt, Pm = D.predictive(truth, xs), D.predictive(lin, xs)
    pi = truth.marginal()
    i = np.arange(len(xs))
    for a in ALPHAS:
        A = TL.tail_mask(pi, a)
        inA = A[xs]
        pa, qa = Pt[:, A].sum(1), np.clip(Pm[:, A].sum(1), 1e-300, 1 - 1e-15)
        st = np.where(inA, np.log(Pt[i, xs]), np.log1p(-np.minimum(pa, 1 - 1e-15)))
        sm = np.where(inA, np.log(np.maximum(Pm[i, xs], 1e-300)), np.log1p(-qa))
        out[f"{a:g}"] = {"ckl_rate": seg(st - sm), "ratio": float(qa.mean() / pa.mean()),
                         "abs_log_ratio": float(np.mean(np.abs(np.log(qa / pa)))), "mass": float(pa.mean())}
    return out


# ------------------------------------------------------------------------------------------------ oracles
def oracle_static(target, d):
    """The true family. 1D: the mixture family member it belongs to (returns its kind); 2D: (theta, q) by maximum
    likelihood on the counts (Nelder-Mead): bivariate t on tcop, Gaussian copula with t margins on gcop."""
    if target in ("t3", "jump", "gauss"):
        return {"t3": "mixt1", "jump": "mix2", "gauss": "mix1"}[target]
    G, c = d["G"], d["counts"]
    e = G["edges"]

    def q_of(z):
        if target == "tcop":
            P = TL._biv_t_cells(1 + np.exp(z[3]), np.tanh(z[2]), e, 16, np.exp(z[0]), np.exp(z[1]))
            return np.maximum(P / P.sum(), 0.0).ravel()
        nu1, s1, nu2, s2, rho = 1 + np.exp(z[0]), np.exp(z[1]), 1 + np.exp(z[2]), np.exp(z[3]), np.tanh(z[4])
        H1 = np.clip(norm.ppf(student.cdf(e / s1, nu1)), -TL.BIG, TL.BIG)
        H2 = np.clip(norm.ppf(student.cdf(e / s2, nu2)), -TL.BIG, TL.BIG)
        return np.asarray(MV.bvn_cells(0.0, 0.0, 1.0, 1.0, rho, H1, H2, 16)).ravel()

    z0 = np.array([0.0, 0.0, 0.5, np.log(4.0)]) if target == "tcop" else np.array([np.log(4.0), 0, np.log(4.0), 0, 0.5])
    f = lambda z: -float(c @ np.log(np.maximum(q_of(z), 1e-300))) / c.sum()
    out = minimize(f, z0, method="Nelder-Mead", options={"maxiter": _it(2000), "xatol": 1e-6, "fatol": 1e-10})
    return out.x, q_of(out.x)


def arch_fit(d):
    G, N = d["G"], jnp.asarray(d["N"])
    v, e = jnp.asarray(G["values"]), jnp.asarray(G["edges_m"])
    Q = lambda z: TL.norm_cells(z[0] + 0.0 * v, jnp.sqrt(jnp.exp(z[1]) + jnp.exp(z[2]) * v**2), e)
    vg = jax.jit(jax.value_and_grad(lambda z: -jnp.sum(N * jnp.log(jnp.maximum(Q(z), 1e-300))) / jnp.sum(N)))
    out = minimize(lambda z: tuple(np.asarray(a, float) for a in vg(jnp.asarray(z))), np.array([0.0, -1.0, -0.5]),
                   jac=True, method="L-BFGS-B", options={"maxiter": _it(1000)})
    return out.x, np.asarray(Q(jnp.asarray(out.x)))


# ------------------------------------------------------------------------------------------------ fitting
def best_stage1(job, model):
    best = None
    for p in sorted((CFG["out"] / f"L{job.level}" / job.dataset / model).glob("*/result.json")):
        r = json.loads(p.read_text())
        if best is None or r["train_ll"] > best[0]:
            best = (r["train_ll"], p.parent)
    if best is None:
        raise RuntimeError(f"no stage 1 fit {model} for {job.dataset}")
    return np.load(best[1] / "params.npz")["theta"], int(best[1].name)


def lin_L3(model, params, G):
    if model.startswith("cq"):
        M, f = TL.cqsm_M(params["theta"], int(model[2:]), G)
        return iolin.IOLin(np.asarray(M), np.asarray(f))
    if model.startswith("sw-"):
        return iolin.IOLin(np.asarray(TL.switch_M(params["theta"], 2, model.split("-")[1], G)), np.ones(2))
    if model.startswith("k-"):
        return iolin.IOLin.from_kernel(np.asarray(TL.cond_probs(params["theta"], model[2:], G)))
    if "Em" in params:
        return iolin.IOLin.from_iohmm(np.einsum("xsk,xky->xsyk", params["R"], params["Em"]))
    if "z" in params:
        return iolin.IOLin.from_iohmm(np.asarray(TL.ms_family_T(jnp.asarray(params["z"]), G)))
    return iolin.IOLin.from_kernel(params["P"])


def _grad_fit(loss, th0, adam, lbfgs, cap, lr):
    res = train.fit_capped(loss, jnp.asarray(th0), _it(adam), _it(lbfgs), _cap(cap), *lr)
    return np.asarray(res["theta"]), {"train_ll": -float(res["final_loss"]), "lbfgs_iters": res["lbfgs_iters"],
                                      "capped": res["capped"], "lbfgs_message": res["lbfgs_message"]}


def fit_job(job):
    d = data(job.level, job.dataset)
    G, m = d["G"], job.model
    target = parse_ds(job.dataset)[0]
    info = {}
    if job.level == 1:
        c = jnp.asarray(d["counts"])
        if m == "tab":
            q = (d["counts"] + 0.5) / (d["counts"] + 0.5).sum()
            return {"n_params": G["K"] - 1, "train_ll": float(d["counts"] @ np.log(q) / d["counts"].sum()),
                    **scores_static(d["truth"], q, G)}, {"q": q}
        kind = m
        if m == "oracle":
            orc = oracle_static(target, d)
            if not isinstance(orc, str):
                z, q = orc
                return {"n_params": len(z), "train_ll": float(d["counts"] @ np.log(np.maximum(q, 1e-300)) /
                                                              d["counts"].sum()), **scores_static(d["truth"], q, G)}, \
                    {"z": z}
            kind = info["oracle_kind"] = orc
        th0 = TL.init(kind, job.init, G, d["b"], False)
        th, inf2 = _grad_fit(lambda t: -jnp.sum(c * jnp.log(jnp.maximum(TL.static_probs(t, kind, G), 1e-300))) /
                             jnp.sum(c), th0, 500, 2000, job.cap, (0.03, 0.003))
        q = np.asarray(TL.static_probs(th, kind, G))
        return {**info, **inf2, "n_params": TL.n_params(kind, G, False), **scores_static(d["truth"], q, G)}, \
            {"theta": th, "q": q}
    if job.level == 2:
        N = jnp.asarray(d["N"])
        tot = float(d["N"].sum())
        if m == "tab":
            Q = (d["N"] + 0.5) / (d["N"] + 0.5).sum(1, keepdims=True)
            return {"n_params": G["K"] * (G["K"] - 1), "train_ll": float(np.sum(d["N"] * np.log(Q)) / tot),
                    **scores_kernel(d["truth"], Q)}, {"Q": Q}
        if m == "oracle" and target == "arch":
            z, Q = arch_fit(d)
            return {"n_params": 3, "train_ll": float(np.sum(d["N"] * np.log(np.maximum(Q, 1e-300))) / tot),
                    **scores_kernel(d["truth"], Q)}, {"z": z}
        kind = "mixt1" if m == "oracle" else m
        th0 = TL.init(kind, job.init, G, d["b"], True, start="copy" if job.init == 0 else "marginal")
        th, inf2 = _grad_fit(lambda t: -jnp.sum(N * jnp.log(jnp.maximum(TL.cond_probs(t, kind, G), 1e-300))) / tot,
                             th0, 500, 2000, job.cap, (0.03, 0.003))
        Q = np.asarray(TL.cond_probs(th, kind, G))
        return {**inf2, "n_params": TL.n_params(kind, G, True), **scores_kernel(d["truth"], Q)}, {"theta": th}
    # level 3
    X = d["X"]
    Xj = jnp.asarray(X)
    if m.startswith(("k-", "sw-", "cq")):
        if m.startswith("k-"):
            kind = m[2:]
            th0 = TL.init(kind, job.init, G, d["b"], True, start="copy" if job.init == 0 else "marginal")
            Mf = lambda t: (TL.cond_probs(t, kind, G)[:, :, None, None], jnp.ones(1))
            npar, steps, lr = TL.n_params(kind, G, True), (500, 2000), (0.01, 0.001)
        elif m.startswith("sw-"):
            kind = m.split("-")[1]
            k0, i1 = best_stage1(job, f"k-{kind}")
            th0, info["stage1_init"] = TL.init_switch(k0, 2, job.init, G), i1
            Mf = lambda t: (TL.switch_M(t, 2, kind, G), jnp.ones(2))
            npar = 2 * (G["n"] + 1) + 2 * TL.n_params(kind, G, True)
            steps, lr = (300, 3000), (0.003, 0.001)
        else:
            mm = int(m[2:])
            k0, i1 = best_stage1(job, "k-qA2L3")
            from archb import cqsm
            th0, info["stage1_init"] = cqsm.embed(k0, job.init, mm, 2, 3, G["n"]), i1
            Mf = lambda t: TL.cqsm_M(t, mm, G)
            npar, steps, lr = len(th0), (300, 3000), (0.003, 0.001)
        th, inf2 = _grad_fit(lambda t: -iolin.loglik(*Mf(t), Xj), th0, steps[0], steps[1], job.cap, lr)
        params, info = {"theta": th}, {**info, **inf2, "n_params": npar}
    elif m == "iohmm-fact-S2":
        T, info = MV.iohmm_fit(X, 2, "fact", job.init, G["K"], max_iter=_it(1000), max_seconds=_cap(job.cap))
        Rr = T.sum(2)
        params = {"R": Rr, "Em": (T[:, 0] / Rr[:, 0, None, :]).transpose(0, 2, 1)}
        info["n_params"] = MV.iohmm_n_params("fact", 2, G["K"])
    elif m == "oracle":
        vg = jax.jit(jax.value_and_grad(lambda z: -iolin.loglik(TL.ms_family_T(z, G).transpose(0, 2, 1, 3),
                                                                jnp.ones(2), Xj)))
        best = None
        for k in range(1 if CFG["smoke"] else 3):
            rng = np.random.default_rng(k)
            z0 = np.concatenate([0.1 * rng.normal(size=2), [0.5, 0.5] + 0.1 * rng.normal(size=2),
                                 np.log([0.5, 1.5]) + 0.2 * rng.normal(size=2), np.log([4.0, 4.0]),
                                 [3.0, 2.0] + rng.normal(size=2)])
            out = minimize(lambda z: tuple(np.asarray(a, float) for a in vg(jnp.asarray(z))), z0, jac=True,
                           method="L-BFGS-B", options={"maxiter": _it(1000), "ftol": 1e-14, "gtol": 1e-9})
            if best is None or out.fun < best.fun:
                best = out
        params, info = {"z": np.asarray(best.x)}, {"n_params": 10, "message": str(best.message)}
    else:  # markov1
        Q = (d["N"] + 0.5) / (d["N"] + 0.5).sum(1, keepdims=True)
        params, info = {"P": Q}, {"n_params": G["K"] * (G["K"] - 1)}
    lin = lin_L3(m, params, G)
    info["train_ll"] = info.get("train_ll", float(lin.path_loglik(X)))
    info["truth_train_ll"] = float(iolin.IOLin.from_iohmm(d["truth"]).path_loglik(X))
    return {**info, **scores_path(lin, d)}, params


# ------------------------------------------------------------------------------------------------ pool
@contextlib.contextmanager
def _log_to(path):
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
    with _log_to(CFG["out"] / "logs" / f"L{job.level}__{job.dataset}__{job.model}__{job.init}.log"):
        print(f"{datetime.now():%H:%M:%S} start {job} pid {os.getpid()}", flush=True)
        try:
            res, params = fit_job(job)
            res.update(level=job.level, dataset=job.dataset, model=job.model, init=job.init, wall_time=time.time() - t0)
            np.savez_compressed(d / "params.npz", **{k: np.asarray(v) for k, v in params.items() if k != "q"})
            tmp = d / "result.json.tmp"
            tmp.write_text(json.dumps(res, indent=1, default=float) + "\n")
            os.replace(tmp, d / "result.json")
            status = "ran"
            key = "kl_rate" if job.level == 3 else "kl"
            msg = f"train {res['train_ll']:.4f} {key} {res[key][0] if job.level == 3 else res[key]:.4g}"
        except Exception:
            traceback.print_exc()
            msg = traceback.format_exc().strip().splitlines()[-1]
    return job, status, time.time() - t0, msg


def _init_worker(out, smoke):
    CFG["out"], CFG["smoke"] = Path(out), smoke


def _match(model, pats):
    return any(fnmatch.fnmatchcase(model, p) for p in pats)


def select(levels, models, skip):
    J = [j for j in jobs_all() if j.level in levels and _match(j.model, models) and not _match(j.model, skip)]
    if CFG["smoke"]:
        keep = {ds_name(t, enc, SIZES[lv][0], 0) for lv, t in ((1, "t3"), (1, "tcop"), (2, "arch"), (3, "crash"))
                for enc in ENCS}
        J = [j for j in J if j.dataset in keep and j.init == 0]
    return J


def fit_stage(levels, models, skip, workers):
    jobs = select(levels, models, skip)
    for j in {(j.level, j.dataset) for j in jobs}:
        data(*j)  # data files written once, before the workers start
    for phase in (1, 2):
        todo = [j for j in jobs if j.stage == phase and not (result_dir(j) / "result.json").exists()]
        todo.sort(key=lambda j: -j.cap)
        print(f"{datetime.now():%F %T} phase {phase}: {len(todo)} fits", flush=True)
        if not todo:
            continue
        t0 = time.time()
        with mp.get_context("spawn").Pool(min(workers, len(todo)), _init_worker, (str(CFG["out"]), CFG["smoke"])) as pool:
            for k, (job, st, sec, msg) in enumerate(pool.imap_unordered(run_job, todo), 1):
                print(f"[{k}/{len(todo)} {(time.time() - t0) / 60:.0f} min] L{job.level} {job.dataset} {job.model} "
                      f"{job.init}: {st} {sec:.0f} s {msg}", flush=True)


def bench():
    """Seconds per value-and-gradient of the level 3 memory models (6 bits, 2,000 x 50 paths)."""
    G = TL.grid("g")
    X = jnp.asarray(np.random.default_rng(0).integers(0, 64, (2000, 50)))
    from archb import cqsm
    k0 = TL.init("qA2L3", 0, G, np.asarray(X).ravel(), True)
    for name, th, Mf in (("k-qA2L3", k0, lambda t: (TL.cond_probs(t, "qA2L3", G)[:, :, None, None], jnp.ones(1))),
                         ("sw-qA2L3-S2", TL.init_switch(k0, 2, 0, G), lambda t: (TL.switch_M(t, 2, "qA2L3", G), jnp.ones(2))),
                         ("sw-mixt16-S2", TL.init_switch(TL.init("mixt16", 0, G, np.asarray(X).ravel(), True), 2, 0, G),
                          lambda t: (TL.switch_M(t, 2, "mixt16", G), jnp.ones(2))),
                         ("cq1", cqsm.embed(k0, 0, 1, 2, 3, 6), lambda t: TL.cqsm_M(t, 1, G)),
                         ("cq2", cqsm.embed(k0, 0, 2, 2, 3, 6), lambda t: TL.cqsm_M(t, 2, G))):
        f = jax.jit(jax.value_and_grad(lambda t, Mf=Mf: -iolin.loglik(*Mf(t), X)))
        t0 = time.time()
        jax.block_until_ready(f(jnp.asarray(th)))
        c, t0 = time.time() - t0, time.time()
        for k in range(3):
            jax.block_until_ready(f(jnp.asarray(th) + 1e-3 * (k + 1)))
        print(f"{jax.devices()[0].platform} {name}: {(time.time() - t0) / 3:.3f} s per gradient (compile {c:.0f} s)",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="fit", choices=["fit", "list", "bench"])
    ap.add_argument("--levels", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--models", nargs="*", default=["*"])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="few iterations, 90 s caps, output to runs/seq10_smoke")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    CFG["smoke"] = a.smoke
    CFG["out"] = Path(a.out) if a.out else (ROOT / "runs" / ("seq10_smoke" if a.smoke else "seq10"))
    if a.stage == "bench":
        bench()
        return
    if a.stage == "list":
        J = select(a.levels, a.models, a.skip)
        done = sum((result_dir(j) / "result.json").exists() for j in J)
        print(f"{len(J)} jobs, {done} done, sum of caps {sum(j.cap for j in J) / 3600:.0f} h")
        for lv in (1, 2, 3):
            print(f"  level {lv}: {sum(j.level == lv for j in J)}")
        return
    fit_stage(a.levels, a.models, a.skip, a.workers)


if __name__ == "__main__":
    main()
