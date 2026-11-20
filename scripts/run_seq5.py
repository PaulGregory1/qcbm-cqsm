"""Sequence 5: can a stochastic QSM (Kraus rank r > 1, dephased memory) beat positive classical models (HMMs,
IOHMMs) at equal memory or equal filter cost? Modules: archb.qseq (stochastic / dephased QSM), archb.hmm (FLW qubit
truth, Hankel ranks, exact-law HMM fits), archb.switch (classical-memory switch), archb.cqsm (dephased C-QSM).

Experiments (tier 1 first; within a tier, longest wall-clock cap first):
  e2a        FLW qubit process (lam = 0.8, alpha = 1; Theorem 5 of arXiv:2209.11225 with p = 1; no finite HMM), T = 5e4
             (seeds 0, 1) and 5e3 (seeds 0-2): QSM D = 2 with r = 1 (exact) and r = 2, 6 inits; HMM S = 2, 3, 4, 6, 8,
             12, 16, 6 inits + the renewal-type hazard chain ('lump'); Markov orders 1-8; truth.            tier 1
  e2a-exact  HMM S = 4, 8, 12, 16 fitted to the exact law of the words of length 16 (L-BFGS on D_16), from the best
             T = 5e4 seed 0 fit ('fit', waits for it) and from the hazard chain ('lump').                    tier 1
  e2b        uniform renewal N = 16 (seed 0) and 32 (seeds 0, 1), T = 5e4: QSM D = 2 (4 inits), D = 4 (6), D = 8
             (renewal32 s0, 3); HMM S = 2, 4, 8, 16 (6 + lump; lump = consecutive countdown states merged), S = 32
             (renewal32 s0, 3); Markov 1-8. Seed 1: QSM_4 and HMM_16 only. The QSM_4 / HMM_16 pair on N = 32 is
             tier 1, the rest tier 2.
  a1         switch between Architecture B kernels (archb.switch) on msar / smsar16 (seq4 data): anc2 kernels S = 2
             (5 inits each), S = 4 on smsar16 (3); B0 kernels S = 2 (5 each). Seq4 stage 2 schedule.          tier 1
  reeval     seq4 best fits (training likelihood) of C-QSM m = 1 / 2 + 2 anc, Architecture B anc2 and the factored
             IOHMM_2, re-evaluated with per-step held-out log-probabilities.                               tier 1
  a2         C-QSM m = 1 + 2 anc with the memory dephased (gamma = 1), msar / smsar16, seq4 seeds 0-4.        tier 2
  e1         QSM D = 4, r = 16 and D = 2, r = 4, gamma in {0, .05, .1, .25, .5, 1} on renewal6 / renewal8 (D = 2 on
             renewal8 only), r = 1 references, HMM_D references, 4 inits.                                     tier 3
  flw05      control: FLW lam = 0.5, T = 5e4, seed 0, the e2a models.                                        tier 4
Fits: QSM Adam 500 steps (lr 0.03 -> 0.003) then L-BFGS (3,000); HMM Baum-Welch (<= 500 iterations, at most half the
cap) then L-BFGS polish; circuits (a1, a2) Adam 300 steps (lr 0.003 -> 0.001) then L-BFGS 3,000 (seq4 stage 2). Every fit
stops at its wall-clock cap and records capped: true; the best point evaluated is kept.
Metrics: held-out KL rate on a held-out string of 201,600 symbols (seed 999; seq4 held-out path for a1 / a2 / reeval)
with the 20-segment SE; per-step held-out log-probabilities (lp_test.npy, float32) for paired SEs; binary processes:
exact KL_L, D_L, d_L for L = 1..16 and the gap law.

-> runs/seq5/<exp>/<dataset>/<model>/<init>/{result.json, params.npz, loss.npy, lp_test.npy}; datasets in
runs/seq5/data/, Hankel ranks of the truths in runs/seq5/hankel/, one log per fit in runs/seq5/logs/. Resumable
(finished fits skipped, lock files with the pid).

Usage: python scripts/run_seq5.py [--exps ...] [--tiers ...] [--datasets ...] [--models ...] [--inits ...] [--workers 6]
                                  [--max-minutes M] [--stop-at HH:MM] [--out DIR] [--list]
       python scripts/run_seq5.py --smoke     (one fit per type, short caps, -> runs/seq5_smoke/)
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import namedtuple  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from archb import cqsm, hmm, iolin, qseq, regimes as R, switch, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEQ4 = ROOT / "runs" / "seq4"
CFG = {"out": ROOT / "runs" / "seq5", "smoke": False}
N_TEST, TEST_SEED, LMAX = 201600, 999, 16
ADAM_QSM, ADAM_STAGE2, LBFGS, EM_ITERS, EXACT_ITERS = 500, 300, 3000, 500, 20000
SMOKE_CAP = 20
HMM_INITS = [0, 1, 2, 3, 4, 5, "lump"]
GAMMAS = [0.0, 1.0, 0.25, 0.05, 0.1, 0.5]  # e1 order: 0, 1, .25 first
Job = namedtuple("Job", "tier exp dataset model init cap")


def configure(out, smoke=False):
    CFG["out"], CFG["smoke"] = Path(out), smoke


# ------------------------------------------------------------------------------------------------ datasets
DATASETS = {}
for _T, _seeds, _base in ((50000, (0, 1), 1000), (5000, (0, 1, 2), 1010)):
    for _s in _seeds:
        DATASETS[f"flw-T{_T}-s{_s}"] = {"kind": "flw", "lam": 0.8, "T": _T, "seed": _base + _s}
DATASETS["flw05-T50000-s0"] = {"kind": "flw", "lam": 0.5, "T": 50000, "seed": 1500}
for _N, _seeds, _base in ((16, (0,), 1100), (32, (0, 1), 1200), (6, (0,), 1300), (8, (0,), 1400)):
    for _s in _seeds:
        DATASETS[f"renewal{_N}-s{_s}"] = {"kind": "renewal", "N": _N, "T": 50000, "seed": _base + _s}
SEQ4_DATA = {"msar": ("msar", None, 700), "smsar16": ("smsar", 16, 730)}  # as scripts/run_seq4.py


def truth_kraus(spec):
    return hmm.flw_qubit(spec["lam"], 1.0)


def truth_lin(spec):
    return qseq.linear(truth_kraus(spec)) if spec["kind"] == "flw" else hmm.hmm_linear(hmm.renewal(spec["N"]))


def truth_logprobs(spec, xs):
    if spec["kind"] == "flw":
        return np.asarray(qseq.logprobs(jnp.asarray(truth_kraus(spec)), jnp.asarray(xs)))
    return hmm.hmm_logprobs(hmm.renewal(spec["N"]), xs)


def prepare_data(names):
    """Training and held-out strings of the binary datasets, and the Hankel ranks of their truths (main process)."""
    for name in names:
        spec = DATASETS[name]
        p = CFG["out"] / "data" / f"{name}.npz"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            if spec["kind"] == "flw":
                lin = truth_lin(spec)
                tr, te = hmm.sample_linear(lin, spec["T"], spec["seed"]), hmm.sample_linear(lin, N_TEST, TEST_SEED)
            else:
                T = hmm.renewal(spec["N"])
                tr, te = hmm.sample(T, spec["T"], spec["seed"]), hmm.sample(T, N_TEST, TEST_SEED)
            tmp = p.with_suffix(".tmp.npz")
            np.savez(tmp, train=tr, test=te)
            os.replace(tmp, p)
        h = CFG["out"] / "hankel" / f"{name}.json"
        if not h.exists():
            h.parent.mkdir(parents=True, exist_ok=True)
            lin = truth_lin(spec)
            rk, sv = hmm.hankel_rank(lin, L=12)
            rg, svg = hmm.hankel_rank(lin, gap=True, n_gap=64)
            out = {"dataset": name, **spec, "rank_words_le6": rk, "sv_words": sv[:12].tolist(), "rank_gap_law_64": rg,
                   "sv_gap": svg[:12].tolist(), "rank_analytic": spec.get("N")}
            h.write_text(json.dumps(out, indent=1) + "\n")


def load_binary(name):
    with np.load(CFG["out"] / "data" / f"{name}.npz") as z:
        return z["train"], z["test"]


def load_seq4(name):
    kind, N, seed = SEQ4_DATA[name]
    return R.load_data(name, kind, N, seed)


_REF = {}


def reference(name):
    """Truth word laws (L <= 16), held-out and training log-probabilities, computed once per process."""
    if name not in _REF:
        if name in SEQ4_DATA:
            kind, N, _ = SEQ4_DATA[name]
            d = load_seq4(name)
            lin = iolin.IOLin.from_iohmm(R.truth_T(kind, N))
            _REF[name] = {"lin": lin, "lp": lin.logprobs(d["test"]), "train_ll": lin.path_loglik(d["X"])}
        else:
            spec = DATASETS[name]
            tr, te = load_binary(name)
            _REF[name] = {"P": truth_lin(spec).word_probs(LMAX), "lp": truth_logprobs(spec, te),
                          "train_ll": float(np.mean(truth_logprobs(spec, tr)))}
    return _REF[name]


# ------------------------------------------------------------------------------------------------ job table
def _hmm_cap(S):
    return 300 if S <= 8 else 600 if S == 12 else 1200 if S == 16 else 1800


def e2a_models(name, tier, exp):
    out = [Job(tier, exp, name, f"qsm-D2-r{r}", i, 300) for r in (1, 2) for i in range(6)]
    out += [Job(tier, exp, name, f"hmm-{S}", i, _hmm_cap(S)) for S in (2, 3, 4, 6, 8, 12, 16) for i in HMM_INITS]
    out += [Job(tier, exp, name, f"markov-{k}", 0, 30) for k in range(1, 9)]
    return out + [Job(tier, exp, name, "truth", 0, 10)]


def jobs_all():
    J = []
    for name in ("flw-T50000-s0", "flw-T50000-s1", "flw-T5000-s0", "flw-T5000-s1", "flw-T5000-s2"):
        J += e2a_models(name, 1, "e2a")
    J += [Job(1, "e2a-exact", "flw-T50000-s0", f"hmm-{S}", i, 1200) for S in (4, 8, 12, 16) for i in ("fit", "lump")]
    # e2b: decisive pair on N = 32 in tier 1
    for name in ("renewal32-s0", "renewal32-s1"):
        J += [Job(1, "e2b", name, "qsm-D4-r1", i, 600) for i in range(6)]
        J += [Job(1, "e2b", name, "hmm-16", i, 1200) for i in HMM_INITS]
        J += [Job(1, "e2b", name, "truth", 0, 10)]
    for name in ("renewal16-s0", "renewal32-s0"):
        J += [Job(2, "e2b", name, "qsm-D2-r1", i, 600) for i in range(4)]
        J += [Job(2, "e2b", name, "qsm-D4-r1", i, 600) for i in range(6)] if name == "renewal16-s0" else []
        J += [Job(2, "e2b", name, f"hmm-{S}", i, _hmm_cap(S)) for S in (2, 4, 8) for i in HMM_INITS]
        J += [Job(2, "e2b", name, "hmm-16", i, 1200) for i in HMM_INITS] if name == "renewal16-s0" else []
        J += [Job(2, "e2b", name, f"markov-{k}", 0, 30) for k in range(1, 9)]
        J += [Job(2, "e2b", name, "truth", 0, 10)] if name == "renewal16-s0" else []
    J += [Job(2, "e2b", "renewal32-s0", "qsm-D8-r1", i, 1200) for i in range(3)]
    J += [Job(2, "e2b", "renewal32-s0", "hmm-32", i, 1800) for i in range(3)]
    # a1 + references
    for name, kind, S, n in (("msar", "anc2", 2, 5), ("smsar16", "anc2", 2, 5), ("smsar16", "anc2", 4, 3),
                             ("msar", "B0", 2, 5), ("smsar16", "B0", 2, 5)):
        J += [Job(1, "a1", name, f"sw-{kind}-S{S}", i, 1800) for i in range(n)]
    for name in SEQ4_DATA:
        J += [Job(1, "reeval", name, mo, "best", 60) for mo in ("cqsm-m1-anc2", "cqsm-m2-anc2", "archb-anc2",
                                                                 "iohmm-fact-2")]
    # a2
    J += [Job(2, "a2", name, "cqsm-m1-anc2-g1", i, 900) for name in SEQ4_DATA for i in range(5)]
    # e1
    for name in ("renewal6-s0", "renewal8-s0"):
        cells = [(4, 16, g) for g in GAMMAS] + ([(2, 4, g) for g in GAMMAS] if name == "renewal8-s0" else [])
        cells += [(4, 1, 0.0)] + ([(2, 1, 0.0)] if name == "renewal8-s0" else [])
        J += [Job(3, "e1", name, f"qsm-D{D}-r{r}" + (f"-g{g:g}" if r > 1 else ""), i, 360) for D, r, g in cells
              for i in range(4)]
        J += [Job(3, "e1", name, f"hmm-{S}", i, 300) for S in ((4, 2) if name == "renewal8-s0" else (4,))
              for i in HMM_INITS]
        J += [Job(3, "e1", name, "truth", 0, 10)]
    J += e2a_models("flw05-T50000-s0", 4, "flw05")
    return J


def _prio(job):
    m = re.search(r"-g([\d.]+)$", job.model)
    return GAMMAS.index(float(m.group(1))) if m else 0


def ordered(jobs):
    return [j for _, j in sorted(enumerate(jobs), key=lambda t: (t[1].tier, -t[1].cap, _prio(t[1]), t[0]))]


def result_dir(job):
    return CFG["out"] / job.exp / job.dataset / job.model / str(job.init)


def deps(job):
    if job.exp == "e2a-exact" and job.init == "fit":
        return [CFG["out"] / "e2a" / job.dataset / job.model / str(i) / "result.json" for i in HMM_INITS]
    return []


# ------------------------------------------------------------------------------------------------ binary fits
def _parse_qsm(model):
    m = re.fullmatch(r"qsm-D(\d+)-r(\d+)(?:-g([\d.]+))?", model)
    return int(m.group(1)), int(m.group(2)), float(m.group(3) or 0.0)


def _adam(n):
    return 0 if CFG["smoke"] else n


def fit_qsm(job, xs, test):
    D, r, g = _parse_qsm(job.model)
    m = int(np.log2(D))
    data = hmm.blocks(xs, 2)
    th0 = np.random.default_rng(job.init).normal(size=qseq.n_params_r(m, 1, r))
    loss = lambda t: -qseq.qsm_r_loglik(t, m, 1, r, g, data)
    res = train.fit_capped(loss, th0, _adam(ADAM_QSM), LBFGS, job.cap, 0.03, 0.003)
    K = qseq.iso_isometry_r(jnp.asarray(res["theta"]), m, 1, r)
    Ms, f = qseq.superops(K, g), qseq.trace_vec(D)
    lp = np.asarray(qseq.sup_logprobs(Ms, f, jnp.asarray(test)))
    info = {"D": D, "r": r, "gamma": g, "m": m, "filter_dim": D * D, "n_params": qseq.n_params_r(m, 1, r),
            "n_effective": qseq.n_effective(m, 1) if r == 1 else None, "adam_ll": -res["adam_loss"],
            "lbfgs_iters": res["lbfgs_iters"], "lbfgs_message": res["lbfgs_message"], "capped": res["capped"]}
    return qseq.sup_linear(Ms, f), lp, -res["final_loss"], info, {"theta": res["theta"], "K": np.asarray(K)}, \
        res["loss"]


def lump_init(name, S):
    """Count-down starting point: renewal N -> consecutive countdown states merged into S groups (pairs for S = N / 2);
    FLW -> the renewal-type hazard chain of the true gap law with S states."""
    spec = DATASETS[name]
    if spec["kind"] == "renewal":
        N = spec["N"]
        return hmm.lump(hmm.renewal(N), np.arange(N) * S // N)
    return hmm.hazard_chain(hmm.gap_dist(truth_lin(spec), 400), S)


def fit_hmm(job, xs, test):
    S, t0 = int(job.model.split("-")[1]), time.time()
    T0 = None if job.init != "lump" else 0.95 * lump_init(job.dataset, S) + 0.05 / (2 * S)
    em = hmm.baum_welch(xs, S, 2, seed=0 if job.init == "lump" else job.init, max_iter=EM_ITERS, T0=T0,
                        max_seconds=0.5 * job.cap)
    T, r = hmm.hmm_polish(em["T"], xs, LBFGS, max_seconds=job.cap - (time.time() - t0))
    info = {"S": S, "filter_dim": S, "n_params": hmm.n_params_hmm(S, 2), "train_ll_fit": -r["final_loss"],
            "em_ll": em["loglik"], "em_iters": em["iters"],
            "em_capped": em["capped"], "lbfgs_iters": r["lbfgs_iters"], "lbfgs_message": r["lbfgs_message"],
            "capped": r["capped"]}
    train_ll = float(np.mean(hmm.hmm_logprobs_abel(T, xs)))  # from the same (Abel) stationary law as the evaluation
    return hmm.hmm_linear_abel(T), hmm.hmm_logprobs_abel(T, test), train_ll, info, {"T": T, "T_em": em["T"]}, \
        np.concatenate([-em["curve"], r["loss"]])


def fit_markov(job, xs, test):
    k = int(job.model.split("-")[1])
    mk = hmm.markov_fit(xs, k, 2)
    info = {"k": k, "filter_dim": 2**k, "n_params": hmm.n_params_markov(k, 2), "capped": False}
    return mk, hmm.markov_logprobs(mk, test), float(np.mean(hmm.markov_logprobs(mk, xs))), info, {"Pc": mk.Pc}, \
        np.zeros(0)


def fit_truth(job, xs, test):
    spec, ref = DATASETS[job.dataset], reference(job.dataset)
    info = {"filter_dim": 4 if spec["kind"] == "flw" else spec["N"], "n_params": None, "capped": False}
    return truth_lin(spec), ref["lp"], ref["train_ll"], info, {}, np.zeros(0)


def best_result(d):
    """(params, init) of the best training likelihood among the result.json under d/<init>/."""
    best = None
    for p in sorted(d.glob("*/result.json")):
        r = json.loads(p.read_text())
        if best is None or r["train_ll"] > best[0]:
            best = (r["train_ll"], p.parent)
    with np.load(best[1] / "params.npz") as z:
        return {k: z[k] for k in z.files}, best[1].name


def fit_exact(job, xs, test):
    S = int(job.model.split("-")[1])
    P16 = reference(job.dataset)["P"][LMAX - 1]
    src = None
    if job.init == "fit":
        prm, src = best_result(CFG["out"] / "e2a" / job.dataset / job.model)
        T0 = prm["T"]
    else:
        T0 = 0.95 * lump_init(job.dataset, S) + 0.05 / (2 * S)
    T, r = hmm.exact_fit(P16, LMAX, T0, EXACT_ITERS, job.cap)
    info = {"S": S, "filter_dim": S, "n_params": hmm.n_params_hmm(S, 2), "objective": "D_16", "D16_objective": r["D_L"],
            "D16_init": r["D_L_init"], "init_from": src, "lbfgs_iters": r["lbfgs_iters"],
            "lbfgs_message": r["lbfgs_message"], "capped": r["capped"]}
    return hmm.hmm_linear_abel(T), hmm.hmm_logprobs_abel(T, test), float(np.mean(hmm.hmm_logprobs_abel(T, xs))), info, \
        {"T": T, "T0": T0}, r["loss"]


def eval_binary(name, mdl, lp, train_ll):
    spec, ref = DATASETS[name], reference(name)
    kl, D, d = hmm.kl_curve(ref["P"], mdl.word_probs(LMAX))
    rate, se = hmm.seg_kl_rate(ref["lp"], lp)
    nmax = 3 * spec["N"] if spec["kind"] == "renewal" else 40
    return {"train_ll": train_ll, "truth_train_ll": ref["train_ll"], "truth_test_ll": float(np.mean(ref["lp"])),
            "heldout_kl_rate": rate, "heldout_se": se, "D16": float(D[-1]), "d8": float(d[7]), "d12": float(d[11]),
            "d16": float(d[15]), "kl": kl.tolist(), "D": D.tolist(), "d": d.tolist(),
            "gap": hmm.gap_dist(mdl, nmax).tolist()}


# ------------------------------------------------------------------------------------------------ seq4 fits
def _stage2(loss, th0, cap):
    return train.fit_capped(loss, jnp.asarray(th0), _adam(ADAM_STAGE2), LBFGS, cap, 0.003, 0.001)


def fit_switch(job, d):
    _, kind, s = job.model.split("-")
    S = int(s[1:])
    prm, i1 = best_result(SEQ4 / job.dataset / f"archb-{kind}")
    th0 = switch.init_switch(prm["theta"], S, job.init, kind)
    Xj = jnp.asarray(d["X"])
    loss = lambda t: -switch.loglik(t, Xj, S, kind)
    res = _stage2(loss, th0, job.cap)
    info = {"S": S, "kernel": kind, "filter_dim": S, "n_params": switch.n_params(S, kind), "stage1_init": i1,
            "init_ll": float(-loss(jnp.asarray(th0))), "adam_ll": -res["adam_loss"], "lbfgs_iters": res["lbfgs_iters"],
            "lbfgs_message": res["lbfgs_message"], "capped": res["capped"]}
    return switch.io(res["theta"], S, kind), -res["final_loss"], info, {"theta": res["theta"]}, res["loss"]


def fit_cqsm_dephased(job, d):
    m, a, g = 1, 2, float(job.model.split("-g")[1])
    prm, i1 = best_result(SEQ4 / job.dataset / "archb-anc2")
    th0 = cqsm.embed(prm["theta"], job.init, m, a, 3)
    Xj = jnp.asarray(d["X"])
    loss = lambda t: -cqsm.loglik(t, Xj, m, a, 3, gamma=g)
    res = _stage2(loss, th0, job.cap)
    info = {"D": 2**m, "n_ancillas": a, "gamma": g, "filter_dim": 4**m, "n_params": cqsm.n_params(m, a, 3),
            "stage1_init": i1, "init_ll": float(-loss(jnp.asarray(th0))), "adam_ll": -res["adam_loss"],
            "lbfgs_iters": res["lbfgs_iters"], "lbfgs_message": res["lbfgs_message"], "capped": res["capped"],
            "mem_angle_rms": float(np.sqrt(np.mean(res["theta"][cqsm.mem_mask(m, a, 3)] ** 2)))}
    return cqsm.io(res["theta"], m, a, 3, gamma=g), -res["final_loss"], info, {"theta": res["theta"]}, res["loss"]


def fit_reeval(job, d):
    """A seq4 fit (best training likelihood) through the same code path as the new fits (gamma = 0)."""
    src = SEQ4 / job.dataset / job.model
    prm, i = best_result(src)
    old = json.loads((src / i / "result.json").read_text())
    if job.model.startswith("iohmm"):
        lin, filt = iolin.IOLin.from_iohmm(prm["T"]), int(job.model.split("-")[2])
    else:
        m = int(job.model.split("-")[1][1:]) if job.model.startswith("cqsm") else 0
        a = 2 if job.model.endswith("anc2") else 0
        lin, filt = cqsm.io(prm["theta"], m, a, 3, gamma=0.0), 4**m
    info = {"seq4_init": i, "seq4_train_ll": old["train_ll"], "seq4_heldout_kl_rate": old["heldout_kl_rate"],
            "n_params": old.get("n_params"), "filter_dim": filt, "capped": False}
    return lin, lin.path_loglik(d["X"]), info, {}, np.zeros(0)


def eval_seq4(name, lin, train_ll, d):
    ref = reference(name)
    lp = lin.logprobs(d["test"])
    rate, se = hmm.seg_kl_rate(ref["lp"], lp)
    return lp, {"train_ll": train_ll, "truth_train_ll": ref["train_ll"], "truth_test_ll": float(np.mean(ref["lp"])),
                "heldout_kl_rate": rate, "heldout_se": se}


def heldout_logprobs(r, d):
    """Per-step held-out log-probabilities of the stored fit in d (result dict r), recomputed from params.npz by the
    code path of fit(); the summaries call it when lp_test.npy (written by the fit, not tracked) is missing."""
    job = Job(0, r["exp"], r["dataset"], r["model"], r["init"], 0)
    prm = dict(np.load(d / "params.npz")) if (d / "params.npz").exists() else {}
    if job.dataset in SEQ4_DATA:
        data = load_seq4(job.dataset)
        if job.exp == "reeval":
            lin = fit_reeval(job, data)[0]
        elif job.exp == "a1":
            _, kind, s = job.model.split("-")
            lin = switch.io(prm["theta"], int(s[1:]), kind)
        else:
            lin = cqsm.io(prm["theta"], 1, 2, 3, gamma=float(job.model.split("-g")[1]))
        return np.asarray(lin.logprobs(data["test"]))
    xs, test = load_binary(job.dataset)
    kind = job.model.split("-")[0]
    if kind == "qsm":
        D, _, g = _parse_qsm(job.model)
        Ms = qseq.superops(jnp.asarray(prm["K"]), g)
        return np.asarray(qseq.sup_logprobs(Ms, qseq.trace_vec(D), jnp.asarray(test)))
    if kind == "hmm":
        return hmm.hmm_logprobs_abel(prm["T"], test)
    if kind == "markov":
        return hmm.markov_logprobs(hmm.markov_fit(xs, int(job.model.split("-")[1]), 2), test)
    return reference(job.dataset)["lp"]


# ------------------------------------------------------------------------------------------------ one job
def fit(job):
    """-> (result dict, params, held-out log-probs, loss curve)."""
    base = {"exp": job.exp, "dataset": job.dataset, "model": job.model, "init": job.init, "cap_seconds": job.cap}
    if job.dataset in SEQ4_DATA:
        d = load_seq4(job.dataset)
        f = {"a1": fit_switch, "a2": fit_cqsm_dephased, "reeval": fit_reeval}[job.exp]
        lin, train_ll, info, params, loss = f(job, d)
        lp, ev = eval_seq4(job.dataset, lin, train_ll, d)
        return {**base, **SEQ4_DATA_SPEC(job.dataset), **info, **ev}, params, lp, loss
    xs, test = load_binary(job.dataset)
    if job.exp == "e2a-exact":
        f = fit_exact
    else:
        f = {"qsm": fit_qsm, "hmm": fit_hmm, "markov": fit_markov, "truth": fit_truth}[job.model.split("-")[0]]
    mdl, lp, train_ll, info, params, loss = f(job, xs, test)
    return {**base, **DATASETS[job.dataset], **info, **eval_binary(job.dataset, mdl, lp, train_ll)}, params, lp, loss


def SEQ4_DATA_SPEC(name):
    kind, N, seed = SEQ4_DATA[name]
    return {"kind": kind, "N": N, "seed": seed}


def _lock(d):
    lock = d / "lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0)
            return False
        except (ValueError, ProcessLookupError):
            lock.write_text(str(os.getpid()))  # stale lock of a killed worker
            return True


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
    """-> (job, status, seconds, message). status: done (result existed), locked, ran, error."""
    if CFG["smoke"]:
        job = job._replace(cap=min(job.cap, SMOKE_CAP))
    d = result_dir(job)
    if (d / "result.json").exists():
        return job, "done", 0.0, ""
    d.mkdir(parents=True, exist_ok=True)
    if not _lock(d):
        return job, "locked", 0.0, ""
    t0, status, msg = time.time(), "error", ""
    with _log_to(CFG["out"] / "logs" / f"{job.exp}__{job.dataset}__{job.model}__{job.init}.log"):
        print(f"{datetime.now():%H:%M:%S} start {job} pid {os.getpid()}", flush=True)
        try:
            res, params, lp, loss = fit(job)
            res["wall_time"] = time.time() - t0
            np.savez(d / "params.npz", **params)
            np.save(d / "loss.npy", np.asarray(loss))
            np.save(d / "lp_test.npy", np.asarray(lp, dtype=np.float32))
            tmp = d / "result.json.tmp"
            tmp.write_text(json.dumps(res, indent=1, default=float) + "\n")
            os.replace(tmp, d / "result.json")
            status = "ran"
            msg = (f"train ll {res['train_ll']:.6f} (truth {res['truth_train_ll']:.6f}) held-out KL {res['heldout_kl_rate']:.3e}"
                   f" ± {res['heldout_se']:.1e}" + (" CAPPED" if res.get("capped") else ""))
            print(msg, flush=True)
        except Exception:
            traceback.print_exc()
            msg = traceback.format_exc().strip().splitlines()[-1]
    (d / "lock").unlink(missing_ok=True)
    return job, status, time.time() - t0, msg


# ------------------------------------------------------------------------------------------------ main
def smoke_selection(jobs):
    """One job per (experiment, model type), never a job that waits for others."""
    seen, out = set(), []
    for j in jobs:
        key = (j.exp, re.sub(r"\d+(\.\d+)?", "#", j.model))
        if key not in seen and not deps(j):
            seen.add(key)
            out.append(j)
    return out


def list_jobs(jobs):
    rows = {}
    for j in jobs:
        r = rows.setdefault((j.tier, j.exp), [0, 0, 0.0])
        r[0] += 1
        if not (result_dir(j) / "result.json").exists():
            r[1] += 1
            r[2] += j.cap / 3600
    print(f"{'tier':>4} {'experiment':<10} {'jobs':>5} {'to do':>6} {'cap CPU-h (to do)':>18}")
    for (t, e), (n, todo, h) in sorted(rows.items()):
        print(f"{t:>4} {e:<10} {n:>5} {todo:>6} {h:>18.1f}")
    tot = np.sum([r for r in rows.values()], axis=0)
    print(f"{'':>4} {'total':<10} {int(tot[0]):>5} {int(tot[1]):>6} {tot[2]:>18.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None, help="experiments (default: all)")
    ap.add_argument("--tiers", nargs="*", type=int, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="model name prefixes")
    ap.add_argument("--inits", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output directory (default runs/seq5)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=1e9, help="start no new fit after this many minutes")
    ap.add_argument("--stop-at", default=None, help="start no new fit after this clock time (HH:MM, next occurrence)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one fit per type, caps <= 20 s, no Adam, -> runs/seq5_smoke/")
    a = ap.parse_args()
    if a.smoke or a.out:
        configure(Path(a.out) if a.out else ROOT / "runs" / "seq5_smoke", smoke=a.smoke)
    jobs = [j for j in ordered(jobs_all()) if (a.exps is None or j.exp in a.exps)
            and (a.tiers is None or j.tier in a.tiers) and (a.datasets is None or j.dataset in a.datasets)
            and (a.models is None or any(j.model.startswith(p) for p in a.models))
            and (a.inits is None or str(j.init) in a.inits)]
    if a.smoke:
        jobs = smoke_selection(jobs)
    if a.list:
        list_jobs(jobs)
        return
    t0 = time.time()
    deadline = t0 + 60 * a.max_minutes
    if a.stop_at:
        h, m = map(int, a.stop_at.split(":"))
        now = datetime.now()
        stop = now.replace(hour=h, minute=m, second=0, microsecond=0)
        deadline = min(deadline, stop.timestamp() + (86400 if stop <= now else 0))
    pending = [j for j in jobs if not (result_dir(j) / "result.json").exists()]
    print(f"{datetime.now():%H:%M:%S} {len(jobs)} fits, {len(pending)} to do, {a.workers} workers, out {CFG['out']}",
          flush=True)
    prepare_data(sorted({j.dataset for j in pending if j.dataset in DATASETS}))
    running, n_done = {}, 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=configure, initargs=(CFG["out"], CFG["smoke"]), maxtasksperchild=8) as pool:
        while True:
            stopping = time.time() > deadline
            if not stopping:
                for j in list(pending):
                    if len(running) >= a.workers:
                        break
                    if all(p.exists() for p in deps(j)):
                        pending.remove(j)
                        running[j] = (pool.apply_async(run_job, (j,)), time.time())
            for j in [j for j, (r, _) in running.items() if r.ready()]:
                r, _ = running.pop(j)
                try:
                    _, st, wt, msg = r.get()
                except Exception as e:  # noqa: BLE001
                    st, wt, msg = "crash", 0.0, repr(e)
                n_done += st == "ran"
                print(f"{datetime.now():%H:%M:%S} [{st}] {j.exp} {j.dataset} {j.model} {j.init}: {msg} [{wt:.0f} s]",
                      flush=True)
            for j in [j for j, (_, ts) in running.items() if time.time() - ts > 2 * j.cap + 900]:
                running.pop(j)
                print(f"{datetime.now():%H:%M:%S} [lost] {j}: no answer after {2 * j.cap + 900} s", flush=True)
            if not running:
                if stopping and pending:
                    print(f"time budget reached: {len(pending)} fits not started", flush=True)
                    break
                if not pending:
                    break
                if not any(all(p.exists() for p in deps(j)) for j in pending):
                    print(f"{len(pending)} fits wait for missing results, skipped: {pending}", flush=True)
                    break
            time.sleep(1)
    print(f"{datetime.now():%H:%M:%S} {n_done} fits ran in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
