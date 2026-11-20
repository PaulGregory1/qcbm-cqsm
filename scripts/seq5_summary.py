"""Sequence 5: aggregate runs/seq5/** into runs/seq5/summary.json and runs/seq5/tables.md, print the four
tables and draw figs/seq5_*.png. Read-only on the run directories; works on partial results (missing cells: "–").

Selection: the reported fit of a cell (dataset, model) is the init with the best TRAINING log-likelihood (exact-law
arm: the lowest D_16 objective), never selected on the truth.
Paired comparisons: per-step held-out log-probabilities (lp_test.npy) of the two reported fits on the same held-out
string; KL_A - KL_B = mean(lp_B - lp_A) with the standard error over 20 segments; "A beats B by > 2 SE" is one-sided.
Outcome labels follow plan_v3 mechanically: Positive / Negative / Inconclusive, "non-decisive (capped)" when a fit
the rule depends on hit its wall-clock cap (HMM_32 in b3 also when L-BFGS stopped at its iteration limit), "–" when a
cell is missing.

T1  E2a (FLW): per dataset the reported fits, Hankel ranks of the truth, decisive QSM_2 r = 1 vs HMM_2 (equal memory)
    and HMM_4 (equal filter cost) at T = 5e4, S* (parsimony, finite sample), practical relevance (HMM_8 KL < 1e-4),
    exact-law arm (D_16, d_8, d_12, d_16, held-out rate, S*_approx, horizon artefact, outcome).
T2  E2b (renewal): b1 QSM_4 vs HMM_16 at N = 32 (both seeds), b2 control at N = 16, b3 HMM_32 vs QSM_8.
T3  A1 switch / A2 dephased C-QSM, every variant its own row; outcome per pre-registered pair against the re-evaluated
    seq4 C-QSM m = 1 + 2 anc (equal memory: switch anc2 S = 2; equal filter cost: anc2 S = 4; parameter-matched: B0
    S = 2; A2: gamma = 1 vs gamma = 0); C-QSM m = 2 + 2 anc, Architecture B anc2 and IOHMM_2 for reference.
T4  E1: KL(gamma), train - truth per gamma, monotonicity on train NLL (paired on the training string, per-step
    log-probabilities recomputed from params.npz), gamma_1/2, validity (gamma = 1 vs HMM_D), r = D^2
    vs r = 1.

Run: PYTHONPATH=src ./.venv/bin/python scripts/seq5_summary.py [--runs DIR] [--figs DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
DPI = 170
FLW_S = (2, 3, 4, 6, 8, 12, 16)
GAMMAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
DASH = "–"


# ------------------------------------------------------------------------------- loading and selection
def load(runs):
    R = {}
    for p in sorted(runs.glob("*/*/*/*/result.json")):
        r = json.loads(p.read_text())
        r["_dir"] = str(p.parent)
        R.setdefault(r["exp"], {}).setdefault(r["dataset"], {}).setdefault(r["model"], []).append(r)
    return R


def _ll(r):
    v = r.get("train_ll")
    return -np.inf if v is None or not np.isfinite(v) else v


def best(R, exp, ds, model):
    rs = R.get(exp, {}).get(ds, {}).get(model)
    if not rs:
        return None
    if exp == "e2a-exact":
        return min(rs, key=lambda r: r.get("D16_objective", np.inf))
    return max(rs, key=_ll)


def n_inits(R, exp, ds, model):
    return len(R.get(exp, {}).get(ds, {}).get(model, []))


def capped(r):
    return bool(r and r.get("capped"))


def iter_limit(r):
    return bool(r and "ITERATIONS REACHED LIMIT" in str(r.get("lbfgs_message", "")))


_LP = {}


def lp(r):
    if r["_dir"] not in _LP:
        p = Path(r["_dir"]) / "lp_test.npy"
        if not p.exists():  # not tracked: recompute from params.npz as the fit did
            sys.path.insert(0, str(ROOT / "scripts"))
            import run_seq5

            np.save(p, np.asarray(run_seq5.heldout_logprobs(r, p.parent), dtype=np.float32))
        _LP[r["_dir"]] = np.load(p).astype(np.float64)
    return _LP[r["_dir"]]


def paired(a, b, n_seg=20):
    """KL_a - KL_b on the common held-out string and its 20-segment SE (None if a fit is missing)."""
    if a is None or b is None or lp(a) is None or lp(b) is None:
        return None
    d = lp(b) - lp(a)
    n = len(d) // n_seg * n_seg
    seg = d[:n].reshape(n_seg, -1).mean(1)
    return float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(n_seg))


_TRAIN = {}


def train_lp(r):
    """Per-step training log-probabilities of a QSM fit (E1), recomputed from its params.npz and the cached training
    string (runs/seq5/data/<dataset>.npz)."""
    if r["_dir"] not in _TRAIN:
        import jax.numpy as jnp

        from archb import qseq

        d = Path(r["_dir"])
        prm, data = d / "params.npz", d.parents[3] / "data" / f"{r['dataset']}.npz"
        if not (prm.exists() and data.exists()):
            _TRAIN[r["_dir"]] = None
        else:
            with np.load(prm) as z, np.load(data) as x:
                K, xs = z["K"], x["train"]
            Ms = qseq.superops(jnp.asarray(K), r["gamma"])
            _TRAIN[r["_dir"]] = np.asarray(qseq.sup_logprobs(Ms, qseq.trace_vec(K.shape[-1]), jnp.asarray(xs)))
    return _TRAIN[r["_dir"]]


def train_paired(a, b, n_seg=20):
    """Training NLL_b - NLL_a and its 20-segment SE (paired on the training string)."""
    la, lb = train_lp(a), train_lp(b)
    if la is None or lb is None:
        return None
    d = la - lb
    n = len(d) // n_seg * n_seg
    seg = d[:n].reshape(n_seg, -1).mean(1)
    return float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(n_seg))


def excess(r):
    """Training NLL minus the truth's (nats per symbol)."""
    return None if r is None or r.get("train_ll") is None else r["truth_train_ll"] - r["train_ll"]


# ------------------------------------------------------------------------------- formatting
def f(x, spec=".2e"):
    return DASH if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def kl_se(r):
    return DASH if r is None else f"{r['heldout_kl_rate']:.2e} ± {r['heldout_se']:.1e}"


def flags(r):
    return ("capped " if capped(r) else "") + ("it-limit" if iter_limit(r) else "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def cell(r):
    """Compact record of a reported fit for summary.json."""
    if r is None:
        return None
    keys = ("init", "train_ll", "truth_train_ll", "heldout_kl_rate", "heldout_se", "D16", "d8", "d12", "d16",
            "n_params", "filter_dim", "capped", "lbfgs_message", "D16_objective", "wall_time")
    return {k: r.get(k) for k in keys if k in r} | {"train_minus_truth": excess(r), "iter_limit": iter_limit(r)}


# ------------------------------------------------------------------------------- outcome rules
def decisive(pairs):
    """pairs: [(quantum fit, classical fit)] over seeds. Positive: the quantum fit beats the classical one by > 2 SE
    (paired) in every seed AND has the better training NLL; Negative: classical better or within 2 SE in every seed."""
    if any(q is None or h is None for q, h in pairs):
        return DASH, []
    if any(capped(q) or capped(h) for q, h in pairs):
        return "non-decisive (capped)", []
    det, beats, hmm_ok = [], [], []
    for q, h in pairs:
        p = paired(h, q)  # KL_h - KL_q
        if p is None:
            return DASH, []
        diff, se = p
        det.append({"KL_classical_minus_quantum": diff, "se": se, "quantum_better_train": q["train_ll"] > h["train_ll"]})
        beats.append(diff > 2 * se and q["train_ll"] > h["train_ll"])
        hmm_ok.append(diff <= 2 * se)
    return ("Positive" if all(beats) else "Negative" if all(hmm_ok) else "Inconclusive"), det


def ratio_outcome(new, ref, lo, hi, labels):
    """new / ref KL ratio rule (A1, A2): ratio <= lo -> labels[0]; ratio >= hi and new worse by > 2 SE -> labels[1]."""
    if new is None or ref is None:
        return DASH, None, None
    if capped(new) or capped(ref):
        return "non-decisive (capped)", None, None
    ratio = new["heldout_kl_rate"] / ref["heldout_kl_rate"]
    p = paired(new, ref)
    if p is None:
        return DASH, ratio, None
    if ratio <= lo:
        return labels[0], ratio, p
    if ratio >= hi and p[0] > 2 * p[1]:
        return labels[1], ratio, p
    return "Inconclusive", ratio, p


# ------------------------------------------------------------------------------- T1 E2a
def t1(R, runs):
    out, md = {}, ["## T1  E2a: FLW qubit process (lam = 0.8) against HMMs"]
    dsets = sorted(R.get("e2a", {}), key=lambda d: (-int(d.split("-T")[1].split("-")[0]), d))
    for ds in dsets:
        q = best(R, "e2a", ds, "qsm-D2-r1")
        h = runs / "hankel" / f"{ds}.json"
        hk = json.loads(h.read_text()) if h.exists() else {}
        rows, cells = [], {}
        models = ["truth", "qsm-D2-r1", "qsm-D2-r2"] + [f"hmm-{S}" for S in FLW_S] + [f"markov-{k}" for k in range(1, 9)]
        for mo in models:
            r = best(R, "e2a", ds, mo)
            cells[mo] = cell(r)
            pq = paired(r, q) if mo != "qsm-D2-r1" else None
            rows.append([mo, f"{n_inits(R, 'e2a', ds, mo)}", kl_se(r), f(excess(r)), f(r and r.get("d16")),
                         DASH if pq is None else f"{pq[0]:+.2e} ± {pq[1]:.1e}", f(r and r.get("n_params"), "d"),
                         flags(r)])
        # S*: smallest S whose reported fit has KL_HMM - eps < 2 SE (paired), eps = KL of QSM_2 r = 1
        s_star, complete = None, q is not None
        for S in FLW_S:
            p = paired(best(R, "e2a", ds, f"hmm-{S}"), q)
            if p is None:
                complete = False
                continue
            if p[0] < 2 * p[1]:
                s_star = S
                break
        s_txt = (str(s_star) if complete else f"<= {s_star} (smaller S missing)") if s_star else \
            (">16" if complete else DASH)
        h8 = best(R, "e2a", ds, "hmm-8")
        rel = None if h8 is None else bool(h8["heldout_kl_rate"] < 1e-4)
        out[ds] = {"cells": cells, "S_star": s_txt, "practical_relevance_hmm8_below_1e-4": rel,
                   "hankel_rank_words": hk.get("rank_words_le6"), "hankel_rank_gap": hk.get("rank_gap_law_64")}
        md += [f"\n### {ds}  (Hankel rank of the truth: words <= 6 {hk.get('rank_words_le6', DASH)}, gap law "
               f"{hk.get('rank_gap_law_64', DASH)}; OOM_3 exact, QSM_2 r = 1 exact, no finite HMM)",
               table(["model", "inits", "held-out KL ± SE", "train − truth", "d_16", "KL − KL(QSM_2) ± SE (paired)",
                      "params", "flags"], rows),
               f"\nS* (parsimony, finite sample: smallest S with KL_HMM − ε < 2 SE) = {s_txt}; practical relevance "
               f"(HMM_8 KL < 1e-4): {DASH if rel is None else rel}"]
    # decisive comparisons at T = 5e4
    dec = {}
    md.append("\n### Decisive (T = 5e4, seeds 0 and 1): QSM_2 r = 1 vs HMM_k")
    rows = []
    for k, lab in ((2, "equal memory"), (4, "equal filter cost")):
        pairs = [(best(R, "e2a", f"flw-T50000-s{s}", "qsm-D2-r1"), best(R, "e2a", f"flw-T50000-s{s}", f"hmm-{k}"))
                 for s in (0, 1)]
        lab_out, det = decisive(pairs)
        dec[f"hmm-{k}"] = {"outcome": lab_out, "per_seed": det}
        rows.append([f"HMM_{k} ({lab})", lab_out] + [f"{d['KL_classical_minus_quantum']:+.2e} ± {d['se']:.1e}"
                                                     f"{'' if d['quantum_better_train'] else ' (QSM worse train)'}"
                                                     for d in det] + [DASH] * (2 - len(det)))
    md.append(table(["comparison", "outcome", "KL_HMM − KL_QSM seed 0", "seed 1"], rows))
    out["decisive"] = dec
    out["exact"] = t1_exact(R, md)
    return out, md


def t1_exact(R, md):
    ds, rows, cells, reach = "flw-T50000-s0", [], {}, {}
    for S in (4, 8, 12, 16):
        r = best(R, "e2a-exact", ds, f"hmm-{S}")
        cells[S] = cell(r)
        if r is not None:
            reach[S] = r["heldout_kl_rate"] < 1e-4 + 2 * r["heldout_se"]
        art = r is not None and r["d16"] < 1e-4 and r["heldout_kl_rate"] >= 1e-4 + 2 * r["heldout_se"]
        binds = r is not None and r["d16"] > r["d12"]
        rows.append([f"hmm-{S}", f(r and r.get("init"), ""), f(r and r.get("D16")), f(r and r.get("d8")),
                     f(r and r.get("d12")), f(r and r.get("d16")), kl_se(r),
                     ("horizon artefact " if art else "") + ("d_L rising at 16 " if binds else "") + flags(r)])
    s_approx = next((S for S in (4, 8, 12, 16) if reach.get(S)), None)
    allin = len(reach) == 4
    s_txt = str(s_approx) if s_approx else (">16" if allin else DASH)
    h4, h8 = best(R, "e2a-exact", ds, "hmm-4"), best(R, "e2a-exact", ds, "hmm-8")
    if h8 is None or h4 is None:
        oc = DASH
    elif capped(h8) or capped(h4):
        oc = "Inconclusive (decisive fit capped)"
    elif reach.get(8) or reach.get(4):
        oc = "Negative"
    else:
        oc = "Positive"
    d16_8 = None if h8 is None else bool(h8["d16"] >= 1e-4)
    md += ["\n### E2a exact-law arm (HMM_S fitted to the exact law of 16-symbol words; QSM_2 r = 1 reaches 0)",
           table(["model", "init", "D_16", "d_8", "d_12", "d_16", "held-out KL ± SE", "flags"], rows),
           f"\nS*_approx (smallest S with held-out rate < 1e-4 within 2 SE) = {s_txt}; HMM_8 d_16 >= 1e-4: "
           f"{DASH if d16_8 is None else d16_8}; outcome: **{oc}**"]
    return {"cells": cells, "S_star_approx": s_txt, "outcome": oc, "hmm8_d16_ge_1e-4": d16_8}


# ------------------------------------------------------------------------------- T2 E2b
def t2(R):
    out, md = {}, ["\n## T2  E2b: uniform renewal at equal filter cost"]
    models = ["truth", "qsm-D2-r1", "qsm-D4-r1", "qsm-D8-r1"] + [f"hmm-{S}" for S in (2, 4, 8, 16, 32)] + \
        [f"markov-{k}" for k in range(1, 9)]
    for ds in ("renewal16-s0", "renewal32-s0", "renewal32-s1"):
        if ds not in R.get("e2b", {}):
            continue
        rows, cells = [], {}
        for mo in models:
            r = best(R, "e2b", ds, mo)
            if r is None and n_inits(R, "e2b", ds, mo) == 0 and mo in ("qsm-D8-r1", "hmm-32"):
                continue
            cells[mo] = cell(r)
            rows.append([mo, n_inits(R, "e2b", ds, mo), kl_se(r), f(excess(r)), f(r and r.get("d16")),
                         f(r and r.get("n_params"), "d"), flags(r)])
        out[ds] = cells
        md += [f"\n### {ds}", table(["model", "inits", "held-out KL ± SE", "train − truth", "d_16", "params", "flags"],
                                    rows)]
    b1, det = decisive([(best(R, "e2b", f"renewal32-s{s}", "qsm-D4-r1"), best(R, "e2b", f"renewal32-s{s}", "hmm-16"))
                        for s in (0, 1)])
    q16, h16 = best(R, "e2b", "renewal16-s0", "qsm-D4-r1"), best(R, "e2b", "renewal16-s0", "hmm-16")
    p = paired(q16, h16)  # KL_QSM - KL_HMM
    b2 = DASH if p is None else ("control OK (HMM_16 beats QSM_4)" if p[0] > 2 * p[1] else
                                 "FLAG: HMM_16 does not beat QSM_4 (optimizer failure?)")
    q8, h32 = best(R, "e2b", "renewal32-s0", "qsm-D8-r1"), best(R, "e2b", "renewal32-s0", "hmm-32")
    p3 = paired(q8, h32)
    if p3 is None:
        b3 = DASH
    elif capped(h32) or iter_limit(h32) or capped(q8):
        b3 = "non-decisive (" + ("capped" if capped(h32) or capped(q8) else "HMM_32 not converged: iteration limit") + ")"
    else:
        b3 = "prediction holds (HMM_32 beats QSM_8)" if p3[0] > 2 * p3[1] else "prediction fails"
    md += ["\n### Outcomes",
           table(["test", "outcome", "detail"],
                 [["b1 QSM_4 vs HMM_16, N = 32 (decisive)", f"**{b1}**",
                   "; ".join(f"seed {s}: KL_HMM − KL_QSM = {d['KL_classical_minus_quantum']:+.2e} ± {d['se']:.1e}"
                             for s, d in enumerate(det)) or DASH],
                  ["b2 control N = 16", b2, DASH if p is None else f"KL_QSM − KL_HMM = {p[0]:+.2e} ± {p[1]:.1e}"],
                  ["b3 HMM_32 vs QSM_8, N = 32", b3, DASH if p3 is None else f"KL_QSM − KL_HMM = {p3[0]:+.2e} ± {p3[1]:.1e}"]])]
    out["b1"], out["b1_per_seed"], out["b2"], out["b3"] = b1, det, b2, b3
    return out, md


# ------------------------------------------------------------------------------- T3 ablations
A1_MODELS = [("reeval", "cqsm-m1-anc2", "C-QSM m=1 +2 anc (seq4)"), ("reeval", "cqsm-m2-anc2", "C-QSM m=2 +2 anc (seq4)"),
             ("a2", "cqsm-m1-anc2-g1", "C-QSM m=1 +2 anc, dephased γ=1"), ("a1", "sw-anc2-S2", "switch anc2 S=2"),
             ("a1", "sw-anc2-S4", "switch anc2 S=4"), ("a1", "sw-B0-S2", "switch B0 S=2"),
             ("reeval", "iohmm-fact-2", "IOHMM_2 factored (seq4)"), ("reeval", "archb-anc2", "Arch. B anc2 (seq4)")]


A1_PAIRS = (("sw-anc2-S2", "equal memory (switch anc2 S = 2)"), ("sw-anc2-S4", "equal filter cost (switch anc2 S = 4)"),
            ("sw-B0-S2", "parameter-matched (switch B0 S = 2)"))


def t3(R):
    """Every variant its own row (reported fit = best training likelihood within the cell); the reference of every
    pre-registered pair is the re-evaluated C-QSM m = 1 + 2 anc (m = 2 + 2 anc shown for reference only)."""
    out, md = {}, ["\n## T3  Ablations of the C-QSM memory (A1 switch, A2 dephasing)"]
    for ds in ("msar", "smsar16"):
        ref = best(R, "reeval", ds, "cqsm-m1-anc2")
        rows, cells = [], {}
        for exp, mo, lab in A1_MODELS:
            r = best(R, exp, ds, mo)
            if r is None and mo == "sw-anc2-S4" and ds == "msar":
                continue
            cells[mo] = cell(r)
            p = paired(r, ref) if r is not None and r is not ref else None
            rows.append([lab, n_inits(R, exp, ds, mo), kl_se(r),
                         DASH if r is None or ref is None else f"{r['heldout_kl_rate'] / ref['heldout_kl_rate']:.2f}",
                         DASH if p is None else f"{p[0]:+.2e} ± {p[1]:.1e}", f(r and r.get("n_params"), "d"),
                         f(r and r.get("filter_dim"), "d"), flags(r)])
        a1, orows = {}, []
        for mo, lab in A1_PAIRS:
            if mo == "sw-anc2-S4" and ds == "msar":
                continue
            o = ratio_outcome(best(R, "a1", ds, mo), ref, 1.1, 1.5, ("Negative for quantum memory", "Positive"))
            a1[mo] = {"pair": lab, "outcome": o[0], "ratio": o[1], "paired": o[2]}
            orows.append([f"A1 {lab} vs C-QSM m = 1 + 2 anc", f"**{o[0]}**", f(o[1], ".2f"),
                          DASH if o[2] is None else f"{o[2][0]:+.2e} ± {o[2][1]:.1e}"])
        a2 = ratio_outcome(best(R, "a2", ds, "cqsm-m1-anc2-g1"), ref, 1.1, 1.5,
                           ("uses its memory classically", "uses coherence"))
        orows.append(["A2 dephased γ = 1 vs γ = 0 (C-QSM m = 1 + 2 anc)", f"**{a2[0]}**", f(a2[1], ".2f"),
                      DASH if a2[2] is None else f"{a2[2][0]:+.2e} ± {a2[2][1]:.1e}"])
        out[ds] = {"cells": cells, "reference": "cqsm-m1-anc2 (re-evaluated seq4, best training likelihood)",
                   "A1": a1, "A2": {"outcome": a2[0], "ratio": a2[1], "paired": a2[2]}}
        md += [f"\n### {ds}  (reference: re-evaluated seq4 C-QSM m = 1 + 2 anc)",
               table(["model", "inits", "held-out KL ± SE", "ratio to C-QSM m=1", "KL − KL(C-QSM m=1) ± SE (paired)",
                      "params", "filter dim", "flags"], rows),
               "", table(["pre-registered pair", "outcome", "KL ratio", "KL difference ± SE (paired)"], orows)]
    return out, md


# ------------------------------------------------------------------------------- T4 E1
def t4(R):
    out, md = {}, ["\n## T4  E1: coherence knob (dephasing gamma) on the QSM memory"]
    for ds, D, r in (("renewal6-s0", 4, 16), ("renewal8-s0", 4, 16), ("renewal8-s0", 2, 4)):
        fits = {g: best(R, "e1", ds, f"qsm-D{D}-r{r}-g{g:g}") for g in GAMMAS}
        r1, hD = best(R, "e1", ds, f"qsm-D{D}-r1"), best(R, "e1", ds, f"hmm-{D}")
        if all(v is None for v in fits.values()) and r1 is None:
            continue
        rows = []
        for g in GAMMAS:
            x = fits[g]
            rows.append([f"{g:g}", n_inits(R, "e1", ds, f"qsm-D{D}-r{r}-g{g:g}"), kl_se(x), f(excess(x)), flags(x)])
        rows += [[f"r = 1, γ = 0", n_inits(R, "e1", ds, f"qsm-D{D}-r1"), kl_se(r1), f(excess(r1)), flags(r1)],
                 [f"HMM_{D}", n_inits(R, "e1", ds, f"hmm-{D}"), kl_se(hD), f(excess(hD)), flags(hD)]]
        # validity: gamma = 1 reproduces HMM_D within 2 SE (paired)
        pv = paired(fits[1.0], hD)
        valid = None if pv is None else bool(abs(pv[0]) <= 2 * pv[1])
        # monotonicity on train NLL: nested classes, the optimal train NLL cannot decrease with gamma
        viol = []
        have = [g for g in sorted(GAMMAS) if fits[g] is not None]
        for g0, g1 in zip(have[:-1], have[1:]):
            a, b = fits[g0], fits[g1]
            p = train_paired(a, b)  # NLL_b - NLL_a on the training string, 20-segment SE
            if p is not None and p[0] < -(2 * p[1] + 0.1 * max(excess(a), 0.0)):
                viol.append(f"{g0:g}->{g1:g} ({p[0]:+.1e} ± {p[1]:.0e})")
        # gamma_1/2: log(KL(gamma) / KL_HMM_D) reaches half its gamma = 0 value (linear interpolation on the grid)
        gh = None
        if hD is not None and fits[0.0] is not None:
            target = 0.5 * np.log(fits[0.0]["heldout_kl_rate"] / hD["heldout_kl_rate"])
            pts = [(g, np.log(fits[g]["heldout_kl_rate"] / hD["heldout_kl_rate"])) for g in sorted(GAMMAS)
                   if fits[g] is not None]
            for (ga, la), (gb, lb) in zip(pts[:-1], pts[1:]):
                if (la - target) * (lb - target) <= 0 and la != lb:
                    gh = ga + (target - la) * (gb - ga) / (lb - la)
                    break
        pr = paired(fits[0.0], r1)  # KL(r = D^2) - KL(r = 1) at gamma = 0
        dtr = None if fits[0.0] is None or r1 is None else excess(fits[0.0]) - excess(r1)
        key = f"{ds} D={D} r={r}"
        out[key] = {"cells": {f"{g:g}": cell(fits[g]) for g in GAMMAS}, "r1": cell(r1), f"hmm{D}": cell(hD),
                    "valid_gamma1_eq_hmm": valid, "monotonicity_violations": viol, "gamma_half": gh,
                    "rD2_minus_r1_heldout": pr, "rD2_minus_r1_train_nll": dtr}
        md += [f"\n### {key}", table(["γ", "inits", "held-out KL ± SE", "train − truth", "flags"], rows),
               f"\nvalidity (γ = 1 = HMM_{D} within 2 SE): {DASH if valid is None else valid}; train-NLL monotonicity "
               f"violations: {', '.join(viol) if viol else ('none' if any(fits.values()) else DASH)}; γ_½ = {f(gh, '.3f')}; "
               f"r = {r} vs r = 1 at γ = 0: held-out ΔKL = {DASH if pr is None else f'{pr[0]:+.2e} ± {pr[1]:.1e}'}, "
               f"train ΔNLL = {f(dtr, '+.2e')} (negative = stochasticity helps)"]
    return out, md


# ------------------------------------------------------------------------------- figures
def fig_gap(R, figs):
    ds = "flw-T50000-s0"
    t = best(R, "e2a", ds, "truth")
    if t is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogy(t["gap"], "o", color=OI["black"], ms=4, label="truth (FLW, λ = 0.8)")
    for mo, c, lab in (("qsm-D2-r1", OI["vermillion"], "QSM$_2$, r = 1"), ("hmm-2", OI["sky"], "HMM$_2$"),
                       ("hmm-4", OI["blue"], "HMM$_4$"), ("hmm-8", OI["green"], "HMM$_8$"),
                       ("hmm-16", OI["purple"], "HMM$_{16}$")):
        r = best(R, "e2a", ds, mo)
        if r is not None:
            ax.semilogy(r["gap"], "-", color=c, lw=1.4, label=lab)
    ax.set_xlabel("gap n (0s between two 1s)")
    ax.set_ylabel("P(gap = n)")
    ax.set_ylim(1e-7, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "seq5_flw_gap.png", dpi=DPI)
    plt.close(fig)


def _kl_panel(ax, R, exp, dsets, sizes, qsm_models, title):
    any_ = False
    for k, ds in enumerate(dsets):
        pts = [(S, best(R, exp, ds, f"hmm-{S}")) for S in sizes]
        pts = [(S, r) for S, r in pts if r is not None]
        if pts:
            any_ = True
            ax.errorbar([S for S, _ in pts], [r["heldout_kl_rate"] for _, r in pts],
                        yerr=[2 * r["heldout_se"] for _, r in pts], marker="os"[k % 2], color=[OI["blue"], OI["sky"]][k % 2],
                        capsize=2, label=f"HMM$_S$ {ds}")
        for j, (mo, lab) in enumerate(qsm_models):
            q = best(R, exp, ds, mo)
            if q is not None:
                any_ = True
                ax.axhline(max(q["heldout_kl_rate"], 1e-7), color=[OI["vermillion"], OI["orange"], OI["purple"]][j],
                           ls=["-", "--"][k % 2], lw=1.2, label=f"{lab} {ds}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("HMM states S")
    ax.set_ylabel("held-out KL rate (nats/symbol)")
    ax.set_title(title, fontsize=10)
    if any_:
        ax.legend(frameon=False, fontsize=6.5)
    return any_


def fig_kl_vs_S(R, figs):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    a = _kl_panel(axs[0], R, "e2a", ["flw-T50000-s0", "flw-T50000-s1"], FLW_S, [("qsm-D2-r1", "QSM$_2$ r=1")],
                  "FLW (λ = 0.8), T = 5e4")
    ex = [(S, best(R, "e2a-exact", "flw-T50000-s0", f"hmm-{S}")) for S in (4, 8, 12, 16)]
    ex = [(S, r) for S, r in ex if r is not None]
    if ex:
        axs[0].plot([S for S, _ in ex], [r["heldout_kl_rate"] for _, r in ex], "^", color=OI["green"],
                    label="HMM$_S$ exact-law fit")
        axs[0].legend(frameon=False, fontsize=6.5)
    b = _kl_panel(axs[1], R, "e2b", ["renewal32-s0", "renewal32-s1"], (2, 4, 8, 16, 32),
                  [("qsm-D2-r1", "QSM$_2$"), ("qsm-D4-r1", "QSM$_4$"), ("qsm-D8-r1", "QSM$_8$")], "renewal N = 32, T = 5e4")
    if a or b:
        fig.tight_layout()
        fig.savefig(figs / "seq5_kl_vs_S.png", dpi=DPI)
    plt.close(fig)


def fig_a1(R, figs):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    drawn = False
    for ax, ds in zip(axs, ("msar", "smsar16")):
        labs, vals, errs = [], [], []
        for exp, mo, lab in A1_MODELS:
            r = best(R, exp, ds, mo)
            if r is not None:
                labs.append(lab.replace(" (seq4)", "*"))
                vals.append(r["heldout_kl_rate"])
                errs.append(2 * r["heldout_se"])
        if vals:
            drawn = True
            cols = [OI["vermillion"] if "C-QSM" in l else OI["blue"] if "switch" in l else OI["green"] for l in labs]
            ax.barh(range(len(vals)), vals, xerr=errs, color=cols, capsize=2)
            ax.set_yticks(range(len(vals)), labs, fontsize=7)
            ax.invert_yaxis()
        ax.set_xlabel("held-out KL rate (nats/step)")
        ax.set_title(ds + "   (* = seq4 fit re-evaluated)", fontsize=9)
    if drawn:
        fig.tight_layout()
        fig.savefig(figs / "seq5_a1_bars.png", dpi=DPI)
    plt.close(fig)


def fig_e1(R, figs):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    drawn = False
    for ax, (ds, D, r) in zip(axs, (("renewal6-s0", 4, 16), ("renewal8-s0", 4, 16), ("renewal8-s0", 2, 4))):
        pts = [(g, best(R, "e1", ds, f"qsm-D{D}-r{r}-g{g:g}")) for g in GAMMAS]
        pts = [(g, x) for g, x in pts if x is not None]
        if pts:
            drawn = True
            ax.errorbar([g for g, _ in pts], [x["heldout_kl_rate"] for _, x in pts],
                        yerr=[2 * x["heldout_se"] for _, x in pts], marker="o", color=OI["vermillion"], capsize=2,
                        label=f"QSM$_{D}$ r = {r}")
        for mo, c, lab in ((f"qsm-D{D}-r1", OI["orange"], f"QSM$_{D}$ r = 1, γ = 0"), (f"hmm-{D}", OI["blue"], f"HMM$_{D}$")):
            x = best(R, "e1", ds, mo)
            if x is not None:
                drawn = True
                ax.axhline(x["heldout_kl_rate"], color=c, ls="--", lw=1.2, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("dephasing γ")
        ax.set_ylabel("held-out KL rate")
        ax.set_title(f"{ds}, D = {D}", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    if drawn:
        fig.tight_layout()
        fig.savefig(figs / "seq5_e1_kl_gamma.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "seq5"))
    ap.add_argument("--figs", default=str(ROOT / "figs"))
    a = ap.parse_args()
    runs, figs = Path(a.runs), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    R = load(runs)
    n = sum(len(v) for e in R.values() for d in e.values() for v in d.values())
    S, md = {"n_fits": n}, [f"# Sequence 5 summary ({n} fits on disk)"]
    for key, fn in (("T1", lambda: t1(R, runs)), ("T2", lambda: t2(R)), ("T3", lambda: t3(R)), ("T4", lambda: t4(R))):
        S[key], m = fn()
        md += m
    text = "\n".join(md) + "\n"
    (runs / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    (runs / "tables.md").write_text(text)
    for fig in (fig_gap, fig_kl_vs_S, fig_a1, fig_e1):
        fig(R, figs)
    print(text)


if __name__ == "__main__":
    main()
