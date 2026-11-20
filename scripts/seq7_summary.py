"""Sequence 7 summary: tables and figures from the fits of scripts/run_seq7.py (runs/seq7/), written so that every
number in a later report can be read from runs/seq7/summary.json.

Per (dataset, model): the fit with the best training likelihood among its inits (the selection rule of sequences 4-6),
its held-out KL rate ± the 20-segment SE, how many inits ran, how many stopped on the wall-clock cap or on an
iteration limit. Then three answers:
  core      per training seed of renewal N = 32: paired held-out difference KL(QSM_4) - KL(HMM_16) and
            KL(QSM_4) - KL(HMM_32) (per-step log-probabilities, 20-segment SE; < 0 = the QSM is better), and the count of
            seeds where the QSM wins by more than 2 SE.
  memory    held-out KL against filter size (QSM: D^2, HMM: S) and against memory dimension (QSM: D, HMM: S), per N,
            median and range over training seeds; equal-memory (QSM_D / HMM_D) and equal-filter (QSM_D / HMM_{D^2})
            ratios per seed.
  datasize  QSM D = 4, 8, 16 and HMM 16, 32 at T = 5e4 (seeds 0-2) and T = 2e5: held-out KL and train - truth
            log-likelihood gap (overfitting shows as a train likelihood above the truth's with a worse held-out KL).
  curve     T in {2.5e4, 5e4, 1e5, 2e5, 4e5}, 5 strings each: QSM_4, QSM_8, HMM_16, early-stopped HMM_32; paired
            QSM_4 - HMM_16, QSM_4 - HMM_32-es, QSM_8 - HMM_32-es per T.
-> runs/seq7/summary.json, runs/seq7/tables.csv, figs/seq7_core.png, figs/seq7_memory.png, figs/seq7_datasize.png,
   figs/seq7_curve.png.
Usage: python scripts/seq7_summary.py [--runs runs/seq7]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from archb import hmm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]
QCOL, HCOL = OKABE[3], OKABE[4]


def load(runs):
    """{(dataset, model): [(result, dir), ...]} over every experiment directory."""
    out = {}
    for p in sorted(runs.glob("*/*/*/*/result.json")):
        if p.parts[-5] in ("data", "logs"):
            continue
        r = json.loads(p.read_text())
        out.setdefault((r["dataset"], r["model"]), []).append((r, p.parent))
    return out


def iter_limited(r):
    msgs = str(r.get("lbfgs_message", "")).upper()
    return "LIMIT" in msgs or "ITERATIONS REACHED" in msgs


def best(F, ds, model):
    fits = F.get((ds, model), [])
    if not fits:
        return None
    key = "val_ll" if model.endswith("-es") else "train_ll"  # early-stopped fits: selected on validation
    r, d = max(fits, key=lambda t: t[0][key])
    return {"dataset": ds, "model": model, "init": r["init"], "n_inits": len(fits),
            "n_capped": sum(bool(f[0].get("capped")) for f in fits),
            "n_iter_limit": sum(iter_limited(f[0]) for f in fits),
            "best_capped": bool(r.get("capped")), "best_iter_limit": iter_limited(r),
            "kl": r["heldout_kl_rate"], "se": r["heldout_se"], "train_minus_truth": r["train_ll"] - r["truth_train_ll"],
            "filter_dim": r["filter_dim"], "n_params": r.get("n_params"), "d16": r.get("d16"),
            "wall_time": r.get("wall_time"), "dir": str(Path(d).resolve().relative_to(ROOT))}


_TEST = {}


def test_string(N):
    """The held-out string of renewal N (seed 999, shared by every training seed of one N, as in run_seq5)."""
    if N not in _TEST:
        _TEST[N] = hmm.sample(hmm.renewal(N), 201600, 999)
    return _TEST[N]


def lp_test(d):
    """Per-step held-out log-probabilities of the fit in d: lp_test.npy when the runner left it (it is not tracked),
    else recomputed from params.npz exactly as the runner computed them (QSM: superoperators of K; HMM: Abel law)."""
    p = d / "lp_test.npy"
    if p.exists():
        return np.load(p).astype(np.float64)
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from archb import qseq

    r = json.loads((d / "result.json").read_text())
    xs = test_string(r["N"])
    with np.load(d / "params.npz") as z:
        if r["model"].startswith("qsm"):
            K = jnp.asarray(z["K"])
            lp = np.asarray(qseq.sup_logprobs(qseq.superops(K, r.get("gamma", 0.0)), qseq.trace_vec(K.shape[-1]),
                                              jnp.asarray(xs)))
        else:
            lp = hmm.hmm_logprobs_abel(z["T"], xs)
    np.save(p, np.asarray(lp, dtype=np.float32))
    return np.asarray(lp, dtype=np.float32).astype(np.float64)


def paired(a, b):
    """KL(a) - KL(b) on the shared held-out string and its 20-segment SE (from the per-step log-probabilities)."""
    if a is None or b is None:
        return None
    la, lb = lp_test(ROOT / a["dir"]), lp_test(ROOT / b["dir"])
    diff, se = hmm.seg_kl_rate(lb, la)
    return {"diff": diff, "se": se, "ratio": a["kl"] / b["kl"] if b["kl"] > 0 else None,
            "qsm_wins_2se": diff < -2 * se, "hmm_wins_2se": diff > 2 * se}


def size(model):
    m = re.fullmatch(r"qsm-D(\d+)-r1", model) or re.fullmatch(r"hmm-(\d+)(?:-es)?", model)
    return int(m.group(1)) if m else None


def core(F):
    rows = []
    for s in range(5):
        ds = f"renewal32-s{s}"
        q, h16, h32 = best(F, ds, "qsm-D4-r1"), best(F, ds, "hmm-16"), best(F, ds, "hmm-32")
        e16, e32 = best(F, ds, "hmm-16-es"), best(F, ds, "hmm-32-es")
        rows.append({"dataset": ds, "qsm4": q, "hmm16": h16, "hmm32": h32, "hmm16_es": e16, "hmm32_es": e32,
                     "qsm4_vs_hmm16": paired(q, h16), "qsm4_vs_hmm32": paired(q, h32),
                     "qsm4_vs_hmm16_es": paired(q, e16), "qsm4_vs_hmm32_es": paired(q, e32)})
    done = [r for r in rows if r["qsm4_vs_hmm16"]]
    out = {"seeds": rows, "n_seeds": len(done),
           "qsm4_beats_hmm16_2se": sum(r["qsm4_vs_hmm16"]["qsm_wins_2se"] for r in done),
           "hmm16_beats_qsm4_2se": sum(r["qsm4_vs_hmm16"]["hmm_wins_2se"] for r in done),
           "median_ratio_qsm4_hmm16": float(np.median([r["qsm4_vs_hmm16"]["ratio"] for r in done])) if done else None}
    # Every comparison: per-seed wins at 2 SE, and the mean difference over the training strings with its SE across
    # strings (the held-out string is shared, so the segment SEs of different seeds are not independent).
    for key in ("qsm4_vs_hmm16", "qsm4_vs_hmm32", "qsm4_vs_hmm16_es", "qsm4_vs_hmm32_es"):
        ps = [r[key] for r in rows if r[key]]
        if ps:
            d = np.array([p["diff"] for p in ps])
            out[key] = {"n_seeds": len(ps), "qsm_wins_2se": sum(p["qsm_wins_2se"] for p in ps),
                        "hmm_wins_2se": sum(p["hmm_wins_2se"] for p in ps), "qsm_better": int((d < 0).sum()),
                        "mean_diff": float(d.mean()),
                        "se_across_seeds": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                        "median_ratio": float(np.median([p["ratio"] for p in ps]))}
    return out


def memory(F):
    out = {}
    for N in (16, 32, 64):
        dss = [f"renewal{N}-s{s}" for s in range(3)]
        curve = {}
        for kind, sizes in (("qsm", (2, 4, 8, 16)), ("hmm", (2, 4, 8, 16, 32, 64))):
            for k in sizes:
                model = f"qsm-D{k}-r1" if kind == "qsm" else f"hmm-{k}"
                b = [best(F, ds, model) for ds in dss]
                b = [x for x in b if x]
                if b:
                    kl = [x["kl"] for x in b]
                    curve[model] = {"kind": kind, "size": k, "filter_dim": k * k if kind == "qsm" else k,
                                    "n_seeds": len(b), "median_kl": float(np.median(kl)), "min_kl": float(min(kl)),
                                    "max_kl": float(max(kl)), "any_capped": any(x["best_capped"] for x in b),
                                    "median_train_minus_truth": float(np.median([x["train_minus_truth"] for x in b]))}
        ratios = []
        for ds in dss:
            for D in (2, 4, 8, 16):
                q = best(F, ds, f"qsm-D{D}-r1")
                for label, S in (("equal_memory", D), ("equal_filter", D * D)):
                    p = paired(q, best(F, ds, f"hmm-{S}"))
                    if p:
                        ratios.append({"dataset": ds, "D": D, "S": S, "comparison": label, **p})
        out[f"N{N}"] = {"curve": curve, "paired": ratios}
    return out


def datasize(F):
    rows = []
    for model in ("qsm-D4-r1", "qsm-D8-r1", "qsm-D16-r1", "hmm-16", "hmm-32", "hmm-32-es"):
        for T, dss in ((50000, [f"renewal32-s{s}" for s in range(3)]),
                       (200000, [f"renewal32-T200k-s{s}" for s in range(3)])):
            b = [x for x in (best(F, ds, model) for ds in dss) if x]
            if b:
                rows.append({"model": model, "T": T, "n_seeds": len(b),
                             "median_kl": float(np.median([x["kl"] for x in b])),
                             "median_train_minus_truth": float(np.median([x["train_minus_truth"] for x in b])),
                             "kl": [x["kl"] for x in b]})
    return rows


CURVE_DS = {25000: "renewal32-T25k-s{}", 50000: "renewal32-s{}", 100000: "renewal32-T100k-s{}",
            200000: "renewal32-T200k-s{}", 400000: "renewal32-T400k-s{}"}
CURVE_MODELS = ("qsm-D4-r1", "qsm-D8-r1", "hmm-16", "hmm-32-es")
CURVE_PAIRS = (("qsm-D4-r1", "hmm-16"), ("qsm-D4-r1", "hmm-32-es"), ("qsm-D8-r1", "hmm-32-es"))


def curve(F):
    """Data-size curve (tier 5 with the core / datasize fits): per T and model, the held-out KL of the 5 strings; per T
    and pair, paired differences per string, wins at 2 SE and the mean difference with its SE across strings."""
    out = {}
    for T, pat in CURVE_DS.items():
        dss = [pat.format(s) for s in range(5)]
        row = {"models": {}, "pairs": {}}
        for mo in CURVE_MODELS:
            b = [x for x in (best(F, ds, mo) for ds in dss) if x]
            if b:
                row["models"][mo] = {"n_seeds": len(b), "kl": [x["kl"] for x in b],
                                     "median_kl": float(np.median([x["kl"] for x in b])),
                                     "min_kl": float(min(x["kl"] for x in b)), "max_kl": float(max(x["kl"] for x in b)),
                                     "train_minus_truth": [x["train_minus_truth"] for x in b],
                                     "n_best_capped": sum(x["best_capped"] for x in b),
                                     "n_best_iter_limit": sum(x["best_iter_limit"] for x in b)}
        for a, b_ in CURVE_PAIRS:
            ps = [p for ds in dss if (p := paired(best(F, ds, a), best(F, ds, b_)))]
            if ps:
                d = np.array([p["diff"] for p in ps])
                row["pairs"][f"{a}_vs_{b_}"] = {
                    "n_seeds": len(ps), "diff": d.tolist(), "se": [p["se"] for p in ps],
                    "qsm_better": int((d < 0).sum()), "qsm_wins_2se": sum(p["qsm_wins_2se"] for p in ps),
                    "hmm_wins_2se": sum(p["hmm_wins_2se"] for p in ps), "mean_diff": float(d.mean()),
                    "se_across_seeds": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                    "median_ratio": float(np.median([p["ratio"] for p in ps]))}
        if row["models"]:
            out[str(T)] = row
    return out


def fig_curve(C, path):
    if not C:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = {"qsm-D4-r1": ("QSM$_4$ (filter 16, 64 par.)", QCOL, "o"), "qsm-D8-r1": ("QSM$_8$ (filter 64)", OKABE[1], "s"),
              "hmm-16": ("HMM$_{16}$ (filter 16, 496 par.)", HCOL, "^"),
              "hmm-32-es": ("HMM$_{32}$, early stopping (true size)", OKABE[0], "v")}
    for k, (mo, (lab, col, mk)) in enumerate(labels.items()):
        pts = [(int(T), r["models"][mo]) for T, r in C.items() if mo in r["models"]]
        if not pts:
            continue
        pts.sort()
        x = np.array([T for T, _ in pts]) * (1 + 0.03 * (k - 1.5))
        med = np.array([v["median_kl"] for _, v in pts])
        err = [med - [v["min_kl"] for _, v in pts], [v["max_kl"] for _, v in pts] - med]
        ax.errorbar(x, med, yerr=err, fmt=mk + "-", color=col, capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training symbols T")
    ax.set_ylabel("held-out KL rate (nats/symbol)\nmedian, range over 5 strings")
    ax.set_title("renewal N = 32: data-size curve", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_core(C, path):
    rows = [r for r in C["seeds"] if r["qsm4_vs_hmm16"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rows))
    for off, key, lab, col in ((-0.2, "qsm4_vs_hmm16", "QSM$_4$ − HMM$_{16}$ (equal filter size)", OKABE[3]),
                               (0.0, "qsm4_vs_hmm32", "QSM$_4$ − HMM$_{32}$ (true size)", OKABE[0]),
                               (0.2, "qsm4_vs_hmm32_es", "QSM$_4$ − HMM$_{32}$, early stopping", OKABE[5])):
        pts = [(i, r[key]) for i, r in enumerate(rows) if r[key]]
        if pts:
            ax.errorbar([i + off for i, _ in pts], [p["diff"] for _, p in pts], yerr=[2 * p["se"] for _, p in pts],
                        fmt="o", color=col, capsize=3, label=lab)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, [r["dataset"].split("-")[1] for r in rows])
    ax.set_xlabel("training string (renewal N = 32, T = 5·10⁴)")
    ax.set_ylabel("paired held-out KL difference\n(nats/symbol, ± 2 SE; < 0: QSM better)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_memory(M, path):
    keys = [k for k in ("N16", "N32", "N64") if M[k]["curve"]]
    if not keys:
        return
    fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
    for ax, k in zip(axs[0], keys):
        for kind, col, lab in (("qsm", QCOL, "QSM (filter D²)"), ("hmm", HCOL, "HMM (filter S)")):
            pts = sorted((v["filter_dim"], v) for v in M[k]["curve"].values() if v["kind"] == kind)
            if not pts:
                continue
            f = [p[0] for p in pts]
            med = np.array([p[1]["median_kl"] for p in pts])
            lo, hi = med - [p[1]["min_kl"] for p in pts], [p[1]["max_kl"] for p in pts] - med
            ax.errorbar(f, med, yerr=[lo, hi], fmt="o-", color=col, capsize=3, label=lab)
            for fd, v in pts:
                ax.annotate(str(v["size"]), (fd, v["median_kl"]), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=col)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"renewal {k[1:]}", fontsize=10)
        ax.set_xlabel("filter size (real numbers)")
    axs[0][0].set_ylabel("held-out KL rate (nats/symbol)\nmedian, range over 3 strings")
    axs[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_datasize(Dz, path):
    if not Dz:
        return
    models = sorted({r["model"] for r in Dz}, key=lambda m: (m[0], size(m)))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(models))
    for off, T, col in ((-0.15, 50000, OKABE[1]), (0.15, 200000, OKABE[3])):
        for i, mo in enumerate(models):
            r = next((r for r in Dz if r["model"] == mo and r["T"] == T), None)
            if r:
                ax.scatter([i + off] * len(r["kl"]), r["kl"], color=col, s=14,
                           label=f"T = {T:,}" if i == 0 else None)
    ax.set_xticks(x, models, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("held-out KL rate (nats/symbol)")
    ax.set_title("renewal N = 32: more training data", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "seq7"))
    a = ap.parse_args()
    runs = Path(a.runs)
    F = load(runs)
    table = [b for (ds, mo) in sorted(F) if (b := best(F, ds, mo))]
    S = {"fits": sum(len(v) for v in F.values()), "best_per_model": table, "core": core(F), "memory": memory(F),
         "datasize": datasize(F), "curve": curve(F)}
    (runs / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    with open(runs / "tables.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["dataset", "model", "init", "n_inits", "kl", "se", "train_minus_truth", "filter_dim", "n_params",
                "best_capped", "best_iter_limit", "n_capped", "n_iter_limit", "wall_time"]
        w.writerow(cols)
        for b in table:
            w.writerow([b[c] for c in cols])
    figs = ROOT / "figs"
    figs.mkdir(exist_ok=True)
    fig_core(S["core"], figs / "seq7_core.png")
    fig_memory(S["memory"], figs / "seq7_memory.png")
    fig_datasize(S["datasize"], figs / "seq7_datasize.png")
    fig_curve(S["curve"], figs / "seq7_curve.png")
    c = S["core"]
    print(f"{S['fits']} fits. Core: {c['n_seeds']} seeds, QSM_4 beats HMM_16 by > 2 SE on {c['qsm4_beats_hmm16_2se']}, "
          f"HMM_16 beats QSM_4 on {c['hmm16_beats_qsm4_2se']}, median KL ratio {c['median_ratio_qsm4_hmm16']}")
    for key in ("qsm4_vs_hmm16", "qsm4_vs_hmm32", "qsm4_vs_hmm16_es", "qsm4_vs_hmm32_es"):
        if key in c:
            v = c[key]
            print(f"  {key}: QSM better on {v['qsm_better']}/{v['n_seeds']}, by > 2 SE on {v['qsm_wins_2se']} "
                  f"(HMM by > 2 SE on {v['hmm_wins_2se']}); mean {v['mean_diff']:+.2e} ± {v['se_across_seeds']:.1e} "
                  f"across strings, median ratio {v['median_ratio']:.2f}")
    for r in c["seeds"]:
        for key in ("qsm4_vs_hmm16", "qsm4_vs_hmm32", "qsm4_vs_hmm16_es", "qsm4_vs_hmm32_es"):
            p = r[key]
            if p:
                print(f"  {r['dataset']} {key}: {p['diff']:+.2e} ± {p['se']:.1e} (ratio {p['ratio']:.2f})")
    for k, v in S["memory"].items():
        for mo, c2 in sorted(v["curve"].items(), key=lambda t: (t[1]["kind"], t[1]["size"])):
            print(f"  memory {k} {mo}: median KL {c2['median_kl']:.2e} [{c2['min_kl']:.1e}, {c2['max_kl']:.1e}]"
                  f" over {c2['n_seeds']}" + (" (capped)" if c2["any_capped"] else ""))
    for T, r in S["curve"].items():
        for key, p in r["pairs"].items():
            print(f"  curve T = {int(T):,} {key}: QSM better on {p['qsm_better']}/{p['n_seeds']}, by > 2 SE on "
                  f"{p['qsm_wins_2se']} (HMM on {p['hmm_wins_2se']}); mean {p['mean_diff']:+.2e}"
                  + (f" ± {p['se_across_seeds']:.1e}" if p["se_across_seeds"] else "")
                  + f", median ratio {p['median_ratio']:.2f}")


if __name__ == "__main__":
    main()
