"""Sequence 9 summary, from runs/seq9/synth/summary.json and runs/seq9/real/<dataset>/eval.json (scripts/run_seq9.py eval).

  synth  held-out KL rate to the truth, predicted correlation by true regime, dependence gain (joint log-score minus the
         product of the model's own predictive marginals), paired differences of the circuit models to their matched
         classical models (> 2 SE counted)
  real   test log-score (nats/day) and paired differences to the GARCH-t + DCC copula, dependence gain, predicted
         correlation against the realized 21-day correlation, joint-tail event calibration, equal-weight portfolio VaR
         hit rates; paired differences of the circuit models to their matched classical models
-> runs/seq9/summary.json, runs/seq9/tables.csv, figs/seq9_synth.png, figs/seq9_real_vs_dcc.png
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIGS = ROOT / "runs" / "seq9", ROOT / "figs"
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def family(mo):
    """Model id without the classical kernel size (rbm22 -> rbm), for tallies across datasets."""
    return re.sub(r"(rbm|mix)\d+", r"\1", mo)


def outcome(d):
    m, se = d
    return 1 if m > 2 * se else (-1 if m < -2 * se else 0)


def synth(out, rows):
    p = RUNS / "synth" / "summary.json"
    if not p.exists():
        return
    S = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ds, r) in zip(axes, S.items()):
        ms = sorted(r["models"], key=lambda m: r["models"][m]["heldout_kl_rate"])
        kl = [max(r["models"][m]["heldout_kl_rate"], 1e-5) for m in ms]
        se = [r["models"][m]["heldout_se"] for m in ms]
        col = [OI[6] if m.startswith(("cq", "k-q", "sw-q")) else OI[5] for m in ms]
        ax.barh(range(len(ms)), kl, xerr=se, color=col)
        ax.set_yticks(range(len(ms)), ms, fontsize=7)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("held-out KL rate to the truth (nats/step)")
        ax.set_title(f"{ds} (truth corr. by regime {r['truth']['corr_by_regime'][0]:.2f} / "
                     f"{r['truth']['corr_by_regime'][1]:.2f})", fontsize=9)
        for m, v in r["models"].items():
            rows.append({"part": "synth", "dataset": ds, "model": m, "n_params": v["n_params"],
                         "score": v["heldout_kl_rate"], "se": v["heldout_se"], "dep_gain": v["dep_gain"],
                         "corr_calm": v["corr_by_regime"][0], "corr_turb": v["corr_by_regime"][1],
                         "corr_rmse": v["corr_rmse"], "capped": v.get("capped")})
        out.setdefault("synth", {})[ds] = {
            "truth": r["truth"], "best": ms[:5],
            "pairs": {k: {"diff": v, "outcome": outcome(v)} for k, v in r["pairs"].items()}}
    fig.tight_layout()
    fig.savefig(FIGS / "seq9_synth.png", dpi=150)
    plt.close(fig)


def real(out, rows):
    evs = {p.parent.name: json.loads(p.read_text()) for p in sorted((RUNS / "real").glob("*/eval.json"))}
    if not evs:
        return
    tally = {}
    for ds, ev in evs.items():
        M = ev["models"]
        for m, e in M.items():
            rows.append({"part": "real", "dataset": ds, "model": m, "n_params": e.get("n_params"),
                         "score": e["test_ll"], "se": e["test_se"], "diff_dcc": e["diff/ref-dcc"][0],
                         "diff_dcc_se": e["diff/ref-dcc"][1], "dep_gain": e["dep_gain"],
                         "corr_vs_realized": e["corr"]["vs_realized_corr"], "corr_mae_dcc": e["corr"]["vs_dcc_mae"],
                         "tail_pred": e["tail"]["pred"], "tail_obs": e["tail"]["obs"],
                         "var1_hits": e["portfolio"]["0.01"]["hit_rate"], "var5_hits": e["portfolio"]["0.05"]["hit_rate"],
                         "var5_cc_p": e["portfolio"]["0.05"]["cc_p"], "capped": e.get("capped")})
        learned = {m: e for m, e in M.items() if not m.startswith("ref-")}
        best = max(learned, key=lambda m: learned[m]["val_ll"]) if learned else None
        out.setdefault("real", {})[ds] = {
            "series": ev["series"], "corr_train": ev["corr_train"][0][1:], "dcc": [ev["refs_info"]["dcc_a"],
                                                                                   ev["refs_info"]["dcc_b"]],
            "best_on_val": best, "best_test": max(M, key=lambda m: M[m]["test_ll"]),
            "test_ll": {m: [e["test_ll"], e["test_se"]] for m, e in M.items()},
            "vs_dcc": {m: e["diff/ref-dcc"] for m, e in M.items()},
            "dep_gain": {m: e["dep_gain"] for m, e in M.items()},
            "pairs": {k: {"diff": v, "outcome": outcome(v)} for k, v in ev["pairs"].items()}}
        for k, v in ev["pairs"].items():
            t = tally.setdefault(k.split("|")[0] + "|" + family(k.split("|")[1]), [0, 0, 0])
            t[1 - outcome(v)] += 1
    out["real_pair_tally"] = {k: {"circuit_better": v[0], "tie": v[1], "classical_better": v[2]} for k, v in tally.items()}
    keys = ["k-qa", "k-rbm", "k-mix", "sw-qa-S2", "sw-rbm", "sw-mix", "cq1", "cq2", "iohmm-fact-S2", "hmm-S4",
            "msvar", "ref-ccc", "ref-ind", "markov1"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    names = list(evs)
    for j, key in enumerate(keys):
        xs, ys, es = [], [], []
        for i, ds in enumerate(names):
            hit = [m for m in evs[ds]["models"] if m == key or (m.startswith(key) and key in ("k-rbm", "k-mix")
                   and not m.endswith("S2")) or (m.startswith(key) and key in ("sw-rbm", "sw-mix"))]
            if key in ("k-rbm", "k-mix", "sw-rbm", "sw-mix") and hit:  # the size matched to qa (the larger one)
                hit = [max(hit, key=lambda m: evs[ds]["models"][m].get("n_params") or 0)]
            if hit:
                d = evs[ds]["models"][hit[0]]["diff/ref-dcc"]
                xs.append(i + (j - len(keys) / 2) * 0.05)
                ys.append(d[0])
                es.append(2 * d[1])
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, color=OI[j % 8] if j < 8 else "0.5",
                        mfc="white" if j >= 8 else None, label=key, capsize=2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylabel("test log-score minus GARCH-t + DCC (nats/day, ± 2 SE)")
    ax.legend(fontsize=7, ncol=7, loc="lower center")
    fig.tight_layout()
    fig.savefig(FIGS / "seq9_real_vs_dcc.png", dpi=150)
    plt.close(fig)


def main():
    global RUNS, FIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--figs", default=str(FIGS))
    a = ap.parse_args()
    RUNS, FIGS = Path(a.runs), Path(a.figs)
    out, rows = {}, []
    FIGS.mkdir(exist_ok=True)
    synth(out, rows)
    real(out, rows)
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("part", "dataset", "model"), k))
    with open(RUNS / "tables.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {RUNS / 'tables.csv'}, {RUNS / 'summary.json'}")
    if "real_pair_tally" in out:
        for k, v in sorted(out["real_pair_tally"].items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
