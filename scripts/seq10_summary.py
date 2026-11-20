"""Sequence 10 summary, from runs/seq10/L<level>/*/*/*/result.json (scripts/run_seq10.py).

Per (level, target, encoding, size, model): the init with the best training likelihood of each data seed, metrics
averaged over the seeds. Families: circuits qB0 / qA2, rbm, mix, mixt, mps / nmps (Born-rule / nonnegative controls),
oracle, tab; level 3: the memory models by id.
  frontier  per family and parameter budget (32, 64, 128, 256, 512): the best censored tail KL (alpha = 1e-2 and 1e-3;
            level 2: per-row, weighted by the stationary law) of the family's models within the budget; the winning
            family per budget, and the ratio best circuit / best classical (< 1: the circuit is more parsimonious)
  encoding  the same best-within-budget value on u and g for every family (does the log-magnitude code help, and whom)
-> runs/seq10/summary.json, runs/seq10/tables.csv, figs/seq10_frontier_L1.png, figs/seq10_frontier_L2.png,
   figs/seq10_L3.png
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIGS = ROOT / "runs" / "seq10", ROOT / "figs"
OI = {"qB0": "#D55E00", "qA2": "#E69F00", "rbm": "#0072B2", "mix": "#56B4E9", "mixt": "#009E73", "mps": "#CC79A7",
      "nmps": "#000000", "oracle": "0.5", "tab": "0.75"}
BUDGETS = (32, 64, 128, 256, 512)
CIRCUITS = ("qB0", "qA2")


def family(model):
    m = re.fullmatch(r"(qB0|qA2)L\d+|(rbm|mixt|mix|nmps|mps)\d+", model)
    if m:
        return m.group(1) or m.group(2)
    return model


def metric(r, level, alpha):
    return r[f"{alpha:g}"]["ckl_rate"][0] if level == 3 else r[f"{alpha:g}"]["ckl"]


def load():
    """{(level, target, enc, size, model): [result of the best init per seed]}."""
    best = {}
    for p in RUNS.glob("L*/*/*/*/result.json"):
        r = json.loads(p.read_text())
        t, enc, n, s = r["dataset"].split("-")
        key = (r["level"], t, enc, int(n[1:]), r["model"], s)
        if key not in best or r["train_ll"] > best[key]["train_ll"]:
            best[key] = r
    out = defaultdict(list)
    for (lv, t, enc, n, mo, _), r in best.items():
        out[(lv, t, enc, n, mo)].append(r)
    return out


def frontier(rows, lv, t, enc, n, alpha):
    """{family: {budget: best metric}} for one setting."""
    fam = defaultdict(dict)
    for (lv_, t_, enc_, n_, mo), rs in rows.items():
        if (lv_, t_, enc_, n_) != (lv, t, enc, n):
            continue
        val = float(np.mean([metric(r, lv, alpha) for r in rs]))
        npar = rs[0]["n_params"]
        f = family(mo)
        for b in BUDGETS:
            if npar <= b and (b not in fam[f] or val < fam[f][b]):
                fam[f][b] = val
    return fam


def main():
    global RUNS, FIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--figs", default=str(FIGS))
    a = ap.parse_args()
    RUNS, FIGS = Path(a.runs), Path(a.figs)
    FIGS.mkdir(exist_ok=True)
    rows = load()
    table = []
    for (lv, t, enc, n, mo), rs in sorted(rows.items()):
        r0 = rs[0]
        row = {"level": lv, "target": t, "enc": enc, "size": n, "model": mo, "family": family(mo),
               "n_params": r0["n_params"], "n_seeds": len(rs)}
        if lv == 3:
            row["kl_rate"] = float(np.mean([r["kl_rate"][0] for r in rs]))
            row["kl_rate_se"] = float(np.mean([r["kl_rate"][1] for r in rs]))
        else:
            row["kl"] = float(np.mean([r["kl"] for r in rs]))
        for al in (1e-2, 1e-3):
            if f"{al:g}" not in r0:  # 2D level 1: alpha 0.05 / 0.01 below
                continue
            row[f"ckl_{al:g}"] = float(np.mean([metric(r, lv, al) for r in rs]))
            row[f"ratio_{al:g}"] = float(np.mean([r[f"{al:g}"]["ratio"] for r in rs if f"{al:g}" in r]))
        if lv == 1 and "0.05" in r0:
            row["joint_ratio_0.05"] = float(np.mean([r["0.05"]["joint_ratio"] for r in rs]))
            row["joint_ratio_0.01"] = float(np.mean([r["0.01"]["joint_ratio"] for r in rs]))
            row["ckl_0.05"] = float(np.mean([r["0.05"]["ckl"] for r in rs]))
            row["ckl_0.01"] = float(np.mean([r["0.01"]["ckl"] for r in rs]))
        table.append(row)
    keys = sorted({k for r in table for k in r}, key=lambda k: (k not in ("level", "target", "enc", "size", "model"), k))
    with open(RUNS / "tables.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    settings = sorted({k[:4] for k in rows})
    summ = {"frontier": {}, "circuit_vs_classical": {}, "encoding": {}}
    tally = defaultdict(lambda: [0, 0, 0])  # circuit better (< 0.8), tie, classical better (> 1.25), per budget
    for lv, t, enc, n in settings:
        if lv == 3:
            continue
        for al in ((0.05, 0.01) if t in ("tcop", "gcop") else (1e-2, 1e-3)):
            fam = frontier(rows, lv, t, enc, n, al)
            key = f"L{lv}/{t}/{enc}/n{n}/a{al:g}"
            summ["frontier"][key] = fam
            for b in BUDGETS:
                circ = [fam[f][b] for f in CIRCUITS if b in fam.get(f, {})]
                clas = [fam[f][b] for f in fam if f not in CIRCUITS + ("oracle", "tab") and b in fam[f]]
                if circ and clas:
                    ratio = min(circ) / max(min(clas), 1e-12)
                    summ["circuit_vs_classical"].setdefault(key, {})[b] = {
                        "ratio": ratio, "best_classical": min((f for f in fam if f not in CIRCUITS + ("oracle", "tab")
                                                               and b in fam[f]), key=lambda f: fam[f][b])}
                    tally[(lv, b)][0 if ratio < 0.8 else (2 if ratio > 1.25 else 1)] += 1
    summ["tally"] = {f"L{lv}/budget{b}": {"circuit_better": v[0], "tie": v[1], "classical_better": v[2]}
                     for (lv, b), v in sorted(tally.items())}
    for lv, t, enc, n in settings:  # encoding effect: g / u ratio of the best-within-256 metric per family
        if lv == 3 or enc != "g":
            continue
        al = 0.01
        fg, fu = frontier(rows, lv, t, "g", n, al), frontier(rows, lv, t, "u", n, al)
        summ["encoding"][f"L{lv}/{t}/n{n}"] = {f: fg[f][256] / max(fu[f][256], 1e-12) for f in fg
                                               if 256 in fg[f] and 256 in fu.get(f, {})}
    l3 = {}
    for (lv, t, enc, n, mo), rs in rows.items():
        if lv == 3:
            l3.setdefault(f"{t}/{enc}/n{n}", {})[mo] = {
                "kl_rate": float(np.mean([r["kl_rate"][0] for r in rs])),
                "ckl_rate_0.01": float(np.mean([r["0.01"]["ckl_rate"][0] for r in rs])),
                "ckl_rate_0.001": float(np.mean([r["0.001"]["ckl_rate"][0] for r in rs])),
                "ratio_0.01": float(np.mean([r["0.01"]["ratio"] for r in rs])), "n_params": rs[0]["n_params"]}
    summ["level3"] = l3
    (RUNS / "summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    for lv, fname in ((1, "seq10_frontier_L1.png"), (2, "seq10_frontier_L2.png")):
        sets = [s for s in settings if s[0] == lv]
        if not sets:
            continue
        size = sorted({s[3] for s in sets})[len({s[3] for s in sets}) // 2]
        targets = sorted({s[1] for s in sets})
        fig, axes = plt.subplots(len(targets), 2, figsize=(10, 2.8 * len(targets)), squeeze=False)
        for i, t in enumerate(targets):
            for j, enc in enumerate(("u", "g")):
                ax = axes[i, j]
                al = 0.01
                pts = defaultdict(list)
                for (lv_, t_, enc_, n_, mo), rs in rows.items():
                    if (lv_, t_, enc_, n_) == (lv, t, enc, size):
                        pts[family(mo)].append((rs[0]["n_params"], np.mean([metric(r, lv, al) for r in rs])))
                for f, ps in sorted(pts.items()):
                    ps = sorted(ps)
                    ax.plot([p[0] for p in ps], [max(p[1], 1e-7) for p in ps], "o-" if len(ps) > 1 else "s",
                            color=OI.get(f, "0.3"), label=f, ms=4, lw=1.2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{t}, {enc}, n = {size}", fontsize=9)
                ax.set_xlabel("free parameters", fontsize=8)
                ax.set_ylabel("censored KL, 1% tails", fontsize=8)
        axes[0, 0].legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(FIGS / fname, dpi=140)
        plt.close(fig)
    if l3:
        keys3 = sorted(l3)
        fig, axes = plt.subplots(1, len(keys3), figsize=(3.2 * len(keys3), 4.2), squeeze=False)
        for ax, k in zip(axes[0], keys3):
            ms = sorted(l3[k], key=lambda m: l3[k][m]["ckl_rate_0.01"])
            ax.barh(range(len(ms)), [max(l3[k][m]["ckl_rate_0.01"], 1e-7) for m in ms],
                    color=["#D55E00" if m.startswith(("cq", "k-q", "sw-q")) else "#0072B2" for m in ms])
            ax.set_yticks(range(len(ms)), ms, fontsize=7)
            ax.set_xscale("log")
            ax.invert_yaxis()
            ax.set_title(k, fontsize=8)
            ax.set_xlabel("censored KL rate, 1% tails", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "seq10_L3.png", dpi=140)
        plt.close(fig)
    print(f"{len(table)} rows -> {RUNS / 'tables.csv'}")
    for k, v in summ["tally"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
