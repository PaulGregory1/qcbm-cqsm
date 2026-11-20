"""Sequence 6: aggregate runs/seq6/** into runs/seq6/summary.json and runs/seq6/tables.md and draw figs/seq6_*.png.
Read-only on the run directories; needs `scripts/run_seq6.py eval` first (scripts/reproduce.py runs it).

Selection, as in the protocols the runs follow: synthetic targets, the init with the best TRAINING likelihood (sequence 5);
real data, the init with the best VALIDATION log-score (step 13), and the variant of a family (QCBM-kernel switches,
classical-kernel switches) with the best validation log-score.
Matched pairs (rule fixed before the full runs, after a smoke run): a QCBM kernel against each classical kernel sized to
it, same number of switch states (anc2, 215 -> mlp6, mlp5, mix36, rbm17; B0, 119 -> mlp3, mix20, rbm8; the conditional
RBM was added after the first results, with the same protocol and rule), and the kernels alone (stage 1, no memory). Paired difference of the per-step log-probabilities on the held-out path (synthetic, SE over 20 segments)
or the test period (real data, 50-day block bootstrap); label "QCBM better" / "classical better" when the difference
exceeds 2 SE, otherwise "tie".

Run: PYTHONPATH=src ./.venv/bin/python scripts/seq6_summary.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from archb import cqsm, hmm, regimes as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIGS = ROOT / "runs" / "seq6", ROOT / "figs"
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
DPI = 170
SYNTH, REAL = ["msar", "smsar16"], ["eurusd-u", "eurusd-q", "eurusd-v", "nasdaq-u", "nasdaq-q", "nasdaq-v"]
PAIRS = [("sw-anc2-S2", "sw-mlp6-S2"), ("sw-anc2-S2", "sw-mlp5-S2"), ("sw-anc2-S2", "sw-mix36-S2"),
         ("sw-anc2-S2", "sw-rbm17-S2"), ("sw-B0-S2", "sw-mlp3-S2"), ("sw-B0-S2", "sw-mix20-S2"), ("sw-B0-S2", "sw-rbm8-S2"),
         ("sw-anc2-S4", "sw-mlp6-S4"), ("sw-anc2-S4", "sw-mix36-S4"), ("sw-anc2-S4", "sw-rbm17-S4"),
         ("archb-anc2", "k-mlp6"), ("archb-anc2", "k-mlp5"), ("archb-anc2", "k-mix36"), ("archb-anc2", "k-rbm17"),
         ("archb-B0", "k-mlp3"), ("archb-B0", "k-mix20"), ("archb-B0", "k-rbm8")]
FAMILIES = {"mlp": "MLP", "mix": "Gaussian mixture", "rbm": "conditional RBM"}
NAMES = {"anc2": "QCBM anc2", "B0": "QCBM B0", "mlp6": "MLP H=6", "mlp5": "MLP H=5", "mlp3": "MLP H=3",
         "mix36": "Gauss. mix. K=36", "mix20": "Gauss. mix. K=20", "rbm17": "cond. RBM H=17", "rbm8": "cond. RBM H=8"}
DS_NAMES = {"eurusd": "EUR/USD", "nasdaq": "NASDAQ"}


def label(d, se):
    return "QCBM better" if d > 2 * se else "classical better" if d < -2 * se else "tie"


def name(mo):
    if mo.startswith(("sw-", "k-")):
        p = mo.split("-")
        kind = NAMES[p[1]]
        return f"{kind}, kernel alone" if p[0] == "k" else f"{kind}, switch S = {p[2][1:]}"
    if mo.startswith("archb-"):
        return f"{NAMES[mo.split('-')[1]]}, kernel alone"
    return mo


def dsname(ds):
    s, g = ds.split("-")
    return f"{DS_NAMES[s]} {g}"


def f3(x):
    return "–" if x is None else f"{x:.2e}"


def fd(x):
    return "–" if x is None else f"{x[0]:+.4f} ± {x[1]:.4f}"


# ------------------------------------------------------------------------------------------------ synthetic
def synth():
    s = json.loads((RUNS / "synth" / "summary.json").read_text())
    seq4 = json.loads((ROOT / "runs" / "seq4" / "summary.json").read_text())
    out = {}
    for ds in SYNTH:
        rows = s[ds]
        kind, N, seed = {"msar": ("msar", None, 700), "smsar16": ("smsar", 16, 730)}[ds]
        d = R.load_data(ds, kind, N, seed)
        truth = R.truth_T(kind, N)
        from archb import iolin
        lp_true = iolin.IOLin.from_iohmm(truth).logprobs(d["test"])
        lps = {}
        a1 = ROOT / "runs" / "seq5" / "a1" / ds
        for mo in rows:
            src = (a1 if mo in ("sw-anc2-S2", "sw-anc2-S4", "sw-B0-S2") else RUNS / "synth" / ds) / mo / str(rows[mo]["init"])
            lps[mo] = np.load(src / "lp_test.npy").astype(float)
        for k in ("anc2", "B0"):  # Architecture B kernels alone (sequence 4 stage 1, best training likelihood)
            b = seq4[ds][f"archb-{k}"]["best"]
            th = np.load(ROOT / "runs" / "seq4" / ds / f"archb-{k}" / str(b["init"]) / "params.npz")["theta"]
            lps[f"archb-{k}"] = cqsm.io(th, 0, 2 if k == "anc2" else 0, 3).logprobs(d["test"])
            rate, se = hmm.seg_kl_rate(lp_true, lps[f"archb-{k}"])
            rows[f"archb-{k}"] = {"heldout_kl_rate": rate, "heldout_se": se, "n_kernel": b["n_params"], "S": 1,
                                  "init": b["init"], "kl_inits_min": None, "kl_inits_max": None}
        pairs = {}
        for q, c in PAIRS:
            if q in lps and c in lps:
                dl = lps[q] - lps[c]
                n = len(dl) // 20 * 20
                seg = dl[:n].reshape(20, -1).mean(1)
                m, se = float(seg.mean()), float(seg.std(ddof=1) / np.sqrt(20))
                pairs[f"{q}|{c}"] = {"diff": m, "se": se, "label": label(m, se)}
        refs = {k: seq4[ds][k]["best"]["heldout_kl_rate"] for k in ("family-msar", "iohmm-fact-2")}
        cq = json.loads((ROOT / "runs" / "seq5" / "reeval" / ds / "cqsm-m1-anc2" / "best" / "result.json").read_text())
        refs["cqsm-m1-anc2"] = cq["heldout_kl_rate"]
        out[ds] = {"rows": rows, "pairs": pairs, "refs": refs}
    return out


# ------------------------------------------------------------------------------------------------ real data
def real():
    out = {}
    for ds in REAL:
        p = RUNS / "real" / ds / "eval.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text())
        arr = np.load(RUNS / "real" / ds / "eval_arrays.npz")
        ref_ev = json.loads((ROOT / "runs" / "seq4_real" / ds / "eval.json").read_text())
        ref_arr = np.load(ROOT / "runs" / "seq4_real" / ds / "eval_arrays.npz")
        s = ref_ev["n_train"] + ref_ev["n_val"]
        from archb import realdata as D
        lps = {k[3:]: arr[k] for k in arr.files if k.startswith("lp/")}
        for k in ("archb-anc2", "archb-B0"):
            if f"lp/{k}" in ref_arr.files:
                lps[k] = ref_arr[f"lp/{k}"]
        pairs = {}
        for q, c in PAIRS:
            if q in lps and c in lps:
                dl = lps[q][s:] - lps[c][s:]
                m, se = float(dl.mean()), float(D.block_bootstrap(dl, 50))
                pairs[f"{q}|{c}"] = {"diff": m, "se": se, "label": label(m, se)}
        ch = e["chosen"]
        fam = {}
        if "qcbm" in ch and "classical" in ch:
            m, se = e["pairs"]["qcbm-classical"]
            fam = {"qcbm": ch["qcbm"], "classical": ch["classical"], "diff": m, "se": se, "label": label(m, se)}
        refs = {k: {"diff_garch_t": None if k == "garch-t-cont" else
                    [float((ref_arr[f"lp/{k}"][s:] - ref_arr["lp/garch-t-cont"][s:]).mean()),
                     float(D.block_bootstrap(ref_arr[f"lp/{k}"][s:] - ref_arr["lp/garch-t-cont"][s:], 50))],
                    "var1": ref_ev["models"][k].get("var1")}
                for k in ("garch-t-cont", "garch-gjr-cont", "msar-cont", "msar-bin", e["cqsm_chosen"])}
        out[ds] = {"models": e["models"], "chosen": ch, "family": fam, "pairs": pairs, "refs": refs,
                   "cqsm_chosen": e["cqsm_chosen"], "n_test": e["n_test"]}
    return out


# ------------------------------------------------------------------------------------------------ tables
def tables(sy, re_):
    L = ["# Sequence 6 — tables", "", "Generated by `scripts/seq6_summary.py` from `runs/seq6/` (fits: `scripts/run_seq6.py`).",
         ""]
    L += ["## T1 — synthetic targets (held-out KL rate, nats/step)", "",
          "Reported fit = init with the best training likelihood. QCBM-kernel switches from sequence 5 A1; Architecture B "
          "kernels alone from sequence 4 stage 1.", ""]
    for ds, d in sy.items():
        L += [f"**{ds}** (references: MS-AR family fit {f3(d['refs']['family-msar'])}, IOHMM factored S = 2 "
              f"{f3(d['refs']['iohmm-fact-2'])}, C-QSM m = 1 + 2 anc {f3(d['refs']['cqsm-m1-anc2'])})", "",
              "| Model | Kernel params | Held-out KL ± SE | Inits (min–max) | L-BFGS limit |", "| --- | ---: | ---: | ---: | --- |"]
        order = sorted(d["rows"], key=lambda m: d["rows"][m]["heldout_kl_rate"])
        for mo in order:
            r = d["rows"][mo]
            L.append(f"| {name(mo)} | {r.get('n_kernel', '–')} | {r['heldout_kl_rate']:.2e} ± {r['heldout_se']:.1e} | "
                     f"{f3(r.get('kl_inits_min'))} – {f3(r.get('kl_inits_max'))} | {'yes' if r.get('it_limit') else ''} |")
        L += ["", "| QCBM | Classical | KL(classical) − KL(QCBM) ± SE | Outcome |", "| --- | --- | ---: | --- |"]
        for k, pr in d["pairs"].items():
            q, c = k.split("|")
            L.append(f"| {name(q)} | {name(c)} | {pr['diff']:+.2e} ± {pr['se']:.1e} | {pr['label']} |")
        L.append("")
    L += ["## T2 — real daily returns (test log-score, nats/day)", "",
          "Reported fit = init with the best validation log-score. Differences: model minus reference on the test period, "
          "50-day block-bootstrap SE (> 0: the model is better).", ""]
    for ds, d in re_.items():
        L += [f"**{dsname(ds)}** ({d['n_test']} test days; chosen on validation: QCBM switch `{d['chosen'].get('qcbm')}`, "
              f"classical switch `{d['chosen'].get('classical')}`; step 13 C-QSM `{d['cqsm_chosen']}`)", "",
              "| Model | Kernel params | − GARCH-t | − MS-AR (cont.) | − C-QSM | VaR 5%: hits, Kupiec p, ES Z | VaR 1%: hits, Kupiec p, ES Z |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        ms = d["models"]
        for mo in sorted(ms, key=lambda m: -ms[m]["test_ll"]):
            r = ms[mo]
            v5, v1 = (r.get("var1") or {}).get("0.05"), (r.get("var1") or {}).get("0.01")
            fv = lambda v: "–" if v is None else f"{100 * v['hit_rate']:.1f}%, {v['kupiec_p']:.2f}, {v['es_z']:+.2f}"
            L.append(f"| {name(mo)} | {r.get('n_kernel')} | {fd(r['diff/garch-t-cont'])} | {fd(r['diff/msar-cont'])} | "
                     f"{fd(r['diff/' + d['cqsm_chosen']])} | {fv(v5)} | {fv(v1)} |")
        for k, rf in d["refs"].items():
            v5, v1 = (rf["var1"] or {}).get("0.05"), (rf["var1"] or {}).get("0.01")
            fv = lambda v: "–" if v is None else f"{100 * v['hit_rate']:.1f}%, {v['kupiec_p']:.2f}, {v['es_z']:+.2f}"
            L.append(f"| *{k} (step 13)* | – | {fd(rf['diff_garch_t']) if rf['diff_garch_t'] else '0'} | – | – | {fv(v5)} | {fv(v1)} |")
        L += ["", "| QCBM | Classical | QCBM − classical ± SE | Outcome |", "| --- | --- | ---: | --- |"]
        if d["family"]:
            f = d["family"]
            L.append(f"| chosen: {name(f['qcbm'])} | chosen: {name(f['classical'])} | {f['diff']:+.4f} ± {f['se']:.4f} | {f['label']} |")
        for k, pr in d["pairs"].items():
            q, c = k.split("|")
            L.append(f"| {name(q)} | {name(c)} | {pr['diff']:+.4f} ± {pr['se']:.4f} | {pr['label']} |")
        L.append("")
    # outcome counts
    L += ["## T3 — outcome counts over matched pairs", "", "| Set | QCBM better | tie | classical better |",
          "| --- | ---: | ---: | ---: |"]
    for nm, groups in (("synthetic, switches", [sy[d]["pairs"] for d in sy]), ("real, switches", [re_[d]["pairs"] for d in re_])):
        for sub, cond in ((" (switch)", lambda k: k.startswith("sw-")), (" (kernel alone)", lambda k: not k.startswith("sw-"))):
            labs = [pr["label"] for g in groups for k, pr in g.items() if cond(k)]
            L.append(f"| {nm.split(',')[0]}{sub} | {labs.count('QCBM better')} | {labs.count('tie')} | "
                     f"{labs.count('classical better')} |")
    labs = [re_[d]["family"]["label"] for d in re_ if re_[d]["family"]]
    L.append(f"| real, chosen families | {labs.count('QCBM better')} | {labs.count('tie')} | {labs.count('classical better')} |")
    L += ["", "By classical family (switches and kernels alone together):", "",
          "| Data | Classical family | QCBM better | tie | classical better |", "| --- | --- | ---: | ---: | ---: |"]
    for nm, groups in (("synthetic", [sy[d]["pairs"] for d in sy]), ("real", [re_[d]["pairs"] for d in re_])):
        for fam, fname in FAMILIES.items():
            labs = [pr["label"] for g in groups for k, pr in g.items() if k.split("|")[1].split("-")[1].startswith(fam)]
            L.append(f"| {nm} | {fname} | {labs.count('QCBM better')} | {labs.count('tie')} | "
                     f"{labs.count('classical better')} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ figures
def figures(sy, re_):
    FIGS.mkdir(exist_ok=True)
    # synthetic: held-out KL of the S = 2 switches and the kernels alone, QCBM vs classical
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, ds in zip(axes, SYNTH):
        rows = sy[ds]["rows"]
        mods = [m for m in ("sw-anc2-S2", "sw-mlp6-S2", "sw-mlp5-S2", "sw-mix36-S2", "sw-rbm17-S2", "sw-B0-S2",
                            "sw-mlp3-S2", "sw-mix20-S2", "sw-rbm8-S2") if m in rows]
        y = np.arange(len(mods))[::-1]
        for yi, m in zip(y, mods):
            r = rows[m]
            col = OI["blue"] if m.split("-")[1] in ("anc2", "B0") else OI["orange"]
            ax.errorbar(r["heldout_kl_rate"], yi, xerr=2 * r["heldout_se"], fmt="o", ms=6, color=col, capsize=0, lw=1.5)
        for k, ls in (("family-msar", ":"), ("cqsm-m1-anc2", "--")):
            ax.axvline(sy[ds]["refs"][k], color=OI["grey"], lw=1, ls=ls)
        ax.set_yticks(y, [name(m).replace(", switch S = 2", "") for m in mods], fontsize=8)
        if ds == "msar":  # spans three decades (the MS-AR family fit is at 2.5e-5); smsar16 fits in one: linear
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.set_xlabel("held-out KL rate (nats/step), ± 2 SE")
        ax.set_title(ds, fontsize=10)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    hs = [axes[1].plot([], [], "o", color=OI["blue"])[0], axes[1].plot([], [], "o", color=OI["orange"])[0],
          axes[1].plot([], [], ls="--", color=OI["grey"])[0], axes[1].plot([], [], ls=":", color=OI["grey"])[0]]
    fig.legend(hs, ["QCBM kernels", "classical kernels", "C-QSM m = 1 + 2 anc", "MS-AR family fit"], fontsize=8,
               frameon=False, ncol=4, loc="lower center")
    fig.suptitle("Switch with S = 2 classical states: QCBM kernels vs classical kernels at equal parameter count", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "seq6_synth_kl.png", dpi=DPI)
    plt.close(fig)
    # real: test log-score minus GARCH-t of the chosen families and references
    if not re_:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [("C-QSM (step 13)", OI["purple"], lambda d: d["refs"][d["cqsm_chosen"]]["diff_garch_t"]),
              ("QCBM switch (chosen)", OI["blue"], lambda d: d["models"][d["chosen"]["qcbm"]]["diff/garch-t-cont"]),
              ("classical switch (chosen)", OI["orange"], lambda d: d["models"][d["chosen"]["classical"]]["diff/garch-t-cont"]),
              ("MS-AR, continuous", OI["grey"], lambda d: d["refs"]["msar-cont"]["diff_garch_t"]),
              ("GJR-GARCH-t", OI["black"], lambda d: d["refs"]["garch-gjr-cont"]["diff_garch_t"])]
    dss = list(re_)
    x = np.arange(len(dss))
    w = 0.14
    for j, (lab, col, get) in enumerate(series):
        vals = [get(re_[d]) for d in dss]
        ax.errorbar(x + (j - 2) * w, [v[0] for v in vals], yerr=[2 * v[1] for v in vals], fmt="o", ms=6, color=col,
                    capsize=0, lw=1.5, label=lab)
    ax.axhline(0, color=OI["grey"], lw=1)
    ax.set_xticks(x, [dsname(d) for d in dss], fontsize=8)
    ax.set_ylabel("test log-score − GARCH-t (nats/day), ± 2 SE")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIGS / "seq6_real_vs_garch.png", dpi=DPI)
    plt.close(fig)


def main():
    sy, re_ = synth(), real()
    out = {"synth": {ds: {"rows": d["rows"], "pairs": d["pairs"], "refs": d["refs"]} for ds, d in sy.items()},
           "real": {ds: {k: v for k, v in d.items() if k != "models"} for ds, d in re_.items()}}
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    t = tables(sy, re_)
    (RUNS / "tables.md").write_text(t)
    figures(sy, re_)
    print(t)


if __name__ == "__main__":
    main()
