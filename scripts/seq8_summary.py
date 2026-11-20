"""Sequence 8 summary: retrained dephasing curve, trained-basis dephasing and the replication across datasets, from the
fits in runs/seq8/ (scripts/run_seq8.py). Per (experiment, dataset, model): the init with the best training likelihood.

  deph / basis  held-out KL rate against gamma for C-QSM m = 1, 2 (+ the trained-basis fit at gamma = 1), paired
                differences to the coherent fit (gamma = 0) of the same run
  rep           per dataset, paired differences to the coherent C-QSM of the same size (m = 1 against the dephased
                m = 1, the classical switch S = 2, IOHMM_2, the MS-AR fit; m = 2 against the dephased m = 2, S = 4,
                IOHMM_4), counts of > 2 SE outcomes over the 5 datasets; spread of the KL rate over inits (optimization)
                and over datasets
  inits         smsar16, C-QSM m = 1 coherent against dephased: best of the first 5 inits and best of all (20 with the
                tier 4 fits), with the KL of every init
-> runs/seq8/summary.json, runs/seq8/tables.csv, figs/seq8_dephasing.png, figs/seq8_replication.png
Paired SEs use lp_test.npy next to each result; when it is missing (it is not tracked) it is recomputed from params.npz.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from archb import hmm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
REP_PAIRS = {"cqsm-m1-anc2-g0": ("cqsm-m1-anc2-g1", "sw-anc2-S2", "iohmm-fact-2", "family-msar"),
             "cqsm-m2-anc2-g0": ("cqsm-m2-anc2-g1", "sw-anc2-S4", "iohmm-fact-4", "cqsm-m1-anc2-g0")}


def load(runs):
    out = {}
    for p in sorted(runs.glob("*/*/*/*/result.json")):
        if p.parts[-5] in ("data", "logs"):
            continue
        r = json.loads(p.read_text())
        out.setdefault((r["exp"], r["dataset"], r["model"]), []).append((r, p.parent))
    return out


def best(F, exp, ds, model, max_init=None):
    """Best training likelihood among the inits (only inits < max_init if given)."""
    fits = [f for f in F.get((exp, ds, model), []) if max_init is None or int(f[0]["init"]) < max_init]
    if not fits:
        return None
    r, d = max(fits, key=lambda t: t[0]["train_ll"])
    kls = [f[0]["heldout_kl_rate"] for f in fits]
    return {"exp": exp, "dataset": ds, "model": model, "init": r["init"], "n_inits": len(fits), "kl": r["heldout_kl_rate"],
            "se": r["heldout_se"], "train_minus_truth": r["train_ll"] - r["truth_train_ll"],
            "kl_inits_min": float(min(kls)), "kl_inits_max": float(max(kls)), "n_params": r.get("n_params"),
            "filter_dim": r.get("filter_dim"), "capped": bool(r.get("capped")),
            "iter_limit": "LIMIT" in str(r.get("lbfgs_message", "")).upper(), "dir": str(Path(d).resolve().relative_to(ROOT))}


_S8 = []


def _runner():
    """scripts/run_seq8.py, configured on runs/seq8 (imported only when a held-out log-probability is missing)."""
    if not _S8:
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_seq8

        run_seq8.S5.configure(ROOT / "runs" / "seq8")
        _S8.append(run_seq8)
    return _S8[0]


def paired(a, b):
    """KL(b) - KL(a) on the shared held-out path (positive: b is worse) with its 20-segment SE."""
    if a is None or b is None:
        return None
    pa, pb = ROOT / a["dir"] / "lp_test.npy", ROOT / b["dir"] / "lp_test.npy"
    for p, x in ((pa, a), (pb, b)):
        if not p.exists():  # not tracked: recompute from params.npz as the fit did
            np.save(p, np.asarray(_runner().heldout_logprobs(x, p.parent), dtype=np.float32))
    diff, se = hmm.seg_kl_rate(np.load(pa).astype(np.float64), np.load(pb).astype(np.float64))
    return {"diff": diff, "se": se, "a_better_2se": diff > 2 * se, "b_better_2se": diff < -2 * se}


def deph(F):
    out = {}
    for ds in ("msar", "smsar16"):
        rows = {}
        for (exp, d, mo) in F:
            if d == ds and exp in ("deph", "basis"):
                b = best(F, exp, ds, mo)
                m = mo.split("-")[1]
                ref = best(F, "deph", ds, f"cqsm-{m}-anc2-g0")
                rows[mo] = {**b, "vs_coherent": paired(ref, b) if mo != f"cqsm-{m}-anc2-g0" else None}
        out[ds] = rows
    return out


def rep(F):
    out = {}
    for base in ("msar", "smsar16"):
        dss = [base] + [f"{base}-d{k}" for k in range(1, 5)]
        per, agg = {}, {}
        for ds in dss:
            row = {}
            for ref, others in REP_PAIRS.items():
                a = best(F, "rep", ds, ref)
                row[ref] = a
                for o in others:
                    row[o] = best(F, "rep", ds, o)
                    row[f"{o}_vs_{ref}"] = paired(a, row[o])
            per[ds] = row
        for ref, others in REP_PAIRS.items():
            for o in others:
                ps = [per[ds][f"{o}_vs_{ref}"] for ds in dss if per[ds].get(f"{o}_vs_{ref}")]
                if ps:
                    agg[f"{o}_vs_{ref}"] = {"n_datasets": len(ps), "mean_diff": float(np.mean([p["diff"] for p in ps])),
                                            "ref_better_2se": sum(bool(p["a_better_2se"]) for p in ps),
                                            "other_better_2se": sum(bool(p["b_better_2se"]) for p in ps)}
        models = sorted({k for ds in dss for k, v in per[ds].items() if isinstance(v, dict) and "kl" in v})
        spread = {}
        for mo in models:
            b = [per[ds][mo] for ds in dss if per[ds].get(mo)]
            spread[mo] = {"n_datasets": len(b), "kl_mean": float(np.mean([x["kl"] for x in b])),
                          "kl_sd_datasets": float(np.std([x["kl"] for x in b], ddof=1)) if len(b) > 1 else None,
                          "kl_range_inits_mean": float(np.mean([x["kl_inits_max"] - x["kl_inits_min"] for x in b]))}
        out[base] = {"datasets": per, "pairs": agg, "spread": spread}
    return out


def inits(F):
    """Tier 4: C-QSM m = 1 coherent (g0) against dephased (g1) on the smsar16 datasets, best of the first 5 inits and
    best of all inits (20 after tier 4); the KL of every init, to see how far best-of-5 is from converged."""
    out, a_, b_ = {}, "cqsm-m1-anc2-g0", "cqsm-m1-anc2-g1"
    for ds in ["smsar16"] + [f"smsar16-d{k}" for k in range(1, 5)]:
        row = {}
        for label, mx in (("best_of_5", 5), ("best_of_all", None)):
            a, b = best(F, "rep", ds, a_, mx), best(F, "rep", ds, b_, mx)
            row[label] = {"coherent": a, "dephased": b, "dephased_vs_coherent": paired(a, b)}
        row["kl_by_init"] = {mo: sorted((int(r["init"]), r["heldout_kl_rate"], r["train_ll"])
                                        for r, _ in F.get(("rep", ds, mo), [])) for mo in (a_, b_)}
        out[ds] = row
    agg = {}
    for label in ("best_of_5", "best_of_all"):
        ps = [r[label]["dephased_vs_coherent"] for r in out.values() if r[label]["dephased_vs_coherent"]]
        if ps:
            agg[label] = {"n_datasets": len(ps), "n_inits": [r[label]["coherent"]["n_inits"] for r in out.values()
                                                             if r[label]["coherent"]],
                          "mean_diff": float(np.mean([p["diff"] for p in ps])),
                          "coherent_better": sum(p["diff"] > 0 for p in ps),
                          "coherent_better_2se": sum(bool(p["a_better_2se"]) for p in ps),
                          "dephased_better_2se": sum(bool(p["b_better_2se"]) for p in ps)}
    return {"datasets": out, "pairs": agg}


def fig_deph(D, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ds in zip(axs, ("msar", "smsar16")):
        rows = D.get(ds, {})
        for m, c in (("m1", OI["vermillion"]), ("m2", OI["orange"])):
            pts = sorted((float(mo.split("-g")[1]), r) for mo, r in rows.items() if mo.startswith(f"cqsm-{m}") and "-g" in mo)
            if pts:
                ax.errorbar([g for g, _ in pts], [r["kl"] for _, r in pts], yerr=[2 * r["se"] for _, r in pts], marker="o",
                            color=c, capsize=2, label=f"C-QSM {m}, dephasing gamma (retrained)")
            u = rows.get(f"cqsm-{m}-anc2-u")
            if u:
                ax.errorbar([1.05], [u["kl"]], yerr=[2 * u["se"]], marker="s", color=c, ls="none", capsize=2,
                            label=f"C-QSM {m}, full dephasing in a trained basis")
        if ax.has_data():
            ax.set_yscale("log")
        ax.set_xlabel("dephasing gamma")
        ax.set_title(f"{ds}: retrained dephasing")
    axs[0].set_ylabel("held-out KL rate (nats/step)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_rep(Rp, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    series = (("coherent C-QSM", ("cqsm-m1-anc2-g0", "cqsm-m2-anc2-g0"), OI["vermillion"], "o"),
              ("dephased C-QSM (retrained)", ("cqsm-m1-anc2-g1", "cqsm-m2-anc2-g1"), OI["sky"], "D"),
              ("classical switch", ("sw-anc2-S2", "sw-anc2-S4"), OI["blue"], "^"),
              ("IOHMM", ("iohmm-fact-2", "iohmm-fact-4"), OI["yellow"], "<"))
    for ax, base in zip(axs, ("msar", "smsar16")):
        per = Rp.get(base, {}).get("datasets", {})
        for k, (lab, mods, c, mk) in enumerate(series):
            for j, mo in enumerate(mods):
                kl = [per[ds][mo]["kl"] for ds in per if per[ds].get(mo)]
                x = (2, 4)[j] * (1 + 0.04 * (k - 1.5))
                if kl:
                    ax.scatter([x] * len(kl), kl, color=c, marker=mk, s=18, label=lab if j == 0 else None)
        fam = [per[ds]["family-msar"]["kl"] for ds in per if per[ds].get("family-msar")]
        if fam:
            ax.axhline(np.median(fam), color=OI["green"], lw=0.8, ls="--", label="MS-AR fit (median)")
        if ax.has_data():
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
        ax.set_xticks([2, 4], ["D = 2", "D = 4"])
        ax.set_title(f"{base}: 5 datasets (best of 5 inits each)")
    axs[0].set_ylabel("held-out KL rate (nats/step)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "seq8"))
    ap.add_argument("--figs", default=str(ROOT / "figs"))
    a = ap.parse_args()
    runs = Path(a.runs)
    F = load(runs)
    table = [b for k in sorted(F) if (b := best(F, *k))]
    S = {"fits": sum(len(v) for v in F.values()), "best": table, "dephasing": deph(F), "replication": rep(F),
         "inits": inits(F)}
    (runs / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    cols = ["exp", "dataset", "model", "init", "n_inits", "kl", "se", "train_minus_truth", "kl_inits_min",
            "kl_inits_max", "n_params", "filter_dim", "capped", "iter_limit"]
    with open(runs / "tables.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for b in table:
            w.writerow([b[c] for c in cols])
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    fig_deph(S["dephasing"], figs / "seq8_dephasing.png")
    fig_rep(S["replication"], figs / "seq8_replication.png")
    print(f"{S['fits']} fits")
    for base, r in S["replication"].items():
        for k, p in r["pairs"].items():
            print(f"  {base} {k}: mean diff {p['mean_diff']:+.2e} over {p['n_datasets']} datasets, reference better "
                  f"(> 2 SE) on {p['ref_better_2se']}, other better on {p['other_better_2se']}")
    for label, p in S["inits"]["pairs"].items():
        print(f"  smsar16 dephased vs coherent m = 1, {label} (inits {p['n_inits']}): mean {p['mean_diff']:+.2e}, "
              f"coherent better on {p['coherent_better']}/{p['n_datasets']} ({p['coherent_better_2se']} at > 2 SE), "
              f"dephased better at > 2 SE on {p['dephased_better_2se']}")


if __name__ == "__main__":
    main()
