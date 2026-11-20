"""Sequence 11 summary: the persistence / contrast grid of scripts/run_seq11.py, with the centre point (A = 0.98,
sigma = (0.3, 1.0)) taken from the five msar datasets of the sequence 8 replication (runs/seq8/rep/).

Per dataset, for the selected fit (best training likelihood) of each model, on the held-out path of 200,000 steps:
  kl         held-out KL rate ± 20-segment SE, and paired differences C-QSM - (switch, dephased C-QSM, IOHMM_2)
  info       G_0 = I(Y; past | X_t) of the truth (oracle memoryless predictor, archb.memdiag), G_theta and
             R_theta = 1 - G_theta / G_0 (undefined when G_0 < 1e-4 or 4 SE: the A = 0.5 control)
  encoded    chi_{H:M|X} and the predictive information of each memory (archb.memdiag, as scripts/run_memdiag.py)
  posterior  excess KL of a least-squares probe on x_t + memory (+ x_t x memory products) to the exact regime
             posterior (second half)
  switch     excess log-loss summed over the 10 steps after a true switch, minus 10 x its mean over the path
             (calm -> turbulent and back, second half); detection delay (misses counted as 50) and missed fraction at
             <= 1 false alarm per 1,000 steps, the setting chosen on the first half (archb.pathstats; not for A = 0.5,
             where the regime changes every other step)
Grid points: mean and range over their datasets. -> runs/seq11/summary.json, runs/seq11/tables.csv, figs/seq11_grid.png

Usage: python scripts/seq11_summary.py [--workers 4]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_memdiag as MDS  # noqa: E402
import run_seq11 as S11  # noqa: E402

from archb import cqsm, hmm, iolin, memdiag as MD, pathstats as PS, regimes as R, switch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIGS = ROOT / "runs" / "seq11", ROOT / "figs"
CENTRE = {"msar": 700, **{f"msar-d{d}": 800 + 10 * d for d in range(1, 5)}}  # as scripts/run_seq8.py
MODELS = ("family-msar", "iohmm-fact-2", "sw-anc2-S2", "cqsm-m1-anc2-g0", "cqsm-m1-anc2-g1", "archb-anc2")
LABEL = {"family-msar": "MS-AR fit", "iohmm-fact-2": "IOHMM S = 2", "sw-anc2-S2": "classical switch S = 2",
         "cqsm-m1-anc2-g0": "C-QSM m = 1", "cqsm-m1-anc2-g1": "C-QSM m = 1, dephased", "archb-anc2": "no memory"}
COL = {"family-msar": MDS.OI["green"], "iohmm-fact-2": MDS.OI["yellow"], "sw-anc2-S2": MDS.OI["blue"],
       "cqsm-m1-anc2-g0": MDS.OI["vermillion"], "cqsm-m1-anc2-g1": MDS.OI["sky"], "archb-anc2": MDS.OI["purple"]}
ETAS, RS_ = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99), (1, 2, 3, 5)
FA_BUDGET, AFTER = 1.0, 10


def point(ds):
    """(grid point, stay, sigma) of a dataset."""
    if ds in CENTRE:
        return "centre", 0.98, (0.3, 1.0)
    stay, sig = S11.PARAMS[ds]
    return ds.rsplit("-d", 1)[0], stay, sig


def data_and_truth(ds):
    if ds in CENTRE:
        p = ROOT / "runs" / ("seq4" if ds == "msar" else "seq8") / "data" / f"{ds}.npz"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **R.make_data("msar", None, CENTRE[ds]))
        T, runs = R.truth_T("msar"), ROOT / "runs" / "seq8" / "rep" / ds
    else:
        p = OUT / "data" / f"{ds}.npz"
        if not p.exists():
            S11.prepare_data([ds])
        T, runs = S11.truth_T(ds), OUT / "grid" / ds
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    return d, iolin.IOLin.from_iohmm(T), runs


def rebuild(model, prm):
    if model.startswith("cqsm"):
        return cqsm.io(prm["theta"], 1, 2, gamma=float(model.split("-g")[1]))
    if model.startswith("archb"):
        return cqsm.io(prm["theta"], 0, 2)
    if model.startswith("sw"):
        return switch.io(prm["theta"], 2, "anc2")
    return iolin.IOLin.from_iohmm(prm["T"])


def best(d):
    fits = [json.loads(p.read_text()) | {"dir": p.parent} for p in d.glob("*/result.json")]
    if not fits:
        return None
    r = max(fits, key=lambda r: r["train_ll"])
    with np.load(r["dir"] / "params.npz") as z:
        prm = {k: z[k] for k in z.files}
    return r, prm, len(fits)


def run(ds):
    d, truth, runs = data_and_truth(ds)
    xs, st = d["test"], d["test_h"]
    h = len(xs) // 2
    lp_true = truth.logprobs(xs)
    g0 = lp_true[1:] - MD.window_logprobs(truth, xs, 0, 1)
    G0 = MD.seg_mean(g0)
    pt, stay, sig = point(ds)
    res = {"dataset": ds, "point": pt, "stay": stay, "sigma": list(sig), "G_0": G0, "models": {}}
    mods, lps = {"truth": {"lin": truth}}, {}
    for model in MODELS:
        b = best(runs / model)
        if b is None:
            continue
        r, prm, n = b
        lin = rebuild(model, prm)
        lp = lin.logprobs(xs)
        dth = lp_true[1:] - lp[1:]
        mods[model], lps[model] = {"lin": lin, "m": 1 if model.startswith("cqsm") else None}, lp
        res["models"][model] = {"init": r["init"], "n_inits": n, "n_params": r.get("n_params"),
                                "capped": bool(r.get("capped")), "kl": hmm.seg_kl_rate(lp_true, lp),
                                "kl_stored": r["heldout_kl_rate"], "G_theta": MD.seg_mean(dth),
                                "R_theta": float(1 - dth.mean() / g0.mean()) if G0[0] > max(4 * G0[1], 1e-4) else None}
    q = "cqsm-m1-anc2-g0"
    if q in lps:
        res["paired"] = {o: hmm.seg_kl_rate(lps[o], lps[q]) for o in ("sw-anc2-S2", "cqsm-m1-anc2-g1",
                                                                       "iohmm-fact-2") if o in lps}
    enc = MDS.encoded_info({k: v for k, v in mods.items()}, xs)
    # posterior probes and switches
    qs = MD.regime_posterior(truth, xs, np.arange(2))
    Fx = MD.features(xs)
    scores = {"exact_filter": qs, "x_only": MD.fit_probe(Fx[:h], qs[:h])(Fx)}
    for model, mo in mods.items():
        if model == "truth" or mo["lin"].d == 1:
            continue
        F = MD.features(xs, mo["lin"].filtered(xs), interact=True)
        scores[model] = MD.fit_probe(F[:h], qs[:h])(F)
        res["models"][model]["posterior_kl"] = MD.posterior_scores(scores[model][h:], qs[h:], st[h:])["kl"]
        res["models"][model]["encoded"] = enc.get(model)
    res["truth_encoded"] = enc["truth"]
    tau, new = PS.switch_times(st)
    sel = tau >= h
    for model, lp in lps.items():
        e = lp_true - lp
        base = float(e[h:].mean())
        sw = {}
        for lab, v in (("to_turbulent", 1), ("to_calm", 0)):
            t_ = tau[sel & (new == v)]
            t_ = t_[t_ + AFTER < len(e)]
            sw[lab] = float(np.mean([e[t:t + AFTER].sum() for t in t_]) - AFTER * base) if len(t_) else None
        res["models"][model]["switch_excess"] = sw
    res["detection"] = {}
    if stay > 0.5:
        for name, sc in scores.items():
            grid_tr = [PS.detection(sc[:h], st[:h], e_, r_) for e_ in ETAS for r_ in RS_]
            ok = [i for i, g in enumerate(grid_tr) if g["fa_per_1000"] <= FA_BUDGET and np.isfinite(g["delay_cens"])]
            if ok:
                i = min(ok, key=lambda i: grid_tr[i]["delay_cens"])
                res["detection"][name] = PS.detection(sc[h:], st[h:], ETAS[i // len(RS_)], RS_[i % len(RS_)])
    print(f"  {ds}: G_0 {G0[0]:.4f}, " + ", ".join(f"{k} {v['kl'][0]:.2e}" for k, v in res["models"].items()),
          flush=True)
    return res


# -------------------------------------------------------------------------------------------------------- aggregate
def aggregate(rows):
    """grid point -> per quantity: mean, min, max over its datasets."""
    out = {}
    for pt in dict.fromkeys(r["point"] for r in rows):
        rs = [r for r in rows if r["point"] == pt]
        a = {"stay": rs[0]["stay"], "sigma": rs[0]["sigma"], "datasets": [r["dataset"] for r in rs]}

        def agg(vals):
            v = [x for x in vals if x is not None]
            return {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)} if v else None

        a["G_0"] = agg([r["G_0"][0] for r in rs])
        a["chi_truth"] = agg([r["truth_encoded"]["chi_second_half"] for r in rs])
        a["models"] = {}
        for mo in MODELS:
            m = [r["models"][mo] for r in rs if mo in r["models"]]
            if not m:
                continue
            a["models"][mo] = {
                "kl": agg([x["kl"][0] for x in m]), "R_theta": agg([x["R_theta"] for x in m]),
                "posterior_kl": agg([x.get("posterior_kl") for x in m]),
                "chi": agg([x["encoded"]["chi_second_half"] if x.get("encoded") else None for x in m]),
                "I_true": agg([x["encoded"]["predictive"]["true"][0] if x.get("encoded") else None for x in m]),
                "switch_to_turbulent": agg([x["switch_excess"]["to_turbulent"] for x in m]),
                "switch_to_calm": agg([x["switch_excess"]["to_calm"] for x in m]),
                "delay": agg([r["detection"].get(mo, {}).get("delay_cens") for r in rs]),
                "missed": agg([r["detection"].get(mo, {}).get("missed") for r in rs]),
                "any_capped": any(x["capped"] for x in m)}
        for mo in ("exact_filter", "x_only"):
            a[mo] = {"delay": agg([r["detection"].get(mo, {}).get("delay_cens") for r in rs]),
                     "missed": agg([r["detection"].get(mo, {}).get("missed") for r in rs])}
        pr = [r["paired"] for r in rs if "paired" in r]
        a["paired_cqsm_minus"] = {o: {"diffs": [p[o][0] for p in pr if o in p], "se": [p[o][1] for p in pr if o in p],
                                      "cqsm_better_2se": sum(p[o][0] < -2 * p[o][1] for p in pr if o in p),
                                      "other_better_2se": sum(p[o][0] > 2 * p[o][1] for p in pr if o in p)}
                                  for o in ("sw-anc2-S2", "cqsm-m1-anc2-g1", "iohmm-fact-2")}
        out[pt] = a
    return out


def figure(A, path):
    pers = sorted([(v["stay"], k) for k, v in A.items() if v["sigma"] == [0.3, 1.0]])
    cont = sorted([(v["sigma"][1] / v["sigma"][0], k) for k, v in A.items() if v["stay"] == 0.98])
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for row, (pts, xlab, xf) in enumerate(((pers, "mean regime duration 1 / (1 - A)", lambda s: 1 / (1 - s)),
                                           (cont, "volatility contrast σ_turbulent / σ_calm", lambda r: r))):
        x = np.array([xf(p[0]) for p in pts])
        keys = [p[1] for p in pts]
        for mo in MODELS:
            for ax, q in zip(axs[row], ("kl", "R_theta", "delay")):
                v = [A[k]["models"].get(mo, {}).get(q) for k in keys]
                ok = [i for i, z in enumerate(v) if z]
                if not ok:
                    continue
                m = np.array([v[i]["mean"] for i in ok])
                lo, hi = m - [v[i]["min"] for i in ok], [v[i]["max"] for i in ok] - m
                ax.errorbar(x[ok], m, yerr=[lo, hi], marker="o", ms=4, capsize=2, color=COL[mo], label=LABEL[mo])
        g = np.array([A[k]["G_0"]["mean"] for k in keys])
        ok = g > 1e-6  # A = 0.5: G_0 = 0 up to rounding, off the log axis
        axs[row, 0].plot(x[ok], g[ok], "k--", marker="s", ms=4, label="$G_0$ (oracle memoryless loss; 0 at A = 0.5)")
        for mo, ls in (("exact_filter", "-"), ("x_only", ":")):
            v = [A[k][mo]["delay"] for k in keys]
            ok = [i for i, z in enumerate(v) if z]
            if ok:
                axs[row, 2].plot(x[ok], [v[i]["mean"] for i in ok], "k" + ls, marker="s", ms=4,
                                 label="exact filter" if mo == "exact_filter" else "probe on $x_t$ only")
        axs[row, 0].set_yscale("log")
        axs[row, 0].set_ylabel("held-out KL rate (nats/step)")
        axs[row, 1].set_ylabel("$R_θ = 1 - G_θ / G_0$")
        axs[row, 1].axhline(1, color="grey", lw=0.6)
        axs[row, 2].set_ylabel(f"detection delay at ≤ {FA_BUDGET:g} false alarm / 1,000 steps\n(misses = 50)")
        for ax in axs[row]:
            ax.set_xlabel(xlab)
            if row == 0:
                ax.set_xscale("log")
    axs[0, 0].legend(fontsize=7)
    axs[0, 2].legend(fontsize=7, loc="lower left")
    axs[0, 0].set_title("msar, A varied (σ = 0.3 / 1.0)", fontsize=10)
    axs[1, 0].set_title("msar, contrast varied (A = 0.98)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=MDS.DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    names = a.datasets or list(CENTRE) + list(S11.DATASETS)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, names))
    A = aggregate(rows)
    S = {"datasets": rows, "points": A}
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    with open(OUT / "tables.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point", "stay", "sigma_calm", "sigma_turbulent", "model", "kl_mean", "R_theta_mean", "chi_mean",
                    "I_true_mean", "posterior_kl_mean", "delay_mean", "missed_mean", "n_datasets"])
        for pt, v in A.items():
            w.writerow([pt, v["stay"], *v["sigma"], "G_0", v["G_0"]["mean"], "", v["chi_truth"]["mean"], "", "", "",
                        "", v["G_0"]["n"]])
            for mo, x in v["models"].items():
                g = lambda q: x[q]["mean"] if x[q] else ""  # noqa: E731
                w.writerow([pt, v["stay"], *v["sigma"], mo, g("kl"), g("R_theta"), g("chi"), g("I_true"),
                            g("posterior_kl"), g("delay"), g("missed"), x["kl"]["n"]])
    FIGS.mkdir(exist_ok=True)
    figure(A, FIGS / "seq11_grid.png")
    for pt, v in A.items():
        p = v["paired_cqsm_minus"]["sw-anc2-S2"]
        print(f"{pt:8s} G_0 {v['G_0']['mean']:.4f}  C-QSM - switch: better {p['cqsm_better_2se']}, worse "
              f"{p['other_better_2se']} of {len(p['diffs'])} at 2 SE")


if __name__ == "__main__":
    main()
