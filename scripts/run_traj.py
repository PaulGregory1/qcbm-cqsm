"""Generated trajectories and regime switches of the sequence 4 / 5 fits on msar and smsar16 (archb.pathstats), for the
models of scripts/run_memdiag.py (the selected fit of each: best training likelihood).

  paths     100 independent paths of 2,000 steps per model from its stationary law; marginal, acf of x, x^2, |x|,
            exceedances of |x| above the truth's 95% and 99% quantiles (rate, cluster length, waiting time until the
            next exceedance); discrepancy to a truth sample of the same size, against the truth-to-truth discrepancy of
            20 more independent truth samples (null band)
  forecast  exact law of x_{t+h}, h in {1, 5, 10, 25, 50, 100}, from the filtered state at 500 origins of the second
            half of the held-out path: TV to the truth's law, log score of the realized value, 90% interval coverage
  switch    excess log-loss e_t = log p_*(x_t | past) - log q(x_t | past) aligned on the true switches (window
            [-20, 50], calm -> turbulent and back, all and isolated switches); detection of the new regime from a
            regime score (exact filter, probes of memdiag on x_t or on x_t + memory, fitted on the first half): delay,
            missed switches and false alarms per 1,000 steps on the second half for eta x r settings, and the setting
            chosen on the first half at false-alarm budgets of 1, 2 and 5 per 1,000 steps
-> runs/traj/<dataset>/summary.json, figs/traj_<dataset>_{paths,forecast,switch}.png

Usage: python scripts/run_traj.py [--datasets msar smsar16] [--workers 2]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_memdiag as MDS  # noqa: E402
import run_seq4 as RS  # noqa: E402

from archb import memdiag as MD, pathstats as PS, regimes as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIGS = ROOT / "runs" / "traj", ROOT / "figs"
N_PATHS, N_STEPS, N_NULL = 100, 2000, 20
HORIZONS = (1, 5, 10, 25, 50, 100)
N_ORIGINS = 500
ETAS, RS_ = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99), (1, 2, 3, 5)
FA_BUDGETS = (1.0, 2.0, 5.0)
LABEL, COL, DPI = MDS.LABEL, MDS.COL, MDS.DPI


def thresholds(truth):
    """|x| levels whose exceedance probability under the truth's marginal is at most 5% and 1%."""
    pi, a = truth.marginal(), np.abs(R.Z)
    levels = np.unique(a)
    tail = np.array([pi[a > u].sum() for u in levels])
    return {"q95": float(levels[np.argmax(tail <= 0.05)]), "q99": float(levels[np.argmax(tail <= 0.01)])}


def run(ds):
    t_start = time.time()
    s, d = RS.DATASETS[ds], RS.load(ds)
    xs, regime = d["test"], R.regime_of(s["kind"], s["N"])
    st = regime[d["test_h"]]
    mods = MDS.models(ds)
    truth = mods["truth"]["lin"]
    thr = thresholds(truth)
    res = {"dataset": ds, "thresholds": thr, "n_paths": N_PATHS, "n_steps": N_STEPS, "n_null": N_NULL}

    # ---------------------------------------------------------------------------------- free generation
    ref = PS.path_stats(PS.sample_paths(truth, N_PATHS, N_STEPS, 1), R.Z, thr)
    null = [PS.discrepancies(PS.path_stats(PS.sample_paths(truth, N_PATHS, N_STEPS, 100 + r), R.Z, thr), ref, thr)
            for r in range(N_NULL)]
    res["paths"] = {"truth_ref": ref, "null": {k: {"median": float(np.median([n[k] for n in null])),
                                                   "q95": float(np.quantile([n[k] for n in null], 0.95))}
                                               for k in null[0]}, "models": {}}
    for name, mo in mods.items():
        if name == "truth":
            continue
        stt = PS.path_stats(PS.sample_paths(mo["lin"], N_PATHS, N_STEPS, 2), R.Z, thr)
        res["paths"]["models"][name] = {"stats": stt, "disc": PS.discrepancies(stt, ref, thr)}
    print(f"  {ds} paths done [{time.time() - t_start:.0f} s]", flush=True)

    # ---------------------------------------------------------------------------------- conditional forecasts
    h = len(xs) // 2
    orig = np.linspace(h, len(xs) - max(HORIZONS) - 1, N_ORIGINS).astype(int)
    states = {name: mo["lin"].filtered(xs) for name, mo in mods.items()}
    Ft = PS.forecast_laws(truth, states["truth"][orig], xs[orig], HORIZONS)
    res["forecast"] = {"horizons": list(HORIZONS), "n_origins": N_ORIGINS, "models": {}}
    for name, mo in mods.items():
        Fm = Ft if name == "truth" else PS.forecast_laws(mo["lin"], states[name][orig], xs[orig], HORIZONS)
        res["forecast"]["models"][name] = {str(H): PS.forecast_scores(Fm[H], Ft[H], xs[orig + H]) for H in HORIZONS}
    print(f"  {ds} forecasts done [{time.time() - t_start:.0f} s]", flush=True)

    # ---------------------------------------------------------------------------------- switches
    lp_true = truth.logprobs(xs)
    tau, new = PS.switch_times(st)
    tau2 = tau[tau >= h]
    new2 = st[tau2]
    res["switch"] = {"n_switches_test": int(len(tau2)), "aligned": {}, "detection": {}}
    for name, mo in mods.items():
        if name == "truth":
            continue
        e = lp_true - mo["lin"].logprobs(xs)
        r = {}
        for lab, sel in (("to_turbulent", new2 == 1), ("to_calm", new2 == 0)):
            for iso in (False, True):
                curve, n = PS.aligned(e, tau2[sel], isolated_from=tau if iso else None)
                r[f"{lab}{'_isolated' if iso else ''}"] = {"curve": curve.tolist(), "n": n}
        res["switch"]["aligned"][name] = r
    q = MD.regime_posterior(truth, xs, regime)
    Fx = MD.features(xs)
    scores = {"exact_filter": q, "probe_x_only": MD.fit_probe(Fx[:h], q[:h])(Fx)}
    for name, mo in mods.items():
        if name == "truth" or mo["lin"].d == 1:
            continue
        F = MD.features(xs, states[name])
        scores[f"probe_{name}"] = MD.fit_probe(F[:h], q[:h])(F)
    for name, sc in scores.items():
        grid_tr = [PS.detection(sc[:h], st[:h], e_, r_) for e_ in ETAS for r_ in RS_]
        grid_te = [PS.detection(sc[h:], st[h:], e_, r_) for e_ in ETAS for r_ in RS_]
        chosen = {}
        for b in FA_BUDGETS:
            ok = [i for i, g in enumerate(grid_tr) if g["fa_per_1000"] <= b and np.isfinite(g["delay_cens"])]
            if ok:
                i = min(ok, key=lambda i: grid_tr[i]["delay_cens"])
                chosen[str(b)] = grid_te[i]
        res["switch"]["detection"][name] = {"grid_test": grid_te, "chosen": chosen}
    print(f"  {ds} switches done [{time.time() - t_start:.0f} s]", flush=True)
    res["wall_seconds"] = time.time() - t_start
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    (OUT / ds / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    figures(res)
    return ds


# -------------------------------------------------------------------------------------------------------- figures
def figures(res):
    ds = res["dataset"]
    FIGS.mkdir(exist_ok=True)
    P = res["paths"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    lags = np.arange(1, len(P["truth_ref"]["acf_abs"]) + 1)
    axs[0].plot(lags, P["truth_ref"]["acf_abs"], color=OI_BLACK, lw=2.2, label="truth")
    sv = np.arange(1, len(P["truth_ref"]["exc_q99"]["survival"]) + 1)
    axs[1].plot(sv, P["truth_ref"]["exc_q99"]["survival"], color=OI_BLACK, lw=2.2, label="truth")
    for name, r in P["models"].items():
        axs[0].plot(lags, r["stats"]["acf_abs"], color=COL[name], lw=1, label=LABEL[name])
        axs[1].plot(sv, r["stats"]["exc_q99"]["survival"], color=COL[name], lw=1, label=LABEL[name])
    axs[0].set_xlabel("lag")
    axs[0].set_ylabel("acf of |x|")
    axs[0].set_title(f"{ds}: volatility clustering, free generation")
    axs[1].set_xlabel("steps j")
    axs[1].set_ylabel("P(no |x| > u99 within j steps)")
    axs[1].set_title(f"waiting time to the next exceedance (u = {res['thresholds']['q99']:.2f})")
    axs[0].legend(fontsize=7)
    keys = ["cdf_ks", "var_rel", "acf_x_max", "acf_x2_max", "acf_abs_max", "exc_q95_rate_rel", "exc_q99_rate_rel",
            "exc_q95_cluster", "exc_q99_survival_sup"]
    names = list(P["models"])
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        v = [P["models"][name]["disc"][k] / max(P["null"][k]["q95"], 1e-12) for k in keys]
        axs[2].bar(np.arange(len(keys)) + i * w, v, w, color=COL[name], label=LABEL[name])
    axs[2].axhline(1, color="grey", lw=0.8, ls="--")
    axs[2].set_yscale("log")
    axs[2].set_xticks(np.arange(len(keys)) + 0.4 - w / 2, keys, rotation=45, ha="right", fontsize=7)
    axs[2].set_ylabel("discrepancy / truth-to-truth 95% quantile")
    axs[2].set_title("path statistics (> 1: outside sampling noise)")
    fig.tight_layout()
    fig.savefig(FIGS / f"traj_{ds}_paths.png", dpi=DPI)
    plt.close(fig)

    Fc = res["forecast"]
    H = Fc["horizons"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in Fc["models"].items():
        if name == "truth":
            continue
        axs[0].errorbar(H, [r[str(h)]["tv"] for h in H], yerr=[2 * r[str(h)]["tv_se"] for h in H], marker="o",
                        color=COL[name], capsize=2, label=LABEL[name])
        axs[1].errorbar(H, [r[str(h)]["logscore_truth"] - r[str(h)]["logscore"] for h in H],
                        yerr=[2 * r[str(h)]["logscore_gap_se"] for h in H], marker="o", color=COL[name], capsize=2)
        axs[2].plot(H, [r[str(h)]["coverage"] for h in H], "o-", color=COL[name])
    axs[2].plot(H, [Fc["models"]["truth"][str(h)]["coverage"] for h in H], "o-", color=OI_BLACK, lw=2, label="truth")
    axs[2].axhline(0.9, color="grey", lw=0.6, ls="--")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("horizon h")
    axs[0].set_ylabel("TV to the truth's law of $x_{t+h}$")
    axs[1].set_ylabel("log-score gap to the truth (nats)")
    axs[2].set_ylabel("90% interval coverage")
    axs[0].legend(fontsize=7)
    axs[0].set_title(f"{ds}: conditional forecasts ({Fc['n_origins']} origins)")
    fig.tight_layout()
    fig.savefig(FIGS / f"traj_{ds}_forecast.png", dpi=DPI)
    plt.close(fig)

    S = res["switch"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    j = np.arange(-20, 51)
    for k, (ax, lab) in enumerate(zip(axs[:2], ("to_turbulent", "to_calm"))):
        for name, r in S["aligned"].items():
            ax.plot(j, r[lab]["curve"], color=COL[name], lw=1, label=LABEL[name] if k == 0 else None)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xlabel("steps from the true switch")
        ax.set_title(f"{ds}: excess loss, {lab.replace('_', ' ')} (n = {next(iter(S['aligned'].values()))[lab]['n']})")
    axs[0].set_ylabel("mean excess log-loss (nats)")
    axs[0].legend(fontsize=7)
    for name, r in S["detection"].items():
        g = r["grid_test"]
        pts = sorted((x["fa_per_1000"], x["delay_cens"]) for x in g if np.isfinite(x["delay_cens"]))
        front, best = [], np.inf
        for fa, dl in pts:  # lower envelope: the best delay at each false-alarm rate
            if dl < best:
                front.append((fa, dl))
                best = dl
        key = name.replace("probe_", "")
        c = OI_BLACK if name == "exact_filter" else "0.55" if name == "probe_x_only" else COL.get(key, "0.3")
        lab = {"exact_filter": "exact filter", "probe_x_only": "probe on $x_t$ only"}.get(name, f"probe on {LABEL.get(key, key)}")
        if front:
            axs[2].plot(*zip(*front), "o-", ms=3, color=c, label=lab)
    axs[2].set_xscale("symlog", linthresh=0.1)
    axs[2].set_xlabel("false alarms per 1,000 steps")
    axs[2].set_ylabel("mean delay, misses counted as 50 (steps)")
    axs[2].set_title("switch detection, second half")
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"traj_{ds}_switch.png", dpi=DPI)
    plt.close(fig)


OI_BLACK = "#000000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["msar", "smsar16"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if a.figures_only:
        for ds in a.datasets:
            figures(json.loads((OUT / ds / "summary.json").read_text()))
        return
    with ProcessPoolExecutor(min(a.workers, len(a.datasets))) as ex:
        for ds in ex.map(run, a.datasets):
            print(f"{ds} done", flush=True)


if __name__ == "__main__":
    main()
