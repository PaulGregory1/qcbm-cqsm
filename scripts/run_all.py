"""Step 5: every run, runs/summary.json, the four figures per encoding and reports/go-no-go.md.

Configurations (2 encodings x seeds 0, 1, 2 each):
  primary           the spec unchanged: lr 0.03, eps 0.01, noise 0.2, 25 warm-started Sinkhorn
                    iterations per step, 800 steps with the spec stop rule. The verdict uses it.
  converged-solver  identical but 100 Sinkhorn iterations per step (a deviation, reported alongside).
Diagnostic (NOT part of the verdict): the first fixes of the spec's troubleshooting table, each a
single change from the primary configuration: eps 0.005, noise 0.1, one extra layer (4 layers).
Step 6 (optional extra, NOT part of the verdict): the single-register 5-qubit model
(archb.single_register, 44 angles), primary settings otherwise, under runs/single_register/.

Outputs: runs/<config>/<encoding>/seed<k>/{theta,kernel,loss}.npy + metrics.json,
runs/diagnostic/<variant>/<encoding>/seed<k>/..., runs/single_register/<encoding>/seed<k>/...,
runs/summary.json, figs/<encoding>_*.png, figs/single_register_stationary.png, reports/go-no-go.md.

Run: ./.venv/bin/python scripts/run_all.py            (trains everything, ~8 min on a CPU)
     ./.venv/bin/python scripts/run_all.py --reuse    (reuses saved runs whose settings match)
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from archb import sinkhorn as sk  # noqa: E402
from archb.metrics import (  # noqa: E402
    ACF_LAGS, LAG1_TOL, SLOPE_TOL, STAT_TV_MAX, TAIL_Z, WTV_IID_RATIO, acf, all_metrics, cond_mean_sd,
    go_no_go, row_tv,
)
from archb.ou import CENTRES, EDGES, ENCODINGS, N_BINS, OU, load_dataset, stationary  # noqa: E402
from archb.references import REFERENCE_NAMES, reference_kernels  # noqa: E402
from archb.train import save_run, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIGS, REPORTS = ROOT / "runs", ROOT / "figs", ROOT / "reports"
SEEDS = (0, 1, 2)
STEPS = 800
BASE = {"lr": 0.03, "eps": 0.01, "noise": 0.2, "sinkhorn_iters": 25, "n_layers": 3}
CONFIGS = {
    "primary": {},
    "converged-solver": {"sinkhorn_iters": 100},
}
DIAGNOSTIC = {
    "eps0.005": {"eps": 0.005},
    "noise0.1": {"noise": 0.1},
    "layers4": {"n_layers": 4},
}
SINGLE = {"model": "single"}  # step 6: single-register model, primary settings otherwise
TIGHT = {"weighted_tv": 0.05, "stationary_tv": 0.02}  # alternative thresholds, not the verdict
METRIC_KEYS = ("weighted_tv", "tail_tv", "stationary_tv", "lag1_acf", "acf_err", "slope")

# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
C_TRUTH, C_QCBM, C_AR1, C_IID = OI["black"], OI["vermillion"], OI["blue"], OI["sky"]
SEED_COLORS = (OI["orange"], OI["blue"], OI["green"])
DPI = 170

# Boundary b | b+1 (b = 0..30): number of bits a binary +1 flips = position of the Gray bit flipped.
DEPTH = np.array([bin(b ^ (b + 1)).count("1") for b in range(N_BINS - 1)])
BULK = np.abs(CENTRES) <= TAIL_Z
NEAR_DEEP = np.zeros(N_BINS, bool)  # rows adjacent to a boundary of depth >= 3
for _b in np.flatnonzero(DEPTH >= 3):
    NEAR_DEEP[[_b, _b + 1]] = True


# ------------------------------------------------------------------------------------- runs
def run_one(root: Path, enc: str, seed: int, settings: dict, reuse: bool, data: dict) -> dict:
    d = root / enc / f"seed{seed}"
    mpath = d / "metrics.json"
    if reuse and mpath.exists():
        saved = json.loads(mpath.read_text())["settings"]
        same = saved.get("steps_max") == STEPS and all(saved.get(k, 3 if k == "n_layers" else None) == v
                                                       for k, v in settings.items())
    else:
        same = False
    if not same:
        res = train(enc, seed, steps=STEPS, log_every=0, data=data, **settings)
        save_run(res, root)
        print(f"  trained {root.name}/{enc}/seed{seed}: {res['steps']} steps, {res['wall_time']:.1f} s, "
              f"S_eps {res['final_loss']:.5f}", flush=True)
    info = json.loads(mpath.read_text())
    info["kernel"] = np.load(d / "kernel.npy")
    info["loss"] = np.load(d / "loss.npy")
    return info


def boundary_flux(P, pi):
    """Two-way probability flux across each boundary b | b+1 under the weights pi, shape (31,)."""
    F = pi[:, None] * P
    return np.array([F[: b + 1, b + 1 :].sum() + F[b + 1 :, : b + 1].sum() for b in range(N_BINS - 1)])


def carry_diagnostic(P, P_star) -> dict:
    """Where the mass goes wrong: flux ratio model/truth per boundary depth (pi*-weighted, so
    the kernel is judged on the states the process visits), and pi*-weighted row TV of the bulk
    rows next to a deep boundary (depth >= 3) vs the other bulk rows."""
    pi = stationary(P_star)
    fm, fs = boundary_flux(P, pi), boundary_flux(P_star, pi)
    rtv = row_tv(P, P_star)
    out = {f"flux_ratio_depth{k}": float(fm[DEPTH == k].sum() / fs[DEPTH == k].sum()) for k in (5, 4, 3)}
    out["flux_ratio_depth1-2"] = float(fm[DEPTH <= 2].sum() / fs[DEPTH <= 2].sum())
    for name, mask in (("near_deep", BULK & NEAR_DEEP), ("other", BULK & ~NEAR_DEEP)):
        out[f"bulk_row_tv_{name}"] = float(pi[mask] @ rtv[mask] / pi[mask].sum())
    return out


def tight_checks(metrics_per_seed, c=OU.c) -> dict:
    """The spec's checks with weighted TV <= 0.05 (absolute) and stationary TV <= 0.02."""
    seeds = []
    for m in metrics_per_seed:
        checks = {
            "stationary_tv": m["stationary_tv"] <= TIGHT["stationary_tv"],
            "lag1_acf": abs(m["lag1_acf"] - m["lag1_acf_exact"]) <= LAG1_TOL,
            "slope": abs(m["slope"] - c) <= SLOPE_TOL,
            "weighted_tv": m["weighted_tv"] <= TIGHT["weighted_tv"],
        }
        seeds.append({"checks": checks, "pass": all(checks.values())})
    n_pass = sum(s["pass"] for s in seeds)
    return {"seeds": seeds, "n_seeds_passing": n_pass, "go": bool(n_pass >= 2)}


def stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def median_seed(runs: dict) -> int:
    """Seed whose weighted TV is the median of the three."""
    order = sorted(runs, key=lambda s: runs[s]["metrics"]["weighted_tv"])
    return order[len(order) // 2]


def single_register_checks(by_enc, P_star) -> dict:
    """Step 6 facts: TV(uniform, pi*) computed exactly, and per run the largest row-sum and
    column-sum errors (doubly stochastic) and the largest |pi - 1/32| of the stationary law."""
    pi_star = stationary(P_star)
    out = {"tv_uniform_vs_pi_star": float(0.5 * np.abs(pi_star - 1.0 / N_BINS).sum()), "runs": {}}
    for enc, by_seed in by_enc.items():
        out["runs"][enc] = {}
        for s, info in by_seed.items():
            P = info["kernel"]
            out["runs"][enc][f"seed{s}"] = {
                "row_sum_err": float(np.abs(P.sum(axis=1) - 1).max()),
                "col_sum_err": float(np.abs(P.sum(axis=0) - 1).max()),
                "stationary_max_dev_from_uniform": float(np.abs(stationary(P) - 1.0 / N_BINS).max())}
    return out


# ------------------------------------------------------------------------------------- figures
def fig_kernels(enc, P_star, P_qcbm, P_tab, seed, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), constrained_layout=True)
    norm = LogNorm(vmin=1e-4, vmax=1.0)
    ext = (EDGES[0], EDGES[-1], EDGES[0], EDGES[-1])
    titles = ("Exact P*", f"QCBM ({enc}, seed {seed})", "Tabular counts")
    for ax, P, t in zip(axes, (P_star, P_qcbm, P_tab), titles):
        im = ax.imshow(np.clip(P, 1e-4, None), origin="lower", extent=ext, cmap="viridis", norm=norm,
                       interpolation="nearest")
        ax.set_title(t)
        ax.set_xlabel(r"$x_{t+1}$")
        ax.axhline(0, color="w", lw=0.5, ls=":")
        ax.axvline(0, color="w", lw=0.5, ls=":")
    axes[0].set_ylabel(r"$x_t$")
    fig.colorbar(im, ax=axes, shrink=0.9, label=r"$P(x_{t+1} \mid x_t)$, log scale (clipped at $10^{-4}$)")
    fig.suptitle(f"Transition kernels, {enc} encoding, primary config "
                 f"(QCBM seed {seed} = median weighted TV of 3 seeds)", fontsize=11)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_cond_mean(enc, P_star, P_qcbm, P_ar1, seed, path):
    z = CENTRES
    (ms, ss), (mq, sq), (ma, sa) = cond_mean_sd(P_star), cond_mean_sd(P_qcbm), cond_mean_sd(P_ar1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(z, ms - ss, ms + ss, color="0.85", label="exact P* ± 1 sd")
    ax.plot(z, ms, color=C_TRUTH, lw=1.5, label="exact P* mean")
    ax.errorbar(z, mq, yerr=sq, color=C_QCBM, marker="o", ms=3, lw=1.2, capsize=2, label="QCBM ± 1 sd")
    ax.plot(z, ma, color=C_AR1, ls="--", lw=1.2, label="AR(1) mean")
    ax.plot(z, ma - sa, color=C_AR1, ls=":", lw=1)
    ax.plot(z, ma + sa, color=C_AR1, ls=":", lw=1, label="AR(1) ± 1 sd")
    ax.set_ylabel(r"$E[x_{t+1} \mid x_t]$")
    ax.set_title("Conditional mean ± 1 sd")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    ax.axhline(0, color=C_TRUTH, lw=1)
    ax.plot(z, mq - ms, color=C_QCBM, marker="o", ms=3, label="QCBM")
    ax.plot(z, ma - ms, color=C_AR1, ls="--", label="AR(1)")
    ax.set_ylabel(r"$E[x_{t+1} \mid x_t] - E_{P^*}[x_{t+1} \mid x_t]$")
    ax.set_title("Conditional-mean error vs exact")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(z, ss, color=C_TRUTH, lw=1.5, label="exact P*")
    ax.plot(z, sq, color=C_QCBM, marker="o", ms=3, label="QCBM")
    ax.plot(z, sa, color=C_AR1, ls="--", label="AR(1)")
    ax.set_ylabel(r"sd$[x_{t+1} \mid x_t]$")
    ax.set_title("Conditional sd")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(r"$x_t$ (bin centre)")
        ax.axvspan(EDGES[0], -TAIL_Z, color="0.95", zorder=0)
        ax.axvspan(TAIL_Z, EDGES[-1], color="0.95", zorder=0)
        ax.set_xlim(EDGES[0], EDGES[-1])
    fig.suptitle(f"Conditional moments, {enc} encoding, primary config (QCBM seed {seed} = median "
                 "weighted TV; shaded: tail rows |z| > 1.41)", fontsize=11)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_acf(enc, P_star, kernels_by_seed, seed_med, P_iid, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(ACF_LAGS, acf(P_star), color=C_TRUTH, lw=2, marker="o", ms=3, label="exact P*")
    for s, P in kernels_by_seed.items():
        if s != seed_med:
            ax.plot(ACF_LAGS, acf(P), color=C_QCBM, lw=1, alpha=0.35)
    ax.plot([], [], color=C_QCBM, lw=1, alpha=0.35, label="QCBM, other seeds")
    ax.plot(ACF_LAGS, acf(kernels_by_seed[seed_med]), color=C_QCBM, lw=2, marker="s", ms=3,
            label=f"QCBM, seed {seed_med} (median weighted TV)")
    ax.plot(ACF_LAGS, acf(P_iid), color=C_IID, lw=1.5, ls="--", label="i.i.d. oracle")
    ax.set_xlabel("lag k")
    ax.set_ylabel(r"corr$(x_t, x_{t+k})$")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f"Autocorrelation, {enc} encoding, primary config", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_loss(enc, losses, floor, path):
    """losses[config][seed] -> per-step loss array."""
    fig, axes = plt.subplots(1, len(losses), figsize=(11, 4.0), sharey=True, constrained_layout=True)
    for ax, (cfg, by_seed) in zip(axes, losses.items()):
        for (s, L), col in zip(by_seed.items(), SEED_COLORS):
            ax.plot(np.arange(len(L)), L, color=col, lw=1.1, label=f"seed {s} ({len(L)} steps)")
            ax.plot(len(L) - 1, L[-1], "o", color=col, ms=4)
        ax.axhline(floor, color=C_TRUTH, ls="--", lw=1, label=f"exact P* ({floor:.1e})")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        iters = BASE["sinkhorn_iters"] if cfg == "primary" else CONFIGS[cfg]["sinkhorn_iters"]
        ax.set_title(f"{cfg}: {iters} Sinkhorn its/step", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"tracked loss $S_\varepsilon(w(\theta), v)$")
    fig.suptitle(f"Training loss, {enc} encoding (dot = stop step; dashed = loss of the exact kernel)",
                 fontsize=11)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_single_register_stationary(P_star, P_cond, P_single, S, path):
    """Stationary law vs bin centre: pi*, the conditional QCBM (Architecture B, primary config,
    median seed) and the single-register model (median seed), both encodings."""
    z = CENTRES
    ms, msr = S["median_seed"]["primary"], S["median_seed"]["single_register"]
    stv = lambda g, e: S["runs"][g][e][f"seed{S['median_seed'][g][e]}"]["metrics"]["stationary_tv"]  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.axhline(1.0 / N_BINS, color="0.55", lw=1, ls=":", zorder=1)
    ax.annotate("1/32", (EDGES[-1], 1.0 / N_BINS), xytext=(-2, 7), textcoords="offset points", ha="right",
                fontsize=8, color="0.4")
    ax.plot(z, stationary(P_star), color=C_TRUTH, lw=2, marker="o", ms=3.5, label=r"exact $\pi^*$", zorder=5)
    styles = {"gray": (OI["vermillion"], "-", "s"), "binary": (OI["orange"], "--", "^")}
    for enc in ("gray", "binary"):
        col, ls, mk = styles[enc]
        ax.plot(z, stationary(P_cond[enc]), color=col, ls=ls, lw=1.4, marker=mk, ms=3.5, zorder=4,
                label=f"conditional QCBM (Arch. B), {enc}, seed {ms[enc]}: TV {stv('primary', enc):.3f}")
    styles = {"gray": (OI["green"], "-", "D", 4.5), "binary": (OI["blue"], "none", "x", 6)}
    for enc in ("gray", "binary"):
        col, ls, mk, sz = styles[enc]
        ax.plot(z, stationary(P_single[enc]), color=col, ls=ls, lw=1.4, marker=mk, ms=sz, mfc="none", zorder=3,
                label=f"single register, {enc}, seed {msr[enc]}: TV {stv('single_register', enc):.3f}")
    ax.set_xlabel(r"bin centre $z$")
    ax.set_ylabel(r"stationary probability $\pi(z)$")
    ax.set_xlim(EDGES[0], EDGES[-1])
    ax.set_ylim(0, 1.45 * stationary(P_star).max())  # headroom so the legend clears the curves
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Stationary law: the single-register kernel is doubly stochastic, so its law is flat at 1/32\n"
                 f"(TV to $\\pi^*$ = TV(uniform, $\\pi^*$) = {S['single_register']['tv_uniform_vs_pi_star']:.4f}; "
                 "primary config, median-weighted-TV seeds)", fontsize=10)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reuse", action="store_true", help="reuse saved runs whose settings match")
    args = ap.parse_args(argv)

    data = load_dataset()
    refs = reference_kernels(data)
    P_star = refs["exact"]
    ref_metrics = {n: all_metrics(refs[n], P_star) for n in REFERENCE_NAMES}
    for n in REFERENCE_NAMES:
        ref_metrics[n].update(carry_diagnostic(refs[n], P_star))

    pi_hat = jnp.asarray(data["pi_hat"], dtype=jnp.float64)
    N = np.asarray(data["N"], float)
    v = jnp.asarray(N / N.sum())
    floor = {eps: sk.divergence(pi_hat[:, None] * jnp.asarray(P_star), v, eps) for eps in (0.01, 0.005)}

    groups = {cfg: (RUNS / cfg, kw) for cfg, kw in CONFIGS.items()}
    groups.update({f"diagnostic/{name}": (RUNS / "diagnostic" / name, kw) for name, kw in DIAGNOSTIC.items()})
    groups["single_register"] = (RUNS / "single_register", SINGLE)
    loaded = {}
    for g, (root, kw) in groups.items():
        settings = {**BASE, **kw}
        print(f"[{g}] {settings}", flush=True)
        loaded[g] = {enc: {s: run_one(root, enc, s, settings, args.reuse, data) for s in SEEDS}
                     for enc in ENCODINGS}

    summary = {
        "settings": {"base": BASE, "steps_max": STEPS, "seeds": list(SEEDS), "configs": CONFIGS,
                     "diagnostic_variants": DIAGNOSTIC, "single_register": SINGLE,
                     "thresholds_spec": {"stationary_tv_max": STAT_TV_MAX, "lag1_tol": LAG1_TOL,
                                         "slope_tol": SLOPE_TOL, "weighted_tv_max": WTV_IID_RATIO
                                         * ref_metrics["iid"]["weighted_tv"], "seed_spread_max": 0.2},
                     "thresholds_tight": TIGHT},
        "references": ref_metrics,
        "loss_floor_exact": {"eps0.01": floor[0.01], "eps0.005": floor[0.005]},
        "runs": {}, "aggregate": {}, "go_no_go": {}, "tight": {}, "median_seed": {},
    }
    for g, by_enc in loaded.items():
        summary["runs"][g], summary["aggregate"][g] = {}, {}
        summary["go_no_go"][g], summary["tight"][g], summary["median_seed"][g] = {}, {}, {}
        for enc, by_seed in by_enc.items():
            runs = {}
            for s, info in by_seed.items():
                m = dict(info["metrics"])
                m.update(carry_diagnostic(info["kernel"], P_star))
                runs[s] = {"metrics": m, "steps": info["steps"], "stopped_early": info["steps"] < STEPS,
                           "final_loss": info["final_loss"], "init_loss": info["init_loss"],
                           "wall_time": info["wall_time"], "settings": info["settings"]}
            summary["runs"][g][enc] = {f"seed{s}": r for s, r in runs.items()}
            keys = list(runs[SEEDS[0]]["metrics"]) + ["final_loss", "steps"]
            summary["aggregate"][g][enc] = {
                k: stats([r["metrics"][k] if k in r["metrics"] else r[k] for r in runs.values()]) for k in keys}
            mlist = [runs[s]["metrics"] for s in SEEDS]
            gng = go_no_go(mlist, ref_metrics["iid"])
            gng["seeds"] = {f"seed{s}": x for s, x in zip(SEEDS, gng["seeds"])}
            summary["go_no_go"][g][enc] = gng
            tc = tight_checks(mlist)
            tc["seeds"] = {f"seed{s}": x for s, x in zip(SEEDS, tc["seeds"])}
            summary["tight"][g][enc] = tc
            summary["median_seed"][g][enc] = median_seed(runs)
    summary["single_register"] = single_register_checks(loaded["single_register"], P_star)
    summary["verdict"] = {
        g: {"go": any(summary["go_no_go"][g][e]["go"] for e in ENCODINGS),
            "go_encodings": [e for e in ENCODINGS if summary["go_no_go"][g][e]["go"]]}
        for g in CONFIGS}
    summary["verdict_tight"] = {
        g: {"go": any(summary["tight"][g][e]["go"] for e in ENCODINGS),
            "go_encodings": [e for e in ENCODINGS if summary["tight"][g][e]["go"]]}
        for g in CONFIGS}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {RUNS / 'summary.json'}; verdict (primary): {summary['verdict']['primary']}")

    FIGS.mkdir(exist_ok=True)
    for enc in ENCODINGS:
        prim = loaded["primary"][enc]
        sm = summary["median_seed"]["primary"][enc]
        fig_kernels(enc, P_star, prim[sm]["kernel"], refs["tabular"], sm, FIGS / f"{enc}_kernels.png")
        fig_cond_mean(enc, P_star, prim[sm]["kernel"], refs["ar1"], sm, FIGS / f"{enc}_cond_mean.png")
        fig_acf(enc, P_star, {s: prim[s]["kernel"] for s in SEEDS}, sm, refs["iid"], FIGS / f"{enc}_acf.png")
        fig_loss(enc, {cfg: {s: loaded[cfg][enc][s]["loss"] for s in SEEDS} for cfg in CONFIGS},
                 floor[0.01], FIGS / f"{enc}_loss.png")
    med = {g: {enc: loaded[g][enc][summary["median_seed"][g][enc]]["kernel"] for enc in ENCODINGS}
           for g in ("primary", "single_register")}
    fig_single_register_stationary(P_star, med["primary"], med["single_register"], summary,
                                   FIGS / "single_register_stationary.png")
    print(f"wrote figures to {FIGS}")

    from write_note import write_note  # noqa: E402  (scripts/write_note.py)
    write_note(summary, REPORTS / "go-no-go.md")
    print(f"wrote {REPORTS / 'go-no-go.md'}")


if __name__ == "__main__":
    main()
