"""Sequence 1: aggregate runs/seq1/** into runs/seq1/summary.json, draw figs/seq1_*.png and
render the tables of reports/seq1.md (the prose lives in seq1_note.py and reads the same summary).

Run: ./.venv/bin/python scripts/seq1_report.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from archb import variants as V  # noqa: E402
from archb.ou import ENCODINGS, exact_kernel  # noqa: E402

SEQ1, FIGS = ROOT / "runs" / "seq1", ROOT / "figs"
MODES = ("ceiling", "train", "mle", "mle_adam", "warm")
ORDER = [c for c in V.LINEUP]  # B0, B0-L4, B0-L5, inout, allpairs, quad, anc1, anc2
KEYS = ("weighted_tv", "tail_tv", "stationary_tv", "lag1_acf", "acf_err", "slope",
        "flux_ratio_depth5", "flux_ratio_depth4", "flux_ratio_depth3", "flux_ratio_depth1-2")
TIGHT = {"weighted_tv": 0.05, "stationary_tv": 0.02}
LAG1_TOL, SLOPE_TOL, C = 0.02, 0.03, float(np.exp(-0.1))

# Okabe-Ito, as in run_all.py
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
STYLE = {  # mode -> (label, color, marker)
    "ceiling": ("step 0 fit to P* (best)", OI["black"], "D"),
    "train": ("Sinkhorn training (seeds)", OI["vermillion"], "o"),
    "mle": ("max. likelihood on data, Adam+L-BFGS", OI["blue"], "s"),
    "mle_adam": ("max. likelihood on data, Adam only", OI["sky"], "v"),
    "warm": ("Sinkhorn from the best fit", OI["green"], "^"),
}
DPI = 170


def load_runs(mode, circuit, enc):
    d = SEQ1 / mode / circuit / enc
    out = []
    for p in sorted(d.glob("*/metrics.json")):
        info = json.loads(p.read_text())
        info["dir"] = str(p.parent.relative_to(ROOT))
        out.append(info)
    return out


def tight_pass(m) -> bool:
    return (m["weighted_tv"] <= TIGHT["weighted_tv"] and m["stationary_tv"] <= TIGHT["stationary_tv"]
            and abs(m["lag1_acf"] - m["lag1_acf_exact"]) <= LAG1_TOL and abs(m["slope"] - C) <= SLOPE_TOL)


def build_summary() -> dict:
    refs = json.loads((ROOT / "runs" / "references.json").read_text())
    S = {"references": refs["metrics"], "circuits": {}}
    for c in ORDER:
        variant, L = V.LINEUP[c]
        S["circuits"][c] = {"n_params": V.n_params(variant, L), "variant": variant, "n_layers": L}
        for enc in ENCODINGS:
            E = {}
            for mode in MODES:
                runs = load_runs(mode, c, enc)
                if not runs:
                    continue
                ms = [r["metrics"] for r in runs]
                wtv = np.array([m["weighted_tv"] for m in ms])
                key = "final_kl" if mode == "ceiling" else ("final_data_kl" if mode.startswith("mle") else "final_loss")
                best = int(np.argmin([r[key] for r in runs])) if mode == "ceiling" else int(np.argmin(wtv))
                E[mode] = {
                    "n": len(runs),
                    "runs": [{"id": r.get("init", r.get("seed")), "objective": r[key], "steps": r["steps"],
                              "wall_time": r["wall_time"], **{k: r["metrics"][k] for k in KEYS},
                              "lag1_acf_exact": r["metrics"]["lag1_acf_exact"], "tight_pass": tight_pass(r["metrics"]),
                              **({"init_loss": r["init_loss"], "final_loss": r["final_loss"]} if "init_loss" in r else {}),
                              "dir": r["dir"]} for r in runs],
                    "best": {k: ms[best][k] for k in KEYS} | {"id": runs[best].get("init", runs[best].get("seed"))},
                    "median": {k: float(np.median([m[k] for m in ms])) for k in KEYS},
                    "min": {k: float(np.min([m[k] for m in ms])) for k in KEYS},
                    "max": {k: float(np.max([m[k] for m in ms])) for k in KEYS},
                    "n_tight_pass": int(sum(tight_pass(m) for m in ms)),
                }
                if mode == "ceiling":
                    b = E[mode]["best"]["weighted_tv"]
                    E[mode]["n_within_1.5x_best"] = int((wtv <= 1.5 * b).sum())
            S["circuits"][c][enc] = E
    gn = json.loads((ROOT / "runs" / "summary.json").read_text())
    S["go_no_go"] = gn.get("aggregate", {})
    (SEQ1 / "summary.json").write_text(json.dumps(S, indent=2) + "\n")
    return S


# ------------------------------------------------------------------------------------ figures
def _present(S, enc):
    return [c for c in ORDER if S["circuits"][c].get(enc)]


def fig_metric(S, key, path, ylabel, refs=True, log=True, unity=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, enc in zip(axes, ENCODINGS):
        cs = _present(S, enc)
        x = np.arange(len(cs))
        for c_i, c in enumerate(cs):
            E = S["circuits"][c][enc]
            if "ceiling" in E:
                vals = [r[key] for r in E["ceiling"]["runs"]]
                ax.scatter(np.full(len(vals), c_i - 0.24), vals, s=9, color=OI["grey"], alpha=0.7, zorder=2,
                           label="step 0 fit to P* (each init)" if c_i == 0 else None)
            for j, mode in enumerate(MODES):
                if mode not in E:
                    continue
                lab, col, mk = STYLE[mode]
                dx = {"ceiling": -0.24, "train": -0.08, "mle": 0.06, "mle_adam": 0.16, "warm": 0.26}[mode]
                vals = [E[mode]["best"][key]] if mode == "ceiling" else [r[key] for r in E[mode]["runs"]]
                ax.scatter(np.full(len(vals), c_i + dx), vals, s=34 if mode != "ceiling" else 46, color=col,
                           marker=mk, edgecolor="white", linewidth=0.8, zorder=3, label=lab if c_i == 0 else None)
        if refs:
            for name, col, ls in (("ar1", OI["purple"], "--"), ("tabular", OI["orange"], ":")):
                ax.axhline(S["references"][name][key], color=col, ls=ls, lw=1.2,
                           label={"ar1": "Gaussian AR(1) reference", "tabular": "tabular counts reference"}[name])
        if unity:
            ax.axhline(1.0, color=OI["black"], lw=0.8, ls="--", label="exact (ratio 1)")
        ax.set_xticks(x, [f"{c}\n{S['circuits'][c]['n_params']}" for c in cs], fontsize=8)
        ax.set_title(f"{enc} encoding", fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(ylabel)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.115, "circuit (number of angles)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_kernels(S, circuit, path):
    P_star = exact_kernel()
    modes = [m for m in ("ceiling", "train", "mle", "warm") if m in S["circuits"][circuit].get("gray", {})]
    fig, axes = plt.subplots(2, 1 + len(modes), figsize=(2.6 * (1 + len(modes)), 5.4))
    norm = LogNorm(1e-4, 1)
    for row, enc in enumerate(ENCODINGS):
        E = S["circuits"][circuit][enc]
        panels = [("exact P*", P_star)]
        for m in modes:
            best = E[m]["best"]["id"]
            run = next(r for r in E[m]["runs"] if r["id"] == best)
            title = {"ceiling": "best fit to P*", "train": "Sinkhorn, best seed", "mle": "max. likelihood, best",
                     "warm": "Sinkhorn from best fit"}[m]
            panels.append((f"{title}\nWTV {run['weighted_tv']:.3f}", np.load(ROOT / run["dir"] / "kernel.npy")))
        for ax, (t, K) in zip(axes[row], panels):
            im = ax.imshow(np.clip(K, 1e-4, 1), norm=norm, cmap="Blues", origin="lower")
            ax.set_title(f"{enc}: {t}" if t == "exact P*" else t, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axhline(15.5, color=OI["vermillion"], lw=0.5, ls=":")
            ax.axvline(15.5, color=OI["vermillion"], lw=0.5, ls=":")
    fig.colorbar(im, ax=axes, shrink=0.6, label="P(x_{t+1} bin | x_t bin), log scale")
    fig.suptitle(f"{circuit}: kernels (rows = x_t, columns = x_t+1; dotted lines = x = 0)", fontsize=9)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_warm_curves(S, circuits, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, enc in zip(axes, ENCODINGS):
        for ci, c in enumerate(circuits):
            E = S["circuits"][c].get(enc, {})
            col = [OI["vermillion"], OI["blue"], OI["green"], OI["purple"]][ci % 4]
            for r in E.get("train", {}).get("runs", []):
                ax.plot(np.load(ROOT / r["dir"] / "loss.npy"), color=col, lw=1, alpha=0.45)
            for r in E.get("warm", {}).get("runs", []):
                ax.plot(np.load(ROOT / r["dir"] / "loss.npy"), color=col, lw=2, label=f"{c}: from best fit")
            ax.plot([], [], color=col, lw=1, alpha=0.45, label=f"{c}: Sinkhorn from copy init")
        ax.set_yscale("log")
        ax.set_title(f"{enc} encoding", fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Sinkhorn divergence (tracked)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------ tables
def f3(x):
    return f"{x:.3f}"


def table_main(S) -> str:
    rows = ["| Circuit | Angles | Enc. | Step 0 best | Step 0 median | Inits ≤ 1.5× best | Sinkhorn median [range] "
            "| Warm start | ML Adam+L-BFGS median | ML Adam only median |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in ORDER:
        for enc in ENCODINGS:
            E = S["circuits"][c].get(enc, {})
            if not E:
                continue
            g = lambda m, k="median": f3(E[m][k]["weighted_tv"]) if m in E else "–"  # noqa: E731
            tr = (f"{f3(E['train']['median']['weighted_tv'])} [{f3(E['train']['min']['weighted_tv'])}–"
                  f"{f3(E['train']['max']['weighted_tv'])}]") if "train" in E else "–"
            within = f"{E['ceiling']['n_within_1.5x_best']}/{E['ceiling']['n']}" if "ceiling" in E else "–"
            rows.append(f"| {c} | {S['circuits'][c]['n_params']} | {enc} | {g('ceiling', 'best')} | {g('ceiling')} | "
                        f"{within} | {tr} | {g('warm', 'best')} | {g('mle')} | {g('mle_adam')} |")
    return "\n".join(rows)


def table_flux(S) -> str:
    rows = ["| Circuit | Enc. | Flux x = 0: step 0 best | Sinkhorn median | Warm | ML median | Flux x = ±1.5: step 0 best "
            "| Sinkhorn median | ML median |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in ORDER:
        for enc in ENCODINGS:
            E = S["circuits"][c].get(enc, {})
            if not E:
                continue
            g = lambda m, k, w="median": f3(E[m][w][k]) if m in E else "–"  # noqa: E731
            rows.append(f"| {c} | {enc} | {g('ceiling', 'flux_ratio_depth5', 'best')} | {g('train', 'flux_ratio_depth5')} | "
                        f"{g('warm', 'flux_ratio_depth5', 'best')} | {g('mle', 'flux_ratio_depth5')} | "
                        f"{g('ceiling', 'flux_ratio_depth4', 'best')} | {g('train', 'flux_ratio_depth4')} | "
                        f"{g('mle', 'flux_ratio_depth4')} |")
    return "\n".join(rows)


def table_tight(S) -> str:
    rows = ["| Circuit | Enc. | Sinkhorn seeds passing | ML seeds passing | Best stationary TV, trained on data | Best weighted TV, trained on data |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for c in ORDER:
        for enc in ENCODINGS:
            E = S["circuits"][c].get(enc, {})
            if not E:
                continue
            tr = [m for m in ("train", "mle", "mle_adam") if m in E]  # not warm: it starts from a fit to P*
            st = min(E[m]["min"]["stationary_tv"] for m in tr) if tr else float("nan")
            wt = min(E[m]["min"]["weighted_tv"] for m in tr) if tr else float("nan")
            p = lambda m: f"{E[m]['n_tight_pass']}/{E[m]['n']}" if m in E else "–"  # noqa: E731
            rows.append(f"| {c} | {enc} | {p('train')} | {p('mle')} | {f3(st)} | {f3(wt)} |")
    return "\n".join(rows)


def main():
    S = build_summary()
    FIGS.mkdir(exist_ok=True)
    fig_metric(S, "weighted_tv", FIGS / "seq1_weighted_tv.png", "weighted TV against P* (log scale)")
    fig_metric(S, "flux_ratio_depth5", FIGS / "seq1_flux_x0.png", "flux across x = 0, model ÷ exact (log scale)",
               refs=False, unity=True)
    fig_metric(S, "stationary_tv", FIGS / "seq1_stationary_tv.png", "stationary TV against π* (log scale)")
    for c in ("B0", "anc2"):
        if S["circuits"][c].get("gray"):
            fig_kernels(S, c, FIGS / f"seq1_kernels_{c}.png")
    warm = [c for c in ("B0", "anc2") if S["circuits"][c].get("gray", {}).get("warm")]
    if warm:
        fig_warm_curves(S, warm, FIGS / "seq1_warm_curves.png")
    tables = {"main": table_main(S), "flux": table_flux(S), "tight": table_tight(S)}
    (SEQ1 / "tables.json").write_text(json.dumps(tables, indent=2) + "\n")
    print(tables["main"])
    print(tables["flux"])
    print(tables["tight"])


if __name__ == "__main__":
    main()
