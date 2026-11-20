"""Sequence 2: aggregate runs/seq2/** into runs/seq2/summary.json, draw figs/seq2_*.png, render the tables used by
scripts/seq2_note.py (runs/seq2/tables.json).

Run: ./.venv/bin/python scripts/seq2_report.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from archb import targets as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEQ2, FIGS = ROOT / "runs" / "seq2", ROOT / "figs"
CIRCUITS = ("B0", "anc1", "anc2")
MODES = ("ceiling", "mle", "train")
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
STYLE = {"ceiling": ("fit to P* (each init)", OI["black"], "D"),
         "mle": ("max. likelihood on data", OI["blue"], "s"),
         "train": ("Sinkhorn on data", OI["vermillion"], "o")}
DPI = 170


_REFS = {}


def refs_for(T):
    """(reference kernels, fitted params) per target, computed once (the var2d fit takes a few minutes)."""
    if T.name not in _REFS:
        _REFS[T.name] = TG.reference_kernels(T, TG.load_dataset(T))
    return _REFS[T.name]


def runs(T, mode, c):
    out = []
    for p in sorted((SEQ2 / T / mode / c).glob("*/metrics.json")):
        r = json.loads(p.read_text())
        r["dir"] = str(p.parent.relative_to(ROOT))
        out.append(r)
    return out


def build_summary():
    S = {"targets": {}}
    for name, T in TG.TARGETS.items():
        data = TG.load_dataset(T)
        refs, fitted = refs_for(T)
        P_star = refs["exact"]
        E = {"d": T.d, "bits": T.bits, "K": T.K, "range": [T.lo, T.hi], "params": T.params,
             "fitted_params": {k: (np.asarray(v).tolist()) for k, v in fitted.items()},
             "data": {"n_transitions": int(data["n_transitions"]), "clip_frac": float(data["clip_frac"])},
             "references": {r: TG.metrics(K, T, P_star) | {"tight_pass": TG.tight_pass(TG.metrics(K, T, P_star), T.d)}
                            for r, K in refs.items() if r != "exact"},
             "circuits": {}}
        for c in CIRCUITS:
            C = {}
            for mode in MODES:
                rs = runs(name, mode, c)
                if not rs:
                    continue
                wtv = np.array([r["metrics"]["weighted_tv"] for r in rs])
                best = int(np.argmin([r["objective"] for r in rs])) if mode == "ceiling" else int(np.argmin(wtv))
                C[mode] = {"n": len(rs), "n_params": rs[0]["n_params"], "n_qubits": rs[0]["n_qubits"],
                           "runs": [{"id": r["id"], "objective": r["objective"], "steps": r["steps"],
                                     "wall_time": r["wall_time"], "tight_pass": r["tight_pass"], "dir": r["dir"],
                                     **r["metrics"]} for r in rs],
                           "best": rs[best]["metrics"] | {"id": rs[best]["id"], "dir": rs[best]["dir"]},
                           "median": {k: float(np.median([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                           "min": {k: float(np.min([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                           "max": {k: float(np.max([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                           "n_tight_pass": int(sum(r["tight_pass"] for r in rs))}
            if C:
                E["circuits"][c] = C
        S["targets"][name] = E
    (SEQ2 / "summary.json").write_text(json.dumps(S, indent=2) + "\n")
    return S


# ------------------------------------------------------------------------------------ figures
def fig_wtv(S, path):
    names = list(S["targets"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.0), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        E = S["targets"][name]
        for ci, c in enumerate(CIRCUITS):
            C = E["circuits"].get(c, {})
            for mode, dx in (("ceiling", -0.2), ("mle", 0.0), ("train", 0.2)):
                if mode not in C:
                    continue
                lab, col, mk = STYLE[mode]
                v = [r["weighted_tv"] for r in C[mode]["runs"]]
                ax.scatter(np.full(len(v), ci + dx), v, color=col, marker=mk, s=34, edgecolor="white", linewidth=0.8,
                           zorder=3, label=lab if ci == 0 else None)
        for r, col, ls, lab in (("parametric", OI["purple"], "--", "true family, fitted"),
                                ("tabular", OI["orange"], ":", "tabular counts")):
            ax.axhline(E["references"][r]["weighted_tv"], color=col, ls=ls, lw=1.2, label=lab)
        ax.set_xticks(range(len(CIRCUITS)), [f"{c}\n{E['circuits'].get(c, {}).get('mle', {}).get('n_params', '')}"
                                            for c in CIRCUITS], fontsize=8)
        ax.set_title(f"{name} ({E['K']} states)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("weighted TV against P* (log scale)")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _best_kernel(E, c, mode):
    C = E["circuits"].get(c, {})
    return np.load(ROOT / C[mode]["best"]["dir"] / "kernel.npy") if mode in C else None


def fig_rows_1d(S, name, rows_z, path):
    """Selected rows P(x' | x) on the decoded grid: exact, true family fitted, best anc2 by likelihood and Sinkhorn."""
    T = TG.TARGETS[name]
    E = S["targets"][name]
    P_star = T.exact_kernel()
    refs, _ = refs_for(T)
    z = T.z
    curves = [("exact P*", P_star, OI["black"], "-"), ("true family, fitted", refs["parametric"], OI["purple"], "--")]
    for c, mode, col, lab in (("anc2", "mle", OI["blue"], "anc2, max. likelihood"),
                              ("anc2", "train", OI["vermillion"], "anc2, Sinkhorn"),
                              ("B0", "mle", OI["green"], "B0, max. likelihood")):
        K = _best_kernel(E, c, mode)
        if K is not None:
            curves.append((lab, K, col, "-"))
    fig, axes = plt.subplots(1, len(rows_z), figsize=(3.6 * len(rows_z), 3.2), sharey=True)
    for ax, x0 in zip(axes, rows_z):
        i = int(np.argmin(np.abs(z - x0)))
        for lab, K, col, ls in curves:
            ax.plot(z, K[i], color=col, ls=ls, lw=1.6 if col != OI["black"] else 2.2, label=lab,
                    drawstyle="steps-mid")
        ax.set_title(f"row x_t = {z[i]:+.2f}", fontsize=9)
        ax.set_xlabel("x_t+1")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("P(x_t+1 bin | x_t)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{name}: kernel rows", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_var2d(S, path):
    """Conditional mean and sd of each coordinate along the diagonal x1 = x2, and the stationary law."""
    T = TG.VAR2D
    E = S["targets"]["var2d"]
    refs, _ = refs_for(T)
    X = T.state_values()
    kernels = [("exact P*", refs["exact"], OI["black"], "-"), ("true family, fitted", refs["parametric"], OI["purple"], "--"),
               ("tabular counts", refs["tabular"], OI["orange"], ":")]
    for c, col in (("anc2", OI["blue"]), ("anc1", OI["sky"]), ("B0", OI["green"])):
        K = _best_kernel(E, c, "mle")
        if K is not None:
            kernels.append((f"{c}, max. likelihood", K, col, "-"))
    diag = np.flatnonzero(np.isclose(X[:, 0], X[:, 1]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for lab, K, col, ls in kernels:
        for k, ax in enumerate(axes[:2]):
            m = K @ X[:, k]
            sd = np.sqrt(np.clip(K @ X[:, k] ** 2 - m**2, 0, None))
            ax.plot(X[diag, 0], m[diag], color=col, ls=ls, lw=1.6, label=lab)
            ax.fill_between(X[diag, 0], m[diag] - sd[diag], m[diag] + sd[diag], color=col, alpha=0.06)
        pi = TG.stationary(K).reshape(T.n_per_dim, T.n_per_dim)
        axes[2].plot(T.z, pi.sum(axis=0), color=col, ls=ls, lw=1.6, drawstyle="steps-mid")
    for k, ax in enumerate(axes[:2]):
        ax.set_title(f"E[x'_{k + 1} | x] ± sd, along x_1 = x_2", fontsize=9)
        ax.set_xlabel("x_1 = x_2")
    axes[2].set_title("stationary law of x_2", fontsize=9)
    axes[2].set_xlabel("x_2")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------- tables
def f3(x):
    return f"{x:.3f}"


def table_target(S, name):
    E = S["targets"][name]
    d = E["d"]
    head = ("| Model | Params | Weighted TV | Stationary TV | Tail TV | Lag-1 error | Cond. mean RMSE (sd) | "
            "Cond. sd rel. error |" + (" Corr. error |" if d == 2 else "") + " Tight |")
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" + (" ---: |" if d == 2 else "") + " ---: |"
    rows = [head, sep]

    def fmt(lab, n, m, tight):
        lag = max(abs(m[f"lag1_{k}"] - m[f"lag1_exact_{k}"]) for k in range(d))
        cm = max(m[f"cond_mean_rmse_{k}"] for k in range(d))
        cs = max(m[f"cond_sd_rel_{k}"] for k in range(d))
        corr = f" {m['corr_err']:+.3f} |" if d == 2 else ""
        return (f"| {lab} | {n} | {f3(m['weighted_tv'])} | {f3(m['stationary_tv'])} | {f3(m['tail_tv'])} | {f3(lag)} | "
                f"{f3(cm)} | {f3(cs)} |{corr} {tight} |")

    nfam = {"arch": 3, "merton": 5, "var2d": 7}[name]
    for r, lab, n in (("parametric", "true family, fitted", nfam), ("tabular", "tabular counts", E["K"] * (E["K"] - 1)),
                      ("iid", "i.i.d. oracle", E["K"] - 1)):
        m = E["references"][r]
        rows.append(fmt(lab, n, m, "✓" if m["tight_pass"] else "✗"))
    for c in CIRCUITS:
        for mode, lab in (("ceiling", "fit to P* (best)"), ("mle", "max. likelihood (median)"), ("train", "Sinkhorn (median)")):
            C = E["circuits"].get(c, {}).get(mode)
            if not C:
                continue
            m = C["best"] if mode == "ceiling" else C["median"]
            tight = f"{C['n_tight_pass']}/{C['n']}"
            rows.append(fmt(f"{c}, {lab}", C["n_params"], m, tight))
    return "\n".join(rows)


def main():
    S = build_summary()
    FIGS.mkdir(exist_ok=True)
    fig_wtv(S, FIGS / "seq2_weighted_tv.png")
    fig_rows_1d(S, "arch", (-2.5, 0.0, 2.5), FIGS / "seq2_rows_arch.png")
    fig_rows_1d(S, "merton", (-2.0, 0.0, 2.0), FIGS / "seq2_rows_merton.png")
    if S["targets"]["var2d"]["circuits"]:
        fig_var2d(S, FIGS / "seq2_var2d.png")
    tables = {name: table_target(S, name) for name in S["targets"]}
    (SEQ2 / "tables.json").write_text(json.dumps(tables, indent=2) + "\n")
    for t in tables.values():
        print(t, "\n")


if __name__ == "__main__":
    main()
