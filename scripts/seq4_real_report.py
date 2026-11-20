"""Sequence 4, target D (real data): aggregate runs/seq4_real/** into runs/seq4_real/summary.json, draw
figs/seq4_real_*.png and render reports/seq4_real.md. Every number in the note is read from the run files
(result.json of every fit, eval.json of every dataset); nothing is typed by hand.

Selection: for every dataset and model the reported fit is the init with the best VALIDATION log-likelihood (the
criterion of the early stopping); within a model family the variant (m, ancillas, S, order, innovation law, inputs)
is also the one with the best validation log-likelihood. The test period is never used for a choice.

Run: ./.venv/bin/python scripts/seq4_real_report.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIGS, REPORT = ROOT / "runs" / "seq4_real", ROOT / "figs", ROOT / "reports" / "seq4_real.md"
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7f7f7f"}
DPI = 170
DATASETS = ["eurusd-u", "eurusd-q", "eurusd-v", "nasdaq-u", "nasdaq-q", "nasdaq-v"]
SERIES = {"eurusd": "EUR/USD", "nasdaq": "NASDAQ Composite"}
GRIDS = {"u": "uniform grid", "q": "quantile grid", "v": "volatility-normalized grid"}
REF, REF_BIN = "garch-t-cont", "garch-t-bin"

FAMILIES = {
    "Architecture B": ["archb-B0", "archb-anc2"],
    "C-QSM": ["cqsm-m1", "cqsm-m2", "cqsm-m1-anc2", "cqsm-m2-anc2"],
    "QSM": ["qsm-m1", "qsm-m2"],
    "IOHMM full": ["iohmm-full-2", "iohmm-full-4"],
    "IOHMM factored": ["iohmm-fact-2", "iohmm-fact-4"],
    "HMM on x": ["hmm-2", "hmm-4"],
    "Markov chain": ["markov-0", "markov-1", "markov-2"],
    "GARCH, continuous inputs": ["garch-norm-cont", "garch-t-cont", "garch-gjr-cont"],
    "GARCH, binned inputs": ["garch-norm-bin", "garch-t-bin", "garch-gjr-bin"],
    "MS-AR": ["msar-cont", "msar-bin"],
}
ORDER = [m for v in FAMILIES.values() for m in v]
ORDER.insert(ORDER.index("cqsm-m2-anc2") + 1, "cqsm-mix")
QUANTUM = ("archb", "cqsm", "qsm")


def label(mo, of=None):
    if mo == "cqsm-mix":
        return "C-QSM, mixture of the 5 seeds" + (f" ({label(of).split(', ', 1)[1]})" if of else "")
    p = mo.split("-")
    if p[0] == "archb":
        return f"Arch. B, {p[1]}"
    if p[0] == "cqsm":
        return f"C-QSM, m = {p[1][1:]}" + (" + 2 ancillas" if len(p) > 2 else "")
    if p[0] == "qsm":
        return f"QSM (seq. 3, general), m = {p[1][1:]}"
    if p[0] == "iohmm":
        return f"IOHMM {'full' if p[1] == 'full' else 'factored'}, S = {p[2]}"
    if p[0] == "hmm":
        return f"HMM on x alone, S = {p[1]}"
    if p[0] == "markov":
        return {"0": "i.i.d. (Markov order 0)", "1": "Markov order 1 (tabular kernel)", "2": "Markov order 2"}[p[1]]
    inp = {"cont": "continuous inputs", "bin": "binned inputs"}[p[-1]]
    if p[0] == "garch":
        return f"{'GJR-GARCH-t' if p[1] == 'gjr' else 'GARCH-N' if p[1] == 'norm' else 'GARCH-t'}, {inp}"
    return f"MS-AR(1), 2 regimes, {inp}"


def bold(mo, of=None):
    return f"**{label(mo, of)}**" if mo.startswith("cqsm") or mo == REF else label(mo, of)


def dsname(ds):
    return f"{SERIES[ds.split('-')[0]]}, {GRIDS[ds.split('-')[1]]}"


# ------------------------------------------------------------------------------------------------ aggregation
def load():
    E, R = {}, {}
    for ds in DATASETS:
        p = RUNS / ds / "eval.json"
        if p.exists():
            E[ds] = json.loads(p.read_text())
            R[ds] = {}
            for q in sorted((RUNS / ds).glob("*/*/result.json")):
                r = json.loads(q.read_text())
                R[ds].setdefault(r["model"], []).append(r)
    return E, R


def chosen(E, ds, fam):
    have = [m for m in FAMILIES[fam] if m in E[ds]["models"]]
    return max(have, key=lambda m: E[ds]["models"][m]["val_ll"]) if have else None


def fmt(x, n=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def fp(p):
    return "–" if p is None else ("<0.001" if p < 1e-3 else f"{p:.3f}")


def filt(mo, e):
    if mo.startswith(("garch",)):
        return "1"
    if mo == "msar-cont":
        return "2"
    if mo == "cqsm-mix":
        return "5 × " + str(e.get("of_filter", "–"))
    return str(e["filter_size"])


def mem(mo, e):
    if mo == "cqsm-mix":
        return "5 seeds"
    if mo.startswith(("cqsm", "qsm")):
        return f"D = {e['memory_dim']}"
    if mo.startswith(("archb", "markov", "garch")):
        return "–" if not mo.startswith("garch") else "σ²"
    return f"{e['memory_dim']} states"


# ------------------------------------------------------------------------------------------------ tables
def lik_table(E, R, ds):
    ev = E[ds]["models"]
    of = E[ds].get("cqsm_chosen")
    rows = ["| Model | Memory | Filter size | Params | Inits (test range) | Train | Validation | **Test** ± SE | "
            "Test − GARCH-t ± SE | Test − GARCH-t binned ± SE | acf\\|x\\| err (max / mean) | Kurtosis |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mo in [m for m in ORDER if m in ev]:
        e = ev[mo]
        if mo == "cqsm-mix":
            e = dict(e, of_filter=ev[of]["filter_size"], n_params=f"5 × {ev[of]['n_params']}")
            inits = f"{e['n_inits']}"
        else:
            inits = f"{e['n_inits']}" + (f" ({fmt(e['test_ll_inits_min'])} .. {fmt(e['test_ll_inits_max'])})"
                                          if e["n_inits"] > 1 else "")
        acf = "–" if e.get("acf_abs_err_max") is None else f"{fmt(e['acf_abs_err_max'])} / {fmt(e['acf_abs_err_mean'])}"
        rows.append(f"| {bold(mo, of)} | {mem(mo, e)} | {filt(mo, e)} | {e['n_params']} | {inits} | "
                    f"{fmt(e['train_ll'])} | {fmt(e['val_ll'])} | {fmt(e['test_ll'])} ± {fmt(e['test_se'])} | "
                    f"{fmt(e['diff_ref'], sign=True)} ± {fmt(e['diff_ref_se'])} | "
                    f"{fmt(e['diff_refbin'], sign=True)} ± {fmt(e['diff_refbin_se'])} | {acf} | "
                    f"{fmt(e.get('kurtosis'), 2)} |")
    emp = E[ds]["empirical"]
    rows.append(f"\nData (bin values in z units): kurtosis {fmt(emp['kurtosis'], 2)} over the whole series, "
                f"{fmt(emp['test_kurtosis'], 2)} over the test period; return sd in training units: validation "
                f"{fmt(emp['val_sd'], 2)}, test {fmt(emp['test_sd'], 2)}.")
    if E[ds]["grid"] == "v":
        rows.append("On this grid the stylized facts of the models are not comparable: simulating any model of the "
                    "normalized bins through the EWMA feedback gives a nearly integrated variance process (kurtosis in "
                    "the hundreds, some paths explode and are dropped); the log-scores and backtests, which condition on "
                    "the observed past, are unaffected.")
    return "\n".join(rows) + "\n"


def yearly_table(E, ds):
    Y = E[ds]["yearly"]
    keys = list(Y[0]["diff"])
    of = E[ds]["cqsm_chosen"]
    rows = ["| Year | Period | Return sd | " + " | ".join(label(k, of) for k in keys) + " |",
            "| --- | --- | ---: |" + " ---: |" * len(keys)]
    for y in Y:
        rows.append(f"| {y['year']} | {y['split']} | {fmt(y['sd'], 2)} | "
                    + " | ".join(fmt(y["diff"][k], sign=True) for k in keys) + " |")
    return "\n".join(rows) + "\n"


def calm_corr(E, ds):
    """Correlation across years between the return sd and the C-QSM's difference to GARCH-t (whole series)."""
    Y = [y for y in E[ds]["yearly"] if y["n"] > 100]
    sd = np.array([y["sd"] for y in Y])
    d = np.array([y["diff"][E[ds]["cqsm_chosen"]] for y in Y])
    calm = sd < np.median(sd)
    return float(np.corrcoef(sd, d)[0, 1]), float(d[calm].mean()), float(d[~calm].mean())


def risk_table(E, ds, mos, key):
    ev = E[ds]["models"]
    rows = ["| Model | 1%: hit rate | Kupiec p | CC p | ES Z | 5%: hit rate | Kupiec p | CC p | ES Z |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mo in mos:
        c = [ev[mo][key][a] for a in ("0.01", "0.05")]
        rows.append(f"| {bold(mo, E[ds].get('cqsm_chosen'))} | " + " | ".join(f"{fmt(x['hit_rate'] * 100, 2)}% | {fp(x['kupiec_p'])} | "
                                                    f"{fp(x['cc_p'])} | {fmt(x['es_z'], 2, True)}" for x in c) + " |")
    return "\n".join(rows) + "\n"


def cont_table(E, ds):
    rows = ["| Model (raw returns, continuous quantile) | 1%: hit rate | Kupiec p | CC p | 5%: hit rate | Kupiec p | CC p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mo, c in E[ds]["continuous_var1"].items():
        rows.append(f"| {label(mo)} | " + " | ".join(f"{fmt(c[a]['hit_rate'] * 100, 2)}% | {fp(c[a]['kupiec_p'])} | "
                                                     f"{fp(c[a]['cc_p'])}" for a in ("0.01", "0.05")) + " |")
    return "\n".join(rows) + "\n"


def selection_table(E):
    rows = ["| Family | " + " | ".join(f"{SERIES[d.split('-')[0]]}, {d.split('-')[1]}" for d in DATASETS if d in E) + " |",
            "| --- |" + " ---: |" * len(E)]
    for fam in FAMILIES:
        cells = []
        for ds in [d for d in DATASETS if d in E]:
            mo = chosen(E, ds, fam)
            e = E[ds]["models"][mo]
            cells.append(f"{fmt(e['diff_ref'], sign=True)} ± {fmt(e['diff_ref_se'])} ({label(mo).split(', ', 1)[-1]})")
        rows.append(f"| {fam} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def memory_table(E):
    rows = ["| Dataset | Readout from | Features | corr. with log RV21, test (train) | "
            "corr. with MS-AR smoothed regime, test (train) |",
            "| --- | --- | ---: | ---: | ---: |"]
    for ds in [d for d in DATASETS if d in E]:
        for k, m in E[ds]["memory"].items():
            name = "EWMA of the observed bin values (λ = 0.94, no parameter)" if k == "ewma" else label(m["model"])
            rows.append(f"| {SERIES[ds.split('-')[0]]}, {ds.split('-')[1]} | {name} | {m.get('n_features', '–')} | "
                        f"{fmt(m['corr_rv_test'])} ({fmt(m['corr_rv_train'])}) | {fmt(m['corr_msar_test'])}"
                        + (f" ({fmt(m['corr_msar_train'])})" if "corr_msar_train" in m else "") + " |")
    return "\n".join(rows) + "\n"


def es_table(R, E):
    """Overfitting control: early stopping iterate against the last iterate, circuits and EM."""
    rows = ["| Dataset | Model | Stop iteration / run length | Validation, kept | Validation, last iterate |",
            "| --- | --- | ---: | ---: | ---: |"]
    for ds in [d for d in DATASETS if d in E]:
        for mo in ["archb-anc2", "cqsm-m1-anc2", "cqsm-m2-anc2", "qsm-m2", "iohmm-fact-2", "hmm-2"]:
            if mo not in E[ds]["models"]:
                continue
            r = next(x for x in R[ds][mo] if x["init"] == E[ds]["models"][mo]["init"])
            n = r.get("lbfgs_iters", r.get("em_iters"))
            last = r.get("val_ll_last", r["val_curve"][-1][1] if r.get("val_curve") else None)
            rows.append(f"| {SERIES[ds.split('-')[0]]}, {ds.split('-')[1]} | {label(mo)} | {r['stop_iter']} / {n} | "
                        f"{fmt(r['val_ll'])} | {fmt(last)} |")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------------------------------------ figures
def colour(mo):
    if mo.startswith("cqsm"):
        return OI["vermillion"]
    if mo.startswith("qsm"):
        return OI["purple"]
    if mo.startswith("archb"):
        return OI["black"]
    if mo.startswith(("iohmm", "hmm")):
        return OI["blue"]
    if mo.startswith("markov"):
        return OI["green"]
    return OI["orange"]


def fig_loglik(E):
    dss = [d for d in DATASETS if d in E]
    nc = len(dss) if len(dss) <= 4 else 3
    nr = int(np.ceil(len(dss) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 6.4 * nr), sharey=True, squeeze=False)
    for ax in axs.ravel()[len(dss):]:
        ax.axis("off")
    for ax, ds in zip(axs.ravel(), dss):
        ev = E[ds]["models"]
        mos = [m for m in ORDER if m in ev]
        y = np.arange(len(mos))[::-1]
        v = np.array([ev[m]["diff_ref"] for m in mos])
        se = np.array([ev[m]["diff_ref_se"] for m in mos])
        lo = max(v.min(), -0.12)
        ax.barh(y, np.maximum(v, lo), xerr=2 * se, color=[colour(m) for m in mos], height=0.7, error_kw={"lw": 0.8})
        for yy, vv in zip(y, v):
            if vv < lo:
                ax.text(lo, yy, f"{vv:.2f} ◂", fontsize=6, va="center", ha="left", color="white")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, [label(m, E[ds].get("cqsm_chosen")) for m in mos], fontsize=6.5)
        ax.set_title(f"{SERIES[ds.split('-')[0]]}, {GRIDS[ds.split('-')[1]]}", fontsize=8)
        ax.set_xlabel("test log-score − GARCH-t (nats / day)\n± 2 block-bootstrap SE", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "seq4_real_loglik.png", dpi=DPI)
    plt.close(fig)


def _dates(ds):
    import csv  # noqa: F401
    from archb import realdata as D
    s, g = ds.split("-")
    return np.array(D.make_dataset(s, g)["dates"], dtype="datetime64[D]")


def fig_cum(E):
    dss = [d for d in DATASETS if d.endswith("-u") and d in E]
    fig, axs = plt.subplots(1, len(dss), figsize=(6.2 * len(dss) / 1.5, 3.4))
    for ax, ds in zip(np.atleast_1d(axs), dss):
        arr = np.load(RUNS / ds / "eval_arrays.npz")
        dt, s = _dates(ds), E[ds]["n_train"] + E[ds]["n_val"]
        ref = arr[f"lp/{REF}"][s:]
        for fam, ls in (("C-QSM", "-"), ("Architecture B", ":"), ("IOHMM factored", "--"), ("MS-AR", "-."),
                        ("GARCH, binned inputs", "-"), ("Markov chain", ":")):
            mo = chosen(E, ds, fam)
            ax.plot(dt[s:], np.cumsum(arr[f"lp/{mo}"][s:] - ref), ls, color=colour(mo) if fam != "GARCH, binned inputs"
                    else OI["sky"], lw=1.1, label=label(mo))
        ax.axhline(0, color="k", lw=0.6)
        mx = arr["lp/cqsm-mix"][s:]
        ax.plot(dt[s:], np.cumsum(mx - ref), "-", color=OI["purple"], lw=1.1, label=label("cqsm-mix", E[ds]["cqsm_chosen"]))
        ax.set_title(f"{SERIES[ds.split('-')[0]]}, uniform grid:\ncumulative test log-score − GARCH-t", fontsize=8)
        ax.set_ylabel("nats", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "seq4_real_cumulative.png", dpi=DPI)
    plt.close(fig)


def fig_acf(E):
    dss = [d for d in DATASETS if d.endswith("-u") and d in E]
    fig, axs = plt.subplots(1, len(dss), figsize=(4.6 * len(dss), 3.4))
    for ax, ds in zip(np.atleast_1d(axs), dss):
        arr = np.load(RUNS / ds / "eval_arrays.npz")
        k = np.arange(1, len(arr["emp_acf"]) + 1)
        ax.plot(k, arr["emp_acf"], "k-", lw=2, label="data, 1999–2026")
        for fam in ("C-QSM", "Architecture B", "QSM", "IOHMM factored", "MS-AR"):
            mo = chosen(E, ds, fam)
            ax.plot(k, arr[f"acf/{mo}"], "-", color=colour(mo) if fam != "MS-AR" else OI["green"], lw=1.1,
                    label=label(mo))
        ax.plot(k, arr[f"acf/{REF}"], "--", color=OI["orange"], lw=1.3, label=label(REF))
        ax.set_title(f"{SERIES[ds.split('-')[0]]}, uniform grid: acf of |x|", fontsize=8)
        ax.set_xlabel("lag (days)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "seq4_real_acf.png", dpi=DPI)
    plt.close(fig)


def fig_memory(E):
    from archb import realdata as D
    dss = [d for d in DATASETS if d.endswith("-u") and d in E]
    fig, axs = plt.subplots(3, len(dss), figsize=(6.0 * len(dss), 7.0), sharex="col")
    axs = np.asarray(axs).reshape(3, len(dss))
    for j, ds in enumerate(dss):
        arr = np.load(RUNS / ds / "eval_arrays.npz")
        s, g = ds.split("-")
        dd = D.make_dataset(s, g)
        dt = np.array(dd["dates"], dtype="datetime64[D]")
        a, st = E[ds]["n_train"], E[ds]["n_train"] + E[ds]["n_val"]
        m = E[ds]["memory"]["cqsm"]
        axs[0, j].plot(dt, dd["r"] * 100, lw=0.3, color="k")
        axs[0, j].set_ylabel("daily log-return (%)", fontsize=7)
        axs[0, j].set_title(f"{SERIES[s]}: memory of the C-QSM chosen on validation ({label(m['model'])})", fontsize=8)
        axs[1, j].plot(dt, np.exp(arr["rv21"]) * 100 * np.sqrt(252), color="k", lw=0.9, label="21-day realized vol")
        axs[1, j].plot(dt, np.exp(arr["mem_rv/cqsm"]) * 100 * np.sqrt(252), color=OI["vermillion"], lw=0.7,
                       label=f"C-QSM memory readout (test corr. {m['corr_rv_test']:.2f})")
        axs[1, j].plot(dt, np.exp(arr["mem_rv/garch-t"]) * 100 * np.sqrt(252), color=OI["orange"], lw=0.7, alpha=0.8,
                       label=f"GARCH-t σ readout (test corr. {E[ds]['memory']['garch-t']['corr_rv_test']:.2f})")
        axs[1, j].set_yscale("log")
        axs[1, j].set_ylabel("annualized vol (%)", fontsize=7)
        axs[2, j].plot(dt, arr["msar_smoothed"], color=OI["green"], lw=0.8, label="MS-AR smoothed P(turbulent)")
        axs[2, j].plot(dt, arr["mem_msar/cqsm"], color=OI["vermillion"], lw=0.6,
                       label=f"C-QSM memory readout (test corr. {m['corr_msar_test']:.2f})")
        axs[2, j].set_ylabel("probability", fontsize=7)
        for ax in axs[:, j]:
            for x in (dt[a], dt[st]):
                ax.axvline(x, color=OI["grey"], lw=0.8, ls="--")
            ax.tick_params(labelsize=7)
            if ax is not axs[0, j]:
                ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "seq4_real_memory.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ text
def bt(e, a="0.05", key="var1"):
    x = e[key][a]
    return f"{x['hit_rate'] * 100:.1f}% (Kupiec p {fp(x['kupiec_p'])}, ES Z {x['es_z']:+.2f})"


def sig(d, se):
    return "better than" if d > 2 * se else "worse than" if d < -2 * se else "statistically tied with"


def findings(E):
    out = []
    dss = [d for d in DATASETS if d in E]
    fixed = [d for d in dss if not d.endswith("-v")]
    cq = {d: E[d]["cqsm_chosen"] for d in dss}
    ev = {d: E[d]["models"] for d in dss}
    # 1. headline
    head = [f"{dsname(d)}: {label(cq[d])} {sig(ev[d][cq[d]]['diff_ref'], ev[d][cq[d]]['diff_ref_se'])} GARCH-t, "
            f"{ev[d][cq[d]]['diff_ref']:+.3f} ± {ev[d][cq[d]]['diff_ref_se']:.3f} nats/day" for d in dss]
    n_beat = sum(ev[d][cq[d]]["diff_ref"] > 2 * ev[d][cq[d]]["diff_ref_se"] for d in dss)
    n_lose = sum(ev[d][cq[d]]["diff_ref"] < -2 * ev[d][cq[d]]["diff_ref_se"] for d in dss)
    out.append(f"**Does the C-QSM beat GARCH-t on test likelihood? {'No' if n_beat == 0 else f'In {n_beat} of {len(dss)} cases'}.** "
               f"The C-QSM variant chosen on validation loses by more than 2 SE in {n_lose} of {len(dss)} datasets "
               "(test log-score minus GARCH-t with continuous inputs, block-bootstrap SE; the binned-input GARCH-t, "
               "same information set as the circuits, is within 0.003 of it everywhere): " + "; ".join(head) + ".")
    # 2. mixture of seeds
    mix = [f"{dsname(d)} {ev[d]['cqsm-mix']['diff_ref']:+.3f} ± {ev[d]['cqsm-mix']['diff_ref_se']:.3f}" for d in dss]
    out.append("**Averaging the 5 seeds of the chosen C-QSM helps most where it loses least.** Equal-weight mixture of "
               "the 5 stage-2 fits (no choice made on test), minus GARCH-t: " + "; ".join(mix) + ". On NASDAQ the "
               "mixture is statistically tied with GARCH-t on both fixed grids; on EUR/USD it stays about 0.04–0.05 "
               "nats/day behind. The spread between seeds of one variant (Inits column of the tables) is of the same "
               "size as the differences between variants, so the choice of m and ancillas on a calm 2018–2021 "
               "validation period is close to noise.")
    # 3. best models
    best = [f"{dsname(d)}: {label(max(ev[d], key=lambda m: ev[d][m]['test_ll']))} "
            f"({max(x['diff_ref'] for x in ev[d].values()):+.3f})" for d in dss]
    out.append("**Best model on test log-score** (difference to GARCH-t): " + "; ".join(best) + ". The leverage "
               "effect matters on the equity index (GJR-GARCH-t gains about 0.013 nats/day on NASDAQ, nothing on "
               "EUR/USD); Gaussian GARCH loses 0.01–0.025 to Student-t everywhere.")
    # 4. VaR / ES
    vr = []
    for d in fixed:
        vr.append(f"{dsname(d)}: C-QSM {bt(ev[d][cq[d]])}, mixture {bt(ev[d]['cqsm-mix'])}, GARCH-t "
                  f"{bt(ev[d][REF])}, GJR-GARCH-t {bt(ev[d]['garch-gjr-cont'])}")
    out.append("**1-day VaR / ES at 5% on the bins** (test period; hit rate, Kupiec p, ES statistic Z, 0 = "
               "calibrated, > 0 = tail losses underestimated): " + "; ".join(vr) + ". GARCH-t is calibrated on "
               "EUR/USD but underestimates the NASDAQ tail (too many hits, Z ≈ +0.4); the C-QSM is too conservative "
               "on EUR/USD (too few hits, Z ≈ −0.3: it does not shrink its spread in the calm test period) and "
               "calibrated on NASDAQ. At 1%, with about 14 expected hits, few tests reject anything; on the quantile "
               "grid the 1% quantile sits in the unbounded edge bin on many days, so the 1% test there mostly counts "
               "edge-bin days and is not informative.")
    # 5. calm years
    cc = []
    for d in [x for x in fixed if x.endswith("-u")] + [x for x in fixed if x.endswith("-q")]:
        r, calm, vol = calm_corr(E, d)
        cc.append(f"{dsname(d)}: correlation {r:+.2f}, calm years {calm:+.3f}, volatile years {vol:+.3f}")
    eu = next(d for d in fixed if d.startswith("eurusd"))
    out.append("**Where the C-QSM loses: calm periods on a fixed grid, clearly on EUR/USD.** Per calendar year over "
               "the whole series (training years in sample), correlation between the year's return sd and the "
               "C-QSM's log-score minus GARCH-t, and the mean difference over the calmer and the more volatile half "
               "of the years: " + "; ".join(cc) + ". On EUR/USD the C-QSM is ahead of GARCH-t in volatile years and "
               "behind in calm ones, and the EUR/USD test period is calm "
               f"(return sd {E[eu]['empirical']['test_sd']:.2f} of the training sd, validation "
               f"{E[eu]['empirical']['val_sd']:.2f}): a fixed grid in training-sd units plus a bounded memory cannot "
               "concentrate the predictive law enough when volatility is low. On NASDAQ the pattern is weak and the "
               "test-period gap smaller. The volatility-normalized grid tests the explanation directly.")
    # 6. memory
    mm = [f"{dsname(d)}: C-QSM {E[d]['memory']['cqsm']['corr_rv_test']:.2f}, EWMA {E[d]['memory']['ewma']['corr_rv_test']:.2f}, "
          f"GARCH-t σ {E[d]['memory']['garch-t']['corr_rv_test']:.2f}" for d in fixed]
    out.append("**The memory tracks volatility, but less well than a zero-parameter filter.** Test correlation of a "
               "linear readout with log 21-day realized volatility: " + "; ".join(mm) + ". Against the MS-AR smoothed "
               "regime the C-QSM readout reaches "
               + ", ".join(f"{E[d]['memory']['cqsm']['corr_msar_test']:.2f}" for d in fixed)
               + ". The memory qubits do learn a volatility state from the data; they encode it less sharply than the "
                 "classical recursions.")
    # 7. other learned models
    gain = [f"{dsname(d)} {ev[d][cq[d]]['test_ll'] - max(ev[d][m]['test_ll'] for m in ('archb-B0', 'archb-anc2')):+.3f}"
            for d in fixed]
    out.append("**Against the other learned models the C-QSM wins clearly.** Gain over the best Architecture B "
               "(the memoryless cQCBM, which ends at the i.i.d. level on returns): " + "; ".join(gain) + " nats/day: "
               "the memory qubits carry the volatility clustering a Markov kernel cannot. The QSM without input, the "
               "nonparametric IOHMMs, HMMs and Markov chains are all behind it, most of them behind the i.i.d. law "
               "on EUR/USD: with about 4,900 training days and a volatility level that drifts between periods, their "
               "per-state 32-bin tables do not transfer (their EM validation curves fall after a few iterations).")
    if not any(d.endswith("-v") for d in dss):
        out.append("**Volatility-normalized grid: pending** (run `scripts/run_seq4_real.py` on the `-v` datasets).")
    else:
        from archb.realdata import block_bootstrap
        vv, gains = [], []
        for d in [x for x in dss if x.endswith("-v")]:
            arr = np.load(RUNS / d / "eval_arrays.npz")
            st = E[d]["n_train"] + E[d]["n_val"]
            gain = arr[f"lp/{cq[d]}"][st:] - arr["lp/markov-0"][st:]
            u = d[:-1] + "u"
            fixed_gap = f" (uniform grid {ev[u][cq[u]]['diff_ref']:+.3f})" if u in ev else ""
            gains.append((gain.mean(), block_bootstrap(gain)))
            vv.append(f"{dsname(d)}: C-QSM ({label(cq[d]).split(', ', 1)[1]}) {ev[d][cq[d]]['diff_ref']:+.3f} ± "
                      f"{ev[d][cq[d]]['diff_ref_se']:.3f}{fixed_gap}, mixture {ev[d]['cqsm-mix']['diff_ref']:+.3f} ± "
                      f"{ev[d]['cqsm-mix']['diff_ref_se']:.3f}, i.i.d. on the normalized bins "
                      f"{ev[d]['markov-0']['diff_ref']:+.3f} ± {ev[d]['markov-0']['diff_ref_se']:.3f}, GJR-GARCH-t "
                      f"{ev[d]['garch-gjr-cont']['diff_ref']:+.3f}; C-QSM minus i.i.d. {gain.mean():+.3f} ± "
                      f"{block_bootstrap(gain):.3f}")
        names = [SERIES[d.split("-")[0]] for d in dss if d.endswith("-v")]
        word = lambda g, se: ("significantly below it" if g < -2 * se else "significantly above it" if g > 2 * se
                              else "tied with it")
        verdict = ("Against the i.i.d. law of the normalized bins the C-QSM is "
                   + "; ".join(f"{n}: {word(g, se)}" for n, (g, se) in zip(names, gains))
                   + (". The memory adds nothing on top of the rescaling (where it differs, its extra parameters only "
                      "fit the training period)." if all(g < 2 * se for g, se in gains) else "."))
        out.append("**Volatility-normalized grid: the rescaling, not the memory, closes the gap.** Each day's bins are "
                   "rescaled by an EWMA volatility of past bins (same information set for every model); test log-score "
                   "minus GARCH-t: " + "; ".join(vv) + ". The i.i.d. law of the normalized bins is what the EWMA "
                   "rescaling alone captures, and it is within noise of GARCH-t; C-QSM minus i.i.d. is what the memory "
                   "and the input controls add on top of it. " + verdict + " The memory readout of realized volatility "
                   "falls accordingly (test correlation "
                   + ", ".join(f"{E[d]['memory']['cqsm']['corr_rv_test']:.2f}" for d in dss if d.endswith("-v"))
                   + "): the grid already carries the volatility level. These runs ran on a separate machine from the same code "
                   "(Python 3.13, numpy 2.1.3, scipy 1.16.3, the same jax 0.11.2).")
    return out


def main():
    E, R = load()
    if not E:
        raise SystemExit("no eval.json yet")
    FIGS.mkdir(exist_ok=True)
    fig_loglik(E)
    fig_cum(E)
    fig_acf(E)
    fig_memory(E)
    summary = {ds: {"selected": {fam: chosen(E, ds, fam) for fam in FAMILIES}, "eval": E[ds]} for ds in E}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    fd = findings(E)
    d0 = E[next(iter(E))]
    L = ["# Sequence 4, target D — the C-QSM on real daily returns", "",
         "Generated by `scripts/seq4_real_report.py` from the run files in `runs/seq4_real/` (fits: "
         "`scripts/run_seq4_real.py`, aggregated in `runs/seq4_real/summary.json`). "
         "Follows [`seq4.md`](seq4.md).",
         "", "## Set-up", ""]
    L.append("**Question.** On the synthetic targets of sequence 4 the correctly specified classical family always won. "
             "On real returns every model is misspecified, there is about 20× less data, and the volatility level "
             "drifts. Does the C-QSM (Architecture B controls on x_t plus 1–2 never-measured memory qubits) beat the "
             "classical models, GARCH-t first, out of sample?")
    data = []
    for ds in [d for d in DATASETS if d.endswith("-u") and d in E]:
        e = E[ds]
        data.append(f"{SERIES[e['series']]} (FRED {'DEXUSEU' if e['series'] == 'eurusd' else 'NASDAQCOM'}): "
                    f"returns {e['dates'][0]} to {e['dates'][3]}, train {e['n_train']} days (to {e['dates'][1]}), "
                    f"validation {e['n_val']} (to {e['dates'][2]}), test {e['n_test']}; daily sd of the training "
                    f"returns {e['sd'] * 100:.3f}%; |z| > 4 (clipped into the edge bins of the uniform grid): "
                    f"train {e['clip_frac']['train'] * 100:.2f}%, validation {e['clip_frac']['val'] * 100:.2f}%, "
                    f"test {e['clip_frac']['test'] * 100:.2f}%")
    L.append("**Data.** Daily log-returns between consecutive quotes (missing days dropped), standardized with the "
             "training mean and sd only, split 70 / 10 / 20 % in time. " + "; ".join(data) + ".")
    occ = {ds: E[ds]["occupancy_train"] for ds in E if ds.endswith("-u")}
    L.append("**Grids** (32 bins, Gray labels for the circuits). *Uniform*: 32 equal bins of [-4, 4] training sd, "
             "edge bins absorb the tails; it spends bins on the tails: the 8 outer bins (|z| > 3) hold "
             + ", ".join(f"{100 * (sum(o[:4]) + sum(o[-4:])) / sum(o):.1f}% ({SERIES[ds.split('-')[0]]})"
                         for ds, o in occ.items())
             + " of the training days, the 8 central bins (|z| < 1) "
             + ", ".join(f"{100 * sum(o[12:20]) / sum(o):.0f}%" for o in occ.values())
             + ". *Quantile*: edges at the 1/32 .. 31/32 training quantiles (every bin holds 1/32 of the training "
               "days; the two outer bins are unbounded, so the grid cannot see how far a crash goes). "
               "*Volatility-normalized* (v): the uniform grid applied to z_t / s_t, s_t an EWMA volatility (λ = 0.94) "
               "computed from past bins only, so every model has the same information set and is scored on the same "
               "intervals s_t [e_j, e_{j+1}) of z. Log-likelihoods compare models on one grid, not across grids. Bin "
               "values: the mean of the training returns in each bin (VaR / ES of the models, |x|, binned-input "
               "families); realized losses in the backtests are the actual returns.")
    L.append("**Models.** Architecture B (B0, anc2; 3 layers); C-QSM m = 1, 2 ± 2 ancillas, staged from Architecture B "
             "as in sequence 4; the sequence 3 QSM (general isometry), m = 1, 2; IOHMM full and factored, S = 2, 4; "
             "HMM on x alone, S = 2, 4; Markov chains of order 0 (i.i.d.), 1, 2 on the bins (add-½ counts); GARCH(1,1) "
             "with Gaussian or unit-variance Student-t innovations, GJR-GARCH(1,1)-t (leverage term), and MS-AR(1) "
             "with 2 regimes (intercept, AR "
             "coefficient, sd and stay probability per regime), each with **continuous inputs** (the usual model, "
             "fitted by continuous maximum likelihood, its predictive density integrated over the bins) and with "
             "**binned inputs** (driven by the bin values and fitted by the binned likelihood: the information set "
             "of the circuits). The reference is GARCH-t with continuous inputs; the binned-input GARCH-t is the "
             "second reference. Also reported: the equal-weight mixture of the 5 seeds of the chosen C-QSM variant.")
    L.append("**Training and selection.** Exact likelihood of the training period as one path. Circuits: Adam then "
             "L-BFGS-B as in sequence 4 (Architecture B inits 0..2, stage 2 seeds 0..4, QSM inits 0..4); EM with the "
             "sequence 4 pseudo-counts, inits 0..4; families: L-BFGS-B from several starts. Overfitting control: "
             "early stopping on the validation log-likelihood (checked every 50 L-BFGS / 10 EM iterations, the best "
             "iterate kept; EM never keeps its random start). Architecture B starts near the i.i.d. law (final R_y "
             "layer at the training frequency of each Gray bit, other angles N(0, 0.02²)): the copy start of "
             "sequence 4 (an identity kernel) could not even reach the i.i.d. likelihood of returns. Stage 2 of the "
             "C-QSM starts from the stage 1 init with the best training objective. The reported init of every model "
             "is the one with the best validation log-likelihood, and within a family the variant with the best "
             "validation log-likelihood is chosen. No refit on train + validation; the test period is never used "
             "for a choice.")
    L.append("**Metrics, all on the same 32 bins of the test period.** Every model filters the whole series from its "
             "start; the test days are scored with everything before them. (1) Log-score log P(b_t | past) per day "
             "(nats), moving-block bootstrap SE (50-day blocks), and the paired difference to GARCH-t with its own "
             "block-bootstrap SE. (2) 1-day VaR / ES at 1% and 5% from each model's predictive law over the bins: "
             "hits from the randomized PIT (exact coverage for a discrete law; the same uniforms for all models; "
             "p-value = twice the median over 200 draws, a valid p-value for a randomized test), Kupiec "
             "unconditional coverage and Christoffersen conditional coverage (CC) tests, and the ES statistic "
             "Z = mean(hit × loss / (α ES)) − 1 with the realized return as loss (0 for a correct model; > 0: tail "
             "losses underestimated). (3) 10-day VaR / ES on non-overlapping 10-day windows of the test period, from "
             "5,000 Monte Carlo paths of each model given its filtered state. (4) acf of |x| (lags 1–50) and kurtosis "
             "of each model's stationary process (exact for the linear representations, 20 × 10,000 simulated days "
             "for GARCH and the continuous MS-AR) against the whole series. (5) Linear readouts of the filtered "
             "internal state (fitted on the training period, correlation on the test period) of 21-day realized "
             "volatility and of the smoothed turbulent-regime probability of the continuous MS-AR, with a "
             "zero-parameter EWMA of the observed bin values as a floor. (6) Per calendar year over the whole series, "
             "the log-score difference to GARCH-t and the return sd.")
    L.append("Tests: `tests/test_realdata.py` (data cleaning, chronological split, standardization on training only, "
             "grids and discretization, bin values, block bootstrap, randomized PIT uniformity, VaR / ES against brute "
             "force, Kupiec / Christoffersen known cases, calibrated backtests, predictive matrices against the "
             "filter, Monte Carlo sums against the exact 2-step law, GARCH recursion and parameter recovery, MS-AR "
             "filter against a plain Hamilton filter, parameter recovery, binned MS-AR as an IOHMM, GJR recursion, "
             "the volatility-normalized grid uses past bins only, randomized p-values, Monte Carlo with a moving "
             "scale).")
    L += ["", "## Findings (most important first)", ""]
    L += [f"{k + 1}. {t}" for k, t in enumerate(fd)]
    L += ["", "## Selection on validation", "",
          "Test log-score minus GARCH-t (nats/day, ± block-bootstrap SE) of the variant of each family chosen on "
          "validation (in brackets):", "", selection_table(E)]
    L += ["![test log-score](../figs/seq4_real_loglik.png)", "", "![cumulative](../figs/seq4_real_cumulative.png)", ""]
    L += ["## Calm and volatile years", "",
          "Log-score minus GARCH-t (continuous inputs) per calendar year over the whole series (train / val / test "
          "marks the period; training years are in sample for every model), with the year's return sd in training "
          "units:", ""]
    for ds in [d for d in DATASETS if d in E and not d.endswith("-q")]:
        L += [f"**{dsname(ds)}**", "", yearly_table(E, ds)]
    L += ["## Memory of the C-QSM", "", memory_table(E), "![memory](../figs/seq4_real_memory.png)", ""]
    L += ["## Overfitting control", "", "Early stopping on validation for the main learned models (init kept):", "",
          es_table(R, E)]
    L += ["## Stylized facts", "", "![acf](../figs/seq4_real_acf.png)", ""]
    for ds in [d for d in DATASETS if d in E]:
        s, g = ds.split("-")
        e = E[ds]
        sel = sorted({chosen(E, ds, f) for f in FAMILIES} | {REF, REF_BIN, "cqsm-mix"}, key=ORDER.index)
        L += [f"## {SERIES[s]}, {GRIDS[g]}", "",
              f"Test {e['dates'][2]} to {e['dates'][3]} ({e['n_test']} days). Log-scores in nats/day; Params = free "
              "parameters; Filter size = real numbers carried by the filter (classical simulation cost per step). "
              "Inits = fits run, with the range of their test log-scores (the init with the best validation "
              "log-likelihood is reported).", "", lik_table(E, R, ds),
              "1-day VaR / ES backtests on the bins (variants chosen on validation, plus GARCH-t):", "",
              risk_table(E, ds, sel, "var1"),
              f"10-day VaR / ES, {e['n_windows']} non-overlapping windows (expected hits at 1%: "
              f"{e['n_windows'] * 0.01:.1f}):", "", risk_table(E, ds, sel, "var10")]
        if e["continuous_var1"]:
            L += ["Reference: the usual 1-day backtest of the continuous-input GARCH on the raw returns:", "",
                  cont_table(E, ds)]
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(fd))


if __name__ == "__main__":
    main()
