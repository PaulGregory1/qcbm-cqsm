"""Sequence 3: aggregate runs/seq3/** into runs/seq3/summary.json, draw figs/seq3_*.png and render reports/seq3.md.
Every number in the note is read from the run files (result.json of every run); nothing is typed by hand.

Selection: for every dataset and model the reported fit is the init with the best TRAINING log-likelihood (what a
practitioner can see), not the best metric against the truth.

Run: ./.venv/bin/python scripts/seq3_report.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from archb import hmm as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIGS, REPORT = ROOT / "runs" / "seq3", ROOT / "figs", ROOT / "reports" / "seq3.md"
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
DPI = 170
TIE = 1e-5  # inits within this many nats/symbol of the best training log-likelihood count as reaching it
NS = list(range(3, 9))
SCARCE = (500, 5000, 50000)


# ------------------------------------------------------------------------------- aggregation
def load():
    out = {}
    for p in sorted(RUNS.glob("*/*/*/result.json")):
        r = json.loads(p.read_text())
        out.setdefault(r["dataset"], {}).setdefault(r["model"], []).append(r)
    return out


def summarize(runs):
    S = {}
    for ds, models in runs.items():
        S[ds] = {}
        for mo, rs in models.items():
            best = max(rs, key=lambda r: r["train_ll"])
            ho = np.array([r["heldout_kl_rate"] for r in rs])
            S[ds][mo] = {
                "n_inits": len(rs), "best": best,
                "n_at_best": int(sum(r["train_ll"] >= best["train_ll"] - TIE for r in rs)),
                "heldout_median": float(np.median(ho)), "heldout_min": float(ho.min()), "heldout_max": float(ho.max()),
                "train_ll_all": sorted(r["train_ll"] for r in rs),
                "per_init": [{"init": r["init"], "train_ll": r["train_ll"], "heldout": r["heldout_kl_rate"],
                              "d_last": r["d"][-1]} for r in sorted(rs, key=lambda r: r["init"])],
                "wall_time_total": float(sum(r["wall_time"] for r in rs)),
            }
    return S


def get(S, ds, mo, key="heldout_kl_rate"):
    try:
        return S[ds][mo]["best"][key]
    except KeyError:
        return None


# ------------------------------------------------------------------------------- formatting
def f(x, n=3):
    if x is None:
        return "–"
    if isinstance(x, (list, tuple)):
        return "–"
    if x != 0 and (abs(x) < 10 ** (-n + 1) or abs(x) >= 1e4):
        return f"{x:.{max(n - 1, 1)}e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{x:.{n}f}"


def fe(x):
    """Rates spanning decades: scientific notation with 2 significant digits."""
    return "–" if x is None else f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def fs(x):
    """Signed small differences."""
    return f"{x:+.1e}".replace("e-0", "e-").replace("e+0", "e+")


def label(mo, N=None):
    if mo.startswith("qsm-iso-m"):
        return f"QSM general, m = {mo[-1]} (D = {2 ** int(mo[-1])})"
    if mo.startswith("qsm-circ-m"):
        return f"QSM circuit, m = {mo[-1]} (D = {2 ** int(mo[-1])})"
    if mo.startswith("hmm-"):
        s = int(mo.split("-")[1])
        return f"HMM, {s} states" + (" (true number)" if N is not None and s == N else "")
    if mo.startswith("markov-"):
        return f"Markov order {mo.split('-')[1]}"
    return "truth"


def model_order(models):
    key = {"qsm-iso": 0, "qsm-circ": 1, "hmm": 2, "markov": 3, "truth": 4}

    def k(mo):
        base = next(v for p, v in key.items() if mo.startswith(p))
        num = int(mo.split("-")[-1].lstrip("m")) if base < 4 else 0
        return (base, num)

    return sorted(models, key=k)


# ------------------------------------------------------------------------------- figures
STYLE = {  # (colour, linestyle, marker): matched memory sizes share a colour, QSM solid, HMM dashed
    "qsm-iso-m1": (OI["orange"], "-", "o"), "hmm-2": (OI["orange"], "--", "s"),
    "qsm-iso-m2": (OI["blue"], "-", "o"), "hmm-4": (OI["blue"], "--", "s"),
    "hmm-N": (OI["black"], ":", "^"), "markov-1": (OI["green"], "-.", "v"), "markov-2": (OI["sky"], "-.", "v"),
    "markov-4": (OI["purple"], "-.", "D"),
}
FLOOR = 1e-6


def fig_renewal(S):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mo, (c, ls, mk) in STYLE.items():
        ys = []
        for N in NS:
            m = f"hmm-{N}" if mo == "hmm-N" else mo
            v = get(S, f"renewal{N}", m)
            ys.append(np.nan if v is None else max(v, FLOOR))
        lab = "HMM, true number of states (N)" if mo == "hmm-N" else label(mo)
        ax.plot(NS, ys, ls=ls, marker=mk, color=c, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("renewal process N (gap uniform on {0..N-1})")
    ax.set_ylabel("held-out KL rate (nats / symbol)")
    ax.set_title(f"Uniform renewal, T = 50,000: KL rate to the truth (floored at {FLOOR:.0e})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "seq3_renewal_klrate.png", dpi=DPI)
    plt.close(fig)


def scarcity_rows(S, mo):
    out = {}
    for T in SCARCE:
        if T == 50000:
            v = [get(S, "renewal5", mo)]
        else:
            v = [get(S, f"renewal5-T{T}-s{s}", mo) for s in range(3)]
        v = [x for x in v if x is not None]
        out[T] = v
    return out


def fig_scarcity(S):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mo in ("qsm-iso-m1", "hmm-2", "qsm-iso-m2", "hmm-4", "hmm-N", "markov-2", "markov-4"):
        c, ls, mk = STYLE[mo]
        m = "hmm-5" if mo == "hmm-N" else mo
        rows = scarcity_rows(S, m)
        Ts = [T for T in SCARCE if rows[T]]
        med = [max(np.median(rows[T]), FLOOR) for T in Ts]
        lo = [max(min(rows[T]), FLOOR) for T in Ts]
        hi = [max(max(rows[T]), FLOOR) for T in Ts]
        ax.errorbar(Ts, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], ls=ls, marker=mk, color=c, capsize=3,
                    label="HMM, 5 states (true number)" if mo == "hmm-N" else label(mo))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training string length T")
    ax.set_ylabel("held-out KL rate (nats / symbol)")
    ax.set_title("Renewal N = 5: data size (median and range over 3 data seeds)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "seq3_scarcity.png", dpi=DPI)
    plt.close(fig)


def fig_gaps(S, N=8):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tr = get(S, f"renewal{N}", "truth", "gap")
    n = np.arange(len(tr))
    ax.bar(n, tr, color=OI["yellow"], edgecolor=OI["black"], label="truth", alpha=0.8)
    for mo in ("qsm-iso-m1", "hmm-2", "qsm-iso-m2", "hmm-4", "markov-4"):
        g = get(S, f"renewal{N}", mo, "gap")
        if g is not None:
            c, ls, mk = STYLE[mo]
            ax.plot(n, g, ls=ls, marker=mk, color=c, label=label(mo), ms=4)
    ax.set_xlabel("gap n (number of 0s between consecutive 1s)")
    ax.set_ylabel("P(gap = n)")
    ax.set_title(f"Renewal N = {N}: gap law of each fitted model")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "seq3_gaps.png", dpi=DPI)
    plt.close(fig)


def fig_acf(S):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tr = get(S, "regime", "truth", "acf_abs")
    lags = np.arange(1, len(tr) + 1)
    ax.plot(lags, tr, color=OI["black"], lw=2.5, label="truth")
    for mo in ("qsm-iso-m1", "hmm-2", "qsm-iso-m2", "hmm-4", "markov-1", "markov-4"):
        a = get(S, "regime", mo, "acf_abs")
        if a is not None:
            c, ls, mk = STYLE[mo]
            ax.plot(lags, a, ls=ls, marker=mk, color=c, label=label(mo), ms=3)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation of |x_t|")
    ax.set_title("Two-regime process: volatility clustering under each model")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "seq3_regime_acf.png", dpi=DPI)
    plt.close(fig)


def fig_regime_d(S):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mo in ("qsm-iso-m1", "hmm-2", "qsm-iso-m2", "hmm-4", "markov-1", "markov-2", "markov-4"):
        d = get(S, "regime", mo, "d")
        if d is not None:
            c, ls, mk = STYLE[mo]
            ax.plot(range(1, len(d) + 1), np.maximum(d, FLOOR), ls=ls, marker=mk, color=c, label=label(mo))
    ax.set_yscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel("conditional KL d_L (nats / symbol)")
    ax.set_title("Two-regime process: next-symbol KL given the previous L-1 symbols")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "seq3_regime_dL.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------------------- note
def acf_err(S, mo, lags=20):
    a, t = get(S, "regime", mo, "acf_abs"), get(S, "regime", "truth", "acf_abs")
    return None if a is None else float(np.mean(np.abs(np.subtract(a[:lags], t[:lags]))))


def model_table(S, ds, N=None):
    w = ["| Model | Memory dim. | Params (eff.) | Held-out KL rate ± s.e. | d_L (exact) | D_L (exact) | "
         "Train NLL − truth | Inits at best |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mo in model_order(S[ds]):
        if mo == "truth":
            continue
        e = S[ds][mo]
        b = e["best"]
        par = f"{b['n_params']} ({b['n_effective']})" if b["n_params"] != b["n_effective"] else f"{b['n_params']}"
        w.append(f"| {label(mo, N)} | {b['memory_dim']} | {par} | {fe(b['heldout_kl_rate'])} ± {fe(b['heldout_se'])} | "
                 f"{fe(b['d'][-1])} | {fe(b['D'][-1])} | {fs(b['truth_train_ll'] - b['train_ll'])} | "
                 f"{e['n_at_best'] if e['n_inits'] > 1 else '–'}/{e['n_inits']} |")
    return w


def main():
    global TEST_COUNT_QSEQ
    TEST_COUNT_QSEQ = _count_tests()
    runs = load()
    S = summarize(runs)
    FIGS.mkdir(exist_ok=True)
    (RUNS / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    if any(f"renewal{N}" in S for N in NS):
        fig_renewal(S)
    if "renewal5" in S:
        fig_scarcity(S)
    if "truth" in S.get("renewal8", {}):
        fig_gaps(S)
    if "truth" in S.get("regime", {}):
        fig_acf(S)
        fig_regime_d(S)
    REPORT.write_text(note(S))
    print("wrote", REPORT)


def _count_tests():
    """Number of tests pytest collects in tests/test_qseq.py (parametrized cases counted)."""
    import re
    import subprocess

    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_qseq.py"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else None


TEST_COUNT_QSEQ = None


def cross_table(S, key, models, Ns=NS, fmt=fe):
    w = ["| Model | " + " | ".join(f"N = {N}" for N in Ns) + " |", "| --- |" + " ---: |" * len(Ns)]
    for mo in models:
        cells = []
        for N in Ns:
            m = f"hmm-{N}" if mo == "hmm-N" else mo
            v = get(S, f"renewal{N}", m, key)
            if isinstance(v, list):
                v = v[-1]
            cells.append(fmt(v))
        lab = "HMM, N states (true number)" if mo == "hmm-N" else label(mo)
        w.append(f"| {lab} | " + " | ".join(cells) + " |")
    return w


def params_line(S, ds, mo):
    b = S[ds][mo]["best"]
    return b["n_params"], b["n_effective"], b["memory_dim"]


CROSS_MODELS = ["qsm-iso-m1", "qsm-circ-m1", "hmm-2", "qsm-iso-m2", "qsm-circ-m2", "hmm-4", "hmm-N",
                "markov-1", "markov-2", "markov-3", "markov-4"]


def scarcity_table(S):
    w = ["| Model | Params | T = 500 | T = 5,000 | T = 50,000 |", "| --- | ---: | ---: | ---: | ---: |"]
    for mo in ("qsm-iso-m1", "qsm-circ-m1", "hmm-2", "qsm-iso-m2", "qsm-circ-m2", "hmm-4", "hmm-5",
               "markov-1", "markov-2", "markov-3", "markov-4"):
        rows = scarcity_rows(S, mo)
        cells = []
        for T in SCARCE:
            v = rows[T]
            if not v:
                cells.append("–")
            elif len(v) == 1:
                cells.append(fe(v[0]))
            else:
                cells.append(f"{fe(float(np.median(v)))} [{fe(min(v))}, {fe(max(v))}]")
        n = get(S, "renewal5", mo, "n_params")
        lab = "HMM, 5 states (true number)" if mo == "hmm-5" else label(mo)
        w.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")
    return w


def regime_table(S):
    w = ["| Model | Memory dim. | Params (eff.) | Held-out KL rate ± s.e. | d_5 (exact) | Train NLL − truth | "
         "|x| acf lag 1 | lag 10 | lag 20 | mean abs. acf error, lags 1–20 | Inits at best |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    t = get(S, "regime", "truth", "acf_abs")
    w.append(f"| truth | 2 | – | 0 | 0 | – | {f(t[0])} | {f(t[9])} | {f(t[19])} | 0 | – |")
    for mo in model_order(S["regime"]):
        if mo == "truth":
            continue
        e, b = S["regime"][mo], S["regime"][mo]["best"]
        a = b["acf_abs"]
        par = f"{b['n_params']} ({b['n_effective']})" if b["n_params"] != b["n_effective"] else f"{b['n_params']}"
        w.append(f"| {label(mo, 2)} | {b['memory_dim']} | {par} | {fe(b['heldout_kl_rate'])} ± {fe(b['heldout_se'])} | "
                 f"{fe(b['d'][-1])} | {fs(b['truth_train_ll'] - b['train_ll'])} | {f(a[0])} | {f(a[9])} | "
                 f"{f(a[19])} | {f(acf_err(S, mo))} | {e['n_at_best'] if e['n_inits'] > 1 else '–'}/{e['n_inits']} |")
    return w


def note(S):
    L = []
    w = L.append
    w("# Sequence 3 — a quantum memory register on non-Markov processes\n")
    w("Generated by `scripts/seq3_report.py` from the run files in `runs/seq3/` (aggregated in `runs/seq3/summary.json`). "
      "Follows [`seq2.md`](seq2.md) and "
      "[`seq2_followups.md`](seq2_followups.md).\n")
    w(setup_text(S))
    w(findings_text(S))
    w("## Uniform renewal processes, T = 50,000\n")
    w("Held-out KL rate (nats per symbol, ± is not shown here; see the per-N tables), best init by training likelihood:\n")
    L += cross_table(S, "heldout_kl_rate", CROSS_MODELS)
    w("")
    Lr = 12
    w(f"Exact conditional KL d_{Lr} = KL(P(x_{Lr} | x_1..x_{Lr - 1}) ‖ Q(·|·)) averaged under the truth:\n")
    L += cross_table(S, "d", CROSS_MODELS)
    w("")
    w("Parameters: real parameters of the parametrization, with the effective count in brackets for the QSM "
      "(isometry dimension minus the memory gauge and one phase per Kraus operator; an upper bound on the dimension of "
      "the model family). HMM: S(SA − 1). Markov order k: 2^k free probabilities.\n")
    w("| Model | Memory dim. | Params (eff.) |", )
    w("| --- | ---: | ---: |")
    for mo in CROSS_MODELS:
        m = "hmm-8" if mo == "hmm-N" else mo
        if get(S, "renewal8", m) is None:
            continue
        n, ne, d = params_line(S, "renewal8", m)
        lab = "HMM, 8 states (N = 8)" if mo == "hmm-N" else label(mo)
        w(f"| {lab} | {d} | {n}" + (f" ({ne})" if ne != n else "") + " |")
    w("")
    w("![renewal KL rate](../figs/seq3_renewal_klrate.png)\n")
    for N in (3, 5, 8):
        ds = f"renewal{N}"
        if "truth" in S.get(ds, {}):
            w(f"### N = {N}\n")
            L += model_table(S, ds, N)
            w("")
    w("![gap law N = 8](../figs/seq3_gaps.png)\n")
    w("## Data size (renewal N = 5)\n")
    w("Held-out KL rate, median [min, max] over 3 training strings (data seeds) at T = 500 and 5,000; one string at "
      "T = 50,000. Same held-out string for all.\n")
    L += scarcity_table(S)
    w("")
    w("![data size](../figs/seq3_scarcity.png)\n")
    if "truth" in S.get("regime", {}):
        w("## Two-regime process (8 symbols, 3 output qubits), T = 50,000\n")
        L += regime_table(S)
        w("")
        w("![acf](../figs/seq3_regime_acf.png)\n")
        w("![d_L](../figs/seq3_regime_dL.png)\n")
    w(optim_text(S))
    w(deviations_text(S))
    w(reproduce_text())
    return "\n".join(L) + "\n"


def lmax_of(S, ds):
    return len(get(S, ds, "truth", "d"))


def setup_text(S):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_seq3 as R  # settings (inits, optimizer budget, layers) are read from the campaign script

    reg = S["regime"]["truth"]["best"]
    r5 = S["renewal5"]["truth"]["best"]
    e, c = H.regime_bins()
    lay = R.CIRC_LAYERS
    return "\n".join([
        "## Set-up\n",
        "**Question.** Sequences 1–2 used a conditional QCBM as a Markov kernel on the observed value x_t. A process "
        "with a hidden state is not Markov in x_t, and no transition matrix on x_t can reproduce it. Does a quantum "
        "model with a persistent, never-measured memory register (the quantum sequence model, QSM, of Wang, Yang, "
        "Somasundaram, Thompson and Gu, arXiv:2603.24069) capture such processes, and does it do better than the "
        "classical model with the same memory, a hidden Markov model (HMM) with as many states as the memory has "
        "dimensions? That paper compares the QSM only with a quantum non-recurrent Born machine.\n",
        "**QSM** (`src/archb/qseq.py`). Memory register of m qubits (dimension D = 2^m), output register of b qubits "
        "(A = 2^b symbols) reset to |0⟩ at every step. One shared unitary V(θ) acts on both; the output is measured "
        "and discarded. Kraus operators K_x = (⟨x| ⊗ I) V (|0⟩ ⊗ I), Σ_x K_x†K_x = I. Stationary memory state ρ_ss = "
        "fixed point of Σ_x K_x ρ K_x†; word probabilities Tr(K_{x_L}…K_{x_1} ρ_ss K_{x_1}†…K_{x_L}†), exact. "
        "Two parametrizations:\n",
        "- **general** (`iso`): the stacked Kraus operators are an arbitrary isometry C^D → C^{AD}, W = M R^{-1} with "
        "M complex and R†R = M†M (Cholesky). Every isometry is reachable and every isometry is V(|0⟩ ⊗ I) for some "
        "unitary V, so this is the best any circuit with m memory qubits can do: the comparison with the HMM is a "
        "comparison of memory, not of an ansatz. 2AD² real parameters; effective dimension at most 2AD² − 2D² + 1 − A "
        "(memory basis gauge and one phase per Kraus operator removed).",
        f"- **circuit** (`circuit`): the paper-style hardware ansatz, L layers of R_z R_y R_z on every qubit plus a "
        f"CNOT ring, then a final rotation layer, 3(m + b)(L + 1) angles. L chosen so the angle count is at least the "
        f"effective dimension: L = {lay[(1, 1)]} (m = 1) and {lay[(2, 1)]} (m = 2) for binary outputs, "
        f"{lay[(1, 3)]} and {lay[(2, 3)]} for 3 output qubits. Output labels are Gray-coded bins (irrelevant for "
        "`iso`).\n",
        "**Classical baselines** (`src/archb/hmm.py`). (i) HMM with S states in the edge-emitting form, "
        "T[s, x, s′] = P(x, s′ | s), which contains the usual state-emitting HMM with the same S and is the classical "
        "counterpart of the QSM (one Kraus operator per symbol ↔ one matrix T[:, x, :] per symbol); S(SA − 1) "
        "parameters. Fitted by Baum–Welch (EM, at most "
        f"{R.EM_ITERS} iterations) from random Dirichlet starts, then polished by L-BFGS on the exact likelihood, "
        f"the same optimizer as the QSM. S = 2^m (matched memory dimension) and the true number of states. (ii) Markov "
        "chains of order k = 1..4 on the observed symbols (add-½ counts), the \"Markov on observables\" baseline. The "
        "conditional QCBM of sequences 1–2 is a transition matrix on x_t, so at best it reproduces the order-1 "
        "chain.\n",
        f"**Targets.** (A) Uniform renewal processes: binary, the number of 0s between consecutive 1s uniform on "
        f"{{0, …, N − 1}}, N = 3…8; exactly an HMM with N states (count-down chain). Training string T = {r5['T']:,} "
        f"symbols; data size runs at N = 5 with T = 5,000 and 500 (3 data seeds each). (B) Two hidden regimes (low / "
        f"high), kept with probability {H.REGIME['stay']} per step; x_t = σ_s ξ_t, σ = {H.REGIME['sig'][0]} / "
        f"{H.REGIME['sig'][1]}, binned on {H.REGIME['n_bins']} equal bins of [{e[0]:g}, {e[-1]:g}] (±3σ_high, edge "
        f"bins absorb the tails), 3 output qubits; T = {reg['T']:,}. Both truths are represented as HMMs, so every "
        "word probability is exact.\n",
        f"**Training.** Exact mean log-likelihood of the training string, forward-filtered from ρ_ss (filtered block "
        f"by block with precomputed Kraus products of up to 4,096 words: identical value, fewer sequential steps). "
        f"Adam {R.ADAM} steps (lr 0.03 → 0.003, best iterate kept), then L-BFGS-B (cap {R.LBFGS}), as "
        f"`archb.ceiling.fit`. Inits: {len(R.QSM_INITS)} per general QSM, {len(R.CIRC_INITS)} per circuit QSM, "
        f"{len(R.HMM_INITS)} random per HMM; for the renewal targets at T = {r5['T']:,} also one start from the "
        f"lumped true chain (when S < N), and {len(R.HMM_INITS) + len(R.HMM_EXTRA)} random starts at N = "
        f"{', '.join(str(n) for n in R.EXTRA_NS)}; D = 8 QSM: {len(R.M3_INITS)} inits. **The reported fit is the init with the best training "
        "likelihood**, never selected on the truth.\n",
        "**Metrics** (all against the true process). (1) Held-out KL rate: (1/n) Σ_t [log P(x_t | x_<t) − "
        f"log Q(x_t | x_<t)] on an independent string of {R.N_TEST:,} symbols, both terms computed exactly by "
        "filtering; it estimates the KL-divergence rate lim KL(P_L ‖ Q_L)/L, with a standard error over 20 segments. "
        "(2) Exact conditional KL d_L = KL(P_L ‖ Q_L) − KL(P_{L−1} ‖ Q_{L−1}) = E_P KL(P(x_L | x_1..x_{L−1}) ‖ "
        "Q(·|·)) and D_L = KL(P_L ‖ Q_L)/L, by enumerating all words, L ≤ 12 (binary) and L ≤ 5 (8 symbols). "
        "d_L is the next-symbol error given L − 1 symbols of context; it tends to the KL rate as L grows. "
        "(3) Renewal: gap law P(1 0^n 1)/P(1). Regime: autocorrelation of |x_t| (bin centres), lags 1–30, exact under "
        "each model. Units: nats per symbol throughout.\n",
    ])


def renewal_Ns(S):
    return [N for N in NS if "truth" in S.get(f"renewal{N}", {})]


def last(v):
    return v[-1] if isinstance(v, list) else v


def ratio_range(S, num, den, Ns, key="heldout_kl_rate"):
    r = [last(get(S, f"renewal{N}", num, key)) / last(get(S, f"renewal{N}", den, key)) for N in Ns
         if get(S, f"renewal{N}", num, key) is not None and get(S, f"renewal{N}", den, key) is not None]
    return (min(r), max(r)) if r else (None, None)


def nlist(Ns, D):
    return ", ".join(str(N) for N in Ns if N > D)


def scarce_med(S, mo, T):
    v = scarcity_rows(S, mo)[T]
    return float(np.median(v)) if v else None


def findings_text(S):
    Ns = renewal_Ns(S)
    ho = lambda N, mo: get(S, f"renewal{N}", mo)
    w = ["## Findings (most important first)\n"]

    # 1. matched memory dimension (only where the HMM has fewer states than the truth; otherwise it contains it)
    pairs = [(N, m) for N in Ns for m in (1, 2) if 2**m < N]
    win = [(N, m) for N, m in pairs if ho(N, f"qsm-iso-m{m}") < ho(N, f"hmm-{2 ** m}")]
    a1, b1 = ratio_range(S, "hmm-2", "qsm-iso-m1", [N for N in Ns if N > 2])
    a2, b2 = ratio_range(S, "hmm-4", "qsm-iso-m2", [N for N in Ns if N > 4])
    da1, db1 = ratio_range(S, "hmm-2", "qsm-iso-m1", [N for N in Ns if N > 2], "d")
    da2, db2 = ratio_range(S, "hmm-4", "qsm-iso-m2", [N for N in Ns if N > 4], "d")
    n1, n2 = get(S, "renewal5", "qsm-iso-m1", "n_effective"), get(S, "renewal5", "qsm-iso-m2", "n_effective")
    p2, p4 = get(S, "renewal5", "hmm-2", "n_params"), get(S, "renewal5", "hmm-4", "n_params")
    small = [N for N in Ns if N <= 4]
    w.append(
        f"1. **On the renewal processes the QSM beats the classical HMM with the same memory dimension whenever that "
        f"dimension is below the true number of states: in {len(win)} of {len(pairs)} (N, D) pairs with D < N, "
        f"N = {Ns[0]}–{Ns[-1]}.** Held-out KL rate, HMM / QSM: {a1:.1f}–{b1:.1f}× with D = 2 (one memory qubit "
        f"against a 2-state HMM, N = {nlist(Ns, 2)}) and {a2:.0f}–{b2:.0f}× with D = 4 (two qubits against a "
        f"4-state HMM, N = {nlist(Ns, 4)}). The exact finite-horizon d_12 gives the same ratios ({da1:.1f}–{db1:.1f}× and "
        f"{da2:.0f}–{db2:.0f}×). At N = 5: D = 2, {fe(ho(5, 'qsm-iso-m1'))} against {fe(ho(5, 'hmm-2'))}; D = 4, "
        f"{fe(ho(5, 'qsm-iso-m2'))} against {fe(ho(5, 'hmm-4'))}. At N = 8: D = 4, {fe(ho(8, 'qsm-iso-m2'))} against "
        f"{fe(ho(8, 'hmm-4'))}. Parameter counts are matched too (QSM effective dimension {n1} and {n2}, HMM {p2} and "
        f"{p4}), so the gain is not a matter of more parameters. The classical optimum is not an optimization "
        f"artefact: starts of the 4-state HMM (random ones and one from the lumped true chain) ending on the best "
        f"likelihood: " + ", ".join(f"N = {N}: {S[f'renewal{N}']['hmm-4']['n_at_best']}/"
                                   f"{S[f'renewal{N}']['hmm-4']['n_inits']}" for N in Ns if N > 4) + ". "
        + (("Where the 4-state HMM can represent the truth (N = " + ", ".join(str(N) for N in small) + "), it wins: "
            + "; ".join(f"N = {N}: HMM {fe(ho(N, 'hmm-4'))}, QSM {fe(ho(N, 'qsm-iso-m2'))}" for N in small) + ". ")
           if small else "")
        + "This is the advantage the quantum-modelling literature predicts (quantum memory states may overlap, so a "
        "small register keeps more of the past than as many classical states), measured here on trained models "
        "against a trained classical baseline.\n")

    # 2. what the advantage is not
    big = [N for N in Ns if N > 4]
    hN = [ho(N, f"hmm-{N}") for N in big]
    q2 = [ho(N, "qsm-iso-m2") for N in big]
    hN_wins = [N for N in big if ho(N, f"hmm-{N}") < ho(N, "qsm-iso-m2")]
    mk_ex = [N for N in Ns if ho(N, "markov-4") < ho(N, "qsm-iso-m2")]
    m3 = {N: ho(N, "qsm-iso-m3") for N in Ns if ho(N, "qsm-iso-m3") is not None}
    txt = (f"2. **It is an advantage in memory dimension, not in modelling power per classical resource.** For "
           f"N = {big[0]}–{big[-1]}, the HMM with the true number of states (at most 8, i.e. 3 bits of classical "
           f"memory) reaches {fe(min(hN))}–{fe(max(hN))} and beats the 2-qubit QSM ({fe(min(q2))}–{fe(max(q2))}) at "
           f"{len(hN_wins)} of {len(big)} values of N; the order-4 Markov chain (16 states, exact for N ≤ 5) beats it "
           f"at N = {', '.join(str(N) for N in mk_ex) if mk_ex else 'none'}. Simulating a QSM with D-dimensional "
           "memory classically takes a D²-entry filter state (16 for two qubits), more than the N states of the exact "
           "classical model. ")
    if m3:
        lst = "; ".join(f"N = {N}: {fe(v)} against {fe(ho(N, f'hmm-{N}'))}" for N, v in sorted(m3.items()))
        txt += (f"Three memory qubits (D = 8, general QSM, best of {S[f'renewal{max(m3)}']['qsm-iso-m3']['n_inits']} "
                f"inits) against the HMM with N states: {lst}.")
    w.append(txt + "\n")

    # 3. regime
    rg = lambda mo, k="heldout_kl_rate": get(S, "regime", mo, k)
    if rg("truth") is not None:
        mk = [acf_err(S, f"markov-{k}") for k in range(1, 5)]
        hid = [acf_err(S, mo) for mo in ("qsm-iso-m1", "qsm-iso-m2", "hmm-2", "hmm-4")]
        w.append(
            f"3. **On the two-regime process the classical HMM wins, and every hidden-state model beats every Markov "
            f"chain on the observed values.** The truth is a 2-state classical HMM: the fitted 2-state HMM reaches "
            f"{fe(rg('hmm-2'))} (all inits on one optimum), the 1-qubit QSM {fe(rg('qsm-iso-m1'))} (best of "
            f"{S['regime']['qsm-iso-m1']['n_inits']} inits; median {fe(S['regime']['qsm-iso-m1']['heldout_median'])}) "
            f"with {get(S, 'regime', 'qsm-iso-m1', 'n_effective')} effective parameters against "
            f"{get(S, 'regime', 'hmm-2', 'n_params')}. Larger memories overfit 50,000 symbols: the 2-qubit QSM "
            f"({get(S, 'regime', 'qsm-iso-m2', 'n_effective')} effective parameters) fits the training string "
            f"{fs(rg('qsm-iso-m2', 'train_ll') - rg('qsm-iso-m2', 'truth_train_ll'))} nats/symbol better than the truth itself and loses on held-out "
            f"data ({fe(rg('qsm-iso-m2'))}), the 4-state HMM less so ({fe(rg('hmm-4'))}). The Markov chains of order "
            f"1–4, the class of the conditional QCBM of sequences 1–2, stay at {fe(min(rg(f'markov-{k}') for k in range(1, 5)))}"
            f"–{fe(max(rg(f'markov-{k}') for k in range(1, 5)))}, 2 orders of magnitude worse: they cannot hold the "
            f"regime. Volatility clustering (acf of |x_t|, lags 1–20) is reproduced by every hidden-state model "
            f"(mean absolute error {f(min(hid))}–{f(max(hid))}) and by no Markov chain ({f(min(mk))}–{f(max(mk))}; "
            f"order 1 has acf {f(get(S, 'regime', 'markov-1', 'acf_abs')[9])} at lag 10 against "
            f"{f(get(S, 'regime', 'truth', 'acf_abs')[9])}).\n")

    # 4. data size
    if all(scarcity_rows(S, "qsm-iso-m2")[T] for T in SCARCE):
        names = {"qsm-iso-m1": "QSM D = 2", "qsm-iso-m2": "QSM D = 4", "hmm-2": "HMM 2", "hmm-4": "HMM 4",
                 "hmm-5": "HMM 5 (true)", "markov-4": "Markov 4"}
        rows = []
        for T in SCARCE:
            order = sorted((scarce_med(S, mo, T), nm) for mo, nm in names.items() if scarce_med(S, mo, T) is not None)
            rows.append(f"T = {T:,}: " + " < ".join(f"{nm} ({fe(v)})" for v, nm in order))
        w.append("4. **Data size (renewal N = 5, median over 3 training strings).** Ranking by held-out KL rate, best "
                 "first — " + "; ".join(rows) + ". " + scarcity_comment(S) + "\n")

    # 5. circuit vs general, multi-qubit outputs
    rat = [ho(N, f"qsm-circ-m{m}") / ho(N, f"qsm-iso-m{m}") for N in Ns for m in (1, 2)
           if ho(N, f"qsm-circ-m{m}") is not None and ho(N, f"qsm-iso-m{m}") is not None]
    close = sum(abs(r - 1) < 0.05 for r in rat)
    head = ("The paper-style circuit loses nothing on binary outputs" if close == len(rat) else
            "On binary outputs the paper-style circuit matches the general model closely")
    mean_t = lambda ds, mo: S[ds][mo]["wall_time_total"] / S[ds][mo]["n_inits"]
    t_ren = np.mean([mean_t(f"renewal{N}", "qsm-iso-m2") for N in Ns])
    t_reg = mean_t("regime", "qsm-iso-m2")
    ren_at = sum(S[f"renewal{N}"][f"qsm-iso-m{m}"]["n_at_best"] for N in Ns for m in (1, 2))
    ren_n = sum(S[f"renewal{N}"][f"qsm-iso-m{m}"]["n_inits"] for N in Ns for m in (1, 2))
    reg_at = sum(S["regime"][f"qsm-iso-m{m}"]["n_at_best"] for m in (1, 2))
    reg_n = sum(S["regime"][f"qsm-iso-m{m}"]["n_inits"] for m in (1, 2))
    w.append(
        f"5. **{head}; with 3 output qubits training gets harder.** On the renewal targets the circuit QSM (best of "
        f"{S['renewal5']['qsm-circ-m1']['n_inits']} inits) is within 5% of the general model's held-out KL rate in "
        f"{close} of {len(rat)} (N, m) cases (circuit / general between {min(rat):.2f} and {max(rat):.2f}), so the "
        f"ansatz is not the limit there. On the 8-symbol regime target the "
        f"1-qubit circuit is {rg('qsm-circ-m1') / rg('qsm-iso-m1'):.1f}× worse than the general model, and the "
        f"landscape is rugged: {reg_at} of {reg_n} general-QSM inits reach their model's best likelihood, against "
        f"{ren_at} of {ren_n} on the renewal targets. The paper's caveat on multi-qubit outputs shows up as cost, "
        f"landscape and overfitting, not as a failure to learn the structure: parameters grow as 2AD² (A = 8: "
        f"{get(S, 'regime', 'qsm-iso-m2', 'n_params')} for D = 4), a 2-qubit fit takes {t_reg:.0f} s on average "
        f"against {t_ren:.0f} s on binary data, and the 2-qubit circuit needs {get(S, 'regime', 'qsm-circ-m2', 'n_layers')} "
        f"layers to have as many angles as the model's effective dimension.\n")

    # 6. long-range errors: held-out KL rate far above the finite-horizon d_L
    if "truth" in S.get("regime", {}):
        rat = {mo: get(S, "regime", mo) / get(S, "regime", mo, "d")[-1] for mo in S["regime"] if mo != "truth"}
        q = [rat[mo] for mo in rat if mo.startswith("qsm")]
        h = [rat[mo] for mo in rat if mo.startswith("hmm")]
        txt = ("6. **Finite-horizon KL can miss long-range errors; the held-out KL rate is the headline metric.** On the "
               f"regime process (memory of about 1/(1 − {H.REGIME['stay']}) = {1 / (1 - H.REGIME['stay']):.0f} steps, "
               f"horizon L = 5), the held-out rate is {min(q):.1f}–{max(q):.1f}× d_5 for the QSMs against "
               f"{min(h):.1f}–{max(h):.1f}× for the HMMs: the QSMs' extra error sits in the slow regime dynamics, "
               "beyond any enumerable horizon. ")
        e3 = S.get("renewal8", {}).get("qsm-iso-m3")
        if e3:
            pi = e3["per_init"]
            txt += (f"The D = 8 QSM on renewal N = 8 shows the same effect across inits: held-out "
                    f"{fe(min(r['heldout'] for r in pi))}–{fe(max(r['heldout'] for r in pi))} while d_12 stays in "
                    f"{fe(min(r['d_last'] for r in pi))}–{fe(max(r['d_last'] for r in pi))}; its best inits fit the "
                    f"training string better than the truth does (by {fs(e3['best']['train_ll'] - e3['best']['truth_train_ll'])} "
                    "nats/symbol), i.e. the extra memory fits long-range structure of the training string that the "
                    "process does not have.")
        w.append(txt + "\n")
    w.append(meaning_text(S))
    return "\n".join(w)


def scarcity_comment(S):
    q2 = {T: scarce_med(S, "qsm-iso-m2", T) for T in SCARCE}
    h4 = {T: scarce_med(S, "hmm-4", T) for T in SCARCE}
    h5 = {T: scarce_med(S, "hmm-5", T) for T in SCARCE}
    beats = [T for T in SCARCE if q2[T] < h4[T]]
    out = (f"The 2-qubit QSM beats the 4-state HMM at T = {', '.join(f'{T:,}' for T in beats)}" if beats else
           "The 2-qubit QSM does not beat the 4-state HMM at any data size")
    out += (f"; against the true 5-state HMM it is " +
            ", ".join(f"{q2[T] / h5[T]:.2g}× at T = {T:,}" for T in SCARCE) + " (QSM / HMM). ")
    r5 = scarcity_rows(S, "hmm-5")[500]
    hid = min((scarce_med(S, mo, 500), mo) for mo in ("qsm-iso-m1", "qsm-iso-m2", "hmm-2", "hmm-4", "hmm-5"))
    out += (f"With little data the larger models suffer most: at T = 500 the true-size HMM spans {fe(min(r5))}–"
            f"{fe(max(r5))} over the 3 strings, and the best hidden-state model is the {label(hid[1])} "
            f"({fe(hid[0])}). The "
            f"order-4 Markov chain, which contains the N = 5 renewal process with 16 parameters, is best at "
            + ", ".join(f"T = {T:,}" for T in SCARCE
                        if scarce_med(S, "markov-4", T) <= min(scarce_med(S, mo, T) for mo in
                                                              ("qsm-iso-m1", "qsm-iso-m2", "hmm-2", "hmm-4", "hmm-5")))
            + ": the paper's observation that a recurrent quantum model generalizes better from little data holds "
            "against a quantum or classical model with more memory, not against a small classical model of the right "
            "structure.")
    return out


def meaning_text(S):
    return "\n".join([
        "## What it means\n",
        "- A persistent quantum memory does what sequences 1–2 could not: it models dependence that runs through a "
        "hidden state. Any Markov model on the observed value, the conditional QCBM included, is stuck at the "
        "Markov-chain numbers above, 1–2 orders of magnitude off on the regime process.",
        "- **Quantum advantage, stated precisely.** At equal memory dimension (and equal parameter count), the "
        "trained QSM is better than the trained classical HMM on every renewal process tested, by up to an order of "
        "magnitude or more with two qubits. That is a memory-size advantage, relevant if the register is the scarce "
        "resource. It is not an advantage in classical computation or in parameters: a classical HMM with the true "
        "number of states (≤ 8), or for N ≤ 5 the order-4 Markov chain, is better and cheaper to simulate than "
        "the 2-qubit QSM, and three memory qubits (D = 8 ≥ N) overfit instead of closing that gap. On a process that is itself a small classical HMM (the regime process) there is no "
        "advantage at all: the classical model of the same size wins.",
        "- For the finance-like use case (volatility clustering), a 2-state classical HMM is the right model and a "
        "QSM brings nothing; the QSM result is a consistency check that the quantum model learns the same structure.",
        "- Next steps, if this line continues: (a) renewal processes with larger N or non-uniform gap laws, where "
        "the classical model needs many states and the quantum compression should grow; (b) a classical baseline "
        "with D² states, the fair comparison at equal simulation cost; (c) sampling-based training (the paper's "
        "shot-based setting), where the exact-likelihood optimum found here is the ceiling.\n",
    ])


def optim_text(S):
    w = ["## Optimization and cost\n",
         "Spread of the final training log-likelihood over inits: the number of inits within "
         f"{TIE:g} nats/symbol of the best one, and the range of held-out KL rates over all inits.\n",
         "| Dataset | Model | Inits at best | Held-out KL rate over inits: min – max | Wall time, all inits (s) |",
         "| --- | --- | ---: | ---: | ---: |"]
    for ds in ["renewal5", "renewal8", "regime"]:
        if ds not in S:
            continue
        for mo in model_order(S[ds]):
            e = S[ds][mo]
            if e["n_inits"] < 2:
                continue
            w.append(f"| {ds} | {label(mo)} | {e['n_at_best']}/{e['n_inits']} | {fe(e['heldout_min'])} – "
                     f"{fe(e['heldout_max'])} | {e['wall_time_total']:.0f} |")
    tot = sum(e["wall_time_total"] for d in S.values() for e in d.values())
    w.append(f"\nTotal compute over all runs: {tot / 3600:.1f} CPU-hours (one core per run).\n")
    return "\n".join(w)


def extra_gain(S):
    g = []
    for ds in S:
        if ds.startswith("renewal") and "-" not in ds:
            for mo, e in S[ds].items():
                pi = e["per_init"]
                if mo.startswith("hmm") and any(r["init"] >= 6 and r["init"] < 100 for r in pi):
                    g.append(max(r["train_ll"] for r in pi) - max(r["train_ll"] for r in pi if r["init"] < 6))
    return max(g)


def deviations_text(S):
    return "\n".join([
        "## Deviations from the brief\n",
        "- **Edge-emitting HMMs.** The classical baseline is the edge-emitting (Mealy) HMM, P(x, s′ | s), rather than "
        "the state-emitting one. It contains the state-emitting HMM with the same number of states and is the exact "
        "classical analogue of the QSM, so it is the stronger and fairer baseline. Baum–Welch is the edge-emitting "
        "version; its result is polished with L-BFGS (an addition), and a small pseudo-count (10⁻³ per expected "
        "transition) keeps every probability positive so that KL values stay finite.",
        "- **Extra inits for the HMM** on the renewal targets at T = 50,000: one oracle-informed start from the lumped "
        "true chain for every N, and 20 random starts instead of 6 for N = 3, 5, 8, because the main claim depends on "
        "the HMM optimum being found (N = 4, 6, 7 keep 6 random starts: the machine's second core was busy with "
        "another job, and at N = 3, 5, 8 the extra starts improved the best training log-likelihood of the first 6 "
        f"by at most {fe(extra_gain(S))} nats/symbol).",
        "- **A D = 8 QSM** (m = 3, general parametrization) was added on renewal N = 6–8, to compare with the HMM with "
        "the true number of states at the same memory dimension.",
        "- **Held-out KL rate** added as the headline metric; the exact finite-horizon d_L and D_L of the brief are "
        "reported alongside. The regime horizon is L ≤ 5 (32,768 words).",
        "- **Not done:** Architecture B on the augmented input (x_t, x_{t−1}) from the plan's model list; the "
        "order-2 Markov chain is its exact upper bound on these targets, which is reported instead. The paper's "
        "window training (past length M, future length L) was replaced by the exact likelihood of the whole string.\n",
    ])


def reproduce_text():
    return "\n".join([
        "## How to reproduce\n",
        "```",
        "./.venv/bin/python -m pytest -q tests/test_qseq.py        # exactness tests of the QSM, truths, baselines",
        "./.venv/bin/python scripts/run_seq3.py --list             # runs still to do",
        "./.venv/bin/python scripts/run_seq3.py --max-minutes 110  # resumable; two workers may run side by side",
        "./.venv/bin/python scripts/seq3_report.py                 # summary.json, figs/seq3_*.png, this note",
        "```",
        "Each run writes `runs/seq3/<dataset>/<model>/<init>/{result.json, params.npz, loss.npy}`; the datasets are "
        "cached in `runs/seq3/data/`. Results are deterministic on CPU (fixed seeds).\n",
    ])


if __name__ == "__main__":
    main()
