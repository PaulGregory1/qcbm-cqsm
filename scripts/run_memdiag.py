"""Memory diagnostics of the sequence 4 / 5 fits on msar and smsar16, without retraining any circuit (archb.memdiag).

For each dataset, the selected fit (best training likelihood) of: the truth; the true family (MS-AR); the factored
IOHMM with 2 states; Architecture B anc2 (no memory: the retrained reset control); the C-QSM m = 1 and m = 2 with
2 ancillas (coherent); the C-QSM m = 1 + 2 anc trained with its memory dephased (sequence 5 A2, gamma = 1: the
retrained dephased control); the classical 2-state (and 4-state on smsar16) switch of anc2 kernels (sequence 5 A1).

  info      oracle G_k (k = 0 .. 128), G_0 = I(Y; past | X_t), each model's G_theta and R_theta = 1 - G_theta / G_0,
            reset-and-replay loss L_k (the model restarted k steps back), on the held-out path, targets t >= 129
  frozen    on the coherent C-QSMs, theta frozen: dephasing lambda in {0, .1, .25, .5, .75, 1} in the computational
            basis and in the basis optimized on the training paths, and a reset to the stationary mean memory state;
            held-out gaps to the coherent model (positive = worse) with paired SE over 20 segments; the retrained
            controls on the same scale
  classical exact classical reduction T[x, y] of the dephased circuits against the circuit with gamma = 1
  posterior regime posterior q*_t of the truth's filter; least-squares probes on x_t, on x_t + memory and on the
            memory alone (fitted on the first half of the held-out path, scored on the second); memory-state swap
            between histories that end at the same x with contrasting posteriors; commutators of the C-QSM states
  encoded   (--encoded: added to an existing summary.json) information held by each memory on the held-out path:
            chi_{H:M|X} (Holevo quantity of the memory states given x_t, <= ln D; Shannon for classical beliefs) on
            each half, and the predictive information I(Y : M | X) under the model's own law, its version scored on
            the true next values, and a within-x shuffle of the memories as bias control (archb.memdiag);
            G_0 - G_theta = G_0 R_theta is the variational lower bound on I(Y : M | X) under the truth
-> runs/memdiag/<dataset>/summary.json, figs/memdiag_<dataset>_{info,frozen,posterior}.png

Usage: python scripts/run_memdiag.py [--datasets msar smsar16] [--encoded]
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_seq4 as RS  # noqa: E402

from archb import cqsm, hmm, iolin, memdiag as MD, regimes as R, switch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIGS, SEQ5 = ROOT / "runs" / "memdiag", ROOT / "figs", ROOT / "runs" / "seq5"
KS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
T0 = KS[-1] + 1
LAMS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
DPI = 170


def best(d):
    p = max(d.glob("*/result.json"), key=lambda q: json.loads(q.read_text())["train_ll"]).parent
    with np.load(p / "params.npz") as z:
        prm = {k: z[k] for k in z.files}
    return prm, p.name, json.loads((p / "result.json").read_text())


def models(ds):
    """name -> dict(lin, role, init, n_params, filter_dim[, theta, m])."""
    s = RS.DATASETS[ds]
    out = {"truth": {"lin": iolin.IOLin.from_iohmm(R.truth_T(s["kind"], s["N"])), "role": "truth"}}
    for mo, role in (("family-msar", "structured filter (MS-AR fit)"), ("iohmm-fact-2", "classical memory (IOHMM)")):
        prm, i, r = best(RS.OUT / ds / mo)
        out[mo] = {"lin": iolin.IOLin.from_iohmm(prm["T"]), "role": role, "init": i}
    prm, i, r = best(RS.OUT / ds / "archb-anc2")
    out["archb-anc2"] = {"lin": cqsm.io(prm["theta"], 0, 2), "role": "reset, retrained (no memory)", "init": i,
                         "n_params": r.get("n_params")}
    for m in (1, 2):
        prm, i, r = best(RS.OUT / ds / f"cqsm-m{m}-anc2")
        out[f"cqsm-m{m}-anc2"] = {"lin": cqsm.io(prm["theta"], m, 2), "role": "coherent", "init": i,
                                  "theta": prm["theta"], "m": m, "n_params": r.get("n_params")}
    prm, i, r = best(SEQ5 / "a2" / ds / "cqsm-m1-anc2-g1")
    out["cqsm-m1-anc2-g1"] = {"lin": cqsm.io(prm["theta"], 1, 2, gamma=1.0), "role": "dephased, retrained",
                              "init": i, "theta": prm["theta"], "m": 1, "n_params": r.get("n_params")}
    for S in (2, 4):
        d = SEQ5 / "a1" / ds / f"sw-anc2-S{S}"
        if d.exists():
            prm, i, r = best(d)
            out[f"sw-anc2-S{S}"] = {"lin": switch.io(prm["theta"], S, "anc2"),
                                    "role": f"classical memory (switch, {S} states)", "init": i,
                                    "n_params": r.get("n_params")}
    for k, v in out.items():
        v["filter_dim"] = v["lin"].d
    return out


def gap(lp_ref, lp):
    """Mean of lp_ref - lp (positive: the second model is worse) with its SE over 20 segments."""
    return hmm.seg_kl_rate(lp_ref, lp)


def run(ds):
    t_start = time.time()
    s, d = RS.DATASETS[ds], RS.load(ds)
    xs, X = d["test"], d["X"]
    regime = R.regime_of(s["kind"], s["N"])
    mods = models(ds)
    truth = mods["truth"]["lin"]
    lp_true = truth.logprobs(xs)
    res = {"dataset": ds, "eval_from": T0, "n_test": len(xs), "ks": list(KS), "lams": list(LAMS), "models": {}}

    # ---------------------------------------------------------------------------------- information and replay
    G = {k: MD.seg_mean(lp_true[T0:] - MD.window_logprobs(truth, xs, k, T0)) for k in KS}
    res["oracle_G"] = {str(k): v for k, v in G.items()}
    g0_seg = (lp_true[T0:] - MD.window_logprobs(truth, xs, 0, T0))
    lps = {}
    for name, mo in mods.items():
        lin = mo["lin"]
        lp = lin.logprobs(xs)
        lps[name] = lp
        dth = lp_true[T0:] - lp[T0:]
        n = len(dth) // 20 * 20
        rs = 1 - dth[:n].reshape(20, -1).mean(1) / g0_seg[:n].reshape(20, -1).mean(1)
        r = {"role": mo["role"], "init": mo.get("init"), "n_params": mo.get("n_params"),
             "filter_dim": mo["filter_dim"], "train_ll": lin.path_loglik(X),
             "heldout_kl_rate": gap(lp_true, lp), "G_theta": MD.seg_mean(dth),
             "R_theta": [float(1 - dth.mean() / g0_seg.mean()), float(rs.std(ddof=1) / np.sqrt(20))]}
        if name != "truth":
            r["replay_L"] = {str(k): MD.seg_mean(lp_true[T0:] - MD.window_logprobs(lin, xs, k, T0)) for k in KS}
        res["models"][name] = r
        print(f"  {ds} {name:18s} KL {r['heldout_kl_rate'][0]:.3e}  R {r['R_theta'][0]:+.3f}"
              f"  [{time.time() - t_start:.0f} s]", flush=True)
    print(f"  {ds} G_0 {G[0][0]:.4e} ± {G[0][1]:.1e}, G_128 {G[128][0]:.2e}", flush=True)

    # ---------------------------------------------------------------------------------- frozen interventions
    res["frozen"] = {}
    for name in ("cqsm-m1-anc2", "cqsm-m2-anc2"):
        mo = mods[name]
        m, D = mo["m"], 2 ** mo["m"]
        M0, f = cqsm.transfer(jnp.asarray(mo["theta"]), m, 2), np.asarray(cqsm.f_vec(m))
        lp_q = lps[name]
        fr = {"computational": {}, "optimized": {}}
        for lam in LAMS:
            lin = iolin.IOLin(np.asarray(MD.dephased(M0, D, lam)), f)
            fr["computational"][str(lam)] = {"gap": gap(lp_q, lin.logprobs(xs)), "train_ll": lin.path_loglik(X)}
        U, ll_u = MD.optimize_basis(M0, f, D, X)
        fr["optimized_train_ll"], fr["U"] = ll_u, [[str(z) for z in row] for row in U]
        for lam in LAMS:
            lin = iolin.IOLin(np.asarray(MD.dephased(M0, D, lam, jnp.asarray(U))), f)
            fr["optimized"][str(lam)] = {"gap": gap(lp_q, lin.logprobs(xs)), "train_ll": lin.path_loglik(X)}
        u = mo["lin"].u.sum(0)
        lin_r = iolin.IOLin(MD.reset(np.asarray(M0), f, u / (u @ f)), f)
        fr["reset"] = {"gap": gap(lp_q, lin_r.logprobs(xs)), "train_ll": lin_r.path_loglik(X)}
        fr["retrained"] = {o: gap(lp_q, lps[o]) for o in mods if o not in ("truth", name)}
        fr["train_ll"] = res["models"][name]["train_ll"]
        res["frozen"][name] = fr
        print(f"  {ds} {name} frozen deph. gap {fr['computational']['1.0']['gap'][0]:+.3e}, optimized basis "
              f"{fr['optimized']['1.0']['gap'][0]:+.3e}, reset {fr['reset']['gap'][0]:+.3e}"
              f"  [{time.time() - t_start:.0f} s]", flush=True)

    # ---------------------------------------------------------------------------------- exact classical reduction
    res["classical_reduction"] = {}
    for name in ("cqsm-m1-anc2-g1", "cqsm-m1-anc2", "cqsm-m2-anc2"):
        mo = mods[name]
        K = np.asarray(cqsm.kraus(jnp.asarray(mo["theta"]), mo["m"], 2))
        cl = MD.classical_from_kraus(K)
        deph = mo["lin"] if name.endswith("-g1") else cqsm.io(mo["theta"], mo["m"], 2, gamma=1.0)
        res["classical_reduction"][name] = {
            "min_entry": float(cl.M.min()), "row_sum_err": float(np.abs(cl.M.sum((1, 3)) - 1).max()),
            "max_abs_dlogp_heldout": float(np.abs(cl.logprobs(xs) - deph.logprobs(xs)).max()),
            "max_abs_dlogp_words3": float(np.abs(MD.word_logprobs(cl, 3) - MD.word_logprobs(deph, 3)).max())}

    # ---------------------------------------------------------------------------------- posterior and memory
    q = MD.regime_posterior(truth, xs, regime)
    st = regime[d["test_h"]]
    h = len(xs) // 2
    tr, te = slice(0, h), slice(h, None)
    Fx = MD.features(xs)
    res["posterior"] = {"exact_filter": MD.posterior_scores(q[te], q[te], st[te]),
                        "x_only": MD.posterior_scores(MD.fit_probe(Fx[tr], q[tr])(Fx[te]), q[te], st[te]),
                        "models": {}}
    states = {}
    probe_q = {}
    for name, mo in mods.items():
        lin = mo["lin"]
        if lin.d == 1:
            continue
        S_ = lin.filtered(xs)
        states[name] = S_
        F = MD.features(xs, S_) if lin.d <= 16 else np.concatenate([Fx, np.real(S_)], 1)
        pr = MD.fit_probe(F[tr], q[tr])
        probe_q[name] = pr(F)
        Fm = F[:, Fx.shape[1]:]
        r = {"x_mem": MD.posterior_scores(probe_q[name][te], q[te], st[te]),
             "mem_only": MD.posterior_scores(MD.fit_probe(Fm[tr], q[tr])(Fm[te]), q[te], st[te])}
        if name != "truth":
            r["swap"] = MD.swap_effect(lin, S_, truth, states["truth"], xs, q)
        if name.startswith("cqsm") and not name.endswith("-g1"):
            r["commutators"] = MD.commutators(MD.rhos(S_, mo["m"]), xs)
        res["posterior"]["models"][name] = r
    print(f"  {ds} posterior probes done  [{time.time() - t_start:.0f} s]", flush=True)

    # ---------------------------------------------------------------------------------- timeline for the figure
    sw = np.flatnonzero(np.diff(st[h + 1000:]) != 0) + h + 1000
    t1 = int(sw[0]) - 100
    win = np.arange(t1, t1 + 500)
    tl = {"t": win.tolist(), "q_star": q[win].tolist(), "regime": st[win].tolist(), "x": R.Z[xs[win]].tolist(),
          "probe": {k: probe_q[k][win].tolist() for k in ("cqsm-m1-anc2", "sw-anc2-S2") if k in probe_q}, "sd": {}}
    for name in ("truth", "cqsm-m1-anc2", "sw-anc2-S2", "archb-anc2"):
        lin = mods[name]["lin"]
        Sw = states[name][win] if name in states else np.ones((len(win), 1))
        P = MD.predictive(lin, Sw, xs[win])
        mu = P @ R.Z
        tl["sd"][name] = np.sqrt(np.clip(P @ R.Z**2 - mu**2, 0, None)).tolist()
    res["timeline"] = tl
    res["wall_seconds"] = time.time() - t_start
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    (OUT / ds / "summary.json").write_text(json.dumps(res, indent=1))
    figures(res)
    return res


def encoded_info(mods, xs):
    """name -> chi on each half of xs, ln D and the predictive information of its memory (models with a memory)."""
    out, h = {}, len(xs) // 2
    for name, mo in mods.items():
        lin = mo["lin"]
        if lin.d == 1:
            continue
        S_ = lin.filtered(xs)
        Rm = MD.memory_rhos(S_, mo.get("m"))
        D = Rm.shape[1]
        out[name] = {"chi_first_half": MD.holevo(Rm[:h], xs[:h]), "chi_second_half": MD.holevo(Rm[h:], xs[h:]),
                     "ln_D": float(np.log(D)), "memory_dim": int(D),
                     "predictive": MD.predictive_info(MD.predictive(lin, S_, xs), xs)}
    return out


def encoded(ds):
    p = OUT / ds / "summary.json"
    res = json.loads(p.read_text())
    res["encoded"] = encoded_info(models(ds), RS.load(ds)["test"])
    for name, r in res["encoded"].items():
        print(f"  {ds} {name:18s} chi {r['chi_first_half']:.4f} / {r['chi_second_half']:.4f} (ln D {r['ln_D']:.3f})"
              f"  I_model {r['predictive']['model']:.4f}  I_true {r['predictive']['true'][0]:+.4f}"
              f"  null {r['predictive']['null'][0]:+.4f}", flush=True)
    p.write_text(json.dumps(res, indent=1))


# -------------------------------------------------------------------------------------------------------- figures
LABEL = {"truth": "truth", "family-msar": "MS-AR fit", "iohmm-fact-2": "IOHMM S = 2", "archb-anc2": "no memory (anc2)",
         "cqsm-m1-anc2": "C-QSM m = 1", "cqsm-m2-anc2": "C-QSM m = 2", "cqsm-m1-anc2-g1": "C-QSM m = 1, dephased (retrained)",
         "sw-anc2-S2": "classical switch S = 2", "sw-anc2-S4": "classical switch S = 4"}
COL = {"truth": OI["black"], "family-msar": OI["green"], "iohmm-fact-2": OI["yellow"], "archb-anc2": OI["purple"],
       "cqsm-m1-anc2": OI["vermillion"], "cqsm-m2-anc2": OI["orange"], "cqsm-m1-anc2-g1": OI["sky"],
       "sw-anc2-S2": OI["blue"], "sw-anc2-S4": OI["blue"]}


def figures(res):
    ds, ks = res["dataset"], np.asarray(res["ks"])
    FIGS.mkdir(exist_ok=True)

    # information horizon: oracle G_k and the models' reset-and-replay losses
    fig, ax = plt.subplots(figsize=(7, 4.2))
    g = np.array([res["oracle_G"][str(k)] for k in ks])
    ax.errorbar(ks + 1, g[:, 0], yerr=2 * g[:, 1], color=OI["black"], marker="o", lw=2, capsize=2,
                label="oracle $G_k$ (truth restarted $k$ steps back)")
    for name, r in res["models"].items():
        if "replay_L" not in r:
            continue
        L = np.array([r["replay_L"][str(k)] for k in ks])
        ls = "--" if name.startswith("sw-anc2-S4") else "-"
        ax.plot(ks + 1, L[:, 0], ls, marker=".", color=COL[name], label=f"{LABEL[name]} (replay)")
        ax.axhline(r["G_theta"][0], color=COL[name], lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1e-4)  # the oracle reaches exactly 0
    ax.set_ylim(bottom=0)
    ax.set_xlabel("window $k + 1$ (observations)")
    ax.set_ylabel("loss vs full-history truth (nats/step)")
    ax.set_title(f"{ds}: predictive information horizon (dotted: long warm-up)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / f"memdiag_{ds}_info.png", dpi=DPI)
    plt.close(fig)

    # frozen and retrained interventions
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axs, ("cqsm-m1-anc2", "cqsm-m2-anc2")):
        fr = res["frozen"][name]
        for basis, c, mk in (("computational", OI["vermillion"], "o"), ("optimized", OI["blue"], "s")):
            v = np.array([fr[basis][str(l)]["gap"] for l in res["lams"]])
            ax.errorbar(res["lams"], v[:, 0], yerr=2 * v[:, 1], marker=mk, color=c, capsize=2,
                        label=f"frozen dephasing, {basis} basis")
        ax.errorbar([1.05], [fr["reset"]["gap"][0]], yerr=[2 * fr["reset"]["gap"][1]], marker="x", color=OI["black"],
                    capsize=2, ls="none", label="frozen reset")
        for o, mk in (("cqsm-m1-anc2-g1", "D"), ("sw-anc2-S2", "^"), ("archb-anc2", "v"), ("iohmm-fact-2", "<")):
            if o in fr["retrained"]:
                gv = fr["retrained"][o]
                ax.errorbar([1.12], [gv[0]], yerr=[2 * gv[1]], marker=mk, color=COL[o], ls="none", capsize=2,
                            label=f"{LABEL[o]}, retrained")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("dephasing strength λ")
        ax.set_title(f"{ds}: {LABEL[name]} (positive = worse than coherent)")
    axs[0].set_ylabel("held-out loss − coherent (nats/step)")
    axs[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / f"memdiag_{ds}_frozen.png", dpi=DPI)
    plt.close(fig)

    # posterior timeline, predictive volatility, reliability
    tl = res["timeline"]
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 3)
    a0 = fig.add_subplot(gs[0, :2])
    t = np.asarray(tl["t"]) - tl["t"][0]
    a0.fill_between(t, 0, tl["regime"], step="mid", color="0.88", label="true regime (turbulent = 1)")
    a0.plot(t, tl["q_star"], color=OI["black"], lw=1.5, label="exact filter $q^*_t$")
    for k, v in tl["probe"].items():
        a0.plot(t, v, color=COL[k], lw=1, label=f"probe on {LABEL[k]}")
    a0.set_ylabel("P(turbulent)")
    a0.legend(fontsize=7, ncol=2, loc="upper right")
    a0.set_title(f"{ds}: held-out window")
    a1 = fig.add_subplot(gs[1, :2], sharex=a0)
    for k, v in tl["sd"].items():
        a1.plot(t, v, color=COL[k], lw=1.6 if k == "truth" else 1, label=LABEL[k])
    a1.set_ylabel("predictive sd of $x_{t+1}$")
    a1.set_xlabel("step")
    a1.legend(fontsize=7, ncol=2)
    a2 = fig.add_subplot(gs[:, 2])
    a2.plot([0, 1], [0, 1], color="grey", lw=0.6)
    for k in ("cqsm-m1-anc2", "cqsm-m2-anc2", "sw-anc2-S2", "truth"):
        rel = np.asarray(res["posterior"]["models"][k]["x_mem"]["reliability"])
        a2.plot(rel[:, 0], rel[:, 1], "o-", ms=3, color=COL[k], label=f"probe on {LABEL[k]}")
    rel = np.asarray(res["posterior"]["x_only"]["reliability"])
    a2.plot(rel[:, 0], rel[:, 1], "s--", ms=3, color="0.5", label="probe on $x_t$ only")
    a2.set_xlabel("predicted P(turbulent)")
    a2.set_ylabel("realized frequency")
    a2.set_title("reliability (second half)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"memdiag_{ds}_posterior.png", dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["msar", "smsar16"])
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--encoded", action="store_true", help="add the encoded information to existing summaries")
    a = ap.parse_args()
    for ds in a.datasets:
        if a.encoded:
            encoded(ds)
        elif a.figures_only:
            figures(json.loads((OUT / ds / "summary.json").read_text()))
        else:
            run(ds)


if __name__ == "__main__":
    main()
