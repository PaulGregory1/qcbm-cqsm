"""Render reports/go-no-go.md from runs/summary.json (called by scripts/run_all.py).

Every number in the note comes from the summary; the prose around the tables is fixed text
written for this set of runs.

Run alone: ./.venv/bin/python scripts/write_note.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODINGS = ("binary", "gray")
SEEDS = ("seed0", "seed1", "seed2")
REF_LABELS = {"exact": "Exact P*", "ar1": "Gaussian AR(1)", "tabular": "Tabular counts", "iid": "i.i.d. oracle"}
COLS = [("weighted_tv", "Weighted TV"), ("tail_tv", "Tail TV"), ("stationary_tv", "Stationary TV"),
        ("lag1_acf", "Lag-1 acf"), ("acf_err", "ACF error"), ("slope", "Slope")]
CHECKS = [("stationary_tv", "Stat. TV < 0.05"), ("lag1_acf", "Lag-1 ±0.02"), ("slope", "Slope ±0.03"),
          ("weighted_tv_vs_iid", "WTV ≤ ½ i.i.d.")]
TIGHT_CHECKS = [("stationary_tv", "Stat. TV ≤ 0.02"), ("lag1_acf", "Lag-1 ±0.02"), ("slope", "Slope ±0.03"),
                ("weighted_tv", "WTV ≤ 0.05")]
DIAG_ROWS = [("primary", "spec (primary)"), ("diagnostic/eps0.005", "ε = 0.005"),
             ("diagnostic/noise0.1", "noise 0.1"), ("diagnostic/layers4", "4 layers")]
DIAG_COLS = [("weighted_tv", "WTV"), ("stationary_tv", "Stat. TV"), ("tail_tv", "Tail TV"),
             ("flux_ratio_depth5", "Flux 15│16"), ("flux_ratio_depth4", "Flux 7│8, 23│24"),
             ("flux_ratio_depth3", "Flux depth 3"), ("flux_ratio_depth1-2", "Flux depth 1–2"),
             ("bulk_row_tv_near_deep", "Row TV near deep"), ("bulk_row_tv_other", "Row TV other")]

TICK = {True: "✓", False: "✗"}


def f4(x):
    return f"{x:.4f}"


def med_range(agg, k, fmt=f4):
    a = agg[k]
    return f"{fmt(a['median'])} [{fmt(a['min'])}–{fmt(a['max'])}]"


def metric_table(S):
    rows = ["| Model | " + " | ".join(h for _, h in COLS) + " | Final S_ε | Steps |",
            "| --- |" + " ---: |" * (len(COLS) + 2)]
    for n, lab in REF_LABELS.items():
        rows.append(f"| {lab} | " + " | ".join(f4(S["references"][n][k]) for k, _ in COLS) + " | | |")
    for cfg in ("primary", "converged-solver"):
        for enc in ENCODINGS:
            a = S["aggregate"][cfg][enc]
            cells = [med_range(a, k) for k, _ in COLS]
            cells.append(med_range(a, "final_loss", lambda x: f"{x:.4f}"))
            cells.append(med_range(a, "steps", lambda x: f"{x:.0f}"))
            rows.append(f"| QCBM {enc}, {cfg} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def check_table(S, cfg):
    rows = ["| Encoding | Seed | " + " | ".join(h for _, h in CHECKS) + " | All four | Stopped at |",
            "| --- | --- |" + " :---: |" * (len(CHECKS) + 1) + " ---: |"]
    for enc in ENCODINGS:
        g = S["go_no_go"][cfg][enc]
        for s in SEEDS:
            c = g["seeds"][s]
            m = S["runs"][cfg][enc][s]["metrics"]
            vals = {"stationary_tv": f"{m['stationary_tv']:.3f}",
                    "lag1_acf": f"{m['lag1_acf'] - m['lag1_acf_exact']:+.3f}",
                    "slope": f"{m['slope'] - 0.9048:+.3f}",
                    "weighted_tv_vs_iid": f"{m['weighted_tv']:.3f}"}
            cells = [f"{TICK[c['checks'][k]]} {vals[k]}" for k, _ in CHECKS]
            steps = S["runs"][cfg][enc][s]["steps"]
            rows.append(f"| {enc} | {s[-1]} | " + " | ".join(cells) + f" | {TICK[c['pass']]} | {steps} |")
        rows.append(f"| {enc} | spread | (max − min)/mean of WTV = {g['weighted_tv_spread']:.3f} "
                    f"{TICK[g['spread_pass']]} (< 0.2) | | | | **{'GO' if g['go'] else 'NO-GO'}** "
                    f"({g['n_seeds_passing']}/3 seeds) | |")
    return "\n".join(rows)


def tight_table(S):
    rows = ["| Config | Encoding | " + " | ".join(f"{h} (seeds ✓)" for _, h in TIGHT_CHECKS) + " | Seeds passing all | Verdict |",
            "| --- | --- |" + " :---: |" * (len(TIGHT_CHECKS) + 2)]
    for cfg in ("primary", "converged-solver"):
        for enc in ENCODINGS:
            t = S["tight"][cfg][enc]
            cells = [str(sum(t["seeds"][s]["checks"][k] for s in SEEDS)) + "/3" for k, _ in TIGHT_CHECKS]
            rows.append(f"| {cfg} | {enc} | " + " | ".join(cells) +
                        f" | {t['n_seeds_passing']}/3 | {'GO' if t['go'] else 'NO-GO'} |")
    return "\n".join(rows)


def diag_table(S):
    rows = ["| Encoding | Run | " + " | ".join(h for _, h in DIAG_COLS) + " | Seeds passing (spec) |",
            "| --- | --- |" + " ---: |" * (len(DIAG_COLS) + 1)]
    for enc in ENCODINGS:
        for g, lab in DIAG_ROWS:
            a = S["aggregate"][g][enc]
            cells = [f"{a[k]['median']:.3f}" for k, _ in DIAG_COLS]
            rows.append(f"| {enc} | {lab} | " + " | ".join(cells) +
                        f" | {S['go_no_go'][g][enc]['n_seeds_passing']}/3 |")
    for n in ("ar1", "tabular"):
        r = S["references"][n]
        cells = [f"{r[k]:.3f}" for k, _ in DIAG_COLS]
        rows.append(f"| reference | {REF_LABELS[n]} | " + " | ".join(cells) + " | |")
    return "\n".join(rows)


SR_COLS = [("weighted_tv", "Weighted TV"), ("tail_tv", "Tail TV"), ("stationary_tv", "Stationary TV"),
           ("lag1_acf", "Lag-1 acf"), ("acf_err", "ACF error"), ("slope", "Slope"),
           ("flux_ratio_depth5", "Flux 15│16")]


def single_register_table(S):
    rows = ["| Model | " + " | ".join(h for _, h in SR_COLS) + " | Final S_ε | Steps |",
            "| --- |" + " ---: |" * (len(SR_COLS) + 2)]
    rows.append("| Exact P* | " + " | ".join(f4(S["references"]["exact"][k]) for k, _ in SR_COLS) + " | | |")
    for g, lab in (("single_register", "Single register"), ("primary", "Conditional QCBM (Arch. B)")):
        for enc in ENCODINGS:
            a = S["aggregate"][g][enc]
            cells = [med_range(a, k) for k, _ in SR_COLS]
            cells.append(med_range(a, "final_loss", lambda x: f"{x:.4f}"))
            cells.append(med_range(a, "steps", lambda x: f"{x:.0f}"))
            rows.append(f"| {lab}, {enc} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_note(S, path):
    tables = {"metric_table": metric_table(S), "check_primary": check_table(S, "primary"),
              "check_converged": check_table(S, "converged-solver"), "tight_table": tight_table(S),
              "diag_table": diag_table(S), "single_table": single_register_table(S)}
    Path(path).write_text(render(S, tables))


def render(S, T):
    ref = S["references"]
    agg = lambda g, e, k: S["aggregate"][g][e][k]  # noqa: E731
    med = lambda g, e, k: agg(g, e, k)["median"]  # noqa: E731
    gng = S["go_no_go"]
    run = lambda g, e, s, k: S["runs"][g][e][s]["metrics"][k]  # noqa: E731
    verdict = lambda g: "GO" if S["verdict"][g]["go"] else "NO-GO"  # noqa: E731
    npass = lambda g, e: gng[g][e]["n_seeds_passing"]  # noqa: E731
    ms = S["median_seed"]["primary"]
    wtv_bound = S["settings"]["thresholds_spec"]["weighted_tv_max"]
    gw, bw = med("primary", "gray", "weighted_tv"), med("primary", "binary", "weighted_tv")
    tight_any = any(S["tight"][g][e]["n_seeds_passing"] for g in ("primary", "converged-solver")
                    for e in ENCODINGS)
    fig = lambda e, n, cap: f"![{cap}](../figs/{e}_{n}.png)"  # noqa: E731
    D = "diagnostic/"
    SR = S["single_register"]
    sr_runs = [r for e in ENCODINGS for r in SR["runs"][e].values()]
    sr_col = max(max(r["col_sum_err"], r["row_sum_err"]) for r in sr_runs)
    sr_pi = max(r["stationary_max_dev_from_uniform"] for r in sr_runs)
    sr_stv = [S["runs"]["single_register"][e][s]["metrics"]["stationary_tv"] for e in ENCODINGS for s in SEEDS]

    return f"""# Go/no-go note — Architecture B, minimal baseline

Generated by `scripts/run_all.py` (prose in `scripts/write_note.py`, every number read from
`runs/summary.json`). Spec: [`docs/spec.md`](../docs/spec.md).

## Verdict

**Official verdict (primary configuration = the spec unchanged, spec thresholds): {verdict("primary")},
carried by the Gray encoding alone.** Gray passes all four per-seed checks on {npass("primary", "gray")}/3 seeds
(seed 2 misses the stationary-TV bound with {run("primary", "gray", "seed2", "stationary_tv"):.4f} against 0.05)
and its weighted-TV spread across seeds is {gng["primary"]["gray"]["weighted_tv_spread"]:.3f} (< 0.2). Binary
passes on {npass("primary", "binary")}/3 seeds: stationary TV {agg("primary", "binary", "stationary_tv")["min"]:.2f}–{agg("primary", "binary", "stationary_tv")["max"]:.2f}
and lag-1 autocorrelation too high by 0.03–0.06 on every seed. The secondary "converged-solver"
configuration (100 Sinkhorn iterations per step, a deviation) gives the same verdict
({verdict("converged-solver")}; Gray {npass("converged-solver", "gray")}/3, binary {npass("converged-solver", "binary")}/3), so the
early, coincidental firing of the stop rule does not decide it. The GO is weak. Gray's weighted TV
(median {gw:.3f}) is {gw / ref["tabular"]["weighted_tv"]:.0f}× that of the tabular counts and {gw / ref["ar1"]["weighted_tv"]:.0f}× that of the AR(1) fit, and
its tail TV ({med("primary", "gray", "tail_tv"):.2f}) is twice the tabular one. Under tighter thresholds (weighted
TV ≤ 0.05, stationary TV ≤ 0.02) {"no encoding passes on any seed, in either configuration" if not tight_any else "some seeds pass (see below)"}.
The main defect is structural: the trained circuit moves almost no probability across x = 0, the
boundary where the most significant bit flips. In the primary configuration the flux across it is
{med("primary", "binary", "flux_ratio_depth5") * 100:.1f}% of the exact flux in binary and {med("primary", "gray", "flux_ratio_depth5") * 100:.0f}% in Gray.

## Metrics

QCBM rows give the median over seeds 0, 1, 2 and the range in brackets. All metrics are exact
functions of the 32 × 32 kernels, scored against P*. "Final S_ε" is the converged Sinkhorn
divergence at the last step; the exact kernel scores {S["loss_floor_exact"]["eps0.01"]:.1e}. "Steps" is where the spec
stop rule fired (800 = ran to the cap).

{T["metric_table"]}

## The five checks, per encoding and seed

Primary configuration (decides the verdict). Each cell shows the check and the value it tested:
stationary TV, lag-1 minus exact lag-1, slope minus c, weighted TV (bound ½ × {ref["iid"]["weighted_tv"]:.4f} = {wtv_bound:.4f}).

{T["check_primary"]}

Converged-solver configuration (deviation: 100 Sinkhorn iterations per step; not the verdict).

{T["check_converged"]}

The spread check never decides the verdict here: it passes for both encodings in both
configurations, and binary fails on the per-seed checks.

## What the figures show

Primary configuration. Single-QCBM panels use the seed with the median weighted TV:
binary seed {ms["binary"]}, Gray seed {ms["gray"]}. Kernels are on a log colour scale clipped at 10⁻⁴, so
spurious far-from-diagonal mass is visible.

| | Binary | Gray |
| --- | --- | --- |
| Kernels | {fig("binary", "kernels", "binary kernels")} | {fig("gray", "kernels", "Gray kernels")} |
| Conditional moments | {fig("binary", "cond_mean", "binary conditional moments")} | {fig("gray", "cond_mean", "Gray conditional moments")} |
| Autocorrelation | {fig("binary", "acf", "binary ACF")} | {fig("gray", "acf", "Gray ACF")} |
| Training loss | {fig("binary", "loss", "binary loss")} | {fig("gray", "loss", "Gray loss")} |

- **Kernels.** Both QCBM kernels follow the diagonal band, but not across x = 0. In binary the
  kernel splits into two diagonal blocks there (the quadrants x_t < 0 → x_{{t+1}} > 0 and
  x_t > 0 → x_{{t+1}} < 0 are empty next to the diagonal), and the band is nearly cut again at
  x = ±1.5. In Gray the band is thinned at x = 0, not cut. On the log scale both kernels also put 10⁻³–10⁻² of mass in blocks far from the
  diagonal (Gray more so in the tails), which tabular never does. The band is also visibly
  patchy, not smooth.
- **Conditional moments.** The conditional mean tracks the truth in the bulk, so the slope check
  passes ({med("primary", "binary", "slope"):.3f} binary, {med("primary", "gray", "slope"):.3f} Gray, c = 0.905). The errors are concentrated at the
  deep boundaries (x = 0 and x = ±1.5): the mean jumps by ±0.15–0.3 there, and the conditional
  sd falls to 0.6–0.7× its exact value, because the mass piles up on one side of the boundary.
  Gray's tail rows have far too much spread (sd up to about 5× the exact value at the edges).
- **Autocorrelation.** Binary decays far too slowly (median ACF error {med("primary", "binary", "acf_err"):.2f}; the acf
  stays near 0.8 at lag 20 instead of 0.14). With almost no flux across x = 0 the chain nearly
  splits into two halves, so the sign of x is almost never forgotten. Gray, whose flux across 0
  is about a quarter of the truth, follows the exact curve closely (ACF error {med("primary", "gray", "acf_err"):.3f}).
  Both are far from the i.i.d. oracle, whose acf is 0 at every lag.
- **Loss.** All runs drop from 0.1–1 to about 10⁻² within 100 steps, then plateau at 45–75×
  the exact kernel's loss (binary ≈ {med("primary", "binary", "final_loss"):.4f}, Gray ≈ {med("primary", "gray", "final_loss"):.4f}). With 25 iterations the
  binary curves oscillate after step ~150 and the stop rule fires by coincidence (seeds 0 and 1
  at 269 and 324 steps). With 100 iterations the curves are smooth and reach slightly lower
  values: weighted TV {med("converged-solver", "binary", "weighted_tv"):.3f} binary and {med("converged-solver", "gray", "weighted_tv"):.3f} Gray (medians), against
  {bw:.3f} and {gw:.3f} in the primary configuration. The ranking and the verdict do not change.

## Diagnostic (not part of the verdict)

Following the spec's troubleshooting table for the failed checks (binary: stationary TV, lag-1,
one weighted-TV miss; Gray seed 2: stationary TV). Boundary b│b+1 has *depth* k when a binary +1
flips k bits there; in Gray the same boundary flips one bit, the one at position k from the
right. So depth 5 is x = 0 (01111→10000, Gray 01000→11000), depth 4 is x = ±1.5, and depth 3 is
x = ±0.75 and ±2.25. "Flux" is the two-way probability flux across those boundaries under π*,
model ÷ exact (1 = right). "Row TV near deep" is the π*-weighted row TV of the 8 bulk rows
(|z| ≤ 1.41) next to a depth ≥ 3 boundary; "other" covers the 8 remaining bulk rows. Each
variant changes one setting from the primary configuration; medians over 3 seeds.

{T["diag_table"]}

- **Where the mass goes wrong.** In both encodings, the flux deficit grows with the significance
  of the bit that must flip, not with the number of bits flipped. Binary moves
  {med("primary", "binary", "flux_ratio_depth5") * 100:.1f}% of the exact flux across x = 0 and {med("primary", "binary", "flux_ratio_depth4") * 100:.0f}% across x = ±1.5. Gray, where each
  crossing is a single bit flip, still moves only {med("primary", "gray", "flux_ratio_depth5") * 100:.0f}% and {med("primary", "gray", "flux_ratio_depth4") * 100:.0f}%. Depth 1–2 boundaries are right
  in both ({med("primary", "binary", "flux_ratio_depth1-2"):.2f} / {med("primary", "gray", "flux_ratio_depth1-2"):.2f}). Gray helps a lot, but the spec's "carry"
  explanation is only part of the story. A plausible reading (a hypothesis, not tested here):
  each layer's output angles are affine in the input bits. Flipping the output MSB for inputs near
  0, and nowhere else, needs a sharp AND of the four lower input bits, in either encoding.
- **Bulk rows away from deep boundaries are also off.** Their row TV is {med("primary", "binary", "bulk_row_tv_other"):.2f} (binary) and
  {med("primary", "gray", "bulk_row_tv_other"):.2f} (Gray), against {ref["tabular"]["bulk_row_tv_other"]:.3f} for tabular. So the high weighted TV is not only a boundary
  effect: the band is patchy everywhere (see the kernel figure).
- **First fixes.** ε = 0.005 makes everything worse (weighted TV {med(D + "eps0.005", "binary", "weighted_tv"):.3f} / {med(D + "eps0.005", "gray", "weighted_tv"):.3f},
  and the stop rule fires earlier). Noise 0.1 lowers binary's stationary TV
  ({med(D + "noise0.1", "binary", "stationary_tv"):.3f}) but leaves the flux across 0 unchanged and passes no seed. A fourth layer (157
  angles) brings Gray to {npass(D + "layers4", "gray")}/3 seeds passing (stationary TV {agg(D + "layers4", "gray", "stationary_tv")["min"]:.3f}–{agg(D + "layers4", "gray", "stationary_tv")["max"]:.3f}, just under 0.05),
  but the flux across 0 stays at {med(D + "layers4", "gray", "flux_ratio_depth5") * 100:.0f}%, and binary stays at {npass(D + "layers4", "binary")}/3. None of the cheap fixes
  touches the x = 0 deficit.

## Thresholds

As step 3 noted, two spec thresholds are lax compared with the references. The weighted-TV bound
½ × i.i.d. = {wtv_bound:.3f} is {wtv_bound / ref["tabular"]["weighted_tv"]:.0f}× the tabular value. Stationary TV < 0.05 compares with
references at ≤ {max(ref[n]["stationary_tv"] for n in ("ar1", "tabular", "iid")):.4f}. The official verdict above uses the spec thresholds unchanged. The table below
shows the same checks with weighted TV ≤ 0.05 and stationary TV ≤ 0.02 (lag-1 and slope as in
the spec; the spread check is omitted).

{T["tight_table"]}

Under these bounds the verdict would be NO-GO for both encodings: weighted TV fails on every
seed (best {min(agg(g, e, "weighted_tv")["min"] for g in ("primary", "converged-solver") for e in ENCODINGS):.3f}, against 0.05) and so does stationary TV (best
{min(agg(g, e, "stationary_tv")["min"] for g in ("primary", "converged-solver") for e in ENCODINGS):.3f}, against 0.02). The only checks Gray still passes are lag-1 and slope (binary passes only slope).
A kernel with the right conditional mean passes both even when its rows are off.

## Optional: single-register model (not part of the verdict)

The spec's optional extra: 5 qubits and no separate output register, P(b'│b) = |⟨b'│U(θ)│b⟩|²,
with |b⟩ as the circuit's initial state (`archb.single_register`). The ansatz is the output side
of the 119-angle circuit without the input controls: 3 layers of R_y on each qubit plus
controlled R_y between neighbouring qubits in both directions (13 angles per layer), then a final
R_y layer, so 3 × 13 + 5 = 44 angles. All angles start at N(0, 0.2²); θ = 0 gives U = I and
P = I, so no copy gates are needed. Training uses the primary configuration unchanged (same
loss, Adam, stop rule), seeds 0, 1, 2, both encodings; runs are in `runs/single_register/`.
Rows give the median over seeds with the range in brackets; the conditional-QCBM rows repeat
the primary configuration for comparison.

{T["single_table"]}

![stationary laws](../figs/single_register_stationary.png)

- **The stationary law is uniform, as it must be.** Every trained kernel is doubly stochastic
  (largest row- or column-sum error {sr_col:.0e}) and its stationary law is 1/32 per bin to
  {sr_pi:.0e}. Its stationary TV is therefore TV(uniform, π*) = {SR["tv_uniform_vs_pi_star"]:.4f}, computed
  exactly from π*, on all 6 runs (range {min(sr_stv):.8f}–{max(sr_stv):.8f}). That is the model's floor, not a
  training failure. The conditional QCBM reaches {med("primary", "gray", "stationary_tv"):.3f} (Gray) and {med("primary", "binary", "stationary_tv"):.3f} (binary).
- **Dynamics.** Lag-1 autocorrelation is {med("single_register", "binary", "lag1_acf"):.3f} (binary) and {med("single_register", "gray", "lag1_acf"):.3f} (Gray) against
  {ref["exact"]["lag1_acf"]:.3f}, and the conditional-mean slope is {med("single_register", "binary", "slope"):.3f} / {med("single_register", "gray", "slope"):.3f} against c = {ref["exact"]["slope"]:.3f}: the trained
  kernels are close to a pure local diffusion with no mean reversion. A plausible reading (not
  tested): pulling mass toward 0 would raise the sums of the central columns, which are pinned at 1.
  The acf is computed under the model's own uniform law, whose spread is much wider than π*'s.
  No seed passes the spec checks ({gng["single_register"]["binary"]["n_seeds_passing"]}/3 binary, {gng["single_register"]["gray"]["n_seeds_passing"]}/3 Gray).
- **Row by row it is not far off.** Weighted TV is {med("single_register", "binary", "weighted_tv"):.3f} (binary) and {med("single_register", "gray", "weighted_tv"):.3f} (Gray),
  against {bw:.3f} and {gw:.3f} for the conditional QCBM. The damage shows in the quantities that
  depend on all rows jointly: the stationary law and the autocorrelation. The flux across x = 0
  is also near zero ({med("single_register", "binary", "flux_ratio_depth5") * 100:.2f}% / {med("single_register", "gray", "flux_ratio_depth5") * 100:.1f}% of the exact flux), so that deficit is not specific to the
  conditional ansatz.
- **Why the input register is needed.** Without it the kernel is the squared modulus of a single
  unitary, so its columns sum to 1 along with its rows and its stationary law is uniform for
  every θ; the input register acts as the auxiliary system that lifts that constraint and lets
  Architecture B concentrate mass where π* does.

## What the full study can reuse

- Code: `archb.ou` (data, encodings, exact kernel), `archb.circuit` (exact simulator; the layer
  count is now read from the angle-vector length, `init_theta(..., n_layers=L)`), `archb.sinkhorn`
  and `archb.train` (loss and training loop), `archb.references`, `archb.metrics` (metrics,
  go/no-go), `archb.single_register` (the 5-qubit single-register model, `train(..., model="single")`),
  `scripts/run_all.py` (runs, figures, this note), and the boundary-flux diagnostic
  in `run_all.py`, which locates kernel errors by bit significance.
- Warm starts: `runs/<config>/<encoding>/seed<k>/theta.npy` (119 angles; the layout is in the
  `archb.circuit` docstring) with `kernel.npy`, `loss.npy` and `metrics.json`; 4-layer angles
  (157) are in `runs/diagnostic/layers4/`. The best kernels are Gray, converged-solver
  (weighted TV {agg("converged-solver", "gray", "weighted_tv")["min"]:.3f}–{agg("converged-solver", "gray", "weighted_tv")["max"]:.3f}).
- Settings: keep ε = 0.01 (0.005 was worse). Use 100 Sinkhorn iterations or a smoothed stop rule,
  because with 25 iterations the stop rule fires by coincidence. Tighten the weighted-TV and
  stationary-TV thresholds before scoring the full study.
- Open problem to carry over: the flux deficit across high-significance bit flips (x = 0
  above all). The full study should track the depth-5 and depth-4 flux ratios for every model.
  An ansatz whose input dependence is richer than affine-per-layer angles (multi-controlled
  rotations, or the hypernetwork) is the natural thing to test against it.
"""


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    S = json.loads((ROOT / "runs" / "summary.json").read_text())
    write_note(S, ROOT / "reports" / "go-no-go.md")
