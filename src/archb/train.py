"""Training loop: Sinkhorn loss S_eps(w(theta), v), envelope gradient, Adam (spec, "Training").

    w_ij(theta) = pi_hat_i P_theta(j | i),   v_ij = N_ij / sum N,   L(theta) = S_eps(w(theta), v)

One step (jitted):
  1. w = w(theta) and its VJP (autodiff through the circuit);
  2. 25 log-domain Sinkhorn updates of the cross potentials (f, g) of OT(w, v) and of the
     symmetric potential p of OT(w, w), warm-started from the previous step (state carries them);
  3. loss = <w, f> + <v, g> - <w, p> - 1/2 OT(v, v)  (dual value at the current potentials;
     OT(v, v) is a constant solved once to convergence);
  4. gradient = VJP(f - p): potentials held fixed (envelope theorem);
  5. Adam update.
Before step 0 the potentials are solved to convergence at theta_0 (eps-scaling, then iterate),
so the warm start begins from exact potentials rather than from zeros.

Stopping: `steps` max, or at step t >= 100 when |L_t - L_{t-100}| / |L_{t-100}| < 1e-3.

Models: `model='conditional'` (Architecture B, default) or 'single' (single-register, step 6).

CLI: python -m archb.train --encoding binary --seed 0 [--model single]
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from archb import circuit, single_register
from archb import sinkhorn as sk
from archb.metrics import all_metrics
from archb.ou import exact_kernel, load_dataset

jax.config.update("jax_enable_x64", True)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
N_SINKHORN = 25
STOP_WINDOW, STOP_RTOL = 100, 1e-3
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
# model name -> module with kernel(theta, encoding) and init_theta(seed, noise, n_layers).
# 'conditional' is Architecture B (the default); 'single' is the step 6 single-register model.
MODELS = {"conditional": circuit, "single": single_register}


def data_measures(data=None):
    """(pi_hat, v) as JAX arrays; v = N / sum N, shape (32, 32)."""
    data = load_dataset() if data is None else data
    N = np.asarray(data["N"], dtype=float)
    return jnp.asarray(data["pi_hat"], dtype=jnp.float64), jnp.asarray(N / N.sum())


def model_measure(theta, encoding, pi_hat, model="conditional"):
    return pi_hat[:, None] * MODELS[model].kernel(theta, encoding)


def make_step(encoding, pi_hat, v, ot_vv, lr, eps, n_iter=N_SINKHORN, model="conditional"):
    lv = sk.safe_log(v)

    @jax.jit
    def step(state):
        theta, m, s, t, f, g, p = state
        w, vjp = jax.vjp(lambda th: model_measure(th, encoding, pi_hat, model), theta)
        lw = sk.safe_log(w)
        f, g = sk.cross_iter(lw, lv, f, g, eps, n_iter)
        p = sk.self_iter(lw, p, eps, n_iter)
        loss = sk.envelope_objective(w, v, f, g, p, ot_vv)
        (grad,) = vjp(f - p)
        t = t + 1
        m = ADAM_B1 * m + (1 - ADAM_B1) * grad
        s = ADAM_B2 * s + (1 - ADAM_B2) * grad**2
        mhat, shat = m / (1 - ADAM_B1**t), s / (1 - ADAM_B2**t)
        theta = theta - lr * mhat / (jnp.sqrt(shat) + ADAM_EPS)
        return (theta, m, s, t, f, g, p), loss

    return step


def converged_loss(theta, encoding, pi_hat, v, ot_vv, eps, f=None, g=None, p=None, model="conditional"):
    """S_eps(w(theta), v) with the cross and self potentials iterated to convergence."""
    w = model_measure(theta, encoding, pi_hat, model)
    f, g, _ = sk.converge_cross(w, v, eps, f, g, tol=1e-11)
    p, _ = sk.converge_self(w, eps, p, tol=1e-11)
    return float(sk.ot_value(w, v, f, g) - 0.5 * sk.ot_self_value(w, p) - 0.5 * ot_vv), (f, g, p)


def train(encoding="binary", seed=0, steps=800, lr=0.03, eps=sk.EPS, noise=0.2, log_every=100, data=None,
          sinkhorn_iters=N_SINKHORN, n_layers=3, model="conditional", stop_smooth=None, theta0=None):
    """Train one model (`model`: a key of MODELS; default Architecture B). Returns a dict with theta, kernel (bin space), loss (per step),
    steps, wall_time (s, including compilation), init_loss / final_loss (converged S_eps).

    Sequence 1 options (defaults keep the spec behaviour): `stop_smooth=k` applies the stop rule
    to the k-step moving average of the loss instead of the raw loss; `theta0` replaces the
    initialization (warm start)."""
    t0 = time.time()
    pi_hat, v = data_measures(data)
    pv, _ = sk.converge_self(v, eps, tol=1e-11)
    ot_vv = sk.ot_self_value(v, pv)

    mod = MODELS[model]
    theta = mod.init_theta(seed, noise, n_layers) if theta0 is None else jnp.asarray(theta0, dtype=jnp.float64)
    init_loss, (f, g, p) = converged_loss(theta, encoding, pi_hat, v, ot_vv, eps, model=model)
    zeros = jnp.zeros_like(theta)
    state = (theta, zeros, zeros, jnp.zeros((), jnp.int64), f, g, p)
    step = make_step(encoding, pi_hat, v, ot_vv, lr, eps, sinkhorn_iters, model)

    losses = []
    for t in range(steps):
        state, loss = step(state)
        losses.append(float(loss))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"non-finite loss at step {t}")
        if log_every and (t % log_every == 0 or t == steps - 1):
            print(f"[{encoding} seed{seed}] step {t:4d}  loss {losses[-1]:.6e}", flush=True)
        k = stop_smooth or 1
        if t >= STOP_WINDOW + k - 1:
            cur = float(np.mean(losses[t - k + 1: t + 1]))
            ref = float(np.mean(losses[t - STOP_WINDOW - k + 1: t - STOP_WINDOW + 1]))
            if abs(cur - ref) < STOP_RTOL * abs(ref):
                if log_every:
                    print(f"[{encoding} seed{seed}] stop at step {t}: loss {losses[t]:.6e}", flush=True)
                break

    theta = state[0]
    final_loss, _ = converged_loss(theta, encoding, pi_hat, v, ot_vv, eps, *state[4:], model=model)
    return {
        "encoding": encoding,
        "seed": seed,
        "theta": np.asarray(theta),
        "kernel": np.asarray(mod.kernel(theta, encoding)),
        "loss": np.asarray(losses),
        "steps": len(losses),
        "wall_time": time.time() - t0,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "settings": {"steps_max": steps, "lr": lr, "eps": eps, "noise": noise, "sinkhorn_iters": sinkhorn_iters,
                     "n_layers": n_layers, **({"model": model} if model != "conditional" else {})},
    }


def save_run(result, runs_dir=RUNS_DIR) -> Path:
    """Write <runs_dir>/<encoding>/seed<k>/{theta.npy, kernel.npy, loss.npy, metrics.json}."""
    out = Path(runs_dir) / result["encoding"] / f"seed{result['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "theta.npy", result["theta"])
    np.save(out / "kernel.npy", result["kernel"])
    np.save(out / "loss.npy", result["loss"])
    info = {k: result[k] for k in ("encoding", "seed", "steps", "wall_time", "init_loss", "final_loss", "settings")}
    info["metrics"] = all_metrics(result["kernel"], exact_kernel())
    (out / "metrics.json").write_text(json.dumps(info, indent=2) + "\n")
    return out


class _Deadline(Exception):
    pass


def lbfgs_capped(fun, x0, maxiter, max_seconds=None, ftol=1e-13, gtol=1e-9):
    """L-BFGS-B (scipy) on fun(x) -> (value, gradient) with a wall-clock cap (sequence 5). Keeps the best point ever
    evaluated (x0 included). Returns dict(x, fun, nit, message, capped, curve: value at each iteration)."""
    from scipy.optimize import minimize

    t0, best, last, curve = time.time(), [np.inf, np.asarray(x0, np.float64)], [np.inf], []

    def f(x):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            raise _Deadline
        v, g = fun(x)
        last[0] = float(v)
        if np.isfinite(last[0]) and last[0] < best[0]:
            best[0], best[1] = last[0], np.array(x, np.float64)
        return last[0], np.asarray(g, np.float64)

    try:
        out = minimize(f, np.asarray(x0, np.float64), jac=True, method="L-BFGS-B",
                       callback=lambda x: curve.append(last[0]),
                       options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": ftol, "gtol": gtol})
        nit, msg, capped = int(out.nit), str(out.message), False
    except _Deadline:
        nit, msg, capped = len(curve), "wall-clock cap", True
    return {"x": best[1], "fun": best[0], "nit": nit, "message": msg, "capped": capped, "curve": np.asarray(curve)}


def fit_capped(loss_fn, theta0, adam_steps, lbfgs_iters, max_seconds, lr0=0.03, lr1=0.003):
    """archb.ceiling.fit with a wall-clock cap: Adam (ceiling.adam, same schedule, not interrupted) then L-BFGS-B on the
    remaining time. Returns the dict of ceiling.fit plus capped (the cap stopped the fit) and wall_time."""
    from archb import ceiling

    t0 = time.time()
    if adam_steps:
        res = ceiling.adam(None, jnp.asarray(theta0), adam_steps, None, loss_fn, lr0, lr1)
    else:
        res = {"theta": np.asarray(theta0), "loss": np.zeros(0), "steps": 0,
               "final_loss": float(loss_fn(jnp.asarray(theta0)))}
    vg = jax.jit(jax.value_and_grad(loss_fn))
    left = max_seconds - (time.time() - t0)
    if left > 0:
        lb = lbfgs_capped(lambda x: vg(jnp.asarray(x)), res["theta"], lbfgs_iters, left)
    else:
        lb = {"x": res["theta"], "fun": np.inf, "nit": 0, "message": "wall-clock cap (Adam)", "capped": True,
              "curve": np.zeros(0)}
    better = lb["fun"] <= res["final_loss"]
    return {"theta": np.asarray(lb["x"] if better else res["theta"]),
            "loss": np.concatenate([res["loss"], lb["curve"]]), "steps": res["steps"] + lb["nit"],
            "adam_loss": res["final_loss"], "final_loss": min(lb["fun"], res["final_loss"]),
            "lbfgs_message": lb["message"], "lbfgs_iters": lb["nit"], "capped": lb["capped"],
            "wall_time": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the conditional QCBM on the OU kernel.")
    ap.add_argument("--encoding", default="binary", choices=["binary", "gray"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--eps", type=float, default=sk.EPS)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sinkhorn-iters", type=int, default=N_SINKHORN)
    ap.add_argument("--model", default="conditional", choices=list(MODELS))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = ap.parse_args(argv)
    res = train(args.encoding, args.seed, args.steps, args.lr, args.eps, args.noise, args.log_every,
                sinkhorn_iters=args.sinkhorn_iters, model=args.model)
    out = save_run(res, args.runs_dir)
    m = json.loads((out / "metrics.json").read_text())["metrics"]
    print(f"saved {out}: steps {res['steps']}, {res['wall_time']:.1f}s, final S_eps {res['final_loss']:.6e}")
    print("  " + "  ".join(f"{k} {val:.4f}" for k, val in m.items()))


if __name__ == "__main__":
    main()
