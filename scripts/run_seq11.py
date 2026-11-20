"""Sequence 11: persistence and volatility-contrast grid of the Markov-switching AR target. Same circuits, fits and budgets as the replication of
sequence 8 (scripts/run_seq8.py, imported, not copied); only the truths change.

Truths: msar with stay probability A (both regimes) and noise sd (sigma_calm, sigma_turbulent), c = 0.95 / 0.7:
  persistence  A in {0.5, 0.8, 0.95, 0.995} at sigma = (0.3, 1.0); A = 0.5 is the negative control (the next regime
               is independent of the past, so G_0 = 0 and no memory can help)
  contrast     sigma = (0.4, 0.6), (0.35, 0.7), (0.2, 1.0) (ratio 1.5, 2, 5) at A = 0.98
The centre point A = 0.98, sigma = (0.3, 1.0) is msar itself: its five sequence 8 replication datasets serve (read
from runs/seq8/rep/ by scripts/seq11_summary.py). Two datasets per grid point (training paths and held-out path from
the seed), each with Architecture B anc2 stage 1 (3 inits), the C-QSM m = 1 + 2 anc coherent and dephased (gamma = 1,
3 inits each), the classical switch of anc2 kernels S = 2 (3 inits), the factored IOHMM S = 2 (3 inits), the MS-AR
family fit and the truth.

-> runs/seq11/grid/<dataset>/<model>/<init>/{result.json, params.npz, loss.npy, lp_test.npy}; data in runs/seq11/data/
(regenerated from the seeds). Resumable (finished fits skipped).

Usage: python scripts/run_seq11.py [--datasets ...] [--models ...] [--workers 8] [--max-minutes M] [--list]
       python scripts/run_seq11.py --smoke     (one fit per type, 20 s caps, -> runs/seq11_smoke/)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_seq5 as S5  # noqa: E402
import run_seq8 as S8  # noqa: E402
from archb import iolin, regimes as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Job = S5.Job
S5.CFG["out"] = ROOT / "runs" / "seq11"
C = (0.95, 0.7)

# name -> (stay, (sigma_calm, sigma_turbulent)); module level, so the spawned workers see the same table
GRID = {}
for _a, _lab in ((0.5, "050"), (0.8, "080"), (0.95, "095"), (0.995, "0995")):
    GRID[f"A{_lab}"] = (_a, (0.3, 1.0))
for _sig, _lab in (((0.4, 0.6), "1.5"), ((0.35, 0.7), "2"), ((0.2, 1.0), "5")):
    GRID[f"r{_lab}"] = (0.98, _sig)
DATASETS = {}
for _k, (_name, _p) in enumerate(GRID.items()):
    for _d in range(2):
        DATASETS[f"{_name}-d{_d}"] = ("msar", None, 1100 + 20 * _k + 2 * _d)  # make_data uses seed and seed + 1
PARAMS = {f"{n}-d{d}": p for n, p in GRID.items() for d in range(2)}
S8.DATASETS.update(DATASETS)


def truth_T(name):
    stay, sig = PARAMS[name]
    return np.asarray(R.msar_T(C, sig, (stay, stay)))


def make_data(name):
    T, seed = truth_T(name), DATASETS[name][2]
    X, H = R.simulate(T, R.N_PATHS, R.N_KEEP, R.BURN_IN, seed)
    xt, ht = R.simulate(T, 1, R.N_TEST, R.BURN_IN, seed + 1)
    return {"X": X, "H": H, "test": xt[0], "test_h": ht[0]}


def prepare_data(names):
    for name in names:
        p = S8.data_path(name)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **make_data(name))
            os.replace(tmp, p)


def reference(name):
    if name not in S8._REF:
        d, lin = S8.load(name), iolin.IOLin.from_iohmm(truth_T(name))
        S8._REF[name] = {"lin": lin, "lp": lin.logprobs(d["test"]), "train_ll": lin.path_loglik(d["X"])}
    return S8._REF[name]


def jobs_all():
    J = []
    for ds in DATASETS:
        J += [Job(1, "grid", ds, "archb-anc2", i, S8.CAP["archb"]) for i in range(3)]
        J += [Job(1, "grid", ds, f"cqsm-m1-anc2-g{g:g}", i, S8.CAP["m1"]) for g in (0.0, 1.0) for i in range(3)]
        J += [Job(1, "grid", ds, "sw-anc2-S2", i, S8.CAP["S2"]) for i in range(3)]
        J += [Job(1, "grid", ds, "iohmm-fact-2", i, S8.CAP["iohmm"]) for i in range(3)]
        J += [Job(1, "grid", ds, "family-msar", 0, S8.CAP["family"]), Job(1, "grid", ds, "truth", 0, S8.CAP["truth"])]
    return J


def stage1_dir(job):
    return S5.CFG["out"] / "grid" / job.dataset / "archb-anc2"


# run_seq8 looks these up at call time (main, fit, fit_cqsm, fit_switch, deps), also in the spawned workers
S8.prepare_data, S8.reference, S8.jobs_all, S8.stage1_dir = prepare_data, reference, jobs_all, stage1_dir

if __name__ == "__main__":
    if "--smoke" in sys.argv:  # the smoke selection keeps one stage 1 init: let the stage 2 fits start from it
        S8.deps = lambda job: [stage1_dir(job) / "0" / "result.json"] if job.model.startswith(("cqsm", "sw")) else []
        if "--out" not in sys.argv:
            sys.argv += ["--out", str(ROOT / "runs" / "seq11_smoke")]
    S8.main()
