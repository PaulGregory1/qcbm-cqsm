"""Sequence 7 job table: datasets, oracle-free inits and budgets (no fit is run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_seq7 as R7  # noqa: E402


def test_seq5_strings_reused():
    # seeds 0, 1 of N = 32 and seed 0 of N = 16 are the sequence 5 training strings
    assert R7.DATASETS["renewal32-s0"]["seed"] == 1200 and R7.DATASETS["renewal32-s1"]["seed"] == 1201
    assert R7.DATASETS["renewal16-s0"]["seed"] == 1100
    assert len({(d["N"], d["T"], d["seed"]) for d in R7.DATASETS.values()}) == len(R7.DATASETS)


def test_no_oracle_init_and_unique_jobs():
    jobs = R7.jobs_all()
    for j in jobs:
        if j.model.startswith("hmm-") and j.init == "lump":
            assert int(j.model.split("-")[1]) < R7.DATASETS[j.dataset]["N"]
    keys = [(j.dataset, j.model, str(j.init)) for j in jobs]
    assert len(keys) == len(set(keys))  # a fit is never run twice under two experiments


def test_core_pair_present():
    jobs = R7.jobs_all()
    for s in range(5):
        ds = f"renewal32-s{s}"
        assert sum(j.dataset == ds and j.model == "qsm-D4-r1" for j in jobs) == 6
        assert sum(j.dataset == ds and j.model == "hmm-16" for j in jobs) == 7
        assert all(j.tier == 1 for j in jobs if j.dataset == ds and j.model in ("qsm-D4-r1", "hmm-16", "hmm-32"))


def test_curve_has_five_strings_per_size():
    # with the core (T = 5e4) and datasize (T = 2e5, seeds 0-2) fits, every model of the curve has 5 strings per T
    jobs = R7.jobs_all()
    pats = {25000: "renewal32-T25k-s{}", 50000: "renewal32-s{}", 100000: "renewal32-T100k-s{}",
            200000: "renewal32-T200k-s{}", 400000: "renewal32-T400k-s{}"}
    for T, pat in pats.items():
        for model in ("qsm-D4-r1", "qsm-D8-r1", "hmm-16", "hmm-32-es"):
            have = {j.dataset for j in jobs if j.model == model}
            assert all(pat.format(s) in have for s in range(5)), (T, model)
            assert all(R7.DATASETS[pat.format(s)]["T"] == T for s in range(5))
    assert all(j.tier == 5 for j in jobs if j.exp == "curve")
