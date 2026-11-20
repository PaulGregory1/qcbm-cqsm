"""Sequence 8 job table (no fit is run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_seq8 as R8  # noqa: E402


def test_unique_jobs_and_more_inits():
    jobs = R8.jobs_all()
    keys = [(j.exp, j.dataset, j.model, j.init) for j in jobs]
    assert len(keys) == len(set(keys))
    for ds in ["smsar16"] + [f"smsar16-d{d}" for d in range(1, 5)]:
        for g in ("0", "1"):
            inits = sorted(j.init for j in jobs if j.exp == "rep" and j.dataset == ds and j.model == f"cqsm-m1-anc2-g{g}")
            assert inits == list(range(20))
