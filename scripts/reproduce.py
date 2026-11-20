"""Rebuild every summary, table, figure and note from the fits stored in runs/ (no training).

Usage: python scripts/reproduce.py [--only seq4 seq7 ...]

Each step reads the run files of one sequence and rewrites runs/<seq>/summary.json, figs/<seq>_*.png and, for the
sequences that have one, the note reports/<seq>.md. With the stored fits unchanged, `git status runs` shows no
change afterwards. To retrain a sequence instead, run scripts/run_<seq>.py (see the README).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [  # (name, script and arguments): in this order, since some steps read the output of earlier ones
    ("baseline", "references.py"),
    ("baseline", "write_note.py"),
    ("seq1", "seq1_report.py"),
    ("seq1", "seq1_note.py"),
    ("seq2", "seq2_report.py"),
    ("seq2", "seq2_note.py"),
    ("seq2", "seq2_followups.py"),
    ("seq3", "seq3_report.py"),
    ("seq4", "seq4_report.py"),
    ("seq4_real", "seq4_real_report.py"),
    ("memdiag", "run_memdiag.py"),  # memory diagnostics of the sequence 4-5 fits, no training
    ("memdiag", "run_memdiag.py --encoded"),
    ("traj", "run_traj.py"),  # generated paths and regime switches of the same fits
    ("seq5", "seq5_summary.py"),
    ("seq6", "run_seq6.py eval"),  # held-out scores of the stored fits, no training
    ("seq6", "seq6_summary.py"),
    ("seq7", "seq7_summary.py"),
    ("seq8", "seq8_summary.py"),
    ("seq9", "seq9_summary.py"),
    ("seq10", "seq10_summary.py"),
    ("seq11", "seq11_summary.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="step names (baseline, seq1, ..., seq11)")
    a = ap.parse_args()
    (ROOT / "reports").mkdir(exist_ok=True)
    failed = []
    for name, step in STEPS:
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        script, *args = step.split()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"{name:10s} {step:24s} {status:6s} {time.time() - t0:6.0f} s", flush=True)
        if r.returncode:
            failed.append(step)
            print(r.stderr[-2000:], file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
