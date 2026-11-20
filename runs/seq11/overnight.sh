#!/bin/sh
# Overnight local run of sequence 11: the fits, then the summary.
cd "$(dirname "$0")/../.."
./.venv/bin/python scripts/run_seq11.py --workers 9
echo "fits exit status $?"
./.venv/bin/python scripts/seq11_summary.py --workers 4
echo "summary exit status $?"
