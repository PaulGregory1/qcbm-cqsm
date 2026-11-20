#!/bin/sh
# The laptop slept from 03:03 to 09:02 on 5 Nov: the fits in flight then ended on their wall-clock cap. After the
# overnight run, remove those results and fit them again, then redo the summary.
cd "$(dirname "$0")/../.."
while pgrep -f "runs/seq11/overnight.sh" > /dev/null; do sleep 60; done
for d in A0995-d0/cqsm-m1-anc2-g0/0 A0995-d0/cqsm-m1-anc2-g0/1 A0995-d0/cqsm-m1-anc2-g0/2 \
         A0995-d0/cqsm-m1-anc2-g1/0 A0995-d0/cqsm-m1-anc2-g1/1 A0995-d0/cqsm-m1-anc2-g1/2 \
         A0995-d0/sw-anc2-S2/0 A0995-d0/sw-anc2-S2/1 A095-d1/cqsm-m1-anc2-g1/1; do
  rm -rf "runs/seq11/grid/$d"
done
./.venv/bin/python scripts/run_seq11.py --workers 9
echo "redo fits exit status $?"
./.venv/bin/python scripts/seq11_summary.py --workers 4
echo "summary exit status $?"
