#!/bin/bash
# The measured mechanism, and the fix it implies.
#
# Measured: raw spectral statistics carry almost perfect blur information
# (linear probe 0.953 / 1.000) but after end-to-end training the descriptor
# retains only 0.593 / 0.913 -- training ERODES identifiability when the
# downstream rectification loss does not reward it.  That is also why the earlier
# auxiliary-supervision attempts failed: they were applied to a representation
# that had no blur information to preserve, so the auxiliary loss could only
# distort.  With spectral inputs the information is available, so an auxiliary
# loss has something to hold on to.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
for i in $(seq 1 400); do
  [ -f output/pr1/stage2_nostride_long.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') BLURFIX2 START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run crossspec_auxce  --dar_mode predicted --est_mode crossspec --aux_ce 0.5 --train_degradations $D4 --steps 8000 --eval_every 4000
echo "### $(date '+%F %T') BLURFIX2 DONE" >> "$L/queue_progress.log"
