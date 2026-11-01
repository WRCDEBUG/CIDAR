#!/bin/bash
# Reviewer control: is the blur fix feature engineering or a mechanism?
# The rule says the stem cannot identify blur because it downsamples 4x before
# pooling.  If that is the mechanism, a NO-STRIDE stem (same convolutions, high
# frequencies preserved to the pooling) should identify blur by itself.  If it
# cannot, the hand-computed spectral prior is necessary rather than convenient.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
for i in $(seq 1 400); do
  [ -f output/pr1/stage2_crossspec_long.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') NOSTRIDE START" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START nostride_long" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag nostride_long --dar_mode predicted \
  --est_mode nostride --train_degradations $D4 --steps 8000 --eval_every 4000 \
  > "$L/nostride_long.log" 2>&1
echo "### $(date '+%F %T') END   nostride_long exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') NOSTRIDE DONE" >> "$L/queue_progress.log"
