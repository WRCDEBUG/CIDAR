#!/bin/bash
# The same mechanism on the cross-modal + temporal descriptor, to see whether the
# video-only cue adds anything on top of demand creation.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
echo "### $(date '+%F %T') DDC2 START" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START ddc_temporal" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag ddc_temporal --dar_mode predicted --est_mode crosstemporal \
  --oracle_mix 0.5 --train_degradations $D4 --steps 8000 --eval_every 4000 \
  > "$L/ddc_temporal.log" 2>&1
echo "### $(date '+%F %T') END   ddc_temporal exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START temporal_nomix" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag temporal_nomix --dar_mode predicted --est_mode crosstemporal \
  --train_degradations $D4 --steps 8000 --eval_every 4000 \
  > "$L/temporal_nomix.log" 2>&1
echo "### $(date '+%F %T') END   temporal_nomix exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') DDC2 DONE" >> "$L/queue_progress.log"
