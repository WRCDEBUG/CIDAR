#!/bin/bash
# DDC: demand-driven conditioning.
#
# The measured failure: the descriptor is 0.953 identifiable from raw statistics
# but only 0.593 after end-to-end training, and blur recovery stays at +0.31%,
# while an oracle descriptor recovers +43.4%.  The rectification loss does not
# reward the information, so the optimiser discards it.  DDC makes the rectifier
# DEPEND on the descriptor (oracle mix-in) so that the loss becomes sensitive to
# descriptor quality; the estimator then has a gradient demanding informativeness.
#
# The temporal fingerprint is a video-only identifiability cue: three cross-frame
# statistics already identify the degradation at 0.813, which no image-restoration
# method can use.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
echo "### $(date '+%F %T') DDC START" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START ddc_mix" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag ddc_mix --dar_mode predicted --est_mode crossspec \
  --oracle_mix 0.5 --train_degradations $D4 --steps 8000 --eval_every 4000 \
  > "$L/ddc_mix.log" 2>&1
echo "### $(date '+%F %T') END   ddc_mix exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') DDC DONE" >> "$L/queue_progress.log"
