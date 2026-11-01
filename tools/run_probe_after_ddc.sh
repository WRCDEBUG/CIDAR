#!/bin/bash
# The decisive closure for the identifiability-collapse claim: re-run the linear
# probe on the DDC-trained descriptor.  If demand-driven conditioning works, the
# probe must STAY HIGH (near the 0.953 of the raw statistics) instead of being
# eroded to 0.593, and blur recovery must appear.  Both have to move, not one.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
for i in $(seq 1 480); do
  [ -f output/pr1/stage2_ddc_mix.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') PROBE-DDC START" >> "$L/queue_progress.log"
timeout 3600 $PY tools/identifiability_probe.py --windows 60 \
  --output output/pr1/identifiability_probe_ddc.json > "$L/probe_ddc.log" 2>&1
echo "### $(date '+%F %T') END   probe_ddc exit=$?" >> "$L/queue_progress.log"

# and the real-video retest of the DDC checkpoint
for i in $(seq 1 480); do
  [ -f output/pr1/stage2_ddc_mix.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') START xvideo_ddc" >> "$L/queue_progress.log"
timeout 7200 $PY tools/crossvideo_degradation_eval.py --datasets VTMOT,HDO --window_stride 20 \
  --dar_ckpt output/pr1/stage2_ddc_mix.pth --est_mode crossspec \
  --output output/pr1/crossvideo_ddc.json > "$L/xvideo_ddc.log" 2>&1
echo "### $(date '+%F %T') END   xvideo_ddc exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') PROBE-DDC DONE" >> "$L/queue_progress.log"
