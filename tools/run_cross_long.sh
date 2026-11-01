#!/bin/bash
# The cross-modal estimator is the one enhancement that worked: at matched 3000
# steps it lifts low-light from +3.3% to +18.1% and the three-degradation mean
# from 9.35% to 13.52%, which validates the identifiability diagnosis (a
# tone-mapped low-light condition is ambiguous from one modality, and the IR/VIS
# relationship breaks the tie).  Training length mattered a lot for every other
# configuration, so run it long, with and without the second injection point,
# and re-test on real night video.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D="noise10,noise20,dark"
for i in $(seq 1 400); do
  [ -f output/pr1/stage2_cross_2stage_3k.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') CROSS START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run cross_noblur_long    --dar_mode predicted --est_mode cross --train_degradations $D --steps 8000 --eval_every 4000
run cross_2stage_long    --dar_mode predicted --est_mode cross --dar_second --train_degradations $D --steps 8000 --eval_every 4000
echo "### $(date '+%F %T') START llvip_cross2stage" >> "$L/queue_progress.log"
timeout 7200 $PY tools/real_lowlight_eval.py --n_runs 8 --frames_per_run 21 --stride 7 \
  --degrade none,noise10,noise20,dark,blur --dar_ckpt output/pr1/stage2_cross_2stage_long.pth \
  --output output/pr1/real_lowlight_cross_2stage_long.json > "$L/llvip_cross.log" 2>&1
echo "### $(date '+%F %T') END   llvip_cross2stage exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') CROSS DONE" >> "$L/queue_progress.log"
