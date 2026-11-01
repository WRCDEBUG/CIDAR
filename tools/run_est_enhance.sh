#!/bin/bash
# Estimator enhancement, matched to the existing 3000-step `predicted` baseline
# (which trained on noise10,noise20,dark,blur with the ORIGINAL stem and got a
# 9.35% three-degradation mean with blur +0.16%).
#   hf    : + a local-contrast channel computed before downsampling, because the
#           original stem downsamples 4x then pools globally and therefore cannot
#           see blur at all (oracle blur 43.4% vs deployable 0.19%).
#   cross : hf plus a single estimator over BOTH modalities, because low-light on
#           one modality is ambiguous (darkened bright scene vs dark scene) and
#           only the IR/VIS relationship breaks the tie.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
echo "### $(date '+%F %T') EST START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 21600 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run hf_3k            --dar_mode predicted --est_mode hf    --train_degradations $D4 --steps 3000 --eval_every 1500
run cross_3k         --dar_mode predicted --est_mode cross --train_degradations $D4 --steps 3000 --eval_every 1500
run cross_2stage_3k  --dar_mode predicted --est_mode cross --dar_second --train_degradations $D4 --steps 3000 --eval_every 1500
echo "### $(date '+%F %T') EST DONE" >> "$L/queue_progress.log"
