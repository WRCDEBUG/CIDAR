#!/bin/bash
# Multi-seed for the headline configuration (cross-modal, 3 degradations, 8k) plus
# the spectral-only ablation that isolates the spectral cue from the cross-modal one.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D3="noise10,noise20,dark"
D4="noise10,noise20,dark,blur"
echo "### $(date '+%F %T') SEEDS START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run spectral_long   --dar_mode predicted --est_mode spectral --train_degradations $D4 --steps 8000 --eval_every 4000
run cross_noblur_s1 --dar_mode predicted --est_mode cross    --train_degradations $D3 --steps 8000 --eval_every 0 --seed 11
run cross_noblur_s2 --dar_mode predicted --est_mode cross    --train_degradations $D3 --steps 8000 --eval_every 0 --seed 22
echo "### $(date '+%F %T') SEEDS DONE" >> "$L/queue_progress.log"
