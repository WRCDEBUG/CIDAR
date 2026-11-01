#!/bin/bash
# Fourth queue: fix the low-light branch's failure to transfer.
# LLVIP protocol B showed noise transfers (32/32 wins) but the fixed-gamma
# low-light branch hurts (-3.81%).  Train it against a domain-randomised
# photometric model and re-test on real night video.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
mkdir -p "$L"
for i in $(seq 1 900); do
  grep -q 'QUEUE3 DONE' "$L/queue_progress.log" 2>/dev/null && break
  sleep 60
done
echo "### $(date '+%F %T') QUEUE4 START" >> "$L/queue_progress.log"

echo "### $(date '+%F %T') START photo_auxce" >> "$L/queue_progress.log"
timeout 14400 $PY tools/stage2_dar.py --tag photo_auxce --dar_mode predicted --aux_ce 0.5 \
    --train_degradations noise10,noise20,darkrand,blur --steps 3000 --eval_every 1500 \
    > "$L/photo_auxce.log" 2>&1
echo "### $(date '+%F %T') END   photo_auxce exit=$?" >> "$L/queue_progress.log"

# re-test the domain-randomised checkpoint on real night video, protocol B
echo "### $(date '+%F %T') START llvip_photo" >> "$L/queue_progress.log"
timeout 7200 $PY tools/real_lowlight_eval.py --n_runs 8 --frames_per_run 21 --stride 7 \
    --degrade none,noise10,noise20,dark,blur --dar_ckpt output/pr1/stage2_photo_auxce.pth \
    --output output/pr1/real_lowlight_photo_auxce.json > "$L/llvip_photo.log" 2>&1
echo "### $(date '+%F %T') END   llvip_photo exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') QUEUE4 DONE" >> "$L/queue_progress.log"
