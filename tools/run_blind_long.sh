#!/bin/bash
# Fair comparison: the earlier blind_matched baseline used 3000 steps while the
# best DAR run used 8000, so the two were not matched.  This runs the blind
# fine-tune at 8000 steps and evaluates it on real LLVIP too.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
for i in $(seq 1 300); do
  [ -f output/pr1/stage2_noblur_2stage.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') START blind_matched_long" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag blind_matched_long --dar_mode off \
  --unfreeze all_except_flow --train_degradations noise10,noise20,dark \
  --steps 8000 --eval_every 4000 --save_full > "$L/blind_matched_long.log" 2>&1
echo "### $(date '+%F %T') END   blind_matched_long exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START llvip_blind_long" >> "$L/queue_progress.log"
timeout 7200 $PY tools/real_lowlight_eval.py --n_runs 8 --frames_per_run 21 --stride 7 \
  --degrade none,noise10,noise20,dark,blur --dar_ckpt output/pr1/stage2_blind_matched_long.pth \
  --dar_mode off --output output/pr1/real_lowlight_blind_long.json > "$L/llvip_blind_long.log" 2>&1
echo "### $(date '+%F %T') END   llvip_blind_long exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') BLINDLONG DONE" >> "$L/queue_progress.log"
