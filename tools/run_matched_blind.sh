#!/bin/bash
# Matched blind fine-tune baseline for the module story.
# Identical training data (noise10, noise20, dark -- the same set DAR uses),
# identical steps, identical objective and identical clean_prob.  The ONLY
# difference is that DAR is switched off and the whole backbone except the flow
# net is trained instead.  Saved in full so it can be evaluated on real LLVIP,
# which is the comparison that decides whether the module or the parameters are
# doing the work.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
echo "### $(date '+%F %T') START blind_matched" >> "$L/queue_progress.log"
timeout 14400 $PY tools/stage2_dar.py --tag blind_matched --dar_mode off \
  --unfreeze all_except_flow --train_degradations noise10,noise20,dark \
  --steps 3000 --eval_every 0 --save_full \
  > "$L/blind_matched.log" 2>&1
echo "### $(date '+%F %T') END   blind_matched exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START llvip_blind" >> "$L/queue_progress.log"
timeout 7200 $PY tools/real_lowlight_eval.py --n_runs 8 --frames_per_run 21 --stride 7 \
  --degrade none,noise10,noise20,dark,blur --dar_ckpt output/pr1/stage2_blind_matched.pth \
  --dar_mode off --output output/pr1/real_lowlight_blind_matched.json \
  > "$L/llvip_blind.log" 2>&1
echo "### $(date '+%F %T') END   llvip_blind exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') BLIND DONE" >> "$L/queue_progress.log"
