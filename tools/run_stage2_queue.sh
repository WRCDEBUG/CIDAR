#!/bin/bash
# Unattended Stage-2 queue.  Each run: 3000 steps (~28 min) + 3 evals (~7 min).
# Every variant is smaller than the 265,573-param blind fine-tune it has to beat.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
mkdir -p output/pr1/stage2_logs
run() {
  name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> output/pr1/stage2_logs/queue_progress.log
  timeout 7200 $PY tools/stage2_dar.py --tag "$name" "$@" \
      > "output/pr1/stage2_logs/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> output/pr1/stage2_logs/queue_progress.log
}

# 1. oracle conditioning: upper bound of what conditioning can buy
run oracle        --dar_mode oracle        --steps 3000 --eval_every 1500
# 2. the actual method: descriptor estimated from the input
run predicted     --dar_mode predicted     --steps 3000 --eval_every 1500
# 3. conditioning ablation: same rectifier, no conditioning path
run noconv        --dar_mode noconv        --steps 3000 --eval_every 1500
# 4. input-side control: restore the raw frames instead of the features
run input_restore --dar_mode off --input_restore --steps 3000 --eval_every 1500
# 5. leave-one-out: never train on blur, test generalisation to an unseen type
run predicted_loo --dar_mode predicted --train_degradations noise10,noise20,dark --steps 3000 --eval_every 1500

echo "### $(date '+%F %T') QUEUE DONE" >> output/pr1/stage2_logs/queue_progress.log
