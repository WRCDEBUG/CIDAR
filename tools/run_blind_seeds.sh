#!/bin/bash
# The baseline also needs more than one seed.  The headline configuration is
# 36.38 +- 1.85 over three seeds; the blind fine-tune it is compared against has
# only one, so the comparison is not yet seed-matched on both sides.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D="noise10,noise20,dark"
echo "### $(date '+%F %T') BLINDSEEDS START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run blind_matched_s1 --dar_mode off --unfreeze all_except_flow --train_degradations $D --steps 8000 --eval_every 0 --seed 11
run blind_matched_s2 --dar_mode off --unfreeze all_except_flow --train_degradations $D --steps 8000 --eval_every 0 --seed 22
echo "### $(date '+%F %T') BLINDSEEDS DONE" >> "$L/queue_progress.log"
