#!/bin/bash
# Sixth queue: combine the two ingredients that each work on their own.
#   blind fine-tune of everything but the flow net   15.99%  (capacity)
#   DAR conditioning, trained without blur           14.66%  (conditioning)
# The oracle shows conditioning can reach 31.49%, so a joint run has both
# ingredients and is the most likely deployable configuration to beat 15.99%.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D="noise10,noise20,darkrand"
for i in $(seq 1 1200); do
  grep -q 'QUEUE5 DONE' "$L/queue_progress.log" 2>/dev/null && break
  sleep 60
done
echo "### $(date '+%F %T') QUEUE6 START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 21600 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run maxcap_noblur       --dar_mode predicted --unfreeze all_except_flow --train_degradations $D --steps 3000 --eval_every 1500
run maxcap_noblur_auxreg --dar_mode predicted --unfreeze all_except_flow --train_degradations $D --aux_reg 1.0 --steps 3000 --eval_every 1500
echo "### $(date '+%F %T') QUEUE6 DONE" >> "$L/queue_progress.log"
