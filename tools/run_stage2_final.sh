#!/bin/bash
# Final chain.  Key facts that set the order:
#   * oracle at 8000 steps jumped from 24.2% to 40.6% mean, and blur from +2.2%
#     to +43.4% -- training length matters a lot, so the long deployable run goes
#     first.
#   * the matched blind fine-tune reaches 22.32% on synthetic and ALSO transfers
#     to real noise, so the module cannot win on transfer alone; it has to win on
#     conditioning, which means the estimator is the critical part.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D="noise10,noise20,dark"
for i in $(seq 1 300); do
  [ -f output/pr1/stage2_noblur_auxreg.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') FINAL START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 21600 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run noblur_long            --dar_mode predicted --train_degradations $D --steps 8000 --eval_every 4000
run noblur_2stage          --dar_mode predicted --dar_second --train_degradations $D --steps 3000 --eval_every 1500
run noblur_2stage_auxreg   --dar_mode predicted --dar_second --train_degradations $D --aux_reg 1.0 --steps 3000 --eval_every 1500
run noblur_2stage_long     --dar_mode predicted --dar_second --train_degradations $D --steps 8000 --eval_every 4000
run maxcap_noblur_2stage   --dar_mode predicted --dar_second --unfreeze all_except_flow --train_degradations $D --steps 3000 --eval_every 1500
run noblur_s1              --dar_mode predicted --train_degradations $D --steps 3000 --eval_every 0 --seed 11
run noblur_s2              --dar_mode predicted --train_degradations $D --steps 3000 --eval_every 0 --seed 22
echo "### $(date '+%F %T') FINAL DONE" >> "$L/queue_progress.log"
