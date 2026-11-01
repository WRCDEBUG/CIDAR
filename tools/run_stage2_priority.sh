#!/bin/bash
# Priority chain v2 -- centres the MODULE story.
#   * noblur_2stage first: the second injection point at the fusion-net input is
#     aimed exactly at low-light, which is the one case where the blind fine-tune
#     still beats the module (it wins there because it adapts fusion_net).
#   * maxcap_noblur: module plus the same capacity budget the blind fine-tune had.
# The matched blind fine-tune + its LLVIP evaluation is running in parallel
# (tools/run_matched_blind.sh) and is the comparison that decides whether the
# module or the parameters are doing the work.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D="noise10,noise20,dark"
for i in $(seq 1 240); do
  [ -f output/pr1/stage2_oracle_long.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') PRIORITY START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 21600 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run noblur_2stage         --dar_mode predicted --dar_second --train_degradations $D --steps 3000 --eval_every 1500
run noblur_2stage_auxreg  --dar_mode predicted --dar_second --train_degradations $D --aux_reg 1.0 --steps 3000 --eval_every 1500
run maxcap_noblur         --dar_mode predicted --unfreeze all_except_flow --train_degradations $D --steps 3000 --eval_every 1500
run maxcap_noblur_2stage  --dar_mode predicted --dar_second --unfreeze all_except_flow --train_degradations $D --steps 3000 --eval_every 1500
run noblur_auxreg         --dar_mode predicted --train_degradations $D --aux_reg 1.0 --steps 3000 --eval_every 1500
run noblur_s1             --dar_mode predicted --train_degradations $D --steps 3000 --eval_every 0 --seed 11
run noblur_s2             --dar_mode predicted --train_degradations $D --steps 3000 --eval_every 0 --seed 22
run noblur_long           --dar_mode predicted --train_degradations $D --steps 8000 --eval_every 4000
echo "### $(date '+%F %T') PRIORITY DONE" >> "$L/queue_progress.log"
