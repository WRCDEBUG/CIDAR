#!/bin/bash
# Fifth queue: make the ESTIMATOR informative enough to approach the oracle.
# Findings so far (mean over the three recoverable degradations):
#   oracle                31.49%   (supervised conditioning)
#   blind fine-tune       15.99%   (265k params, everything but the flow net)
#   predicted_loo         14.66%   (unsupervised, trained without blur)
#   noconv / input_restore ~9%     (no conditioning)
#   aux_ce lambda=0.5      4.95%   (CE dominated and distorted the optimisation)
# So the conditioning mechanism works; the estimator is the bottleneck.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D="noise10,noise20,darkrand"
for i in $(seq 1 900); do
  grep -q 'QUEUE4 DONE' "$L/queue_progress.log" 2>/dev/null && break
  sleep 60
done
echo "### $(date '+%F %T') QUEUE5 START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 14400 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run noblur_auxreg  --dar_mode predicted --train_degradations $D --aux_reg 1.0  --steps 3000 --eval_every 1500
run noblur_auxce02 --dar_mode predicted --train_degradations $D --aux_ce 0.02  --steps 3000 --eval_every 1500
run noblur_s1      --dar_mode predicted --train_degradations $D --steps 3000 --eval_every 0 --seed 11
run noblur_s2      --dar_mode predicted --train_degradations $D --steps 3000 --eval_every 0 --seed 22
echo "### $(date '+%F %T') QUEUE5 DONE" >> "$L/queue_progress.log"
