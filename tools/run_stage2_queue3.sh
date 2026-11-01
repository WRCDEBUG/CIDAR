#!/bin/bash
# Third unattended queue: estimator supervision.
# The first `predicted` run showed the descriptor is free to stay
# non-discriminative (dark +3.3% vs oracle +58.8%), so the estimator now gets an
# auxiliary degradation-classification loss on the synthetic labels.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
mkdir -p "$L"

for i in $(seq 1 900); do
  grep -q 'QUEUE2 DONE' "$L/queue_progress.log" 2>/dev/null && break
  sleep 60
done
echo "### $(date '+%F %T') QUEUE3 START" >> "$L/queue_progress.log"

run() {
  name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 14400 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"
}

run predicted_auxce      --dar_mode predicted --aux_ce 0.5 --steps 3000 --eval_every 1500
run predicted_auxce_w1   --dar_mode predicted --aux_ce 1.0 --steps 3000 --eval_every 1500
run predicted_auxce_s1   --dar_mode predicted --aux_ce 0.5 --steps 3000 --eval_every 0 --seed 11
run predicted_auxce_long --dar_mode predicted --aux_ce 0.5 --steps 8000 --eval_every 4000

echo "### $(date '+%F %T') QUEUE3 DONE" >> "$L/queue_progress.log"
