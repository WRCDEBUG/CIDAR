#!/bin/bash
# Second unattended queue: chained after queue 1 finishes.
#   * longer schedules for the two best variants
#   * multi-seed repeats (the paper needs >=3 seeds for a paired test)
#   * a max-capacity run (DAR + everything except the flow net)
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
mkdir -p "$L"

# wait for queue 1 to report completion
for i in $(seq 1 720); do
  grep -q 'QUEUE DONE' "$L/queue_progress.log" 2>/dev/null && break
  sleep 60
done
echo "### $(date '+%F %T') QUEUE2 START" >> "$L/queue_progress.log"

run() {
  name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 14400 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"
}

# longer schedule: does the DAR keep improving past 3k steps?
run oracle_long    --dar_mode oracle    --steps 8000 --eval_every 4000
run predicted_long --dar_mode predicted --steps 8000 --eval_every 4000
# multi-seed for the actual method
run predicted_s1   --dar_mode predicted --steps 3000 --eval_every 0 --seed 11
run predicted_s2   --dar_mode predicted --steps 3000 --eval_every 0 --seed 22
# max capacity: DAR plus everything except the frozen flow net
run predicted_maxcap --dar_mode predicted --unfreeze all_except_flow --steps 3000 --eval_every 0

echo "### $(date '+%F %T') QUEUE2 DONE" >> "$L/queue_progress.log"
