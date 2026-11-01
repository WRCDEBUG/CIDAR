#!/bin/bash
# Score methods as their fusion runs finish, one at a time.
#
# score_baselines.py reads+rewrites output/pr1/baseline_metrics/all.json, so two
# scoring processes must never overlap; this queue serialises them and waits for
# the matching fusion process to disappear first.  MATLAB's contract_metrics is
# CPU-bound and the fusion jobs are GPU-bound, so the two run concurrently
# without hurting throughput.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs

for m in "$@"; do
  echo "### $(date '+%F %T') SCORE-WAIT $m" >> "$L/queue_progress.log"
  while pgrep -f "bench_baselines.py --method ${m} " > /dev/null 2>&1; do
    sleep 30
  done
  echo "### $(date '+%F %T') SCORE-START $m" >> "$L/queue_progress.log"
  $PY tools/score_baselines.py "$m" >> "$L/score_${m}.log" 2>&1
  echo "### $(date '+%F %T') SCORE-END   $m exit=$?" >> "$L/queue_progress.log"
done
echo "### $(date '+%F %T') SCORING DONE" >> "$L/queue_progress.log"
