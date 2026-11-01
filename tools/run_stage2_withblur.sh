#!/bin/bash
# Follow-up: blur turns out to be largely RECOVERABLE once there is enough
# training (oracle: +2.2% at 3000 steps -> +43.4% at 8000), which contradicts the
# earlier "blur is irreversible" conclusion and means the earlier decision to drop
# blur from training may have been premature.  Re-test with blur included, long.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
# start as soon as the currently running estimator run finishes, so this chain
# and the final chain run as two concurrent jobs (the throughput sweet spot on a
# single A100)
for i in $(seq 1 400); do
  [ -f output/pr1/stage2_noblur_auxreg.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') WITHBLUR START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }
run withblur_long    --dar_mode predicted --train_degradations $D4 --steps 8000 --eval_every 4000
run withblur_2stage_long --dar_mode predicted --dar_second --train_degradations $D4 --steps 8000 --eval_every 4000
echo "### $(date '+%F %T') WITHBLUR DONE" >> "$L/queue_progress.log"
