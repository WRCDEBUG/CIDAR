#!/bin/bash
# The gate, motivated by a defect that only the real-VIDEO validation exposed:
# on clean VTMOT and HDO the ungated module loses 3.5-4.1%, because the descriptor
# is out of domain there and reports a degradation that is not present.  The gate
# lets the module abstain, supervised for free by the clean/degraded label.
# Then evaluate on the real video benchmarks, which is where the defect appeared.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D3="noise10,noise20,dark"
for i in $(seq 1 400); do
  [ -f output/pr1/stage2_ddc_temporal.json ] && [ -f output/pr1/stage2_temporal_nomix.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') GATE START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run gate_cross --dar_mode predicted --est_mode cross --dar_gate --train_degradations $D3 --steps 8000 --eval_every 4000

echo "### $(date '+%F %T') START xvideo_gate" >> "$L/queue_progress.log"
timeout 7200 $PY tools/crossvideo_degradation_eval.py --datasets VTMOT,HDO --window_stride 20 \
  --dar_ckpt output/pr1/stage2_gate_cross.pth --est_mode cross \
  --output output/pr1/crossvideo_gate.json > "$L/xvideo_gate.log" 2>&1
echo "### $(date '+%F %T') END   xvideo_gate exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') GATE DONE" >> "$L/queue_progress.log"
