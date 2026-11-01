#!/bin/bash
# temporal_nomix crashed on a bug that is now fixed, and the gate chain was
# waiting for a json that never appeared.  Re-run both, gate first because it
# addresses the clean-video defect that the real-video test exposed.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D3="noise10,noise20,dark"; D4="noise10,noise20,dark,blur"
echo "### $(date '+%F %T') AFTER-DDC START" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START gate_cross" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag gate_cross --dar_mode predicted --est_mode cross \
  --dar_gate --train_degradations $D3 --steps 8000 --eval_every 4000 > "$L/gate_cross.log" 2>&1
echo "### $(date '+%F %T') END   gate_cross exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START xvideo_gate" >> "$L/queue_progress.log"
timeout 7200 $PY tools/crossvideo_degradation_eval.py --datasets VTMOT,HDO --window_stride 20 \
  --dar_ckpt output/pr1/stage2_gate_cross.pth --est_mode cross \
  --output output/pr1/crossvideo_gate.json > "$L/xvideo_gate.log" 2>&1
echo "### $(date '+%F %T') END   xvideo_gate exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') START temporal_nomix" >> "$L/queue_progress.log"
timeout 28800 $PY tools/stage2_dar.py --tag temporal_nomix --dar_mode predicted --est_mode crosstemporal \
  --train_degradations $D4 --steps 8000 --eval_every 4000 > "$L/temporal_nomix.log" 2>&1
echo "### $(date '+%F %T') END   temporal_nomix exit=$?" >> "$L/queue_progress.log"
echo "### $(date '+%F %T') AFTER-DDC DONE" >> "$L/queue_progress.log"
