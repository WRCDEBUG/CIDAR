#!/bin/bash
# Prototype conditioning, motivated by the sharpest measurement in the project:
#   oracle descriptor (DISCRETE, one constant vector per degradation) -> blur +43.4%
#   continuous descriptor with the same information (probe 0.873)     -> blur +0.4%
# Identifiability is not the bottleneck; the FORM of the conditioning is.  A soft
# prototype assignment gives the rectifier an almost-discrete condition while
# keeping the estimator differentiable.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"
for i in $(seq 1 600); do
  [ -f output/pr1/stage2_temporal_nomix.json ] && break
  sleep 60
done
echo "### $(date '+%F %T') VQ START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run vq_temporal --dar_mode predicted --est_mode crosstemporal --dar_vq 8  --train_degradations $D4 --steps 8000 --eval_every 4000
run vq_spec     --dar_mode predicted --est_mode crossspec    --dar_vq 8  --train_degradations $D4 --steps 8000 --eval_every 4000
run vq_temporal_k16 --dar_mode predicted --est_mode crosstemporal --dar_vq 16 --train_degradations $D4 --steps 8000 --eval_every 4000
echo "### $(date '+%F %T') VQ DONE" >> "$L/queue_progress.log"
