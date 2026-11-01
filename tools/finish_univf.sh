#!/bin/bash
# Wait for the two UniVF (VF-Bench) runs to finish, then flatten their
# [ir | vis | fused] triptychs into the contract layout with the fused panel.
#
# The runs write per-sequence PNGs under
#   VF-Bench-main/output/UniVF-IVF/test_results/latest/eval_visual/IVF-<ds>/<seq>/
# and the contract needs
#   output/pr1/fusion_results/univf/<ds>/fused/<seq>_<frame>.jpg
# plus matching ir/vis links, which tools/stage_method_outputs.py provides.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
VB="${VB:-/path/to/VF-Bench-main/output/UniVF-IVF/test_results/latest/eval_visual}"
L=output/pr1/stage2_logs

for ds in M3SVD HDO; do
  echo "### $(date '+%F %T') UNIVF-STAGE-WAIT $ds" >> "$L/queue_progress.log"
  while pgrep -f "test.py --task_name IVF --dataset_name ${ds} " > /dev/null 2>&1; do
    sleep 60
  done
  n=$(find "$VB/IVF-$ds" -name '*.png' 2>/dev/null | wc -l)
  echo "### $(date '+%F %T') UNIVF-STAGE $ds ($n png)" >> "$L/queue_progress.log"
  if [ "$n" -gt 0 ]; then
    $PY tools/stage_method_outputs.py --method univf --dataset "$ds" \
        --src "$VB/IVF-$ds" --panel right >> "$L/stage_univf_${ds}.log" 2>&1
  fi
done
echo "### $(date '+%F %T') UNIVF-STAGE DONE" >> "$L/queue_progress.log"
