#!/bin/bash
# Run the third-party fusion baselines over the frozen five-sequence holdout of
# all three datasets, saving every fused frame so the results can be re-scored
# later without re-running the networks.  Resumable by construction.
#
# Usage:  tools/run_baselines.sh [method ...]
#         tools/run_baselines.sh tardal ignet tdfusion fusionmamba s4fusion coconet
#
# Two interpreters are needed:
#   videofusion - everything except the Mamba methods
#   ifcaf       - FusionMamba / S4Fusion (mamba_ssm + selective_scan_cuda)
# Up to $JOBS methods run concurrently; each method is a separate process so a
# crash only costs that method's remaining frames (existing output is skipped).
cd "$(dirname "$0")/.." || exit 1
L=output/pr1/stage2_logs
mkdir -p "$L"
PY_MAIN="${PY_MAIN:-python}"
PY_MAMBA="${PY_MAMBA:-python}"
JOBS=${JOBS:-3}
DATASETS=${DATASETS:-M3SVD,HDO,VTMOT}
STEP=${STEP:-5}

methods=("$@")
if [ ${#methods[@]} -eq 0 ]; then
  methods=(tardal ignet tdfusion fusionmamba s4fusion coconet)
fi

py_for() {
  case "$1" in
    fusionmamba|s4fusion) echo "$PY_MAMBA" ;;
    *) echo "$PY_MAIN" ;;
  esac
}

run_one() {
  local m=$1 py
  py=$(py_for "$m")
  echo "### $(date '+%F %T') START baseline_$m ($py)" >> "$L/queue_progress.log"
  timeout 43200 "$py" tools/bench_baselines.py --method "$m" \
      --datasets "$DATASETS" --sample_step "$STEP" \
      > "$L/baseline_${m}.log" 2>&1
  local rc=$?
  echo "### $(date '+%F %T') END   baseline_$m exit=$rc" >> "$L/queue_progress.log"
  return $rc
}

echo "### $(date '+%F %T') BASELINES START (${methods[*]}, jobs=$JOBS, data=$DATASETS)" >> "$L/queue_progress.log"

# simple bounded job pool
pids=()
for m in "${methods[@]}"; do
  if [ ${#pids[@]} -ge "$JOBS" ]; then
    wait -n 2>/dev/null || wait "${pids[0]}"
    # drop finished entries
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  fi
  run_one "$m" &
  pids+=($!)
  sleep 2
done
wait
echo "### $(date '+%F %T') BASELINES DONE" >> "$L/queue_progress.log"
