#!/bin/bash
cd "$(dirname "$0")"

for ds in HDO M3SVD VTMOT; do
    echo "=== MS2R: $ds ==="
    "${PY:-python}" evaluate_metrics.py \
      --dataset_name $ds \
      --exp_path MAVFusion-$ds \
      --base_data_dir ./data2 \
      --num_test_sequences 1000 \
      --num_workers 32 \
      --device cuda:0 \
      --include_ms2r
done