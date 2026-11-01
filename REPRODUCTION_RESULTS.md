# MAVFusion Reproduction Results

## Evaluation setup

- Environment: Conda environment `videofusion` with Python 3.10 and PyTorch 2.5.1+cu121.
- Device: NVIDIA A100 80 GB.
- Checkpoint: `output/<experiment>/checkpoint/latest/model.pth`.
- Evaluation script: `evaluate_metrics.py`.
- Sequence selection: five sequences sampled with `seed=2025` from each prepared test split.
- The repository's available image metrics are `Metric_VIF`, `Metric_SSIM`, `Metric_MI`, and `Metric_Qabf`.
- VTMOT requires `batch_size=1` because its sequences contain different spatial resolutions.

## Commands

```bash
python evaluate_metrics.py \
  --dataset_name M3SVD --exp_path MAVFusion-M3SVD \
  --num_test_sequences 5 --seed 2025 --batch_size 4 \
  --base_data_dir /home/ubuntu/Code/datasets --num_workers 4 --device cuda

python evaluate_metrics.py \
  --dataset_name HDO --exp_path MAVFusion-HDO \
  --num_test_sequences 5 --seed 2025 --batch_size 1 \
  --base_data_dir /home/ubuntu/Code/datasets --num_workers 4 --device cuda

python evaluate_metrics.py \
  --dataset_name VTMOT --exp_path MAVFusion \
  --num_test_sequences 5 --seed 2025 --batch_size 1 \
  --base_data_dir /home/ubuntu/Code/datasets --num_workers 4 --device cuda
```

## Results

Results are frame-window averages over the selected sequences. The `samples` column counts five-frame inference windows, not raw individual image files.

| Dataset | Selected sequences | Samples | VIF | SSIM | MI | Qabf |
|---|---|---:|---:|---:|---:|---:|
| M3SVD | `0117_1605`, `0111_1803`, `0118_1904`, `1208_1711`, `0115_1834` | 3930 | 0.457664 | 0.638369 | 2.562323 | 0.736008 |
| HDO | `seq_25`, `seq_11`, `seq_3`, `seq_9`, `seq_23` | 1350 | 0.494449 | 0.572164 | 2.247477 | 0.748497 |
| VTMOT | `photo-0310-53`, `LasHeR-056`, `photo-0318-37`, `wurenji-0303-09`, `photo-0310-35` | 4128 | 0.412151 | 0.685075 | 2.459109 | 0.648148 |

CSV outputs:

- `output/MAVFusion-M3SVD/metrics_M3SVD.csv`
- `output/MAVFusion-HDO/metrics_HDO.csv`
- `output/MAVFusion/metrics_VTMOT.csv`

## Difference from paper Table 1

The paper reports eight metrics: `QG`, `QM`, `QP`, `QS`, `QCB`, `VIF`, `QAB/F`, and `MS2R`. The current repository implements only the four image metrics listed above for this evaluation script. Its `MS2R` implementation is present but does not document enough detail to establish that it is identical to the paper protocol, and it was not included in these runs.

Therefore, these numbers are reproducible repository-level checkpoint results, not an exact numerical reproduction of Table 1. The paper's reported MAVFusion reference values are:

| Dataset | QG | QM | QP | QS | QCB | VIF | QAB/F | MS2R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M3SVD | 0.6897 | 1.1544 | 0.6122 | 0.8549 | 0.5535 | 0.3814 | 0.7550 | 0.1058 |
| HDO | 0.6629 | 0.9996 | 0.5660 | 0.8464 | 0.5578 | 0.3359 | 0.6837 | 0.9826 |
| VTMOT | 0.6325 | 0.9741 | 0.4947 | 0.8504 | 0.4036 | 0.3885 | 0.7182 | 0.8301 |

The PDF describes 30 M3SVD, 24 HDO, and 90 VTMOT sequences. The prepared downloads contain 30, 27, and 120 sequences respectively, so the HDO and VTMOT results cannot be treated as a strict dataset-identical reproduction without obtaining the paper's exact subset and metric implementations.
