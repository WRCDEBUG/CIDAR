# CIDAR: Cross-Modal Degradation-Aware Feature Rectification for Infrared–Visible Video Fusion

Code and evaluation artifacts for the manuscript submitted to *Image and Vision
Computing*. CIDAR inserts zero-initialized, FiLM-conditioned residual rectifiers
between the shallow feature extractors and the temporal alignment of a
motion-guided infrared–visible video fusion network. The rectifiers are driven by
a learned five-dimensional cross-modal descriptor and trained by feature matching
to a frozen clean-input teacher, without fused ground truth.

## Repository layout

```
src/                 model and training code
tools/               training, evaluation, baseline and figure scripts
config/              YAML configuration files
train.py, test.py    entry points
data_split/          frozen five-sequence splits
results/             per-window QABF results and the evaluation manifest
```

## Environment

The reported experiments used Python 3.10, PyTorch 2.5.1+cu121 and CUDA 12.1 on
an NVIDIA A100. Install PyTorch for your CUDA version, then the usual packages:

```bash
pip install numpy opencv-python scipy pyyaml tqdm
```

Some scripts need two interpreters (the Mamba-based baselines require a separate
environment). They read the interpreter from `PY`, `PY_MAIN` and `PY_MAMBA`;
export whichever apply:

```bash
export PY=/path/to/envs/videofusion/bin/python
export PY_MAIN="$PY"
export PY_MAMBA=/path/to/envs/ifcaf/bin/python
```

## Data

The three benchmarks are public and are **not** redistributed here:

- **M3SVD** (VideoFusion, CVPR 2026) — 30 sequences
- **HDO** (RCVS, IEEE Trans. Multimedia 2024) — 27 sequences in our local copy
- **VTMOT** (Zhu et al., Pattern Recognition 2025) — 120 sequences

Expected layout:

```
data2/<DATASET>/<sequence>/{infrared,visible}/*.jpg
```

Download each benchmark from its original source and place it under `data2/`.
Set `TB_ROOT` and `THIRD_ROOT` if the third-party baseline outputs live outside
`third_party/`, and `LLVIP_ROOT` for the optional LLVIP night-image check.

## Training and evaluation

Entry points:

```bash
python train.py -h
python tools/stage1_finetune.py -h
python tools/stage2_dar.py -h
python tools/crossvideo_degradation_eval.py -h
```

The metric contract used in the paper is `tools/contract_metrics.m` (MATLAB;
QABF on [0, 255]). The real-video protocol uses five-frame windows, scores the
center frame, uses stride 20 and degradation seed 4242 on the frozen
five-sequence holdout.

## Results

`results/dar_real_video_stride20_per_window.json` contains 1375 rows
(275 windows x 5 conditions: clean, noise sigma=10, noise sigma=20, low-light,
5x5 blur) with baseline and CIDAR QABF, the delta and the win flag.
`results/dar_real_video_stride20_summary.json` aggregates them and
`results/dar_real_video_stride20_manifest.json` records the command, git commit,
checkpoint SHA-256, stride, seed and runtime.

## License

MIT. This repository contains code derived from
[MAVFusion](https://github.com/lixilai/MAVFusion); the original MAVFusion
copyright is retained in `LICENSE`.

## Citation

The manuscript is under review. Until it is published, please cite:

```bibtex
@misc{wang2026cidar,
  title  = {CIDAR: Cross-Modal Degradation-Aware Feature Rectification for
            Infrared--Visible Video Fusion},
  author = {Wang, Rongchao and Zhou, Zhaofa and Zhang, Zhili},
  year   = {2026},
  note   = {Manuscript under review}
}
```
