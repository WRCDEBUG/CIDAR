#!/usr/bin/env python3
"""Real-VIDEO cross-dataset degradation validation.

Why this file exists: the earlier real-data check used LLVIP, which is a dataset
of image pairs that we stitched into pseudo-sequences by assuming consecutive file
numbers are consecutive video frames.  For a video-fusion paper that is the wrong
instrument.  This script runs the same well-posed protocol -- real frames, a
synthetic degradation added on top, quality measured against the CLEAN sources --
on the project's own real video benchmarks (M3SVD / HDO / VTMOT), which differ in
sensor, resolution, scene and capture conditions.

Protocol
  for each dataset and each degradation:
     baseline (released weights) and DAR (given checkpoint) both see the degraded
     frames, and Q_AB/F is measured against the CLEAN sources, so
     recovery = (Q_dar - Q_base) / (Q_clean - Q_base) is well defined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.util import metric  # noqa: E402
from tools.stage2_dar import degrade  # noqa: E402

YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
        "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
        "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}
DEGS = ("none", "noise10", "noise20", "dark", "blur")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="VTMOT,HDO")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--dar_ckpt", default="output/pr1/stage2_cross_noblur_long.pth")
    p.add_argument("--est_mode", default="cross")
    p.add_argument("--window_stride", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/crossvideo_degradation.json")
    return p.parse_args()


def build_loader(dataset, man, stride):
    key = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test",
           "VTMOT": "VTMOT_frozen5_test"}[dataset]
    seqs = man["sequence_sets"][key]["sequences"]
    dc = OmegaConf.load(YAML[dataset])
    nm = f"_xv_{dataset}.json"
    (Path(dc.csv_dir) / nm).write_text(json.dumps({"train": [], "test": seqs}))
    dc.split_filename = nm
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    idx = list(range(0, len(ds), stride))
    return DataLoader(Subset(ds, idx), batch_size=1, shuffle=False), len(idx), Path(dc.csv_dir) / nm


def main():
    args = parse_args()
    device = torch.device(args.device)
    man = json.loads((REPO / args.manifest).read_text())
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    out = {}
    for dataset in [d for d in args.datasets.split(",") if d]:
        loader, n_win, split = build_loader(dataset, man, args.window_stride)
        ckpt = torch.load(REPO / "output" / EXP[dataset] / "checkpoint/latest/model.pth",
                          map_location=device)
        base = VideoFusion(model_config=cfg).to(device).eval()
        base.load_state_dict(ckpt, strict=False)
        dar = VideoFusion(model_config=cfg,
                          dar={"mode": "predicted", "est_mode": args.est_mode}).to(device).eval()
        dar.load_state_dict(ckpt, strict=False)
        cp = REPO / args.dar_ckpt
        if cp.exists():
            dar.load_state_dict(torch.load(cp, map_location=device), strict=False)
        print(f"=== {dataset}: {n_win} windows, DAR={cp.name} (est_mode={args.est_mode}) ===", flush=True)
        rows = {}
        try:
            with torch.no_grad():
                for d in DEGS:
                    qb = qd = 0.0
                    wins = 0
                    n = 0
                    for b in loader:
                        ir0 = b["ir"].to(device)
                        rgb0 = b["rgb"].to(device)
                        gen = torch.Generator(device=device).manual_seed(4242)
                        ir = ir0 if d == "none" else degrade(ir0, d, gen)
                        rgb = rgb0 if d == "none" else degrade(rgb0, d, gen)
                        ir_ref = torch.round(ir0[:, 2] * 255)
                        vi_ref = torch.round(rgb0[:, 2] * 255)
                        ob = base(ir, rgb).clamp(0, 1)[:, 1]
                        od = dar(ir, rgb).clamp(0, 1)[:, 1]
                        qb_ = float(metric.Metric_Qabf(torch.round(ob * 255), ir_ref, vi_ref).mean())
                        qd_ = float(metric.Metric_Qabf(torch.round(od * 255), ir_ref, vi_ref).mean())
                        qb += qb_
                        qd += qd_
                        wins += int(qd_ > qb_)
                        n += 1
                    rows[d] = {"baseline": qb / n, "dar": qd / n,
                               "delta_pct": (qd / qb - 1) * 100 if qb else None,
                               "wins": wins, "n": n}
                    print(f"  {d:8} base {rows[d]['baseline']:.4f}  DAR {rows[d]['dar']:.4f}  "
                          f"{rows[d]['delta_pct']:+6.2f}%  wins {wins}/{n}", flush=True)
        finally:
            split.unlink(missing_ok=True)
        out[dataset] = rows
    path = REPO / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
