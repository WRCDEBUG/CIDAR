#!/usr/bin/env python3
"""Compute UniVF VIF / MS2R from the *saved* fused images instead of raw tensors.

Why
  evaluate_metrics.py (and therefore the UniVF reference harness) scores the raw
  network output tensor.  The MATLAB half of the pipeline scores the JPEGs in
  eval_data/, which are produced by src/util/io.pred_2_8bit with
  ``inherit_vi_chroma=True`` -- the network's 3-channel output keeps its luma but
  takes the visible frame's Cb/Cr.  VIF is a 3-channel metric, so the two halves
  are not scoring the same object, and neither is the paper necessarily.

  This script closes that gap: it reads exactly the files the MATLAB metrics
  read, applies the same frame sampling (step 5), and calls the same UniVF
  reference implementations.

Inputs   eval_data/<dataset>/{ir,vis,fused}/<seq>_<frame>.jpg
Methods  src/util/metric.py  Metric_VIF  and  MS2R_Evaluator.compute_ms2r_metric
         (UniVF / VF-Bench reference, https://github.com/Zhaozixiang1228/VF-Bench)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.util import metric  # noqa: E402

IMAGE_EXT = (".jpg", ".png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
    p.add_argument("--sample_step", type=int, default=5)
    p.add_argument("--metrics", default="vif,ms2r")
    p.add_argument("--eval_root", default=None,
                   help="override the manifest eval_root, e.g. output/pr1/deploy/fast")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="output/pr1/python")
    return p.parse_args()


def load_rgb(path: Path, size=None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None and (img.height, img.width) != size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # [H,W,3] in [0,255]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def resolve(directory: Path, stem: str):
    for ext in IMAGE_EXT:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    manifest = json.loads((REPO / args.manifest).read_text())
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {m.strip() for m in args.metrics.split(",") if m.strip()}

    ms2r = metric.MS2R_Evaluator(device=str(device)) if "ms2r" in want else None
    results = []

    for ds in args.datasets.split(","):
        spec = manifest["datasets"][ds]
        set_name = spec["default_set"]
        sequences = manifest["sequence_sets"][set_name]["sequences"]
        root = REPO / (args.eval_root or spec["eval_root"])
        ir_dir, vis_dir, fused_dir = root / "ir", root / "vis", root / "fused"

        vif_vals, ms2r_vals = [], []
        for seq in sequences:
            frames = sorted(p for p in fused_dir.glob(f"{seq}_*") if p.suffix.lower() in IMAGE_EXT)
            frames = frames[:: args.sample_step]
            for f in frames:
                stem = f.stem
                ir_p, vis_p = resolve(ir_dir, stem), resolve(vis_dir, stem)
                if ir_p is None or vis_p is None:
                    continue
                fused = load_rgb(f).to(device)
                size = (fused.shape[-2], fused.shape[-1])
                ir = load_rgb(ir_p, size).to(device)
                vis = load_rgb(vis_p, size).to(device)
                if "vif" in want:
                    vif_vals.append(float(metric.Metric_VIF(fused, ir, vis).mean().item()))
            if ms2r is not None and len(frames) >= 3:
                for i in range(1, len(frames) - 1):
                    trio = frames[i - 1: i + 2]
                    ir_trio = [resolve(ir_dir, p.stem) for p in trio]
                    vis_trio = [resolve(vis_dir, p.stem) for p in trio]
                    if any(x is None for x in ir_trio + vis_trio):
                        continue
                    g = torch.cat([load_rgb(p) for p in trio], dim=0)
                    size = (g.shape[-2], g.shape[-1])
                    r1 = torch.cat([load_rgb(p, size) for p in ir_trio], dim=0)
                    r2 = torch.cat([load_rgb(p, size) for p in vis_trio], dim=0)
                    clip = lambda t: t.unsqueeze(0).to(device)  # noqa: E731  [3,C,H,W] -> [1,3,C,H,W]
                    ms2r_vals.append(float(
                        ms2r.compute_ms2r_metric(clip(g), clip(r1), clip(r2)).mean().item()
                    ))

        entry = {
            "dataset": ds, "set": set_name,
            "source": "saved images in eval_data (same objects as the MATLAB metrics)",
            "vif_frames": len(vif_vals),
            "Metric_VIF": float(np.mean(vif_vals)) if vif_vals else None,
            "ms2r_windows": len(ms2r_vals),
            "Metric_MS2R": float(np.mean(ms2r_vals)) if ms2r_vals else None,
            "paper": {"VIF": spec["paper_table1"]["VIF"], "MS2R": spec["paper_table1"]["MS2R"]},
        }
        results.append(entry)
        print(f"[{ds}] VIF={entry['Metric_VIF']} (paper {entry['paper']['VIF']})  "
              f"MS2R={entry['Metric_MS2R']} (paper {entry['paper']['MS2R']})")

    (out_dir / "univf_from_images.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwritten {out_dir / 'univf_from_images.json'}")


if __name__ == "__main__":
    main()
