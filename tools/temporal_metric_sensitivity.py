#!/usr/bin/env python3
"""Compare the sensitivity of MS2R against a warping-error temporal metric.

MS2R was measured to be blind to design-level changes: perturbing the temporal
aggregation of LightweightAAM moved it by <= 0.0004, and adding sigma=8/255 noise
moved it by 0.0000, while substituting a distant frame moved it by +0.34.  A
metric that only reacts to gross corruption cannot be the target of a temporal
method.

This script evaluates a candidate replacement on exactly the same perturbations:

    warping error  = mean | F_t - warp(F_{t-1}, flow(F_{t-1} -> F_t)) |
    tOF            = same, but the warping flow is taken from the IR reference
                     clip's motion instead of the generated clip's own flow

and reports both metrics for:

    baseline / uniform / current_heavy / neighbour_only   (model perturbations)
    true / swap_neighbours / distant_frame / add_noise    (synthetic corruptions)

so the two metrics can be compared head to head on the same frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.util import metric  # noqa: E402

DATASET_YAML = {
    "M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
    "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
    "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml",
}
EXP_PATH = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}

VARIANTS = {
    "baseline": None,
    "uniform": (0.0, 0.0, 0.0),
    "current_heavy": (-3.0, 3.0, -3.0),
    "neighbour_only": (6.0, 0.0, 0.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--num_sequences", type=int, default=1)
    p.add_argument("--window_stride", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="output/pr1/temporal_metric_sensitivity.json")
    return p.parse_args()


def build_loader(args, sequences):
    data_cfg = OmegaConf.load(DATASET_YAML[args.dataset])
    split_name = "_temporal_metric_sensitivity.json"
    (Path(data_cfg.csv_dir) / split_name).write_text(json.dumps({"train": [], "test": sequences}))
    data_cfg.split_filename = split_name
    dataset = get_multi_frame_dataset(data_cfg, base_data_dir="data2", mode=DatasetMode.TEST)
    subset = Subset(dataset, list(range(0, len(dataset), args.window_stride)))
    return DataLoader(subset, batch_size=1, shuffle=False, num_workers=0), Path(data_cfg.csv_dir) / split_name


def set_temp_weight(model, logits):
    if logits is None:
        return
    tensor = torch.tensor(logits, dtype=torch.float32).view(1, 1, 3, 1, 1)
    for aam in (model.aam_ir, model.aam_vi):
        with torch.no_grad():
            aam.temp_weight.copy_(tensor.to(aam.temp_weight.device))


def gray(x):
    return x.mean(dim=1, keepdim=True)


def warping_error(flow_net, clip, flow_from=None):
    """clip: [1,3,C,H,W] on device, values in [0,1]. Returns mean |F_t - warp(F_{t-1})|."""
    src = flow_from if flow_from is not None else clip
    errs = []
    for t in (1, 2):
        prev = clip[:, t - 1]
        cur = clip[:, t]
        flow = flow_net(src[:, t - 1] * 255.0, src[:, t] * 255.0)["final"]
        grid = torch.stack(torch.meshgrid(
            torch.arange(cur.shape[-2], device=cur.device, dtype=torch.float32),
            torch.arange(cur.shape[-1], device=cur.device, dtype=torch.float32),
            indexing="ij")[::-1], dim=-1).unsqueeze(0)
        norm = torch.empty_like(flow)
        norm[..., 0] = flow[..., 0] * (2.0 / (cur.shape[-1] - 1))
        norm[..., 1] = flow[..., 1] * (2.0 / (cur.shape[-2] - 1))
        warped = torch.nn.functional.grid_sample(
            prev, grid + norm.permute(0, 2, 3, 1), align_corners=True, padding_mode="border")
        errs.append(float((gray(cur) - gray(warped)).abs().mean().item()))
    return float(np.mean(errs))


def main():
    args = parse_args()
    device = torch.device(args.device)
    manifest = json.loads((REPO / args.manifest).read_text())
    sequences = manifest["sequence_sets"][manifest["datasets"][args.dataset]["default_set"]]["sequences"]
    sequences = sequences[: args.num_sequences]

    model_cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = REPO / "output" / EXP_PATH[args.dataset] / "checkpoint" / "latest" / "model.pth"
    model = VideoFusion(model_config=model_cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.requires_grad_(False)

    loader, split_path = build_loader(args, sequences)
    ms2r = metric.MS2R_Evaluator(device=str(device))

    keys = list(VARIANTS) + ["swap_neighbours", "distant_frame", "add_noise"]
    acc = {k: {"ms2r": 0.0, "warp": 0.0, "tof": 0.0} for k in keys}
    n = 0
    try:
        with torch.no_grad():
            for batch in loader:
                ir = batch["ir"].to(device)
                rgb = batch["rgb"].to(device)
                r1 = torch.round(ir[:, 1:4].clamp(0, 1) * 255)
                r2 = torch.round(rgb[:, 1:4].clamp(0, 1) * 255)
                d_ref1 = ms2r.compute_flow(r1[:, 1], r1[:, 2]) - ms2r.compute_flow(r1[:, 0], r1[:, 1])
                d_ref2 = ms2r.compute_flow(r2[:, 1], r2[:, 2]) - ms2r.compute_flow(r2[:, 0], r2[:, 1])
                last = None
                for name, logits in VARIANTS.items():
                    set_temp_weight(model, logits)
                    pred = model(ir, rgb).clamp(0, 1)
                    if name == "baseline":
                        last = pred.clone()
                    g = torch.round(pred * 255)
                    d_gen = ms2r.compute_flow(g[:, 1], g[:, 2]) - ms2r.compute_flow(g[:, 0], g[:, 1])
                    acc[name]["ms2r"] += float((0.5 * ((d_gen - d_ref1).abs().mean()
                                                      + (d_gen - d_ref2).abs().mean())).item())
                    acc[name]["warp"] += warping_error(ms2r.flow_net, pred)
                    acc[name]["tof"] += warping_error(ms2r.flow_net, pred, flow_from=ir[:, 1:4] / 255.0)

                # synthetic corruptions on the baseline clip
                synth = {
                    "swap_neighbours": last.flip(1),
                    "distant_frame": torch.cat([last[:, 0:1], last[:, 1:2], last[:, 0:1]], dim=1),
                    "add_noise": (last + torch.randn_like(last) * (8.0 / 255.0)).clamp(0, 1),
                }
                for name, clip in synth.items():
                    g = torch.round(clip * 255)
                    d_gen = ms2r.compute_flow(g[:, 1], g[:, 2]) - ms2r.compute_flow(g[:, 0], g[:, 1])
                    acc[name]["ms2r"] += float((0.5 * ((d_gen - d_ref1).abs().mean()
                                                      + (d_gen - d_ref2).abs().mean())).item())
                    acc[name]["warp"] += warping_error(ms2r.flow_net, clip)
                    acc[name]["tof"] += warping_error(ms2r.flow_net, clip, flow_from=ir[:, 1:4] / 255.0)
                n += 1
    finally:
        split_path.unlink(missing_ok=True)
        set_temp_weight(model, None)

    rows = {k: {m: acc[k][m] / n for m in ("ms2r", "warp", "tof")} for k in keys}
    base = rows["baseline"]
    payload = {"dataset": args.dataset, "sequences": sequences, "windows": n,
               "window_stride": args.window_stride, "metrics": rows,
               "delta_vs_baseline": {k: {m: rows[k][m] - base[m] for m in ("ms2r", "warp", "tof")}
                                     for k in keys}}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== temporal metric sensitivity · {args.dataset} · {n} windows ===")
    print(f"{'variant':20} {'MS2R':>9} {'dMS2R':>8} | {'warpErr':>9} {'dWarp':>9} | {'tOF':>9} {'dtOF':>9}")
    for k in keys:
        r, d = rows[k], payload["delta_vs_baseline"][k]
        print(f"{k:20} {r['ms2r']:9.4f} {d['ms2r']:+8.4f} | {r['warp']:9.5f} {d['warp']:+9.5f} | "
              f"{r['tof']:9.5f} {d['tof']:+9.5f}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
