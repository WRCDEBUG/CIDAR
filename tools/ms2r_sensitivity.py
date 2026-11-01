#!/usr/bin/env python3
"""Is MS2R sensitive enough to serve as a target metric for a temporal method?

A method paper does not need MS2R to match the paper's absolute value -- it needs
a metric that responds reliably to real changes in temporal behaviour, compared
against a local baseline on identical frames.

`LightweightAAM` aggregates the three warped frames with a single softmax over
`temp_weight`, i.e. the same three temporal weights at every pixel of every
frame.  This script perturbs only that parameter (no retraining, no other
change) and measures whether MS2R moves:

    baseline      the trained weights
    uniform       logits [0,0,0]        -> (1/3, 1/3, 1/3)
    current_heavy logits [-3,3,-3]      -> ~(0.05, 0.91, 0.05)
    prev_only     logits [6,0,0]        -> ~(0.997, 0.003, 0.003)

Reference differential flows depend only on the source clips, so they are
computed once per window and reused across variants.

For VTMOT/HDO the per-sequence resolution differs, so batch is always 1.
"""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--num_sequences", type=int, default=2)
    p.add_argument("--window_stride", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="output/pr1/ms2r_sensitivity.json")
    return p.parse_args()


def build_loader(args, sequences):
    data_cfg = OmegaConf.load(DATASET_YAML[args.dataset])
    split_name = "_ms2r_sensitivity.json"
    (Path(data_cfg.csv_dir) / split_name).write_text(json.dumps({"train": [], "test": sequences}))
    data_cfg.split_filename = split_name
    dataset = get_multi_frame_dataset(data_cfg, base_data_dir="data2", mode=DatasetMode.TEST)
    subset = Subset(dataset, list(range(0, len(dataset), args.window_stride)))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    return loader, Path(data_cfg.csv_dir) / split_name


def set_temp_weight(model, logits):
    """Set the temporal softmax logits on every LightweightAAM. None = leave as trained."""
    if logits is None:
        return
    tensor = torch.tensor(logits, dtype=torch.float32).view(1, 1, 3, 1, 1)
    for aam in (model.aam_ir, model.aam_vi):
        with torch.no_grad():
            aam.temp_weight.copy_(tensor.to(aam.temp_weight.device))


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
    assert model.fusion_net.dec1.dynamic_branch.record_enabled is False

    loader, split_path = build_loader(args, sequences)
    ms2r = metric.MS2R_Evaluator(device=str(device))

    totals = {k: {"ms2r": 0.0, "flicker": 0.0} for k in VARIANTS}
    count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                ir = batch["ir"].to(device)
                rgb = batch["rgb"].to(device)
                r1 = torch.round(ir[:, 1:4].clamp(0, 1) * 255)
                r2 = torch.round(rgb[:, 1:4].clamp(0, 1) * 255)
                # reference differential flows: independent of the variant
                d_ref1 = ms2r.compute_flow(r1[:, 1], r1[:, 2]) - ms2r.compute_flow(r1[:, 0], r1[:, 1])
                d_ref2 = ms2r.compute_flow(r2[:, 1], r2[:, 2]) - ms2r.compute_flow(r2[:, 0], r2[:, 1])

                for name, logits in VARIANTS.items():
                    set_temp_weight(model, logits)
                    pred = model(ir, rgb)
                    g = torch.round(pred.clamp(0, 1) * 255)
                    d_gen = ms2r.compute_flow(g[:, 1], g[:, 2]) - ms2r.compute_flow(g[:, 0], g[:, 1])
                    value = 0.5 * ((d_gen - d_ref1).abs().mean() + (d_gen - d_ref2).abs().mean())
                    totals[name]["ms2r"] += float(value.item())
                    # independent flicker proxy: mean |F_t - F_{t-1}| over the clip
                    totals[name]["flicker"] += float((pred[:, 1:] - pred[:, :-1]).abs().mean().item())
                count += 1
                if count % 10 == 0:
                    print(f"  {count} windows ...", flush=True)
    finally:
        split_path.unlink(missing_ok=True)

    set_temp_weight(model, None)
    results = {}
    for name in VARIANTS:
        results[name] = {
            "MS2R": totals[name]["ms2r"] / count,
            "flicker_mean_abs_frame_diff": totals[name]["flicker"] / count,
        }
    base = results["baseline"]
    for name in VARIANTS:
        results[name]["MS2R_delta_vs_baseline"] = results[name]["MS2R"] - base["MS2R"]
        results[name]["MS2R_relative"] = (results[name]["MS2R"] / base["MS2R"] - 1.0) if base["MS2R"] else None

    payload = {"dataset": args.dataset, "sequences": sequences, "windows": count,
               "window_stride": args.window_stride, "variants": results,
               "interpretation": "if MS2R cannot separate these clearly different temporal "
                                 "aggregations, it cannot serve as a target metric for a temporal method"}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== MS2R sensitivity · {args.dataset} · {count} windows ===")
    print(f"{'variant':16} {'MS2R':>9} {'delta':>9} {'rel':>8} {'flicker':>9}")
    for name in VARIANTS:
        r = results[name]
        print(f"{name:16} {r['MS2R']:9.4f} {r['MS2R_delta_vs_baseline']:+9.4f} "
              f"{(r['MS2R_relative'] or 0)*100:7.1f}% {r['flicker_mean_abs_frame_diff']:9.5f}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
