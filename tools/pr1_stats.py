#!/usr/bin/env python3
"""Collect the pre-registered PR1 support-set statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import get_multi_frame_dataset
from src.dataset.base_two_modal_dataset import DatasetMode
from src.model.net import VideoFusion


LAYERS = ("e1", "dec1", "bt")
DATASET = "M3SVD"
CONFIG = "config/train/ivf-train.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", default="MAVFusion-M3SVD")
    parser.add_argument("--ckpt_path", default="latest")
    parser.add_argument("--base_data_dir", default="data2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="output/pr1")
    return parser.parse_args()


def seq_name(batch):
    path = batch["data_path_ls_dict"]["ir"][2][0]
    return Path(path).parts[-3]


def summarize_scores(scores, active_idx, k):
    scores = scores.flatten().float()
    active_idx = active_idx.flatten().long()
    n = int(scores.numel())
    sorted_scores = torch.sort(scores, descending=True).values
    boundary = sorted_scores[k - 1]
    tie_level = int(torch.count_nonzero(scores == boundary).item())
    tie_in = int(torch.count_nonzero(scores.index_select(0, active_idx) == boundary).item())
    tie_out = tie_level - tie_in
    active_scores = scores.index_select(0, active_idx)
    return {
        "N": n,
        "K": int(k),
        "A": int(torch.count_nonzero(scores > 0).item()),
        "A_lt_K": bool(torch.count_nonzero(scores > 0).item() < k),
        "wasted": float(torch.count_nonzero(active_scores == 0).item() / k),
        "n_sat": int(torch.count_nonzero(scores >= 1 - 1e-6).item()),
        "tie_boundary": bool(n > k and bool(sorted_scores[k - 1] == sorted_scores[k])),
        "tie_level": tie_level,
        "tie_in": tie_in,
        "tie_out": tie_out,
        "interchangeable": bool(tie_in > 0 and tie_out > 0),
    }


def leakage(record):
    dynamic = record["dynamic"].float()
    mask = record["mask"].float()
    output = record["out"].float()
    smooth = record["smooth"].float()
    zero_pixels = mask == 0
    gated = dynamic * mask
    denom = gated.abs().sum().item()
    gate_ratio = float((gated.abs() * zero_pixels).sum().item() / max(denom, 1e-12))
    smooth_denom = output.abs().sum().item()
    smooth_ratio = float((smooth.abs() * zero_pixels).sum().item() / max(smooth_denom, 1e-12))
    return gate_ratio, smooth_ratio


def build_loader(args):
    data_cfg = OmegaConf.load("config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml")
    frozen = json.loads(Path("output/pr1/frozen_split.json").read_text())
    selected_split = Path(data_cfg.csv_dir) / "_pr1_development.json"
    selected_split.write_text(json.dumps({"train": [], "test": frozen["development"]}, indent=2))
    data_cfg.split_filename = selected_split.name
    dataset = get_multi_frame_dataset(data_cfg, base_data_dir=args.base_data_dir, mode=DatasetMode.TEST)
    return dataset, selected_split


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, selected_split = build_loader(args)
    cfg = OmegaConf.load(CONFIG)
    model = VideoFusion(model_config=cfg).to(args.device).eval()
    checkpoint = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    model.set_recording(True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    frame_rows = []
    video_rows = defaultdict(lambda: defaultdict(list))
    leakage_rows = []
    with torch.no_grad():
        for batch in loader:
            seq = seq_name(batch)
            ir = batch["ir"].to(args.device)
            rgb = batch["rgb"].to(args.device)
            model.clear_pr1_records()
            model(ir, rgb)
            records = model.get_pr1_records()
            num_frames = len(records[LAYERS[0]]["selection"])
            for frame_offset in range(num_frames):
                per_layer = {}
                for layer in LAYERS:
                    selection = records[layer]["selection"]
                    if len(selection) != num_frames:
                        raise RuntimeError(f"Layer record count mismatch for {layer}")
                    record = selection[frame_offset]
                    stats = summarize_scores(record["scores"].cpu(), record["active_idx"].cpu(), record["k"])
                    per_layer[layer] = stats
                    for key, value in stats.items():
                        if key not in {"N", "K"}:
                            video_rows[seq][f"{layer}.{key}"].append(float(value))
                frame_index = len([row for row in frame_rows if row["video"] == seq])
                dec1_gates = records["dec1"]["gate"]
                if dec1_gates and frame_index < 20:
                    gate_ratio, smooth_ratio = leakage(dec1_gates[frame_offset])
                    leakage_rows.append({
                        "video": seq,
                        "frame": frame_index,
                        "gate_mismatch_ratio": gate_ratio,
                        "smooth_leakage_ratio": smooth_ratio,
                    })
                frame_rows.append({"video": seq, "frame": frame_index, "layers": per_layer})

    selected_split.unlink(missing_ok=True)
    by_video = []
    for video in sorted(video_rows):
        row = {"video": video}
        for key, values in sorted(video_rows[video].items()):
            row[key] = sum(values) / len(values)
        row["frames"] = len([item for item in frame_rows if item["video"] == video])
        by_video.append(row)

    payload = {
        "dataset": DATASET,
        "batch_size": 1,
        "layers": LAYERS,
        "decision_layer": "dec1",
        "videos": sorted(video_rows),
        "frames": frame_rows,
        "by_video": by_video,
        "leakage_check": leakage_rows,
    }
    (output_dir / "stats.json").write_text(json.dumps(payload, indent=2))
    fields = sorted({key for row in by_video for key in row})
    with (output_dir / "stats_by_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_video)
    print(json.dumps({"videos": len(by_video), "frames": len(frame_rows), "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
