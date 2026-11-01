#!/usr/bin/env python3
"""End-to-end comparison of the fixed budget vs the motion-support budget.

Runs the real deployment path used by test.py:

    dataset (5-frame windows) -> model(ir, vi) -> pred_2_8bit -> JPEG q95

for both budget policies, measures the forward latency with CUDA events exactly
as test.py does (skipping the first five batches), and writes a fused_result
directory per policy plus a report.  The saved JPEGs are what the metric
contract is then applied to, so the comparison is on the published artefacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.util.io import pred_2_8bit, save_image  # noqa: E402
from tools.support_stats import LAYERS  # noqa: E402

DATASET_YAML = {
    "M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="M3SVD")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--set", dest="set_name", default="frozen5_test")
    p.add_argument("--base_data_dir", default="data2")
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--ckpt_path", default="latest")
    p.add_argument("--limit_batches", type=int, default=0, help="0 = no limit")
    p.add_argument("--batch_stride", type=int, default=1,
                   help="process every n-th window; both modes always see the identical set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="output/pr1/e2e")
    p.add_argument("--modes", default="fixed,support")
    p.add_argument("--selector_mode", default=None,
                   help="optional PatchSelector mode applied to --selector_layer "
                        "(legacy | tie_aux | tie_var | tie_rand); None keeps the untouched path")
    p.add_argument("--selector_layer", default="dec1")
    p.add_argument("--selector_seed", type=int, default=0)
    return p.parse_args()


def build_loader(args, sequences):
    yaml_path = DATASET_YAML[args.dataset_name]
    data_cfg = OmegaConf.load(yaml_path)
    csv_dir = Path(data_cfg.csv_dir)
    split_name = "_e2e_selected.json"
    (csv_dir / split_name).write_text(json.dumps({"train": [], "test": sequences}, indent=2))
    data_cfg.split_filename = split_name
    dataset = get_multi_frame_dataset(
        data_cfg, base_data_dir=args.base_data_dir, mode=DatasetMode.TEST
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return dataset, loader, csv_dir / split_name


def main():
    args = parse_args()
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    sequences = manifest["sequence_sets"][args.set_name]["sequences"]

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.requires_grad_(False)

    dataset, loader, split_path = build_loader(args, sequences)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": args.dataset_name,
        "set": args.set_name,
        "sequences": sequences,
        "checkpoint": str(ckpt),
        "device": str(device),
        "num_batches": len(loader),
        "modes": {},
    }

    try:
        for mode in args.modes.split(","):
            model.configure_budget(mode)
            if args.selector_mode:
                model.configure_pr1_selector(
                    mode=args.selector_mode, layer=args.selector_layer, seed=args.selector_seed
                )
            mode_dir = out_root / mode
            vis_dir = mode_dir / "fused"
            if vis_dir.exists():
                shutil.rmtree(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)
            # the contract metric script expects <root>/{ir,vis,fused}; reuse the
            # already-materialised sources so both modes are scored on identical
            # inputs and on the JPEGs this run actually saved
            eval_root = Path("eval_data") / args.dataset_name
            for modality in ("ir", "vis"):
                link = mode_dir / modality
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(eval_root.resolve() / modality, link)

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            total_ms = 0.0
            total_frames = 0
            counted = 0
            model.reset_budget_stats()

            with torch.no_grad():
                for i, batch in enumerate(tqdm(loader, desc=f"{args.dataset_name}/{mode}")):
                    if args.limit_batches and i >= args.limit_batches:
                        break
                    if i % args.batch_stride:
                        continue
                    ir_img = batch["ir"].to(device)
                    rgb_img = batch["rgb"].to(device)

                    torch.cuda.synchronize()
                    starter.record()
                    fusion_pred = model(ir_img, rgb_img)
                    ender.record()
                    torch.cuda.synchronize()

                    if i > 5:  # same warm-up skip as test.py
                        total_ms += starter.elapsed_time(ender)
                        total_frames += fusion_pred.shape[1]
                        counted += 1

                    vis_8bit = pred_2_8bit(fusion_pred, ir_img, rgb_img)
                    middle_idx = ir_img.shape[1] // 2
                    ir_path = batch["data_path_ls_dict"]["ir"][middle_idx][0]
                    seq_name = os.path.dirname(ir_path).split(os.sep)[0]
                    frame_stem = os.path.splitext(os.path.basename(ir_path))[0]
                    save_image(vis_8bit, os.path.join(vis_dir, f"{seq_name}_{frame_stem}.jpg"))

            # deployed budget actually used (cheap integer counters, no tensors)
            stats = model.budget_stats()
            k_sum = stats["k_total"]
            support_sum = stats["support_total"]
            zero_support_windows = stats["zero_support_calls"]

            entry = {
                "mode": mode,
                "batches": counted,
                "output_frames": total_frames,
                "avg_forward_ms": total_ms / counted if counted else None,
                "ms_per_fused_image": total_ms / total_frames if total_frames else None,
                "fps": 1000.0 * total_frames / total_ms if total_ms else None,
                "k_selected_total": k_sum,
                "support_total": support_sum if support_sum else None,
                "zero_support_windows": zero_support_windows,
                "fused_dir": str(vis_dir),
                "saved_images": len(list(vis_dir.glob("*.jpg"))),
            }
            report["modes"][mode] = entry
            print(json.dumps(entry, indent=2))

            # manifest copy for the MATLAB contract metrics (relative to repo root,
            # because MATLAB fullfile() concatenates rather than resolving absolute roots)
            eval_root_arg = os.path.relpath(mode_dir, Path.cwd())
            local = dict(manifest)
            local["eval_root"] = eval_root_arg
            (mode_dir / "eval_manifest.json").write_text(json.dumps(local, indent=2))
            entry["eval_root"] = eval_root_arg
    finally:
        split_path.unlink(missing_ok=True)

    fixed = report["modes"].get("fixed", {})
    support = report["modes"].get("support", {})
    if fixed.get("k_selected_total") and support.get("k_selected_total"):
        report["budget_reduction_measured"] = 1.0 - support["k_selected_total"] / fixed["k_selected_total"]
    if fixed.get("avg_forward_ms") and support.get("avg_forward_ms"):
        report["latency_reduction_fraction"] = 1.0 - support["avg_forward_ms"] / fixed["avg_forward_ms"]

    (out_root / "e2e_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "modes"}, indent=2))
    print(f"written {out_root / 'e2e_report.json'}")


if __name__ == "__main__":
    main()
