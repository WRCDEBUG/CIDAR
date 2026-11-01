import argparse
import csv
import json
import logging
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_multi_frame_dataset
from src.dataset.base_two_modal_dataset import DatasetMode
from src.model.net import VideoFusion
from src.util import metric

DATASET_TO_YAML = {
    "M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
    "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
    "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml",
}
IMAGE_METRICS = ("Metric_VIF", "Metric_SSIM", "Metric_MI", "Metric_Qabf")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAVFusion outputs on selected sequences")
    parser.add_argument("--dataset_name", required=True, choices=sorted(DATASET_TO_YAML))
    parser.add_argument("--exp_path", required=True)
    parser.add_argument("--ckpt_path", default="latest")
    parser.add_argument("--base_data_dir", default="data2")
    parser.add_argument("--num_test_sequences", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_ms2r", action="store_true")
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def select_sequences(split_path, explicit, count, seed):
    split = json.loads(Path(split_path).read_text())
    available = list(split["test"])
    if explicit:
        selected = [name for name in explicit.split(",") if name]
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Sequences are not in test split: {missing}")
        return selected
    if count >= len(available):
        return available
    return random.Random(seed).sample(sorted(available), count)


def make_dataset_config(yaml_path, selected):
    data_cfg = OmegaConf.load(yaml_path)
    csv_dir = Path(data_cfg.csv_dir)
    split_filename = f"_selected_{Path(yaml_path).stem}_{len(selected)}.json"
    split_path = csv_dir / split_filename
    split_path.write_text(json.dumps({"train": [], "test": selected}, indent=2))
    data_cfg.split_filename = split_filename
    return data_cfg, split_path


def calculate_image_metrics(pred, ir, rgb):
    fusion = torch.round(pred[:, 1].clamp(0, 1) * 255)
    ir_middle = torch.round(ir[:, 2].clamp(0, 1) * 255)
    rgb_middle = torch.round(rgb[:, 2].clamp(0, 1) * 255)
    values = {}
    for name in IMAGE_METRICS:
        values[name] = getattr(metric, name)(fusion, ir_middle, rgb_middle)
    return values


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = torch.device(args.device)
    yaml_path = DATASET_TO_YAML[args.dataset_name]
    base_data_dir = Path(args.base_data_dir).expanduser().resolve()
    original_cfg = OmegaConf.load(yaml_path)
    selected = select_sequences(
        Path(original_cfg.csv_dir) / original_cfg.get("split_filename", "split.json"),
        args.sequences,
        args.num_test_sequences,
        args.seed,
    )
    data_cfg, selected_split = make_dataset_config(yaml_path, selected)
    output_csv = Path(args.output_csv or f"output/{args.exp_path}/metrics_{args.dataset_name}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model_path = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    model = VideoFusion(model_config=model_cfg).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = get_multi_frame_dataset(data_cfg, base_data_dir=str(base_data_dir), mode=DatasetMode.TEST)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ms2r_evaluator = metric.MS2R_Evaluator(device=device) if args.include_ms2r else None
    totals = {name: 0.0 for name in IMAGE_METRICS}
    if args.include_ms2r:
        totals["Metric_MS2R"] = 0.0
    sample_count = 0

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {args.dataset_name}"):
                ir = batch["ir"].to(device)
                rgb = batch["rgb"].to(device)
                pred = model(ir, rgb)
                values = calculate_image_metrics(pred, ir, rgb)
                if ms2r_evaluator is not None:
                    generated = torch.round(pred.clamp(0, 1) * 255)
                    ir_clip = torch.round(ir[:, 1:4].clamp(0, 1) * 255)
                    rgb_clip = torch.round(rgb[:, 1:4].clamp(0, 1) * 255)
                    values["Metric_MS2R"] = ms2r_evaluator.compute_ms2r_metric(
                        generated, ir_clip, rgb_clip
                    )
                for name, value in values.items():
                    totals[name] += float(value.sum().item())
                sample_count += ir.shape[0]
    finally:
        selected_split.unlink(missing_ok=True)

    averages = {name: value / sample_count for name, value in totals.items()}
    rows = [
        {"dataset": args.dataset_name, "sequences": ",".join(selected), "samples": sample_count, **averages}
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"dataset": args.dataset_name, "selected_sequences": selected, "samples": sample_count, "metrics": averages}, indent=2))
    print(f"Saved metrics to {output_csv}")


if __name__ == "__main__":
    main()
