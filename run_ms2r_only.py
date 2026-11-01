"""
只计算 MS2R 指标，跳过 VIF/SSIM/MI/Qabf
用法: python run_ms2r_only.py --dataset_name HDO --exp_path MAVFusion-HDO \
                              --base_data_dir ./data2 \
                              --sequences seq_1,seq_2,seq_3,seq_4,seq_5
"""
import argparse, json, logging, random
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
    "HDO":   "config/dataset/IVF/HDO/HDO_5-frame.yaml",
    "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=sorted(DATASET_TO_YAML))
    p.add_argument("--exp_path", required=True)
    p.add_argument("--ckpt_path", default="latest")
    p.add_argument("--base_data_dir", default="./data2")
    p.add_argument("--sequences", default=None, help="逗号分隔，如 seq_1,seq_2")
    p.add_argument("--num_test_sequences", type=int, default=5)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_csv", default=None)
    return p.parse_args()


def select_sequences(split_path, explicit, count, seed):
    import json
    split = json.loads(Path(split_path).read_text())
    available = list(split["test"])
    if explicit:
        selected = [n for n in explicit.split(",") if n]
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"不在 test split: {missing}")
        return selected
    if count >= len(available):
        return available
    return random.Random(seed).sample(sorted(available), count)


def make_dataset_config(yaml_path, selected):
    data_cfg = OmegaConf.load(yaml_path)
    csv_dir = Path(data_cfg.csv_dir)
    split_filename = f"_selected_ms2r_{Path(yaml_path).stem}_{len(selected)}.json"
    split_path = csv_dir / split_filename
    split_path.write_text(json.dumps({"train": [], "test": selected}, indent=2))
    data_cfg.split_filename = split_filename
    return data_cfg, split_path


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = torch.device(args.device)

    yaml_path = DATASET_TO_YAML[args.dataset_name]
    base_data_dir = Path(args.base_data_dir).expanduser().resolve()
    original_cfg = OmegaConf.load(yaml_path)
    selected = select_sequences(
        Path(original_cfg.csv_dir) / original_cfg.get("split_filename", "split.json"),
        args.sequences, args.num_test_sequences, args.seed,
    )
    data_cfg, selected_split = make_dataset_config(yaml_path, selected)
    output_csv = Path(args.output_csv or f"output/{args.exp_path}/metrics_ms2r_{args.dataset_name}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model_cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model_path = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    model = VideoFusion(model_config=model_cfg).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 数据
    dataset = get_multi_frame_dataset(data_cfg, base_data_dir=str(base_data_dir), mode=DatasetMode.TEST)
    # === 抽样：每 5 个窗口取 1 ===
    from torch.utils.data import Subset
    indices = list(range(0, len(dataset), 5))
    dataset = Subset(dataset, indices)
    print(f"抽样后 {len(dataset)} 个窗口")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # MS2R 评估器
    ms2r_evaluator = metric.MS2R_Evaluator(device=device)

    total_ms2r = 0.0
    sample_count = 0

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"MS2R {args.dataset_name}"):
                ir = batch["ir"].to(device)
                rgb = batch["rgb"].to(device)
                pred = model(ir, rgb)

                # 转成 0-255 整数（MS2R 需要）
                generated = torch.round(pred.clamp(0, 1) * 255)
                ir_clip   = torch.round(ir[:, 1:4].clamp(0, 1) * 255)
                rgb_clip  = torch.round(rgb[:, 1:4].clamp(0, 1) * 255)

                val = ms2r_evaluator.compute_ms2r_metric(generated, ir_clip, rgb_clip)
                total_ms2r += float(val.sum().item())
                sample_count += ir.shape[0]
    finally:
        selected_split.unlink(missing_ok=True)

    ms2r_avg = total_ms2r / max(sample_count, 1)

    result = {
        "dataset": args.dataset_name,
        "selected_sequences": selected,
        "samples": sample_count,
        "Metric_MS2R": ms2r_avg,
    }
    print(json.dumps(result, indent=2))

    with output_csv.open("w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["dataset", "sequences", "samples", "Metric_MS2R"])
        w.writeheader()
        w.writerow({"dataset": args.dataset_name,
                    "sequences": ",".join(selected),
                    "samples": sample_count,
                    "Metric_MS2R": ms2r_avg})
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()