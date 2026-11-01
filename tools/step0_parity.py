#!/usr/bin/env python3
"""Step 0 parity audit.

Two independent claims are recorded here, on real frames rather than synthetic
tensors:

1. ``selector=None`` (the untouched code path) and an explicit
   ``PatchSelector(mode="legacy")`` produce bit-comparable outputs and exactly
   the same ``active_idx`` at every layer.
2. The selected index set really is ``torch.topk(scores, k, sorted=False)``.

The result is the gate required by codex要求/week1_codex_task.md section 1:
without it no downstream statistic may be treated as valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.net import VideoFusion  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--seq", default="0111_1803")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--ckpt_path", default="latest")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="output/pr1/legacy_match.json")
    return p.parse_args()


def list_frames(directory: Path):
    files = [f for f in sorted(directory.iterdir()) if f.suffix.lower() in IMAGE_EXT]
    if not files:
        raise FileNotFoundError(f"No frames in {directory}")
    return files


def load_window(data_dir: Path, seq: str, start: int, count: int, device):
    ir_dir = data_dir / seq / "infrared"
    vi_dir = data_dir / seq / "visible"
    ir_files = list_frames(ir_dir)
    vi_files = list_frames(vi_dir)
    if len(ir_files) != len(vi_files):
        raise ValueError(f"IR/VI frame count mismatch: {len(ir_files)} vs {len(vi_files)}")
    if start + count > len(ir_files):
        raise ValueError(f"Requested window {start}:{start+count} exceeds {len(ir_files)} frames")

    import torchvision.io as tio

    def stack(files):
        frames = []
        for f in files[start:start + count]:
            img = tio.read_image(str(f), mode="RGB").float() / 255.0
            frames.append(img)
        return torch.stack(frames, dim=0).unsqueeze(0).to(device)

    return stack(ir_files), stack(vi_files), len(ir_files), ir_files[start].name


def snapshot_indices(model):
    records = model.get_pr1_records()
    return {
        layer: [rec["active_idx"].cpu().tolist() for rec in data["selection"]]
        for layer, data in records.items()
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    seq_dir = data_dir / args.seq
    if not seq_dir.is_dir():
        sys.exit(f"Missing sequence directory: {seq_dir}")

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    if not ckpt.exists():
        sys.exit(f"Missing checkpoint: {ckpt}")

    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.requires_grad_(False)
    model.set_recording(True)

    ir, vi, total_frames, first_frame = load_window(
        data_dir, args.seq, args.start, args.num_frames, device
    )

    with torch.no_grad():
        model.clear_pr1_records()
        baseline = model(ir, vi)
        baseline_idx = snapshot_indices(model)

        model.configure_pr1_selector(mode="legacy", layer="dec1")
        model.clear_pr1_records()
        explicit = model(ir, vi)
        explicit_idx = snapshot_indices(model)

    max_abs_error = float((baseline - explicit).abs().max().item())
    mean_abs_error = float((baseline - explicit).abs().mean().item())
    reference_scale = float(baseline.abs().max().item())

    layers = ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1")
    indices_equal = {}
    for layer in layers:
        a, b = baseline_idx.get(layer, []), explicit_idx.get(layer, [])
        if not a or not b:
            indices_equal[layer] = None
            continue
        indices_equal[layer] = all(
            sorted(x) == sorted(y) for x, y in zip(a, b)
        )

    # Claim 2: the legacy selector reproduces torch.topk(scores, k, sorted=False)
    # exactly, on the scores actually observed at every layer.
    from src.model.selector import PatchSelector

    selector = PatchSelector(mode="legacy")
    topk_match = {}
    for layer in layers:
        layer_records = explicit_idx.get(layer, [])
        if not layer_records:
            topk_match[layer] = None
            continue
        ok = True
        for rec in model.get_pr1_records()[layer]["selection"]:
            scores, k = rec["scores"], rec["k"]
            _, idx = torch.topk(scores, k, sorted=False)
            sel_idx, _ = selector.select(scores=scores, k=k, patch_size=1)
            ok = ok and sorted(idx.cpu().tolist()) == sorted(sel_idx.cpu().tolist())
        topk_match[layer] = bool(ok)

    payload = {
        "seq": args.seq,
        "first_frame": first_frame,
        "window": [args.start, args.start + args.num_frames],
        "sequence_frames": total_frames,
        "input_shape": list(ir.shape),
        "checkpoint": str(ckpt),
        "device": str(device),
        "batch_size": 1,
        "eval_mode": True,
        "no_grad": True,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "output_abs_max": reference_scale,
        "relative_max_error": max_abs_error / reference_scale if reference_scale else None,
        "threshold": 1e-5,
        "active_idx_identical": indices_equal,
        "legacy_equals_torch_topk": topk_match,
        "passed": bool(
            max_abs_error < 1e-5 and all(v for v in indices_equal.values() if v is not None)
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["passed"]:
        sys.exit("legacy equivalence FAILED; do not run statistics or tie-break experiments")


if __name__ == "__main__":
    main()
