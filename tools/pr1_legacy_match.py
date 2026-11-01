#!/usr/bin/env python3
"""Verify that explicit legacy mode matches the untouched model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.model.net import VideoFusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", default="MAVFusion-M3SVD")
    parser.add_argument("--ckpt_path", default="latest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", default="output/pr1/legacy_match.json")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    checkpoint = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    model = VideoFusion(model_config=cfg).to(args.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    ir = torch.rand(1, 5, 3, args.height, args.width, device=args.device)
    vi = torch.rand_like(ir)
    with torch.no_grad():
        baseline = model(ir, vi)
        model.configure_pr1_selector(mode="legacy", layer="dec1")
        explicit_legacy = model(ir, vi)
    max_abs_error = float((baseline - explicit_legacy).abs().max().item())
    payload = {
        "batch_size": 1,
        "eval": True,
        "no_grad": True,
        "max_abs_error": max_abs_error,
        "threshold": 1e-5,
        "passed": max_abs_error < 1e-5,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("legacy equivalence failed; stop before statistics")


if __name__ == "__main__":
    main()
