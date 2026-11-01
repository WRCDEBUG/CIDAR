#!/usr/bin/env python3
"""Motion-compensation direction: proof of the convention, and its impact.

Part A -- proof.  Shift a real frame by a known displacement and ask which call
order reconstructs the target when fed to ``flow_warp``:

    flow_warp(prev, get_flow(prev, curr))   <- what VideoFusion.forward does
    flow_warp(prev, get_flow(curr, prev))

Part B -- impact.  With ``model.flow_direction`` set to each value, report

    * the neighbour alignment error on real consecutive frames,
    * Metric_Qabf (UniVF reference implementation) and MS2R-motion of the output.

The shipped order is kept as the default; "corrected" only changes the two
``get_flow`` call sites in ``VideoFusion.forward``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.model.utils import flow_warp  # noqa: E402
from src.util import metric  # noqa: E402
from tools.step0_parity import load_window  # noqa: E402

DATASET_YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml"}
EXP_PATH = {"M3SVD": "MAVFusion-M3SVD"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--num_sequences", type=int, default=1)
    p.add_argument("--window_stride", type=int, default=20)
    p.add_argument("--mask_frac", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/flow_direction.json")
    return p.parse_args()


def load_model(device):
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    m = VideoFusion(model_config=cfg).to(device).eval()
    m.load_state_dict(torch.load(REPO / "output" / EXP_PATH["M3SVD"] / "checkpoint/latest/model.pth",
                                 map_location=device))
    m.requires_grad_(False)
    return m


def part_a(model, device):
    ir, vi, _, _ = load_window(Path("data2/M3SVD"), "0111_1803", 40, 5, device)
    curr = vi[:, 2]
    for shift in (5, 10, 20):
        prev = torch.roll(curr, shifts=-shift, dims=-1)   # known displacement
        with torch.no_grad():
            coded = flow_warp(prev, model.get_flow(prev, curr))
            reversed_ = flow_warp(prev, model.get_flow(curr, prev))
        yield {"shift_px": shift,
               "err_coded_flow_prev_curr": float((coded - curr).abs().mean()),
               "err_reversed_flow_curr_prev": float((reversed_ - curr).abs().mean())}


def ms2r_motion(ev, g, r1, r2, mask_frac):
    dg = ev.compute_flow(g[:, 1], g[:, 2]) - ev.compute_flow(g[:, 0], g[:, 1])
    d1 = ev.compute_flow(r1[:, 1], r1[:, 2]) - ev.compute_flow(r1[:, 0], r1[:, 1])
    d2 = ev.compute_flow(r2[:, 1], r2[:, 2]) - ev.compute_flow(r2[:, 0], r2[:, 1])
    D1, D2 = (dg - d1).abs(), (dg - d2).abs()
    mag = (r1[:, 1] - r1[:, 0]).abs().mean(dim=1, keepdim=True)
    keep = mag >= torch.quantile(mag.flatten(), 1 - mask_frac)
    return float(0.5 * (D1[keep.expand_as(D1)].mean() + D2[keep.expand_as(D2)].mean()))


def part_b(args, model, device):
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][man["datasets"][args.dataset]["default_set"]]["sequences"][: args.num_sequences]
    data_cfg = OmegaConf.load(DATASET_YAML[args.dataset])
    name = "_flow_direction.json"
    (Path(data_cfg.csv_dir) / name).write_text(json.dumps({"train": [], "test": seqs}))
    data_cfg.split_filename = name
    ds = get_multi_frame_dataset(data_cfg, base_data_dir="data2", mode=DatasetMode.TEST)
    loader = DataLoader(Subset(ds, list(range(0, len(ds), args.window_stride))), batch_size=1, shuffle=False)
    ev = metric.MS2R_Evaluator(device=str(device))

    out = {}
    try:
        with torch.no_grad():
            for direction in ("released", "corrected"):
                model.flow_direction = direction
                align_err, qabf, m2r, n = 0.0, 0.0, 0.0, 0
                for b in loader:
                    ir = b["ir"].to(device)
                    rgb = b["rgb"].to(device)
                    prev, curr = ir[:, 1], ir[:, 2]
                    fp = model.pair_flow(prev, curr)
                    align_err += float((flow_warp(prev, fp) - curr).abs().mean())
                    pred = model(ir, rgb).clamp(0, 1)
                    qabf += float(metric.Metric_Qabf(
                        torch.round(pred[:, 1] * 255), torch.round(ir[:, 2] * 255),
                        torch.round(rgb[:, 2] * 255)).mean())
                    m2r += ms2r_motion(ev, torch.round(pred * 255),
                                       torch.round(ir[:, 1:4] * 255), torch.round(rgb[:, 1:4] * 255),
                                       args.mask_frac)
                    n += 1
                out[direction] = {"windows": n, "ir_neighbour_alignment_err": align_err / n,
                                  "Metric_Qabf_middle_frame": qabf / n, "MS2R_motion": m2r / n}
    finally:
        (Path(data_cfg.csv_dir) / name).unlink(missing_ok=True)
        model.flow_direction = "released"
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(device)

    proof = list(part_a(model, device))
    print("=== Part A: which call order reconstructs the target? ===")
    print(f"{'shift(px)':>10} {'coded get_flow(prev,curr)':>26} {'reversed get_flow(curr,prev)':>30}")
    for r in proof:
        print(f"{r['shift_px']:10} {r['err_coded_flow_prev_curr']:26.5f} "
              f"{r['err_reversed_flow_curr_prev']:30.5f}")

    impact = part_b(args, model, device)
    print("\n=== Part B: impact on alignment and metrics ===")
    print(f"{'direction':12} {'align err':>10} {'Qabf':>8} {'MS2R-motion':>12}")
    for k, v in impact.items():
        print(f"{k:12} {v['ir_neighbour_alignment_err']:10.5f} "
              f"{v['Metric_Qabf_middle_frame']:8.4f} {v['MS2R_motion']:12.4f}")

    payload = {"part_a_convention_proof": proof, "part_b_impact": impact,
               "note": "the checkpoint was trained with the 'released' order, so 'corrected' is an "
                       "out-of-distribution input for the trained weights; it measures the defect, "
                       "not the ceiling of a corrected-and-retrained model"}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
