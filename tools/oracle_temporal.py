#!/usr/bin/env python3
"""B2: is there headroom in the temporal aggregation?  Oracle upper bounds.

The endpoint is MS2R-motion: the MS2R quantity (|differential flow| error against
the two reference clips) averaged over the motion support instead of the whole
frame.  tools/temporal_metric_dilution.py showed that this is 3.7x more sensitive
to temporal corruption than the whole-frame mean and is not fooled by noise.

Two oracles, cheapest first.

  A. Global sweep -- the temporal aggregation weight is a single 3-vector shared
     by every pixel.  Sweep it over the simplex.  This bounds any method whose
     only lever is that global vector.

  B. Per-pixel oracle -- generate the three one-hot outputs (prev-only,
     current-only, next-only) and, for every pixel and frame, keep the candidate
     whose photometric temporal consistency is best:

         cost_k(t) = |F^(k)_t - warp(F^(k)_{t-1}, flow_{t-1->t})|
                   + |F^(k)_t - warp(F^(k)_{t+1}, flow_{t+1->t})|

     This bounds any method that can choose, per pixel, how much of each
     neighbour to mix in -- which is exactly what the proposed occlusion-aware
     aggregation would do.

Decision rule (pre-registered): if the per-pixel oracle does not reach a 5%
relative improvement in MS2R-motion over the baseline, the temporal-aggregation
direction is abandoned before any training is spent on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.util import metric  # noqa: E402

DATASET_YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
                "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
                "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP_PATH = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}

# coarse simplex coverage (softmax logits), plus the trained point
GLOBAL_SWEEP = {
    "trained": None,
    "current_only": (-6.0, 6.0, -6.0),
    "prev_only": (6.0, -6.0, -6.0),
    "next_only": (-6.0, -6.0, 6.0),
    "uniform": (0.0, 0.0, 0.0),
    "half_prev_curr": (0.0, 0.0, -6.0),
    "half_next_curr": (-6.0, 0.0, 0.0),
    "quarter_curr_rest_uniform": (-2.0, 0.0, -2.0),
}
ONE_HOT = {"prev": (10.0, -10.0, -10.0), "curr": (-10.0, 10.0, -10.0), "next": (-10.0, -10.0, 10.0)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--num_sequences", type=int, default=2)
    p.add_argument("--window_stride", type=int, default=10)
    p.add_argument("--mask_frac", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flow_direction", default="released", choices=["released", "corrected"])
    p.add_argument("--output", default="output/pr1/oracle_temporal.json")
    return p.parse_args()


def set_temp_weight(model, logits):
    if logits is None:
        return
    t = torch.tensor(logits, dtype=torch.float32).view(1, 1, 3, 1, 1)
    for aam in (model.aam_ir, model.aam_vi):
        with torch.no_grad():
            aam.temp_weight.copy_(t.to(aam.temp_weight.device))


def flow_gray(evaluator, a, b):
    return evaluator.compute_flow(a, b)


def ms2r_motion(evaluator, g, r1, r2, mask_frac):
    """g/r1/r2: [1,3,C,H,W] in [0,255]. Returns (masked, whole_frame)."""
    d_gen = flow_gray(evaluator, g[:, 1], g[:, 2]) - flow_gray(evaluator, g[:, 0], g[:, 1])
    d_r1 = flow_gray(evaluator, r1[:, 1], r1[:, 2]) - flow_gray(evaluator, r1[:, 0], r1[:, 1])
    d_r2 = flow_gray(evaluator, r2[:, 1], r2[:, 2]) - flow_gray(evaluator, r2[:, 0], r2[:, 1])
    D1 = (d_gen - d_r1).abs()
    D2 = (d_gen - d_r2).abs()
    mag = (r1[:, 1] - r1[:, 0]).abs().mean(dim=1, keepdim=True)
    keep = mag >= torch.quantile(mag.flatten(), 1 - mask_frac)
    masked = 0.5 * (D1[keep.expand_as(D1)].mean() + D2[keep.expand_as(D2)].mean())
    whole = 0.5 * (D1.mean() + D2.mean())
    return float(masked), float(whole)


def warp_by(flow, img):
    """Warp img [1,C,H,W] with flow [1,2,H,W] (flow points img -> target)."""
    h, w = img.shape[-2:]
    grid = torch.stack(torch.meshgrid(
        torch.arange(h, device=img.device, dtype=torch.float32),
        torch.arange(w, device=img.device, dtype=torch.float32), indexing="ij")[::-1], dim=-1).unsqueeze(0)
    norm = torch.empty_like(flow)
    norm[..., 0] = flow[..., 0] * (2.0 / (w - 1))
    norm[..., 1] = flow[..., 1] * (2.0 / (h - 1))
    return F.grid_sample(img, grid + norm.permute(0, 2, 3, 1), align_corners=True, padding_mode="border")


def photometric_cost(evaluator, clip):
    """Per-pixel temporal inconsistency for a 3-frame clip [1,3,C,H,W]; returns [1,3,H,W]."""
    costs = []
    for t in range(3):
        c = torch.zeros_like(clip[:, 0, :1])
        if t > 0:
            f_back = evaluator.compute_flow(clip[:, t] * 255.0, clip[:, t - 1] * 255.0)
            c = c + (clip[:, t] - warp_by(f_back, clip[:, t - 1])).abs().mean(dim=1, keepdim=True)
        if t < 2:
            f_fwd = evaluator.compute_flow(clip[:, t] * 255.0, clip[:, t + 1] * 255.0)
            c = c + (clip[:, t] - warp_by(f_fwd, clip[:, t + 1])).abs().mean(dim=1, keepdim=True)
        costs.append(c)
    return torch.cat(costs, dim=1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    man = json.loads((REPO / args.manifest).read_text())
    seqs = man["sequence_sets"][man["datasets"][args.dataset]["default_set"]]["sequences"][: args.num_sequences]

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(REPO / "output" / EXP_PATH[args.dataset] / "checkpoint/latest/model.pth",
                                     map_location=device))
    model.requires_grad_(False)
    model.flow_direction = args.flow_direction
    stash = []
    model.aam_ir.stash = stash
    model.aam_vi.stash = stash

    data_cfg = OmegaConf.load(DATASET_YAML[args.dataset])
    split_name = "_oracle_temporal.json"
    (Path(data_cfg.csv_dir) / split_name).write_text(json.dumps({"train": [], "test": seqs}))
    data_cfg.split_filename = split_name
    ds = get_multi_frame_dataset(data_cfg, base_data_dir="data2", mode=DatasetMode.TEST)
    loader = DataLoader(Subset(ds, list(range(0, len(ds), args.window_stride))), batch_size=1, shuffle=False)
    ev = metric.MS2R_Evaluator(device=str(device))

    acc = {k: {"masked": 0.0, "whole": 0.0} for k in list(GLOBAL_SWEEP) + ["oracle_perpixel"]}
    n = 0
    try:
        with torch.no_grad():
            for b in loader:
                ir = b["ir"].to(device)
                rgb = b["rgb"].to(device)
                r1 = torch.round(ir[:, 1:4].clamp(0, 1) * 255)
                r2 = torch.round(rgb[:, 1:4].clamp(0, 1) * 255)

                for name, logits in GLOBAL_SWEEP.items():
                    set_temp_weight(model, logits)
                    pred = model(ir, rgb).clamp(0, 1)
                    m, w = ms2r_motion(ev, torch.round(pred * 255), r1, r2, args.mask_frac)
                    acc[name]["masked"] += m
                    acc[name]["whole"] += w

                # per-pixel oracle over the three one-hot outputs
                cands, costs = {}, []
                for name, logits in ONE_HOT.items():
                    set_temp_weight(model, logits)
                    clip = model(ir, rgb).clamp(0, 1)
                    cands[name] = clip
                    costs.append(photometric_cost(ev, clip))
                stacked_cost = torch.stack(costs, dim=0)        # [3,1,3,H,W]
                pick = stacked_cost.argmin(dim=0)               # [1,3,H,W], values in {0,1,2}
                oracle = torch.zeros_like(cands["curr"])
                for k, name in enumerate(ONE_HOT):
                    mask = (pick == k).unsqueeze(2)             # [1,3,1,H,W]
                    oracle = torch.where(mask, cands[name], oracle)
                m, w = ms2r_motion(ev, torch.round(oracle * 255), r1, r2, args.mask_frac)
                acc["oracle_perpixel"]["masked"] += m
                acc["oracle_perpixel"]["whole"] += w
                n += 1
    finally:
        (Path(data_cfg.csv_dir) / split_name).unlink(missing_ok=True)
        set_temp_weight(model, None)

    rows = {k: {m: v / max(n, 1) for m, v in acc[k].items()} for k in acc}
    base = rows["trained"]
    out_rows = {}
    for k, v in rows.items():
        out_rows[k] = {
            "MS2R_motion": v["masked"],
            "MS2R_whole": v["whole"],
            "rel_gain_motion_pct": (base["masked"] - v["masked"]) / base["masked"] * 100 if base["masked"] else None,
            "rel_gain_whole_pct": (base["whole"] - v["whole"]) / base["whole"] * 100 if base["whole"] else None,
        }
    best = max((k for k in rows if k != "trained"),
               key=lambda k: (base["masked"] - rows[k]["masked"]) / base["masked"])
    import numpy as _np
    diversity = ({k: float(_np.mean([s[k] for s in stash])) for k in stash[0]} if stash else {})
    payload = {
        "flow_direction": args.flow_direction,
        "temporal_stream_diversity": diversity,
        "dataset": args.dataset, "sequences": seqs, "windows": n,
        "window_stride": args.window_stride, "mask_frac": args.mask_frac,
        "endpoint": "MS2R-motion (MS2R quantity averaged over the motion support)",
        "global_sweep_and_oracle": out_rows,
        "best_global": best,
        "global_best_rel_gain_pct": out_rows[best]["rel_gain_motion_pct"],
        "per_pixel_oracle_rel_gain_pct": out_rows["oracle_perpixel"]["rel_gain_motion_pct"],
        "decision_threshold_pct": 5.0,
        "decision": ("GO: per-pixel oracle reaches the threshold"
                     if out_rows["oracle_perpixel"]["rel_gain_motion_pct"] >= 5.0
                     else "STOP: per-pixel oracle is below the 5% threshold; abandon temporal aggregation"),
    }
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== B2 oracle · {args.dataset} · flow_direction={args.flow_direction} · {n} windows ===")
    if diversity:
        print("temporal stream diversity (mean |stream_i - stream_j|):", diversity)
    print(f"{'variant':26} {'MS2R-motion':>12} {'gain%':>8} {'MS2R-whole':>11} {'gain%':>8}")
    for k in list(GLOBAL_SWEEP) + ["oracle_perpixel"]:
        r = out_rows[k]
        print(f"{k:26} {r['MS2R_motion']:12.4f} {r['rel_gain_motion_pct']:8.2f} "
              f"{r['MS2R_whole']:11.4f} {r['rel_gain_whole_pct']:8.2f}")
    print(f"\nbest global: {best} ({out_rows[best]['rel_gain_motion_pct']:.2f}%)")
    print(f"per-pixel oracle: {out_rows['oracle_perpixel']['rel_gain_motion_pct']:.2f}%")
    print(f"DECISION: {payload['decision']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
