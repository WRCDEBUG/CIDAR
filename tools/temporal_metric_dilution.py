#!/usr/bin/env python3
"""Is the temporal signal diluted by the static background?

MS2R is a whole-frame mean of |differential flow|.  A method that fixes ghosting
at moving edges changes a small fraction of pixels, so its effect on the mean is
tiny while the metric's own spread dominates.  This script asks whether the same
quantity, restricted to the moving part of the frame or summarised by its tail,
separates the perturbations that the mean cannot.

Reported per variant, from the |D1| / |D2| maps defined by MS2R:
    mean        = the MS2R quantity itself
    p95 / p99   = tail of |D|
    masked      = mean |D| over pixels whose reference flow magnitude is in the
                  top 10% (i.e. where the method can actually act)
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.dataset import get_multi_frame_dataset           # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion                       # noqa: E402
from src.util import metric                                 # noqa: E402

DATASET_YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
                "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
                "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP_PATH = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}
VARIANTS = {"baseline": None, "uniform": (0.0, 0.0, 0.0),
            "current_heavy": (-3.0, 3.0, -3.0), "neighbour_only": (6.0, 0.0, 0.0)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--num_sequences", type=int, default=1)
    p.add_argument("--window_stride", type=int, default=10)
    p.add_argument("--mask_frac", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/temporal_metric_dilution.json")
    return p.parse_args()


def set_temp_weight(model, logits):
    if logits is None:
        return
    t = torch.tensor(logits, dtype=torch.float32).view(1, 1, 3, 1, 1)
    for aam in (model.aam_ir, model.aam_vi):
        with torch.no_grad():
            aam.temp_weight.copy_(t.to(aam.temp_weight.device))


def stats(ms2r, g, r1, r2, mask_frac):
    d_gen = ms2r.compute_flow(g[:, 1], g[:, 2]) - ms2r.compute_flow(g[:, 0], g[:, 1])
    d_ref1 = ms2r.compute_flow(r1[:, 1], r1[:, 2]) - ms2r.compute_flow(r1[:, 0], r1[:, 1])
    d_ref2 = ms2r.compute_flow(r2[:, 1], r2[:, 2]) - ms2r.compute_flow(r2[:, 0], r2[:, 1])
    D1 = (d_gen - d_ref1).abs()          # [1,2,H,W]
    D2 = (d_gen - d_ref2).abs()
    D = torch.cat([D1.flatten(), D2.flatten()])
    mag = (r1[:, 1] - r1[:, 0]).abs().mean(dim=1, keepdim=True)   # reference motion proxy
    thresh = torch.quantile(mag.flatten(), 1 - mask_frac)
    keep = (mag >= thresh)
    masked = 0.5 * (D1[keep.expand_as(D1)].mean() + D2[keep.expand_as(D2)].mean())
    return {"mean": float(D.mean()), "p95": float(torch.quantile(D, 0.95)),
            "p99": float(torch.quantile(D, 0.99)), "masked": float(masked)}


def main():
    args = parse_args(); device = torch.device(args.device)
    man = json.loads((REPO / args.manifest).read_text())
    seqs = man["sequence_sets"][man["datasets"][args.dataset]["default_set"]]["sequences"][: args.num_sequences]
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(REPO / "output" / EXP_PATH[args.dataset] / "checkpoint/latest/model.pth",
                                     map_location=device))
    model.requires_grad_(False)
    data_cfg = OmegaConf.load(DATASET_YAML[args.dataset])
    name = "_temporal_dilution.json"
    (Path(data_cfg.csv_dir) / name).write_text(json.dumps({"train": [], "test": seqs}))
    data_cfg.split_filename = name
    ds = get_multi_frame_dataset(data_cfg, base_data_dir="data2", mode=DatasetMode.TEST)
    loader = DataLoader(Subset(ds, list(range(0, len(ds), args.window_stride))), batch_size=1, shuffle=False)
    ms2r = metric.MS2R_Evaluator(device=str(device))
    keys = list(VARIANTS) + ["swap_neighbours", "distant_frame", "add_noise"]
    acc = {k: [] for k in keys}
    n = 0
    try:
        with torch.no_grad():
            for b in loader:
                ir = b["ir"].to(device); rgb = b["rgb"].to(device)
                r1 = torch.round(ir[:, 1:4].clamp(0, 1) * 255); r2 = torch.round(rgb[:, 1:4].clamp(0, 1) * 255)
                last = None
                for v, lg in VARIANTS.items():
                    set_temp_weight(model, lg)
                    pred = model(ir, rgb).clamp(0, 1)
                    if v == "baseline":
                        last = pred.clone()
                    acc[v].append(stats(ms2r, torch.round(pred * 255), r1, r2, args.mask_frac))
                synth = {"swap_neighbours": last.flip(1),
                         "distant_frame": torch.cat([last[:, 0:1], last[:, 1:2], last[:, 0:1]], 1),
                         "add_noise": (last + torch.randn_like(last) * (8.0 / 255.0)).clamp(0, 1)}
                for v, clip in synth.items():
                    acc[v].append(stats(ms2r, torch.round(clip * 255), r1, r2, args.mask_frac))
                n += 1
    finally:
        (Path(data_cfg.csv_dir) / name).unlink(missing_ok=True)
        set_temp_weight(model, None)
    rows = {k: {m: float(np.mean([a[m] for a in acc[k]])) for m in ("mean", "p95", "p99", "masked")} for k in keys}
    base = rows["baseline"]
    out = REPO / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "windows": n, "mask_frac": args.mask_frac,
                               "metrics": rows,
                               "delta_vs_baseline": {k: {m: rows[k][m] - base[m] for m in rows[k]} for k in keys}},
                              indent=2, ensure_ascii=False))
    print(f"\n=== dilution test · {args.dataset} · {n} windows · mask top {args.mask_frac:.0%} ===")
    print(f"{'variant':18} {'mean':>8} {'d':>9} {'p95':>8} {'d':>9} {'masked':>8} {'d':>9} {'rel%masked':>11}")
    for k in keys:
        r, d = rows[k], {m: rows[k][m] - base[m] for m in rows[k]}
        rel = (r["masked"] / base["masked"] - 1) * 100
        print(f"{k:18} {r['mean']:8.4f} {d['mean']:+9.4f} {r['p95']:8.4f} {d['p95']:+9.4f} "
              f"{r['masked']:8.4f} {d['masked']:+9.4f} {rel:10.2f}%")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
