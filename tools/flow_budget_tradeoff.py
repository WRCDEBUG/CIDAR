#!/usr/bin/env python3
"""F1: quality-latency tradeoff of the optical-flow budget.

`get_flow` is 4 calls per 5-frame window and 62% of device time, so it is the
only large lever the profiling found.  This measures what each reduction costs:

    full_256   shipped: 4 calls, SEA-RAFT input capped at 256px
    half_256   2 calls, later iterations reuse the cached field
    full_128   4 calls at a 128px cap
    half_128   both
    full_64    4 calls at a 64px cap

For each: forward latency (CUDA events), IR neighbour alignment error,
Metric_Qabf (middle frame, UniVF implementation) and MS2R-motion.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.dataset import get_multi_frame_dataset            # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion                       # noqa: E402
from src.model.utils import flow_warp                       # noqa: E402
from src.util import metric                                 # noqa: E402

YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
        "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
        "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}
# SEA-RAFT builds a 4-level correlation pyramid that halves the 8x-downsampled
# feature map once per level, so *both* padded input sides must stay >= 128.
# For a 3:4 frame that means max_res >= ~171; 96 and 128 crash inside CorrBlock.
# This also means the shipped 256 already reduces 480x640 to 192x256, i.e. the
# resolution lever is narrow -- the number of calls is the real lever.
VARIANTS = [("full_256", "full", 256), ("half_256", "half", 256),
            ("single_256", "single", 256), ("full_192", "full", 192)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--num_sequences", type=int, default=2)
    p.add_argument("--window_stride", type=int, default=10)
    p.add_argument("--mask_frac", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/flow_budget_tradeoff.json")
    return p.parse_args()


def ms2r_motion(ev, g, r1, r2, mf):
    dg = ev.compute_flow(g[:, 1], g[:, 2]) - ev.compute_flow(g[:, 0], g[:, 1])
    d1 = ev.compute_flow(r1[:, 1], r1[:, 2]) - ev.compute_flow(r1[:, 0], r1[:, 1])
    d2 = ev.compute_flow(r2[:, 1], r2[:, 2]) - ev.compute_flow(r2[:, 0], r2[:, 1])
    D1, D2 = (dg - d1).abs(), (dg - d2).abs()
    mag = (r1[:, 1] - r1[:, 0]).abs().mean(dim=1, keepdim=True)
    keep = mag >= torch.quantile(mag.flatten(), 1 - mf)
    return float(0.5 * (D1[keep.expand_as(D1)].mean() + D2[keep.expand_as(D2)].mean()))


def main():
    a = parse_args(); device = torch.device(a.device)
    man = json.loads((REPO / a.manifest).read_text())
    seqs = man["sequence_sets"][man["datasets"][a.dataset]["default_set"]]["sequences"][: a.num_sequences]
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(REPO / "output" / EXP[a.dataset] / "checkpoint/latest/model.pth",
                                     map_location=device))
    model.requires_grad_(False)
    dc = OmegaConf.load(YAML[a.dataset]); nm = "_flow_budget.json"
    (Path(dc.csv_dir) / nm).write_text(json.dumps({"train": [], "test": seqs})); dc.split_filename = nm
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    loader = DataLoader(Subset(ds, list(range(0, len(ds), a.window_stride))), batch_size=1, shuffle=False)
    ev = metric.MS2R_Evaluator(device=str(device))
    out = {}
    try:
        with torch.no_grad():
            for name, budget, res in VARIANTS:
                print(f"  running {name} ...", flush=True)
                model.flow_budget, model.flow_max_res = budget, res
                for _ in range(2):
                    b = next(iter(loader)); model(b["ir"].to(device), b["rgb"].to(device))
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                total_ms = qabf = m2r = align = 0.0; n = 0
                for b in loader:
                    ir = b["ir"].to(device); rgb = b["rgb"].to(device)
                    torch.cuda.synchronize(); s.record()
                    pred = model(ir, rgb)
                    e.record(); torch.cuda.synchronize()
                    total_ms += s.elapsed_time(e)
                    pred = pred.clamp(0, 1)
                    prev, curr = ir[:, 1], ir[:, 2]
                    align += float((flow_warp(prev, model.pair_flow(prev, curr)) - curr).abs().mean())
                    qabf += float(metric.Metric_Qabf(torch.round(pred[:, 1] * 255),
                                                     torch.round(ir[:, 2] * 255),
                                                     torch.round(rgb[:, 2] * 255)).mean())
                    m2r += ms2r_motion(ev, torch.round(pred * 255), torch.round(ir[:, 1:4] * 255),
                                       torch.round(rgb[:, 1:4] * 255), a.mask_frac)
                    n += 1
                out[name] = {"budget": budget, "flow_max_res": res, "windows": n,
                             "ms_per_window": total_ms / n, "align_err": align / n,
                             "Metric_Qabf": qabf / n, "MS2R_motion": m2r / n}
                print(f"  {name:10} {out[name]['ms_per_window']:8.1f} ms  "
                      f"Qabf={out[name]['Metric_Qabf']:.4f}  MS2R-m={out[name]['MS2R_motion']:.4f}")
    finally:
        (Path(dc.csv_dir) / nm).unlink(missing_ok=True)
        model.flow_budget, model.flow_max_res = "full", 256
    base = out["full_256"]
    for k, v in out.items():
        v["speedup_vs_full"] = base["ms_per_window"] / v["ms_per_window"]
        v["qabf_drop"] = base["Metric_Qabf"] - v["Metric_Qabf"]
        v["ms2r_motion_rel_change_pct"] = (v["MS2R_motion"] / base["MS2R_motion"] - 1) * 100
    payload = {"dataset": a.dataset, "sequences": seqs, "window_stride": a.window_stride,
               "variants": out}
    o = REPO / a.output; o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n{'variant':10} {'ms':>8} {'speedup':>8} {'Qabf':>8} {'dQabf':>8} {'MS2R-m':>8} {'rel%':>8}")
    for k, v in out.items():
        print(f"{k:10} {v['ms_per_window']:8.1f} {v['speedup_vs_full']:8.2f} {v['Metric_Qabf']:8.4f} "
              f"{v['qabf_drop']:+8.4f} {v['MS2R_motion']:8.4f} {v['ms2r_motion_rel_change_pct']:+8.2f}")
    print(f"\nwritten {o}")


if __name__ == "__main__":
    main()
