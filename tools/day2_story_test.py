#!/usr/bin/env python3
"""Day 2: does the story hold?  --  gate decoupling and saliency-based allocation.

The hypothesis under test: motion-guided allocation structurally denies
interaction to static-but-thermally-salient regions, so reallocating the budget
to salient patches (and decoupling the two motion gates so they can actually
contribute) should improve quality.

Seven variants, all with the same K, the same frozen weights, no training:

    baseline        motion / amp=val / pixel gate      (shipped)
    decoupled_only  motion / amp=1   / no pixel gate   -> is the gate suppressing quality?
    sta_w050        motion + 0.5*saliency, decoupled
    sta_w200        motion + 2.0*saliency, decoupled
    saliency_only   saliency only, decoupled
    random_ctrl     random allocation, decoupled       -> cheap control
    sta_released    motion + 2.0*saliency, gates as shipped
                    -> should equal the baseline: proves the filler patches are inert
                       while the gates are on

Endpoints: Metric_Qabf (middle frame, UniVF implementation) and MS2R-motion.
Diagnostics: fraction of the selected budget that has zero motion ("filler").
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
from src.util import metric  # noqa: E402

YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
        "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
        "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}

VARIANTS = [
    ("baseline",       dict(score_mode="motion",          sal_weight=0.0, amp_mode="released", gate_mode="released")),
    ("decoupled_only", dict(score_mode="motion",          sal_weight=0.0, amp_mode="unity",    gate_mode="none")),
    ("sta_w050",       dict(score_mode="motion_saliency", sal_weight=0.5, amp_mode="unity",    gate_mode="none")),
    ("sta_w200",       dict(score_mode="motion_saliency", sal_weight=2.0, amp_mode="unity",    gate_mode="none")),
    ("saliency_only",  dict(score_mode="saliency",        sal_weight=0.0, amp_mode="unity",    gate_mode="none")),
    ("random_ctrl",    dict(score_mode="random",          sal_weight=0.0, amp_mode="unity",    gate_mode="none")),
    ("sta_released",   dict(score_mode="motion_saliency", sal_weight=2.0, amp_mode="released", gate_mode="released")),
]
LAYERS = ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--window_stride", type=int, default=20)
    p.add_argument("--mask_frac", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/day2_story_test.json")
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
    args = parse_args()
    device = torch.device(args.device)
    man = json.loads((REPO / args.manifest).read_text())
    seqs = man["sequence_sets"][man["datasets"][args.dataset]["default_set"]]["sequences"]

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(REPO / "output" / EXP[args.dataset] / "checkpoint/latest/model.pth",
                                     map_location=device))
    model.requires_grad_(False)
    model.set_recording(True)

    dc = OmegaConf.load(YAML[args.dataset])
    nm = "_day2.json"
    (Path(dc.csv_dir) / nm).write_text(json.dumps({"train": [], "test": seqs}))
    dc.split_filename = nm
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    loader = DataLoader(Subset(ds, list(range(0, len(ds), args.window_stride))), batch_size=1, shuffle=False)
    ev = metric.MS2R_Evaluator(device=str(device))

    results = {}
    try:
        with torch.no_grad():
            for name, kw in VARIANTS:
                model.configure_allocation(**kw)
                qabf = m2r = filler = 0.0
                n = 0
                for b in loader:
                    ir = b["ir"].to(device)
                    rgb = b["rgb"].to(device)
                    model.clear_pr1_records()
                    pred = model(ir, rgb).clamp(0, 1)
                    recs = model.get_pr1_records()
                    # allocation diagnostic: share of the budget with zero motion score
                    sel = [r for r in recs[LAYERS[0]]["selection"]]
                    for r in sel:
                        sc, idx = r["scores"], r["active_idx"]
                        filler += float((sc.index_select(0, idx) == 0).float().mean())
                    qabf += float(metric.Metric_Qabf(torch.round(pred[:, 1] * 255),
                                                     torch.round(ir[:, 2] * 255),
                                                     torch.round(rgb[:, 2] * 255)).mean())
                    m2r += ms2r_motion(ev, torch.round(pred * 255),
                                       torch.round(ir[:, 1:4] * 255), torch.round(rgb[:, 1:4] * 255),
                                       args.mask_frac)
                    n += 1
                results[name] = {"windows": n, "Metric_Qabf": qabf / n, "MS2R_motion": m2r / n,
                                 "filler_fraction_dec1": filler / max(n * 3, 1)}
                print(f"  {name:15} Qabf={results[name]['Metric_Qabf']:.4f} "
                      f"MS2R-m={results[name]['MS2R_motion']:.4f} filler={results[name]['filler_fraction_dec1']:.3f}",
                      flush=True)
    finally:
        (Path(dc.csv_dir) / nm).unlink(missing_ok=True)
        model.configure_allocation()
        model.set_recording(False)

    base = results["baseline"]
    for k, v in results.items():
        v["qabf_gain_pct"] = (v["Metric_Qabf"] - base["Metric_Qabf"]) / base["Metric_Qabf"] * 100
        v["ms2r_motion_change_pct"] = (v["MS2R_motion"] / base["MS2R_motion"] - 1) * 100

    best_q = max((k for k in results if k != "baseline"), key=lambda k: results[k]["qabf_gain_pct"])
    best_m = min((k for k in results if k != "baseline"), key=lambda k: results[k]["MS2R_motion"])
    payload = {"dataset": args.dataset, "sequences": seqs, "windows": base["windows"],
               "window_stride": args.window_stride, "endpoints": ["Metric_Qabf", "MS2R-motion"],
               "threshold_pct": 5.0, "variants": results,
               "best_qabf_variant": best_q, "best_qabf_gain_pct": results[best_q]["qabf_gain_pct"],
               "best_ms2r_variant": best_m, "best_ms2r_gain_pct": -results[best_m]["ms2r_motion_change_pct"],
               "story_holds": bool(results[best_q]["qabf_gain_pct"] >= 5.0
                                   or -results[best_m]["ms2r_motion_change_pct"] >= 5.0)}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== Day 2 story test · {args.dataset} · {base['windows']} windows ===")
    print(f"{'variant':16} {'Qabf':>8} {'gain%':>8} {'MS2R-m':>8} {'chg%':>8} {'filler':>8}")
    for k in [v[0] for v in VARIANTS]:
        r = results[k]
        print(f"{k:16} {r['Metric_Qabf']:8.4f} {r['qabf_gain_pct']:+8.2f} "
              f"{r['MS2R_motion']:8.4f} {r['ms2r_motion_change_pct']:+8.2f} {r['filler_fraction_dec1']:8.3f}")
    print(f"\nstory holds (>=5% on either endpoint): {payload['story_holds']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
