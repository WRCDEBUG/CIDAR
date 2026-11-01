#!/usr/bin/env python3
"""Day 3-4 gate: does reliability-weighted motion gating help on degraded input?

Hypothesis (the only quality lead not refuted on Day 2): under degradations the
optical flow produces *spurious* motion, so the budget is spent on fake motion
patches and misaligned features are injected.  Damping the motion mask where the
flow is unreliable should recover quality there, while being neutral on clean
input.  Note this is consistent with the Day-2 finding that the motion gate is
beneficial -- it asks for LESS interaction on unreliable motion, not for more
interaction on static regions.

Degradations are applied to the model input only (both modalities):
    clean, noise10, noise20, dark(gamma), blur

Reliability modes:
    none (shipped), grad (flow high-frequency), fb (forward/backward consistency)

Endpoints: Metric_Qabf (middle frame, UniVF implementation) and MS2R-motion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
        "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
        "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--window_stride", type=int, default=20)
    p.add_argument("--mask_frac", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="output/pr1/day3_reliability_gate.json")
    return p.parse_args()


def degrade(x, kind, device):
    if kind == "clean":
        return x
    if kind == "noise10":
        return (x + torch.randn_like(x) * (10.0 / 255.0)).clamp(0, 1)
    if kind == "noise20":
        return (x + torch.randn_like(x) * (20.0 / 255.0)).clamp(0, 1)
    if kind == "dark":
        return x.clamp(0, 1) ** 1.8
    if kind == "blur":
        k = torch.ones(1, 1, 5, 5, device=device) / 25.0
        b, t, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * t, c, h, w), k.repeat(c, 1, 1, 1), padding=2, groups=c)
        return y.reshape(b, t, c, h, w)
    raise ValueError(kind)


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
    dc = OmegaConf.load(YAML[args.dataset])
    nm = "_day3.json"
    (Path(dc.csv_dir) / nm).write_text(json.dumps({"train": [], "test": seqs}))
    dc.split_filename = nm
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    idx = list(range(0, len(ds), args.window_stride))
    if args.limit:
        idx = idx[: args.limit]
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False)
    ev = metric.MS2R_Evaluator(device=str(device))

    configs = []
    for deg in ("clean", "noise10", "noise20", "dark", "blur"):
        for mode, alpha in (("none", 0.0), ("grad", 8.0), ("grad", 20.0), ("fb", 4.0)):
            configs.append((deg, mode, alpha))

    results = {}
    try:
        with torch.no_grad():
            for deg, mode, alpha in configs:
                model.reliability_mode = mode
                model.rel_alpha = alpha
                qabf = m2r = 0.0
                n = 0
                for b in loader:
                    ir0 = b["ir"].to(device)
                    rgb0 = b["rgb"].to(device)
                    ir = degrade(ir0, deg, device)
                    rgb = degrade(rgb0, deg, device)
                    pred = model(ir, rgb).clamp(0, 1)
                    qabf += float(metric.Metric_Qabf(torch.round(pred[:, 1] * 255),
                                                     torch.round(ir0[:, 2] * 255),
                                                     torch.round(rgb0[:, 2] * 255)).mean())
                    m2r += ms2r_motion(ev, torch.round(pred * 255),
                                       torch.round(ir0[:, 1:4] * 255), torch.round(rgb0[:, 1:4] * 255),
                                       args.mask_frac)
                    n += 1
                key = f"{deg}|{mode}|a{alpha:g}"
                results[key] = {"degradation": deg, "mode": mode, "alpha": alpha, "windows": n,
                                "Metric_Qabf": qabf / n, "MS2R_motion": m2r / n}
                print(f"  {key:22} Qabf={results[key]['Metric_Qabf']:.4f} MS2R-m={results[key]['MS2R_motion']:.4f}",
                      flush=True)
    finally:
        (Path(dc.csv_dir) / nm).unlink(missing_ok=True)
        model.reliability_mode, model.rel_alpha = "none", 1.0

    # reference per degradation = shipped model on that degradation
    ref = {d: results[f"{d}|none|a0"] for d in ("clean", "noise10", "noise20", "dark", "blur")}
    best = {}
    for d in ref:
        cands = [k for k in results if k.startswith(f"{d}|") and not k.endswith("|none|a0")]
        q = max(cands, key=lambda k: results[k]["Metric_Qabf"] - ref[d]["Metric_Qabf"])
        m = min(cands, key=lambda k: results[k]["MS2R_motion"] - ref[d]["MS2R_motion"])
        best[d] = {
            "ref_qabf": ref[d]["Metric_Qabf"], "ref_ms2r": ref[d]["MS2R_motion"],
            "best_qabf_key": q, "qabf_gain_pct": (results[q]["Metric_Qabf"] / ref[d]["Metric_Qabf"] - 1) * 100,
            "best_ms2r_key": m, "ms2r_gain_pct": (1 - results[m]["MS2R_motion"] / ref[d]["MS2R_motion"]) * 100,
        }
    clean = best["clean"]
    degraded_ok = [d for d in ref if d != "clean"
                   and (best[d]["qabf_gain_pct"] >= 5.0 or best[d]["ms2r_gain_pct"] >= 5.0)]
    payload = {"dataset": args.dataset, "sequences": seqs, "windows": next(iter(results.values()))["windows"],
               "threshold_pct": 5.0, "results": results, "per_degradation_best": best,
               "clean_side_effect_pct": {"qabf": clean["qabf_gain_pct"], "ms2r": clean["ms2r_gain_pct"]},
               "degradations_with_ge5pct_gain": degraded_ok,
               "gate_passes": bool(degraded_ok and abs(clean["qabf_gain_pct"]) <= 1.0)}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== Day 3 reliability gate · {args.dataset} · {payload['windows']} windows ===")
    print(f"{'degradation':12} {'ref Qabf':>9} {'ref MS2R':>9} {'best gain%':>11} {'best MS2R gain%':>16}")
    for d in ref:
        b = best[d]
        print(f"{d:12} {b['ref_qabf']:9.4f} {b['ref_ms2r']:9.4f} {b['qabf_gain_pct']:+11.2f} "
              f"{b['ms2r_gain_pct']:+16.2f}")
    print(f"\ndegradations with >=5% gain: {degraded_ok}")
    print(f"clean side-effect: Qabf {clean['qabf_gain_pct']:+.2f}%  MS2R {clean['ms2r_gain_pct']:+.2f}%")
    print(f"gate passes: {payload['gate_passes']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
