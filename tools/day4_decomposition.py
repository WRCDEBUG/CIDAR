#!/usr/bin/env python3
"""Day 4: turn the degradation hypotheses into evidence before designing modules.

Question: when the model collapses under degradation (-21% to -71% Qabf), how much
of that collapse comes from the OPTICAL-FLOW path (flow error -> wrong warping and
spurious motion mask) and how much from the FEATURE path (corrupted shallow
features)?

Design
  A  clean input                          -> reference quality
  B  degraded input                       -> full collapse
  C  degraded input but the flow/mask is replayed from the clean run
                                          -> isolates the flow path

  flow-path collapse  ~= Q_C - Q_B
  feature-path collapse ~= Q_A - Q_C

Also measured directly:
  * flow EPE between clean and degraded VIS (how wrong does SEA-RAFT get?)
  * spurious-motion rate: pixels that are motion in the degraded mask but not in
    the clean mask, i.e. budget spent on motion that is not there.

The clean flows are collected by wrapping ``pair_flow`` on a clean forward and
replayed on the degraded forward, so no model code is modified.
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
    p.add_argument("--window_stride", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/day4_decomposition.json")
    return p.parse_args()


def degrade(x, kind):
    if kind == "clean":
        return x
    if kind == "noise10":
        return (x + torch.randn_like(x) * (10.0 / 255.0)).clamp(0, 1)
    if kind == "noise20":
        return (x + torch.randn_like(x) * (20.0 / 255.0)).clamp(0, 1)
    if kind == "dark":
        return x.clamp(0, 1) ** 1.8
    if kind == "blur":
        k = torch.ones(1, 1, 5, 5, device=x.device) / 25.0
        b, t, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * t, c, h, w), k.repeat(c, 1, 1, 1), padding=2, groups=c)
        return y.reshape(b, t, c, h, w)
    raise ValueError(kind)


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
    nm = "_day4.json"
    (Path(dc.csv_dir) / nm).write_text(json.dumps({"train": [], "test": seqs}))
    dc.split_filename = nm
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    loader = DataLoader(Subset(ds, list(range(0, len(ds), args.window_stride))), batch_size=1, shuffle=False)

    orig_pair_flow = model.pair_flow
    orig_mask = model.generate_motion_mask_from_flow
    store = {"flows": [], "masks": []}
    mode = {"kind": "record"}

    def patched_pair_flow(source, target):
        if mode["kind"] == "record":
            f = orig_pair_flow(source, target)
            store["flows"].append(f.detach().clone())
            return f
        f = store["flows"][patched_pair_flow.i]
        patched_pair_flow.i += 1
        return f

    def patched_mask(fp, fn):
        if mode["kind"] == "record":
            m = orig_mask(fp, fn)
            store["masks"].append(m.detach().clone())
            return m
        m = store["masks"][patched_mask.i]
        patched_mask.i += 1
        return m

    model.pair_flow = patched_pair_flow
    model.generate_motion_mask_from_flow = patched_mask

    degrees = ("noise10", "noise20", "dark", "blur")
    res = {k: {"qabf": 0.0, "n": 0} for k in (("clean",) + degrees + tuple(f"{d}_cleanflow" for d in degrees))}
    flow_err = {d: {"epe": 0.0, "spurious": 0.0, "missed": 0.0, "n": 0} for d in degrees}

    try:
        with torch.no_grad():
            for b in loader:
                ir0 = b["ir"].to(device)
                rgb0 = b["rgb"].to(device)
                # A: clean  (also records the clean flows/masks)
                mode["kind"] = "record"
                store["flows"], store["masks"] = [], []
                pred = model(ir0, rgb0).clamp(0, 1)
                res["clean"]["qabf"] += float(metric.Metric_Qabf(
                    torch.round(pred[:, 1] * 255), torch.round(ir0[:, 2] * 255),
                    torch.round(rgb0[:, 2] * 255)).mean())
                res["clean"]["n"] += 1
                clean_flows = list(store["flows"])
                clean_masks = list(store["masks"])

                # direct flow-error measurement on the VIS pair used by the model
                with torch.no_grad():
                    for d in degrees:
                        rgbd = degrade(rgb0, d)
                        for t in (1, 2):
                            fd = model.get_flow(rgbd[:, t], rgbd[:, t + 1])
                            fc = model.get_flow(rgb0[:, t], rgb0[:, t + 1])
                            flow_err[d]["epe"] += float((fd - fc).abs().mean())
                            flow_err[d]["n"] += 1

                for d in degrees:
                    ir = degrade(ir0, d)
                    rgb = degrade(rgb0, d)
                    # B: full degradation
                    mode["kind"] = "record"
                    store["flows"], store["masks"] = [], []
                    pB = model(ir, rgb).clamp(0, 1)
                    deg_masks = list(store["masks"])
                    res[d]["qabf"] += float(metric.Metric_Qabf(
                        torch.round(pB[:, 1] * 255), torch.round(ir0[:, 2] * 255),
                        torch.round(rgb0[:, 2] * 255)).mean())
                    res[d]["n"] += 1
                    # spurious / missed motion
                    for mc, md in zip(clean_masks, deg_masks):
                        cs, ds_ = mc > 0, md > 0
                        flow_err[d]["spurious"] += float((ds_ & ~cs).float().mean())
                        flow_err[d]["missed"] += float((cs & ~ds_).float().mean())
                    # C: degraded features, clean flow/mask
                    mode["kind"] = "replay"
                    patched_pair_flow.i = 0
                    patched_mask.i = 0
                    store["flows"], store["masks"] = clean_flows, clean_masks
                    pC = model(ir, rgb).clamp(0, 1)
                    res[f"{d}_cleanflow"]["qabf"] += float(metric.Metric_Qabf(
                        torch.round(pC[:, 1] * 255), torch.round(ir0[:, 2] * 255),
                        torch.round(rgb0[:, 2] * 255)).mean())
                    res[f"{d}_cleanflow"]["n"] += 1
    finally:
        model.pair_flow = orig_pair_flow
        model.generate_motion_mask_from_flow = orig_mask
        (Path(dc.csv_dir) / nm).unlink(missing_ok=True)

    q = {k: v["qabf"] / max(v["n"], 1) for k, v in res.items()}
    clean = q["clean"]
    payload = {"dataset": args.dataset, "sequences": seqs, "windows": res["clean"]["n"],
               "qabf": q,
               "flow_error_vs_clean_flow": {d: {"epe": v["epe"] / max(v["n"], 1),
                                                "spurious_mask_pct": v["spurious"] / max(v["n"], 1) * 100,
                                                "missed_mask_pct": v["missed"] / max(v["n"], 1) * 100}
                                            for d, v in flow_err.items()}}
    for d in degrees:
        b, c = q[d], q[f"{d}_cleanflow"]
        payload.setdefault("decomposition", {})[d] = {
            "total_drop_pct": (1 - b / clean) * 100,
            "flow_path_recovery_pct_of_clean": (c - b) / clean * 100,
            "feature_path_remainder_pct_of_clean": (clean - c) / clean * 100,
            "flow_path_share_of_drop_pct": (c - b) / (clean - b) * 100 if clean > b else None,
        }

    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== Day 4 decomposition · {args.dataset} · {payload['windows']} windows ===")
    print(f"clean Qabf = {clean:.4f}\n")
    print(f"{'degradation':11} {'Qabf(deg)':>10} {'Qabf(+cleanflow)':>17} {'total drop%':>12} "
          f"{'flow-path%':>11} {'feat-path%':>11} {'flow share%':>12}")
    for d in degrees:
        s = payload["decomposition"][d]
        print(f"{d:11} {q[d]:10.4f} {q[f'{d}_cleanflow']:17.4f} {s['total_drop_pct']:12.2f} "
              f"{s['flow_path_recovery_pct_of_clean']:11.2f} {s['feature_path_remainder_pct_of_clean']:11.2f} "
              f"{s['flow_path_share_of_drop_pct']:12.1f}")
    print(f"\n{'degradation':11} {'flow EPE':>10} {'spurious mask%':>15} {'missed mask%':>14}")
    for d in degrees:
        e = payload["flow_error_vs_clean_flow"][d]
        print(f"{d:11} {e['epe']:10.3f} {e['spurious_mask_pct']:15.3f} {e['missed_mask_pct']:14.3f}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
