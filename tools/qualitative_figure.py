#!/usr/bin/env python3
"""Qualitative comparison panel, in the layout this field's reviewers expect.

For each scene there are TWO rows: the full frame, and an enlarged crop of the
dynamic region, so that noise removal and structure preservation can be judged
without zooming the PDF.  Columns: infrared | visible | degraded input |
baseline | DAR (ours) | target (clean input through the same network).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from tools.stage2_dar import degrade  # noqa: E402
from src.util import metric  # noqa: E402

YAML = {"VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml",
        "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml"}
KEY = {"VTMOT": "VTMOT_frozen5_test", "HDO": "HDO_frozen5_test"}
EXP = {"VTMOT": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO"}
COLS = ["infrared", "visible", "degraded", "baseline", "DAR (ours)", "target (clean)"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="VTMOT")
    p.add_argument("--degrade", default="noise20")
    p.add_argument("--dar_ckpt", default="output/pr1/stage2_cross_noblur_long.pth")
    p.add_argument("--est_mode", default="cross")
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--window_stride", type=int, default=151)
    p.add_argument("--zoom", type=int, default=64)
    p.add_argument("--output", default="runs/00_contract/fig_qualitative.png")
    return p.parse_args()


def load(dataset):
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    dc = OmegaConf.load(YAML[dataset])
    nm = f"_qual_{dataset}.json"
    (Path(dc.csv_dir) / nm).write_text(
        json.dumps({"train": [], "test": man["sequence_sets"][KEY[dataset]]["sequences"]}))
    dc.split_filename = nm
    return get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST), Path(dc.csv_dir) / nm


def main():
    a = args()
    dev = torch.device("cuda")
    ds, split = load(a.dataset)
    cand = list(range(0, len(ds), a.window_stride))[:12]
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ck = torch.load(REPO / "output" / EXP[a.dataset] / "checkpoint/latest/model.pth",
                    map_location=dev)
    base = VideoFusion(model_config=cfg).to(dev).eval()
    base.load_state_dict(ck, strict=False)
    dar = VideoFusion(model_config=cfg,
                      dar={"mode": "predicted", "est_mode": a.est_mode}).to(dev).eval()
    dar.load_state_dict(ck, strict=False)
    dar.load_state_dict(torch.load(REPO / a.dar_ckpt, map_location=dev), strict=False)

    # pick the scenes where the module helps most, measured, not hand-picked
    scored = []
    try:
        with torch.no_grad():
            for i in cand:
                b = ds[i]
                ir0, rgb0 = b["ir"][None].to(dev), b["rgb"][None].to(dev)
                g = torch.Generator(device=dev).manual_seed(4242)
                irn, rgbn = degrade(ir0, a.degrade, g), degrade(rgb0, a.degrade, g)
                iref = torch.round(ir0[:, 2] * 255); vref = torch.round(rgb0[:, 2] * 255)
                ob = base(irn, rgbn).clamp(0, 1)[:, 1]
                od = dar(irn, rgbn).clamp(0, 1)[:, 1]
                qb = float(metric.Metric_Qabf(torch.round(ob * 255), iref, vref).mean())
                qd = float(metric.Metric_Qabf(torch.round(od * 255), iref, vref).mean())
                scored.append((qd - qb, i, qb, qd))
    finally:
        pass
    scored.sort(reverse=True)
    idx = [i for _, i, _, _ in scored[:a.scenes]]
    qmap = {i: (qb, qd) for _, i, qb, qd in scored}
    print("chosen windows:", [(i, round(qmap[i][0], 4), round(qmap[i][1], 4)) for i in idx])
    nrow = 2 * len(idx)
    fig, axes = plt.subplots(nrow, 6, figsize=(14, 2.35 * nrow))
    axes = np.atleast_2d(axes)
    try:
        with torch.no_grad():
            for r, i in enumerate(idx):
                b = ds[i]
                ir0, rgb0 = b["ir"][None].to(dev), b["rgb"][None].to(dev)
                g = torch.Generator(device=dev).manual_seed(4242)
                irn, rgbn = degrade(ir0, a.degrade, g), degrade(rgb0, a.degrade, g)
                ob = base(irn, rgbn).clamp(0, 1)[0, 1].mean(0).cpu().numpy()
                od = dar(irn, rgbn).clamp(0, 1)[0, 1].mean(0).cpu().numpy()
                tg = base(ir0, rgb0).clamp(0, 1)[0, 1].mean(0).cpu().numpy()
                panels = [ir0[0, 2].mean(0).cpu().numpy(), rgb0[0, 2].mean(0).cpu().numpy(),
                          rgbn[0, 2].mean(0).cpu().numpy(), ob, od, tg]
                H, W = ob.shape
                z = min(a.zoom, H, W)
                yy, xx = np.unravel_index(np.argmax(tg), tg.shape)
                cy = int(np.clip(yy, z // 2, H - z // 2))
                cx = int(np.clip(xx, z // 2, W - z // 2))
                qb, qd = qmap[i]
                for c in range(6):
                    ax = axes[2 * r, c]
                    ax.imshow(panels[c], cmap="gray", vmin=0, vmax=1)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.add_patch(plt.Rectangle((cx - z / 2, cy - z / 2), z, z,
                                               edgecolor="#00e5ff", lw=0.8, fill=False))
                    if r == 0:
                        ax.set_title(COLS[c], fontsize=9)
                    if c == 0:
                        ax.set_ylabel(f"scene {r + 1}", fontsize=8)
                    if c == 3:
                        ax.set_xlabel(f"$Q_{{AB/F}}$={qb:.3f}", fontsize=8)
                    if c == 4:
                        ax.set_xlabel(f"$Q_{{AB/F}}$={qd:.3f}  ({qd - qb:+.3f})", fontsize=8)
                    az = axes[2 * r + 1, c]
                    az.imshow(panels[c][cy - z // 2:cy + z // 2, cx - z // 2:cx + z // 2],
                              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    az.set_xticks([]); az.set_yticks([])
                    if c == 0:
                        az.set_ylabel("zoom", fontsize=8)
        fig.tight_layout(pad=0.15)
        fig.savefig(REPO / a.output, dpi=200, bbox_inches="tight")
        print(f"written {a.output}")
    finally:
        split.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
