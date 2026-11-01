#!/usr/bin/env python3
"""Turn "identifiability" from a narrative claim into a measurement.

The paper's design rule is that conditioned rectification can only recover what
the descriptor can identify from the input.  A reviewer is entitled to ask how we
know what a descriptor can and cannot identify, and the answer should not be a
story about downsampling.  This script measures it:

    take the degradation descriptor, fit a LINEAR classifier to predict which
    degradation was applied, and report the accuracy.

A linear probe is the right instrument because it asks whether the information is
present in the representation, not whether a larger network could recover it.
Configurations compared:

    stem only          the released estimator stem
    + cross-modal      both modalities into one estimator
    spectral stats     the hand-computed multi-scale energies, no network at all
    stem+cross+spectral

We also report the binary blur-vs-rest accuracy, which is the quantity the blur
fix has to move.
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
from src.model.dar import DegradationEstimator  # noqa: E402
from tools.stage2_dar import degrade  # noqa: E402

YAML = "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml"
DEGS = ("clean", "noise10", "noise20", "dark", "blur")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--windows", type=int, default=150)
    p.add_argument("--window_stride", type=int, default=25)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/identifiability_probe.json")
    return p.parse_args()


def build_loader(args):
    man = json.loads((REPO / args.manifest).read_text())
    seqs = man["sequence_sets"]["frozen5_test"]["sequences"]
    dc = OmegaConf.load(YAML)
    nm = "_probe.json"
    (Path(dc.csv_dir) / nm).write_text(json.dumps({"train": [], "test": seqs}))
    dc.split_filename = nm
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    idx = list(range(0, len(ds), args.window_stride))[: args.windows]
    return DataLoader(Subset(ds, idx), batch_size=1, shuffle=False), Path(dc.csv_dir) / nm


def fit_probe(Xtr, ytr, Xte, yte, n_cls, steps=600):
    clf = torch.nn.Linear(Xtr.shape[1], n_cls).to(Xtr.device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = clf(Xte).argmax(1)
        acc = (pred == yte).float().mean().item()
        # binary blur-vs-rest
        blur = DEGS.index("blur")
        tb = yte == blur
        pb = pred == blur
        acc_b = ((pb == tb).float().mean().item()) if tb.any() else float("nan")
    return acc, acc_b


def main():
    args = parse_args()
    device = torch.device(args.device)
    loader, split_path = build_loader(args)
    cfg = OmegaConf.load("config/train/ivf-train.yaml")

    # configurations to probe: (label, dar config, checkpoint or None)
    CONFIGS = [
        ("stem only (untrained)", {"mode": "predicted", "est_mode": "base"}, None),
        ("stem only (trained, 4 degs)", {"mode": "predicted", "est_mode": "base"},
         "output/pr1/stage2_predicted.pth"),
        ("cross-modal (trained, 3 degs)", {"mode": "predicted", "est_mode": "cross"},
         "output/pr1/stage2_cross_noblur_long.pth"),
        ("+ spectral (trained, 4 degs)", {"mode": "predicted", "est_mode": "crossspec"},
         "output/pr1/stage2_crossspec_long.pth"),
        ("no-stride stem (control)", {"mode": "predicted", "est_mode": "nostride"},
         "output/pr1/stage2_nostride_long.pth"),
        ("DDC (oracle mixed in)", {"mode": "predicted", "est_mode": "crossspec"},
         "output/pr1/stage2_ddc_mix.pth"),
        ("DDC + temporal", {"mode": "predicted", "est_mode": "crosstemporal"},
         "output/pr1/stage2_ddc_temporal.pth"),
        ("spectral stats only (no net)", None, "spec"),
        ("temporal stats only (no net)", None, "temp"),
        ("spectral + temporal (no net)", None, "both"),
    ]

    feats: dict[str, list] = {c[0]: [] for c in CONFIGS}
    labels = []
    try:
        with torch.no_grad():
            for i, b in enumerate(loader):
                ir0 = b["ir"].to(device)
                rgb0 = b["rgb"].to(device)
                for di, d in enumerate(DEGS):
                    gen = torch.Generator(device=device).manual_seed(999)
                    ir = degrade(ir0, d, gen)
                    rgb = degrade(rgb0, d, gen)
                    for label, dcfg, ckpt in CONFIGS:
                        if dcfg is None:
                            # hand-computed statistics, no network at all
                            x = torch.cat([ir, rgb], dim=2)
                            bb, t, c, h, w = x.shape
                            xx = x.reshape(bb * t, c, h, w)
                            parts = []
                            if ckpt in ("spec", "both"):
                                parts.append(DegradationEstimator._spectral_stats(xx)
                                             .reshape(bb, t, -1).mean(1))
                            if ckpt in ("temp", "both"):
                                parts.append(DegradationEstimator._temporal_stats(x))
                            st = torch.cat(parts, dim=1)
                            feats[label].append(st[0].cpu())
                            continue
                        model = getattr(main, "_cache", None)
                        key = (label,)
                        cache = getattr(main, "_cache", {}) or {}
                        if key not in cache:
                            # fixed seed: an UNTRAINED descriptor's probe accuracy
                            # otherwise depends on the random initialisation, which
                            # made the earlier untrained rows unreproducible
                            torch.manual_seed(1234)
                            m = VideoFusion(model_config=cfg, dar=dcfg).to(device).eval()
                            m.load_state_dict(torch.load(REPO / "output/MAVFusion-M3SVD/"
                                                         "checkpoint/latest/model.pth",
                                                         map_location=device), strict=False)
                            has_ckpt = bool(ckpt and (REPO / ckpt).exists())
                            if has_ckpt:
                                m.load_state_dict(torch.load(REPO / ckpt, map_location=device),
                                                  strict=False)
                            main._trained = getattr(main, "_trained", {})
                            main._trained[label] = has_ckpt
                            cache[key] = m
                            main._cache = cache
                        m = cache[key]
                        cond = m.dar_estimator(ir, rgb)
                        feats[label].append(cond[0].cpu())
                    labels.append(di)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1} windows", flush=True)
    finally:
        split_path.unlink(missing_ok=True)

    y = torch.tensor(labels)
    n_cls = len(DEGS)
    # split by window so the probe cannot memorise a window
    n = len(y)
    tr = torch.zeros(n, dtype=torch.bool)
    tr[::2] = True
    te = ~tr

    results = {}
    for label, _, _ in CONFIGS:
        X = torch.stack(feats[label]).float()
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        acc, acc_blur = fit_probe(X[tr].to(device), y[tr].to(device),
                                  X[te].to(device), y[te].to(device), n_cls, args.steps)
        trained = getattr(main, "_trained", {}).get(label, None)
        results[label] = {"dim": int(X.shape[1]),
                          "accuracy_all": round(acc, 4),
                          "accuracy_blur_vs_rest": round(acc_blur, 4),
                          "checkpoint_found": trained}
        flag = "" if trained is not False else "   [no checkpoint: untrained]"
        print(f"  {label:32} dim={X.shape[1]:4d}  acc={acc:.3f}  "
              f"blur-vs-rest={acc_blur:.3f}{flag}", flush=True)

    payload = {"windows": n, "classes": list(DEGS), "chance_all": 1 / n_cls,
               "chance_blur_vs_rest": 0.5, "results": results}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
