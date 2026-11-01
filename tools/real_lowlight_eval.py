#!/usr/bin/env python3
"""External validation on REAL low-light video (LLVIP), not synthetic degradation.

LLVIP is captured at night: the visible modality has mean intensity 0.20 against
0.40 for the infrared, so it is genuinely dark rather than synthetically
darkened.  Consecutive file numbers are consecutive video frames (13 runs,
longest 500), which lets the 5-frame model be driven over real sequences.

What this measures
  Q_AB/F between the fused middle frame and the two REAL source frames, for
     baseline (released weights)
     DAR (predicted conditioning, i.e. the deployable variant)
  plus a paired sign test over windows, because the frozen 5-sequence holdout is
  too small for a meaningful paired test on its own.

No ground-truth fused image exists, so this is a no-reference comparison; it is
reported as external evidence, not as the primary endpoint.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.net import VideoFusion  # noqa: E402
from src.util import metric  # noqa: E402
from tools.stage2_dar import degrade  # noqa: E402  (identical degradation models)

LLVIP = Path(os.environ.get("LLVIP_ROOT", "data2/LLVIP"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=8)
    p.add_argument("--frames_per_run", type=int, default=25)
    p.add_argument("--stride", type=int, default=4, help="windows are taken every `stride` frames")
    p.add_argument("--dar_ckpt", default="output/pr1/stage2_predicted.pth")
    p.add_argument("--dar_mode", default="predicted")
    p.add_argument("--est_mode", default="base", choices=["base", "hf", "cross", "spectral", "crossspec", "nostride",
                            "temporal", "crosstemporal"],
                   help="must match the checkpoint: the estimator stem width depends on it")
    p.add_argument("--oracle_label", default=None,
                   help="for --dar_mode oracle, the one-hot label to feed, e.g. dark")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/real_lowlight.json")
    p.add_argument("--degrade", default="none",
                   help="comma list of degradations to ADD on top of the real frames "
                        "(none,noise10,noise20,dark,blur).  Adding one makes the test "
                        "well-posed: a clean reference genuinely exists, so it measures "
                        "content-domain transfer rather than 'should night be brightened'.")
    return p.parse_args()


def consecutive_runs(stems):
    nums = sorted(int(s) for s in stems if s.isdigit())
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n != prev + 1:
            runs.append((start, prev))
            start = n
        prev = n
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= 25]


def load_pair(a: int, device):
    ir = np.asarray(Image.open(LLVIP / "infrared" / "test" / f"{a}.jpg").convert("RGB"),
                    dtype=np.float32) / 255.0
    vi = np.asarray(Image.open(LLVIP / "visible" / "test" / f"{a}.jpg").convert("RGB"),
                    dtype=np.float32) / 255.0
    return (torch.from_numpy(ir).permute(2, 0, 1).to(device),
            torch.from_numpy(vi).permute(2, 0, 1).to(device))


def main():
    args = parse_args()
    device = torch.device(args.device)
    runs = consecutive_runs([p.stem for p in (LLVIP / "visible" / "test").glob("*.jpg")])
    chosen = runs[: args.n_runs]
    print(f"LLVIP runs available: {len(runs)}; using {len(chosen)}")

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = torch.load(REPO / "output" / "MAVFusion-M3SVD" / "checkpoint/latest/model.pth",
                      map_location=device)

    def build(dar):
        m = VideoFusion(model_config=cfg, dar=dar).to(device).eval()
        m.load_state_dict(ckpt, strict=False)
        return m

    base = build(None)
    dar = build({"mode": args.dar_mode, "est_mode": args.est_mode})
    dar_ckpt_path = REPO / args.dar_ckpt
    if dar_ckpt_path.exists():
        dar.load_state_dict(torch.load(dar_ckpt_path, map_location=device), strict=False)
        print(f"loaded DAR weights from {dar_ckpt_path}")
    else:
        print(f"[warn] {dar_ckpt_path} not found; DAR is at its identity initialisation")

    from tools.stage2_dar import cond_onehot  # noqa: E402  (same one-hot convention)
    deg_list = [d for d in args.degrade.split(",") if d] or ["none"]
    rows = []
    with torch.no_grad():
        for (a, b) in chosen:
            starts = list(range(a, min(b - 4, a + args.frames_per_run) + 1, args.stride))
            for s in starts:
                idx = list(range(s, s + 5))
                pairs = [load_pair(i, device) for i in idx]      # load each pair once
                # [1,5,3,H,W]: stack on dim 0 gives [5,3,H,W] for the time axis
                ir = torch.stack([q[0] for q in pairs], dim=0).unsqueeze(0)
                vi = torch.stack([q[1] for q in pairs], dim=0).unsqueeze(0)
                mid = 2
                ir_mid = torch.round(ir[:, mid] * 255)
                vi_mid = torch.round(vi[:, mid] * 255)
                rec = {"run": f"{a}-{b}", "start": s}
                for dg in deg_list:
                    gen = torch.Generator(device=device).manual_seed(4242)
                    ir_in = ir if dg == "none" else degrade(ir, dg, gen)
                    vi_in = vi if dg == "none" else degrade(vi, dg, gen)
                    for name, model in (("baseline", base), ("dar", dar)):
                        if model.dar_mode == "oracle" and args.oracle_label:
                            model.dar_cond_vec = cond_onehot(args.oracle_label, device)
                        out = model(ir_in, vi_in).clamp(0, 1)
                        pre = f"{name}_" if dg == "none" else f"{name}_{dg}_"
                        q = float(metric.Metric_Qabf(torch.round(out[:, 1] * 255), ir_mid, vi_mid).mean())
                        g = float(metric.Metric_VIF(torch.round(out[:, 1] * 255),
                                                    ir_mid, vi_mid).mean())
                        rec[f"{pre}qabf"] = q
                        rec[f"{pre}vif"] = g
                rows.append(rec)
            print(f"  run {a}-{b}: {len(starts)} windows done", flush=True)

    from math import comb
    summary = {}
    for dg in deg_list:
        pre = "baseline_" if dg == "none" else f"baseline_{dg}_"
        pre2 = "dar_" if dg == "none" else f"dar_{dg}_"
        bq = np.array([r[f"{pre}qabf"] for r in rows])
        dq = np.array([r[f"{pre2}qabf"] for r in rows])
        bg = np.array([r[f"{pre}vif"] for r in rows])
        dgs = np.array([r[f"{pre2}vif"] for r in rows])
        wins = int((dq > bq).sum())
        n = len(rows)
        k = max(wins, n - wins)
        p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
        summary[dg] = {"Qabf": {"baseline_mean": float(bq.mean()), "dar_mean": float(dq.mean()),
                                "delta": float((dq - bq).mean()), "wins": wins, "n": n,
                                "sign_test_p": p_sign},
                       "VIF": {"baseline_mean": float(bg.mean()), "dar_mean": float(dgs.mean()),
                               "delta": float((dgs - bg).mean())}}
    payload = {
        "dataset": "LLVIP (real low-light night video)", "n_windows": len(rows),
        "runs_used": [f"{a}-{b}" for a, b in chosen],
        "degradations": deg_list, "summary": summary,
        "dar_checkpoint": str(dar_ckpt_path), "dar_mode": args.dar_mode,
        "oracle_label": args.oracle_label,
        "per_window": rows,
    }
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== LLVIP real-content validation ({len(rows)} windows, DAR={args.dar_mode}) ===")
    print(f"{'added degradation':18} {'base Qabf':>10} {'DAR Qabf':>10} {'delta':>9} {'pct':>8} {'wins':>8} {'p':>8}")
    for dg in deg_list:
        q = summary[dg]["Qabf"]
        print(f"{dg:18} {q['baseline_mean']:10.4f} {q['dar_mean']:10.4f} {q['delta']:+9.4f} "
              f"{(q['dar_mean']/q['baseline_mean']-1)*100:+7.2f}% {q['wins']:5d}/{q['n']:<3d} "
              f"{q['sign_test_p']:8.4f}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
