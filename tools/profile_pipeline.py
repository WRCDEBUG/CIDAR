#!/usr/bin/env python3
"""Component-level latency profile of the MAVFusion inference path.

Wraps each stage of ``VideoFusion.forward`` with CUDA event pairs (inclusive
timing, no per-call synchronisation) and reports milliseconds and share of the
5-frame window at two resolutions.  Also counts ``flow_warp`` calls, because
grid_sample on full-resolution feature maps is the prime suspect for the
alignment module's cost.

Nothing here changes the model; it only measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.model.net as net_module  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from tools.step0_parity import load_window  # noqa: E402


class Stage:
    __slots__ = ("name", "ms", "calls", "depth")

    def __init__(self, name):
        self.name = name
        self.ms = 0.0
        self.calls = 0


def timed(label, register):
    """Decorate a callable so each call's GPU time is accumulated under label."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            register.setdefault(label, []).append((start, end))
            return out

        wrapper.__name__ = getattr(fn, "__name__", label)
        return wrapper

    return decorator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--seq", default="0111_1803")
    p.add_argument("--start", type=int, default=40)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument("--synthetic_hw", default=None,
                   help="e.g. 1024,1280 — replace the real window with random tensors "
                        "at this size to emulate another dataset resolution")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/profile_pipeline.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = Path("output") / args.exp_path / "checkpoint" / "latest" / "model.pth"
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.requires_grad_(False)

    ir, vi, _, _ = load_window(Path(args.data_dir), args.seq, args.start, 5, device)
    if args.synthetic_hw:
        sh, sw = (int(x) for x in args.synthetic_hw.split(","))
        ir = torch.rand(1, 5, 3, sh, sw, device=device)
        vi = torch.rand_like(ir)
    h, w = ir.shape[-2:]

    records: dict[str, list] = {}

    model.get_flow = timed("get_flow(SEA-RAFT)", records)(model.get_flow)
    model.generate_motion_mask_from_flow = timed("motion_mask", records)(
        model.generate_motion_mask_from_flow
    )
    model.feat_extractor_ir.forward = timed("feat_extractor_ir", records)(
        model.feat_extractor_ir.forward
    )
    model.feat_extractor_vi.forward = timed("feat_extractor_vi", records)(
        model.feat_extractor_vi.forward
    )
    model.aam_ir.forward = timed("aam_ir", records)(model.aam_ir.forward)
    model.aam_vi.forward = timed("aam_vi", records)(model.aam_vi.forward)
    model.fusion_net.forward = timed("fusion_net", records)(model.fusion_net.forward)
    net_module.flow_warp = timed("flow_warp (inside AAM)", records)(net_module.flow_warp)

    # full-forward timing
    full_events = []

    with torch.no_grad():
        for _ in range(3):
            model(ir, vi)
        records.clear()
        full_events.clear()
        for _ in range(args.iterations):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(ir, vi)
            e.record()
            full_events.append((s, e))
        torch.cuda.synchronize()

    def total(events):
        return sum(s.elapsed_time(e) for s, e in events)

    full_ms = total(full_events) / args.iterations
    stages = []
    for name, events in records.items():
        calls_per_window = len(events) / args.iterations
        stages.append(
            {
                "stage": name,
                "ms_per_window": total(events) / args.iterations,
                "calls_per_window": round(calls_per_window, 2),
                "share_of_full": (total(events) / args.iterations) / full_ms,
            }
        )
    stages.sort(key=lambda d: -d["ms_per_window"])

    payload = {
        "input_shape": list(ir.shape),
        "resolution": f"{h}x{w}",
        "iterations": args.iterations,
        "full_forward_ms": full_ms,
        "ms_per_fused_image": full_ms / 3,
        "stages": stages,
        "note": "stage times are inclusive (aam_* contain their flow_warp calls); "
                "flow_warp is listed separately as an exact subtotal",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== {h}x{w} · full forward {full_ms:.2f} ms ({full_ms/3:.2f} ms per fused image) ===")
    print(f"{'stage':26} {'ms/window':>10} {'calls':>7} {'share':>7}")
    for s in stages:
        print(f"{s['stage']:26} {s['ms_per_window']:10.2f} {s['calls_per_window']:7.2f} "
              f"{s['share_of_full']*100:6.1f}%")
    print(f"written {out}")


if __name__ == "__main__":
    main()
