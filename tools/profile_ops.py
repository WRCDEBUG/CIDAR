#!/usr/bin/env python3
"""Op-level CUDA profile of the MAVFusion inference path (torch.profiler).

Uses kineto rather than hand-rolled CUDA events, and labels the stages with
``record_function`` so the op table can be attributed to a stage.  Reports the
top ops by self CUDA time and the stage subtotals.
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
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402
from tools.step0_parity import load_window  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--seq", default="0111_1803")
    p.add_argument("--start", type=int, default=40)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--synthetic_hw", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/profile_ops_480p.json")
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

    def wrapped(label, fn):
        def inner(*a, **kw):
            with record_function(label):
                return fn(*a, **kw)

        return inner

    model.get_flow = wrapped("stage:get_flow", model.get_flow)
    model.generate_motion_mask_from_flow = wrapped(
        "stage:motion_mask", model.generate_motion_mask_from_flow
    )
    model.feat_extractor_ir.forward = wrapped("stage:feat_ir", model.feat_extractor_ir.forward)
    model.feat_extractor_vi.forward = wrapped("stage:feat_vi", model.feat_extractor_vi.forward)
    model.aam_ir.forward = wrapped("stage:aam_ir", model.aam_ir.forward)
    model.aam_vi.forward = wrapped("stage:aam_vi", model.aam_vi.forward)
    model.fusion_net.forward = wrapped("stage:fusion_net", model.fusion_net.forward)
    net_module.flow_warp = wrapped("op:flow_warp", net_module.flow_warp)

    with torch.no_grad():
        for _ in range(3):
            model(ir, vi)
        torch.cuda.synchronize()
        wall_start = torch.cuda.Event(enable_timing=True)
        wall_end = torch.cuda.Event(enable_timing=True)
        wall_start.record()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                     record_shapes=False) as prof:
            for _ in range(args.iterations):
                model(ir, vi)
            torch.cuda.synchronize()
        wall_end.record()
        torch.cuda.synchronize()

    n = args.iterations
    wall_ms = wall_start.elapsed_time(wall_end) / n

    key_avg = prof.key_averages()
    device_total_us = 0.0
    kernel_launches = 0.0
    cpu_total_us = 0.0
    op_rows = []
    stage_rows = []
    for evt in key_avg:
        is_stage = evt.key.startswith(("stage:", "op:"))
        dev_us = getattr(evt, "self_device_time_total", None)
        if dev_us is None or dev_us == 0:
            dev_us = getattr(evt, "self_cuda_time_total", 0) or 0
        cpu_us = getattr(evt, "self_cpu_time_total", 0) or 0
        if is_stage:
            stage_rows.append({
                "key": evt.key,
                "calls_per_window": evt.count / n,
                "cpu_ms_per_window": cpu_us / 1000.0 / n,
                "device_ms_per_window": dev_us / 1000.0 / n,
            })
            continue
        cpu_total_us += cpu_us
        if dev_us > 0:
            device_total_us += dev_us
            kernel_launches += evt.count
            op_rows.append({
                "key": evt.key,
                "self_cuda_ms_per_window": dev_us / 1000.0 / n,
                "calls_per_window": evt.count / n,
            })

    op_rows.sort(key=lambda r: -r["self_cuda_ms_per_window"])
    stage_rows.sort(key=lambda r: -r["cpu_ms_per_window"])

    device_ms = device_total_us / 1000.0 / n
    payload = {
        "resolution": f"{h}x{w}",
        "input_shape": list(ir.shape),
        "iterations": n,
        "wall_ms_per_window": wall_ms,
        "device_cuda_ms_per_window": device_ms,
        "gpu_busy_fraction": device_ms / wall_ms if wall_ms else None,
        "kernel_launches_per_window": kernel_launches / n,
        "cpu_self_ms_per_window": cpu_total_us / 1000.0 / n,
        "stage_cpu_ms_per_window": stage_rows,
        "top_ops_self_cuda_ms_per_window": op_rows[: args.top],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== {h}x{w} ===")
    print(f"  wall              {wall_ms:8.2f} ms/window")
    print(f"  device (CUDA)     {device_ms:8.2f} ms/window   -> GPU busy "
          f"{device_ms/wall_ms*100:.1f}%")
    print(f"  CPU self          {cpu_total_us/1000.0/n:8.2f} ms/window")
    print(f"  kernel launches   {kernel_launches/n:8.1f} per window")
    print("\n-- stages (CPU self time; wall-clock attribution including launch) --")
    for r in stage_rows:
        print(f"  {r['key']:20} cpu {r['cpu_ms_per_window']:7.2f} ms   "
              f"device {r['device_ms_per_window']:7.2f} ms   calls/window {r['calls_per_window']:5.2f}")
    print("\n-- top ops by self CUDA time --")
    for r in op_rows[: args.top]:
        print(f"  {r['key'][:58]:58} {r['self_cuda_ms_per_window']:7.2f} ms  "
              f"{r['calls_per_window']:7.1f} calls")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
