#!/usr/bin/env python3
"""Test whether the inference path is launch-bound by capturing a CUDA graph.

tools/profile_ops.py measures ~14.6k kernel launches per 5-frame window and a
GPU busy fraction near 35%, which points at launch overhead rather than compute.
A CUDA graph replays the identical kernels with a single launch per graph, so:

  * if graph replay is much faster  -> the gap was launch overhead, and there is
    a zero-quality-change speedup available;
  * if it is not faster             -> the model is genuinely compute-bound and
    the earlier reading was an artefact.

The graph path is bit-compared against eager to make sure nothing changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.net import VideoFusion  # noqa: E402
from tools.step0_parity import load_window  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--seq", default="0111_1803")
    p.add_argument("--start", type=int, default=40)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--synthetic_hw", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/cuda_graph_480p.json")
    return p.parse_args()


def bench(fn, n, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n


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

    eager_ms = bench(lambda: model(ir, vi), args.iterations)

    with torch.no_grad():
        reference = model(ir, vi).clone()

    static_ir = ir.clone()
    static_vi = vi.clone()
    status = "ok"
    graph_ms = None
    max_abs_error = None
    try:
        pool = torch.cuda.graphs.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # warm up on a side stream, as required before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(static_ir, static_vi)
        torch.cuda.current_stream().wait_stream(s)
        model.clear_pr1_records()
        with torch.cuda.graph(graph, pool=pool):
            static_out = model(static_ir, static_vi)
        model.clear_pr1_records()

        graph_ms = bench(lambda: graph.replay(), args.iterations, warmup=3)
        max_abs_error = float((static_out - reference).abs().max().item())

        # Essential check: the graph must be correct for *new* input data, not
        # just reproduce the tensor it was captured with.
        ir2, vi2, _, _ = load_window(Path(args.data_dir), args.seq, args.start + 60,
                                     ir.shape[1], device)
        if ir2.shape == ir.shape:
            with torch.no_grad():
                reference2 = model(ir2, vi2).clone()
            static_ir.copy_(ir2)
            static_vi.copy_(vi2)
            graph.replay()
            torch.cuda.synchronize()
            max_abs_error_new_input = float((static_out - reference2).abs().max().item())
            static_ir.copy_(ir)
            static_vi.copy_(vi)
        else:
            max_abs_error_new_input = None
    except Exception as exc:  # noqa: BLE001
        status = f"capture failed: {type(exc).__name__}: {exc}"
        max_abs_error_new_input = None

    payload = {
        "resolution": f"{h}x{w}",
        "input_shape": list(ir.shape),
        "iterations": args.iterations,
        "eager_ms_per_window": eager_ms,
        "eager_ms_per_fused_image": eager_ms / 3,
        "graph_ms_per_window": graph_ms,
        "graph_ms_per_fused_image": graph_ms / 3 if graph_ms else None,
        "speedup": eager_ms / graph_ms if graph_ms else None,
        "max_abs_error_graph_vs_eager": max_abs_error,
        "max_abs_error_graph_vs_eager_new_input": max_abs_error_new_input,
        "status": status,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
