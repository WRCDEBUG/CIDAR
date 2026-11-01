#!/usr/bin/env python3
"""Deployment package: measure the accelerated inference path end to end.

Variants
  baseline : shipped forward (4 SEA-RAFT calls per 5-frame window)
  fast     : flow_budget="half"        (2 calls; measured 1.18x, dQabf 0.0000)
             + CUDA Graph replay       (measured 1.14x, bit-identical)
             + flow_direction="corrected"  (fixes the proven reversed
                                            motion-compensation direction)

The fast path keeps the same outputs as closely as possible; the metrics are
then computed from the saved JPEGs by tools/contract_metrics.m (6 image metrics)
and tools/run_univf_metrics.py (VIF, MS2R), i.e. the same endpoints as the rest
of the project.

Per-resolution CUDA graphs are cached, because HDO and VTMOT mix resolutions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.util.io import pred_2_8bit, save_image  # noqa: E402

YAML = {"M3SVD": "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml",
        "HDO": "config/dataset/IVF/HDO/HDO_5-frame.yaml",
        "VTMOT": "config/dataset/IVF/VTMOT/VTMOT_5-frame.yaml"}
EXP = {"M3SVD": "MAVFusion-M3SVD", "HDO": "MAVFusion-HDO", "VTMOT": "MAVFusion-M3SVD"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="M3SVD")
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--variants", default="baseline,fast",
                   help="use fast alone to skip the baseline: eval_data/<ds> already holds it")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="output/pr1/deploy")
    p.add_argument("--limit", type=int, default=0, help="0 = all windows")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def build_loader(args):
    man = json.loads((REPO / args.manifest).read_text())
    seqs = man["sequence_sets"][man["datasets"][args.dataset]["default_set"]]["sequences"]
    dc = OmegaConf.load(YAML[args.dataset])
    name = "_deploy_selected.json"
    (Path(dc.csv_dir) / name).write_text(json.dumps({"train": [], "test": seqs}))
    dc.split_filename = name
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
    idx = list(range(len(ds)))
    if args.limit:
        idx = idx[: args.limit]
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=args.num_workers)
    return loader, seqs, Path(dc.csv_dir) / name


class GraphCache:
    """One CUDA graph per input resolution, so mixed-resolution datasets work."""

    def __init__(self, model):
        self.model = model
        self.graphs = {}

    def get(self, shape, device):
        key = tuple(shape[2:])
        if key in self.graphs:
            return self.graphs[key]
        static_ir = torch.zeros(shape, device=device)
        static_vi = torch.zeros(shape, device=device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.model(static_ir, static_vi)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.model(static_ir, static_vi)
        self.graphs[key] = (graph, static_ir, static_vi, static_out)
        print(f"    captured graph for {key}", flush=True)
        return self.graphs[key]


def main():
    args = parse_args()
    device = torch.device(args.device)
    loader, seqs, split_path = build_loader(args)
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    out_root = REPO / args.output_dir
    report = {"dataset": args.dataset, "sequences": seqs, "windows": len(loader.dataset), "variants": {}}

    try:
        for variant in args.variants.split(","):
            model = VideoFusion(model_config=cfg).to(device).eval()
            model.load_state_dict(torch.load(REPO / "output" / EXP[args.dataset] / "checkpoint/latest/model.pth",
                                             map_location=device))
            model.requires_grad_(False)
            use_graph = variant == "fast"
            if variant == "baseline":
                model.flow_budget, model.flow_direction = "full", "released"
            else:
                model.flow_budget, model.flow_direction = "half", "corrected"
            graphs = GraphCache(model) if use_graph else None

            # per-dataset sub-directory: a shared <variant>/fused path would let a
            # second dataset overwrite the first one's outputs
            vis_dir = out_root / args.dataset / variant / "fused"
            if vis_dir.exists():
                shutil.rmtree(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)
            for modality in ("ir", "vis"):
                link = out_root / args.dataset / variant / modality
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink((REPO / "eval_data" / args.dataset / modality).resolve(), link)

            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            total_ms = 0.0
            total_frames = 0
            counted = 0
            wall0 = time.time()
            print(f"  running {variant} ...", flush=True)
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    ir = batch["ir"].to(device)
                    rgb = batch["rgb"].to(device)
                    if use_graph:
                        graph, sir, svi, sout = graphs.get(ir.shape, device)
                        sir.copy_(ir)
                        svi.copy_(rgb)
                        torch.cuda.synchronize()
                        s.record()
                        graph.replay()
                        e.record()
                        torch.cuda.synchronize()
                        pred = sout
                    else:
                        torch.cuda.synchronize()
                        s.record()
                        pred = model(ir, rgb)
                        e.record()
                        torch.cuda.synchronize()
                    if i > 5:
                        total_ms += s.elapsed_time(e)
                        total_frames += pred.shape[1]
                        counted += 1
                    vis = pred_2_8bit(pred, ir, rgb)
                    mid = ir.shape[1] // 2
                    p = batch["data_path_ls_dict"]["ir"][mid][0]
                    seq = os.path.dirname(p).split(os.sep)[0]
                    stem = os.path.splitext(os.path.basename(p))[0]
                    save_image(vis, str(vis_dir / f"{seq}_{stem}.jpg"))
                    if (i + 1) % 200 == 0:
                        print(f"    {i+1}/{len(loader.dataset)}", flush=True)

            entry = {"windows": counted, "output_frames": total_frames,
                     "avg_forward_ms": total_ms / max(counted, 1),
                     "ms_per_fused_image": total_ms / max(total_frames, 1),
                     "fps": 1000.0 * total_frames / total_ms if total_ms else None,
                     "wall_seconds": time.time() - wall0,
                     "saved_images": len(list(vis_dir.glob("*.jpg"))),
                     "flow_budget": model.flow_budget, "flow_direction": model.flow_direction,
                     "cuda_graph": use_graph,
                     "fused_dir": str(vis_dir.relative_to(REPO))}
            report["variants"][variant] = entry
            print(json.dumps(entry, indent=2))
            del model, graphs
            torch.cuda.empty_cache()
    finally:
        split_path.unlink(missing_ok=True)

    if "baseline" in report["variants"] and "fast" in report["variants"]:
        b = report["variants"]["baseline"]["avg_forward_ms"]
        f = report["variants"]["fast"]["avg_forward_ms"]
        report["speedup"] = b / f
    out = out_root / f"deploy_report_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n=== {args.dataset} ===")
    for k, v in report["variants"].items():
        print(f"  {k:9} {v['avg_forward_ms']:8.1f} ms/window  {v['fps']:7.2f} FPS  "
              f"{v['ms_per_fused_image']:7.2f} ms/image")
    if "speedup" in report:
        print(f"  speedup: {report['speedup']:.3f}x")
    print(f"written {out}")


if __name__ == "__main__":
    main()
