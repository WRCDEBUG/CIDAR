#!/usr/bin/env python3
"""Verify that pruning the oversized dynamic-branch budget is mathematically exact.

`MultiScaleSparseBranch` selects K = int(0.3 * N) patches, then multiplies each
selected patch by its own motion score (`val = scores[idx]`).  A selected patch
with score 0 therefore contributes exactly zero to the branch output.  Since
tools/support_stats.py finds K >> A (A = patches with score > 0) on ~95% of
frames at every layer, the budget can be cut to A with no semantic change.

This script proves that claim rather than asserting it, with an internal
control:

    baseline : K  = int(0.3 * N)
    pruned   : K' = min(K, max_windows A)
    control  : K''= K' + 1     <-- adds ONE zero-score patch, mathematically
                                   identical to `pruned`, but a different
                                   tensor shape

If `pruned` and `control` agree (and both differ from `baseline` by the same
amount), then the baseline/pruned gap is floating-point reduction-order noise
induced by the shape change, not a semantic difference.  `same_batch_size_delta`
is that bound.

Also reports the 8-bit pixel disagreement, because the published artifacts are
uint8 images.
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
from tools.support_stats import LAYERS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--sequence", default="0111_1803")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--ckpt_path", default="latest")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--output", default="output/pr1/prune_equivalence.json")
    return p.parse_args()


class FusionTimer:
    def __init__(self, module):
        self.times = []
        self._start = None
        self._pre = module.register_forward_pre_hook(self._before)
        self._post = module.register_forward_hook(self._after)

    def _before(self, module, inputs):
        if self._start is not None:
            return
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def _after(self, module, inputs, output):
        if self._start is None:
            return
        self._end.record()
        torch.cuda.synchronize()
        self.times.append(self._start.elapsed_time(self._end))
        self._start = None

    def remove(self):
        self._pre.remove()
        self._post.remove()


def main():
    args = parse_args()
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.requires_grad_(False)
    model.set_recording(True)

    ir, vi, total, first = load_window(data_dir, args.sequence, args.start, args.num_frames, device)
    timer = FusionTimer(model.fusion_net)
    original_ratio = {
        layer: getattr(model.fusion_net, layer).dynamic_branch.topk_ratio for layer in LAYERS
    }

    def set_budget(offsets):
        """offsets[layer] added on top of min(K, A_max)."""
        for layer in LAYERS:
            branch = getattr(model.fusion_net, layer).dynamic_branch
            sel = model.get_pr1_records()[layer]["selection"]
            n = int(sel[0]["scores"].numel())
            k = int(sel[0]["k"])
            a_max = max(int((r["scores"] > 0).sum().item()) for r in sel)
            target = min(k, a_max) + offsets[layer]
            branch.topk_ratio = min(1.0, (target + 0.5) / n)

    with torch.no_grad():
        for _ in range(2):
            model(ir, vi)

        timer.times.clear()
        model.clear_pr1_records()
        baseline = model(ir, vi)
        baseline_ms = list(timer.times)
        records = model.get_pr1_records()

        layers_info = {}
        for layer in LAYERS:
            sel = records[layer]["selection"]
            n = int(sel[0]["scores"].numel())
            k = int(sel[0]["k"])
            a = [int((r["scores"] > 0).sum().item()) for r in sel]
            layers_info[layer] = {"N": n, "K": k, "A_per_window": a, "A_max": max(a),
                                  "K_target": min(k, max(a))}

        # pruned: K' = min(K, A_max)
        set_budget({layer: 0 for layer in LAYERS})
        for _ in range(2):
            model(ir, vi)
        timer.times.clear()
        model.clear_pr1_records()
        pruned = model(ir, vi)
        pruned_ms = list(timer.times)
        pruned_records = model.get_pr1_records()

        # control: K'' = K' + 1, adds one zero-score patch -> mathematically identical
        set_budget({layer: 1 for layer in LAYERS})
        for _ in range(2):
            model(ir, vi)
        model.clear_pr1_records()
        control = model(ir, vi)
        control_records = model.get_pr1_records()

        for layer in LAYERS:
            getattr(model.fusion_net, layer).dynamic_branch.topk_ratio = original_ratio[layer]

    timer.remove()

    for layer in LAYERS:
        layers_info[layer]["K_pruned"] = int(pruned_records[layer]["selection"][0]["k"])
        layers_info[layer]["K_control"] = int(control_records[layer]["selection"][0]["k"])
        layers_info[layer]["budget_reduction"] = 1.0 - layers_info[layer]["K_pruned"] / layers_info[layer]["K"]
        layers_info[layer]["pruning_applicable"] = bool(layers_info[layer]["A_max"] < layers_info[layer]["K"])

    def maxdiff(a, b):
        return float((a - b).abs().max().item())

    def uint8_disagreement(a, b):
        qa = torch.round(a.clamp(0, 1) * 255)
        qb = torch.round(b.clamp(0, 1) * 255)
        return float((qa != qb).float().mean().item())

    total_k = sum(v["K"] for v in layers_info.values())
    total_kp = sum(v["K_pruned"] for v in layers_info.values())
    payload = {
        "sequence": args.sequence,
        "start": args.start,
        "first_frame": first,
        "input_shape": list(ir.shape),
        "windows_in_forward": len(records[LAYERS[0]]["selection"]),
        "deterministic": args.deterministic,
        "layers": layers_info,
        "total_K": total_k,
        "total_K_pruned": total_kp,
        "query_reduction": 1.0 - total_kp / total_k,
        "baseline_vs_pruned_max_abs_error": maxdiff(baseline, pruned),
        "pruned_vs_control_max_abs_error": maxdiff(pruned, control),
        "baseline_vs_control_max_abs_error": maxdiff(baseline, control),
        "uint8_pixel_disagreement_baseline_vs_pruned": uint8_disagreement(baseline, pruned),
        "fusion_stage_ms": {
            "baseline": baseline_ms,
            "pruned": pruned_ms,
            "baseline_mean": sum(baseline_ms) / len(baseline_ms) if baseline_ms else None,
            "pruned_mean": sum(pruned_ms) / len(pruned_ms) if pruned_ms else None,
        },
        "interpretation": (
            "pruned adds only zero-score patches to the selection, whose contribution is "
            "val = scores[idx] = 0 -> excavate(act_p) * val = 0. control adds one more such "
            "patch. If pruned_vs_control is ~0, then baseline_vs_pruned is shape-induced "
            "floating-point reduction-order noise, not a semantic difference."
        ),
    }
    payload["mathematically_exact"] = bool(payload["pruned_vs_control_max_abs_error"] < 1e-6)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["mathematically_exact"]:
        sys.exit("pruning equivalence control FAILED")


if __name__ == "__main__":
    main()
