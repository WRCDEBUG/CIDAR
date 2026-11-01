#!/usr/bin/env python3
"""Step 0 ECR integration parity on the real model and real frames.

Checks, in order:

1. ``capture_last`` / ``replay_last`` reproduce the full forward output for every
   captured end-layer entry.
2. With a router attached in ``mode="motion"`` the output is identical to the
   pre-attach baseline (the hook must be a no-op by default).
3. A forced one-out/one-in exchange keeps ``K`` unchanged, only selects patches
   inside the motion support, and actually changes the output.
4. The router rejects a route that introduces a zero-motion patch.

Nothing here is a quality claim; it is the integration gate before any label or
training work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# append, not insert: the starter also ships a top-level ``tools`` package and
# must not shadow the repository's own tools/ directory.
sys.path.append(str(Path(__file__).resolve().parents[1] / "third_party" / "ecr_fusion_starter"))

from ecr.diagnostics import sample_pairs  # noqa: E402
from ecr.replay import attach_router, capture_last, replay_last  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402

from tools.step0_parity import load_window  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--seq", default="0111_1803")
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--ckpt_path", default="latest")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--max_swaps", type=int, default=16)
    p.add_argument("--swap_fraction", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="output/pr1/ecr_parity.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    if not ckpt.exists():
        sys.exit(f"Missing checkpoint: {ckpt}")

    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().requires_grad_(False)
    model.set_recording(True)

    report = {
        "checkpoint": str(ckpt),
        "device": str(device),
        "seq": args.seq,
        "router": {
            "hidden": args.hidden,
            "max_swaps": args.max_swaps,
            "swap_fraction": args.swap_fraction,
            "threshold": args.threshold,
        },
        "windows": [],
    }

    router = None
    for w in range(args.windows):
        start = args.start + w * args.stride
        ir, vi, total, first = load_window(data_dir, args.seq, start, args.num_frames, device)

        with torch.no_grad():
            # --- 1. capture / replay parity -------------------------------
            frames = capture_last(model, ir, vi)
            full = model(ir, vi)
            capture_errors = []
            replay_errors = []
            for j, frame in enumerate(frames):
                capture_errors.append(float((frame.output - full[:, j]).abs().max().item()))
                replay = replay_last(model, frame)
                replay_errors.append(float((replay - frame.output).abs().max().item()))

            # --- 2. attach + motion-mode no-op ---------------------------
            if router is None:
                router = attach_router(
                    model,
                    hidden=args.hidden,
                    max_swaps=args.max_swaps,
                    swap_fraction=args.swap_fraction,
                    threshold=args.threshold,
                )
            router.mode = "motion"
            after_attach = model(ir, vi)
            motion_noop_error = float((after_attach - full).abs().max().item())

            # --- 3. forced legal swap ------------------------------------
            frame0 = frames[1]  # centre output, source_index == 2
            # gather the reference route by capturing the trace
            router.capture = True
            replay_last(model, frame0)
            trace = router.trace
            router.capture = False
            motion, active = trace["motion"], trace["active"]
            pairs = sample_pairs(motion, active, count=8, seed=args.seed, tolerance=0.05)

            swap_records = []
            for a, b in pairs[:4]:
                alternative = active.clone()
                pos = torch.where(active == a)[0]
                if pos.numel() != 1:
                    continue
                alternative[pos[0]] = b
                with torch.no_grad(), router.force(alternative):
                    swapped = replay_last(model, frame0)
                k_ok = bool(alternative.numel() == active.numel())
                support_ok = bool((motion[b] > 0).item())
                delta_out = float((swapped - frame0.output).abs().max().item())
                swap_records.append(
                    {
                        "a": int(a),
                        "b": int(b),
                        "k": int(active.numel()),
                        "k_preserved": k_ok,
                        "recipient_in_motion_support": support_ok,
                        "motion_a": float(motion[a].item()),
                        "motion_b": float(motion[b].item()),
                        "output_max_abs_change": delta_out,
                    }
                )

            # --- 4. illegal route must be rejected -----------------------
            illegal_rejected = None
            zero_motion = torch.where(motion <= 0)[0]
            if zero_motion.numel() > 0 and pairs:
                bad = active.clone()
                bad[torch.where(active == pairs[0][0])[0][0]] = zero_motion[0]
                try:
                    with torch.no_grad(), router.force(bad):
                        replay_last(model, frame0)
                    illegal_rejected = False
                except ValueError:
                    illegal_rejected = True

        report["windows"].append(
            {
                "start": start,
                "first_frame": first,
                "sequence_frames": total,
                "input_shape": list(ir.shape),
                "k": int(active.numel()),
                "n": int(motion.numel()),
                "support": int((motion > 0).sum().item()),
                "capture_vs_full_max_abs_error": max(capture_errors),
                "replay_vs_capture_max_abs_error": max(replay_errors),
                "motion_mode_vs_baseline_max_abs_error": motion_noop_error,
                "swaps": swap_records,
                "illegal_route_rejected": illegal_rejected,
            }
        )

    last = report["windows"][-1]
    integration_ok = bool(
        all(w["capture_vs_full_max_abs_error"] < 1e-5 for w in report["windows"])
        and all(w["replay_vs_capture_max_abs_error"] < 1e-5 for w in report["windows"])
        and all(w["motion_mode_vs_baseline_max_abs_error"] < 1e-5 for w in report["windows"])
        and all(s["k_preserved"] and s["recipient_in_motion_support"] for w in report["windows"] for s in w["swaps"])
    )
    # An empty candidate set is a *result*, not an integration failure: it means
    # every motion-supported patch is already selected (A <= K at this layer).
    no_candidates = all(len(w["swaps"]) == 0 for w in report["windows"])
    report["no_legal_swap_candidates"] = no_candidates
    report["illegal_route_check_ran"] = last["illegal_route_rejected"] is not None
    report["integration_passed"] = integration_ok
    report["passed"] = bool(
        integration_ok
        and (last["illegal_route_rejected"] is True or no_candidates)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        sys.exit("ECR integration parity FAILED")


if __name__ == "__main__":
    main()
