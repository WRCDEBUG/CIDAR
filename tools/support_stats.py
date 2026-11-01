#!/usr/bin/env python3
"""Motion-support / budget statistics for every fusion layer (Step 0 diagnostic).

For each sampled 5-frame window this records, per layer:

    N  number of patches
    K  the layer's fixed query budget (int(0.3 * N))
    A  patches with motion score > 0
    selected_in_support / zero_score_selected
    recipients  patches with motion > 0 that are NOT selected
                (== the candidate pool ECR's one-out/one-in exchange needs)
    tie statistics at the Top-K boundary

This is the measurement the ECR plan depends on: a budget-conserving exchange
requires ``recipients > 0``.  It costs one forward pass per window and no
training, and it is the number that decides whether the method can even be
executed as specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.net import VideoFusion  # noqa: E402
from tools.step0_parity import list_frames, load_window  # noqa: E402

LAYERS = ("e1", "e2", "e3", "bt", "dec3", "dec2", "dec1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data2/M3SVD")
    p.add_argument("--sequences", default=None, help="comma list; default = all sequences in data_dir")
    p.add_argument("--windows_per_seq", type=int, default=2)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--exp_path", default="MAVFusion-M3SVD")
    p.add_argument("--ckpt_path", default="latest")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="output/pr1")
    return p.parse_args()


def summarize(scores: torch.Tensor, active_idx: torch.Tensor, k: int):
    s = scores.flatten().float()
    idx = active_idx.flatten().long()
    n = int(s.numel())
    sorted_scores = torch.sort(s, descending=True).values
    boundary = sorted_scores[k - 1]
    tie_level = int(torch.count_nonzero(s == boundary).item())
    active_scores = s.index_select(0, idx)
    tie_in = int(torch.count_nonzero(active_scores == boundary).item())
    supports = torch.nonzero(s > 0, as_tuple=False).flatten()
    selected = torch.zeros(n, dtype=torch.bool)
    selected[idx] = True
    selected_in_support = int(selected[supports].sum().item()) if supports.numel() else 0
    a = int(supports.numel())
    return {
        "N": n,
        "K": int(k),
        "A": a,
        "A_lt_K": bool(a < k),
        "A_over_K": float(a / k) if k else float("nan"),
        "wasted": float(torch.count_nonzero(active_scores == 0).item() / k) if k else float("nan"),
        "zero_score_selected": int(k - selected_in_support),
        "selected_in_support": selected_in_support,
        "recipients": int(a - selected_in_support),
        "tie_boundary": bool(n > k and bool(sorted_scores[k - 1] == sorted_scores[k])),
        "tie_level": tie_level,
        "tie_in": tie_in,
        "tie_out": tie_level - tie_in,
        "interchangeable": bool(tie_in > 0 and (tie_level - tie_in) > 0),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    if args.sequences:
        sequences = [s for s in args.sequences.split(",") if s]
    else:
        sequences = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    ckpt = Path("output") / args.exp_path / "checkpoint" / args.ckpt_path / "model.pth"
    if not ckpt.exists():
        sys.exit(f"Missing checkpoint: {ckpt}")
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.requires_grad_(False)
    model.set_recording(True)

    frame_rows = []
    per_video = defaultdict(lambda: defaultdict(list))
    window_count = defaultdict(int)
    summary_rows = []

    with torch.no_grad():
        for seq in sequences:
            total = len(list_frames(data_dir / seq / "infrared"))
            if total < args.num_frames:
                print(f"[skip] {seq}: only {total} frames")
                continue
            if args.windows_per_seq == 1:
                starts = [0]
            else:
                step = max(1, (total - args.num_frames) // (args.windows_per_seq - 1))
                starts = [min(i * step, total - args.num_frames) for i in range(args.windows_per_seq)]

            for start in starts:
                ir, vi, _, first = load_window(data_dir, seq, start, args.num_frames, device)
                model.clear_pr1_records()
                model(ir, vi)
                records = model.get_pr1_records()
                window_count[seq] += 1
                line = []
                for layer in LAYERS:
                    for rec in records[layer]["selection"]:
                        stats = summarize(rec["scores"].cpu(), rec["active_idx"].cpu(), rec["k"])
                        frame_rows.append({"video": seq, "start": start, "first_frame": first,
                                           "layer": layer, **stats})
                        for key in ("A", "K", "N", "wasted", "recipients", "A_over_K"):
                            per_video[seq][f"{layer}.{key}"].append(float(stats[key]))
                        for key in ("A_lt_K", "tie_boundary", "interchangeable"):
                            per_video[seq][f"{layer}.{key}"].append(float(stats[key]))
                    rec0 = records[layer]["selection"][0]
                    st0 = summarize(rec0["scores"].cpu(), rec0["active_idx"].cpu(), rec0["k"])
                    line.append(f"{layer}:A={st0['A']}/K={st0['K']}")
                summary_rows.append(f"  {seq}@{start}  " + " ".join(line))
                print(summary_rows[-1])

    by_video = []
    for video in sorted(per_video):
        row = {"video": video, "windows": window_count[video]}
        for key, values in sorted(per_video[video].items()):
            row[key] = sum(values) / len(values)
        by_video.append(row)

    video_equal = {}
    for layer in LAYERS:
        for suffix in ("N", "K", "A", "A_over_K", "wasted", "recipients",
                       "A_lt_K", "tie_boundary", "interchangeable"):
            vals = [row.get(f"{layer}.{suffix}") for row in by_video]
            vals = [v for v in vals if v is not None]
            video_equal[f"{layer}.{suffix}"] = sum(vals) / len(vals) if vals else None

    payload = {
        "dataset": data_dir.name,
        "sequences": sequences,
        "windows_per_seq": args.windows_per_seq,
        "batch_size": 1,
        "layers": LAYERS,
        "video_equal": video_equal,
        "by_video": by_video,
        "frames": frame_rows,
        "key_finding": {
            "dec1_A_lt_K_video_equal_rate": video_equal.get("dec1.A_lt_K"),
            "dec1_recipients_video_equal": video_equal.get("dec1.recipients"),
            "dec1_wasted_video_equal": video_equal.get("dec1.wasted"),
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "support_stats.json").write_text(json.dumps(payload, indent=2))
    fields = ["video", "windows"] + sorted(k for k in by_video[0] if k not in {"video", "windows"}) if by_video else ["video"]
    with (out_dir / "support_stats_by_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_video)

    print("\n=== video-equal statistics ===")
    print(f"{'layer':6} {'N':>7} {'K':>7} {'A':>7} {'A/K':>6} {'wasted':>7} {'recipients':>11} {'A<K':>5} {'interchg':>9}")
    for layer in LAYERS:
        print(f"{layer:6} {video_equal[f'{layer}.N']:7.0f} {video_equal[f'{layer}.K']:7.0f} "
              f"{video_equal[f'{layer}.A']:7.1f} {video_equal[f'{layer}.A_over_K']:6.3f} "
              f"{video_equal[f'{layer}.wasted']:7.3f} {video_equal[f'{layer}.recipients']:11.1f} "
              f"{video_equal[f'{layer}.A_lt_K']:5.2f} {video_equal[f'{layer}.interchangeable']:9.2f}")
    print(f"\nwritten: {out_dir/'support_stats.json'}")
    print(f"written: {out_dir/'support_stats_by_video.csv'}")


if __name__ == "__main__":
    main()
