#!/usr/bin/env python3
"""Export DAR's clean fused frames in the contract layout.

The Stage-2 training/evaluation script measured Qabf in memory and never wrote
fused images, which is why the clean-quality table had to fall back on the base
network's row.  This script closes that gap: it runs the released base network
plus a DAR checkpoint on the *clean* frozen holdout, at exactly the frames the
contract scores, and writes

    output/pr1/fusion_results/dar/<dataset>/fused/<seq>_<frame>.jpg

so that `tools/score_baselines.py dar` and `tools/score_vif.py dar` can score
DAR with the same eight metrics as every baseline.

Inference protocol is the one used when the base network's clean results were
produced (5-frame window centred on the scored frame, middle output frame), so
the only difference between the `dar` and `MAVFusion` rows is the rectifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402

SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}
YAML = {ds: f"config/dataset/IVF/{ds}/{ds}_5-frame.yaml" for ds in SETS}
EXT = (".jpg", ".png", ".jpeg", ".bmp")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="M3SVD,VTMOT,HDO")
    p.add_argument("--dar_ckpt", default="output/pr1/stage2_cross_noblur_long.pth")
    p.add_argument("--est_mode", default="cross")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out_root", default="output/pr1/fusion_results")
    return p.parse_args()


def sampled_names(dataset: str) -> set[str]:
    """<seq>_<frame> names the contract scores (every 5th, sorted)."""
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    root = REPO / "eval_data" / dataset / "ir"
    out = set()
    for seq in seqs:
        irs = [p for p in root.iterdir() if p.suffix.lower() in EXT and p.name.startswith(seq + "_")]
        vis = {p.name for p in (REPO / "eval_data" / dataset / "vis").iterdir()}
        for p in sorted(irs, key=lambda q: q.name)[::5]:
            if p.name in vis:
                out.add(f"{seq}_{p.stem[len(seq) + 1:]}")
    return out


def main():
    a = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    dar_cfg = {"mode": "predicted", "est_mode": a.est_mode, "input_restore": False,
               "second": False, "gate": False, "vq": 0}
    model = VideoFusion(model_config=cfg, dar=dar_cfg).to(device)
    base_ck = torch.load(REPO / "output/MAVFusion-M3SVD/checkpoint/latest/model.pth",
                         map_location=device)
    model.load_state_dict(base_ck, strict=False)
    dar_ck = torch.load(REPO / a.dar_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(dar_ck, strict=False)
    print(f"DAR ckpt: {len(dar_ck)} tensors; missing {len(missing)} unexpected {len(unexpected)}",
          flush=True)
    model.eval()

    for ds in [d for d in a.datasets.split(",") if d]:
        want = sampled_names(ds)
        if a.limit:
            want = set(sorted(want)[:a.limit])
        outdir = REPO / a.out_root / "dar" / ds / "fused"
        outdir.mkdir(parents=True, exist_ok=True)

        dc = OmegaConf.load(YAML[ds])
        split_name = "_tmp_export_split.json"
        (REPO / dc.csv_dir).mkdir(parents=True, exist_ok=True)
        man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
        (REPO / dc.csv_dir / split_name).write_text(json.dumps(
            {"train": [], "test": man["sequence_sets"][SETS[ds]]["sequences"]}))
        dc.split_filename = split_name
        ds_obj = get_multi_frame_dataset(dc, base_data_dir="data2", mode=DatasetMode.TEST)
        loader = DataLoader(ds_obj, batch_size=1, shuffle=False, num_workers=2)
        print(f"{ds}: dataset {len(ds_obj)} frames, {len(want)} to score", flush=True)

        done = 0
        for b in loader:
            ir_path = b["data_path_ls_dict"]["ir"][2][0]
            name = Path(ir_path).as_posix()
            seq, _, rest = name.partition("/")
            stem = Path(rest).stem
            key = f"{seq}_{stem}"
            if key not in want:
                continue
            dst = outdir / f"{key}.jpg"
            if dst.exists():
                done += 1
                continue
            ir = b["ir"].to(device)
            rgb = b["rgb"].to(device)
            with torch.no_grad():
                out = model(ir, rgb)
            f = out[0, 1].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            g = cv2.cvtColor((f * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cv2.imwrite(str(dst), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            done += 1
            if done % 50 == 0:
                print(f"  {ds}: {done}/{len(want)}", flush=True)
        (REPO / dc.csv_dir / split_name).unlink(missing_ok=True)
        print(f"dar {ds}: {done}/{len(want)} frames -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
