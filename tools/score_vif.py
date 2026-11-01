#!/usr/bin/env python3
"""Compute VIF for every baseline with the pinned UniVF reference implementation.

The evaluation contract pins VIF to `src/util/metric.py::Metric_VIF` (pixel-based
VIF, 4 scales), not to the vendored MATLAB `vifvec`; the manifest records why
(the MATLAB implementation deviates by +0.10..+0.20 from the paper).  The MATLAB
contract still emits a VIF column for historical reasons, but it must not be used
for the comparison table.

This driver reads the frames the contract scores, averages VIF over F vs the
infrared and F vs the visible reference exactly like the reference implementation
(`Metric_VIF(F, I, V)` already averages the two), and reports the frame-equal
mean -- the same aggregation used for our own VIF row in
`output/pr1/merged/table1_local_vs_paper.csv`.

Output:
  output/pr1/baseline_metrics/vif_<method>_<dataset>.json
  output/pr1/baseline_metrics/vif_all.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.util.metric import Metric_VIF  # noqa: E402

RES = REPO / "output/pr1/fusion_results"
OUT = REPO / "output/pr1/baseline_metrics"
SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}
EXT = (".jpg", ".png", ".jpeg", ".bmp")


def expected_frames(dataset: str, step: int = 5) -> int:
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    root = REPO / "eval_data" / dataset / "ir"
    n = 0
    for seq in seqs:
        irs = [p for p in root.iterdir() if p.suffix.lower() in EXT and p.name.startswith(seq + "_")]
        vis = {p.name for p in (REPO / "eval_data" / dataset / "vis").iterdir()}
        n += len([p for p in sorted(irs)[::step] if p.name in vis])
    return n


def contract_frames(method: str, dataset: str, step_full: int = 5):
    """(seq, filename) pairs the contract will score, in the contract's order."""
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    fused = RES / method / dataset / "fused"
    if not fused.is_dir():
        return []
    n_fused = len(list(fused.iterdir()))
    step = 1 if n_fused <= 1.1 * expected_frames(dataset, step_full) else step_full
    out = []
    for seq in seqs:
        files = sorted(p for p in fused.iterdir()
                       if p.suffix.lower() in EXT and p.name.startswith(seq + "_"))
        out += [(seq, p) for p in files[::step]]
    return out


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float64)).to(device)[None, None]


def vif(method: str, dataset: str, device) -> dict | None:
    frames = contract_frames(method, dataset)
    if not frames:
        print(f"  skip {method}/{dataset}: no fused frames")
        return None
    ir_root = REPO / "eval_data" / dataset / "ir"
    vis_root = REPO / "eval_data" / dataset / "vis"
    vals, skipped = [], 0
    for seq, p in frames:
        f = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        i = cv2.imread(str(ir_root / p.name), cv2.IMREAD_GRAYSCALE)
        v = cv2.imread(str(vis_root / p.name), cv2.IMREAD_GRAYSCALE)
        if f is None or i is None or v is None:
            skipped += 1
            continue
        h, w = f.shape
        if i.shape != (h, w):
            i = cv2.resize(i, (w, h), interpolation=cv2.INTER_LINEAR)
        if v.shape != (h, w):
            v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
        with torch.no_grad():
            vals.append(float(Metric_VIF(to_tensor(f, device), to_tensor(i, device), to_tensor(v, device))))
    if not vals:
        return None
    res = {"VIF": float(np.mean(vals)), "frames": len(vals), "skipped": skipped,
           "implementation": "UniVF VF-Bench src/util/metric.py::Metric_VIF",
           "aggregation": "frame-equal"}
    (OUT / f"vif_{method}_{dataset}.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = sys.argv[1:] or sorted(p.name for p in RES.iterdir()
                                     if p.is_dir() and p.name != "univf")
    allv = {}
    for m in methods:
        for ds in SETS:
            print(f"  VIF {m}/{ds} ...", flush=True)
            r = vif(m, ds, device)
            if r:
                allv.setdefault(m, {})[ds] = r
                print(f"    {r['frames']} frames, VIF={r['VIF']:.4f}", flush=True)
    merged = OUT / "vif_all.json"
    # Rebuild the merged file from every per-dataset result on disk, so a run
    # that stops early (e.g. one method missing a dataset) never loses numbers
    # that were already computed.
    prev = {}
    for f in sorted(OUT.glob("vif_*_*.json")):
        stem = f.stem[len("vif_"):]
        method, _, ds = stem.rpartition("_")
        if method and ds in SETS:
            prev.setdefault(method, {})[ds] = json.loads(f.read_text())
    merged.write_text(json.dumps(prev, indent=2))
    print(f"written {merged} ({sum(len(v) for v in prev.values())} cells)")


if __name__ == "__main__":
    main()
