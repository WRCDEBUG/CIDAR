#!/usr/bin/env python3
"""Stage a method's own output tree into the flat layout the contract reads.

Sources
  1. --src <per-seq tree>   flatten <seq>/<frame>.jpg -> fused/<seq>_<frame>.jpg
     (the layout written by a method's own test.py, e.g. VF-Bench's
      eval_visual/IVF-M3SVD/<seq>/<frame>.jpg)
  2. --flat-src <dir>       take an already flat directory of frames
  3. neither                use the existing <out>/fused directory as is

Options
  --panel {left,middle,right}
     Some implementations save a qualitative triptych [ir | vis | fused] instead
     of the fused image alone (VF-Bench's pred_2_8bit does exactly this).  The
     contract compares F against the *single* fused frame, so the panel must be
     cropped out before scoring.  Use --panel right for [ir | vis | fused].

  --move-src-to <dir>
     When the triptychs are already in fused/, move them aside first (preserved
     as evidence) and re-read them from there.

The contract (tools/contract_metrics.m) reads <root>/ir, <root>/vis and
<root>/fused with the *same* file name in all three, so ir/vis links are added
for every holdout frame.  Symlinks only; no pixels are duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
EXT = (".jpg", ".png", ".jpeg", ".bmp")
SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(os.path.abspath(src), dst)


def frozen_frames(dataset: str) -> dict[str, Path]:
    """name -> eval_data/ir path for every frame of the holdout sequences."""
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    root = REPO / "eval_data" / dataset / "ir"
    out = {}
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in EXT and any(p.name.startswith(s + "_") for s in seqs):
            out[p.name] = p
    return out


def crop_panel(img, panel: str):
    h, w = img.shape[:2]
    # the triptychs are built by np.concatenate(..., axis=1), so the split is exact
    w0 = w // 3
    x0 = {"left": 0, "middle": w0, "right": 2 * w0}[panel]
    return img[:, x0:x0 + w0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True, choices=list(SETS))
    ap.add_argument("--src", default=None, help="per-sequence tree to flatten")
    ap.add_argument("--flat-src", default=None, help="flat directory of named frames")
    ap.add_argument("--panel", choices=["left", "middle", "right"], default=None,
                    help="crop this third out of a [ir | vis | fused] triptych")
    ap.add_argument("--move-src-to", default=None,
                    help="move the existing fused/ directory here before rewriting it")
    ap.add_argument("--out_root", default="output/pr1/fusion_results")
    a = ap.parse_args()

    dst = REPO / a.out_root / a.method / a.dataset
    fused = dst / "fused"
    fused.mkdir(parents=True, exist_ok=True)

    flat_src: Path | None = Path(a.flat_src) if a.flat_src else None
    if a.move_src_to:
        target = dst / a.move_src_to
        if fused.is_dir() and any(fused.iterdir()) and not target.is_dir():
            fused.rename(target)
            fused.mkdir(parents=True, exist_ok=True)
            flat_src = target
            print(f"moved existing fused frames -> {target}")
        elif target.is_dir():
            flat_src = target

    written = 0
    if a.src:
        src = Path(a.src)
        for seq_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            for f in sorted(seq_dir.iterdir()):
                if f.suffix.lower() not in EXT:
                    continue
                out = fused / f"{seq_dir.name}_{f.stem}.jpg"
                if not out.exists():
                    img = cv2.imread(str(f))
                    if a.panel:
                        img = crop_panel(img, a.panel)
                    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    written += 1
    elif flat_src:
        for f in sorted(flat_src.iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            out = fused / f"{f.stem}.jpg"
            if not out.exists():
                img = cv2.imread(str(f))
                if img is None:
                    continue
                if a.panel:
                    img = crop_panel(img, a.panel)
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                written += 1
    if written:
        print(f"wrote {written} frames (panel={a.panel})")

    have = {p.name for p in fused.iterdir() if p.suffix.lower() in EXT}
    src_ir = frozen_frames(a.dataset)
    vis_root = REPO / "eval_data" / a.dataset / "vis"
    linked = 0
    for name in sorted(have):
        if name in src_ir:
            link(src_ir[name], dst / "ir" / name)
            link(vis_root / name, dst / "vis" / name)
            linked += 1
    print(f"{a.method}/{a.dataset}: fused={len(have)} ir/vis linked={linked} "
          f"({len(src_ir)} holdout frames available)")
    missing = len(have) - linked
    if missing:
        print(f"  note: {missing} fused frames have no eval_data source "
              f"(non-holdout frames; the contract ignores them)")


if __name__ == "__main__":
    main()
