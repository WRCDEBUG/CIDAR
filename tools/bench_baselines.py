#!/usr/bin/env python3
"""Unified comparison harness for third-party image-fusion baselines.

Why this exists: the paper needs the comparison table a TIP reviewer expects
(fusion methods x datasets x the eight metrics), and the fused frames themselves
are worth keeping because they can be reused for future papers.  So this driver
does three things and nothing else:

  1. enumerates the exact frames the evaluation contract scores (the frozen
     five-sequence holdout of each dataset, every --sample_step frames),
  2. runs one third-party method on those pairs, writing
        output/pr1/fusion_results/<method>/<dataset>/fused/<seq>_<frame>.jpg
     with the naming the metric contract expects, and
  3. is resumable: existing outputs are skipped, so a crash costs only the
     frames in flight.

The fused frames are ordinary PNG/JPEG files, so they can be re-scored with any
metric later without re-running the network.

Conventions that make the table defensible
------------------------------------------
* Every method receives the *luminance* channel of the visible frame and the
  infrared frame as uint8 grayscale, at the infrared frame's resolution (the
  visible frame is resized only when the pair is not co-registered).
* Every method writes the *luminance* of its fusion result as a 3-channel JPEG,
  because the MATLAB contract reads F with rgb2gray().  Saving Y directly and
  saving a YCbCr-recoloured image both yield the same rgb2gray(F) up to
  clipping, so this keeps all methods on one scale.
* The output normalisation of each method follows its own official test script
  (min-max where the release normalises, clamp-only where it does not); the
  method-specific line is stated in its adapter below.

Environments
------------
Most adapters need only torch/cv2/numpy.  FusionMamba and S4Fusion need
mamba_ssm + selective_scan_cuda, which live in the `ifcaf` conda env, so those
two must be launched with that interpreter (tools/run_baselines.sh does it).
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}
EXT = (".jpg", ".png", ".jpeg", ".bmp")

TB = Path(os.environ.get("TB_ROOT", "third_party/DuiBiShiYan"))
THIRD = Path(os.environ.get("THIRD_ROOT", "third_party/Others"))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
    p.add_argument("--sample_step", type=int, default=5)
    p.add_argument("--out_root", default="output/pr1/fusion_results")
    p.add_argument("--limit", type=int, default=0, help="debug: only N frames per dataset")
    p.add_argument("--batch", type=int, default=0,
                   help="reserved; adapters are single-frame so memory stays predictable")
    return p.parse_args()


def frame_list(dataset, sample_step, limit=0):
    """The frames the metric contract actually scores, as (seq, stem, ir, vis)."""
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    root = REPO / "eval_data" / dataset
    items = []
    for seq in seqs:
        irs = sorted([p for p in (root / "ir").iterdir() if p.suffix.lower() in EXT and p.name.startswith(seq + "_")])
        vis = {p.name: p for p in (root / "vis").iterdir() if p.suffix.lower() in EXT}
        keep = irs[::sample_step]
        if limit:
            keep = keep[:limit]
        for p in keep:
            q = vis.get(p.name)
            if q is not None:
                items.append((seq, p.stem[len(seq) + 1:], p, q))
    return items


def _t(x, device):
    """uint8 HxW -> float32 [1,1,H,W] in [0,1]."""
    return torch.from_numpy(x.astype(np.float32) / 255.0)[None, None].to(device)


def _to_u8(x):
    """[H,W] float tensor in [0,1] -> uint8 HxW."""
    return (x.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _pad_to(x, multiple):
    h, w = x.shape[-2], x.shape[-1]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
    return x, h, w


def load_method(name, device):
    """Return a callable (ir_bgr_uint8, vis_bgr_uint8) -> fused uint8 (H, W)."""
    # ---------------------------------------------------------------- CDDFuse
    if name == "cddfuse":
        sys.path.insert(0, str(TB / "CDDFuse_old"))
        from net import (Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction,
                         DetailFeatureExtraction)
        ck = torch.load(str(TB / "CDDFuse_old/models/CDDFuse_IVF.pth"), map_location=device)
        enc = torch.nn.DataParallel(Restormer_Encoder()).to(device)
        dec = torch.nn.DataParallel(Restormer_Decoder()).to(device)
        base = torch.nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
        det = torch.nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
        enc.load_state_dict(ck["DIDF_Encoder"]); dec.load_state_dict(ck["DIDF_Decoder"])
        base.load_state_dict(ck["BaseFuseLayer"]); det.load_state_dict(ck["DetailFuseLayer"])
        for m in (enc, dec, base, det):
            m.eval()

        def fuse(ir, vi):
            a = _t(ir, device)
            b = _t(vi, device)
            with torch.no_grad():
                fb, fd, _ = enc(b)
                ib, idd, _ = enc(a)
                x = dec(b, det(fd + idd), base(fb + ib))[0]
                x = _minmax(x)
            return _to_u8(x[0, 0])

        return fuse

    # ------------------------------------------------------------------ TarDAL
    # Official infer_f.py feeds [0,1] grayscale and saves the generator output
    # clipped to [0,1] (no min-max).  The released tardal-dt.pth carries
    # use_eval=False and infer_f.py honours it, so the generator is kept in
    # train mode: eval mode (running BN statistics) produces a much darker,
    # infrared-dominated image and collapses QABF from 0.44 to 0.24.  We use
    # batch 1 instead of the official 4 so a frame's score cannot depend on
    # which other frames happen to share its batch (measured effect: <0.01).
    if name == "tardal":
        root = TB / "TarDAL-main"
        sys.path.insert(0, str(root))
        from module.fuse.generator import Generator
        ck = torch.load(str(root / "weights/v1/tardal-dt.pth"), map_location="cpu")
        ck = ck.get("fuse", ck)
        ck.pop("use_eval", None)
        gen = Generator(dim=32, depth=3)
        gen.load_state_dict(ck)
        gen = gen.to(device).train()

        def fuse(ir, vi):
            with torch.no_grad():
                o = gen(_t(ir, device), _t(vi, device))
            return _to_u8(o[0, 0])

        return fuse

    # ------------------------------------------------------------------- IGNet
    # Official test.py: visible YCrCb (only Y is used) + infrared -> fused Y,
    # then clip to [0,1] and min-max normalise (batch size 1 in the release).
    # test.py never calls .eval(), so the network runs in train mode.
    if name == "ignet":
        root = THIRD / "IGNet-master"
        sys.path.insert(0, str(root))
        from model_gnn import backbone
        from rgb2ycbcr import RGB2YCrCb
        m = backbone()
        m.load_state_dict(torch.load(str(root / "model/IGNet.pth"), map_location=device))
        m = m.to(device).train()

        def fuse(ir, vi):
            vis3 = np.repeat(vi[:, :, None], 3, axis=2)
            a = torch.from_numpy(vis3.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            b = _t(ir, device)
            with torch.no_grad():
                y = RGB2YCrCb(a)
                o = m(y, b)
                o = _minmax(o.clamp(0, 1))
            return _to_u8(o[0, 0])

        return fuse

    # ---------------------------------------------------------------- TDFusion
    # Official test.py: Y of both inputs, concatenated on the channel axis,
    # then min-max normalise the output.  We use the M3FD checkpoint for every
    # dataset (the release only ships per-dataset checkpoints and M3FD is the
    # closest general-purpose infrared/visible pair).
    if name == "tdfusion":
        root = TB / "TDFusion-main"
        sys.path.insert(0, str(root))
        from nets.ReFusion import ReFusion
        m = ReFusion()
        m.load_state_dict(torch.load(str(root / "models/TDFusion_M3FD.pth"), map_location="cpu"))
        m = m.to(device).eval()

        def fuse(ir, vi):
            x = torch.cat([_t(ir, device), _t(vi, device)], dim=1)
            with torch.no_grad():
                o = _minmax(m(x))
            return _to_u8(o[0, 0])

        return fuse

    # -------------------------------------------------------------- FusionMamba
    # Official test.py: pad to a multiple of 32, clamp to [0,1], then min-max.
    if name == "fusionmamba":
        root = TB / "FusionMamba-main"
        sys.path.insert(0, str(root))
        from models.vmamba_Fusion_efficross import VSSM_Fusion
        m = VSSM_Fusion(in_channel=1)
        m.load_state_dict(torch.load(str(root / "model_last/my_cross/model.pth"), map_location="cpu"))
        m = m.to(device).eval()

        def fuse(ir, vi):
            a, h, w = _pad_to(_t(ir, device), 32)
            b, _, _ = _pad_to(_t(vi, device), 32)
            with torch.no_grad():
                o = m(a, b)
            return _to_u8(_minmax(o.clamp(0, 1))[0, 0, :h, :w])

        return fuse

    # ---------------------------------------------------------------- S4Fusion
    # Official test.py: visible Y + infrared, padded to the 12k+1 grid its
    # down/up-sampling requires, output cropped back and saved clamped.
    if name == "s4fusion":
        root = TB / "S4Fusion-main"
        sys.path.insert(0, str(root))
        from S4Fusion import MambaNet
        ck = torch.load(str(root / "model/model.pkl"), map_location="cpu")
        m = MambaNet()
        m.load_state_dict(ck["model"])
        m = m.to(device).eval()

        def rest_of(x):
            add = 0
            while (x - 1) % 3 != 0 or ((x - 1) // 3) % 4 != 0:
                add += 1
                x += 1
            return add

        def fuse(ir, vi):
            a = _t(ir, device)
            b = _t(vi, device)
            h, w = a.shape[-2:]
            rh, rw = rest_of(h), rest_of(w)
            l1, l2 = rh // 2, rw // 2
            a = torch.nn.functional.pad(a, (l2, rw - l2, l1, rh - l1), value=0.0)
            b = torch.nn.functional.pad(b, (l2, rw - l2, l1, rh - l1), value=0.0)
            with torch.no_grad():
                # test.py calls model(Y_visible, ir)
                o = m(b, a)
            if isinstance(o, (list, tuple)):
                o = o[0]
            if o.dim() == 4:
                o = o[0]
            return _to_u8(o[0, l1:l1 + h, l2:l2 + w])

        return fuse

    # --------------------------------------------------------------- SeAFusion
    # Official test.py: visible Y + infrared -> fused Y in [0,1], then recoloured
    # with the visible chroma; test.py never calls .eval(), but the network's
    # BatchNorm is in the decoder only and train/eval give bit-identical output
    # here (checked on a frame: MAE 0.00).  We therefore keep train mode to
    # match the released script.  Saving Y directly equals rgb2gray of the
    # recoloured image, because the visible inputs used here are luminance.
    if name == "seafusion":
        root = THIRD / "SeAFusion-main"
        sys.path.insert(0, str(root))
        from FusionNet import FusionNet
        m = FusionNet(output=1)
        m.load_state_dict(torch.load(str(root / "model/Fusion/fusionmodel_final.pth"),
                                     map_location="cpu"))
        m = m.to(device).train()

        def fuse(ir, vi):
            with torch.no_grad():
                o = m(_t(vi, device), _t(ir, device))
            return _to_u8(o[0, 0])

        return fuse

    # ----------------------------------------------------------------- CoCoNet
    # Official test.py: inputs in [0,1], output tanh scaled by 127.5+127.5 and
    # written with cv2 (saturating cast), so no min-max.  Like IGNet, main.py's
    # test branch never calls .eval(); with the released epoch-1 checkpoint the
    # running BN statistics are stale, so eval mode washes the image out and
    # collapses QABF from 0.43 to 0.12.  Train mode is what the authors ran.
    if name == "coconet":
        root = THIRD / "CoCoNet-main"
        sys.path.insert(0, str(root))
        from models.model import Unet_resize_conv
        ck = torch.load(str(root / "logs/latest.pth"), map_location="cpu")
        m = Unet_resize_conv()
        m.load_state_dict(ck["state_dict"])
        m = m.to(device).train()

        def fuse(ir, vi):
            with torch.no_grad():
                o = m(_t(vi, device), _t(ir, device))[0, 0]
            return ((o * 127.5 + 127.5).clamp(0, 255)).round().cpu().numpy().astype(np.uint8)

        return fuse

    raise SystemExit(f"unknown method {name}")


def main():
    a = parse()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fuse = load_method(a.method, dev)
    for ds in [d for d in a.datasets.split(",") if d]:
        items = frame_list(ds, a.sample_step, a.limit)
        outdir = REPO / a.out_root / a.method / ds / "fused"
        outdir.mkdir(parents=True, exist_ok=True)
        # keep the sources next to the outputs so the pair is never separated
        for sub in ("ir", "vis"):
            (REPO / a.out_root / a.method / ds / sub).mkdir(parents=True, exist_ok=True)
        done = 0
        for seq, stem, pir, pvis in items:
            dst = outdir / f"{seq}_{stem}.jpg"
            if dst.exists():
                done += 1
                continue
            ir = cv2.imread(str(pir), cv2.IMREAD_GRAYSCALE)
            vi = cv2.imread(str(pvis), cv2.IMREAD_GRAYSCALE)
            if ir is None or vi is None:
                continue
            if ir.shape != vi.shape:
                vi = cv2.resize(vi, (ir.shape[1], ir.shape[0]), interpolation=cv2.INTER_LINEAR)
            f = fuse(ir, vi)
            # the metric contract reads with rgb2gray(), so every image must be
            # 3-channel; a single-channel JPEG makes rgb2gray fail on the MAP
            cv2.imwrite(str(dst), cv2.cvtColor(f, cv2.COLOR_GRAY2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            for p, sub in ((pir, "ir"), (pvis, "vis")):
                t = REPO / a.out_root / a.method / ds / sub / p.name
                if not t.exists():
                    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(str(t), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 100])
            done += 1
            if done % 50 == 0:
                print(f"  {ds}: {done}/{len(items)}", flush=True)
        print(f"{a.method} {ds}: {done}/{len(items)} frames -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
