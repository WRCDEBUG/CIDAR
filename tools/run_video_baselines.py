#!/usr/bin/env python3
"""Run the video-fusion baselines on the frozen holdout in the contract layout.

Three sequence-based methods are supported; each is loaded from its own release
with its own official preprocessing, and their temporal windows are respected:

  cmvf        -- per frame (CMVF is a spatial network; its official test also
                 fuses frame by frame, only wrapping the frames into an mp4)
  videofusion -- 7-frame clips (options/test/test_VideoFusion.yml: num_frame=7)
  temcoco     -- 10-frame clips (the window the released checkpoint was trained
                 with; its test config uses 30 frames with 5-frame overlap)

Everything downstream is identical to tools/bench_baselines.py: the fused
luminance is written as a 3-channel JPEG named <seq>_<frame>.jpg under
output/pr1/fusion_results/<method>/<dataset>/fused/, so the same MATLAB contract
and the same VIF implementation score it.

Usage
  python tools/run_video_baselines.py --method videofusion --datasets M3SVD
  python tools/run_video_baselines.py --method cmvf --datasets M3SVD,HDO,VTMOT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VB = REPO / "对比视频融合方法"
SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}
EXT = (".jpg", ".png", ".jpeg", ".bmp")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["cmvf", "videofusion", "temcoco", "ofvfusion"])
    p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
    p.add_argument("--limit", type=int, default=0, help="debug: first N frames per dataset")
    p.add_argument("--window", type=int, default=0, help="override the method's temporal window")
    p.add_argument("--out_root", default="output/pr1/fusion_results")
    return p.parse_args()


def natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def sequence_frames(dataset: str):
    """(seq, [(ir_path, vis_path, frame_stem), ...]) for the frozen holdout."""
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    ir_root = REPO / "eval_data" / dataset / "ir"
    vis_root = REPO / "eval_data" / dataset / "vis"
    out = []
    for seq in seqs:
        irs = [p for p in ir_root.iterdir()
               if p.suffix.lower() in EXT and p.name.startswith(seq + "_")]
        vis = {p.name: p for p in vis_root.iterdir() if p.suffix.lower() in EXT}
        items = []
        for p in sorted(irs, key=lambda q: natkey(q.stem)):
            q = vis.get(p.name)
            if q is not None:
                items.append((p, q, p.stem[len(seq) + 1:]))
        out.append((seq, items))
    return out


# --------------------------------------------------------------------------
def load_cmvf(device):
    sys.path.insert(0, str(VB / "CMVF-master" / "fusion"))
    from fusion_model import get_fusion_model

    model = get_fusion_model()
    sd = torch.load(str(VB / "CMVF-master" / "fusion" / "models" / "best_model.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    def fuse_window(pairs):
        """pairs: list of (ir_bgr, vis_bgr) uint8. Returns list of gray uint8."""
        outs = []
        for ir_bgr, vis_bgr in pairs:
            ir = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            vi_y = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
            ir_t = torch.from_numpy(ir)[None, None].repeat(1, 2, 1, 1).to(device)
            vi_t = torch.from_numpy(vi_y)[None, None].repeat(1, 2, 1, 1).to(device)
            with torch.no_grad():
                f = model(ir_t, vi_t).squeeze().float().cpu().numpy()
            outs.append((np.clip(f, 0, 1) * 255).round().astype(np.uint8))
        return outs

    return fuse_window, 1


def load_videofusion(device, window):
    sys.path.insert(0, str(VB / "VideoFusion-main"))
    from basicsr.archs.videofusion_Trans_arch import VideoFusion_Trans

    net = VideoFusion_Trans(
        in_channels=3, out_channels=3, num_features=32, num_blocks=[2, 2, 4],
        num_decoder_blocks=2, num_heads=[2, 4, 8], ffn_expansion_factor=1,
        bias=False, LayerNorm_type="WithBias", embed_dim=32, groups=[4, 8, 16])
    ck = torch.load(str(VB / "VideoFusion-main" / "experiments" / "VideoFusion" / "net_g_latest.pth"),
                    map_location="cpu")
    ck = ck.get("params", ck)
    net.load_state_dict(ck)
    net = net.to(device).eval()
    T = window or 7

    def fuse_window(pairs):
        ir = np.stack([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a, _ in pairs]).astype(np.float32) / 255.0
        vi = np.stack([cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for _, b in pairs]).astype(np.float32) / 255.0
        ir_t = torch.from_numpy(ir).permute(0, 3, 1, 2)[None].to(device)
        vi_t = torch.from_numpy(vi).permute(0, 3, 1, 2)[None].to(device)
        with torch.no_grad():
            out = net(ir_t, vi_t)["fusion"][0]           # [T,3,H,W] in [0,1]
        out = out.clamp(0, 1).cpu().numpy()
        return [cv2.cvtColor((f.transpose(1, 2, 0) * 255).round().astype(np.uint8),
                             cv2.COLOR_RGB2GRAY) for f in out]

    return fuse_window, T


def load_temcoco(device, window):
    sys.path.insert(0, str(VB / "TemCoCo-main"))
    from archs.Net import Fusion

    net = Fusion()
    sd = torch.load(str(VB / "TemCoCo-main" / "experiments" / "VideoFusion_IRnoise2" / "models"
                         / "VideoFusionNet_495000.pth"), map_location="cpu")
    net.load_state_dict(sd, strict=True)
    net = net.to(device).eval()
    T = window or 10

    def fuse_window(pairs):
        ir = np.stack([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a, _ in pairs]).astype(np.float32) / 255.0
        vi = np.stack([cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for _, b in pairs]).astype(np.float32) / 255.0
        ir_t = torch.from_numpy(ir).permute(0, 3, 1, 2)[None].to(device)
        vi_t = torch.from_numpy(vi).permute(0, 3, 1, 2)[None].to(device)
        with torch.no_grad():
            out = net(ir_t, vi_t)[0][0]                  # [3,T,H,W]
        out = out.clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()   # [T,H,W,3]
        return [cv2.cvtColor((f * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in out]

    return fuse_window, T


def load_ofvfusion(device, window):
    sys.path.insert(0, str(VB / "OFVFusion-main"))
    from Models.OFVFusion import OFVFusion

    net = OFVFusion()
    sd = torch.load(str(VB / "OFVFusion-main" / "Logs" / "20251216" / "083421"
                         / "best_network.pth"), map_location="cpu")
    net.load_state_dict(sd, strict=False)
    net = net.to(device).eval()
    T = window or 6

    FB = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=7,
              poly_sigma=1.5, flags=0)

    def farneback(frames_u8):
        """frames_u8: [T,H,W] uint8 -> [1,T-1,2,H,W] float, as in their dataset."""
        out = []
        for t in range(1, len(frames_u8)):
            a = cv2.GaussianBlur(frames_u8[t - 1], (5, 5), 0)
            b = cv2.GaussianBlur(frames_u8[t], (5, 5), 0)
            f = cv2.calcOpticalFlowFarneback(a, b, None, **FB)
            f[np.abs(f) > 20] = 0
            out.append(f)
        arr = np.transpose(np.stack(out), (0, 3, 1, 2))
        return torch.from_numpy(arr).float()[None].to(device)

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def fuse_window(pairs):
        irs, ys = [], []
        for ir_bgr, vis_bgr in pairs:
            ir = ir_bgr[:, :, 0].astype(np.float32) / 255.0
            b, g, r = [vis_bgr[:, :, i].astype(np.float32) / 255.0 for i in range(3)]
            vis = np.stack([r, g, b])                      # RGB
            vis = norm01(vis)
            y = 0.299 * vis[0] + 0.587 * vis[1] + 0.114 * vis[2]
            irs.append(norm01(ir))
            ys.append(y)
        ir_t = torch.from_numpy(np.stack(irs))[:, None].float()[None].to(device)   # [1,T,1,H,W]
        y_t = torch.from_numpy(np.stack(ys))[:, None].float()[None].to(device)
        ir_u8 = (np.stack(irs) * 255).astype(np.uint8)
        y_u8 = (np.stack(ys) * 255).astype(np.uint8)
        flow_ir = farneback(ir_u8)
        flow_vi = farneback(y_u8)
        with torch.no_grad():
            out = net(ir_t, y_t, flow_ir, flow_vi)[0][0]   # [T,1,H,W]
        out = out[:, 0].clamp(0, 1).cpu().numpy()           # [T,H,W]
        return [(f * 255).round().astype(np.uint8) for f in out]

    return fuse_window, T


def main():
    a = parse()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.method == "cmvf":
        fuse_window, T = load_cmvf(dev)
    elif a.method == "videofusion":
        fuse_window, T = load_videofusion(dev, a.window)
    elif a.method == "temcoco":
        fuse_window, T = load_temcoco(dev, a.window)
    else:
        fuse_window, T = load_ofvfusion(dev, a.window)
    mult = 32 if a.method in ("videofusion", "temcoco") else 1
    print(f"{a.method}: window={T}, pad_multiple={mult}, device={dev}", flush=True)

    for ds in [d for d in a.datasets.split(",") if d]:
        outdir = REPO / a.out_root / a.method / ds / "fused"
        outdir.mkdir(parents=True, exist_ok=True)
        total = done = 0
        for seq, items in sequence_frames(ds):
            if a.limit:
                items = items[:a.limit]
            total += len(items)
            for start in range(0, len(items), T):
                chunk = items[start:start + T]
                if not chunk:
                    continue
                dsts = [outdir / f"{seq}_{stem}.jpg" for _, _, stem in chunk]
                if all(p.exists() for p in dsts):
                    done += len(chunk)
                    continue
                irs, vis = [], []
                for pir, pvis, _ in chunk:
                    ir = cv2.imread(str(pir), cv2.IMREAD_COLOR)
                    vi = cv2.imread(str(pvis), cv2.IMREAD_COLOR)
                    if ir is None or vi is None:
                        raise RuntimeError(f"cannot read {pir} / {pvis}")
                    if ir.shape != vi.shape:
                        vi = cv2.resize(vi, (ir.shape[1], ir.shape[0]), interpolation=cv2.INTER_LINEAR)
                    irs.append(ir)
                    vis.append(vi)
                # VideoFusion/TemCoCo downsample by 2 repeatedly, so pad odd
                # sizes (e.g. VTMOT's 481x631) to a multiple of 32 and crop back
                h0, w0 = irs[0].shape[:2]
                if mult > 1 and (h0 % mult or w0 % mult):
                    H = (h0 + mult - 1) // mult * mult
                    W = (w0 + mult - 1) // mult * mult
                    irs = [cv2.copyMakeBorder(x, 0, H - h0, 0, W - w0, cv2.BORDER_REFLECT_101) for x in irs]
                    vis = [cv2.copyMakeBorder(x, 0, H - h0, 0, W - w0, cv2.BORDER_REFLECT_101) for x in vis]
                # pad a short final window by repeating the last frame
                pad = T - len(chunk)
                if pad:
                    irs += [irs[-1]] * pad
                    vis += [vis[-1]] * pad
                fused = [g[:h0, :w0] for g in fuse_window(list(zip(irs, vis)))[:len(chunk)]]
                for dst, g in zip(dsts, fused):
                    if not dst.exists():
                        cv2.imwrite(str(dst), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 100])
                done += len(chunk)
                if done % 50 < T:
                    print(f"  {ds}/{seq}: {done}/{total}", flush=True)
        print(f"{a.method} {ds}: {done}/{total} frames -> {outdir}", flush=True)
    print("add ir/vis with: python tools/stage_method_outputs.py --method <m> --dataset <ds>",
          flush=True)


if __name__ == "__main__":
    main()
