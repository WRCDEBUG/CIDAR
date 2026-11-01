#!/usr/bin/env python3
"""Stage 2: the degradation-aware rectifier (DAR) and all its controls.

The bar set by Stage 1: a BLIND full-model fine-tune (265,573 trainable params,
everything except the frozen flow net) recovers 12.07% of the degradation
collapse on average, 29.38% on low-light and 9.68% on noise10, with low-light
being the case that clearly clears the 15% bar.  Blur recovers 0.31% because a
5x5 blur destroys high-frequency information irreversibly.

Every DAR variant here is SMALLER than that blind fine-tune (222k-253k params),
so "a smaller conditional module beats a larger blind fine-tune" is a claim
about conditioning rather than about capacity.

Variants
  oracle     true degradation one-hot fed in        -> upper bound of conditioning
  predicted  descriptor estimated from the input    -> the actual method
  noconv     same rectifier with the conditioning path removed -> is it conditioning
             that helps, or just extra capacity?
  input_restore  rectifier applied to the RAW frames instead of the features
             -> is feature-side rectification better than "clean the input first"?

Unseen-degradation probes (never used for training): noise15 and dark22.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import get_multi_frame_dataset  # noqa: E402
from src.dataset.base_two_modal_dataset import DatasetMode  # noqa: E402
from src.model.net import VideoFusion  # noqa: E402
from src.model.dar import count_params  # noqa: E402
from src.util import metric  # noqa: E402

YAML = "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml"
EXP = "MAVFusion-M3SVD"
COND_ORDER = ("clean", "noise10", "noise20", "dark", "blur")
ALL_DEGS = ("noise10", "noise20", "dark", "blur")
EVAL_DEGS = ("clean", "noise10", "noise20", "dark", "blur", "noise15", "dark22")
# Day 3/4 reference on the frozen 5 at stride 25 (clean sources as reference)
REF = {"clean": 0.7364, "noise10": 0.5823, "noise20": 0.4777, "dark": 0.5638,
       "blur": 0.2162, "noise15": None, "dark22": None}
# Stage-1 blind fine-tune, the bar to beat
BLIND = {"mean": 12.07, "clean": -1.48, "noise10": 9.68, "noise20": 8.90,
         "dark": 29.38, "blur": 0.31}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dar_mode", default="predicted",
                   choices=["off", "oracle", "predicted", "noconv"])
    p.add_argument("--input_restore", action="store_true")
    p.add_argument("--est_mode", default="base", choices=["base", "hf", "cross", "spectral", "crossspec", "nostride",
                            "temporal", "crosstemporal"],
                   help="base = original estimator stem; hf = add a high-frequency "
                        "channel (the stem downsamples 4x and cannot see blur); "
                        "cross = hf plus both modalities in one estimator (the IR/VIS "
                        "ratio disambiguates a darkened scene from a dark one)")
    p.add_argument("--dar_second", action="store_true",
                   help="also rectify at the fusion-network input (second injection point)")
    p.add_argument("--unfreeze", default="dar_extractors",
                   choices=["dar", "dar_extractors", "all_except_flow"])
    p.add_argument("--train_degradations", default=",".join(ALL_DEGS),
                   help="leave-one-out support, e.g. noise10,noise20,dark")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clean_prob", type=float, default=0.5)
    p.add_argument("--dar_vq", type=int, default=0,
                   help="quantise the descriptor into K prototypes before FiLM; motivated by "
                        "the measurement that a discrete oracle condition works where a "
                        "continuous descriptor with the same information does not")
    p.add_argument("--dar_gate", action="store_true",
                   help="scale the rectifier residual by a learned degradation gate so the "
                        "module can abstain on clean frames (real-video validation showed a "
                        "3.5-4.1% loss on clean input without it)")
    p.add_argument("--gate_w", type=float, default=0.5,
                   help="weight of the gate supervision (binary: was this step degraded?)")
    p.add_argument("--oracle_mix", type=float, default=0.0,
                   help="demand-driven conditioning: with this probability per step, feed the "
                        "TRUE descriptor to the rectifier instead of the estimated one, so the "
                        "rectification loss becomes sensitive to descriptor quality")
    p.add_argument("--aux_reg", type=float, default=0.0,
                   help="weight of an auxiliary regression of the true degradation "
                        "parameters (noise sigma, gamma) on the descriptor")
    p.add_argument("--aux_ce", type=float, default=0.0,
                   help="weight of the auxiliary degradation-classification loss on the estimator")
    p.add_argument("--eval_every", type=int, default=1500)
    p.add_argument("--eval_stride", type=int, default=40)
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--train_split", default="data_split/IVF/M3SVD/split_train25.json")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_full", action="store_true",
                   help="save the whole state dict, needed when the run also trains "
                        "modules outside DAR (e.g. a matched blind fine-tune baseline) so "
                        "it can be re-evaluated on real data")
    p.add_argument("--tag", default="dar")
    p.add_argument("--output", default=None)
    p.add_argument("--ckpt", default=None)
    return p.parse_args()


def degrade(x, kind, gen):
    if kind == "clean":
        return x
    if kind.startswith("noise"):
        sigma = {"noise10": 10.0, "noise15": 15.0, "noise20": 20.0}[kind]
        return (x + torch.randn(x.shape, generator=gen, device=x.device) * (sigma / 255.0)).clamp(0, 1)
    if kind == "dark":
        return x.clamp(0, 1) ** 1.8
    if kind == "dark22":
        return x.clamp(0, 1) ** 2.2
    if kind == "darkrand":
        # Domain-randomised photometric low-light.  A fixed gamma is a monotone
        # tone map that the rectifier can simply learn to invert, which is why the
        # fixed-gamma branch did not transfer to real LLVIP night footage.  Drawing
        # gamma, gain and a little sensor noise per step forces a correction that is
        # not tied to one specific transform.
        g = 1.2 + 2.6 * float(torch.rand(1, generator=gen, device=x.device))
        s_ = 0.5 + 0.65 * float(torch.rand(1, generator=gen, device=x.device))
        y = (x.clamp(0, 1) ** g) * s_
        y = y + torch.randn(x.shape, generator=gen, device=x.device) * (3.0 / 255.0)
        return y.clamp(0, 1)
    if kind == "blur":
        k = torch.ones(1, 1, 5, 5, device=x.device) / 25.0
        b, t, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * t, c, h, w), k.repeat(c, 1, 1, 1), padding=2, groups=c)
        return y.reshape(b, t, c, h, w)
    raise ValueError(kind)


# (noise sigma, gamma) actually applied by each training degradation
DEG_PARAMS = {"clean": (0.0, 1.0), "noise10": (10 / 255, 1.0), "noise20": (20 / 255, 1.0),
              "noise15": (15 / 255, 1.0), "dark": (0.0, 1.8), "dark22": (0.0, 2.2),
              "darkrand": (0.0, 1.8), "blur": (0.0, 1.0)}


def cond_index(kind):
    """Map a training degradation to its descriptor class ('darkrand' is 'dark')."""
    if kind == "darkrand" or kind == "dark22":
        return COND_ORDER.index("dark")
    return COND_ORDER.index(kind)


def cond_onehot(kind, device):
    v = torch.zeros(1, len(COND_ORDER), device=device)
    if kind in COND_ORDER:
        v[0, COND_ORDER.index(kind)] = 1.0     # unseen types stay all-zero = "unknown"
    return v


def build_split(args, frozen):
    p = REPO / args.train_split
    p.parent.mkdir(parents=True, exist_ok=True)
    all_seqs = sorted(d.name for d in (REPO / "data2/M3SVD").iterdir() if d.is_dir())
    train = [s for s in all_seqs if s not in set(frozen)]
    p.write_text(json.dumps({"train": train, "test": list(frozen)}, indent=2))
    return train, p


def make_loader(split_path, mode, stride=1):
    dc = OmegaConf.load(YAML)
    dc.split_filename = split_path.name
    ds = get_multi_frame_dataset(dc, base_data_dir="data2", mode=mode)
    idx = list(range(0, len(ds), stride))
    return ds, DataLoader(Subset(ds, idx), batch_size=1, shuffle=(mode == DatasetMode.TRAIN),
                          num_workers=2, drop_last=True)


def set_trainable(model, which):
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []
    if which in ("dar", "dar_extractors", "all_except_flow"):
        for n in ("dar_ir", "dar_vi", "dar_ir2", "dar_vi2", "dar_estimator", "input_restorer"):
            mod = getattr(model, n, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
                groups.append(n)
    if which in ("dar_extractors", "all_except_flow"):
        for n in ("feat_extractor_ir", "feat_extractor_vi"):
            for p in getattr(model, n).parameters():
                p.requires_grad_(True)
        groups += ["feat_extractor_ir", "feat_extractor_vi"]
    if which == "all_except_flow":
        for p in model.fusion_net.parameters():
            p.requires_grad_(True)
        for n in ("aam_ir", "aam_vi"):
            for p in getattr(model, n).parameters():
                p.requires_grad_(True)
        groups += ["fusion_net", "aam_ir", "aam_vi"]
    return [p for p in model.parameters() if p.requires_grad], groups


def evaluate(model, loader, device, args):
    model.dar_force_oracle = False          # evaluation always uses the estimate
    out = {}
    for d in EVAL_DEGS:
        gen = torch.Generator(device=device).manual_seed(777)   # fixed realisation
        q = 0.0
        n = 0
        with torch.no_grad():
            for b in loader:
                ir0 = b["ir"].to(device)
                rgb0 = b["rgb"].to(device)
                ir = degrade(ir0, d, gen)
                rgb = degrade(rgb0, d, gen)
                if model.dar_mode == "oracle":
                    model.dar_cond_vec = cond_onehot(d, device)
                pred = model(ir, rgb).clamp(0, 1)
                q += float(metric.Metric_Qabf(torch.round(pred[:, 1] * 255),
                                              torch.round(ir0[:, 2] * 255),
                                              torch.round(rgb0[:, 2] * 255)).mean())
                n += 1
        out[d] = q / max(n, 1)
        print(f"    [{args.tag}] {d:8} Qabf={out[d]:.4f}", flush=True)
    return out


def main():
    args = parse_args()
    tag = args.tag
    out_path = REPO / (args.output or f"output/pr1/stage2_{tag}.json")
    ckpt_path = REPO / (args.ckpt or f"output/pr1/stage2_{tag}.pth")
    train_degs = [s for s in args.train_degradations.split(",") if s]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    man = json.loads((REPO / args.manifest).read_text())
    frozen = man["sequence_sets"]["frozen5_test"]["sequences"]
    train_seqs, split_path = build_split(args, frozen)

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    dar_cfg = {"mode": args.dar_mode, "input_restore": args.input_restore,
               "second": args.dar_second, "est_mode": args.est_mode,
               "gate": args.dar_gate, "vq": args.dar_vq}
    model = VideoFusion(model_config=cfg, dar=dar_cfg).to(device)
    model.load_state_dict(torch.load(REPO / "output" / EXP / "checkpoint/latest/model.pth",
                                     map_location=device), strict=False)
    model.eval()

    train_ds, train_loader = make_loader(split_path, DatasetMode.TRAIN)
    test_ds, test_loader = make_loader(split_path, DatasetMode.TEST, stride=args.eval_stride)

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.dar_mode = "off"
    teacher.dar_ir = teacher.dar_vi = teacher.dar_estimator = None
    teacher.input_restorer = None

    print(f"=== stage2 tag={tag} dar={args.dar_mode} restore={args.input_restore} "
          f"train_degs={train_degs} steps={args.steps} ===", flush=True)
    print("--- before ---", flush=True)
    history = {"before": evaluate(model, test_loader, device, args)}

    params, groups = set_trainable(model, args.unfreeze)
    n_train = sum(p.numel() for p in params)
    dar_params = sum(count_params(getattr(model, n)) for n in
                     ("dar_ir", "dar_vi", "dar_ir2", "dar_vi2", "dar_estimator",
                      "input_restorer")
                     if getattr(model, n, None) is not None)
    print(f"trainable: {n_train:,} ({groups}); DAR params {dar_params:,}; "
          f"blind-fine-tune reference {BLIND}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    step = 0
    t0 = time.time()
    model.train()
    while step < args.steps:
        for b in train_loader:
            if step >= args.steps:
                break
            ir0 = b["ir"].to(device)
            rgb0 = b["rgb"].to(device)
            if random.random() < args.clean_prob:
                d = "clean"
                ir, rgb = ir0, rgb0
            else:
                d = random.choice(train_degs)
                gen = torch.Generator(device=device).manual_seed(args.seed + step)
                ir = degrade(ir0, d, gen)
                rgb = degrade(rgb0, d, gen)

            if model.dar_mode == "oracle":
                model.dar_cond_vec = cond_onehot(d, device)
            elif args.oracle_mix > 0:
                use_oracle = random.random() < args.oracle_mix
                model.dar_force_oracle = use_oracle
                model.dar_cond_vec = cond_onehot(d, device) if use_oracle else None

            with torch.no_grad():
                teacher_out = teacher(ir0, rgb0).clamp(0, 1)

            student = model(ir, rgb).clamp(0, 1)
            loss = F.l1_loss(student, teacher_out)
            if args.dar_gate and model.dar_estimator is not None and args.gate_w > 0:
                # free supervision: we know whether this step was degraded
                g_logit = model.dar_estimator.gate_head(model.dar_estimator(ir, rgb))
                tgt_g = torch.zeros_like(g_logit) if d == "clean" else torch.ones_like(g_logit)
                loss = loss + args.gate_w * F.binary_cross_entropy_with_logits(g_logit, tgt_g)
            if args.aux_reg > 0 and model.dar_estimator is not None:
                desc = model.dar_estimator(ir, rgb)
                tgt = torch.tensor([DEG_PARAMS[d]], dtype=torch.float32, device=device)
                loss = loss + args.aux_reg * F.mse_loss(model.dar_estimator.reg(desc), tgt)
            if args.aux_ce > 0 and model.dar_estimator is not None:
                desc = model.dar_estimator(ir, rgb)
                target_cls = torch.tensor([cond_index(d)], device=device)
                loss = loss + args.aux_ce * F.cross_entropy(model.dar_estimator.cls(desc), target_cls)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"  [{tag}] step {step}/{args.steps} loss={loss.item():.5f} deg={d} "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
            if args.eval_every and step % args.eval_every == 0:
                model.eval()
                history[f"step{step}"] = evaluate(model, test_loader, device, args)
                model.train()

    model.eval()
    history["after"] = evaluate(model, test_loader, device, args)
    split_path.unlink(missing_ok=True)

    base, after = history["before"], history["after"]
    rec = {}
    for d in EVAL_DEGS:
        if d == "clean":
            # normalised recovery is meaningless for clean (its gap is ~0); report
            # the raw change instead so the printout cannot show a bogus number
            rec[d] = {"baseline": base[d], "after": after[d], "recovery_pct": None,
                      "change_pct": (after[d] - base[d]) / base[d] * 100}
            continue
        gap = REF["clean"] - base[d]
        rec[d] = {"baseline": base[d], "after": after[d],
                  "recovery_pct": (after[d] - base[d]) / gap * 100 if gap > 0 else None}
    seen = [d for d in ALL_DEGS if d in train_degs]
    unseen = ["noise15", "dark22"] + [d for d in ALL_DEGS if d not in train_degs]
    mean_seen = (sum(rec[d]["recovery_pct"] for d in seen) / len(seen)) if seen else None
    clean_change = (after["clean"] - base["clean"]) / base["clean"] * 100
    payload = {"tag": tag, "dar_mode": args.dar_mode, "input_restore": args.input_restore,
               "aux_ce": args.aux_ce, "aux_reg": args.aux_reg,
               "oracle_mix": args.oracle_mix, "dar_gate": args.dar_gate,
               "dar_vq": args.dar_vq,
               "unfreeze": args.unfreeze, "dar_second": args.dar_second,
               "est_mode": args.est_mode,
               "train_degradations": train_degs,
               "steps": args.steps, "lr": args.lr, "clean_prob": args.clean_prob,
               "trainable_params": n_train, "dar_params": dar_params, "groups": groups,
               "history": history, "recovery": rec,
               "mean_recovery_seen_degs_pct": mean_seen,
               "unseen_degs": unseen,
               "clean_change_pct": clean_change,
               "blind_finetune_reference": BLIND}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.save_full:
        torch.save(model.state_dict(), ckpt_path)
    else:
        torch.save({k: v for k, v in model.state_dict().items()
                    if k.startswith(("dar_", "input_restorer", "feat_extractor"))}, ckpt_path)

    print(f"\n=== stage2 {tag} ===")
    print(f"{'degradation':11} {'baseline':>9} {'after':>9} {'recovery%':>10}  seen?")
    for d in EVAL_DEGS:
        r = rec[d]
        rs = f"{r['recovery_pct']:+10.2f}" if r["recovery_pct"] is not None else "       n/a"
        print(f"{d:11} {r['baseline']:9.4f} {r['after']:9.4f} {rs}  {'yes' if d in train_degs else 'NO'}")
    print(f"clean change: {clean_change:+.2f}%   mean recovery (seen degs) = "
          f"{mean_seen:+.2f}%" if mean_seen is not None else "")
    print(f"BAR: blind fine-tune mean {BLIND['mean']}%, dark {BLIND['dark']}%, noise10 {BLIND['noise10']}%")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
