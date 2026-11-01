#!/usr/bin/env python3
"""Stage 1 gate: is >=15% of the degradation collapse recoverable by fine-tuning
only the shallow feature extractors?

Context (measured, see day4_decomposition_report.md): the whole degradation
collapse (-21% to -71% Qabf) comes from the FEATURE path; replaying a perfect
clean optical flow recovers nothing.  So the only justified module is a
feature-side one, and the first question is whether that path is even trainable
towards robustness with a small, bounded intervention.

Protocol
  train  : the 25 M3SVD sequences that are not in the frozen test holdout
  test   : the frozen 5 sequences, evaluated with the SAME protocol as Day 3/4
           (degraded input, Qabf against the CLEAN sources, identical windows
           and identical noise realisations)
  student: the frozen model with only feat_extractor_ir/vi unfrozen
  teacher: the original frozen model run on CLEAN input (free paired supervision;
           this is exactly the signal a degradation-aware rectifier would train on)
  loss   : L1(student(deg), teacher(clean))  [+ optional official fusion loss]

Pre-registered gate
  mean recovery over the four degradations >= 15% of the clean-vs-degraded gap,
  and clean Qabf no worse than -1%.
    recovery(d) = (Q_ft(d) - Q_base(d)) / (Q_clean - Q_base(d))
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
from src.util import metric  # noqa: E402

# src/util/loss.py imports kornia, which is not installed in this environment.
# The distillation objective below needs no external loss, so the import is lazy
# and only happens when --objective distill+fusion is requested.
IVF_MVF_Loss = None

YAML = "config/dataset/IVF/M3SVD/M3SVD_5-frame.yaml"
EXP = "MAVFusion-M3SVD"
DEGRADATIONS = ("noise10", "noise20", "dark", "blur")
# Day 3/4 reference numbers on the frozen 5 at stride 25 (158 windows)
REF = {"clean": 0.7364, "noise10": 0.5823, "noise20": 0.4777, "dark": 0.5638, "blur": 0.2162}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--train_split", default="data_split/IVF/M3SVD/split_train25.json")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--clean_prob", type=float, default=0.5,
                   help="fraction of steps fed CLEAN input, which acts as a regulariser "
                        "against forgetting the clean-input behaviour")
    p.add_argument("--unfreeze", default="extractors",
                   choices=["extractors", "extractors_aam", "all_except_flow"])
    p.add_argument("--objective", default="distill", choices=["distill", "distill+fusion"])
    p.add_argument("--eval_every", type=int, default=300)
    p.add_argument("--eval_stride", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/pr1/stage1_finetune.json")
    p.add_argument("--ckpt", default="output/pr1/stage1_extractors.pth")
    return p.parse_args()


def degrade(x, kind, gen):
    if kind == "clean":
        return x
    if kind == "noise10":
        return (x + torch.randn(x.shape, generator=gen, device=x.device) * (10.0 / 255.0)).clamp(0, 1)
    if kind == "noise20":
        return (x + torch.randn(x.shape, generator=gen, device=x.device) * (20.0 / 255.0)).clamp(0, 1)
    if kind == "dark":
        return x.clamp(0, 1) ** 1.8
    if kind == "blur":
        k = torch.ones(1, 1, 5, 5, device=x.device) / 25.0
        b, t, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * t, c, h, w), k.repeat(c, 1, 1, 1), padding=2, groups=c)
        return y.reshape(b, t, c, h, w)
    raise ValueError(kind)


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
    """Given the parameter census, feat_extractor_* alone is only 10,176 params
    (0.11% of the model: flow_net holds 97.1%), so a negative result there is
    confounded by capacity.  'all_except_flow' is the honest maximum-recovery
    gate: 265,573 params (2.9%)."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = ["feat_extractor_ir", "feat_extractor_vi"]
    if which in ("extractors_aam", "all_except_flow"):
        names += ["aam_ir", "aam_vi"]
    if which == "all_except_flow":
        names += ["fusion_net"]
    for n in names:
        for p in getattr(model, n).parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def evaluate(model, loader, device, stride, tag=""):
    out = {}
    for d in ("clean",) + DEGRADATIONS:
        gen = torch.Generator(device=device).manual_seed(1234 + hash(d) % 1000)
        q = 0.0
        n = 0
        with torch.no_grad():
            for b in loader:
                ir0 = b["ir"].to(device)
                rgb0 = b["rgb"].to(device)
                ir = degrade(ir0, d, gen)
                rgb = degrade(rgb0, d, gen)
                pred = model(ir, rgb).clamp(0, 1)
                q += float(metric.Metric_Qabf(torch.round(pred[:, 1] * 255),
                                              torch.round(ir0[:, 2] * 255),
                                              torch.round(rgb0[:, 2] * 255)).mean())
                n += 1
        out[d] = q / max(n, 1)
        print(f"    [{tag}] {d:8} Qabf={out[d]:.4f} (ref {REF[d]:.4f})", flush=True)
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    man = json.loads((REPO / args.manifest).read_text())
    frozen = man["sequence_sets"]["frozen5_test"]["sequences"]
    train_seqs, split_path = build_split(args, frozen)
    print(f"train sequences: {len(train_seqs)}  frozen test: {len(frozen)}")

    cfg = OmegaConf.load("config/train/ivf-train.yaml")
    model = VideoFusion(model_config=cfg).to(device)
    model.load_state_dict(torch.load(REPO / "output" / EXP / "checkpoint/latest/model.pth",
                                     map_location=device))
    model.eval()

    train_ds, train_loader = make_loader(split_path, DatasetMode.TRAIN)
    test_ds, test_loader = make_loader(split_path, DatasetMode.TEST, stride=args.eval_stride)
    print(f"train windows/epoch: {len(train_ds)}   eval windows: {len(test_ds)}")

    crit = None
    if args.objective == "distill+fusion":
        from src.util.loss import IVF_MVF_Loss as _Loss  # noqa: WPS433
        crit = _Loss(coef=(1, 5, 0, 2), use_occlusion=False, occ_threshold=1.0,
                     max_offset_mask=False).to(device)

    print("\n--- before fine-tuning ---")
    history = {"before": evaluate(model, test_loader, device, args.eval_stride, "before")}

    # A FROZEN copy is the teacher.  Using the model itself would make the target
    # move with the parameters, and the objective L1(f(deg), f(clean)) then has the
    # degenerate solution "make f constant" -- which is exactly what collapsed the
    # first attempt (clean Qabf 0.736 -> 0.241).
    teacher_model = copy.deepcopy(model).eval()
    for p_ in teacher_model.parameters():
        p_.requires_grad_(False)

    params = set_trainable(model, args.unfreeze)
    n_train = sum(p.numel() for p in params)
    print(f"\ntrainable ({args.unfreeze}): {n_train:,} params")
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
                d = random.choice(DEGRADATIONS)
                gen = torch.Generator(device=device).manual_seed(args.seed + step)
                ir = degrade(ir0, d, gen)
                rgb = degrade(rgb0, d, gen)

            with torch.no_grad():
                teacher = teacher_model(ir0, rgb0).clamp(0, 1)   # FROZEN model, CLEAN input

            student = model(ir, rgb).clamp(0, 1)
            loss = F.l1_loss(student, teacher)
            if args.objective == "distill+fusion":
                lo = crit(student, ir, rgb, model.flow_net)
                loss = loss + 0.1 * lo["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1

            if step % 50 == 0:
                print(f"  step {step}/{args.steps}  loss={loss.item():.5f}  deg={d}  "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
            if step % args.eval_every == 0:
                model.eval()
                key = f"step{step}"
                history[key] = evaluate(model, test_loader, device, args.eval_stride, key)
                model.train()

    model.eval()
    history["after"] = evaluate(model, test_loader, device, args.eval_stride, "after")
    torch.save({k: v for k, v in model.state_dict().items()
                if k.startswith(("feat_extractor_ir", "feat_extractor_vi"))},
               REPO / args.ckpt)
    split_path.unlink(missing_ok=True)

    base, after = history["before"], history["after"]
    rec = {}
    for d in DEGRADATIONS:
        gap = REF["clean"] - base[d]
        rec[d] = {"baseline": base[d], "after": after[d],
                  "recovery_pct_of_gap": (after[d] - base[d]) / gap * 100 if gap > 0 else None}
    mean_rec = sum(v["recovery_pct_of_gap"] for v in rec.values()) / len(DEGRADATIONS)
    clean_change = (after["clean"] - base["clean"]) / base["clean"] * 100
    payload = {"train_sequences": len(train_seqs), "steps": args.steps, "lr": args.lr,
               "unfreeze": args.unfreeze, "trainable_params": n_train,
               "objective": args.objective, "history": history, "recovery": rec,
               "mean_recovery_pct_of_gap": mean_rec, "clean_change_pct": clean_change,
               "gate_threshold_pct": 15.0, "gate_passes": bool(mean_rec >= 15.0 and clean_change >= -1.0)}
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== Stage 1 gate · {args.unfreeze} · {args.steps} steps ===")
    print(f"{'degradation':11} {'baseline':>9} {'after':>9} {'recovery%':>10}")
    for d in DEGRADATIONS:
        print(f"{d:11} {rec[d]['baseline']:9.4f} {rec[d]['after']:9.4f} {rec[d]['recovery_pct_of_gap']:+10.2f}")
    print(f"clean: {base['clean']:.4f} -> {after['clean']:.4f} ({clean_change:+.2f}%)")
    print(f"mean recovery = {mean_rec:+.2f}%   gate (>=15%): {payload['gate_passes']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
