"""
从融合图 + 原图构建评估用扁平目录

用法:
    python build_eval_data.py M3SVD
    python build_eval_data.py M3SVD --missing-only
    python build_eval_data.py M3SVD --sequences 1208_1711,1204_1139

修复记录 (Step 0, 2026-10-15):
    旧版本用 `fused_stem.rsplit("_", 1)` 推断帧号，对帧名本身含下划线的序列
    （例如 1208_1711 的帧名是 `12_251.png`，融合图名 `1208_1711_12_251.jpg`）
    会把序列名切成 `1208_1711_12`，随后原图查找失败并被静默跳过。
    M3SVD 因此有 8 条序列从未进入 eval_data（含冻结测试集里的 1208_1711）。

    新版本以 data2 的原图为基准枚举帧：序列名已知，帧号取原图 stem，
    融合图固定为 f"{seq}_{stem}{ext}"。这样任意帧名规则都能正确配对，
    并把缺失项显式报告出来，而不是静默丢弃。
"""
import argparse
import shutil
from pathlib import Path

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build(dataset, missing_only=False, sequences=None, verbose=True):
    fused_src = Path(f"output/MAVFusion-{dataset}/test_results/latest/eval_visual/IVF-{dataset}/fused_result")
    if not fused_src.is_dir():
        print(f"❌ 融合图目录不存在: {fused_src}")
        return 1

    data_root = Path("data2") / dataset
    if not data_root.is_dir():
        print(f"❌ 数据目录不存在: {data_root}")
        return 1

    out_root = Path("eval_data") / dataset
    ir_out, vis_out, fused_out = (out_root / k for k in ("ir", "vis", "fused"))
    for d in (ir_out, vis_out, fused_out):
        d.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if sequences:
        wanted = set(sequences)
        seq_dirs = [d for d in seq_dirs if d.name in wanted]
        missing_seqs = wanted - {d.name for d in seq_dirs}
        if missing_seqs:
            print(f"⚠️  数据目录中不存在这些序列: {sorted(missing_seqs)}")

    copied = skipped = absent = 0
    per_seq = {}
    for seq_dir in seq_dirs:
        seq = seq_dir.name
        n_copied = n_absent = 0
        ir_src_dir = seq_dir / "infrared"
        vis_src_dir = seq_dir / "visible"
        if not ir_src_dir.is_dir() or not vis_src_dir.is_dir():
            print(f"❌ 缺少模态目录: {seq_dir}")
            continue

        for f in sorted(ir_src_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXT:
                continue
            vis_src = next(
                (vis_src_dir / f"{f.stem}{e}" for e in IMAGE_EXT
                 if (vis_src_dir / f"{f.stem}{e}").exists()), None
            )
            fused_src_file = next(
                (fused_src / f"{seq}_{f.stem}{e}" for e in IMAGE_EXT
                 if (fused_src / f"{seq}_{f.stem}{e}").exists()), None
            )
            if vis_src is None or fused_src_file is None:
                absent += 1
                n_absent += 1
                continue

            # 三个目录统一使用融合图名，保持 ir/vis/fused 一一对应
            name = fused_src_file.name
            targets = [(f, ir_out / name), (vis_src, vis_out / name), (fused_src_file, fused_out / name)]
            if missing_only and all(t.exists() for _, t in targets):
                skipped += 1
                continue
            for src, dst in targets:
                if missing_only and dst.exists():
                    continue
                shutil.copy(src, dst)
            copied += 1
            n_copied += 1

        per_seq[seq] = n_copied
        if verbose:
            flag = "⚠️" if n_absent else "  "
            print(f"{flag} {seq}: copied={n_copied} absent={n_absent}")

    total_fused = len(list(fused_out.glob("*")))
    print(f"\n✅ {dataset}: 本次复制融合图 {copied} 张，跳过已存在 {skipped}，原图无对应融合图 {absent}")
    print(f"   eval_data 现有融合图 {total_fused} 张，覆盖 {len(per_seq)} 条序列")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="M3SVD | VTMOT | HDO")
    p.add_argument("--missing-only", action="store_true", help="只补 eval_data 中尚不存在的文件")
    p.add_argument("--sequences", default=None, help="逗号分隔的序列白名单")
    args = p.parse_args()
    seqs = [s for s in (args.sequences or "").split(",") if s] or None
    raise SystemExit(build(args.dataset, missing_only=args.missing_only, sequences=seqs))


if __name__ == "__main__":
    main()
