#!/usr/bin/env python3
"""Run the Python MS2R evaluation for every dataset/set in the manifest.

The MATLAB side (VIF等参数计算/run_all_matlab.m) produces the seven image
metrics; MS2R is produced here and the two are combined by
tools/merge_metrics.py.

Sequence sets and evaluation roots come from runs/00_contract/eval_manifest.json,
so the MATLAB and Python halves always score the same videos with the same
frame sampling (both use step 5).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# output/<exp_path>/checkpoint/latest/model.pth.  The three dataset checkpoints
# are byte-identical (sha256 7b3b5001f042bcff...), and output/MAVFusion/ has no
# checkpoint directory, so VTMOT reuses the M3SVD experiment path deliberately.
EXP_PATH = {
    "M3SVD": "MAVFusion-M3SVD",
    "HDO": "MAVFusion-HDO",
    "VTMOT": "MAVFusion-M3SVD",
}
BASE_DATA_DIR = {
    "M3SVD": "data2",
    "HDO": "data2",
    "VTMOT": "data2",
}
# HDO and VTMOT mix sequences of different spatial resolutions, so a batch
# larger than 1 cannot be collated (torch default_collate cannot stack them).
# REPRODUCTION_RESULTS.md notes the same constraint for VTMOT.
BATCH_SIZE = {
    "M3SVD": 4,
    "HDO": 1,
    "VTMOT": 1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
    p.add_argument("--set_name", default=None, help="override the manifest default_set")
    p.add_argument("--batch_size", type=int, default=0, help="0 = per-dataset default")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output_dir", default="output/pr1/python")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.loads((REPO / args.manifest).read_text())
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for ds in args.datasets.split(","):
        spec = manifest["datasets"][ds]
        set_name = args.set_name or spec["default_set"]
        sequences = manifest["sequence_sets"][set_name]["sequences"]
        out_csv = out_dir / f"metrics_ms2r_{ds}_{set_name}.csv"
        cmd = [
            args.python, "run_ms2r_only.py",
            "--dataset_name", ds,
            "--exp_path", EXP_PATH[ds],
            "--base_data_dir", BASE_DATA_DIR[ds],
            "--sequences", ",".join(sequences),
            "--batch_size", str(args.batch_size or BATCH_SIZE[ds]),
            "--num_workers", str(args.num_workers),
            "--device", args.device,
            "--output_csv", str(out_csv),
        ]
        print(f"\n=== {ds} / {set_name} ===\n$ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=REPO)
        entry = {"dataset": ds, "set": set_name, "returncode": result.returncode,
                 "output_csv": str(out_csv if not out_csv.is_absolute()
                                  else out_csv.relative_to(out_csv.anchor)),
                 "sequences": sequences, "exp_path": EXP_PATH[ds],
                 "checkpoint_note": "all dataset checkpoints are byte-identical; "
                                    "VTMOT reuses the M3SVD experiment path"}
        if out_csv.exists():
            import csv
            with out_csv.open() as fh:
                entry["values"] = next(csv.DictReader(fh))
        summary.append(entry)

    (out_dir / "ms2r_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== MS2R summary ===")
    for e in summary:
        print(f"  {e['dataset']:6} rc={e['returncode']} {e.get('values', {})}")
    print(f"written {out_dir / 'ms2r_summary.json'}")
    if any(e["returncode"] != 0 for e in summary):
        sys.exit("at least one MS2R run failed")


if __name__ == "__main__":
    main()
