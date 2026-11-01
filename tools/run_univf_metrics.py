#!/usr/bin/env python3
"""Compute VIF and MS2R with the UniVF reference implementations.

Why these two come from Python and not MATLAB
  The UniVF reference (VF-Bench, Zhaoxiang Zhao et al., NeurIPS 2025 Spotlight,
  https://github.com/Zhaozixiang1228/VF-Bench) implements both metrics in
  ``src/util/metric.py``:

    * ``Metric_VIF``   -- pixel-based VIF (vifp), 4 scales, float64,
                          inputs are the middle frames rounded to [0, 255],
                          averaged over the two source references.
    * ``MS2R_Evaluator.compute_ms2r_metric``
                      -- 0.5 * (mean|D1| + mean|D2|) with
                         D = (flow(G1,G2) - flow(G0,G1)) - (flow(R1,R2) - flow(R0,R1)).

  MAVFusion's own ``src/util/metric.py`` is byte-identical to the UniVF version
  for ``Metric_VIF`` and computes the same MS2R formula (rewritten to batch the
  six flow forwards into one).  The MATLAB ``vifvec`` in the vendored metric
  repository is a *different* implementation and deviates by +0.10 to +0.20
  against the paper, so it is not used.

  ``compute_sample_smoothness`` (a histogram-entropy quantity) exists in the
  UniVF reference too and is unused there as well -- it is upstream dead code,
  not a MAVFusion defect.

This driver runs ``evaluate_metrics.py`` (which already wires both reference
implementations) per dataset/set taken from runs/00_contract/eval_manifest.json.

HDO and VTMOT mix sequence resolutions, so batch_size=1 is required there.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EXP_PATH = {
    "M3SVD": "MAVFusion-M3SVD",
    "HDO": "MAVFusion-HDO",
    "VTMOT": "MAVFusion-M3SVD",   # all dataset checkpoints are byte-identical
}
BATCH_SIZE = {"M3SVD": 4, "HDO": 1, "VTMOT": 1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
    p.add_argument("--set_name", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output_dir", default="output/pr1/python")
    p.add_argument("--num_workers", type=int, default=4)
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
        out_csv = out_dir / f"univf_{ds}_{set_name}.csv"
        cmd = [
            args.python, "evaluate_metrics.py",
            "--dataset_name", ds,
            "--exp_path", EXP_PATH[ds],
            "--base_data_dir", "data2",
            "--sequences", ",".join(sequences),
            "--batch_size", str(BATCH_SIZE[ds]),
            "--num_workers", str(args.num_workers),
            "--device", args.device,
            "--include_ms2r",
            "--output_csv", str(out_csv),
        ]
        print(f"\n=== {ds} / {set_name} (UniVF Metric_VIF + MS2R) ===\n$ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        entry = {"dataset": ds, "set": set_name, "returncode": rc,
                 "output_csv": out_csv.name, "sequences": sequences,
                 "implementation": "UniVF VF-Bench src/util/metric.py",
                 "source_repo": "https://github.com/Zhaozixiang1228/VF-Bench"}
        if out_csv.exists():
            with out_csv.open() as fh:
                entry["values"] = next(csv.DictReader(fh))
        summary.append(entry)

    (out_dir / "univf_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== UniVF VIF / MS2R summary ===")
    for e in summary:
        v = e.get("values", {})
        print(f"  {e['dataset']:6} rc={e['returncode']} "
              f"VIF={v.get('Metric_VIF')} MS2R={v.get('Metric_MS2R')}")
    print(f"written {out_dir / 'univf_summary.json'}")
    if any(e["returncode"] != 0 for e in summary):
        sys.exit("at least one run failed")


if __name__ == "__main__":
    main()
