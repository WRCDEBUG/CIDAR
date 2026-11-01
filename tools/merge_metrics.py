#!/usr/bin/env python3
"""Merge the MATLAB image metrics with the Python MS2R into one table.

Inputs
  output/pr1/matlab/metrics_<ds>_<set>.csv        (7 image metrics, MATLAB)
  output/pr1/python/metrics_ms2r_<ds>_<set>.csv   (MS2R, Python)
Outputs
  output/pr1/merged/metrics_<ds>_<set>.csv        per-dataset table vs the paper
  output/pr1/merged/table1_local_vs_paper.csv     all datasets side by side

Weighting note: the MATLAB script reports its six image metrics video-equal
(per-sequence mean, then mean over sequences).  VIF is likewise video-equal
(evaluate_metrics.py averages per frame; with a fixed 5-sequence set the
difference is small but is noted here).  MS2R is sample-equal because it is a
temporal metric over sliding windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# MATLAB (VIF等参数计算/run_all_matlab.m) supplies these six.
MATLAB_METRICS = ("QG", "QM", "QP", "QS", "QCB", "QABF")
# Python supplies VIF and MS2R from the UniVF reference implementation
# (VF-Bench src/util/metric.py); see tools/run_univf_metrics.py.
PYTHON_METRICS = ("VIF", "MS2R")
COLUMNS = MATLAB_METRICS + PYTHON_METRICS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/00_contract/eval_manifest.json")
    p.add_argument("--matlab_dir", default="output/pr1/matlab")
    p.add_argument("--python_dir", default="output/pr1/python")
    p.add_argument("--univf_dir", default="output/pr1/python")
    p.add_argument("--output_dir", default="output/pr1/merged")
    p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
    return p.parse_args()


def read_single_row(path: Path) -> dict:
    with path.open() as fh:
        return next(csv.DictReader(fh))


def main():
    args = parse_args()
    manifest = json.loads((REPO / args.manifest).read_text())
    matlab_dir = REPO / args.matlab_dir
    python_dir = REPO / args.python_dir
    univf_dir = REPO / args.univf_dir
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for ds in args.datasets.split(","):
        spec = manifest["datasets"][ds]
        set_name = spec["default_set"]
        summary_path = matlab_dir / f"metrics_{ds}_{set_name}.csv"
        ms2r_path = python_dir / f"metrics_ms2r_{ds}_{set_name}.csv"   # fallback path
        if not summary_path.exists():
            print(f"[skip] {ds}: missing {summary_path}")
            continue

        with summary_path.open() as fh:
            rows = {r["Row"]: r for r in csv.DictReader(fh)}
        video_equal = rows["video_equal"]
        paper = rows["paper"]

        local = {m: float(video_equal[m]) for m in MATLAB_METRICS}
        # VIF and MS2R come from the UniVF reference implementation
        univf = univf_dir / f"univf_{ds}_{set_name}.csv"
        if univf.exists():
            row = read_single_row(univf)
            local["VIF"] = float(row["Metric_VIF"])
            local["MS2R"] = float(row["Metric_MS2R"])
        elif ms2r_path.exists():
            local["MS2R"] = float(read_single_row(ms2r_path)["Metric_MS2R"])
            print(f"[warn] {ds}: no UniVF csv, VIF missing and MS2R taken from run_ms2r_only.py")

        paper_row = {m: float(paper[m]) for m in MATLAB_METRICS}
        paper_row["VIF"] = float(spec["paper_table1"]["VIF"])
        paper_row["MS2R"] = float(spec["paper_table1"]["MS2R"])

        out_csv = out_dir / f"metrics_{ds}_{set_name}.csv"
        with out_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Row", "dataset", "set"] + list(COLUMNS))
            w.writerow(["video_equal"] + [ds, set_name] + [f"{local.get(m, float('nan')):.6f}" for m in COLUMNS])
            w.writerow(["paper" ] + [ds, set_name] + [f"{paper_row.get(m, float('nan')):.6f}" for m in COLUMNS])
            w.writerow(["diff"] + [ds, set_name] +
                       [f"{local.get(m, float('nan')) - paper_row.get(m, float('nan')):+.6f}" for m in COLUMNS])

        combined_rows.append({
            "dataset": ds, "set": set_name,
            **{f"local_{m}": local.get(m) for m in COLUMNS},
            **{f"paper_{m}": paper_row.get(m) for m in COLUMNS},
            "vif_ms2r_source": "UniVF" if univf.exists() else "partial",
            "sequences": ",".join(manifest["sequence_sets"][set_name]["sequences"]),
        })
        nan = float("nan")
        print(f"[{ds}] " + " ".join(f"{m}={local.get(m, nan):.4f}" for m in COLUMNS))
        print(f"       vs paper: " + " ".join(
            f"{m}{local.get(m, 0) - paper_row.get(m, 0):+.4f}" for m in COLUMNS))

    combined = out_dir / "table1_local_vs_paper.csv"
    if combined_rows:
        fields = list(combined_rows[0].keys())
        with combined.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(combined_rows)
    print(f"\nwritten {combined}")
    print(f"written per-dataset tables under {out_dir}")


if __name__ == "__main__":
    main()
