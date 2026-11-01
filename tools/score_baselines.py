#!/usr/bin/env python3
"""Score every fused result on disk with the sanctioned metric contract.

For each (method, dataset) it writes a temporary manifest whose eval_root points at
the method's own directory, calls the same MATLAB contract used for our own
numbers, and collects the eight metrics.  The point is that every method is scored
by exactly the same code, on exactly the same frames, with the same input scale --
which is what makes a comparison table defensible.

Outputs
  output/pr1/baseline_metrics/<method>_<dataset>.json   per-metric means
  output/pr1/baseline_metrics/all.json                  everything, merged
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "output/pr1/fusion_results"
OUT = REPO / "output/pr1/baseline_metrics"
SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}
METRICS = ["QG", "QABF", "QM", "QP", "QS", "QCB", "VIF"]


def _expected_frames(dataset: str, step: int = 5) -> int:
    """Number of holdout frames the contract samples for this dataset."""
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    seqs = man["sequence_sets"][SETS[dataset]]["sequences"]
    root = REPO / "eval_data" / dataset
    n = 0
    for seq in seqs:
        irs = [p for p in (root / "ir").iterdir()
               if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp") and p.name.startswith(seq + "_")]
        vis = {p.name for p in (root / "vis").iterdir()}
        n += len([p for p in sorted(irs)[::step] if p.name in vis])
    return n


def score(method: str, dataset: str, sample_step: int = 5) -> dict | None:
    root = RES / method / dataset
    if not (root / "fused").is_dir() or not any((root / "fused").iterdir()):
        print(f"  skip {method}/{dataset}: no fused frames")
        return None
    man = json.loads((REPO / "runs/00_contract/eval_manifest.json").read_text())
    man["eval_root"] = str(root)
    tmp = REPO / "runs/00_contract" / f"_tmp_manifest_{method}_{dataset}.json"
    tmp.write_text(json.dumps(man, indent=2))
    dst_dir = OUT / f"{method}_{dataset}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"contract_metrics_{SETS[dataset]}.json"
    # Two kinds of output directory exist:
    #   * harness runs (this repo's bench_baselines.py) store exactly the frames
    #     the contract scores, so the directory must be read at step 1;
    #   * a method's own test.py stores *every* frame, so it must be read at
    #     step 5 to land on the same pre-registered frames.
    n_fused = len(list((root / "fused").iterdir()))
    n_expect = _expected_frames(dataset, sample_step)
    step = 1 if n_fused <= 1.1 * n_expect else sample_step
    cmd = ["matlab", "-batch",
           f"addpath('tools'); contract_metrics('{tmp}', '{dst_dir}', "
           f"'{SETS[dataset]}', {step}, inf)"]
    print(f"  scoring {method}/{dataset} ...", flush=True)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=7200)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0 or not dst.exists():
        print(f"  FAILED {method}/{dataset}\n{(r.stdout or '')[-600:]}\n{(r.stderr or '')[-400:]}")
        return None
    if not dst.exists():
        cands = list(dst_dir.glob("contract_metrics_*.json"))
        if not cands:
            print(f"  no metrics json in {dst_dir}")
            return None
        dst = cands[0]
    d = json.loads(dst.read_text())
    mean = d.get("video_equal_mean", d)
    return {k: mean.get(k) for k in METRICS} | {
        "MS2R": None, "frames": d.get("frames"), "set": SETS[dataset]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    argv = sys.argv[1:]
    only = None
    if "--datasets" in argv:
        i = argv.index("--datasets")
        only = [d for d in argv[i + 1].split(",") if d]
        del argv[i:i + 2]
    methods = argv or sorted([p.name for p in RES.iterdir() if p.is_dir()])
    datasets = only or list(SETS)
    allm = {}
    for m in methods:
        for ds in datasets:
            r = score(m, ds)
            if r:
                allm.setdefault(m, {})[ds] = r
    merged = OUT / "all.json"
    prev = json.loads(merged.read_text()) if merged.exists() else {}
    for m, v in allm.items():
        prev.setdefault(m, {}).update(v)
    merged.write_text(json.dumps(prev, indent=2, ensure_ascii=False))
    print(f"written {merged}")
    for m, v in prev.items():
        row = " ".join(f"{ds}:{(v[ds].get('QABF') or float('nan')):.4f}" if ds in v else ""
                       for ds in SETS)
        print(f"  {m:16} {row}")


if __name__ == "__main__":
    main()
