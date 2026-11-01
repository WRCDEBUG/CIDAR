#!/usr/bin/env python3
"""Assemble the quantitative comparison table from the recorded scores.

Inputs
  output/pr1/baseline_metrics/all.json          every baseline, every dataset
  output/pr1/merged/table1_local_vs_paper.csv   our own video-equal numbers
Outputs
  output/pr1/baseline_metrics/comparison.csv    long form, one row per cell
  output/pr1/baseline_metrics/comparison.md     readable, one panel per dataset
  runs/00_contract/comparison_table.tex         paste-ready LaTeX floats

Only numbers produced by tools/score_baselines.py are used; this script never
runs a network and never edits a score, so a table cannot drift from the
artifacts on disk.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = ("QG", "QABF", "QM", "QP", "QS", "QCB", "VIF", "MS2R")
DATASETS = ("M3SVD", "HDO", "VTMOT")
# presentation order: ours first, then the published baselines
METHOD_ORDER = ("univf", "cddfuse", "seafusion", "tardal", "ignet", "tdfusion",
                "fusionmamba", "s4fusion", "coconet")
LABEL = {
    "univf": "UniVF (ours)", "cddfuse": "CDDFuse", "seafusion": "SeAFusion",
    "tardal": "TarDAL", "ignet": "IGNet", "tdfusion": "TDFusion",
    "fusionmamba": "FusionMamba", "s4fusion": "S4Fusion", "coconet": "CoCoNet",
}
TEX_LABEL = {
    "QG": r"$Q^{G}$", "QABF": r"$Q^{AB/F}$", "QM": r"$Q^{M}$", "QP": r"$Q^{P}$",
    "QS": r"$Q^{S}$", "QCB": r"$Q^{CB}$", "VIF": r"VIF", "MS2R": r"MS2R",
}
HIGHER_IS_BETTER = {m: True for m in METRICS}


def load_scores() -> dict[str, dict[str, dict[str, float | None]]]:
    f = REPO / "output/pr1/baseline_metrics/all.json"
    if not f.exists():
        return {}
    raw = json.loads(f.read_text())
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for method, ds_map in raw.items():
        for ds, vals in ds_map.items():
            out.setdefault(method, {})[ds] = {m: vals.get(m) for m in METRICS}
    # VIF is pinned to the UniVF reference implementation (manifest), so the
    # MATLAB vifvec column in all.json must not be used.
    vif_f = REPO / "output/pr1/baseline_metrics/vif_all.json"
    if vif_f.exists():
        for method, ds_map in json.loads(vif_f.read_text()).items():
            for ds, vals in ds_map.items():
                if method in out and ds in out[method]:
                    out[method][ds]["VIF"] = vals.get("VIF")
    return out


def add_ours(scores):
    """Pull our video-equal row out of the merged reproduction table."""
    f = REPO / "output/pr1/merged/table1_local_vs_paper.csv"
    if not f.exists():
        return
    with f.open() as fh:
        for row in csv.DictReader(fh):
            ds = row["dataset"]
            scores.setdefault("univf", {})[ds] = {
                m: (float(row[f"local_{m}"]) if row.get(f"local_{m}") not in (None, "") else None)
                for m in METRICS
            }


def fmt(v) -> str:
    return "---" if v is None else f"{v:.4f}"


def main():
    scores = load_scores()
    add_ours(scores)
    if not scores:
        raise SystemExit("no scores on disk yet (run tools/score_baselines.py first)")
    methods = [m for m in METHOD_ORDER if m in scores] + \
              [m for m in sorted(scores) if m not in METHOD_ORDER]
    methods = [m for m in methods if any(scores[m].get(ds) for ds in DATASETS)]

    out_dir = REPO / "output/pr1/baseline_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # long form
    with (out_dir / "comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset"] + list(METRICS))
        for m in methods:
            for ds in DATASETS:
                v = scores.get(m, {}).get(ds)
                if v:
                    w.writerow([m, ds] + [("" if v.get(k) is None else f"{v[k]:.6f}") for k in METRICS])

    # best per (dataset, metric)
    best = {}
    for ds in DATASETS:
        for k in METRICS:
            vals = [scores[m][ds][k] for m in methods
                    if scores.get(m, {}).get(ds, {}).get(k) is not None]
            best[(ds, k)] = max(vals) if (vals and HIGHER_IS_BETTER[k]) else (min(vals) if vals else None)

    # markdown
    md = ["# Baseline comparison (video-equal means on the frozen five-sequence holdout)", ""]
    md.append("Every row is scored on the same frozen frames: the six image metrics by "
              "`tools/contract_metrics.m` and VIF by the pinned UniVF reference "
              "(`tools/score_vif.py`). **Bold** marks the best value in a column.")
    for ds in DATASETS:
        md += ["", f"## {ds}", "",
               "| Method | " + " | ".join(METRICS) + " |",
               "|---|" + "---|" * len(METRICS)]
        for m in methods:
            v = scores.get(m, {}).get(ds)
            if not v:
                continue
            cells = []
            for k in METRICS:
                x = v.get(k)
                if x is None:
                    cells.append("---")
                elif best[(ds, k)] is not None and abs(x - best[(ds, k)]) < 1e-12:
                    cells.append(f"**{x:.4f}**")
                else:
                    cells.append(f"{x:.4f}")
            md.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    md += ["", "Notes: all rows are scored on the same frozen frames with the same implementations: "
               "QG/QABF/QM/QP/QS/QCB by `tools/contract_metrics.m` at the scales declared in "
               "`runs/00_contract/eval_manifest.json` (QS on [0,255], where [0,1] saturates near 0.995), "
               "and VIF by the pinned UniVF reference (`Metric_VIF`, `tools/score_vif.py`). "
               "MS2R is a temporal metric that per-frame baselines cannot produce, so it is blank for "
               "them. Video-equal = per-sequence means averaged over the five holdout sequences."]
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n")

    # LaTeX: one float per dataset, 8 metric columns
    tex = ["% generated by tools/make_baseline_table.py -- do not edit by hand"]
    for ds in DATASETS:
        body = []
        for m in methods:
            v = scores.get(m, {}).get(ds)
            if not v:
                continue
            cells = []
            for k in METRICS:
                x = v.get(k)
                if x is None:
                    cells.append("---")
                elif best[(ds, k)] is not None and abs(x - best[(ds, k)]) < 1e-12:
                    cells.append(r"\textbf{" + f"{x:.4f}" + "}")
                else:
                    cells.append(f"{x:.4f}")
            body.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
        tex += [
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\caption{Quantitative comparison on " + ds + r" (video-equal means, five frozen sequences).}",
            r"\label{tab:comparison-" + ds.lower() + "}",
            r"\begin{tabular}{l" + "c" * len(METRICS) + "}",
            r"\toprule",
            "Method & " + " & ".join(TEX_LABEL[k] for k in METRICS) + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule", r"\end{tabular}", r"\end{table*}", "",
        ]
    (REPO / "runs/00_contract/comparison_table.tex").write_text("\n".join(tex) + "\n")

    print(f"methods: {methods}")
    for ds in DATASETS:
        print(f"\n== {ds}")
        print("  " + "met    " + " ".join(f"{k:>7}" for k in METRICS))
        for m in methods:
            v = scores.get(m, {}).get(ds)
            if v:
                print(f"  {m:15}" + " ".join(f"{fmt(v.get(k)):>7}" for k in METRICS))
    print(f"\nwritten {out_dir/'comparison.csv'}, {out_dir/'comparison.md'}, "
          f"runs/00_contract/comparison_table.tex")


if __name__ == "__main__":
    main()
