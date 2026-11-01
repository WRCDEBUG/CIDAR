#!/usr/bin/env python3
"""Build the paper's result tables straight from the recorded run artifacts.

Every number in the paper should be regenerable by running this file, so that a
reviewer-facing claim cannot silently drift from the experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEGS = ("clean", "noise10", "noise20", "dark", "blur", "noise15", "dark22")
CORE = ("noise10", "noise20", "dark")          # the recoverable, same-step subset
LABEL = {
    "oracle": "DAR, true descriptor (oracle), 3k",
    "oracle_long": "DAR, true descriptor (oracle), 8k",
    "predicted": "DAR, estimated descriptor",
    "predicted_loo": "DAR, estimated, trained without blur",
    "noblur_long": "DAR, estimated, trained without blur, 8k",
    "withblur_long": "DAR, estimated, trained with blur, 8k",
    "noblur_2stage": "DAR + second injection point",
    "noblur_2stage_long": "DAR + second injection point, 8k",
    "hf_3k": "DAR + high-frequency estimator",
    "cross_3k": "DAR + cross-modal estimator",
    "cross_2stage_3k": "DAR + cross-modal estimator + second injection",
    "noconv": "rectifier, no conditioning (capacity control)",
    "input_restore": "input-side restorer (placement control)",
    "auxce_early": "DAR + auxiliary classification",
    "noblur_auxreg": "DAR + auxiliary regression",
    "blind_matched": "blind fine-tune, 3k (matched)",
    "blind_matched_long": "blind fine-tune, 8k (matched)",
    "predicted_noblur_rand": "DAR, domain-randomised low-light",
}


def load_runs():
    runs = {}
    for f in sorted((REPO / "output/pr1").glob("stage2_*.json")):
        if f.name in ("stage2_summary.json",):
            continue
        d = json.loads(f.read_text())
        runs[d.get("tag", f.stem.replace("stage2_", ""))] = d
    return runs


def fmt(v, plus=True):
    if v is None:
        return "---"
    return f"{v:+.1f}" if plus else f"{v:.1f}"


def main():
    runs = load_runs()
    if not runs:
        print("no runs found")
        return

    def core_mean(rec):
        vals = [rec.get(d, {}).get("recovery_pct") for d in CORE]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    print("=" * 132)
    print("MAIN TABLE -- recovery of the degradation gap (%), 0 = no better than the degraded model")
    print("=" * 132)
    print(f"{'configuration':46} {'noise10':>8} {'noise20':>8} {'dark':>8} {'blur':>8} "
          f"{'3-mean':>8} {'clean d%':>9} {'params':>9}")
    print("-" * 132)
    order = ["blind_matched_long", "blind_matched", "oracle_long", "oracle",
             "noblur_long", "withblur_long", "noblur_2stage_long", "noblur_2stage",
             "cross_2stage_3k", "cross_3k", "hf_3k", "predicted_loo",
             "predicted_noblur_rand", "predicted", "noconv", "input_restore",
             "noblur_auxreg", "auxce_early"]
    seen = set()
    for tag in order + [t for t in runs if t not in order]:
        if tag not in runs or tag in seen:
            continue
        seen.add(tag)
        d = runs[tag]
        rec = d.get("recovery", {})
        print(f"{(LABEL.get(tag) or tag)[:46]:46} "
              f"{fmt(rec.get('noise10', {}).get('recovery_pct')):>8} "
              f"{fmt(rec.get('noise20', {}).get('recovery_pct')):>8} "
              f"{fmt(rec.get('dark', {}).get('recovery_pct')):>8} "
              f"{fmt(rec.get('blur', {}).get('recovery_pct')):>8} "
              f"{fmt(core_mean(rec)):>8} "
              f"{fmt(d.get('clean_change_pct')):>9} "
              f"{d.get('trainable_params', 0):>9,}")
    print("-" * 132)

    # LaTeX
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Degradation robustness. Recovery is normalised against each run's own "
             r"clean reference: $0\%$ is no better than the unrecovered degraded model and "
             r"$100\%$ is fully restored. The three-degradation mean covers the recoverable "
             r"cases; a $5{\times}5$ blur is listed separately.}",
             r"\label{tab:main}", r"\small",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Configuration & noise $\sigma{=}10$ & noise $\sigma{=}20$ & low-light & "
             r"$3$-mean & blur & clean $\Delta$ \\", r"\midrule"]
    for tag in order + [t for t in runs if t not in order]:
        if tag not in runs:
            continue
        d = runs[tag]
        rec = d.get("recovery", {})
        name = (LABEL.get(tag) or tag).replace("_", r"\_").replace("+", r"+")
        bold = tag in ("oracle_long", "noblur_long", "cross_3k", "cross_2stage_3k")
        cm = core_mean(rec)
        cells = [fmt(rec.get(k, {}).get("recovery_pct")) for k in ("noise10", "noise20", "dark")]
        row = (f"{name} & " + " & ".join(f"${c}\\%$" for c in cells)
               + f" & ${fmt(cm)}\\%$ & ${fmt(rec.get('blur', {}).get('recovery_pct'))}\\%$"
               + f" & ${fmt(d.get('clean_change_pct'))}\\%$ \\\\")
        if bold:
            row = r"\textbf{" + row + "}"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = REPO / "runs/00_contract/table_main_dar.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nLaTeX table written to {out}")


if __name__ == "__main__":
    main()
