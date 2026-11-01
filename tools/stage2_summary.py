#!/usr/bin/env python3
"""Collect every Stage-2 run into one comparison table.

Reads output/pr1/stage2_*.json and prints recovery per degradation next to the
Stage-1 blind fine-tune reference, so the DAR claim (a smaller conditional module
beats a larger blind fine-tune) can be read off directly.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF = {"clean": 0.7364, "noise10": 0.5823, "noise20": 0.4777, "dark": 0.5638, "blur": 0.2162}
DEGS = ("clean", "noise10", "noise20", "dark", "blur", "noise15", "dark22")
BLIND = {"clean": -1.48, "noise10": 9.68, "noise20": 8.90, "dark": 29.38, "blur": 0.31}


def main():
    files = sorted((REPO / "output/pr1").glob("stage2_*.json"))
    if not files:
        print("no stage2 results yet")
        return
    rows = []
    for f in files:
        if f.name == "stage2_summary.json":
            continue
        d = json.loads(f.read_text())
        rows.append(d)

    print(f"{'run':18} {'mode':10} {'restore':7} {'params':>9} {'trained on':28} "
          + " ".join(f"{x:>8}" for x in DEGS))
    print("-" * 130)
    print(f"{'BLIND fine-tune':18} {'-':10} {'-':7} {265573:9,} {'all 4 + clean':28} "
          + " ".join(f"{BLIND.get(x, float('nan')):+8.2f}" for x in DEGS))
    for d in rows:
        rec = d.get("recovery", {})
        cells = []
        for x in DEGS:
            v = rec.get(x, {}).get("recovery_pct")
            cells.append(f"{v:+8.2f}" if isinstance(v, (int, float)) else f"{'n/a':>8}")
        print(f"{d['tag']:18} {d['dar_mode']:10} {str(d.get('input_restore', False)):7} "
              f"{d['trainable_params']:9,} {','.join(d['train_degradations'])[:28]:28} " + " ".join(cells))
    print("-" * 130)
    print("\nrecovery% is measured against the clean reference of that same run, so it is\n"
          "normalised: 0% = no better than the unrecovered degraded model, 100% = fully restored.\n")
    for d in rows:
        rec, un = d.get("recovery", {}), d.get("unseen_degs", [])
        # clean must never enter the recovery average: its gap is ~0, so the
        # normalised number is meaningless (early runs stored a bogus -336%)
        seen = [x for x in DEGS if x != "clean" and x not in un
                and rec.get(x, {}).get("recovery_pct") is not None]
        if not seen:
            continue
        mean_seen = sum(rec[x]["recovery_pct"] for x in seen) / len(seen)
        unseen_txt = "  ".join(
            f"{x}={rec[x]['recovery_pct']:+.1f}%" for x in un
            if rec.get(x, {}).get("recovery_pct") is not None)
        common = [x for x in ("noise10", "noise20", "dark")
                  if rec.get(x, {}).get("recovery_pct") is not None]
        mean_common = (sum(rec[x]["recovery_pct"] for x in common) / len(common)) if common else float("nan")
        print(f"{d['tag']:18} mean(seen) = {mean_seen:+6.2f}%   mean over the 3 recoverable = "
              f"{mean_common:+6.2f}%   clean {d.get('clean_change_pct', 0):+5.2f}%"
              f"   unseen: {unseen_txt}")

    out = REPO / "output/pr1/stage2_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
