#!/usr/bin/env python3
"""Check that every number quoted in the paper still matches the experiment JSON.

Run this before submitting.  It exists because a paper that silently drifts from
its experiments is the most common way a result becomes unreproducible.
"""
from __future__ import annotations
import json
import pathlib
import sys

O = pathlib.Path(__file__).resolve().parents[1] / "output/pr1"


def rd(n):
    p = O / n
    return json.loads(p.read_text()) if p.exists() else None


def core(tag):
    d = rd(f"stage2_{tag}.json")
    if not d:
        return None
    r = d["recovery"]
    v = [r[k]["recovery_pct"] for k in ("noise10", "noise20", "dark")
         if r.get(k, {}).get("recovery_pct") is not None]
    return sum(v) / len(v)


CONDITIONING = [("noconv", 8.8), ("input_restore", 9.6), ("predicted", 9.4),
                ("hf_3k", 7.3), ("cross_3k", 13.5), ("auxce_early", 4.9),
                ("noblur_auxreg", 7.5), ("predicted_noblur_rand", 5.4),
                ("predicted_loo", 14.7), ("noblur_long", 19.0),
                ("cross_2stage_long", 36.0), ("blind_matched", 22.3),
                ("blind_matched_long", 26.1), ("oracle", 31.5), ("oracle_long", 39.6)]
PROBE = {"stem only (untrained)": (0.447, 0.707), "stem only (trained, 4 degs)": (0.447, 0.700),
         "cross-modal (trained, 3 degs)": (0.433, 0.567), "+ spectral (trained, 4 degs)": (0.593, 0.913),
         "no-stride stem (control)": (0.413, 0.800), "spectral stats only (no net)": (0.953, 1.000)}
BLUR = [("predicted", 0.16), ("crossspec_long", 0.31), ("spectral_long", 0.02),
        ("nostride_long", 0.08), ("crossspec_auxce", 0.14), ("oracle_long", 43.41)]


def main():
    bad = 0
    print("== conditioning table ==")
    for tag, claimed in CONDITIONING:
        m = core(tag)
        ok = m is not None and abs(m - claimed) <= 0.15
        bad += 0 if ok else 1
        print(f"  {tag:22} {claimed:+7.1f}  {m if m is None else round(m, 2)!s:>7}  {'OK' if ok else 'MISMATCH'}")

    print("== identifiability probe ==")
    pr = (rd("identifiability_probe.json") or {}).get("results", {})
    for k, (a, b) in PROBE.items():
        v = pr.get(k)
        ok = v and abs(v["accuracy_all"] - a) < 0.002 and abs(v["accuracy_blur_vs_rest"] - b) < 0.002
        bad += 0 if ok else 1
        print(f"  {k:32} {a:.3f}/{b:.3f}  "
              f"{'MISSING' if not v else f'{v[chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121)+chr(95)+chr(97)+chr(108)+chr(108)]:.3f}/{v[chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121)+chr(95)+chr(98)+chr(108)+chr(117)+chr(114)+chr(95)+chr(118)+chr(115)+chr(95)+chr(114)+chr(101)+chr(115)+chr(116)]:.3f}'}"
              f"  {'OK' if ok else 'MISMATCH'}")

    print("== blur table ==")
    for tag, claimed in BLUR:
        d = rd(f"stage2_{tag}.json")
        m = d["recovery"]["blur"]["recovery_pct"] if d else None
        ok = m is not None and abs(m - claimed) < 0.02
        bad += 0 if ok else 1
        print(f"  {tag:22} {claimed:+7.2f}  {m if m is None else round(m, 2)!s:>7}  {'OK' if ok else 'MISMATCH'}")

    print("== headline seeds ==")
    s = rd("headline_seeds.json")
    ok = s and abs(s["mean_of_means"] - 36.4) <= 0.15
    bad += 0 if ok else 1
    print(f"  3-seed mean            +36.4  {s['mean_of_means'] if s else 'MISSING'}  {'OK' if ok else 'MISMATCH'}"
          f"   (sd {s['sd_of_means']:.2f} vs quoted 1.9)")

    print("== seed-matched comparison ==")
    bs = rd("blind_seeds.json")
    st = rd("seed_comparison_test.json")
    for label, claimed, got, tol in (
            ("DAR 3-seed mean", 36.4, (s or {}).get("mean_of_means"), 0.15),
            ("blind 3-seed mean", 27.1, (bs or {}).get("mean_of_means"), 0.15),
            ("paired mean difference", 9.3, (st or {}).get("paired", {}).get("mean"), 0.1),
            ("paired t statistic", 23.6, (st or {}).get("paired", {}).get("t"), 0.2),
            ("paired p (two-sided)", 0.0018, (st or {}).get("paired", {}).get("p_two_sided_t"), 0.0006)):
        ok = got is not None and abs(got - claimed) <= tol
        bad += 0 if ok else 1
        print(f"  {label:24} {claimed:>8}  {got if got is None else round(got, 4)!s:>8}  {'OK' if ok else 'MISMATCH'}")

    print(f"\nmismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
