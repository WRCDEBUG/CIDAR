#!/usr/bin/env python3
"""Apply the frozen PR1 Gate A and tie-break decision rules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PRIMARY = "Q_M"
DELTA_MIN = 0.005
MS2R_REL_WORSE_MAX = 0.01
QABF_DROP_MAX = 0.003
BOOTSTRAP_SEED = 2025
BOOTSTRAP_N = 10000
DEV_DATASET = "M3SVD"
LAYERS = ("e1", "dec1", "bt")
DECISION_LAYER = "dec1"
PRIMARY_RULE = "tie_aux"

OUTPUT_DIR = Path("output/pr1")
STATS_PATH = OUTPUT_DIR / "stats.json"
METRICS_PATH = OUTPUT_DIR / "metrics_by_video.csv"
REPRO_GAP_PATH = OUTPUT_DIR / "repro_gap.json"
DECISION_PATH = OUTPUT_DIR / "decision.txt"


def load_stats():
    stats = json.loads(STATS_PATH.read_text())
    if stats.get("dataset") != DEV_DATASET or stats.get("batch_size") != 1:
        raise ValueError("stats.json does not match the frozen dataset/batch contract")
    frames = stats.get("frames", [])
    if not frames:
        raise ValueError("stats.json has no per-frame records")
    a_lt_k = []
    interchangeable = []
    for frame in frames:
        layer = frame["layers"][DECISION_LAYER]
        a_lt_k.append(float(layer["A_lt_K"]))
        interchangeable.append(float(layer["interchangeable"]))
    leakage = stats.get("leakage_check", [])
    max_gate_mismatch = max((row["gate_mismatch_ratio"] for row in leakage), default=0.0)
    return {
        "a_lt_k_rate": float(np.mean(a_lt_k)),
        "interchangeable_rate": float(np.mean(interchangeable)),
        "max_gate_mismatch": float(max_gate_mismatch),
    }


def gate_a(summary):
    if summary["max_gate_mismatch"] > 1e-4:
        return False, "GATE_MISMATCH"
    passed = summary["a_lt_k_rate"] >= 0.10 or summary["interchangeable_rate"] >= 0.20
    return passed, None


def load_metrics():
    rows = list(csv.DictReader(METRICS_PATH.open()))
    required = {"video", "rule", "seed", "Q_M", "Qabf", "MS2R", "latency_ms"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"metrics_by_video.csv requires columns {sorted(required)}")
    return rows


def mean_by_video(rows, rule, metric, seed=None):
    selected = []
    for row in rows:
        if row["rule"] != rule:
            continue
        if seed is not None and int(row["seed"]) != seed:
            continue
        selected.append((row["video"], float(row[metric])))
    return dict(selected)


def paired(rows, rule, metric, seed=0):
    legacy = mean_by_video(rows, "legacy", metric, seed=0)
    current = mean_by_video(rows, rule, metric, seed=seed)
    videos = sorted(set(legacy) & set(current))
    if not videos:
        raise ValueError(f"No paired {metric} rows for {rule}, seed={seed}")
    return np.asarray([current[v] - legacy[v] for v in videos], dtype=np.float64), videos


def bootstrap_ci(values):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_N, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def rand_is_same_scale(rows, aux_mean):
    rand_means = []
    for seed in (0, 1, 2):
        values, _ = paired(rows, "tie_rand", PRIMARY, seed=seed)
        rand_means.append(float(values.mean()))
    best_rand = max(rand_means)
    same_scale = best_rand > 0 and best_rand >= 0.5 * max(aux_mean, 0.0)
    return same_scale, rand_means


def format_report(decision, gate, numbers, notes):
    lines = [decision]
    repro = json.loads(REPRO_GAP_PATH.read_text()) if REPRO_GAP_PATH.exists() else None
    if repro and repro.get("absolute_gap", 0.0) > DELTA_MIN:
        lines.insert(0, "REPRO_GAP_LARGER_THAN_THRESHOLD")
    lines.extend([
        f"GateA dec1 A<K frame rate: {gate['a_lt_k_rate']:.6f}",
        f"GateA dec1 interchangeable frame rate: {gate['interchangeable_rate']:.6f}",
        f"Gate audit max gated-zero ratio: {gate['max_gate_mismatch']:.6g}",
    ])
    for key, value in numbers.items():
        lines.append(f"{key}: {value}")
    lines.extend(notes)
    return "\n".join(lines) + "\n"


def main():
    gate = load_stats()
    passed, gate_error = gate_a(gate)
    if gate_error:
        report = format_report("GATE_A_FAIL", gate, {}, [gate_error])
    elif not passed:
        report = format_report("GATE_A_FAIL", gate, {}, [])
    else:
        if not METRICS_PATH.exists():
            raise FileNotFoundError("Gate A passed; run tools/pr1_tiebreak.py before deciding")
        rows = load_metrics()
        q_diff, videos = paired(rows, PRIMARY_RULE, PRIMARY, seed=0)
        q_mean = float(q_diff.mean())
        q_low, q_high = bootstrap_ci(q_diff)
        qabf_diff, _ = paired(rows, PRIMARY_RULE, "Qabf", seed=0)
        qabf_drop = max(0.0, -float(qabf_diff.mean()))
        legacy_ms2r = mean_by_video(rows, "legacy", "MS2R", seed=0)
        aux_ms2r = mean_by_video(rows, PRIMARY_RULE, "MS2R", seed=0)
        common = sorted(set(legacy_ms2r) & set(aux_ms2r))
        base_ms2r = float(np.mean([legacy_ms2r[v] for v in common]))
        rule_ms2r = float(np.mean([aux_ms2r[v] for v in common]))
        ms2r_rel_worse = (rule_ms2r - base_ms2r) / base_ms2r
        rand_same_scale, rand_means = rand_is_same_scale(rows, q_mean)

        is_pass = (
            q_mean >= DELTA_MIN
            and q_low > 0
            and ms2r_rel_worse <= MS2R_REL_WORSE_MAX
            and qabf_drop <= QABF_DROP_MAX
            and REPRO_GAP_PATH.exists()
        )
        if is_pass and rand_same_scale:
            decision = "INCONCLUSIVE"
        elif is_pass:
            decision = "PASS"
        elif q_high < DELTA_MIN:
            decision = "FAIL"
        else:
            decision = "INCONCLUSIVE"
        notes = []
        if rand_same_scale:
            notes.append("SECONDARY_KEY_NOT_IDENTIFIED")
        numbers = {
            "videos": len(videos),
            "tie_aux Q_M paired mean": f"{q_mean:.9f}",
            "tie_aux Q_M bootstrap 95% CI": f"[{q_low:.9f}, {q_high:.9f}]",
            "tie_aux MS2R relative worse": f"{ms2r_rel_worse:.9f}",
            "tie_aux Qabf mean drop": f"{qabf_drop:.9f}",
            "tie_rand Q_M paired means seeds 0,1,2": ", ".join(f"{x:.9f}" for x in rand_means),
        }
        report = format_report(decision, gate, numbers, notes)

    DECISION_PATH.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
