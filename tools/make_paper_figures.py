#!/usr/bin/env python3
"""Figures 1-2 for the DAR paper, built from the measured JSON artifacts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "runs/00_contract/figures"
FIG.mkdir(parents=True, exist_ok=True)
DEGS = ["noise10", "noise20", "dark", "blur"]
LABEL = {"noise10": r"noise $\sigma$=10", "noise20": r"noise $\sigma$=20",
         "dark": "low-light", "blur": "5x5 blur"}

# ---------- Figure 1: where the degradation collapse comes from ----------
dec = json.loads((REPO / "output/pr1/day4_decomposition.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
feat = [dec["decomposition"][d]["feature_path_remainder_pct_of_clean"] for d in DEGS]
flow = [dec["decomposition"][d]["flow_path_recovery_pct_of_clean"] for d in DEGS]
x = np.arange(len(DEGS)); w = 0.38
ax[0].bar(x - w/2, feat, w, label="feature path", color="#c0392b")
ax[0].bar(x + w/2, flow, w, label="optical-flow path", color="#7f8c8d")
ax[0].set_xticks(x); ax[0].set_xticklabels([LABEL[d] for d in DEGS])
ax[0].set_ylabel("Q$_{AB/F}$ drop (% of clean)")
ax[0].set_title("(a) The collapse is entirely in the feature path")
ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
for i, v in enumerate(feat):
    ax[0].text(i - w/2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
ax[0].text(0.56, 0.74, "replaying a perfect clean\nflow recovers nothing\n(flow path $\\approx$ 0.0%)",
           transform=ax[0].transAxes, ha="center", fontsize=9,
           bbox=dict(boxstyle="round", fc="#fdf3d0", ec="#d0b040"))
qe = dec["flow_error_vs_clean_flow"]
sp = [qe[d]["spurious_mask_pct"] for d in DEGS]; ms = [qe[d]["missed_mask_pct"] for d in DEGS]
ax[1].bar(x - w/2, sp, w, label="spurious motion", color="#e67e22")
ax[1].bar(x + w/2, ms, w, label="missed motion", color="#2980b9")
ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[d] for d in DEGS])
ax[1].set_ylabel("motion-mask pixels (%)")
ax[1].set_title("(b) ...even though the flow really does degrade")
ax[1].legend(); ax[1].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "fig1_path_decomposition.png", dpi=200)
print("wrote fig1")

# ---------- Figure 2: recovery vs the blind fine-tune ----------
s2 = json.loads((REPO / "output/pr1/stage2_oracle.json").read_text())
BLIND = {"noise10": 9.68, "noise20": 8.90, "dark": 29.38, "blur": 0.31}
dar = [s2["recovery"][d]["recovery_pct"] for d in DEGS]
blind = [BLIND[d] for d in DEGS]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
x = np.arange(len(DEGS)); w = 0.38
b1 = ax.bar(x - w/2, blind, w, label=f"blind fine-tune ({s2['blind_finetune_reference'] and 265573:,} params)",
            color="#95a5a6")
b2 = ax.bar(x + w/2, dar, w, label=f"DAR, oracle cond. ({s2['dar_params']:,} params)", color="#27ae60")
ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in DEGS])
ax.set_ylabel("recovery of the degradation gap (%)")
ax.set_title("DAR doubles the recovery of a larger blind fine-tune")
ax.axhline(15, ls="--", c="k", lw=1); ax.text(3.45, 16, "15% target", fontsize=8, ha="right")
ax.legend(); ax.grid(axis="y", alpha=.3)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1, f"{r.get_height():.1f}",
                ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_recovery_vs_blind.png", dpi=200)
print("wrote fig2")

# ---------- Figure 3: the conditioning ladder ----------
LADDER = [
    ("no fix\n(baseline)", 0.0, "#95a5a6"),
    ("feature-side\n(no cond.)", 8.80, "#bdc3c7"),
    ("input-side\n(no cond.)", 9.62, "#bdc3c7"),
    ("feature-side\n(unsup. cond., 3k)", 14.66, "#3498db"),
    ("blind fine-tune\n3k (266k params)", 22.32, "#8e44ad"),
    ("feature-side\n(unsup. cond., 8k)", 20.74, "#3498db"),
    ("feature-side\n(SUPERVISED, 8k)", 39.63, "#27ae60"),
]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
names = [l[0] for l in LADDER]; vals = [l[1] for l in LADDER]; cols = [l[2] for l in LADDER]
b = ax.bar(np.arange(len(vals)), vals, color=cols)
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("recovery of the degradation gap (%)\nmean over noise10 / noise20 / low-light")
ax.set_title("Where the gain comes from: conditioning, and it must be supervised")
ax.axhline(22.32, ls="--", c="#8e44ad", lw=1)
ax.text(len(vals)-0.5, 22.9, "blind fine-tune (3k)", fontsize=8, ha="right", color="#8e44ad")
ax.grid(axis="y", alpha=.3)
for r in b:
    ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.5, f"{r.get_height():.1f}",
            ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"fig3_conditioning_ladder.png", dpi=200)
print("wrote fig3")

# ---------- Figure 4: LLVIP real-data transfer ----------
ll = json.loads((REPO/"output/pr1/real_lowlight_predicted_multi.json").read_text())
DEGS4 = ["none", "noise10", "noise20", "dark", "blur"]
LBL = {"none": "real night\n(no added deg.)", "noise10": r"$+\sigma$=10",
       "noise20": r"$+\sigma$=20", "dark": "low-light\n(added)", "blur": "+5x5 blur"}
pct = [(ll["summary"][d]["Qabf"]["dar_mean"]/ll["summary"][d]["Qabf"]["baseline_mean"]-1)*100 for d in DEGS4]
wins = [ll["summary"][d]["Qabf"]["wins"] for d in DEGS4]
n = ll["summary"]["none"]["Qabf"]["n"]
cols = ["#27ae60" if p > 0 else "#c0392b" for p in pct]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
b = ax.bar(np.arange(len(pct)), pct, color=cols)
ax.axhline(0, c="k", lw=1)
ax.set_xticks(np.arange(len(pct))); ax.set_xticklabels([LBL[d] for d in DEGS4], fontsize=9)
ax.set_ylabel("Q$_{AB/F}$ change vs baseline (%)")
ax.set_title(f"Real night video (LLVIP, {n} windows): noise transfers, low-light does not")
ax.grid(axis="y", alpha=.3)
for r, w in zip(b, wins):
    y = r.get_height()
    ax.text(r.get_x()+r.get_width()/2, y + (0.25 if y >= 0 else -0.6), f"{y:+.2f}%\n{w}/{n}",
            ha="center", fontsize=8.5)
ax.set_ylim(min(pct)-2.2, max(pct)+2.2)
fig.tight_layout(); fig.savefig(FIG/"fig4_llvip_transfer.png", dpi=200)
print("wrote fig4")


# ---------- Figure 5: identifiability ----------
IDENT = [
    ("noise $\\sigma$=10\n(identifiable)", 16.11, 12.38),
    ("noise $\\sigma$=20\n(identifiable)", 13.47, 12.18),
    ("low-light\n(ambiguous)", 27.39, 42.40),
    ("5x5 blur\n(unseen by stem)", 0.19, 0.45),
]
fig, ax = plt.subplots(figsize=(8.0, 4.4))
x = np.arange(len(IDENT)); w = 0.38
d1 = [a for _, a, _ in IDENT]; d2 = [b for _, _, b in IDENT]
b1 = ax.bar(x - w/2, d1, w, label="DAR (estimated descriptor)", color="#27ae60")
b2 = ax.bar(x + w/2, d2, w, label="blind fine-tune (266k)", color="#8e44ad")
ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in IDENT], fontsize=9)
ax.set_ylabel("recovery of the degradation gap (%)")
ax.set_title("The module wins where the degradation is identifiable from the input")
ax.grid(axis="y", alpha=.3); ax.legend()
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.6, f"{r.get_height():.1f}",
                ha="center", fontsize=8.5)
ax.annotate("oracle recovers\n+43.4% here", xy=(3-w/2, 0.19), xytext=(2.35, 20),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b")
fig.tight_layout(); fig.savefig(FIG/"fig5_identifiability.png", dpi=200)
print("wrote fig5")
