#!/usr/bin/env python3
"""Architecture figure for the paper: where DAR sits and what each part does."""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = pathlib.Path(__file__).resolve().parents[1] / "runs/00_contract/figures"
FIG.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, text, fc="#eef3f8", ec="#2c3e50", fs=9.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.5, linestyle=ls))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, color="#2c3e50", lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                                 lw=lw, linestyle=ls, mutation_scale=14,
                                 connectionstyle=f"arc3,rad={rad}"))

ax.text(50, 60, "DAR: degradation-aware feature rectification", ha="center",
        fontsize=13.5, fontweight="bold")

# ---------- estimator (top) ----------
box(30, 49, 40, 7.5, "degradation estimator   $c = g(x)\\in\\mathbb{R}^{5}$",
    fc="#fcf3cf", ec="#b7950b", fs=11)
arrow(50, 49, 40, 42.8, color="#b7950b")     # down to IR rectifier
arrow(50, 49, 40, 25.8, color="#b7950b")     # down to VIS rectifier

# ---------- IR row ----------
box(1, 35, 13, 7.5, "IR window", fc="#fdebd0")
box(17, 35, 12, 7.5, "feat ext$_{ir}$")
box(32, 35, 14, 7.5, "DAR rect$_{ir}$", fc="#d5f5e3", ec="#1e8449", bold=True)
box(49, 35, 15, 7.5, "AAM$_{ir}$")
arrow(14, 38.75, 17, 38.75); arrow(29, 38.75, 32, 38.75); arrow(46, 38.75, 49, 38.75)

# ---------- VIS row ----------
box(1, 18, 13, 7.5, "VIS window", fc="#d6eaf8")
box(17, 18, 12, 7.5, "feat ext$_{vi}$")
box(32, 18, 14, 7.5, "DAR rect$_{vi}$", fc="#d5f5e3", ec="#1e8449", bold=True)
box(49, 18, 15, 7.5, "AAM$_{vi}$")
arrow(14, 21.75, 17, 21.75); arrow(29, 21.75, 32, 21.75); arrow(46, 21.75, 49, 21.75)

# raw frames feed the estimator (dotted)
arrow(7, 42.5, 33, 49, color="#b7950b", ls=":", lw=1.2, rad=-0.12)
arrow(7, 18, 33, 49.4, color="#b7950b", ls=":", lw=1.2, rad=0.22)
ax.text(15.5, 53.2, "raw frames, both modalities", fontsize=8.5, color="#b7950b", style="italic")

# ---------- optional second injection ----------
box(70, 26.5, 13, 8.5, "DAR rect$^{(2)}$\n(optional)", fc="#eafaf1", ec="#1e8449",
    fs=9, ls="--")
arrow(64, 38.75, 70, 33.5); arrow(64, 21.75, 70, 27.5)
arrow(50+25.5, 49, 76.5, 35.2, color="#b7950b", ls="--")

# ---------- fusion ----------
box(87, 26.5, 12, 8.5, "fusion net\n(sparse\ninteraction)", fc="#e8daef", ec="#6c3483", fs=9)
arrow(83, 30.75, 87, 30.75)
box(87, 12, 12, 6, "fused\nframes", fc="#fadbd8", ec="#922b21", fs=9.5)
arrow(93, 26.5, 93, 18, color="#922b21")

# ---------- annotations ----------
ax.text(39, 45.4, "injection point 1: before any cross-frame mixing",
        ha="center", fontsize=8.6, color="#1e8449")
ax.text(76.5, 24.2, "injection point 2", ha="center", fontsize=8.2, color="#1e8449")

box(2, 1.5, 44, 7.5,
    "zero-initialised output $\\Rightarrow$ at step 0 the modified model is\n"
    "bit-identical to the released one (verified, max$|\\Delta|$ = 0)",
    fc="#fef9e7", ec="#b7950b", fs=8.6)
box(52, 1.5, 46, 7.5,
    "recovers what the descriptor can identify:  noise (one modality) and\n"
    "low-light (both modalities).  Blur stays open: the stem probes at chance\n"
    "(0.44 vs 0.95 spectrally), so the fix is a representation change.",
    fc="#eafaf1", ec="#1e8449", fs=8.2)

fig.savefig(FIG / "fig_architecture.png", dpi=200, bbox_inches="tight")
print("wrote fig_architecture.png")
