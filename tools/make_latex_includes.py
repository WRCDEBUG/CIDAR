#!/usr/bin/env python3
"""Regenerate the LaTeX table includes used by paper_full_draft.tex.

Kept separate from tools/make_paper_tables.py (which prints the human-readable
comparison and writes the single big table) because these are the small tables
the draft \\inputs.
"""

from __future__ import annotations

import json
import pathlib
import shutil

R = pathlib.Path(__file__).resolve().parents[1]
O = R / "output/pr1"
H = R / "runs/00_contract"


def rd(p, default=None):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def core_mean(d):
    v = [d.get("recovery", {}).get(k, {}).get("recovery_pct") for k in ("noise10", "noise20", "dark")]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def f1(x):
    return f"${x:+.1f}\\%$" if x is not None else "---"


# best deployable checkpoint, quoted in the identifiability and real-data tables
BEST = "cross_noblur_long"


def main():
    dec = rd(O / "day4_decomposition.json")
    runs = {}
    for f in O.glob("stage2_*.json"):
        if f.name == "stage2_summary.json":
            continue
        d = rd(f)
        runs[d.get("tag", f.stem)] = d

    LBL = {"noise10": r"noise $\sigma{=}10$", "noise20": r"noise $\sigma{=}20$",
           "dark": "low-light", "blur": r"$5{\times}5$ blur"}

    # ---- decomposition ----
    rows = []
    for d in ("noise10", "noise20", "dark", "blur"):
        s = dec["decomposition"][d]
        e = dec["flow_error_vs_clean_flow"][d]
        rows.append(f"{LBL[d]} & ${s['total_drop_pct']:.2f}\\%$ & "
                    f"${s['flow_path_recovery_pct_of_clean']:.2f}\\%$ & "
                    f"$\\mathbf{{{s['feature_path_remainder_pct_of_clean']:.2f}\\%}}$ & "
                    f"${e['epe']:.3f}$ \\\\")
    (H / "tab_decomposition.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Path decomposition.  Configuration C replays a perfect clean flow and
motion mask into the degraded pass, so the flow path is given everything it
could have had; it recovers nothing, and the collapse is entirely in the
feature path.}
\label{tab:decomposition}
\small
\begin{tabular}{lcccc}
\toprule
Degradation & Total drop & Flow path & Feature path & Flow EPE \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ---- conditioning ablation ----
    # The training-degradation set differs between groups, and that difference is
    # large enough to change the conclusion, so it is stated in the label.
    ORDER = [("noconv", "no conditioning (capacity control)", 222112),
             ("input_restore", "input-side restorer (placement control)", 38931),
             ("predicted", "trained on 4 degradations", 262915),
             ("hf_3k", "4 degs + high-frequency channel", 262915),
             ("cross_3k", "4 degs + CROSS-MODAL descriptor", 262915),
             ("auxce_early", "4 degs + auxiliary classification", 262915),
             ("noblur_auxreg", "4 degs + auxiliary regression", 262927),
             ("predicted_noblur_rand", "4 degs + randomised low-light", 262915),
             ("predicted_loo", "trained on 3 degradations (no blur)", 262915),
             ("cross_noblur_long", "3 degs + CROSS-MODAL, 8k", 264367),
             ("cross_2stage_long", "3 degs + CROSS-MODAL + 2nd injection, 8k", 343087),
             ("noblur_long", "3 degs, 8k steps", 262927),
             ("noblur_2stage_long", "3 degs, + second injection, 8k", 341647),
             ("withblur_long", "4 degs, 8k steps", 262927),
             ("blind_matched", "blind fine-tune, 3k steps", 265573),
             ("blind_matched_long", "blind fine-tune, 8k steps", 265573),
             ("oracle", "TRUE descriptor, 3k", 242272),
             ("oracle_long", "TRUE descriptor, 8k", 242272)]
    seeds = rd(O / "headline_seeds.json") or {}
    lines = []
    for tag, label, params in ORDER:
        d = runs.get(tag)
        if d is None:
            continue
        cm = core_mean(d)
        lines.append(f"{label} & ${params:,}$ & {f1(cm)} & {f1(d.get('clean_change_pct'))} \\\\")
        if tag == "blind_matched_long":
            bs = rd(O / "blind_seeds.json") or {}
            if bs:
                lines.append(f"\\quad (same, mean of 3 seeds) & ${params:,}$ & "
                             f"${bs['mean_of_means']:.1f} \\pm {bs['sd_of_means']:.1f}\\%$ & $-3.4\\%$ \\\\")
        if tag == "cross_noblur_long" and seeds:
            m, sd = seeds["mean_of_means"], seeds["sd_of_means"]
            lines.append(f"\\quad (same, mean of 3 seeds) & ${params:,}$ & "
                         f"$\\mathbf{{{m:.1f} \\pm {sd:.1f}}}\\%$ & $-0.6\\%$ \\\\")
    (H / "tab_conditioning.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Conditioning ablation, mean recovery over the three recoverable
degradations.  Placement does not matter without conditioning; capacity is not
the limiting resource; a supervised descriptor is worth nearly twice a blind
fine-tune of the whole network, with fewer parameters and far less forgetting.}
\label{tab:conditioning}
\small
\begin{tabular}{lrrr}
\toprule
Configuration & Trainable & Recovery & Clean $\Delta$ \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ---- identifiability ----
    def g(tag, k):
        d = runs.get(tag, {})
        v = d.get("recovery", {}).get(k, {}).get("recovery_pct")
        return f"${v:+.1f}\\%$" if v is not None else "---"
    # Blind column uses the STEP-MATCHED 8k run so the comparison is fair.
    BL = "blind_matched_long"
    (H / "tab_identifiability.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Recovery per degradation, DAR against a step-matched blind fine-tune
(both 8k steps).  DAR wins on sensor noise, and --- because the cross-modal
descriptor removes the single-modality ambiguity --- on low-light too.  A
$5{\times}5$ blur is the remaining open case: an oracle descriptor recovers
$43.4\%$ of it, so it is a descriptor problem rather than a limit of the module.}
\label{tab:identifiability}
\small
\begin{tabular}{llcc}
\toprule
Degradation & Identifiable from & DAR & Blind fine-tune \\
\midrule
noise $\sigma{=}10$ & one modality & """ + g(BEST, "noise10") + " & " + g(BL, "noise10") + r""" \\
noise $\sigma{=}20$ & one modality & """ + g(BEST, "noise20") + " & " + g(BL, "noise20") + r""" \\
low-light & both modalities & """ + g(BEST, "dark") + " & " + g(BL, "dark") + r""" \\
$5{\times}5$ blur & neither (yet) & """ + g(BEST, "blur") + " & " + g(BL, "blur") + r""" \\
\midrule
blur, oracle descriptor & --- & $+43.4\%$ & --- \\
\bottomrule
\end{tabular}
\end{table}
""")

    # ---- LLVIP ----
    # best DAR checkpoint on real data, against the step-matched blind fine-tune
    ll = (rd(O / "real_lowlight_cross_noblur_long.json")
          or rd(O / "real_lowlight_withblur_long.json"))
    bl = rd(O / "real_lowlight_blind_long.json") or rd(O / "real_lowlight_blind_matched.json")
    S = {"none": "none (real night)", "noise10": r"noise $\sigma{=}10$",
         "noise20": r"noise $\sigma{=}20$", "dark": "low-light (added)",
         "blur": r"$5{\times}5$ blur"}
    rows = []
    for k in ("none", "noise10", "noise20", "dark", "blur"):
        a = ll["summary"][k]["Qabf"]
        b = (bl or {}).get("summary", {}).get(k, {}).get("Qabf")
        cell = f"${b['delta']/b['baseline_mean']*100:+.2f}\\%$" if b else "---"
        rows.append(f"{S[k]} & ${a['baseline_mean']:.4f}$ & ${a['dar_mean']:.4f}$ & "
                    f"$\\mathbf{{{a['delta']/a['baseline_mean']*100:+.2f}\\%}}$ & "
                    f"${a['wins']}/{a['n']}$ & {cell} \\\\")
    (H / "tab_llvip.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Real night video (LLVIP, 32 windows), with a synthetic degradation
added on top of the real frames so that a clean reference genuinely exists.
The blind fine-tune helps only on noise and damages every other condition,
including clean night footage; DAR is positive on four of five conditions.}
\label{tab:llvip}
\small
\begin{tabular}{lccccc}
\toprule
Added degradation & Baseline & DAR & Change & Wins & Blind \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ---- deployment ----
    dep = rd(O / "deploy/deploy_summary.json", {})
    RS = {"M3SVD": r"$480{\times}640$", "VTMOT": r"$576{\times}768$", "HDO": r"$1024{\times}1280$"}
    rows = []
    for k in ("M3SVD", "VTMOT", "HDO"):
        v = dep.get(k)
        if v:
            rows.append(f"{k} & {RS[k]} & ${v['baseline_ms']:.1f}$ & ${v['fast_ms']:.1f}$ & "
                        f"$\\mathbf{{{v['speedup']:.2f}\\times}}$ & ${v['max_abs_metric_delta']:.4f}$ \\\\")
    (H / "tab_deploy.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Deployment package: cached flow reuse, CUDA-graph replay and a
corrected motion-compensation direction.  Only the CUDA-graph component is
bit-identical.}
\label{tab:deploy}
\small
\begin{tabular}{llcccc}
\toprule
Dataset & Resolution & Base (ms) & Ours (ms) & Speedup & max $|\Delta|$ \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ---- identifiability probe ----
    pr = rd(O / "identifiability_probe.json")
    if pr:
        # keys are the probe script's own labels; keep them in sync
        order = [("stem only (untrained)", "stem only, untrained"),
                 ("stem only (trained, 4 degs)", "stem only, trained"),
                 ("cross-modal (trained, 3 degs)", "stem + cross-modal"),
                 ("+ spectral (trained, 4 degs)", "stem + cross-modal + spectral"),
                 ("spectral stats only (no net)", "spectral statistics alone (no network)")]
        rows = []
        for k, lab in order:
            v = pr["results"].get(k)
            if not v:
                continue
            rows.append(lab + " & $" + str(v['dim']) + "$ & $"
                        + f"{v['accuracy_all']:.3f}" + "$ & $"
                        + f"{v['accuracy_blur_vs_rest']:.3f}" + "$ " + chr(92) * 2)
        (H / "tab_probe.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Measuring identifiability.  A linear probe on the degradation
descriptor must name which of five degradations was applied; chance is $0.20$
for the five-way task and $0.50$ for blur-versus-rest.  The released stem is at
chance and \emph{training does not change that}, so the limitation is in the
representation rather than the optimisation.  Hand-computed spectral statistics
alone, with no network, are almost perfectly separable: blur is linearly
identifiable by construction once the representation carries frequency content.}
\label{tab:probe}
\small
\begin{tabular}{lccc}
\toprule
Descriptor & dim & 5-way acc. & blur vs rest \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")
        print("probe table written")

    # ---- blur: identifiability before vs after training ----
    BS = chr(92)                      # a LaTeX row terminator is two backslashes
    prb = rd(O / "identifiability_probe.json") or {}
    res = prb.get("results", {})
    def pr(key):
        v = res.get(key)
        if not v:
            return "--- & ---"
        return f"${v['accuracy_all']:.3f}$ & ${v['accuracy_blur_vs_rest']:.3f}$"
    def blur(k):
        v = runs.get(k, {}).get("recovery", {}).get("blur", {}).get("recovery_pct")
        return f"${v:+.2f}\\%$" if v is not None else "---"
    def row(*cells):
        return " & ".join(cells) + " " + BS + BS
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Supplying the information is not the same as using it, and the "
        r"natural explanations all fail.  Raw "
        r"spectral statistics carry almost perfect blur information and a linear "
        r"probe recovers it, but after end-to-end training the descriptor retains "
        r"only part of it ($0.953\to0.593$) and blur recovery stays near zero, while "
        r"an oracle descriptor recovers $43.4\%$.  Identifiability is therefore "
        r"\emph{necessary but not sufficient}: information the downstream "
        r"rectification loss does not reward is discarded by the optimiser.  A "
        r"no-stride stem, which refutes our own downsampling story, and an "
        r"auxiliary classification loss both leave blur near zero.}",
        r"\label{tab:blur}", r"\small",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Descriptor & 5-way probe & blur vs rest & blur recovery " + BS + BS,
        r"\midrule",
        row("released stem, trained", pr("stem only (trained, 4 degs)"), blur("predicted")),
        row("spectral statistics alone, no network", pr("spectral stats only (no net)"), "---"),
        row("stem + cross-modal + spectral, trained", pr("+ spectral (trained, 4 degs)"), blur("crossspec_long")),
        row("spectral only, trained", "--- & ---", blur("spectral_long")),
        row("no-stride stem, trained (refutes the downsampling story)",
            pr("no-stride stem (control)"), blur("nostride_long")),
        row("spectral + auxiliary classification", "--- & ---", blur("crossspec_auxce")),
        r"\midrule",
        row("oracle descriptor (constant per degradation)", "--- & ---", blur("oracle_long")),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (H / "tab_blur.tex").write_text(chr(10).join(lines) + chr(10))
    print("blur table written")

    # ---- real-video transfer table ----
    xv = rd(O / "crossvideo_degradation.json")
    if xv:
        BS2 = chr(92) * 2
        LBL = [("none", "none (clean)"), ("noise10", r"noise $\sigma{=}10$"),
               ("noise20", r"noise $\sigma{=}20$"), ("dark", "low-light"), ("blur", "blur")]
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Transfer to real video.  The reference is the real (clean) frame "
                 r"and a synthetic degradation is added on top, so recovery is well defined. "
                 r"Noise transfers on both datasets, winning every window; the first row of "
                 r"each block is the defect this test exposed --- on clean real video the "
                 r"ungated module actively hurts.}",
                 r"\label{tab:xvideo}", r"\small",
                 r"\begin{tabular}{llccc}", r"\toprule",
                 r"Dataset & Degradation & Baseline & DAR & Change " + BS2, r"\midrule"]
        for ds in ("VTMOT", "HDO"):
            if ds not in xv:
                continue
            blocks = [LBL[0]] + LBL[1:]
            for i, (k, lab) in enumerate(blocks):
                r_ = xv[ds].get(k)
                if not r_:
                    continue
                head = ds if i == 0 else ""
                chg = r_["delta_pct"]
                star = r"$\mathbf{" + f"{chg:+.2f}" + r"\%}$" if k == "none" else f"${chg:+.2f}\\%$"
                wins = f" ({r_['wins']}/{r_['n']})" if k.startswith("noise") else ""
                lines.append(f"{head} & {lab} & ${r_['baseline']:.4f}$ & ${r_['dar']:.4f}$ & "
                             f"{star}{wins} " + BS2)
            if ds != "HDO":
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (H / "tab_xvideo.tex").write_text(chr(10).join(lines) + chr(10))
        print("real-video table written")

    # ---- new mechanisms: identifiability collapse, DDC, gate ----
    BS3 = chr(92) * 2
    MECH = [("crossspec_long", "conditioned, no protection", "0.593", "--"),
            ("ddc_mix", "DDC (demand-driven, spectral)", None, None),
            ("ddc_temporal", "DDC + temporal fingerprint", None, None),
            ("temporal_nomix", "temporal fingerprint, no DDC", None, None),
            ("gate_cross", "degradation gate", None, None)]
    prb_ddc = rd(O / "identifiability_probe_ddc.json") or {}
    rows = []
    for tag, label, probe_all, _ in MECH:
        d = runs.get(tag)
        if not d:
            continue
        rec = d.get("recovery", {})
        blur = rec.get("blur", {}).get("recovery_pct")
        dark = rec.get("dark", {}).get("recovery_pct")
        n10 = rec.get("noise10", {}).get("recovery_pct")
        core = core_mean(d)
        rows.append(" & ".join([
            label,
            f"${blur:+.2f}{chr(92)}%$" if blur is not None else "---",
            f"${dark:+.1f}{chr(92)}%$" if dark is not None else "---",
            f"${core:.1f}{chr(92)}%$" if core is not None else "---",
            f"${probe_all}$" if probe_all else "---"]) + " " + BS3)
    if rows:
        (H / "tab_mechanisms.tex").write_text(chr(10).join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{The two new mechanisms.  DDC is supposed to restore blur "
            r"recovery by making the consumer demand the information; the gate is "
            r"supposed to remove the clean-video loss.  Both are reported against the "
            r"unprotected baseline in the first row.}",
            r"\label{tab:mech}", r"\small",
            r"\begin{tabular}{lcccc}", r"\toprule",
            r"Configuration & blur & low-light & 3-mean & probe " + BS3,
            r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
        print("mechanism table written")

    for f in ("fig1_path_decomposition.png", "fig5_identifiability.png"):
        src = R / "runs/00_contract/figures" / f
        if src.exists():
            shutil.copy(src, H / f)
    print("LaTeX includes regenerated:", sorted(p.name for p in H.glob("tab_*.tex")))


if __name__ == "__main__":
    main()
