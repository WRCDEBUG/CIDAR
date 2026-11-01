#!/usr/bin/env python3
"""Stage the exact frames the metric contract scores as a flat ir/vis pair of
directories, so a third-party method with a directory-based CLI can be run
unmodified.  Symlinks, so staging costs nothing."""
import argparse, json, os, pathlib
SETS = {"M3SVD": "frozen5_test", "HDO": "HDO_frozen5_test", "VTMOT": "VTMOT_frozen5_test"}
EXT = (".jpg", ".png", ".jpeg", ".bmp")
p = argparse.ArgumentParser()
p.add_argument("--datasets", default="M3SVD,HDO,VTMOT")
p.add_argument("--sample_step", type=int, default=5)
p.add_argument("--out", default="output/pr1/fusion_inputs")
a = p.parse_args()
man = json.loads(pathlib.Path("runs/00_contract/eval_manifest.json").read_text())
for ds in [d for d in a.datasets.split(",") if d]:
    root = pathlib.Path("eval_data") / ds
    out = pathlib.Path(a.out) / ds
    for sub in ("ir", "vis"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for seq in man["sequence_sets"][SETS[ds]]["sequences"]:
        irs = sorted(q for q in (root / "ir").iterdir()
                     if q.suffix.lower() in EXT and q.name.startswith(seq + "_"))
        vis = {q.name: q for q in (root / "vis").iterdir() if q.suffix.lower() in EXT}
        for q in irs[::a.sample_step]:
            w = vis.get(q.name)
            if w is None:
                continue
            for src, sub in ((q, "ir"), (w, "vis")):
                t = out / sub / q.name
                if not t.exists():
                    os.symlink(os.path.abspath(src), t)
            n += 1
    print(f"  {ds}: {n} pairs -> {out}")
