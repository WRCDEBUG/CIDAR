#!/bin/bash
# Blur fix, designed to answer the three questions a reviewer will ask.
#
# Q1 "how do you know the estimator cannot see blur?"  -> tools/identifiability_probe.py
#    measures it: a linear probe on the descriptor must name the degradation, and
#    the stem-only descriptor is at chance for blur while the spectral statistics
#    are linearly separable by construction.
# Q2 "is the fix a special case or a principle?"       -> the same rule (recovery is
#    bounded by identifiability) predicted the cross-modal fix for low-light; this
#    is its second application, and the spectral-only run is the ablation that
#    isolates the cue.
# Q3 "is the oracle's 43.4% an artefact of feeding a label?"  -> if the fix is real,
#    the ESTIMATED descriptor should close most of the gap to the oracle.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
L=output/pr1/stage2_logs
D4="noise10,noise20,dark,blur"      # blur must be in the training set to be recovered
echo "### $(date '+%F %T') BLURFIX START" >> "$L/queue_progress.log"
run() { name="$1"; shift
  echo "### $(date '+%F %T') START $name" >> "$L/queue_progress.log"
  timeout 28800 $PY tools/stage2_dar.py --tag "$name" "$@" > "$L/${name}.log" 2>&1
  echo "### $(date '+%F %T') END   $name exit=$?" >> "$L/queue_progress.log"; }

run crossspec_long   --dar_mode predicted --est_mode crossspec --train_degradations $D4 --steps 8000 --eval_every 4000
run crossspec_s1     --dar_mode predicted --est_mode crossspec --train_degradations $D4 --steps 8000 --eval_every 0 --seed 11
echo "### $(date '+%F %T') BLURFIX DONE" >> "$L/queue_progress.log"
