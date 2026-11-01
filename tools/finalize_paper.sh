#!/bin/bash
# One-shot finalisation: regenerate every table and figure from the recorded run
# artifacts, then rebuild the PDF.  Run this after any new experiment so the
# paper can never drift from the measurements.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"
echo "== tables from output/pr1/stage2_*.json =="
$PY tools/make_paper_tables.py
echo "== figures =="
$PY tools/make_paper_figures.py
echo "== regenerate the LaTeX table includes =="
$PY tools/make_latex_includes.py
# the IEEEtran submission version; IEEEtran.cls is vendored next to it because
# the local TeX Live does not ship the class
cd runs/00_contract || exit 1
if [ -f paper_ieee.tex ]; then
  echo "== compile (IEEEtran) =="
  pdflatex -interaction=nonstopmode paper_ieee.tex > /dev/null 2>&1
  pdflatex -interaction=nonstopmode paper_ieee.tex > /dev/null 2>&1
  echo -n "IEEE pages:      "; pdfinfo paper_ieee.pdf 2>/dev/null | awk '/Pages/{print $2}'
  echo "IEEE pdf:        runs/00_contract/paper_ieee.pdf"
fi
cd ../.. || exit 1

echo "== compile =="
cd runs/00_contract
pdflatex -interaction=nonstopmode -halt-on-error paper_full_draft.tex > /tmp/final_tex1.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error paper_full_draft.tex > /tmp/final_tex2.log 2>&1
pdfinfo paper_full_draft.pdf | grep -E "Pages"
echo "done: runs/00_contract/paper_full_draft.pdf"
