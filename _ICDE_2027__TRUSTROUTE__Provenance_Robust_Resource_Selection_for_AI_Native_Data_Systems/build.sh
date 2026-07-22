#!/usr/bin/env bash
# Build the Elsevier submission. Requires pdflatex, bibtex, and elsarticle.
set -euo pipefail

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
else
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
fi

echo "Built main.pdf"
