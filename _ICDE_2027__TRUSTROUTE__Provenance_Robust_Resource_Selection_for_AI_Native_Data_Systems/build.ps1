$ErrorActionPreference = "Stop"

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -pdf -interaction=nonstopmode main.tex
}
else {
    if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
        throw "pdflatex was not found on PATH. Install TeX Live or MiKTeX with elsarticle support."
    }

    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
}

Write-Host "Built main.pdf"
