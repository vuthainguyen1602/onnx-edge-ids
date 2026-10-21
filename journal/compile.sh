#!/usr/bin/env bash
set -euo pipefail

echo "[*] Compiling Elsevier CAS Journal Manuscript..."
xelatex -halt-on-error main.tex
if [ -f references.bib ]; then
    bibtex main || true
fi
xelatex -halt-on-error main.tex
xelatex -halt-on-error main.tex

echo "[+] Compilation successful! Output generated at main.pdf"
