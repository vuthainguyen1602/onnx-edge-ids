#!/usr/bin/env bash
set -euo pipefail

xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
