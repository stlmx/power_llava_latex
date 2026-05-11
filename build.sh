#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="${ROOT_DIR}/TII-Articles-LaTeX-template"

cd "${MAIN_DIR}"
latexmk -pdf -interaction=nonstopmode -halt-on-error tii-articles-template.tex
