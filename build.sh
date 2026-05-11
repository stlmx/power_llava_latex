#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="${LATEX_DOCKER_BUILD:-/mnt/ssd_data/limingxuan/service/scripts/latex-docker-build.sh}"

"${BUILD_SCRIPT}" \
  --engine pdflatex \
  "${ROOT_DIR}" \
  TII-Articles-LaTeX-template/tii-articles-template.tex
