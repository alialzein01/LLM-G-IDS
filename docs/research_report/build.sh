#!/usr/bin/env bash
# Compile the research report. First run downloads packages (~2 min); later runs ~1 s.
#   ./build.sh          compile once
#   ./build.sh watch    recompile on every save
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH

if [ "${1:-}" = "watch" ]; then
    exec tectonic -X watch
fi

tectonic main.tex --outdir build
echo "-> $(pwd)/build/main.pdf"
