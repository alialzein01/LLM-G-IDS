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

# Regenerate the table fragments from the contracts, so a stale fragment cannot
# survive a build. Figures are drawn externally and are not generated here.
PYGEN=../../.venv/bin/python
[ -x "$PYGEN" ] || PYGEN=$(command -v python3)
if [ -n "$PYGEN" ]; then
    echo "--- tables ---"
    OMP_NUM_THREADS=1 "$PYGEN" tables/make_tables.py
fi

# Number and vocabulary audit. A warning step: it reports, it does not block the
# compile. Run scripts/check_report_numbers.py directly for the exit code.
PYTHON=../../.venv/bin/python
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
if [ -n "$PYTHON" ]; then
    echo "--- number audit ---"
    OMP_NUM_THREADS=1 "$PYTHON" ../../scripts/check_report_numbers.py --quiet || true
    echo "--------------------"
fi

tectonic main.tex --outdir build
echo "-> $(pwd)/build/main.pdf"
