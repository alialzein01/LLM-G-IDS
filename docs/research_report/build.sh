#!/usr/bin/env bash
# Compile the research report. First run downloads packages (~2 min); later runs ~1 s.
#   ./build.sh          compile once
#   ./build.sh watch    recompile on every save
set -euo pipefail
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH

if [ "${1:-}" = "watch" ]; then
    exec tectonic -X watch
fi

# Regenerate the table fragments from the contracts, so a stale fragment cannot
# survive a build.
PYGEN=../../.venv/bin/python
[ -x "$PYGEN" ] || PYGEN=$(command -v python3)
if [ -n "$PYGEN" ]; then
    echo "--- tables ---"
    OMP_NUM_THREADS=1 "$PYGEN" tables/make_tables.py
fi

# Figures. The six data plots come from one script; the three diagrams are
# standalone TikZ documents. Pass "figures" to redraw them, which needs tectonic
# and takes a few seconds; otherwise the committed PDFs are used as they are.
if [ "${1:-}" = "figures" ]; then
    echo "--- figures ---"
    OMP_NUM_THREADS=1 "$PYGEN" figures/src/make_plots.py
    for f in fig_architecture fig_loop_flow fig_ceiling; do
        (cd figures/src && tectonic -c minimal "$f.tex" --outdir .. >/dev/null)
        echo "  $f.pdf"
    done
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
cp build/main.pdf build/finalReport.pdf
echo "-> $(pwd)/build/finalReport.pdf"
