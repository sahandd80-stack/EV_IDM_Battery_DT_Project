#!/usr/bin/env bash
# run_all.sh -- one-shot reproducibility script for the EV IDM Battery DT project.
set -e
cd "$(dirname "$0")"

# Make our local modules importable from any CWD
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "============================================================"
echo "EV IDM Battery Digital Twin -- full pipeline"
echo "============================================================"

# Step 1: preprocessing
echo
echo "[1/9] Preprocessing ..."
python -m preprocessing.preprocess \
    --raw-dir data/raw \
    --output   data/processed_segments.csv \
    --manifest data/segment_manifest.json

# Step 2: tests
echo
echo "[2/9] Running IDM unit tests ..."
python tests/test_idm.py

# Step 3: calibration
echo
echo "[3/9] IDM calibration ..."
python -m idm.calibration \
    --input  data/processed_segments.csv \
    --output results/calibration_results.json

# Step 4: multi-objective optimisation
echo
echo "[4/9] Multi-objective Pareto scan ..."
python -m optimization.multi_objective_optimization \
    --calibration results/calibration_results.json \
    --output     results/pareto_front.json

# Step 5: closed-loop experiments
echo
echo "[5/9] Closed-loop driver-profile experiments ..."
python -m experiments.closed_loop \
    --calibration results/calibration_results.json \
    --output     results/closed_loop_results.json

# Step 6: sensitivity analysis
echo
echo "[6/9] OAT sensitivity analysis ..."
python -m analysis.sensitivity \
    --baseline results/pareto_front.json \
    --output   results/sensitivity_analysis.json

# Step 7: external validation (VED subset)
echo
echo "[7/9] External VED validation ..."
python -m experiments.ved_validation \
    --ved-dir data/ved_subset \
    --output  results/ved_validation.json || \
    echo "[7/9] VED validation skipped (module optional)"

# Step 8: figures
echo
echo "[8/9] Generating figures ..."
python -m figures.make_figures

# Step 9: manuscript
echo
echo "[9/9] Compiling manuscript ..."
if command -v tectonic >/dev/null 2>&1; then
    tectonic manuscript.tex
elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode manuscript.tex
    bibtex   manuscript
    pdflatex -interaction=nonstopmode manuscript.tex
    pdflatex -interaction=nonstopmode manuscript.tex
else
    echo "[9/9] No LaTeX compiler available; skipping PDF compilation."
fi

echo
echo "Done. Outputs in results/ and figures_output/, manuscript.pdf in repo root."
