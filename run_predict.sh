#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./run_predict.sh [MODEL_DIR] [TEST_CSV] [BASE_MODEL] [OUTPUT_CSV]
# Example:
# ./run_predict.sh ./checkpoint-2414 ./data/test.csv bert-large-cased-whole-word-masking ./submission.csv

MODEL_DIR=${1:-./checkpoint-2414}
TEST_CSV=${2:-./data/test.csv}
BASE_MODEL=${3:-bert-large-cased-whole-word-masking}
OUTPUT_CSV=${4:-./submission.csv}

# Ensure python environment is activated (user should set up venv or conda).
# Optionally, uncomment next line if using a virtualenv located at .venv
# source .venv/Scripts/activate

echo "Running prediction with model_dir=${MODEL_DIR}, test_csv=${TEST_CSV}, base_model=${BASE_MODEL}, output_csv=${OUTPUT_CSV}"

python code/predict_for_report.py --model-dir "${MODEL_DIR}" --test-csv "${TEST_CSV}" --base-model "${BASE_MODEL}" --output-csv "${OUTPUT_CSV}"

echo "Prediction finished. Output written to ${OUTPUT_CSV}"