#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 PYTHON BASE_MODEL ADAPTER PROMPTS PERSONWORK_ZIP RCAEVAL_ZIP OUTPUT_ROOT" >&2
  exit 2
fi

PYTHON="$1"
BASE_MODEL="$2"
ADAPTER="$3"
PROMPTS="$4"
PERSONWORK_ZIP="$5"
RCAEVAL_ZIP="$6"
OUTPUT_ROOT="$7"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MAX_PER_DATASET="${MAX_PER_DATASET:-100}"
MAX_NEW_TOKENS_STRUCTURE="${MAX_NEW_TOKENS_STRUCTURE:-256}"
MAX_NEW_TOKENS_ANSWER="${MAX_NEW_TOKENS_ANSWER:-192}"

mkdir -p "$OUTPUT_ROOT"
for mode in direct lexical_rag full random proxy_oracle no_structuring no_decomposition; do
  output="$OUTPUT_ROOT/$mode"
  "$PYTHON" "$SCRIPT_DIR/src/evaluate_external_five_stage_datasets.py" \
    --personwork-zip "$PERSONWORK_ZIP" --rcaeval-zip "$RCAEVAL_ZIP" \
    --base-model "$BASE_MODEL" --router-adapter "$ADAPTER" --prompts-dir "$PROMPTS" \
    --output-dir "$output" --max-samples "$MAX_PER_DATASET" --mode "$mode" \
    --max-new-tokens-structure "$MAX_NEW_TOKENS_STRUCTURE" \
    --max-new-tokens-answer "$MAX_NEW_TOKENS_ANSWER"
done

for structure in chunk graph table algorithm catalogue; do
  output="$OUTPUT_ROOT/fixed_$structure"
  "$PYTHON" "$SCRIPT_DIR/src/evaluate_external_five_stage_datasets.py" \
    --personwork-zip "$PERSONWORK_ZIP" --rcaeval-zip "$RCAEVAL_ZIP" \
    --base-model "$BASE_MODEL" --router-adapter "$ADAPTER" --prompts-dir "$PROMPTS" \
    --output-dir "$output" --max-samples "$MAX_PER_DATASET" --mode fixed \
    --fixed-structure "$structure" --max-new-tokens-structure "$MAX_NEW_TOKENS_STRUCTURE" \
    --max-new-tokens-answer "$MAX_NEW_TOKENS_ANSWER"
done

# human_oracle is intentionally omitted until both annotation CSVs contain
# human labels; empty templates must never be treated as a gold standard.
