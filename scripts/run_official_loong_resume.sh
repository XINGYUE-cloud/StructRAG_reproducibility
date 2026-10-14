#!/usr/bin/env bash
set -euo pipefail

# Set these paths for the machine that runs the experiment; no machine-local
# paths or credentials are embedded in this script.
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m src.run_official_loong_structrag_qwen_resume "$@"
