#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"

"$PYTHON_BIN" -m src.train_official_dpo_router_v3 \
  --train upstream/train_router/data/train.json \
  --validation upstream/train_router/data/test.json \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/router_dpo \
  --epochs 3 --learning-rate 1e-5 --batch-size 4 --gradient-accumulation 2

"$PYTHON_BIN" -m src.evaluate_official_dpo_router \
  --test upstream/train_router/data/test.json \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --adapter outputs/router_dpo \
  --output-dir outputs/router_eval

"$PYTHON_BIN" scripts/build_lexicon.py --out data/test_world/lexicon.json --pretty
"$PYTHON_BIN" scripts/build_profiles.py \
  --lexicon data/test_world/lexicon.json --out data/test_world/profiles \
  --n-units 3 --families-per-unit 6 --seed 20261012
"$PYTHON_BIN" scripts/build_views.py \
  --profiles data/test_world/profiles/people_profiles.jsonl \
  --relations data/test_world/profiles/relations_basic.jsonl \
  --out data/test_world/kb_views
"$PYTHON_BIN" scripts/build_hard_person_questions.py \
  --profiles data/test_world/profiles/people_profiles.jsonl \
  --out data/test_world/hard_questions --seed 20261012 --per-type-limit 20

"$PYTHON_BIN" -m src.run_single_gpu_structrag \
  --kb-dir data/test_world/kb_views \
  --questions data/test_world/hard_questions/all_hard_questions.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --router-adapter outputs/router_dpo \
  --output-dir outputs/end_to_end_dynamic \
  --max-samples 20 --max-docs 6 --max-subquestions 2 --stratified \
  --structurization-mode dynamic

"$PYTHON_BIN" -m src.run_single_gpu_structrag \
  --kb-dir data/test_world/kb_views \
  --questions data/test_world/hard_questions/all_hard_questions.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --router-adapter outputs/router_dpo \
  --output-dir outputs/end_to_end_provided_control \
  --max-samples 20 --max-docs 6 --max-subquestions 2 --stratified \
  --structurization-mode provided
