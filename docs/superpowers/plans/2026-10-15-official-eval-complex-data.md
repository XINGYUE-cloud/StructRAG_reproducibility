# Official Evaluation and Complex Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the official Loong evaluation path, add auditable semantic judging, and generate a larger, harder five-type training corpus without data leakage.

**Architecture:** Keep the existing resumable runner and add optional structure/metric hooks rather than replacing its checkpoint format. Put synthetic generation in a deterministic script with unit-disjoint splits. Put semantic judging in a separate cache-first command so official model inference remains usable without an API key.

**Tech Stack:** Python 3.11, JSONL, PyTorch/Transformers for server inference, OpenAI-compatible HTTP API for optional judging, `unittest` for dependency-light local tests.

**Spec:** `docs/superpowers/specs/2026-10-14-official-eval-and-complex-data-design.md`

## Global Constraints

- Never delete or overwrite existing server content; every remote output uses a new timestamped directory.
- Official Loong rows are test-only and never enter synthetic training or validation files.
- All generated splits are `unit_id`-disjoint and record difficulty controls.
- No API keys, model weights, private responses, or server passwords enter git.
- Reports distinguish official data, synthetic data, diagnostic string metrics, and semantic LLM metrics.

### Task 1: Build Complex Five-Type Dataset v3

**Files:**
- Create: `scripts/build_complex_five_type_data_v3.py`
- Create: `tests/test_complex_five_type_data_v3.py`

**Interfaces:**
- CLI accepts `--profiles`, `--out-dir`, `--train-count`, `--validation-count`, `--test-count`, `--seed`.
- Output rows retain `id`, `unit_id`, `label`, `difficulty_level`, `difficulty_controls`, `query`, `answer`, `answer_keywords`, and `documents`.
- Defaults are train 16,000, validation 2,000, test 3,200.

- [ ] Write tests for split disjointness, unique IDs, all five labels, and monotonic difficulty controls using a tiny fixture.
- [ ] Run the tests and observe the expected import failure.
- [ ] Implement generation by extending the existing profile/document helpers with multi-hop, multi-condition, distractor, length, and hierarchy controls.
- [ ] Run the tests and a tiny CLI generation.
- [ ] Record metadata proving the split counts and unit isolation.

### Task 2: Official Structure Metrics and Runner Metadata

**Files:**
- Create: `src/evaluate_official_structure_metrics.py`
- Modify: `src/run_official_loong_structrag_qwen_resume.py`
- Create: `tests/test_official_structure_metrics.py`

**Interfaces:**
- `evaluate_official_structure_metrics.py --predictions ... --output-dir ...` writes overall and grouped JSON metrics.
- Group keys are selected structure, official document type, level, and coverage status.
- Runner retains existing checkpoint files and adds no destructive behavior.

- [ ] Write tests for grouping, zero-row groups, and preparation exclusions versus model errors.
- [ ] Run tests to verify the new module is missing.
- [ ] Implement pure JSONL aggregation and optional runner metadata needed for grouping.
- [ ] Run tests and aggregate the existing local sanitized sample.

### Task 3: Cache-First LLM Semantic Judge

**Files:**
- Create: `src/evaluate_loong_llm_judge.py`
- Create: `tests/test_loong_llm_judge.py`
- Modify: `requirements.txt` only if an existing dependency is insufficient

**Interfaces:**
- CLI reads prediction JSONL and writes `judgments.jsonl`, `metrics.json`, and `errors.jsonl`.
- `--dry-run` validates prompt/response parsing and never opens a network connection.
- Network mode reads `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `OPENAI_MODEL` from the environment.
- One judgment is cached by prediction ID and model; reruns never resubmit cached IDs.

- [ ] Write tests for valid JSON parsing, malformed judge responses, cache reuse, and dry-run no-network behavior.
- [ ] Run tests to verify the new module is missing.
- [ ] Implement bounded score parsing, retries for transient failures, and aggregate means by overall/type/level/structure.
- [ ] Run tests and dry-run against a fixture.

### Task 4: Documentation and Reproducible Commands

**Files:**
- Create: `docs/OFFICIAL_FULL_EVALUATION_20261015.md`
- Modify: `README.md`
- Modify: `docs/REPRODUCTION_REPORT_20261014.md`

- [ ] Document local smoke commands and new-server commands with timestamped output paths.
- [ ] Document API setup without embedding secrets.
- [ ] Document model/data/evaluator differences from the paper and separate synthetic results from official results.
- [ ] Run markdown/link and Python syntax checks.

### Task 5: Server Execution and Evidence Collection

**Files:**
- Create remotely: new timestamped output/data directories only
- Create locally: `docs/results/official_full_20261015.json`

- [ ] Confirm server GPU availability and existing process ownership without killing or changing processes.
- [ ] Generate v3 data in a new remote directory and verify counts/checksums.
- [ ] Run official Loong full test with checkpoint resume in a new directory.
- [ ] Run structure metric aggregation; run semantic judge only after API variables are configured.
- [ ] Download sanitized metrics and write the comparison report.

