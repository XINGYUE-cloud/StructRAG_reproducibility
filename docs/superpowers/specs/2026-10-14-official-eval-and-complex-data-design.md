# Official Evaluation and Complex Data Design

## Goal

Run the full official Loong test set for the StructRAG pipeline, add a
reproducible LLM semantic evaluator, report metrics by structure, and create a
larger, harder five-type training set without leaking official test data.

## Scope

The work has three isolated tracks:

1. An official Loong evaluation track that preserves the existing three-class
   baseline and adds an optional five-class route/construct path.
2. An OpenAI-compatible semantic judge that reads completed prediction JSONL,
   caches one judgment per example, and writes aggregate reports.
3. A deterministic v3 synthetic generator for five training structures with
   larger scale and explicit complexity controls.

The official Loong files are test-only. Synthetic profiles and generated rows
are the only data permitted for training. Existing server directories are
read-only inputs; every new output uses a new directory.

## Interfaces and artifacts

- `src/run_official_loong_structrag_qwen_resume.py` keeps its current command
  line compatible and gains `--structure-set {released,all-five}`. `released`
  accepts `chunk/table/graph`; `all-five` accepts all five labels and dispatches
  the matching public or repository prompt.
- `src/evaluate_loong_llm_judge.py` consumes an official predictions JSONL and
  writes `judgments.jsonl`, `metrics.json`, and `errors.jsonl`. It requires
  `OPENAI_API_KEY`; `OPENAI_BASE_URL` and `OPENAI_MODEL` are optional.
- `scripts/build_complex_five_type_data_v3.py` consumes generated profile JSONL
  and writes unit-disjoint `train.jsonl`, `validation.jsonl`, and `test.jsonl`,
  plus `metadata.json`. Defaults are 16,000/2,000/3,200 rows.
- `src/evaluate_official_structure_metrics.py` aggregates official predictions
  by selected structure, document type, difficulty level, and coverage status.
- `docs/OFFICIAL_FULL_EVALUATION_*.md` records commands, model substitutions,
  API evaluator settings, exclusions, and the comparison with the paper.

## Data and leakage policy

- A `unit_id` is assigned to exactly one synthetic split.
- Official Loong rows, documents, answers, and derived labels never enter
  synthetic training or validation files.
- The v3 generator records the number of hops, conditions, distractors,
  document count, approximate token/character length, and hierarchy depth in
  every row.
- The two labels absent from released router data (`algorithm` and
  `catalogue`) remain explicitly marked as synthetic extensions.

## Evaluation contract

The official runner must finish every row that fits the configured whole-document
budget and record preparation exclusions separately from model failures. String
metrics remain diagnostic. The semantic judge sends only the question, gold
answer, and prediction to the configured evaluator, requests a bounded numeric
score plus a short reason, validates the response, and retries transient errors.
No API key or private response is committed.

Every report must distinguish:

- official Loong results versus synthetic results;
- released three-class baseline versus synthetic five-class router;
- exact/leaf-recall diagnostics versus the paper's LLM score;
- completed rows versus preparation exclusions.

## Acceptance criteria

- A dry-run test proves v3 split disjointness, row uniqueness, and difficulty
  controls without loading a model.
- An official-run smoke test processes at least one row in each supported
  structure mode and refuses to overwrite an existing output directory.
- The semantic evaluator dry-run validates response parsing and cache reuse
  without making a network call.
- Full server runs use a new timestamped directory, retain old outputs, and
  produce per-structure and overall metrics.
- Reports include the exact model, data, evaluator, context budget, exclusions,
  and any unavailable paper-comparison fields.
