# Official Loong Evaluation Record (2026-10-15)

## Status

This record separates completed local checks from work that requires a free GPU
and a user-supplied evaluator key. No existing server directory is modified or
deleted. New remote runs must use a fresh timestamped output directory.

## Verified locally

- Python test suite: 15 tests passed.
- Complex v3 corpus: 16,000 train, 2,000 validation, 3,200 test rows.
- Five labels are present: `chunk`, `table`, `graph`, `algorithm`, `catalogue`.
- Split source roots are disjoint: 18 train roots, 6 validation roots, 6 test roots.
- All 21,200 rows have unique IDs and grounded evidence in their source documents.
- Official Loong rows are not present in the synthetic corpus.

The completed v3 router run used Qwen2.5-1.5B with one epoch of LoRA-DPO
(batch size 8, gradient accumulation 2, learning rate 1e-5). Its validation
DPO loss was 0.00130675. On the held-out 3,200-row v3 test split, Accuracy and
Macro-F1 were both 1.0000; the confusion matrix was diagonal for all five
labels. This is a routed structure classification result, not an end-to-end
answer accuracy result.

The corpus is generated educational data. It is not the official Loong benchmark,
and its difficulty controls are experimental controls rather than official Loong
difficulty annotations.

## Official data protocol

The official track uses only the public Loong test records and the released
three-label router (`chunk`, `table`, `graph`). The Qwen2.5-1.5B model is a
resource-constrained substitution for the paper's larger Qwen models. The
pipeline order remains:

```text
Router -> Structurizer -> Decomposer -> Precise extractor -> Answer merger
```

Preparation exclusions (for example, no complete document fitting the selected
context budget) are recorded separately from model errors. No official test row
is used for training or validation.

## Commands

Run the resumable official pipeline on one available GPU, using a new output
directory each time:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run_official_loong_structrag_qwen_resume \
  --dataset data/official_loong_processed/loong_qwen18k_whole_docs.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --router-adapter outputs/router_dpo_full \
  --prompts-dir prompts \
  --output-dir outputs/official_loong_20261015_<timestamp> \
  --context-limit 32768
```

Aggregate structural diagnostics after the run:

```bash
python -m src.evaluate_official_structure_metrics \
  --predictions outputs/official_loong_20261015_<timestamp>/predictions.jsonl \
  --output-dir outputs/official_metrics_20261015_<timestamp>
```

## Optional semantic evaluation

The local template `.env.judge.local` keeps the requested model and provider:

```env
OPENAI_MODEL=gpt-5.6terra
OPENAI_BASE_URL=https://api.vip1129.cc/v1
OPENAI_API_KEY=
JUDGE_API_STYLE=chat_completions
```

Fill `OPENAI_API_KEY` only on the machine that runs the evaluation. Do not send
the key in chat, commit it, or upload `.env.judge.local`. After filling it:

```bash
python -m src.evaluate_loong_llm_judge \
  --predictions outputs/official_loong_20261015_<timestamp>/predictions.jsonl \
  --output-dir outputs/llm_judge_20261015_<timestamp> \
  --config .env.judge.local
```

The evaluator caches by question, gold answer, prediction, model, and base URL.
It validates a bounded score and writes `judgments.jsonl`, `metrics.json`, and
`errors.jsonl`. Until a valid key is supplied, no live semantic score is
reported. A dry run only exercises parsing fixtures and cannot stand in for the
paper's external LLM evaluation.

## Comparison boundary

The paper reports Loong `LLM score=60.38`, `EM=0.23`, and router
`Accuracy=94.38%`, `Macro-F1=93.42%`. Those values use different model sizes,
data coverage, and evaluator conditions. This repository must report its
Qwen2.5-1.5B diagnostics and any `gpt-5.6terra` results as a separate track;
they must not be described as a numerical reproduction of the paper.
