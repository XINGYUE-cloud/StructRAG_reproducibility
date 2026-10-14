# StructRAG Reproducibility Study

This repository records a resource-constrained reproduction study of
[StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-Time
Hybrid Information Structurization](https://arxiv.org/abs/2410.08815).

The main experiment uses the authors' public Loong source set and preserves
the released inference order:

```text
Router -> Structurizer -> Decomposer -> Precise extractor -> Answer merger
```

It runs locally with `Qwen2.5-1.5B-Instruct` on one GPU. This is a protocol
aligned engineering reproduction, not a numerical reproduction of the paper:
the paper uses Qwen2-7B/Qwen2-72B-scale models, five structure classes, and an
LLM/GPT-4 evaluation protocol.

## What Is Included

- `src/prepare_official_loong_structrag.py`: restores the public Loong
  documents into the title-marker format expected by the released StructRAG
  code. It includes whole documents only and records omitted documents.
- `src/run_official_loong_structrag_qwen.py`: single-GPU implementation of
  the released five-stage order for the three public router labels.
- `src/run_official_loong_structrag_qwen_resume.py`: checkpointed runner that
  appends each route and answer to JSONL, so an interrupted SSH session can be
  resumed without repeating completed examples.
- `src/train_official_dpo_router_v3.py` and related adapters: Qwen2.5-1.5B
  LoRA-DPO training/evaluation on the released router preference pairs.
- `src/run_paper_aligned_five_type_structrag_v1.py` and the synthetic data
  scripts: a separately labelled five-type extension for studying the missing
  `algorithm` and `catalogue` classes.
- `prompts/`: public StructRAG prompt templates used by the runners.
- `docs/`: paper notes, protocol, limitations, and experiment records.

## Public Data

Download the official Loong files from the upstream projects rather than
committing them:

- Questions and answers:
  `https://raw.githubusercontent.com/MozerWang/Loong/main/data/loong.jsonl`
- Documents: `http://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip`
- StructRAG source and router data:
  `https://github.com/Li-Z-Q/StructRAG`

The verified `loong.jsonl` used in this study has 1,600 records and SHA-256:

```text
CD640D3CA0E41D3638153506E3CAEE109850B30DE0A4E829862232A2B6302DA8
```

The public router release contains 1,444 training pairs and 362 test pairs,
but only `chunk`, `table`, and `graph`. The paper describes five classes;
`algorithm` and `catalogue` are therefore not silently treated as available
official labels.

## Official Loong Run

Use Python 3.10+, PyTorch with CUDA, `transformers`, `peft`, `tiktoken`, and
the packages in `requirements.txt`. Set paths to local copies of the model,
router adapter, prompts, and prepared data. A clean preparation command is:

```bash
python -m src.prepare_official_loong_structrag \
  --input data/official_loong_source/official_loong.jsonl \
  --doc-root data/official_loong_source/doc \
  --tokenizer models/Qwen2.5-1.5B-Instruct \
  --output data/official_loong_processed/loong_qwen18k_whole_docs.jsonl \
  --report data/official_loong_processed/preparation_report.json \
  --max-input-tokens 18000
```

For a resumable run on one selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run_official_loong_structrag_qwen_resume \
  --dataset data/official_loong_processed/loong_qwen18k_whole_docs.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --router-adapter outputs/router_dpo_full \
  --prompts-dir prompts \
  --output-dir outputs/official_loong_checkpoint \
  --context-limit 32768
```

The server run for this study uses the same command with `CUDA_VISIBLE_DEVICES=3`.
It writes `routes.jsonl`, `predictions.jsonl`, `errors.jsonl`,
`progress.json`, and `metrics.json` incrementally. The runner never silently
truncates a document. Samples with no whole document fitting the input budget
are counted as preparation/route exclusions.

The paper's official Loong score cannot be reproduced without the evaluator
API used by `Loong/src/step3_model_evaluate.py`. This repository reports only
auditable auxiliary metrics (`exact_canonical_match`, `gold_leaf_recall`,
route counts, completion counts) unless a compatible evaluator is configured.

## Results So Far

These tracks must be read separately:

| Track | Data/classes | Result |
| --- | --- | --- |
| Released router | 1,444/362 public pairs, 3 classes | Accuracy 98.34%; Macro-F1 98.33% |
| Synthetic five-type router | Generated, unit-disjoint, 5 classes | Accuracy 100%; Macro-F1 100% (template-controlled task) |
| Synthetic five-stage pipeline | 10 held-out examples | Exact match 0%; answer-contains 0%; keyword recall 5% |
| Official Loong Qwen run | 1,600 official records, local 1.5B model | 1,435 completed; canonical exact match 0.00%; mean gold-leaf recall 4.10% |

The paper reports overall Loong LLM/EM scores of `60.38/0.23` and a router
accuracy/Macro-F1 of `94.38%/93.42%`. Those numbers are included for context
only; model, labels, data coverage, and evaluator differ here.

## Known Limitations

1. The released router data exposes three labels, not the paper's five.
2. Qwen2.5-1.5B is much smaller than the paper's main models.
3. The synthetic five-class judge is deterministic provenance logic, not the
   authors' unreleased LLM preference-judging pipeline.
4. Long Loong documents exceed the one-GPU context budget. The preparation
   script keeps complete documents and records omissions instead of truncating.
5. The public graph construction prompt is hard-coded around paper-reference
   triples, so it is not equally natural for financial and legal examples.
6. EM and leaf recall are diagnostic metrics, not the paper's GPT-4 semantic
   score.

Detailed Chinese notes are in
[`docs/PAPER_STUDY_AND_REPRODUCTION_20261014.md`](docs/PAPER_STUDY_AND_REPRODUCTION_20261014.md),
the consolidated report is in
[`docs/REPRODUCTION_REPORT_20261014.md`](docs/REPRODUCTION_REPORT_20261014.md),
and the issue log is in
[`docs/REPRODUCTION_ISSUES_20261014.md`](docs/REPRODUCTION_ISSUES_20261014.md).

The official Loong run separates four counts: source rows, preparation exclusions
(no complete document fits the input budget), route completions, and completed
end-to-end predictions. The final run had 1,600 source rows, 165 preparation
exclusions, 1,435 route completions, and 1,435 end-to-end completions. An
exclusion is not a model error. Per-type metrics and route counts are recorded
in [`docs/RESULTS_TEMPLATE_20261014.md`](docs/RESULTS_TEMPLATE_20261014.md).

## Attribution

- Paper and implementation: [Li-Z-Q/StructRAG](https://github.com/Li-Z-Q/StructRAG)
- Loong benchmark: [MozerWang/Loong](https://github.com/MozerWang/Loong)
- This repository is an educational reproduction record; it does not claim
  the original paper's numerical result.
