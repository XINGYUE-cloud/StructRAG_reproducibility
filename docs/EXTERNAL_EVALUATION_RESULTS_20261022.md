# External Evaluation Results (2026-10-22)

## Test Sets and Protocol

The only external test sources in this run were the supplied archives:

| Dataset | Evaluated rows | Split |
| --- | ---: | --- |
| VALUE-KAO-PersonWork-EN-v2-Experiment-1k | 1,000 | archive runtime set |
| RCAEval-RE2-Microservice-v1 | 1,538 | official `test` split |

The router receives the question and a data-derived document overview, and
outputs one of `chunk`, `graph`, `table`, `algorithm`, or `catalogue`. It does
not extract documents or generate the final answer. Preference training used
1,191 accepted, document-grounded LLM preference pairs from two non-overlapping
generation batches. The generator created
five structure-specific plans, ranked them with `gpt-5.6-terra`, and used the
top-versus-second pair for LoRA-DPO. Qwen2.5-1.5B-Instruct was trained for three
epochs on one server GPU; best validation DPO loss was 0.2667 at epoch 1.

## Router Results

The independent preference holdout contains 120 rows. Accuracy was **80.00%**
and Macro-F1 was **74.90%**. This holdout is labeled by the same LLM plan-ranking
process used to create preferences, not by human annotators.

The supplied archives do not contain human-annotated StructRAG structure
targets. The following external scores use task-derived proxy mappings only:

| Dataset | Rows | Proxy Accuracy | Proxy Macro-F1 | Supported proxy labels |
| --- | ---: | ---: | ---: | --- |
| PersonWork | 1,000 | 42.10% | 20.11% | table, graph, algorithm |
| RCAEval test | 1,538 | 74.45% | 73.59% | table, graph |

On PersonWork, the router predicted `graph` for 973/1,000 rows and `catalogue`
for 22; 313 of 314 proxy-`algorithm` rows were predicted as `graph`. On RCAEval,
the proxy score covers only `root_cause` (graph) and `fault_type` (table). These
figures are diagnostic agreement with task-derived mappings, not human-gold
five-class routing accuracy. The archive tasks do not test `chunk` or
`catalogue` targets.

## End-to-End Pilot Matrix

All 12 runnable configurations were evaluated on the same fixed pilot sample:
2 rows from each archive (4 total per configuration). Answer generation was
capped at 32 new tokens. These settings validate the experiment pipeline and
are too small and too truncated for benchmark-level conclusions.

| Configuration | PersonWork EM | RCA fault accuracy | RCA root-cause Hit@1 | Mean latency (s) | Format-valid rate | Gold-fact recall proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 0% | 0% | 0% | 8.04 | n/a | n/a |
| Lexical RAG | 0% | 0% | 0% | 2.54 | n/a | n/a |
| Full StructRAG | 0% | 0% | 0% | 13.41 | 25% | 20% |
| Random route | 0% | 0% | 0% | 12.03 | 50% | 60% |
| Proxy oracle route | 0% | 0% | 0% | 13.25 | 0% | 0% |
| No structuring | 0% | 0% | 0% | 11.26 | 25% | 80% |
| No decomposition | 0% | 0% | 0% | 12.11 | 25% | 20% |
| Fixed chunk | 0% | 0% | 0% | 10.91 | 100% | 80% |
| Fixed graph | 0% | 0% | 0% | 13.45 | 0% | 0% |
| Fixed table | 0% | 0% | 0% | 13.08 | 0% | 0% |
| Fixed algorithm | 0% | 0% | 0% | 13.41 | 25% | 0% |
| Fixed catalogue | 0% | 0% | 0% | 14.23 | 75% | 0% |

The task-derived proxy route reached 100% router agreement by construction; it
is not a learned-model or human-oracle result. No multi-document examples were
present in this four-row pilot, so document-count robustness is not measurable.

## Semantic Judge

`gpt-5.6-terra` scored 48 answers (2 examples per dataset for each of 12
configurations) on a 1-100 scale. The overall mean was **3.79/100**. Per-mode
means were:

| Configuration | Mean score |
| --- | ---: |
| Direct | 4.75 |
| Lexical RAG | 5.00 |
| Full StructRAG | 3.00 |
| Random route | 3.00 |
| Proxy oracle route | 4.00 |
| No structuring | 5.50 |
| No decomposition | 3.50 |
| Fixed chunk | 2.25 |
| Fixed graph | 4.00 |
| Fixed table | 5.50 |
| Fixed algorithm | 2.75 |
| Fixed catalogue | 2.25 |

The low scores align with the 32-token cap and weak 1.5B answer generation.
The pilot does not show an end-to-end gain from the full pipeline. A larger
paired evaluation with adequate answer budgets is required before drawing
quality or cost-effectiveness conclusions.

## Human Annotation and Reproducibility Notes

The blinded annotation CSVs remain empty. Consequently human Accuracy,
Macro-F1, confusion matrix, tie-aware accuracy, and true human-oracle end-to-end
results are pending annotation. Do not interpret proxy scores as those
measurements.

The complete code and metric JSON files are in the experiment workspace/server
output directory and are intentionally not committed as raw per-example output.
The supplied ZIP files, credentials, API key, model weights, and adapter
checkpoints are not included in this repository.
