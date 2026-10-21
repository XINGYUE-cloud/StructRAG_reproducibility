# Evaluation Matrix (2026-10-21)

This is the evaluation contract for the expanded experiment. The router's only
output is one structure label: `chunk`, `graph`, `table`, `algorithm`, or
`catalogue`. It receives the question and a document overview. It does not
extract document facts or generate the final answer.

## Data and Preference Construction

The final external tests are exactly the two supplied archives:

- `VALUE-KAO-PersonWork-EN-v2-Experiment-1k.zip`: 1,000 test questions.
- `RCAEval-RE2-Microservice-v1.zip`: 1,538 questions in the official `test`
  split; `fit` and `validation` rows are excluded from test scoring.

The generated corpus is used only as a source for training-task synthesis; its
test split and the two external test archives are not used to create training
preferences.

Training sources are drawn from the training split of the generated document
corpus. Existing task labels and DPO negatives are ignored. The LLM workflow
must, for each source document:

1. Synthesize an answerable knowledge-intensive question and a short answer.
2. Provide a verbatim evidence quotation that can be checked against the source.
3. Write one candidate solving plan for each of the five structures.
4. Rank all five plans by evidence fit, completeness, and complexity.
5. Produce the DPO pair `best structure > second-best structure`.

`src/build_llm_five_type_preferences.py` implements this workflow with resumable
JSONL output and a source-unit-disjoint train/validation/test split. Gold answers
and evidence are retained only for task-quality and downstream evaluation; the
router prompt contains only the synthesized question and document overview.

## Evaluation Matrix

| Aspect | Measure | Data / status |
| --- | --- | --- |
| Task adaptation | Counts and scores by comparison, relation, planning, summary, and single-hop QA | Report the source task type separately from any structure label. |
| Router | Accuracy, macro-F1, confusion matrix, top-2 accuracy, and tie-aware accuracy | Require blinded human preferred/second-choice annotations. `src/export_router_annotation_template.py` exports stratified sheets; blank sheets are not gold labels. |
| Structured extraction | Evidence recall, fact precision, schema/format validity, and manual audit | Compare generated structures with source evidence, stratified by selected structure. |
| Final answer | Exact match, token F1, semantic score, and evidence citation correctness | Report semantic judge and deterministic metrics separately. |
| End-to-end | Direct long-context, lexical RAG, fixed structure, full StructRAG, and oracle-router StructRAG | Paired questions, same model and decoding settings. |
| Ablation | Fixed route, random route, no structuring, and no decomposition | Change one component at a time; hold the sample IDs fixed. |
| Robustness | Score by document length/count and evidence-hop buckets | Report sample counts for each bucket. |
| Cost and speed | Stage latency, input/output tokens, total latency, and API cost | Record route, structuring, extraction, and answer stages separately. |

## Annotation Protocol

The annotation sheets show no model prediction. For each row, annotators record
the best structure, second choice, any acceptable tie, and a short rationale.
Use the following task question: *which representation makes the required
evidence easiest to preserve and retrieve for this question?* Do not label by
the answer's wording alone. Adjudicate disagreements before computing the
primary router metrics. Multi-label/tied cases are retained for tie-aware
scoring and reported separately from strict single-label accuracy.

The external test bundles do not provide StructRAG structure labels. Any
task-to-structure mapping is a proxy and must never be presented as human gold.
The RCAEval `root_cause` and `fault_type` task names alone do not establish a
unique best representation.

## Existing Results That Must Stay Separate

- The existing 1,600-row, five-label synthetic test reports 100% accuracy, but
  its labels were assigned by the deterministic generator. It is a controlled
  smoke result, not the primary router result for this matrix.
- The 3,200-row complex synthetic test also reports 100% under generator-derived
  labels. It is retained as a data-pipeline diagnostic only.
- The existing Loong run completed 1,435 rows through an end-to-end pipeline
  and recorded answer/leaf-recall metrics. Its route counts cover only
  `chunk`, `graph`, and `table`; it is not a five-class router evaluation.
- Earlier external full-pipeline attempts are exploratory and are not used as
  results for this matrix.

## Execution Record

The primary inference hardware is server GPU 3 only; other server GPUs may be
occupied by unrelated jobs. All new server files must be written below a new
experiment directory. Existing server files are read-only. When available disk
space under `/home` is insufficient, use the new experiment directory under
`/data` and do not clean up or overwrite old data.
