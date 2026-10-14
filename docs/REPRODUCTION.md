# Reproduction Plan

## 1. Define the Reproduction Boundary

This project has two layers:

1. **Available now:** generate the synthetic person dataset, four knowledge-base views, representation-agnostic QA files, and a small local Qwen RAG runner.
2. **Resource-bounded routing baseline:** train a four-class Qwen2.5-1.5B QLoRA router using deterministic task-to-representation proxy labels, then evaluate it on information units unseen during training.
3. **Not reproduced:** the paper's Qwen2-7B five-class DPO router, LLM-generated preference data, Qwen2-72B structurizer/utilizer modules, and the full StructRAG pipeline.

The full StructRAG paper is a broader system: a router chooses among table, graph, algorithm, catalogue, and text chunk structures; an LLM structurizer converts raw documents; then an LLM utilizer decomposes the question, extracts evidence, and answers. This repository starts with a controlled four-view comparison rather than claiming a full replication.

## 2. Fix the Environment

Record the following before every run:

- Operating system and Python version.
- Git commit SHA.
- Random seed, number of units, and families per unit.
- Model name, revision, inference backend, quantization, temperature, and maximum generated tokens.
- Retriever type, embedding model, top-k, context limit, and whether retrieval is restricted to one unit.

## 3. Generate the Dataset

Use the commands in the repository README. For the first smoke test, keep --n-units 1, --families-per-unit 3, and --seed 42. Confirm that these files exist:

~~~
data/profiles/people_profiles.jsonl
data/profiles/relations_basic.jsonl
data/profiles/metadata.json
data/kb_views/kb_text.jsonl
data/kb_views/kb_table.jsonl
data/kb_views/kb_triples.jsonl
data/kb_views/kb_graph.jsonl
data/kb_views/kb_all_views.jsonl
data/questions/all_questions.jsonl
~~~

For the recorded local experiment, use 30 independent units, 6 families per unit, and seed 42. The generated benchmark has 1,838 people, 5,740 basic relations, and 2,921 questions. See the `Local Qwen Reproduction` section in the README for exact commands.

## 4. Build Router Data and Train

Run `src.build_router_dataset` with the generated `all_questions.jsonl`. It splits by `unit_id`, rather than randomly by question, to keep each family/network in exactly one split. With the fixed command in the README, the data is balanced as follows:

| Split | Units | Examples | Labels |
|---|---:|---:|---|
| Train | 18 | 900 | 225 each of chunk, table, triples, graph |
| Validation | 6 | 240 | 60 each |
| Test | 6 | 240 | 60 each |

The labels are a transparent four-view proxy: direct personal attributes map to `chunk`, family-member questions to `triples`, kinship reasoning to `graph`, and attribute filtering to `table`. They are not the paper's LLM-judged DPO preferences.

Use `src.train_router` to train LoRA on `Qwen/Qwen2.5-1.5B-Instruct`. It requires a CUDA GPU. The script uses 4-bit QLoRA only when bitsandbytes has a compatible CUDA binary; otherwise it automatically uses FP16 LoRA and records that actual mode in the metrics. Use `src.evaluate_router` to save exact router predictions, accuracy, macro F1, per-label metrics, and the confusion matrix. Finally, `src.compare_with_paper` writes the qualified comparison table.

## 5. Run the RAG Baseline

`src.run_rag` is a bounded, deterministic answer-generation baseline. It selects one supplied view, ranks only records from the question's unit using names and lexical overlap, sends the top contexts to local Qwen, and saves predictions. It is not the paper's full StructRAG retrieval/utilization pipeline.

Example:

~~~powershell
.\.venv\Scripts\python.exe -m src.run_rag --kb-dir data\benchmark\kb_views --questions data\benchmark\questions\all_questions.jsonl --representation graph --output-dir results\rag --max-samples 40 --top-k 3 --stratified
~~~

Run all four representations with identical questions, seed, model, top-k, prompt, and decoding settings. Change only `--representation`.

## 6. Report Results

Report metrics by task type, not only an aggregate score:

- person_attribute
- family_member
- derived_kinship
- multi_hop_kinship
- attribute_filter

For every aggregate result, include sample count and a short error analysis. Distinguish retrieval failures, missing evidence, reasoning failures, and answer-format failures.

## 7. Next Extensions

- Scale the synthetic world and add distractor entities.
- Increase kinship-chain depth and attribute constraints.
- Add project and time-feasibility entities.
- Compare a fixed representation against task-adaptive routing, which is the central StructRAG hypothesis.
