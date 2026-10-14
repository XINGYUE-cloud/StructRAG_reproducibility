# Local Experiment Results

Date: 2026-10-12

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM
- CUDA runtime: 12.8
- Python: 3.11.9
- Base model: Qwen2.5-1.5B-Instruct, stored in a local path supplied by the runner
- Adapter: FP16 LoRA, rank 8, 2 epochs, learning rate 2e-4

The Windows bitsandbytes package did not include a CUDA 12.8 binary. Therefore this run uses FP16 LoRA, not QLoRA. The training and evaluation scripts record the actual mode.

## Dataset and Router Setup

The supplied generator created 30 independent information units with 1,838 people, 5,740 basic relations, and 2,921 questions. Router labels are deterministic four-view proxies: `chunk`, `table`, `triples`, and `graph`.

| Split | Units | Examples | Balance |
|---|---:|---:|---|
| Train | 18 | 900 | 225 per label |
| Validation | 6 | 240 | 60 per label |
| Test | 6 | 240 | 60 per label |

Splits are by information unit, so no family/network occurs in more than one split.

## Router Result

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Qwen2.5-1.5B zero shot | 25.00% | 10.00% |
| Qwen2.5-1.5B + FP16 LoRA | 100.00% | 100.00% |
| StructRAG paper router | 94.38% | 93.42% |

The local test confusion matrix is diagonal: all four labels have 60 correct predictions out of 60.

This is not a direct paper comparison. The paper uses a five-class Qwen2-7B DPO router trained from 900 LLM-constructed preference pairs. This project uses a four-class Qwen2.5-1.5B FP16 LoRA router with transparent deterministic labels on synthetic, unit-disjoint data. The 100% local value shows this proxy task is easy after fine-tuning; it does not establish that the paper's full StructRAG system is matched or exceeded.

## Four-View RAG Baseline

Each view used the same 40 stratified questions (8 per task type), within-unit lexical retrieval, top-k 3 contexts, greedy local Qwen generation, and FP16 inference.

| Representation | Exact match | Answer contains |
|---|---:|---:|
| Chunk | 42.5% | 52.5% |
| Table | 30.0% | 45.0% |
| Triples | 40.0% | 47.5% |
| Graph | 50.0% | 55.0% |

Graph is the best local representation under this bounded baseline. Its direct-attribute and attribute-filter exact match rates are each 87.5%; multi-hop kinship is 50.0%; family-member queries are 12.5%; derived kinship is 12.5%. For derived kinship, evidence recall is 100%, so the weak answer score points to model reasoning/output limitations rather than a retrieval miss.

This RAG baseline is deliberately small. It does not include the paper's LLM structurizer, question decomposition/utilizer, five-way adaptive router, or external benchmarks.

## Reproducible Outputs

- Router metrics: `results/router_qlora/router_metrics.json`
- Router comparison: `results/router_qlora/paper_comparison.md`
- Zero-shot router metrics: `results/router_zero_shot/router_metrics.json`
- RAG metrics and predictions: `results/rag/`
- RAG summary: `results/rag/rag_summary.md`
