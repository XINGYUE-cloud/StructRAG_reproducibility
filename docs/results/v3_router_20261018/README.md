# Complex v3 router result

This directory contains sanitized metrics downloaded from the new server run.

- Training: 16,000 rows, one epoch, Qwen2.5-1.5B LoRA-DPO.
- Validation DPO loss: `0.001306752267992124`.
- Test: 3,200 unit/source-root-disjoint synthetic rows.
- Accuracy: `1.0`.
- Macro-F1: `1.0`.

The result is a five-class synthetic router result. It does not measure the
StructRAG answer pipeline and cannot be compared numerically with the paper's
Loong LLM score. The raw test predictions and model adapter remain on the
server and are not committed here.
