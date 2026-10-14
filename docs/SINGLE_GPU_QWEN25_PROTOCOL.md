# Qwen2.5-1.5B Single-GPU Protocol

This experiment is a resource-constrained adaptation, not a numerical reproduction of the paper.

## Training Data

The router uses `train_router/data/train.json` and `train_router/data/test.json` from the official StructRAG repository. The released files contain 1,444 training and 362 held-out DPO preference pairs, over three labels only:

- `chunk`
- `table`
- `graph`

This is a material difference from the five candidate types described in the paper. The released router prompt and released router training data do not include `catalogue` or `algorithm`; this protocol does not invent those labels.

The model is Qwen2.5-1.5B-Instruct with BF16 LoRA DPO, three epochs, learning rate `1e-5`, batch size one, and gradient accumulation eight. These preserve the paper's DPO epoch count and learning rate while changing the base model and the fine-tuning method needed for one GPU.

## Test Data

The test world is generated only with the supplied scripts in `scripts/`:

1. `build_lexicon.py`
2. `build_profiles.py`
3. `build_views.py`
4. `build_hard_person_questions.py`

The type mapping follows the official released router labels:

| Router label | Supplied test representation |
| --- | --- |
| `chunk` | `kb_text.jsonl` |
| `table` | `kb_table.jsonl` |
| `graph` | `kb_triples.jsonl` |

`kb_graph.jsonl` is retained as an additional supplied representation but is not a separate router class. Treating it as a fourth router label would violate the official released DPO data.

## End-to-End Modes

`dynamic` is the primary result. It feeds raw text profiles to the trained router, dynamically builds tables or triples for the selected representation, decomposes the question, extracts precise evidence, and generates an answer.

`provided` uses the supplied pre-rendered table or triples view after routing. It is a representation-matching control, not a StructRAG structurizer result.

Both modes use lexical top-document selection before structurization to bound one-GPU runtime. The original paper restructures its full document collection with Qwen2-72B; that remains an explicit unmatched condition.
