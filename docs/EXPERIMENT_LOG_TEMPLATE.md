# Experiment Log Template

Copy this section for each run. Keep the filled logs in pull requests or in a versioned docs/experiments/ directory; do not commit raw model outputs unless they are small and non-sensitive.

## Run ID

- Date:
- Commit SHA:
- Goal:

## Dataset

- Seed:
- Units / families per unit:
- Question file:
- Total questions and task-type counts:

## Retrieval and Model

- Representation: text / table / triples / graph
- Retriever and embedding model:
- Top-k and context limit:
- LLM and revision:
- Backend and hardware:
- Temperature / max new tokens:

## Results

| Metric | Overall | Attributes | Family | Derived kinship | Multi-hop | Attribute filter |
|---|---:|---:|---:|---:|---:|---:|
| Exact match | | | | | | |
| Answer contains | | | | | | |
| Gold Recall | | | | | | |
| All Gold | | | | | | |
| Median latency | | | | | | |

## Error Analysis

- Retrieval failures:
- Reasoning failures:
- Formatting failures:
- Surprising cases:
- Next change to test:
