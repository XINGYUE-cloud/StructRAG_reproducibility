# Task 3 report: cache-first Loong semantic judge

Implemented `src/evaluate_loong_llm_judge.py` with standard-library JSONL I/O,
OpenAI-compatible HTTP judging, strict 0-100 score/reason parsing, transient
request retries, and cache keys composed of prediction ID and model. The
`--dry-run` path parses inline `judge_response` (or `mock_judge_response`)
fixtures and never opens a network connection. Metrics include overall means
and means grouped by type, level, and selected structure; errors are written
separately.

Verification:

```text
python -m unittest tests.test_loong_llm_judge -v
Ran 4 tests ... OK
```
