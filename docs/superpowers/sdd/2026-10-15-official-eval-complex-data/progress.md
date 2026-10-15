# SDD ledger - plan: docs/superpowers/plans/2026-10-15-official-eval-complex-data.md

## Preflight

- Worktree: existing main checkout; user explicitly approved continuing in the current repository.
- Existing output policy: preserve all server content; new timestamped directories only.
- Task overlap: Task 1 creates an independent generator; Task 3 creates an independent evaluator; Task 2 consumes runner JSONL and does not alter those modules.
- Ruling: use dependency-light `unittest` fixtures because pytest is not installed locally; this tests the same public Python interfaces without adding a test dependency.

