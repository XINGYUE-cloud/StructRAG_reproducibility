# Paper-Method Alignment Update

## Scope

This update limits changes to the methods and execution details present in the
public StructRAG implementation. It does not enable the separate
`--query-only-decomposition` or `--strict-grounding` variants.

## Changes

- Added the paper's `algorithm` and `catalogue` construction and evidence-filtering
  paths to the official resumable runner, using the released prompts and module
  order: router, structurizer, decomposer, precise extractor, answer merger.
- Added explicit `--router-classes {released-three,paper-five}` selection. The
  default preserves the released three-class prompt; `paper-five` exposes all
  five paper structures for a compatible five-class adapter.
- Matched the upstream router's label parsing priority (`table`, `graph`,
  `algorithm`, `catalogue`, then `chunk` fallback).
- Preserved the source document title in chunk retrieval results, as the upstream
  utilizer does before answer merging.
- Set the local generation allowance to 4096 tokens by default, matching the
  released `QwenAPI.response()` default. Context-limit checks remain enabled.
- Updated the resumable run revision and prompt fingerprint so an old checkpoint
  cannot be mixed with this implementation.

## Important Boundary

The released router preference data contains only three labels. The runner can
now execute all five structure modules, but this does not create an official
five-class Loong router or establish that the public adapter can select
`algorithm` and `catalogue` reliably. Keep `released-three` for the official
adapter. Any evaluation using a synthetic five-class adapter must use
`paper-five` and be reported separately from the official-router result.

## Verification Status

- Local unit tests: 26 passed.
- Python compilation: passed.
- Local runner CLI check: unavailable because PyTorch is not installed in the
  workstation environment.
- Remote official Loong inference: not run in this update. SSH public-key
  authentication to the configured server was rejected, and no existing local
  credential was available for a safe non-interactive run.
- Therefore no semantic-score improvement is claimed. The next valid comparison
  must use a new output directory, the same official sample IDs and evaluator,
  and report the chosen router adapter explicitly.
