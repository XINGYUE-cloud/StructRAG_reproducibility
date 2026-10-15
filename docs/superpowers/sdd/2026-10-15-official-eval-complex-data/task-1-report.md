# Task 1 Report: Verified Complex Five-Type Dataset v3

Implemented the grounded generator and tests in this task. The generator clones
the supplied profile units into deterministic synthetic worlds with fresh IDs
and names, assigns source roots to exactly one split, and writes the requested
schema plus `source_evidence_ids`, structured `task_spec`, and independently
checkable `oracle` fields. Existing output directories are never overwritten.

Generated corpus: `data/complex_five_type_v3_verified_20261015/`

Measured full-corpus results:

- Counts: train 16,000; validation 2,000; test 3,200; total 21,200.
- Unique row IDs: 21,200/21,200.
- Unique `(query, documents)` meanings: 21,200/21,200.
- Labels: catalogue 4,286; algorithm 4,284; table 4,270; chunk 4,251; graph 4,109.
- Levels: 1: 3,313; 2: 3,975; 3: 8,493; 4: 5,419.
- Grounding audit: every evidence ID occurs in the selected document text, 21,200/21,200 rows.
- Source-root isolation: train 18 roots, validation 6 roots, test 6 roots, with no overlap.

Graph rows use explicit relation steps and are skipped when a requested path is
not available; no self or revisit fallback is used. Chunk rows remain
single-hop. Table and algorithm rows carry distinct structured conditions/rule
oracles. Catalogue rows include an explicit target ID and requested hierarchy
depth. Document counts and character lengths are computed from emitted
documents rather than estimated.

Verification:

```text
python -m unittest tests/test_complex_five_type_data_v3.py
Ran 4 tests ... OK

python -m unittest discover -s tests
Ran 15 tests ... OK
```

Tests cover grounded multi-hop paths, impossible graph fixtures, exact document
metadata, deterministic same-seed output, split/disjointness behavior, all
five labels, and unsatisfiable selection requests. This corpus is synthetic;
its level distribution is a sampling control and is not evidence of paper or
official Loong difficulty equivalence.
