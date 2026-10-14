# Included Synthetic Data

This directory contains the fixed-seed generated dataset used by the repository's
controlled local experiments. It is included so that the four-view baseline and
router measurements can be inspected without regenerating the data.

## Contents

| Path | Contents |
| --- | --- |
| `benchmark/profiles/` | 1,838 generated person profiles and 5,740 generated base relations across 30 independent units |
| `benchmark/kb_views/` | Text, table, triple, graph, and combined knowledge-base views |
| `benchmark/questions/` | 2,921 generated questions and per-task question files |
| `benchmark/lexicon.json` | Synthetic lexicon used to generate the benchmark |
| `router/` | Unit-disjoint router splits: 900 train, 240 validation, and 240 test rows |

The data are deterministic outputs of the scripts in `scripts/`, using seed
`42`, 30 units, and 6 families per unit. Names, organizations, occupations,
attributes, and relationships are generated values for an educational benchmark;
they are not real-person records.

## Scope

This is not a redistribution of the official Loong benchmark or the paper's
released router preference data. Download those third-party sources from their
upstream repositories as described in the project README. The files here support
the repository's separately documented synthetic four-view experiment only.

## Official-Difficulty Five-Type Extension

`official_difficulty_five_type_v2/router_v3/` is a separate five-label extension
with `chunk`, `table`, `graph`, `algorithm`, and `catalogue`. It contains 3,200
training, 800 validation, and 1,600 test preference rows. The test difficulty
counts are 250/300/641/409, matching the official Loong level proportions, and
`unit_id` is split-disjoint. These are synthetic educational rows generated from
the supplied person-profile world; they are not the official Loong benchmark or
the unreleased five-type router set. See
`docs/OFFICIAL_DIFFICULTY_FIVE_TYPE_EXPERIMENT_20261014.md` for the protocol and
limitations.
