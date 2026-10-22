# Runtime and Token Accounting (2026-10-22)

## Completed full router test

The completed router-only run evaluated both supplied archives in full:

| Dataset | Rows |
| --- | ---: |
| VALUE-KAO-PersonWork-EN-v2-Experiment-1k | 1,000 |
| RCAEval-RE2-Microservice-v1 official test | 1,538 |
| **Total** | **2,538** |

The prediction file contains one router call per row. Re-tokenizing the exact
saved prompts and completions with the run's tokenizer gives:

| Quantity | Tokens |
| --- | ---: |
| Router input tokens | 875,707 |
| Saved completion tokens | 2,560 |
| **Total reconstructed tokens** | **878,267** |

The original run did not record `perf_counter` timestamps, so its wall-clock
duration cannot be recovered exactly from the artifacts. The result must not be
reported with an invented duration. The evaluation scripts now record
`cost_and_speed.wall_clock_seconds`, per-row latency, input tokens, output
tokens, and total tokens for future full runs.

## Existing end-to-end pilot

The 12-condition pilot used four rows per condition (two from each archive),
not the full archives. Across all 12 conditions it consumed 904,050 input
tokens and 4,413 generated tokens (908,463 total). The sum of recorded stage
latencies was 549.59 seconds; this is accumulated per-row stage time, not a
single wall-clock duration, because the pilot artifacts did not record one
process-level timer.

## Preference generation accounting

The English preference-generation retry is resumable. Accepted rows carry the
API's `prompt_tokens`, `completion_tokens`, `total_tokens`, and request elapsed
time when the provider returns usage. Rows accepted before this instrumentation
were retained but are marked as usage-unrecorded; they are not silently
estimated as exact provider usage.

