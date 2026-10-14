# 官方 Loong 最终结果记录

本页来自服务器断点任务完成后写出的 `metrics.json`。模型为
Qwen2.5-1.5B-Instruct；仅使用一张 GPU 执行。

| 指标 | 数值 |
| --- | ---: |
| selected official examples | 1,600 |
| preparation exclusions (no document fits) | 165 |
| routed examples | 1,435 |
| completed examples | 1,435 |
| pipeline errors | 0 |
| exact canonical match | 0.00% (0 / 1,435) |
| mean gold-leaf recall | 4.10% |

“Preparation exclusions” are rows removed before model routing because no complete
document fits the configured input budget. They are not model errors. “Pipeline
errors” are failures after routing or during structurization/decomposition/
extraction/merging.

## Route distribution

| Structure | Completed predictions |
| --- | ---: |
| `chunk` | 461 |
| `graph` | 213 |
| `table` | 761 |

## Per-type

| Type | Completed | Mean gold-leaf recall | Exact canonical match |
| --- | ---: | ---: | ---: |
| financial | 558 | 5.47% | 0.00% |
| legal | 500 | 5.62% | 0.00% |
| paper | 377 | 0.05% | 0.00% |

所有 1,435 条已路由样本均完成端到端 pipeline。原始 `metrics.json` 的
`error_records=165` 是没有完整文档可纳入输入预算的 preparation exclusions，
不是 pipeline error。

## 解释

这里的指标不是论文的 GPT-4/LLM score。论文对比表应单独引用论文原始数值，并明确模型、标签、上下文覆盖和评测器差异。
