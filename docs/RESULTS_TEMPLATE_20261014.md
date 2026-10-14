# 官方 Loong 最终结果记录

本页在服务器断点任务完成后填写，数据源为同一输出目录中的 `metrics.json`。

| 指标 | 数值 |
| --- | ---: |
| selected official examples | pending |
| preparation exclusions (no document fits) | pending |
| routed examples | pending |
| completed examples | pending |
| pipeline errors | pending |
| exact canonical match | pending |
| mean gold-leaf recall | pending |

“Preparation exclusions” are rows removed before model routing because no complete
document fits the configured input budget. They are not model errors. “Pipeline
errors” are failures after routing or during structurization/decomposition/
extraction/merging.

## Route distribution

待 `metrics.json` 的 `route_counts` 填入 `chunk`、`table`、`graph` 数量。

## Per-type

待 `metrics.json` 生成后填入 financial、legal、paper 的样本数和辅助指标；同时
注明每一类的 completed count，而不是把 preparation exclusions 当作预测失败。

## 解释

这里的指标不是论文的 GPT-4/LLM score。论文对比表应单独引用论文原始数值，并明确模型、标签、上下文覆盖和评测器差异。
