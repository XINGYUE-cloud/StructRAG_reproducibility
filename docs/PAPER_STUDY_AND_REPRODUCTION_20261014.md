# 论文学习与复现记录

日期：2026-10-14

## 1. 论文要解决的问题

StructRAG 关注知识密集型问题中的一个结构性困难：答案证据往往分散在多份文档中。普通的 chunk RAG 把文档切成相似的文本片段后检索，容易带入噪声，也不擅长进行全局比较、关系链推理、算法步骤整理和层级总结。论文的核心假设是，问题类型与知识表示形态之间存在认知匹配关系；先选择合适的表示，再进行推理，比始终使用文本块更合适。

## 2. 论文方法

论文的推理链是：

```text
问题 q + 文档核心信息 C
        |
        v
Hybrid Structure Router: t = R(q, C)
        |
        v
Scattered Knowledge Structurizer: (k_i, b_i) = S(q, t, d_i)
        |
        v
Structured Knowledge Utilizer
  Q_hat = U_decompose(q, B_t)
  K_hat = U_extract(q_hat, K_t)
  answer = U_infer(q, Q_hat, K_hat)
```

五种临时知识形态及其典型用途如下：

| 形态 | 适合的问题 | 生成结果 |
| --- | --- | --- |
| chunk | 单跳问答、直接定位 | 原始文本块 |
| table | 统计、比较、属性筛选 | 保留标题和来源的表格 |
| graph | 多跳关系链、引用关系 | head-relation-tail 三元组 |
| algorithm | 规划、条件决策、步骤推理 | 与原文事实对应的伪代码 |
| catalogue | 层级组织和全局总结 | 多级编号的目录式摘要 |

路由器训练采用 synthesize -> simulate -> judge 生成偏好对，再用 DPO 优化。论文中的路由偏好对规模约为 900，主实验使用 Qwen2-7B 路由器和 Qwen2-72B 结构化/利用模块。

## 3. 论文评测协议与原始结果

Loong 长文评测包含 Spotlight、Comparison、Clustering、Chain 四类任务，并按文档长度划分四个集合。论文报告的总体 Loong 结果为：

| 集合 | LLM score | EM |
| --- | ---: | ---: |
| Set 1 | 69.43 | 0.35 |
| Set 2 | 60.95 | 0.24 |
| Set 3 | 57.92 | 0.21 |
| Set 4 | 51.42 | 0.10 |
| Overall | 60.38 | 0.23 |

论文路由器表中的准确率/Macro-F1 为 94.38%/93.42%。Loong 的语义分数由外部 LLM 评判，不等同于简单字符串准确率。

## 4. 本次复现做了什么

本项目分为三条互相隔离的实验线：

1. **公开路由数据线**：使用作者公开的 1,444 条训练偏好对和 362 条测试偏好对。公开数据实际只有 `chunk`、`table`、`graph` 三类。Qwen2.5-1.5B 使用 BF16 LoRA-DPO 训练，测试 Accuracy 98.34%、Macro-F1 98.33%。
2. **合成五类扩展线**：用已有数据生成器构造 `algorithm` 和 `catalogue`，与三类公开标签组成五类路由器。测试 Accuracy/Macro-F1 均为 100%，但任务模板化，不能当作自然文档泛化结果。
3. **官方 Loong 端到端线**：下载并校验官方 `loong.jsonl` 和 `doc.zip`，恢复文档标题标记，按官方代码的路由、结构化、分解、抽取、合并顺序运行 Qwen2.5-1.5B。由于模型上下文为 32K，准备脚本只保留完整文档，不对正文静默截断。

## 5. 官方数据核验

官方题目文件共 1,600 条，SHA-256：

```text
CD640D3CA0E41D3638153506E3CAEE109850B30DE0A4E829862232A2B6302DA8
```

类型分布为 financial 700、legal 500、paper 400；level 分布为 1:250、2:300、3:641、4:409。18K 输入预算的准备结果为：完整覆盖 45 条、部分整文档覆盖 1,390 条、无文档可纳入 165 条。30K 输入预算的结果为：完整覆盖 177 条、部分整文档覆盖 1,389 条、无文档可纳入 34 条。两种准备都没有 preparation error，也没有切断任何单篇文档。

## 6. 单卡执行状态

服务器只使用 GPU 3，且所有新增文件均放在新的日期目录中。第一次非断点运行在路由阶段停在 433/1600，旧目录保留作为中断记录。随后启动了 checkpoint runner：

- 路由成功记录写入 `routes.jsonl`；
- 每条答案立即写入 `predictions.jsonl`；
- 无文档样本写入 `errors.jsonl`；
- `progress.json` 记录 selected/routed/completed/remaining；
- 重启时按 ID 跳过已有答案，不覆盖旧结果。

截至本文创建时，断点任务已完成路由记录 1,435 条，165 条因没有任何完整文档可放入预算而记录为 route exclusion，并已进入答案生成阶段。最终 `metrics.json` 以服务器输出为准；在任务完成前不把部分结果称为全量准确率。

## 7. 结果如何解释

公开路由的 98.34% 只说明 Qwen2.5-1.5B 在作者公开的三类偏好测试上表现良好；它不能证明五类路由器或整个 StructRAG 达到论文水平。合成五类的 100% 主要反映任务模板和标签规则清晰。端到端结果必须同时考虑路由、结构化、分解、抽取和合并，不能只看路由准确率。

没有可用的 GPT-4/OpenAI 评测接口时，本项目只报告可审计的 `exact_canonical_match`、`gold_leaf_recall`、路由分布、完成数和错误数，不报告论文意义上的 LLM score。
