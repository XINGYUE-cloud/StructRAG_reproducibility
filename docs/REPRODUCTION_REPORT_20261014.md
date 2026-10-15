# StructRAG 复现学习与实验报告

日期：2026-10-14

## 结论先行

本项目是一个**协议对齐、资源受限的工程复现**：保留公开 StructRAG 的
`Router -> Structurizer -> Decomposer -> Precise extractor -> Answer merger`
顺序、提示词和官方 Loong 数据来源，使用 Qwen2.5-1.5B 在单张 GPU 上运行。
它不是论文数值的逐项重现，因为论文使用 Qwen2-7B/Qwen2-72B、五类路由和
外部 LLM 评测器，而公开仓库只提供三类路由标签。

服务器全量任务已完成；本报告的最终结果来自同一输出目录的 `metrics.json`。

## 论文内容要点

StructRAG 的出发点是：不同问题适合不同的知识表示。将所有资料统一切成
文本 chunk 会损失表格中的比较关系、图中的多跳关系、算法的步骤约束和目录
的层级结构。系统先由路由器选择结构，再把分散文档结构化，最后分解问题、
抽取证据并合并答案。

论文讨论的五类结构是：

| 结构 | 典型问题 | 表示 |
| --- | --- | --- |
| `chunk` | 单跳定位 | 原始文本片段 |
| `table` | 统计、比较、筛选 | 带标题/来源的表格 |
| `graph` | 多跳、引用或关系链 | head-relation-tail 三元组 |
| `algorithm` | 规划、条件决策、步骤执行 | 事实约束下的伪代码 |
| `catalogue` | 层级总结、全局组织 | 多级编号目录 |

路由偏好数据按 synthesize -> simulate -> judge 构造，再用 DPO 训练；论文的
主实验使用更大的 Qwen 系列模型。论文报告 Loong 总体 `LLM score=60.38`、
`EM=0.23`，路由器 `Accuracy=94.38%`、`Macro-F1=93.42%`。

## 本次复现流程

1. 下载并校验官方 Loong 问答文件和文档压缩包；问答文件共 1,600 条，SHA-256
   为 `CD640D3CA0E41D3638153506E3CAEE109850B30DE0A4E829862232A2B6302DA8`。
2. 用 `src/prepare_official_loong_structrag.py` 恢复公开 StructRAG 需要的标题
   标记。只纳入完整文档，不静默截断，并记录每条样本的覆盖状态。
3. 用公开三类路由偏好数据训练/评测 Qwen2.5-1.5B LoRA 路由器。
4. 按论文模块顺序执行官方 Loong：路由、结构化、问题分解、精确抽取、答案
   合并。断点 runner 每条记录写入 JSONL，可在 SSH 中断后继续。
5. 用可审计的 canonical exact match、答案叶节点召回率、路由分布和完成数做
   辅助评测；没有 GPT-4/OpenAI 评测配置时不伪造论文的 LLM score。

## 数据覆盖与运行约束

18K 输入预算下：完整覆盖 45 条、部分覆盖 1,390 条、没有任何完整文档可纳入
的 165 条。后者是**准备阶段排除**，不是模型预测错误。30K 预算下对应为
177、1,389、34。当前服务器任务使用 18K 预处理数据、32K 模型上下文和 GPU 3。

## 2026-10-15 复杂五类扩展数据

为补充公开路由数据中缺失的 `algorithm` 和 `catalogue`，另行生成了
`data/complex_five_type_v3_verified_20261015/`。该目录不是官方 Loong 数据，
不能与官方测试结果合并解读。

| split | rows | source roots | labels |
| --- | ---: | ---: | --- |
| train | 16,000 | 18 | five labels |
| validation | 2,000 | 6 | five labels |
| test | 3,200 | 6 | five labels |

全量 21,200 行的 `unit_id`、`(query, documents)` 均唯一，证据 grounding audit
为 21,200/21,200；三个 split 的 source roots 互不重叠。每行保存图跳数、条件数、
干扰项数量、文档数、近似字符长度和目录深度等控制字段。这里的 level 是生成器的
实验控制，不能称为官方 Loong 难度等级。

## 结果记录

| 项目 | 本次结果 | 论文结果 | 备注 |
| --- | ---: | ---: | --- |
| 公开三类路由 Accuracy | 98.34% | 94.38% | 数据/模型不同，不能直接宣称超越 |
| 公开三类路由 Macro-F1 | 98.33% | 93.42% | 同上 |
| 官方 Loong selected rows | 1,600 | 论文全量协议 | 官方 source rows |
| 官方 Loong preparation exclusions | 165 | 未按本预算设置 | 18K 下没有完整文档可纳入 |
| 官方 Loong completed | 1,435 | 论文全量 | 全部已路由行完成 pipeline |
| canonical exact match | 0.00% (0/1,435) | EM 0.23 | 本项目是字符串诊断指标 |
| gold-leaf recall | 4.10% | 无对应论文指标 | 辅助证据覆盖指标 |

最终路由分布：`table=761`、`chunk=461`、`graph=213`。按文档类型的
gold-leaf recall：financial 5.47%（558 条）、legal 5.62%（500 条）、
paper 0.05%（377 条）；三类 canonical exact match 均为 0%。`metrics.json`
中的 165 条 error record 都是准备阶段的 no-document-fits exclusion，pipeline
阶段没有错误。

## 与论文不一致的地方

- 公开路由数据只有 `chunk/table/graph`，缺少 `algorithm/catalogue`；两类扩展
  数据单独生成，不能混入官方指标。
- Qwen2.5-1.5B 替代论文中的 Qwen2-7B/Qwen2-72B。
- 单卡上下文预算使部分长文档无法纳入；论文使用更大的上下文和多卡推理。
- 论文的 Loong 分数由外部 LLM 判断，本项目没有相同评测 API。
- 公开 graph prompt 偏向论文引用关系，对 financial/legal 文档并不完全自然。

## 遇到的问题与解决

- 标题解析格式不一致：预处理脚本补回 `<标题起始符>`/`<标题终止符>`。
- 全量运行容易因 SSH 断线丢失结果：新增 append-and-flush checkpoint runner。
- 长文档超上下文：采用整篇文档选择并显式记录遗漏，不做静默截断。
- 发布安全：仓库不包含密码、askpass 文件、模型、官方数据、日志或 checkpoint；
  运行脚本中的服务器路径改为环境变量。

## 复现命令入口

环境安装见 `requirements.txt`；官方数据准备和单卡命令见 README。可执行：

```bash
PROJECT_DIR=/path/to/StructRAG_reproducibility \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_official_loong_resume.sh \
  --dataset data/official_loong_processed/loong_qwen18k_whole_docs.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --router-adapter outputs/router_dpo_full \
  --prompts-dir prompts \
  --output-dir outputs/official_loong_checkpoint
```

生成的 `metrics.json`、`progress.json`、`routes.jsonl`、`predictions.jsonl` 和
`errors.jsonl` 应保留在本地实验目录，不提交到公开仓库。
