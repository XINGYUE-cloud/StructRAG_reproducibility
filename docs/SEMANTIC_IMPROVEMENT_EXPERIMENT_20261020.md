# 官方 Loong 语义质量改进实验

## 目的与基线

该实验尝试提升 Qwen2.5-1.5B 官方 Loong 端到端答案的证据贴合度和格式遵循能力，不改变模型、官方测试集或路由 adapter。既有基线是 1,435 条完成预测，`gpt-5.6-terra` 第三方评测均分 `5.3415/100`；其中 paper 为 `4.1565`，graph 路由为 `3.2113`。该分数不是论文作者 GPT-4 评测的直接复现。

## 改动

- `--query-only-decomposition`：直接将原始问题作为唯一子问题，避免 1.5B 模型改写题意或生成不完整的分解格式。
- `--strict-grounding`：抽取和合并仅能使用提供的证据，禁止补造数字、实体、标题和引用；合并时优先遵循题目要求的 JSON、判决结果编号等格式，并回答证据支持的部分。
- resumable runner 在输出目录记录 `run_config.json`，其中包含提示词模板 SHA-256、模型/数据路径和实验参数。不同配置不能在一个 checkpoint 目录续跑；没有配置记录的旧 checkpoint 也必须使用新目录。

默认关闭两个开关时，原有提示词行为保持不变。这是 pipeline prompt ablation，不是论文原方法中的新训练方法，也不改变官方训练集或测试集。

## 建议运行流程

先选择 60 条官方样本做按文档类型和难度分层的 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run_official_loong_structrag_qwen_resume \
  --dataset data/official_loong_processed/loong_qwen18k_whole_docs.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --router-adapter outputs/router_dpo_full \
  --prompts-dir prompts \
  --output-dir outputs/official_loong_semantic_strict_smoke_20261020 \
  --context-limit 32768 \
  --max-samples 60 --stratified --seed 20261020 \
  --query-only-decomposition --strict-grounding
```

只有在预测完整、格式有效且没有系统性退化后，才用相同配置及独立目录对全部准备完成的官方样本运行。保留原有预测和评测结果，不要复用旧 checkpoint 目录。

## 对照评测

应以 smoke 输出中的 ID 从旧基线预测及旧基线 `judgments.jsonl` 取出相同样本，按 ID 做成对比较；两组都使用 `gpt-5.6-terra`、同一评测提示和同一 endpoint。主要报告成对的均分变化、类型/难度/结构分组变化、有效 JSON 或指定答案格式比例、请求错误数。smoke 的初步差异不能当作全量结果；若全量运行则在同一 1,435 条 ID 上完成评测，并同时报告旧基线与新结果。

第三方 evaluator 的随机性、endpoint 更新和原始材料覆盖限制仍可能影响分数。不要根据 Router 合成数据的 100% 分类准确率推断端到端问答会提升，也不要将本实验分数直接等同论文 `LLM score=60.38`。

## 当前状态

服务器 smoke 已完成并记录在
[`docs/SEMANTIC_IMPROVEMENT_RESULTS_20261020.md`](SEMANTIC_IMPROVEMENT_RESULTS_20261020.md)。
Query-only 在 53 条成对样本上整体基本持平，strict grounding 明显下降，
因此没有启动全量增强推理。所有输出都写入新目录；服务器原有目录和进程未被删除、终止或修改。
