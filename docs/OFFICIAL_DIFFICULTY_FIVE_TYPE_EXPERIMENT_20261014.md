# 官方难度校准五类数据实验

日期：2026-10-14

## 目的

为五类 StructRAG 结构生成训练集和独立测试集，并把难度分布校准到官方
Loong 数据的四个 level。公开 StructRAG 路由数据只有 `chunk/table/graph`，
所以 `algorithm` 和 `catalogue` 仍然是合成扩展，不能当作官方五类数据。

## 数据协议

生成脚本：`scripts/build_official_difficulty_five_type_data_v2.py`

| Split | 样本数 | level 1 | level 2 | level 3 | level 4 | 信息单元 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 3,200 | 500 | 600 | 1,282 | 818 | 18 |
| validation | 800 | 125 | 150 | 321 | 204 | 6 |
| test | 1,600 | 250 | 300 | 641 | 409 | 6 |

测试集比例严格复用官方 Loong 的 `250/300/641/409`。难度控制同时写入每条
样本：关系跳数、条件数量、干扰项数量和目录层级深度分别随 level 递增。
同一个 `unit_id` 只出现在一个 split，避免人物和关系网络泄漏。

数据目录：`data/official_difficulty_five_type_v2/router_v3/`

## 训练与评测

使用 Qwen2.5-1.5B-Instruct、BF16 LoRA-DPO，3 epochs，学习率 `1e-5`，
batch size 8，梯度累积 2，LoRA rank 16。训练适配器只保存在服务器实验目录，
仓库只提交数据、脚本和摘要指标。

服务器评测结果：

| 指标 | 结果 |
| --- | ---: |
| 测试样本 | 1,600 |
| 五类 Accuracy | 100.00% |
| 五类 Macro-F1 | 100.00% |
| level 1 Accuracy | 100.00% (250/250) |
| level 2 Accuracy | 100.00% (300/300) |
| level 3 Accuracy | 100.00% (641/641) |
| level 4 Accuracy | 100.00% (409/409) |

训练摘要和完整指标见 `docs/difficulty_five_type_metrics_20261014/`。

## 与四条实验线的关系

| 实验线 | 新数据能否直接使用 | 当前解释 |
| --- | --- | --- |
| 公开三类路由 | 不能混入 | 仍报告官方公开三类数据的 98.34%/98.33%，保持可比性 |
| 合成五类路由 | 可以 | 使用本数据训练和同难度测试，结果为 100%/100% |
| 合成五阶段端到端 | 可以作为新控制集 | 应使用同一 test split 运行 `run_paper_aligned_five_type_structrag_v1.py`；路由分数提升不等于答案质量提升 |
| 官方 Loong 端到端 | 不能替换官方测试集 | 仍报告 1,435 条完成、canonical exact match 0%、leaf recall 4.10%；合成数据不会改变该官方结果 |

因此，当前已证实的是五类路由线在官方难度比例测试上的提升。端到端合成线
需要另行运行完整五阶段推理；官方 Loong 线必须继续使用官方 Loong 测试集，不能
用合成样本宣称提升。

本次先完成 10 条分层端到端控制（五类各 2 条）：路由准确率 100%，exact
match 0%，answer-contains 30%，平均关键词召回 43.93%。这说明路由器已经选对
结构，但 1.5B 模型的结构化、证据抽取和答案合并仍是瓶颈；不能把路由 100% 解释
成端到端问答 100%。

## 复现命令

```bash
python scripts/build_official_difficulty_five_type_data_v2.py \
  --profiles /path/to/data/benchmark/profiles/people_profiles.jsonl \
  --out-dir data/official_difficulty_five_type_v2/router_v3 \
  --train-count 3200 --validation-count 800 --test-count 1600 \
  --candidates-per-level 25 --seed 20261014

CUDA_VISIBLE_DEVICES=1 python -m src.train_synthetic_five_type_dpo_router_v1 \
  --train data/official_difficulty_five_type_v2/router_v3/train.jsonl \
  --validation data/official_difficulty_five_type_v2/router_v3/validation.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --output-dir models/router_official_difficulty_v2

CUDA_VISIBLE_DEVICES=1 python -m src.evaluate_official_difficulty_five_type_router_v2 \
  --test data/official_difficulty_five_type_v2/router_v3/test.jsonl \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --adapter models/router_official_difficulty_v2 \
  --output-dir results/router_official_difficulty_v2
```

数据是固定种子的教育性合成数据，不是官方 Loong 或作者未公开的五类路由数据。
100% 结果反映生成任务的结构边界和模板一致性，不能直接外推到自然文档或论文
的 GPT-4/LLM 评测分数。
