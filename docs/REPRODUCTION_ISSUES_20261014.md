# 复现过程中的问题与处理

## 已处理的问题

### 1. 官方路由数据不是五类

论文描述五种知识形态，但作者公开的 `train_router/data/train.json` 和 `test.json` 只包含 `chunk`、`table`、`graph`。因此公开数据线严格保留三类；缺少的 `algorithm`、`catalogue` 只在单独的合成扩展线中生成，不能混入官方三类指标。

### 2. 上游标题标记不一致

Loong 预处理代码读取文档后输出普通标题文本，而公开 StructRAG 的 `split_content_and_tile()` 按 `<标题起始符>` 和 `<标题终止符>` 拆分。准备脚本在不改正文内容的前提下补回这两个标记，并记录来源文档名，使输入能被公开结构化模块解析。

### 3. 长文档超过 Qwen2.5-1.5B 上下文

模型配置的最大位置长度为 32,768。若直接把所有 Loong 文档拼入 prompt，会产生超长输入。准备脚本采用“整篇文档选择”策略：能放入预算就加入，放不下就跳过整篇，并在每条记录写入 `included_doc_names`、`omitted_doc_names` 和 `coverage_status`。这样会降低覆盖率，但不会制造静默截断的不可审计结果。

### 4. 全量任务容易因 SSH 断开丢失进度

原始 runner 只在全程结束时写结果，路由到 433 条后中断时没有可用的 predictions 文件。新增的 checkpoint runner 每条 route/answer 都追加并 flush 到 JSONL，重启可复用已完成 ID。

### 5. 官方评分需要外部评测器

论文 Loong 结果使用作者的 step-3 LLM 评测提示，最终输出 score、成功率和 perfect rate。当前没有配置 GPT-4/OpenAI API，所以本项目不伪造论文 LLM score，只保留字符串和证据召回辅助指标。

### 6. GitHub HTTPS 推送进程崩溃

本机的 `git-remote-https.exe` 在向指定 GitHub 仓库推送时出现内存访问冲突。代码、README 和报告均已在本地 Git 提交；远端读取可用，但写入需要在修复本机 Git HTTPS 组件后重新推送，或改用已绑定 GitHub 账户的 SSH key / 细粒度 PAT。该问题与实验代码和服务器任务无关。

## 尚未消除的差异

- Qwen2.5-1.5B 与论文 Qwen2-7B/Qwen2-72B 的能力和上下文行为不同。
- 公共路由器只有三类，五类扩展的 judge 是确定性规则，不是作者的 LLM judge。
- 上游 graph prompt 硬编码为论文标题引用关系，对 financial/legal 样本并不完全自然。
- 文档预算导致一部分 Loong 样本无法完整覆盖，不能与论文 10K-250K 长上下文结果直接比较。
- 端到端辅助 EM/contains/leaf recall 不能替代作者的 GPT-4 语义评分。

## 发布安全

服务器密码、askpass 脚本、模型权重、官方数据和运行日志不进入 Git。公开仓库只包含代码、公开 prompt 模板、脱敏文档和可复现命令。
