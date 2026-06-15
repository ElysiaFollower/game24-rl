<!--
职责：记录会影响后续 agent 决策的重要选择及其理由。
边界：不要记录每次小改动、聊天摘要或可从代码直接看出的事实。
-->

# 决策日志

## 记录规则

重要决策必须写明：日期、决策、原因、否决方案、后续约束。

## 决策

### 2026-06-13 - 初始化 repo-native harness

- 决策：采用 repo-native harness，仓库保存指令、状态、验证、交接和质量信息。
- 原因：降低冷启动成本、上下文丢失、范围漂移、验证缺口和返工。
- 否决方案：只依赖聊天 prompt 或单个巨型 `AGENTS.md`。
- 后续约束：项目事实必须进入仓库；重复失败优先转成测试、脚本或检查。

### 2026-06-13 - 高分优先，SFT 保底后 GRPO 冲顶

- 决策：先用 solver-generated short success traces 做 LoRA SFT，达到强 fallback 后再做 GRPO。
- 原因：课程交付要求可展示强结果；SFT 先建立格式和基础求解能力，GRPO 才有稳定优化起点。
- 否决方案：从零开始纯 RL；把 SFT 调成无底洞；直接复制外部仓库。
- 后续约束：SFT v1 solve rate 低于 50% 时优先排查 pipeline bug，不进入 GRPO。
- ADR：`docs/adr/0002-establish-sft-before-grpo.md`、`docs/adr/0013-use-explicit-sft-success-gates.md`

### 2026-06-13 - 课程模型作为主模型

- 决策：主线使用 `Qwen/Qwen2.5-1.5B-Instruct`，不先做 0.5B smoke target。
- 原因：时间优先投向最终可报告模型；4090 级别算力可以支撑 LoRA SFT/GRPO。
- 否决方案：以 0.5B 作为主线；频繁切模型做探索。
- 后续约束：除非课程限制或硬件不可用，否则报告主结果围绕该模型。
- ADR：`docs/adr/0001-use-course-model-as-primary.md`

### 2026-06-13 - 主答案契约只认 `<answer>...</answer>`

- 决策：本仓库只评估 `<answer>...</answer>` 内表达式。
- 原因：主契约越单一，训练、评估、reward 和报告越可靠；没有必要为外部 benchmark 输出格式牺牲维护性。
- 否决方案：兼容 LLM4Game24 的 `expression:` 作为主路径。
- 后续约束：如需外部格式兼容，只能作为独立 adapter/ablation，不能污染主 verifier。
- ADR：`docs/adr/0011-use-single-answer-contract.md`

### 2026-06-13 - 严格 AST + Fraction verifier

- 决策：verifier 使用 AST allowlist + `Fraction`，禁止 Python `eval`。
- 原因：reward 和可展示分数必须可审计、抗注入、无浮点误差。
- 否决方案：regex-only verifier；直接 `eval` 模型输出。
- 后续约束：所有训练 reward 和评估分数必须经过同一 strict verifier。
- ADR：`docs/adr/0012-use-strict-ast-fraction-verifier.md`

### 2026-06-13 - Split 按 multiset 隔离

- 决策：train/validation/test 按 sorted multiset 划分，不能按 trace row 随机划分。
- 原因：同一个 puzzle 的多条 trace 如果跨 split，会虚高验证分数。
- 否决方案：随机拆 SFT 样本行；训练覆盖所有 solvable puzzle 后再报告 heldout。
- 后续约束：所有报告必须说明 split manifest；OOD 单独报告。
- ADR：`docs/adr/0004-use-multiset-isolated-splits.md`

### 2026-06-13 - LLM4Game24 是 reference baseline，不是 target

- 决策：把 [LLM4Game24](https://github.com/LiaoMengqi/LLM4Game24) 作为方法和分数参考，不复制成目标实现。
- 原因：它证明 solver-generated trace + SFT 可行，但本项目需要代码可控、契约清晰、评估可审计。
- 否决方案：fork 后小改；为了对齐其 bench 而改变本项目主契约。
- 后续约束：baseline 对比必须说明模型、split、answer format、verifier 差异。
- 文档：`docs/baselines.md`

### 2026-06-15 - 先训练固定单路径状态转移 trace

- 决策：下一轮 SFT 探索先使用单条正确路径的固定状态转移格式：`<think>` 内只包含 `<s0>` 初始数列和 `<s1>`/`<s2>`/`<s3>` 三步 `a op b = c | left: ...`，`<answer>` 内输出同一路径的最终表达式。暂不加入 rollback、错误探索或多候选路径。
- Assumption：`Qwen/Qwen2.5-1.5B-Instruct` 对 24 点这种窄任务可能更适合学习低熵、强约束、任务专用的状态转移语法，而不是先学习长自然语言搜索叙述。固定 trace 保留 `<think>/<answer>` 分工和 `left:` 状态提示，但减少格式漂移和无效搜索噪声。
- 决策过程：小样本 overfit 已证明本仓库的核心训练、LoRA 保存加载和 strict eval 链路能学会训练样本；baseline 转换实验显示包含搜索状态的信息强于纯短成功 trace，但长 rollback/search trace 的主要失败转为 `<answer>` 契约不闭合；原 short-success trace 格式稳定但 wrong-value 多。综合看，当前更像 SFT teacher 设计问题，而不是明显 tensor/mask/eval 代码 bug。
- 原因：固定单路径 trace 是 short-success 与 search-state 的折中：它显式暴露每一步剩余状态，能约束模型学会局部算术状态更新，同时不引入 rollback 叙事的长上下文和多路径噪声。这个实验也足够短，适合在继续 full finetune、提高学习率或加入多候选搜索前先拿证据。
- 否决方案：立即进入 GRPO；继续只调 v3 checked-chat；直接复制 baseline 训练实现；先上长 rollback/multi-candidate trace；先改主模型、split、answer contract 或 verifier。
- 后续约束：保持主模型、split、strict verifier 和 `<answer>...</answer>` 主契约不变；先跑短 LoRA 消融并用 validation strict solve rate 与 raw outputs case study 判断方向。若 fixed trace 明显改善，再考虑 full finetune、更强学习率或第二阶段多候选状态选择。

### 2026-06-15 - Solver 应支持多路径/search-trace 数据导出

- 决策：后续应把 solver/data generation 扩展为可导出多条正确路径或压缩搜索树轨迹；训练数据可以从这些路径中选择、采样或生成多条样本，而不是长期只依赖 DFS 返回的第一条 canonical path。
- Assumption：24 点搜索空间很小，全路径或搜索树导出在计算上不是瓶颈；真正的设计问题是选择哪些路径作为 teacher signal。模型评测应继续按 strict verifier 判定任意合法表达式为正确，而不是要求匹配训练集中第一条路径。
- 决策过程：当前 fixed single-path 实验显示模型很容易学会格式，但泛化正确率仍低；LLM4Game24 的强 baseline 使用每题多条记录和 rollback/search trace，说明多路径/搜索过程信号可能是高分关键差异之一。用户指出“先搜全路径不代表全用”，这能同时支持 fixed path、多路径 SFT、search trace SFT 和后续 verifier/RL 方案。
- 原因：把全路径/search trace 作为可复用中间数据，可以让我们做有控制的消融：第一条路径 vs 多条成功路径 vs 压缩搜索树，而不反复改 solver 主逻辑。
- 后续约束：不改变主 split、strict verifier 或 `<answer>...</answer>` 契约；多路径样本仍必须按 puzzle multiset 隔离 split，不能让同一 puzzle 的不同路径跨 train/validation/test。
