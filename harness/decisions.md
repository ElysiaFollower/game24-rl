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
