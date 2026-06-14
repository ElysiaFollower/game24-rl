<!--
职责：记录目标项目当前生效的 agent 协作模式和开发原则。
边界：不要把历史模式档案、聊天记录、长期项目事实或 profile 来源追踪放在这里。
-->

# Current Harness Mode

<!-- mode-schema: current-harness-mode/v1 -->

## 当前策略摘要

- 生效范围：`M2-sft-audit-and-repair` 严格审计、修复、验证和复训启动。
- Collaboration mode：高自治执行，关键边界升级。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 生效时间：2026-06-15。
- 切换原因：负责人准备离开，希望 agent 在目标、边界和验收标准已写入 active plan 后端到端完成审计、修复、验证、文档和复训启动。

## Collaboration Mode

Agent 先读取仓库事实，再直接推进当前 active plan。保持 WIP=1，不开启无关任务；遇到验证失败先诊断和修复，不因一次失败就停止。每个完成声明必须带验证证据、artifact 路径和清洁状态说明。

只在需要改变目标边界、外部接口契约、持久化语义、主模型、split 策略、answer contract、verifier 接受标准、训练路线，或需要扩大到 active plan 明确排除的非目标时停止并等待负责人确认。

## Development Style Principles

- 研究原型要可复现：数据、split、checkpoint、decoding、verifier version、raw output 和报告路径都要能回溯。
- 先建立反馈环再修复：评测口径、数据样本、训练 mask、checkpoint/load 和 generation 行为必须有可复现证据。
- Python 代码遵循轻量 Google Python Style：公共函数/类用 Google-style docstring，命名清晰，少全局状态。
- solver、verifier、data generation、evaluation 是基础设施，优先写可测试的小函数。
- 训练脚本可以薄，但评估和 verifier 不能薄弱；分数必须来自 strict verifier。
- 允许局部工程取舍，但核心不变量要进测试、诊断脚本、审计报告或 ADR。

## 授权边界

- Agent 可以自主执行：读取项目事实源、审计代码/数据/报告、运行本地测试、运行远端只读诊断、修复确认的 pipeline bug、补 focused tests、更新 docs/harness、同步代码、在修复后启动一轮新的 SFT 训练。
- Agent 可以自主对比 reference baseline 的公开实现和文档，但不能复制外部代码为主实现，不能用 baseline 的 `eval` 路径替代本仓库 verifier。
- 未经负责人确认，不改主模型、不改 split 策略、不改 answer contract、不放宽 verifier/reward 口径、不进入 GRPO、不扩展到 Countdown/OOD、不引入新服务或重型框架。
- 发现必须改变上述边界才能推进时，停止并在 handoff 中记录证据、选项和推荐。

## 验证策略

- 最小验证：`./scripts/harness-check.sh`。
- Python 改动：`python -m compileall src scripts` 和相关 focused tests。
- 基础链路：`pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py tests/test_training_pipeline.py`。
- 质量门禁：`ruff check .` 和 `ruff format --check .`，如因环境不可用需记录原因。
- 远端实验：每个分数必须声明 model、split manifest、checkpoint、decoding config、answer contract、verifier version 和 raw output artifact。
- 不能作为完成证据：训练 loss 单独下降、模型自我判断、非 strict verifier、Python `eval`、存在 split 泄漏的分数、未声明 decoding 的分数。

## 阻塞和升级规则

- 阻塞条件：无法复现核心数据数字；strict verifier 与标准 24 点规则冲突；训练服务器不可用；外部 baseline 访问失败且其实现细节成为关键判断；下一步必须改变主模型/split/answer contract/verifier/训练路线；负责人要求暂停。
- 升级方式：停止实现，更新 `harness/session-handoff.md`，列出证据、影响、可选方案和推荐。

## 本任务局部调整

当前任务是 `M2-sft-audit-and-repair`。执行顺序固定为：先审评测代码，再审数据和 case study，再审训练代码和 generation/checkpoint 行为；必要时对比 LLM4Game24 reference baseline。只有审计报告和修复证据足以说明问题已处理后，才启动新的 SFT 训练。最终训练跑起来之后可以结束会话。
