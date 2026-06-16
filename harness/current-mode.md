<!--
职责：记录目标项目当前生效的 agent 协作模式和开发原则。
边界：不要把历史模式档案、聊天记录、长期项目事实或 profile 来源追踪放在这里。
-->

# Current Harness Mode

<!-- mode-schema: current-harness-mode/v1 -->

## 当前策略摘要

- 生效范围：`M3-grpo-frontier` conservative GRPO pilot 设计、实现和验证。
- Collaboration mode：高自治执行，关键边界升级。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 生效时间：2026-06-16。
- 切换原因：strong full fine-tuning SFT 已达到 `110/136 = 80.88%`，rollout audit 证明存在可用 group-level reward variance；负责人明确要求进入 RLVR + GRPO 后训练设计，以 `90%+` 为下一目标。

## Collaboration Mode

Agent 先读取仓库事实，再直接推进当前 active plan。保持 WIP=1，不开启无关任务；遇到验证失败先诊断和修复，不因一次失败就停止。每个完成声明必须带验证证据、artifact 路径和清洁状态说明。

只在需要改变目标边界、外部接口契约、持久化语义、主模型、split 策略、answer contract、verifier 接受标准，或需要扩大到 active plan 明确排除的非目标时停止并等待负责人确认。进入 conservative GRPO pilot 已由负责人在本阶段授权。

## Development Style Principles

- 研究原型要可复现：数据、split、checkpoint、decoding、verifier version、raw output 和报告路径都要能回溯。
- 先建立反馈环再修复：评测口径、数据样本、训练 mask、checkpoint/load 和 generation 行为必须有可复现证据。
- Python 代码遵循轻量 Google Python Style：公共函数/类用 Google-style docstring，命名清晰，少全局状态。
- solver、verifier、data generation、evaluation 是基础设施，优先写可测试的小函数。
- 训练脚本可以薄，但评估和 verifier 不能薄弱；分数必须来自 strict verifier。
- 允许局部工程取舍，但核心不变量要进测试、诊断脚本、审计报告或 ADR。

## 授权边界

- Agent 可以自主执行：读取项目事实源、审计代码/数据/报告、运行本地测试、运行远端只读诊断、设计并实现 conservative GRPO pilot、补 focused tests、更新 docs/harness、同步代码、在本地/远端先跑 dry-run 和 compatibility probe。
- Agent 可以自主对比 GRPO/RLVR 公开论文、官方文档和 reference baseline，但不能复制外部代码为主实现，不能用 baseline 的 `eval` 路径替代本仓库 verifier。
- 未经负责人确认，不改主模型、不改 split 策略、不改 answer contract、不放宽 verifier/reward 口径、不扩展到 Countdown/OOD、不引入新重型训练框架作为唯一主路径。
- 真实长 GRPO 训练前必须先有本地 dry-run、reward tests、train-pool evidence 和远端兼容性 probe；发现必须改变上述边界才能推进时，停止并在 handoff 中记录证据、选项和推荐。

## 验证策略

- 最小验证：`./scripts/harness-check.sh`。
- Python 改动：`python -m compileall src scripts` 和相关 focused tests。
- 基础链路：`pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py tests/test_training_pipeline.py`；GRPO 实现后追加 reward/pool/training dry-run focused tests。
- 质量门禁：`ruff check .` 和 `ruff format --check .`，如因环境不可用需记录原因。
- 远端实验：每个分数必须声明 model、initial checkpoint、GRPO checkpoint、split manifest、reward version、TRL/config version、decoding config、answer contract、verifier version 和 raw output artifact。
- 不能作为完成证据：reward_mean 单独上升、训练 loss 单独下降、模型自我判断、非 strict verifier、Python `eval`、存在 split 泄漏的分数、未声明 decoding/generation budget 的分数。

## 阻塞和升级规则

- 阻塞条件：无法复现核心数据数字；strict verifier 与标准 24 点规则冲突；训练服务器不可用；TRL/GRPO API 与设计关键需求不兼容；下一步必须改变主模型/split/answer contract/verifier；负责人要求暂停。
- 升级方式：停止实现，更新 `harness/session-handoff.md`，列出证据、影响、可选方案和推荐。

## 本任务局部调整

当前任务是 `M3-grpo-frontier` 的 conservative pilot 设计与后续最小实现。第一步只把方案落成可执行合同：目标 `90%+` 等价于 validation 至少 `123/136`，reward 以 strict verifier correctness 为主，训练池优先 active-difficulty/mixed-reward prompts，监控 greedy solve rate、pass@k、completion length、truncation rate、reward std、zero-std group rate 和 KL。实现阶段必须先 dry-run 和 compatibility probe，再启动短 pilot。
