<!--
职责：描述本项目的持久目标、受众、边界和主要工作流。
边界：不要存放临时任务状态、实现日志、密钥，或应放在 docs/architecture/ 下的详细架构。
-->

# 项目概览

## 目标

训练并评估一个面向标准 24 点游戏的专用模型。项目以高分和可复现为优先：先通过严格 solver/verifier/evaluation 基座拿到可靠 SFT warm start，再用 GRPO 从该 checkpoint 继续提升。

## 受众

- 课程项目开发者：需要快速接手实现、训练、评估和报告。
- 课程评审者：需要看到清晰的 baseline、实验设置、指标和可复现证据。
- 后续研究者：需要能在 SFT/GRPO 基础上扩展数据、奖励、OOD 评测或消融。

## 范围内

- 标准 24 点：四个 `1..13` 的整数，目标值固定为 `24`。
- 精确枚举 solver、严格 AST + `Fraction` verifier、multiset-isolated split。
- `Qwen/Qwen2.5-1.5B-Instruct` 的 LoRA SFT 和后续 GRPO。
- R1 wrapper：`<think>...</think><answer>...</answer>`。
- 结果报告必须声明 model、split、answer contract、decoding、verifier version。

## 范围外

- 第一版不训练 Countdown 任意 target；只保留为 OOD 或后续扩展。
- 第一版不兼容 LLM4Game24 的 `expression:` 输出作为主契约。
- 不复制外部 baseline 代码；可以借鉴方法和对比分数。
- 不使用 Python `eval` 做 verifier 或 reward。
- 不把 SFT 调成无底洞；SFT 过关后进入 GRPO。

## 核心工作流

- M1：实现 solver、verifier、split/data manifest，并用测试锁定 1,820 / 1,362 / 458 等基准数字。
- M2：生成 first-pass SFT set，训练 LoRA SFT，达到 `solve_rate >= 70%` 作为可展示 fallback，`>= 80%` 作为强 fallback。
- M3：从 SFT checkpoint 启动 GRPO，监控 reward hacking、长度崩塌、KL、format rate、solve rate。
- M4：在强结果之后做数据规模、rollback trace、难例采样、OOD 等消融。
- 研究资料位于 `docs/research/`，baseline 说明位于 `docs/baselines.md`，稳定决策位于 `docs/adr/`。

## 验证

- 聚焦验证：`python -m compileall src scripts`；M1 起运行 `pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py`。
- 完整验证：`ruff check . && ruff format --check . && pytest`。
- 训练/评估验证：所有分数必须绑定 split manifest、checkpoint、decoding config、verifier version 和原始输出 artifact。
