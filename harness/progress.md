<!--
职责：记录跨会话进度、状态变化、阻塞和验证摘要，让新会话能快速恢复。
边界：不要存放聊天记录、原始日志、密钥，或更适合由代码、测试、ADR、任务计划表达的内容。
-->

# 进度日志

## 当前状态

- 当前功能项：`M2-sft-audit-and-repair`
- 当前任务计划：`plans/active/20260615-sft-audit-and-repair.md`
- 当前模式：高自治执行；最终训练跑起来后可结束会话。
- 背景：SFT v2 修复 prompt/completion 边界和 stale JSONL 后，final validation 仍只有 `42/136 = 30.88%`，format/valid expression 为 100%，因此继续作为低分审计与修复任务处理。
- 下一步最佳动作：同步当前修复到 AutoDL，启动 `configs/sft_v3_checked_chat.yaml` 训练，并观察前几条日志确认正常运行。

## 状态约定

- `not_started`：尚未开始。
- `active`：当前唯一在制任务。
- `blocked`：缺少输入、环境、依赖或决策。
- `passing`：验证通过且 evidence 已记录。

## 近期证据

### 2026-06-15 - SFT 审计、修复和 v3 复训准备

- 评测代码审计结论：strict verifier/aggregate scoring 未发现明显算分 bug；主要失败是模型输出 correct-format/correct-numbers 但 wrong-value。
- SFT v2 final validation：`outputs/eval/sft_v2_fixed_prompt_final_validation_20260615-010152/validation-eval-report.json`，solve_rate=`42/136=30.88%`，format_rate=`1.0`，valid_expr_rate=`1.0`。
- Case study artifact：`outputs/diagnostics/sft_case_study/summary.json`，checkpoint-1500 为 `29` ok / `107` wrong_value，train/validation overlap count 为 `0`。
- Baseline 对比：临时只读克隆 `https://github.com/LiaoMengqi/LLM4Game24` 到 `/tmp/LLM4Game24`；其强 format-v2 数据约 45k 条、使用 Qwen chat framing、显式最终 target statement 和部分 rollback/search traces。
- 修复：`trace_type` 和 `prompt_style` 从 config/CLI 贯通到 data generation、training dry-run 和 eval report；新增 `configs/sft_v3_checked_chat.yaml`。
- v3 dry-run：`python scripts/train_sft.py --config configs/sft_v3_checked_chat.yaml --dry-run` 通过；生成 `11706` 条 strict-verifier-valid checked/chat SFT records。
- 验证：`./scripts/harness-check.sh`、`python -m compileall src scripts`、focused pytest 38 tests、`pytest` 38 tests、`ruff check src scripts tests`、`ruff format --check src scripts tests configs` 均通过。
- 审计报告：`docs/experiments/sft_audit_report.md`。

### 2026-06-15 - SFT 低分后切换到严格审计任务

- 旧 active plan `0002-sft-training-readiness` 已归档到 `plans/archive/`。
- 新 active plan：`plans/active/20260615-sft-audit-and-repair.md`。
- 新 feature：`M2-sft-audit-and-repair`，唯一 active。
- `M2-first-pass-sft` 标记为 blocked：SFT v2 `checkpoint-500` solve_rate=21/136，`checkpoint-1500` solve_rate=29/136，final solve_rate=42/136。

### 2026-06-14/15 - AutoDL GitHub 网络经验

- AutoDL 直连 `https://github.com` 超时；通过本地 `127.0.0.1:2080` 代理的 SSH 反向转发可通。
- 经验文档：`docs/experiments/remote-git-sync-via-proxy.md`。

## 历史摘要

- M1 solver/verifier/data split foundation 已 passing；验收数字为 1,820 total / 1,362 solvable / 458 unsolvable。
- M2 training readiness 已实现并归档：配置解析、dry-run、checkpoint resume、strict eval artifacts、Miniconda/bootstrap、AutoDL 环境验证。
- 当前不得进入 GRPO；GRPO 仍等待 SFT warm start 有可信高分后再开始。

### 2026-06-15 - SFT v3 remote training started

- Commit synced to AutoDL: `372e17a`.
- Remote dirty worktree before sync was preserved in stash `autodl-before-sft-v3-sync-20260615-011851`.
- AutoDL validation before launch: `./scripts/harness-check.sh` passed with expected warning for ignored case-study artifact absence; `python -m compileall src scripts` passed; `pytest tests/test_data_gen.py tests/test_training_pipeline.py` passed with 17 tests; v3 dry-run generated `11706/11706` strict-verifier-valid records.
- Training tmux session: `game24-sft-v3-checked-chat`.
- Run dir: `outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat`.
- Log file: `outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat/logs/train-20260615-011937.log`.
- Observed normal progress: tokenization completed, trainer reached about `57/4392` steps, `checkpoint-50` exists, GPU active around `6065 MiB / 49140 MiB` and `30%` utilization.
