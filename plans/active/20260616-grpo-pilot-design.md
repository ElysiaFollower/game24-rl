# Conservative GRPO Pilot Design

## 目标

基于 strong full fine-tuning SFT checkpoint
`outputs/experiments/baseline_format_v2_full_5000_from800/final` 设计下一阶段
RLVR + GRPO 方案，把当前 validation strict greedy solve rate
`110/136 = 80.88%` 推向 `90%+`。本任务的完成物是可执行设计和任务合同：
明确 reward、训练池、配置、监控、停止条件、实现步骤和验证门禁，让后续实现者可以直接接入 TRL
`GRPOTrainer` 或等价 GRPO 训练入口。

## 非目标

- 本任务不启动远端 GRPO 训练，不生成 checkpoint，不声称已达到 `90%+`。
- 不更换主模型 `Qwen/Qwen2.5-1.5B-Instruct`。
- 不改变 split 策略、主 answer contract 或 strict verifier 接受标准。
- 不用 Python `eval` 做 reward。
- 不为了抬分无限放大 generation budget；`max_new_tokens` 是报告口径的一部分。
- 不引入新重型训练框架作为唯一主路径；优先复用 Transformers/TRL/PEFT 栈。

## 当前仓库事实

- 当前强 SFT 结果：
  `docs/experiments/sft_full_finetune_search_trace_20260616.md`。
- 当前 SFT validation greedy：`110/136 = 80.88%`，剩余 `26` 个失败均为
  answer-contract/truncation，未输出 `<answer>`。
- GRPO rollout audit：
  `docs/experiments/grpo_rollout_audit_20260616.md`。
- Rollout evidence：validation pilot pass@4 `30/32`，mixed groups `16/32`；
  targeted greedy-failure pass@8 `22/26`，mixed groups `19/26`。
- 项目 verifier：`src/game24_rl/verifier.py`，版本
  `strict-ast-fraction-v1`。
- 项目评测入口：`scripts/eval_checkpoint.py` 和
  `scripts/experiments/run_rollback_sft_experiment.py --mode eval`。
- 训练栈 ADR：`docs/adr/0009-use-transformers-peft-trl-stack.md`。

## 允许改动

- 新增或更新 GRPO/RLVR 设计文档、任务计划、决策日志、progress、feature list 和 handoff。
- 设计后续实现所需的配置、reward schema、训练池生成逻辑、eval/monitoring artifact schema。
- 若本任务转入实现，可新增 `src/game24_rl/rewards.py`、`src/game24_rl/train_grpo.py`、
  `scripts/train_grpo.py`、GRPO config 和 focused tests。

## 禁止改动

- 不修改 strict verifier 接受标准。
- 不让同一个 puzzle multiset 的不同 trace 跨 split。
- 不把 external baseline 输出格式纳入主评估契约。
- 不提交 checkpoint、raw rollout、远端日志、模型缓存或生成训练数据。
- 不让 reward 主要由格式、长度或 `<think>` 风格驱动。

## 验收标准

- 设计文档说明为什么当前 SFT 已满足 GRPO 条件，以及为什么目标可以设为 `90%+`。
- 设计文档给出具体 reward v1、训练池验收、GRPO probe 配置、监控指标、停止/回滚门禁和实现步骤。
- 设计明确 `90%+` 的 validation 门槛：`123/136` 及以上。
- 设计引用当前 rollout audit、strong SFT 分析和外部 GRPO/RLVR 资料。
- Harness 状态切到 `M3-grpo-frontier`，且 WIP=1。
- 旧 M2 审计计划已归档，feature evidence 说明 M2 已以 strong SFT + rollout audit 收束。

## 关键锚点

配套检查文件：`plans/active/20260616-grpo-pilot-design.check.json`

- GRPO pilot 设计文档存在，并包含 `123/136`、reward、training pool audit、answer-close monitoring、stop gate。
- `M3-grpo-frontier` 是唯一 active feature。
- 决策日志记录进入 conservative GRPO pilot 的原因和后续约束。
- Progress 和 session handoff 都指向当前 active plan 和下一步实现动作。

## 验证命令

```sh
./scripts/harness-check.sh
python -m compileall src scripts
```

若本任务只更新设计和 harness，不强制跑完整 pytest；若随后进入代码实现，必须补 focused
tests 并按风险扩大到 `pytest`、`ruff check .` 和 `ruff format --check .`。

## Evidence 记录要求

验证通过后，将命令、结果、关键文档路径和设计摘要写入
`harness/feature_list.json` 的 `M3-grpo-frontier.evidence`。

## 完成定义

- 本计划和 `.check.json` 已创建。
- 设计文档可独立指导下一步实现和远端 pilot。
- Harness/feature/progress/handoff 与 active plan 一致。
- 验证命令已运行并记录 evidence。
- 清洁状态说明已更新。

## 阻塞条件

- 发现当前 TRL 版本不支持所需 GRPO config、`mask_truncated_completions`、reward function 额外列或 `remove_unused_columns=False` 时，先记录兼容性风险，不猜 API。
- 若后续实现需要改变 verifier、split、answer contract、主模型或 reported metric，停止并由负责人确认。
- 若远端训练环境缺少 TRL/Transformers 版本或显存不足，先做 dry-run/compat probe，不启动长训。

## 下一步最佳动作

实现最小 GRPO pilot：reward 函数、training-pool builder、answer-close 指标、
TRL `GRPOTrainer` dry-run/compatibility probe、focused reward tests；train pool
验收通过后，再在 AutoDL 上做 `beta=0/0.001`、`scale_rewards=none/group` 的 short probe。
