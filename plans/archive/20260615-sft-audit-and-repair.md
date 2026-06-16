# SFT 严格审计与修复

## 目标

在 SFT v2 中期验证仍低于预期后，建立并执行一轮严格的代码、数据、评测口径审计：先审评测代码，再抽样审数据，再审训练代码和生成链路，必要时对比 LLM4Game24 reference baseline，产出可复现审计报告，修复确认的问题，并在修复后启动一轮新的 SFT 训练。

## 非目标

- 不进入 GRPO，不改 reward 设计。
- 不更换主模型 `Qwen/Qwen2.5-1.5B-Instruct`。
- 不放宽 strict verifier、answer contract、split policy 来抬分。
- 不把外部 baseline 代码直接复制为本仓库主实现。
- 不在没有审计证据的情况下继续盲目调参或开多轮长训。

## 当前仓库事实

- 入口规则：`AGENTS.md`
- 初始化契约：`harness/bootstrap-contract.md`
- 当前功能项：`M2-sft-audit-and-repair`
- 相关文件/模块：`src/game24_rl/evaluate.py`、`src/game24_rl/data_gen.py`、`src/game24_rl/train_sft.py`、`src/game24_rl/verifier.py`、`scripts/eval_checkpoint.py`、`scripts/audit_sft_dataset.py`、`scripts/diagnose_sft_batch.py`、`scripts/diagnose_sft_overfit.py`、`docs/baselines.md`
- 已知约束：SFT 成功门槛原定 `solve_rate >= 70%`；当前 v2 `checkpoint-500` 为 21/136，`checkpoint-1500` 为 29/136，远低于预期；`checkpoint-1500` 的 format/valid expression 已达 100%，但 solve rate 未提升。

## 允许改动

- 审计并修复 evaluation、data generation、training、diagnostic scripts、tests 和相关文档。
- 增加 focused tests、诊断脚本、case-study artifact 和审计报告。
- 在负责人确认过的修复后，用 AutoDL 空闲 GPU 启动一轮新的 SFT 训练。
- 为对比 reference baseline，可读取、总结或临时拉取外部仓库；长期结论必须写回本仓库文档。

## 禁止改动

- 未经负责人确认，不改主模型、split 策略、answer contract、verifier 接受标准或 reported metric 口径。
- 未经负责人确认，不进入 GRPO、不扩展到 OOD/Countdown、不做大型架构迁移。
- 不提交训练 checkpoint、raw logs、模型缓存、数据缓存或外部仓库副本。
- 不用 Python `eval` 替代本仓库 strict verifier。

## 验收标准

- 评测代码审计完成：逐行检查 `evaluate.py` / `eval_checkpoint.py` 的 prompt、adapter load、generation slicing、decode、raw-output schema 和 strict-verifier scoring，结论写入审计报告。
- 数据审计完成：抽样检查训练 JSONL 的 prompt/completion/trace/answer、token 长度、目标表达式有效性、split 隔离和分布；至少包含若干成功/失败 case study。
- 训练代码审计完成：检查 TRL prompt-completion 处理、completion-only mask、EOS/pad、LoRA 保存加载、resume、checkpoint 和 teacher-forced loss；必要时用 tiny probe 复现。
- 若发现代码 bug，已补 focused test 或诊断门禁，并修复。
- 若更可能是设计/数据策略问题，报告必须明确说明证据，不把它伪装成代码 bug。
- 修复后，在 AutoDL 启动一轮新的 SFT 训练，并记录 run name、config、checkpoint/eval artifact 路径。

## 关键锚点

配套检查文件：`plans/active/20260615-sft-audit-and-repair.check.json`

- 审计报告：证明评测、数据、训练和 baseline 对比不是只存在聊天里。
- Case study artifact：证明我们理解模型失败样本和训练样本，而不是只看 aggregate metric。
- Focused tests/diagnostics：证明修复的问题不会再次 silent 复发。
- Feature evidence：证明验证命令、报告路径和复训路径已写入长期状态。

## 验证命令

```sh
./scripts/harness-check.sh
python -m compileall src scripts
pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py tests/test_training_pipeline.py
ruff check .
ruff format --check .
# 手动/远端：运行审计脚本、case-study 脚本、修复后的 checkpoint validation eval，并记录 artifact 路径。
```

## Evidence 记录要求

验证通过后，将命令、结果、关键输出摘要或 artifact 路径写入 `harness/feature_list.json` 的 `evidence`。

## 完成定义

- 请求行为已实现。
- 非目标没有被触碰。
- 关键锚点已满足；若锚点因方案变化不再合理，已先更新 active plan 和 `.check.json` 并记录原因。
- 上方验证命令已运行；未运行的命令必须说明原因。
- `harness/feature_list.json` 状态和 evidence 已更新。
- 职责、接口、setup 或边界改变时，docs、注释、测试或 harness 文件已更新。
- `harness/session-handoff.md` 写明当前状态、风险和下一步。
- 清洁状态检查已说明。

## 阻塞条件

- SFT v2 final 仍在运行且需要避免抢占资源时，只能做只读本地审计或等待。
- 外部 baseline 仓库无法访问且其实现细节成为判断关键时，先记录 blocker，不猜。
- 发现需要改变 verifier/split/answer contract/主模型/训练路线时，停止并交给负责人确认。
- 无法建立可复现 case study 或 focused test 时，先报告测试缝隙，不继续声称已修复。

## 下一步最佳动作

1. 只读审计 `src/game24_rl/evaluate.py` 和 `scripts/eval_checkpoint.py`，逐项核对 prompt、generation、decode、adapter、scoring 和 artifact schema。
