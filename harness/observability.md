<!--
职责：定义本项目 harness 的运行时信号、过程工件和验证证据采集方式。
边界：不要存放完整日志；日志应由工具产生，本文只说明采集与解释规则。
-->

# 可观测性

## 运行时信号

- 应用启动/就绪：无服务；`./init.sh` 和 `./scripts/harness-check.sh` 能运行即表示开发入口可用。
- 关键用户路径：split 生成、SFT 数据生成、checkpoint 评估、GRPO reward/eval 日志。
- 数据/副作用检查：manifest 中 split 名称、seed、puzzle multiset、trace count、checkpoint 路径、verifier version 必须可回溯。
- 错误上下文：solver/verifier 失败要输出 puzzle、answer text、解析阶段、失败原因；训练/评估失败要输出 config path、checkpoint、split、decoding。

## 过程工件

- 任务合同：`plans/active/`
- 功能状态：`harness/feature_list.json`
- 验证证据：feature item 的 `evidence`
- 会话交接：`harness/session-handoff.md`
- 质量评估：`harness/evaluator-rubric.md` 和 `harness/quality.md`
- 训练 artifact：`outputs/`、`runs/`、`wandb/`、`checkpoints/` 默认不提交，只在报告或 evidence 中记录路径和摘要。

## 面向 agent 的错误消息规则

验证失败时，错误消息应说明：

- 哪个命令失败；
- 失败的可观察症状；
- 最可能的检查位置；
- 下一步修复建议。

不要只写 “test failed”。
