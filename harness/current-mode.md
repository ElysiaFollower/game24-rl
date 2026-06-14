<!--
职责：记录目标项目当前生效的 agent 协作模式和开发原则。
边界：不要把历史模式档案、聊天记录、长期项目事实或 profile 来源追踪放在这里。
-->

# Current Harness Mode

<!-- mode-schema: current-harness-mode/v1 -->

## 当前策略摘要

- 生效范围：M2 真实训练准备、远程/AutoDL 实验运行和后续 GRPO 初始阶段
- Collaboration mode：强人类把关，小步授权执行
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行
- 生效时间：2026-06-14
- 切换原因：初始实验阶段需要负责人亲自监工关键判断，避免训练、环境和评测决策浪费 GPU 时间或污染可报告证据。

## Collaboration Mode

Agent 只执行负责人明确授权的下一步。每一步完成后给出事实、证据、未决问题、选项和推荐，不自动跨到下一阶段。遇到设计分叉、训练/评测口径变化、验证不可用、实验假设冲突、远程资源风险或需要扩大范围时，先停止并交给负责人确认。

## Development Style Principles

- 研究原型要可复现：数据、split、checkpoint、decoding、verifier version 都要能回溯。
- Python 代码遵循轻量 Google Python Style：公共函数/类用 Google-style docstring，命名清晰，少全局状态。
- solver、verifier、data generation、evaluation 是基础设施，优先写可测试的小函数。
- 训练脚本可以薄，但评估和 verifier 不能薄弱；分数必须来自严格 verifier。
- 允许工程取舍，但核心不变量要进测试或 ADR。

## 授权边界

- Agent 可以在明确授权的一步内执行：读取项目事实源、运行指定或必要的轻量验证、做已确认范围内的小改动、收集实验证据、整理选项和推荐。
- 未经负责人确认，不启动真实长训练、不改训练路线、不改主模型、不改 answer contract、不改 split 策略、不改 verifier/reward 口径、不扩大到 Countdown/OOD 训练、不提交或推送仓库。
- 必须先问负责人：任何会改变架构方向、任务边界、验证标准、协作模式、开发原则、实验结论口径、远程运行方式、资源占用策略或可报告分数解释的决定。
- 必须停止并升级：发现 train/validation/test multiset 泄漏；verifier 规则会错误接受非法答案；需要使用 Python `eval` 才能推进；课程指定模型不可用；训练环境无法验证；SFT v1 低于 50% 且已排除常见 pipeline bug；GRPO 指标显示 reward hacking、长度崩塌或 zero-std 过高但原因不明。

## 验证策略

- 最小验证：`./scripts/harness-check.sh`；如果触碰 Python 路径，再运行 `python -m compileall src scripts` 和相关测试。
- 风险升高时追加：`pytest` 的相关子集、`ruff check .`、`ruff format --check .`、tiny fixture 训练/评估 dry run、真实模型最小闭环 smoke。
- 不能作为完成证据：训练 loss 单独下降、模型自我判断、非严格 verifier、Python `eval` 结果、存在 split 泄漏的分数、未声明 decoding 的分数。
- 实验分数必须绑定 split manifest、checkpoint、decoding config、answer contract、verifier version 和 raw output artifact；缺任一项则只能算未验证观察。

## 阻塞和切换规则

- 阻塞条件：无法复现 1,820/1,362/458 基准数字；严格 verifier 与标准 24 点规则冲突；训练服务器缺少可用 CUDA/PyTorch；课程硬约束与 ADR 冲突；负责人要求暂停；下一步需要负责人授权但授权尚未给出。
- 允许切换模式的时机：新任务开始、阶段结束、验证失败暴露方法问题、用户要求改变参与深度。
- 切换方式：覆盖本文件，并在 `harness/progress.md` 记录原因。

## 本阶段局部调整

当前处于 M2 first-pass SFT 的真实训练准备阶段。Agent 的职责是实验把关和运行协助：先验证环境、模型加载、数据、checkpoint、resume、eval artifact 和 strict verifier 闭环，再在负责人授权后放大训练。GRPO 只在 SFT 过成功门槛后进入；进入前必须再次确认 reward、日志指标和停止条件。
