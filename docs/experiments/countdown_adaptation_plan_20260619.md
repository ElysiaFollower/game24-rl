# Countdown Adaptation 轻量计划

## 目标

在 handoff2 GRPO checkpoint-500 基础上，做 `Jiayi-Pan/Countdown-Tasks-3to4`
的轻量任务迁移：

```text
给定 3-4 个数字 + target -> 用给定数字各一次，通过 + - * / 拼出 target
```

目标不是重新学习搜索，而是让模型学会 Countdown 输入语义：

- `Numbers:` 是唯一可用数字；
- `Target:` 是目标值，不能当数字使用；
- 最终 `<answer>` 表达式必须等于 target；
- 不再默认凑 24。

## 当前证据

handoff2 模型 zero-shot eval 已显示明显迁移失败：

- 模型仍会进行 rollback/search；
- 但经常把 target 混入搜索状态；
- 仍有强烈“凑 24”惯性；
- 100 题 stratified eval 上 `0/100`；
- 400 条 / 200 step SFT 后，partial eval 前 8 条只有 `1/8` 正确；
- 2000 条 / 600 step SFT 后，格式恢复，但 smoke eval 仍只有 `1/16` 正确；
- 2000 条 / 600 step SFT 后，16 题 sampled audit 的 pass@8 只有 `1/16`。

这说明当前 SFT 已能修复输出契约，但模型尚未可靠学会任意 target 搜索。
直接 GRPO 的正样本密度过低，不适合盲目长训。

## 原则

时间紧，不做笨重 pipeline。尽量复用现有代码：

- prompt：复用 `qwen_chat_target`；
- solver：复用 `game24_rl.countdown_solver`；
- verifier：复用现有 strict verifier，已支持 3-4 数字任意 target；
- eval：复用 `scripts/eval_checkpoint.py`；
- GRPO：如需要，复用 `scripts/train_grpo.py`；
- 数据：复用 AutoDL 已下载的 `Jiayi-Pan/Countdown-Tasks-3to4`。

## 阶段 1：构建轻量 SFT 数据

从 Countdown 数据集中流式扫描 solvable 题。

建议规模：

```text
200-400 train prompts
```

构建方式：

- DFS 搜索每题的一个正确表达式；
- 生成与 Game24 一致的 completion：

```text
<think>
...
</think>
<answer>...</answer>
```

- prompt 使用 `qwen_chat_target`：

```text
Numbers: ...
Target: ...
```

数据可以按 DFS 解路径数做轻量分层，但不要为了分层复杂化流程。

## 阶段 2：短 SFT warm start

从 handoff2 GRPO checkpoint-500 出发做短 SFT。

建议初始配置：

```text
training_mode = LoRA / adapter continuation if feasible
max_steps = 100-300
max_seq_length = 4096
prompt_style = qwen_chat_target
```

目的只解决任务格式迁移，不追求长训。

验收：

- 跑同一个 100 题 Countdown stratified eval；
- 指标至少应明显高于 zero-shot 的 `0/100`；
- 重点看 wrong_numbers 是否下降，尤其 target 被当数字的问题是否消失；
- 如果 format / target-following 已明显恢复，可以先把 SFT 结果作为 handoff3 加分项结果。

## 阶段 3：必要时再 GRPO

只有当短 SFT 后仍有明显错误、且模型能采样到正确答案时，再做短 GRPO。

建议初始配置：

```text
train prompts = 100-300 solvable Countdown prompts
prompt_style = qwen_chat_target
num_generations = 4-8
max_completion_length = 2048 or 4096
max_steps = 100-300
reward = strict verifier correctness
```

不要做复杂 gate。当前目标是快速拿到可展示的加分项结果。

GRPO reward 重点：

- correct target：强正奖励；
- wrong_numbers：强负奖励；
- wrong_value：强负奖励；
- no-answer / answer contract failure：负奖励；
- 暂时不增加复杂长度 shaping，避免偏离主目标。

当前执行状态：暂不启动 GRPO。原因是 sampled audit 只有 `1/16` prompts
在 8 次采样中出现正确答案，15 个 prompt 是 all-wrong group。GRPO 在这种
pool 上主要会得到零方差信号，训练效率和结果都不可控。

如果继续推进，优先路线应改为 curriculum SFT：

- 先构建更容易的 Countdown 子集，例如 `5_plus` solution bucket；
- 使用 3-number 和高解数题做第一阶段；
- 再逐步加入 `4/3/2/1` solution bucket；
- 每个阶段用 balanced smoke eval 检查是否真的提升搜索命中，而不是只提升格式。

## 输出物

本阶段最终需要：

- Countdown eval manifest；
- zero-shot eval 结果；
- 如果执行 SFT：SFT checkpoint、eval 结果；
- 如果执行 GRPO：GRPO checkpoint、eval 结果；
- `docs/handoff/handoff3.md` 中只写最终交付结论；
- 过程记录继续写在 `docs/experiments/countdown_transfer_eval_20260619.md`
  或新的实验记录文件中。

## 推荐执行顺序

```text
1. 固化 zero-shot eval 结果
2. 构建 200-400 条 solver SFT 数据
3. 跑短 SFT
4. 评估同一个 100 题 stratified eval
5. 若仍明显不足，再做短 GRPO
6. 整理 handoff3 交付结果
```
