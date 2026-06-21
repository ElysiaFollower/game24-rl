# 给队友的简短交接：Countdown 迁移实验

## 结论

handoff3 做的是加分项：把 handoff2 的 24 点 GRPO 模型迁移到
`Jiayi-Pan/Countdown-Tasks-3to4`，尝试求解“给定 3-4 个数，凑任意 target”的任务。

最终最好结果是：

```text
held-out full100 direct greedy accuracy = 26/100 = 26%
```

这个结果不够强，不能包装成成功迁移。但它不是无效实验：我们已经确认模型能学会
Countdown 的输出格式，主要失败不是格式崩溃，而是表达式值不等于 target。

## 实验起点

handoff3 从 handoff2 交付的 24 点 GRPO 模型出发：

```text
outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500
```

Hugging Face 留档：

```text
https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500
```

这不是从原始 Qwen 重新训练，而是在 handoff2 的 24 点模型上做迁移。

## 评估口径

目标数据集：

```text
Jiayi-Pan/Countdown-Tasks-3to4
```

由于数据集很大，本阶段构建了一个 100 题 held-out 小评估集：

```text
outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json
```

构建规则：

- 只保留 DFS 可解题；
- 按唯一可行表达式数量分为 `1/2/3/4` 四组；
- 每组 25 题，共 100 题；
- 训练数据排除这 100 题；
- 评估用 direct greedy，`max_new_tokens=4096`；
- strict verifier 检查 `<answer>...</answer>` 内表达式是否只使用输入数字且等于 target。

注意：这个 full100 评估集刻意选择低解数题，偏难；它不是 Countdown 全量随机准确率。

## Prompt 口径

我们采用最小迁移 prompt：不引入新的 `Numbers:` / `Target:` 格式，只把原 24 点 prompt
里的目标数字换成 Countdown 的 target，user 段仍只放裸数字。

原 24 点 prompt：

```text
Play the 24-point game. Given four numbers, reach 24 ...
user: 1 6 9 12
```

Countdown prompt：

```text
Play the 50-point game. Given four numbers, reach 50 ...
user: 74 5 20 88
```

仓库实现：

```text
prompt_style = qwen_chat_minimal_target
```

早期的 `qwen_chat_target` / `Check:` completion 路线改变了输入和输出语义，已废弃，
只保留为失败记录。

## 实验路线

核心路线：

```text
handoff2 GRPO
-> minimal-transfer zero-shot audit
-> target-replacement SFT
-> balanced low-solution SFT
-> target-distance GRPO
```

另外还做了一次 final broad-solvable SFT，用来验证“扩大普通可解题覆盖”是否能继续提升。
结果是退化，因此不作为最终模型。

## 结果表

| stage | full100 greedy | sampled audit | 结论 |
| --- | ---: | ---: | --- |
| handoff2 zero-shot minimal transfer | `0/16` | `pass@4=0/8` | 会长搜索，但不能稳定完成 Countdown |
| target-replacement SFT | `5/100` | `pass@8=13/100` | 格式基本修复，求解很弱 |
| balanced low-solution SFT | `23/100` | `pass@8=45/100` | 最有效的 SFT 路线 |
| target-distance GRPO | `26/100` | `pass@8=46/100` | 最好 direct greedy 结果 |
| final broad-solvable SFT | `19/100` | `pass@8=43/100` | 扩大普通可解题覆盖反而退化 |

最终采用的最好结果：

```text
model:
outputs/experiments/handoff3_countdown_balanced_grpo_probe/grpo_targetdistance_from_balanced_sft_train256_g8_lr1e6_beta001_1200/final

accuracy:
26/100 = 26%
```

## 失败类型

SFT 之后格式问题基本被解决：

- `format_ok` 基本是 `100/100`；
- 模型能稳定输出 `<answer>...</answer>`；
- 主要失败变成 `wrong_value`。

典型错误：

```text
target = 50
numbers = 74 5 20 88
```

模型输出：

```text
<think>
(74) - (5) = 69, left: 69, 20, 88
(20) - (69) = -49, left: -49, 88
(-49) + (88) = 39, left: 39
</think>
<answer>((20 - (74 - 5)) + 88)</answer>
```

strict verifier：

```text
reason = wrong_value
value = 39
target = 50
```

也就是说，模型不是没输出，也不是 verifier 算错，而是把一个值不等于 target 的表达式提交成答案。

## 为什么停止继续训练

最后一次 broad-solvable SFT 用 20000 道唯一可解 Countdown 题继续训练，但结果从
`26/100` 退到 `19/100`。这说明当前问题不是“普通可解题覆盖不够”这么简单。

当前 full100 是低解数难题；继续混入大量普通 solvable 题，会稀释低解数题的训练信号。
因此继续盲目扩大 SFT 或 GRPO 步数不值得。

## 可交付说法

报告里可以这样写：

> 作为加分项，我们尝试将 24 点模型迁移到 Countdown 任意 target 任务。我们采用最小迁移
> prompt，只把目标值从 24 替换为题目 target，并保持原有 `<think>` / `<answer>` 输出契约。
> 在 100 道低解数 held-out 可解题上，zero-shot 不能直接完成任务；经过 target-replacement
> SFT、balanced low-solution SFT 和 target-distance GRPO 后，最好 direct greedy 准确率为
> `26/100 = 26%`。失败主要来自 `wrong_value`，说明模型已经基本掌握输出格式，但任意 target
> 下的表达式值校验和目标对齐仍不足。

## 留档

- SFT / balanced SFT 过程：`docs/experiments/countdown_handoff3_sft_baseline_20260620.md`
- GRPO / final broad-solvable SFT：`docs/experiments/countdown_handoff3_grpo_20260620.md`
- 旧 prompt 失败记录：`docs/experiments/countdown_transfer_eval_20260619.md`
