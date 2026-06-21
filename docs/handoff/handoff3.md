# 给队友的简短交接：Countdown 迁移实验

一句话版：

> handoff3 做的是加分项：把 handoff2 的 24 点 GRPO 模型迁移到
> `Jiayi-Pan/Countdown-Tasks-3to4`，也就是“给定 3-4 个数，凑任意 target”。
> 我们用最小迁移 prompt、SFT、balanced SFT 和 GRPO 都试过了。最好 direct greedy
> 结果是 `26/100 = 26%`。这个结果不够强，但实验链路和失败原因是清楚的：
> 格式基本学会了，主要错在表达式值不等于 target。

## 这次实验接在 handoff2 后面

handoff2 交付的是 24 点 GRPO 模型：

```text
outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500
```

Hugging Face 留档：

```text
https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500
```

handoff3 没有从原始 Qwen 重新训练，而是从这个 handoff2 模型出发，测试它能不能迁移到
Countdown 任意 target 任务。

## 数据和评估口径

目标数据集：

```text
Jiayi-Pan/Countdown-Tasks-3to4
```

这个数据集很大，所以本阶段没有全量评估，而是构建了一个 100 题的可复现 held-out
评估集：

```text
outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json
```

构建逻辑：

- 只保留 DFS 可解题；
- 按唯一可行表达式数量分成 `1/2/3/4` 四组；
- 每组 25 题，总计 100 题；
- 训练数据排除这 100 题；
- 评估使用 direct greedy，`max_new_tokens=4096`；
- strict verifier 检查 `<answer>...</answer>` 中的表达式是否只用输入数字并等于 target。

这个 100 题集合偏难，因为它刻意选择低解数题；所以结果不能直接理解成 Countdown
全量随机题准确率。

## Prompt 选择

最关键的工程判断是：不能换成全新的 `Numbers:` / `Target:` 输入格式。

原 24 点 prompt 类似：

```text
<|im_start|>system
Play the 24-point game. Given four numbers, reach 24 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
1 6 9 12
<|im_end|>
<|im_start|>assistant
```

Countdown 最小迁移 prompt 只替换目标数字，user 段仍然只放裸数字：

```text
<|im_start|>system
Play the 50-point game. Given four numbers, reach 50 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
74 5 20 88
<|im_end|>
<|im_start|>assistant
```

仓库里的 prompt style 是：

```text
qwen_chat_minimal_target
```

旧的 `qwen_chat_target` 和 `Check:` completion 路线改变了输入/输出语义，已经废弃，只作为失败记录保留。

## 实验路线和结果

核心路线是：

```text
handoff2 GRPO -> minimal-transfer eval -> target-replacement SFT
-> balanced low-solution SFT -> target-distance GRPO
```

结果表：

| stage | full100 greedy | sampled audit | 主要结论 |
| --- | ---: | ---: | --- |
| handoff2 zero-shot minimal transfer | `0/16` | `pass@4=0/8` | 会长搜索，但不闭合答案 |
| target-replacement SFT | `5/100` | `pass@8=13/100` | 格式修复，求解仍弱 |
| balanced low-solution SFT | `23/100` | `pass@8=45/100` | 当前最有效的 SFT 路线 |
| target-distance GRPO | `26/100` | `pass@8=46/100` | 最好 direct greedy 结果 |
| final broad-solvable SFT | `19/100` | `pass@8=43/100` | 扩大普通可解题覆盖反而退化 |

最终可报告的最好 direct greedy 结果是：

```text
26/100 = 26%
```

对应 checkpoint：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/grpo_targetdistance_from_balanced_sft_train256_g8_lr1e6_beta001_1200/final
```

对应结果文档：

```text
docs/experiments/countdown_handoff3_grpo_20260620.md
```

## 怎么解释这个结果

这条迁移没有达到我们一开始希望的效果。比较稳的解释是：

> handoff2 的 24 点模型确实学到了搜索格式和一部分算式组合能力，但“固定 target=24”
> 到“任意 target”的迁移并不只是替换一个数字。模型很容易生成格式正确、数字也基本正确、
> 但表达式值不等于 target 的答案。

target-replacement SFT 之后，格式问题基本解决：

- `format_ok` 基本达到 `100/100`；
- 输出能稳定闭合 `<answer>`；
- 主要失败从 no-answer 变成 `wrong_value`。

这说明训练不是完全无效。问题从“不会按格式完成任务”推进到了“会提交错误值的表达式”。

## 典型错误

题目：

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

verifier：

```text
reason = wrong_value
value = 39
target = 50
```

也就是说，它不是没有输出，也不是 verifier 错了，而是模型把一个不等于 target 的表达式当答案提交了。

## 为什么最后不继续训了

我们最后还试了一次 broad-solvable SFT：用 20000 道唯一可解 Countdown 题继续训练。
结果从 `26/100` 退到 `19/100`。

这说明简单增加普通可解题覆盖不是有效方向。当前 held-out full100 是低解数难题，
训练分布一旦变宽，反而会稀释这类题的信号。

所以 handoff3 最合理的交付说法是：

> Countdown 迁移作为加分项完成了系统实验，但结果有限。最好 direct greedy 准确率为
> `26/100 = 26%`。失败主要是 `wrong_value`，不是格式崩溃或答案不闭合。后续若继续做，
> 应重新设计 target-aware 数据和 verifier-guided 训练信号，而不是继续盲目扩大 SFT 或 GRPO 步数。

## 留档位置

建议阅读顺序：

1. `docs/handoff/handoff2.md`
   先看 handoff3 的起点模型是什么。

2. `docs/experiments/countdown_handoff3_sft_baseline_20260620.md`
   看 minimal-transfer prompt、SFT 和 balanced SFT 的结果。

3. `docs/experiments/countdown_handoff3_grpo_20260620.md`
   看 target-distance GRPO、final broad-solvable SFT 和最终结论。

4. `docs/experiments/countdown_transfer_eval_20260619.md`
   只作为旧 prompt 路线的失败记录，不作为当前主结论。
