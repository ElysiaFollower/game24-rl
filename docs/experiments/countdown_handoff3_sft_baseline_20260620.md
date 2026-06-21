# Countdown Handoff3 SFT Baseline

## 目的

记录 handoff3 起步阶段的基线审计结果，为后续基于 handoff2 的 Countdown SFT 提供对照。

本阶段目标不是重新设计任务格式，而是在 handoff2 的 Game24 模型能力上做最小迁移：
把原来的 `24` 换成 Countdown 题目给定的 `target`，尽量不改变输入、搜索和输出语义。

## 模型起点

handoff3 的基座模型是 handoff2 交付的 GRPO 模型：

```text
base model: outputs/experiments/baseline_format_v2_full_5000_from800/final
LoRA adapter: outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500
HF mirror: https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500
```

这里的“基座模型”指 handoff3 实验起点，不是原始 `Qwen/Qwen2.5-1.5B-Instruct`。

## 数据口径

目标数据集：

```text
Jiayi-Pan/Countdown-Tasks-3to4
```

当前先使用 100 题 stratified eval manifest 做可控评估：

```text
outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json
```

构建口径：

| item | value |
| --- | --- |
| 数据来源 | `Jiayi-Pan/Countdown-Tasks-3to4` |
| 题目筛选 | 只保留 DFS 可解题 |
| 难度分层 | 按唯一可行表达式条数分为 `1/2/3/4` |
| 每层数量 | 25 |
| 总量 | 100 |
| 扫描规则 | 按数据集原始顺序流式扫描，四层凑满即停止 |

## 最小迁移 Prompt

原 Game24 prompt：

```text
<|im_start|>system
Play the 24-point game. Given four numbers, reach 24 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
1 6 9 12
<|im_end|>
<|im_start|>assistant
```

Countdown 最小迁移 prompt：

```text
<|im_start|>system
Play the 50-point game. Given four numbers, reach 50 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
74 5 20 88
<|im_end|>
<|im_start|>assistant
```

仓库实现：

```text
prompt_style = qwen_chat_minimal_target
```

关键约束：

| 项 | 口径 |
| --- | --- |
| system prompt | 只把 `24` 替换成题目 target |
| user prompt | 仍然只放裸数字 |
| 不使用 | `Numbers:` / `Target:` 新格式 |
| 不使用 | `Check:` completion |
| 输出契约 | 仍然是 `<think>...</think>` + 唯一 `<answer>...</answer>` |
| verifier | 检查表达式只使用输入数字，且值等于 target |

## Greedy Baseline

Artifact：

```text
outputs/experiments/handoff3_countdown_minimal/eval_handoff2_minimal_greedy16
```

配置：

| item | value |
| --- | --- |
| prompt_style | `qwen_chat_minimal_target` |
| split | `countdown_eval` |
| sample size | 16 |
| decoding | greedy |
| max_new_tokens | 4096 |
| batch_size | 4 |

结果：

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0.00% | 1 | 0 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 15 |
| `wrong_numbers` | 1 |

## Sampled Audit

先尝试较大审计：

```text
outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_32_g8
```

该审计为 `32 prompts x 8 samples x 4096 tokens`，运行 60 分钟仍无 summary。
GPU 持续生成，判断为当前模型在该 prompt 下容易打满长搜索，停止该审计。

随后改跑小样本审计：

```text
outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_8_g4
```

配置：

| item | value |
| --- | --- |
| prompt_style | `qwen_chat_minimal_target` |
| prompts | 8 |
| samples / prompt | 4 |
| total outputs | 32 |
| decoding | sampled |
| temperature | 0.8 |
| top_p | 0.95 |
| max_new_tokens | 4096 |

结果：

| prompts | total outputs | solved | pass@4 | format_ok | valid_expr | len mean | len p50 | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 0 | 0/8 | 3 | 0 | 4096 | 4096 | 4096 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 29 |
| `wrong_numbers` | 3 |

## 错误类型

### 长搜索不闭合

题目：

```text
target = 50
numbers = 74 5 20 88
```

模型输出恢复为 rollback/search，但完整 4096 token 没有输出 `<answer>`：

```text
(20) - (5) = 15, left: 15, 74, 88
(88) * (15) = 1260, left: 1260, 74
roll back, left: 15, 74, 88
...
roll back,
```

verifier：

```text
reason = answer_contract:expected exactly one <answer>...</answer> block
has_answer_open = false
has_answer_close = false
completion_tokens = 4096
```

### 24 点惯性

题目：

```text
target = 44
numbers = 26 4 2
```

greedy 输出中间找到过与 44 相关的状态，但最终仍提交 24 点答案：

```text
(26) - (4) = 22, left: 22, 2, 4
(2) * (22) = 44, left: 44, 4
...
(22) - (-2) = 24, left: 24
reach 24! expression: ((26 - 4) - (2 - 4))
</think>
<answer>((26 - 4) - (2 - 4))</answer>
```

verifier：

```text
value = 24
target = 44
reason = wrong_numbers
```

### 目标意识存在但答案回填错误

同一题在 sampled 输出中出现另一类错误：

```text
(26) - (4) = 22, left: 22, 2
(2) * (22) = 44, left: 44
reach 44! expression: (2 * ((26 - 4) - 2))
</think>
<answer>(2 * ((26 - 4) - 2))</answer>
```

verifier：

```text
expression = (2 * ((26 - 4) - 2))
value = 40
target = 44
used_numbers = [2, 2, 4, 26]
reason = wrong_numbers
```

## 结论

handoff2 模型在最小迁移 prompt 下不是完全失去搜索行为；它仍会输出与 Game24
一致的 rollback/search trace。但当前基线结果是：

| audit | result |
| --- | --- |
| greedy direct | `0/16` |
| sampled pass@4 | `0/8` |
| sampled solved outputs | `0/32` |
| main failure | 长搜索不闭合 |
| secondary failures | 24 点惯性、表达式回填/用数错误 |

因此当前不适合直接做 GRPO。原因不是 reward 难写，而是 sampled audit 没有观察到
可被 GRPO 放大的正确轨迹。

## 下一步 SFT 口径

下一步应先做最小迁移 SFT，目标是让模型学会“target 可变，但搜索过程和输出格式不变”。

SFT 数据应遵守：

| 项 | 口径 |
| --- | --- |
| 起点模型 | handoff2 GRPO adapter |
| prompt_style | `qwen_chat_minimal_target` |
| prompt 改动 | 只替换目标数字 |
| user 输入 | 裸数字 |
| completion | solver verified trace + `<answer>` |
| 不使用 | `Numbers:` / `Target:` |
| 不使用 | `Check:` |
| 训练题 | Countdown solvable train subset，排除 eval manifest |
| 评估 | 先跑同一 100 题 manifest，再做 sampled audit |

这一步不是为了重新教模型搜索，而是把已有搜索格式和任意 target 绑定起来，并修复
answer closure / 24 点惯性 / final expression 回填问题。

## Handoff3 Target-Replacement SFT 后结果

按上述口径完成一轮 SFT：

```text
outputs/experiments/handoff3_countdown_sft/sft_handoff2_target_replacement_20000_2400steps/final
```

训练口径：

| item | value |
| --- | --- |
| train records | 20000 |
| max_steps | 2400 |
| learning_rate | `1e-5` |
| max_length | 4096 |
| initial adapter | handoff2 GRPO checkpoint-500 |
| prompt | target 替换，user 裸数字 |
| completion | solver trace + `<answer>` |

Full100 greedy eval：

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 5 | 5.00% | 100 | 98 |

失败分布：

| reason | count |
| --- | ---: |
| `wrong_value` | 93 |
| `wrong_numbers` | 2 |
| `ok` | 5 |

Full100 sampled audit：

```text
outputs/experiments/handoff3_countdown_sft/rollout_sft_handoff2_target_replacement_20000_2400steps_full100_g8_4096
```

| prompts | samples / prompt | total outputs | solved outputs | pass@8 | mixed groups | all-wrong groups | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8 | 800 | 31 | 13/100 | 13 | 87 | 90.84 | 113 |

失败分布：

| reason | count |
| --- | ---: |
| `wrong_value` | 722 |
| `wrong_numbers` | 35 |
| `syntax_error:unmatched ')'` | 9 |
| `syntax_error:'(' was never closed` | 3 |
| `ok` | 31 |

结论：SFT 已经解决输出闭合和格式问题，但 full100 上 `pass@8=13%`、
mixed groups 只有 13 个，GRPO 信号偏弱。此时不适合直接长训 GRPO；更稳的下一步
是继续提高 SFT/rejection-sampled trace 质量，或先构建更大的 train-side sampled
audit，确认训练池是否有足够 mixed groups。

## Balanced Low-Solution SFT 后结果

前一轮 SFT 数据中 `5_plus` 容易题占比过高，而 full100 eval 是 `1/2/3/4`
solution buckets 各 25 题。为对齐评估分布，追加一轮 balanced low-solution SFT。

数据 artifact：

```text
outputs/experiments/handoff3_countdown_balanced_sft/data/countdown-balanced-low-solution-sft-20000.jsonl
```

数据构建口径：

| item | value |
| --- | --- |
| 数据来源 | `Jiayi-Pan/Countdown-Tasks-3to4` train |
| 排除 | 100 题 eval manifest |
| prompt_style | `qwen_chat_minimal_target` |
| trace_type | `short_success` |
| bucket quotas | `1=5000, 2=5000, 3=5000, 4=5000` |
| total records | 20000 |
| scanned rows | 147955 |

训练 artifact：

```text
outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final
```

训练口径：

| item | value |
| --- | --- |
| initial adapter | target-replacement SFT final |
| max_steps | 2400 |
| learning_rate | `1e-5` |
| max_length | 4096 |
| prompt | target 替换，user 裸数字 |
| completion | solver trace + `<answer>` |

Full100 greedy eval：

```text
outputs/experiments/handoff3_countdown_balanced_sft/eval_balanced_low_solution_20000_2400steps_full100_4096/countdown_eval-eval-report.json
```

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 23 | 23.00% | 100 | 98 |

失败分布：

| reason | count |
| --- | ---: |
| `wrong_value` | 75 |
| `wrong_numbers` | 2 |
| `ok` | 23 |

Full100 sampled audit：

```text
outputs/experiments/handoff3_countdown_balanced_sft/rollout_balanced_low_solution_20000_2400steps_full100_g8_4096/summary.json
```

| prompts | samples / prompt | total outputs | solved outputs | pass@8 | mixed groups | all-wrong groups | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8 | 800 | 134 | 45/100 | 42 | 55 | 88.84 | 112 |

失败分布：

| reason | count |
| --- | ---: |
| `wrong_value` | 638 |
| `wrong_numbers` | 25 |
| `syntax_error:unmatched ')'` | 2 |
| `unsupported_expression:UnaryOp` | 1 |
| `ok` | 134 |

对比前一轮 target-replacement SFT：

| metric | target-replacement SFT | balanced low-solution SFT |
| --- | ---: | ---: |
| greedy solve | `5/100` | `23/100` |
| sampled solved outputs | `31/800` | `134/800` |
| pass@8 | `13/100` | `45/100` |
| mixed groups | `13` | `42` |

结论：balanced low-solution SFT 明显有效，说明前一轮低分主要来自训练分布与
full100 评估分布不匹配。当前模型已经稳定闭合答案，主要失败从格式/闭合问题转为
`wrong_value`。`pass@8=45/100` 和 `mixed_groups=42` 表明 sampled distribution
里已经有较多正确轨迹，可以进入一轮保守 GRPO probe；但 greedy `23/100` 仍不能
作为最终结果。
