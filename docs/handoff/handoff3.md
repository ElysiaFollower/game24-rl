# Handoff3 Start Point: Countdown Transfer

## 当前目标

handoff3 阶段只做一件事：

> 基于 handoff2 交付的 GRPO 模型，在
> `Jiayi-Pan/Countdown-Tasks-3to4` 上做“给定 3-4 个数，凑任意 target”的迁移实验。

如果 zero-shot / direct eval 不够好，再考虑继续做强化学习后训练。

## 基座模型

本阶段从 handoff2 GRPO 模型出发：

```text
outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500
```

Hugging Face 对应目录：

```text
https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500
```

它的上游是 handoff1 SFT-final，不是从 base model 直接训练出来的。

## 数据口径

目标数据集：

```text
Jiayi-Pan/Countdown-Tasks-3to4
```

本阶段只评估 / 训练 solvable 题目。不可解题不纳入主实验，否则无法区分模型能力不足和题目本身无解。

## 为什么这个迁移是合理的

当前 Game24 模型的输入并不是完全裸四个数。训练和评估时，模型实际看到的是一个 chat prompt：

```text
<|im_start|>system
Play the 24-point game. Given four numbers, reach 24 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
1 6 9 12
<|im_end|>
<|im_start|>assistant
```

也就是说，目标 `24` 是通过 system prompt 明确告诉模型的；user 段只放数字。
模型并不是只靠“看到四个数就默认做 24 点”。

因此迁移到 Countdown 时，正确的最小迁移不是换成新的 `Numbers:` / `Target:`
输入格式，而是尽量保持 handoff2 的 Game24 prompt 行为，只把目标数字从 `24`
替换成 Countdown 的 `target`，user 段仍只放裸数字：

```text
<|im_start|>system
Play the 50-point game. Given four numbers, reach 50 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
74 5 20 88
<|im_end|>
<|im_start|>assistant
```

这个格式与原 Game24 prompt 的行为最接近：

- system prompt 说明任务规则；
- user prompt 只给数字；
- assistant 输出 `<think>` 推理和唯一 `<answer>...</answer>` 表达式；
- verifier 仍然检查表达式是否只使用给定数字并等于 target。

对应仓库 prompt style：

```text
qwen_chat_minimal_target
```

旧的 `qwen_chat_target` prompt 会引入新的 `Numbers:` / `Target:` 输入格式，
同时前期 SFT 数据还使用了 `Check:` completion，改变了模型原本的搜索输出语义。
这条旧路线的结果可以作为失败记录保留，但不能作为“最小迁移是否可行”的结论。

## Eval 设计口径

本阶段不直接从 `Jiayi-Pan/Countdown-Tasks-3to4` 随机抽全量 eval。该数据集很大，
直接全量评估成本不可控，因此先构建一个 100 题的可复现小评估集。

构建脚本：

```text
scripts/build_countdown_stratified_eval_manifest.py
```

构建逻辑：

- 使用 DFS 暴力搜索每道题的可行表达式数量；
- 只保留 solvable 题目，0 条可行路径的题不进入主评估；
- 使用 dataset 原始顺序做流式扫描，不先下载或扫描全量数据；
- 按唯一可行表达式数量分层；
- 当前目标分层为 `1/2/3/4` 条可行路径四组；
- 每组 25 题，总计 100 题；
- 当四组都凑满 25 题时立即停止扫描；
- 如果某一组不够 25 题，脚本会失败并打印缺口，不会自动改口径。

旧 zero-shot 评估口径：

- model base：`outputs/experiments/baseline_format_v2_full_5000_from800/final`
- LoRA checkpoint：`outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500`
- prompt style：`qwen_chat_target`
- max_new_tokens：`4096`
- batch_size：`4`
- decoding：greedy

过程性运行记录放在：

```text
docs/experiments/countdown_transfer_eval_20260619.md
docs/experiments/countdown_handoff3_sft_baseline_20260620.md
```

## 当前结果

### Handoff2 minimal-transfer audit

在 handoff2 GRPO 模型上，使用 `qwen_chat_minimal_target` 先跑了 16 题 greedy
审计：

```text
outputs/experiments/handoff3_countdown_minimal/eval_handoff2_minimal_greedy16
```

| total | solved | solve_rate | format_ok | valid_expr | max_new_tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0.00% | 1 | 0 | 4096 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 15 |
| `wrong_numbers` | 1 |

结论：最小迁移 prompt 恢复了 handoff2 的 rollback/search 行为，但 greedy
仍不能直接完成 Countdown；主要失败是 4096 token 内没有闭合 `<answer>`。

原计划跑 32 题、每题 8 采样、4096 token 的 sampled audit：

```text
outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_32_g8
```

该审计运行 60 分钟仍无 summary，GPU 持续生成，说明当前模型在最小迁移 prompt
下倾向于打满长搜索。为避免继续沉没成本，停止该审计，改跑小样本 8 题、每题
4 采样、4096 token：

```text
outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_8_g4
```

| prompts | samples / prompt | total outputs | solved | pass@4 | format_ok | valid_expr | len mean | len p50 | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 32 | 0 | 0/8 | 3 | 0 | 4096 | 4096 | 4096 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 29 |
| `wrong_numbers` | 3 |

结论：handoff2 在最小迁移 prompt 下能恢复搜索形式，但当前采样空间里没有观察到
正确轨迹，并且所有样本都打满 4096 token。直接从 handoff2 对 Countdown 做 GRPO
缺少正样本信号，风险很高。下一步更合理的是先用最小迁移 prompt 重新构建
Countdown SFT 数据，让模型学会“目标数可变但搜索/输出格式不变”，再重新做
sampled audit 和 GRPO。

起步阶段的完整实验过程记录：

```text
docs/experiments/countdown_handoff3_sft_baseline_20260620.md
```

### Handoff3 target-replacement SFT

按最小迁移口径完成一轮 SFT：

```text
outputs/experiments/handoff3_countdown_sft/sft_handoff2_target_replacement_20000_2400steps/final
```

训练口径：

| item | value |
| --- | --- |
| start | handoff2 GRPO checkpoint-500 |
| train records | 20000 |
| max_steps | 2400 |
| learning_rate | `1e-5` |
| max_length | 4096 |
| prompt | target 替换，user 裸数字 |
| completion | solver trace + `<answer>` |

Full100 greedy eval：

| total | solved | solve_rate | format_ok | valid_expr | main failure |
| ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 5 | 5.00% | 100 | 98 | `wrong_value=93` |

Full100 sampled audit：

| prompts | samples / prompt | solved outputs | pass@8 | mixed groups | all-wrong groups |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8 | 31/800 | 13/100 | 13 | 87 |

结论：这轮 SFT 基本修复了格式和闭合问题，但训练数据中 `5_plus` 容易题占比过高，
与当前 full100 eval 的 `1/2/3/4` 低解数分布不匹配，导致求解率仍低。

### Handoff3 balanced low-solution SFT

随后按 eval 难度分布追加 balanced low-solution SFT：

```text
outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final
```

训练口径：

| item | value |
| --- | --- |
| start | target-replacement SFT final |
| train records | 20000 |
| bucket quotas | `1=5000, 2=5000, 3=5000, 4=5000` |
| max_steps | 2400 |
| learning_rate | `1e-5` |
| max_length | 4096 |
| prompt | target 替换，user 裸数字 |
| completion | solver trace + `<answer>` |

Full100 greedy eval：

| total | solved | solve_rate | format_ok | valid_expr | main failure |
| ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 23 | 23.00% | 100 | 98 | `wrong_value=75` |

Full100 sampled audit：

| prompts | samples / prompt | solved outputs | pass@8 | mixed groups | all-wrong groups | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8 | 134/800 | 45/100 | 42 | 55 | 88.84 | 112 |

对比：

| metric | target-replacement SFT | balanced low-solution SFT |
| --- | ---: | ---: |
| greedy solve | `5/100` | `23/100` |
| sampled solved outputs | `31/800` | `134/800` |
| pass@8 | `13/100` | `45/100` |
| mixed groups | `13` | `42` |

结论：balanced low-solution SFT 是当前 handoff3 的有效路线。它还不是最终结果，
但已经把问题从“格式/闭合失败”推进到“表达式值不等于 target”，并且 sampled
distribution 中已有足够正确轨迹，下一步可以从该 checkpoint 做一轮保守 GRPO
probe。

### Target-distance GRPO probe

在 balanced low-solution SFT 基础上做了一轮 target-distance GRPO：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/grpo_targetdistance_from_balanced_sft_train256_g8_lr1e6_beta001_1200/final
```

结果：

| metric | balanced SFT | GRPO target-distance |
| --- | ---: | ---: |
| held-out full100 greedy | `23/100` | `26/100` |
| held-out full100 solved samples | `134/800` | `177/800` |
| held-out full100 strict pass@8 | `45/100` | `46/100` |
| held-out full100 valid_expr | `98/100` | `96/100` |

结论：GRPO 有小幅 greedy 提升，但没有把 `pass@8` 有效转成 greedy direct
accuracy。主要失败仍是 `wrong_value`。旧 `pass_at_k=87/100` 是
target-distance reward 非零口径，不是 strict solve pass@8。

详细记录：

```text
docs/experiments/countdown_handoff3_grpo_20260620.md
```

### Final chance broad-solvable SFT

针对 balanced low-solution SFT 的一个关键缺口，追加最后一次训练：此前
`countdown-balanced-low-solution-sft-20000.jsonl` 是按 records 配额构建，
每题多 trace，unique puzzle 覆盖明显小于 20000；而 Countdown 的 target/number
空间远大于 24 点，低 unique 覆盖可能是泛化差的重要原因。

因此最后一次尝试不再继续加 GRPO，而是从 balanced low-solution SFT final 继续，
使用 20000 道唯一可解 Countdown 题构建 fast broad-solvable SFT 数据。构建时只
搜索每题第一条解，不做完整解数统计；排除当前 full100 eval manifest，保持
held-out 评估不泄漏。

脚本：

```text
scripts/experiments/run_countdown_final_chance_fast.sh
```

运行目录：

```text
outputs/experiments/handoff3_countdown_final_chance_fast
```

训练口径：

| item | value |
| --- | --- |
| start base | `outputs/experiments/baseline_format_v2_full_5000_from800/final` |
| initial adapter | balanced low-solution SFT final |
| train records | 20000 unique solvable puzzles |
| excluded eval | `countdown-stratified-eval-manifest.json` full100 |
| max_steps | 6000 |
| learning_rate | `1e-5` |
| max_length | 1024 |
| eval | full100 greedy, `max_new_tokens=4096` |
| sampled audit | full100 `G=8`, `max_new_tokens=4096` |

结果：

| checkpoint | solved | solve_rate | format_ok | valid_expr |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-1500` | 15/100 | 15.00% | 100 | 94 |
| `checkpoint-3000` | 19/100 | 19.00% | 100 | 92 |
| `checkpoint-4500` | 16/100 | 16.00% | 100 | 94 |
| `checkpoint-6000` / `final` | 19/100 | 19.00% | 100 | 94 |

Final sampled audit：

| prompts | samples / prompt | solved outputs | strict pass@8 | strict mixed | all-correct | all-wrong | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8 | 137/800 | 43/100 | 39 | 4 | 57 | 92.45 | 118 |

Failure mix：

| reason | count |
| --- | ---: |
| `wrong_value` | 599 |
| `wrong_numbers` | 59 |
| `syntax_error:'(' was never closed` | 3 |
| `syntax_error:unmatched ')'` | 1 |
| `unsupported_expression:UnaryOp` | 1 |
| `ok` | 137 |

结论：broad-solvable SFT 没有提升 held-out full100，反而低于 balanced SFT
`23/100` 和 target-distance GRPO `26/100`。这说明当前 100 题低解数 eval 对
训练分布很敏感；简单增加普通 solvable unique puzzles 会稀释难度分布，不能作为
继续烧算力的方向。

### 旧 target-aware 路线

以下旧实验使用 `qwen_chat_target` prompt 或 `Check:` completion，不是当前认可的
最小迁移口径，只作为失败记录保留：

| route | result |
| --- | --- |
| old zero-shot full100 | `0/100` |
| old SFT-2000-600 sampled audit | `pass@8=1/16` |
| old curriculum SFT full100 | `4/100` |
| old GRPO target_alignment | balanced16 `0/16`, train128 `56/128` |
| old GRPO target_distance | balanced16 `0/16`, train128 `57/128` |

旧路线的详细记录：

```text
docs/experiments/countdown_transfer_eval_20260619.md
docs/experiments/countdown_adaptation_plan_20260619.md
docs/experiments/countdown_grpo_alignment_20260620.md
```
