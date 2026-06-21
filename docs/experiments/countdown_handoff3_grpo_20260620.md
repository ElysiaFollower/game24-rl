# Countdown Handoff3 GRPO

## 目的

在 handoff3 当前有效路线 `balanced low-solution SFT` 基础上，做一轮保守 GRPO，
验证 sampled 正确轨迹能否被推到 greedy 解码中。

## 起点

```text
base model:
outputs/experiments/baseline_format_v2_full_5000_from800/final

initial LoRA adapter:
outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final
```

起点 full100 指标：

| eval | result |
| --- | ---: |
| greedy | `23/100` |
| sampled pass@8 | `45/100` |
| sampled solved outputs | `134/800` |
| mixed groups | `42` |

主要失败是 `wrong_value`，不是 answer closure。

## 本轮固定口径

脚本：

```text
scripts/experiments/run_countdown_handoff3_balanced_grpo.sh
```

训练数据从 balanced SFT JSONL 抽取 unique prompts：

```text
outputs/experiments/handoff3_countdown_balanced_sft/data/countdown-balanced-low-solution-sft-20000.jsonl
```

GRPO train manifest：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/data/countdown-balanced-grpo-train-manifest.json
```

抽样口径：

| bucket | prompts |
| --- | ---: |
| `1` | 64 |
| `2` | 64 |
| `3` | 64 |
| `4` | 64 |
| total | 256 |

训练前 rollout / pool：

| item | value |
| --- | --- |
| num_generations | 8 |
| max_new_tokens | 512 |
| temperature | 0.8 |
| top_p | 0.95 |
| reward_profile | `target_distance` |
| pool selection | reward-variance groups |

GRPO 训练：

| item | value |
| --- | --- |
| initial adapter | balanced low-solution SFT final |
| reward_profile | `target_distance` |
| max_steps | 1200 |
| save_steps | 300 |
| learning_rate | `1e-6` |
| beta | `0.001` |
| scale_rewards | `none` |
| loss_type | `dr_grpo` |
| num_generations | 8 |
| gradient_accumulation_steps | 8 |
| max_completion_length | 512 |
| mask_truncated_completions | `false` |
| remove_unused_columns | `false` |

评估：

| eval | value |
| --- | --- |
| full100 greedy | `max_new_tokens=4096` |
| full100 sampled audit | `G=8`, `max_new_tokens=4096`, `target_distance` diagnostics |

## 当前状态

2026-06-20 23:55 CST 首次启动 `512 prompts / 1024 max_new_tokens`，
但 train-side rollout 运行 34 分钟仍未产出 artifact。为避免把夜里算力耗在
建 pool 上，已停止该 run，切换为当前 `256 prompts / 512 max_new_tokens`
probe 口径。

当前 AutoDL 运行目录：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe
```

## Train-Side Rollout / Pool

Artifact：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/rollout_train256_g8_targetdistance_512/summary.json
outputs/experiments/handoff3_countdown_balanced_grpo_probe/pool_train256_g8_targetdistance_512/pool-manifest.json
```

rollout strict summary：

| prompts | outputs | solved outputs | strict pass@8 | strict mixed | all-correct | all-wrong | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2048 | 592 | 149/256 | 122 | 27 | 107 | 91.08 | 113 |

以 `target_distance` reward 重新计算方差后，pool manifest 选出 `221` 个
reward-variance prompt groups：

| selected prompts | reward-variance rate | reward zero-std rate |
| ---: | ---: | ---: |
| 221 | 86.33% | 13.67% |

解释：strict valid 口径下有 122 个 mixed groups；`target_distance` reward 又能从
strict all-wrong 但接近程度不同的组里提供额外梯度，所以最终可训练 prompt pool 为 221。

## 训练与评估结果

2026-06-21 00:52 CST 开始 GRPO 训练，02:35 CST 左右完成训练并自动进入评估。
训练过程中无 OOM，`completions/clipped_ratio=0`，平均 completion 长度约
80-100 token。

最终 checkpoint：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/grpo_targetdistance_from_balanced_sft_train256_g8_lr1e6_beta001_1200/final
```

Held-out full100 greedy eval：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/eval_targetdistance_1200_full100_4096/countdown_eval-eval-report.json
```

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 26 | 26.00% | 100 | 96 |

Held-out full100 sampled audit：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/rollout_targetdistance_1200_full100_g8_4096/summary.json
```

| prompts | outputs | solved outputs | strict pass@8 | strict mixed | all-correct | all-wrong | reward-var groups | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 800 | 177 | 46/100 | 40 | 6 | 54 | 91 | 88.92 | 115 |

注：旧 summary 里的 `pass_at_k=87/100` 是 target-distance reward 非零口径，
不是 strict valid pass@8；该值不能作为准确率解读。

失败分布：

| reason | count |
| --- | ---: |
| `wrong_value` | 581 |
| `wrong_numbers` | 38 |
| `syntax_error:unmatched ')'` | 4 |
| `ok` | 177 |

Train256 greedy eval：

```text
outputs/experiments/handoff3_countdown_balanced_grpo_probe/eval_targetdistance_1200_train256_4096/countdown_grpo_train-eval-report.json
```

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 111 | 43.36% | 256 | 255 |

Checkpoint sweep on held-out full100：

| checkpoint | solved | solve_rate | format_ok | valid_expr |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-300` | 24/100 | 24.00% | 100 | 93 |
| `checkpoint-600` | 24/100 | 24.00% | 100 | 95 |
| `checkpoint-900` | 26/100 | 26.00% | 100 | 96 |
| `checkpoint-1200` / `final` | 26/100 | 26.00% | 100 | 96 |

`checkpoint-1200` 与 `final` 的 `adapter_model.safetensors` sha256 一致。

## 对比结论

| metric | balanced SFT | GRPO target-distance |
| --- | ---: | ---: |
| held-out full100 greedy | `23/100` | `26/100` |
| held-out full100 solved samples | `134/800` | `177/800` |
| held-out full100 strict pass@8 | `45/100` | `46/100` |
| held-out full100 valid_expr | `98/100` | `96/100` |

结论：GRPO 带来小幅 greedy 提升，并增加 strict 正确样本数；
但 strict pass@8 基本没变，greedy 仍只有 `26%`，主要失败还是 `wrong_value`。
这说明 target-distance GRPO 的学习信号不足以把正确轨迹推成 greedy direct 解码。

## 2026-06-21 Final Chance: Broad-Solvable SFT

target-distance GRPO 后的主要失败仍是 `wrong_value`，而不是格式或闭合失败。
同时复盘发现 balanced low-solution SFT 的 `20000` 是 record 数，不是 unique
puzzle 数；由于每题多 trace，真实 unique 覆盖远低于 20000。Countdown 的目标值和
数字组合空间远大于 24 点，这可能让模型只学到局部分布，无法泛化到 held-out
full100。

因此最后一次尝试改为扩大 unique puzzle 覆盖，而不是继续加大 GRPO 强度。

脚本：

```text
scripts/experiments/run_countdown_final_chance_fast.sh
```

运行目录：

```text
outputs/experiments/handoff3_countdown_final_chance_fast
```

数据构建：

| item | value |
| --- | --- |
| source | `Jiayi-Pan/Countdown-Tasks-3to4` local JSONL |
| builder | `scripts/build_countdown_fast_sft_data.py` |
| train records | 20000 |
| uniqueness | one record per unique `(target, nums)` puzzle |
| solver | first exact solution only, no full solution counting |
| excluded eval | `outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json` |
| prompt | `qwen_chat_minimal_target` |
| completion | short success trace + `<answer>` |

训练：

| item | value |
| --- | --- |
| base model | `outputs/experiments/baseline_format_v2_full_5000_from800/final` |
| initial adapter | `outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final` |
| output | `outputs/experiments/handoff3_countdown_final_chance_fast/sft_fast20k_from_balanced_6000steps_lr1e5/final` |
| max_steps | 6000 |
| save_steps | 1500 |
| learning_rate | `1e-5` |
| max_length | 1024 |
| gradient_accumulation_steps | 8 |

后续评估：

| eval | config | output |
| --- | --- | --- |
| full100 greedy | `max_new_tokens=4096` | `eval_fast20k_full100_4096` |
| full100 sampled audit | `G=8`, `temperature=0.8`, `top_p=0.95`, `max_new_tokens=4096` | `rollout_fast20k_full100_g8_4096` |

2026-06-21 14:41 CST 启动训练，17:33 CST 完成训练并自动评估，17:43 CST 完成
sampled audit。

训练日志摘要：

| item | value |
| --- | --- |
| runtime | `10310s` |
| train_loss | `0.02657` |
| final lr | `1.667e-09` |
| mean_token_accuracy near end | about `0.991-0.993` |
| checkpoints | `1500/3000/4500/6000` |

Full100 greedy checkpoint sweep：

| checkpoint | solved | solve_rate | format_ok | valid_expr |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-1500` | 15/100 | 15.00% | 100 | 94 |
| `checkpoint-3000` | 19/100 | 19.00% | 100 | 92 |
| `checkpoint-4500` | 16/100 | 16.00% | 100 | 94 |
| `checkpoint-6000` | 19/100 | 19.00% | 100 | 94 |
| `final` | 19/100 | 19.00% | 100 | 94 |

Final greedy artifact：

```text
outputs/experiments/handoff3_countdown_final_chance_fast/eval_fast20k_full100_4096/countdown_eval-eval-report.json
```

Final sampled audit artifact：

```text
outputs/experiments/handoff3_countdown_final_chance_fast/rollout_fast20k_full100_g8_4096/summary.json
```

Final sampled audit：

| prompts | outputs | solved outputs | strict pass@8 | strict mixed | all-correct | all-wrong | valid_expr | len mean | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 800 | 137 | 43/100 | 39 | 4 | 57 | 736 | 92.45 | 118 |

Failure mix：

| reason | count |
| --- | ---: |
| `wrong_value` | 599 |
| `wrong_numbers` | 59 |
| `syntax_error:'(' was never closed` | 3 |
| `syntax_error:unmatched ')'` | 1 |
| `unsupported_expression:UnaryOp` | 1 |
| `ok` | 137 |

典型错误：

```text
target = 50
numbers = 74 5 20 88

<think>
(74) - (5) = 69, left: 69, 20, 88
(20) - (69) = -49, left: -49, 88
(-49) + (88) = 39, left: 39
</think>
<answer>((20 - (74 - 5)) + 88)</answer>

verifier: wrong_value, value = 39
```

结论：该实验没有超过当前 GRPO `26/100`，也低于 balanced SFT `23/100`。
checkpoint sweep 说明不是 final 过训掩盖了中途好点。更广的普通 solvable
unique coverage 会稀释当前 held-out full100 的低解数分布，不能作为继续烧算力的
有效方向。handoff3 当前可报告的最好 direct greedy 结果仍是
target-distance GRPO 的 `26/100`；Countdown 迁移应作为受限加分项/负结果记录。
