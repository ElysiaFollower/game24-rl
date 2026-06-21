# Countdown Target Alignment GRPO 记录

## 目标

基于 handoff3 Countdown curriculum SFT adapter，尝试用 GRPO 对齐模型对
`Target:` 的认识，减少“格式正确但表达式不等于 target”的 `wrong_value`。

## 起点

```text
outputs/experiments/handoff3_countdown_adapt/sft_train_handoff2_curriculum_5plus8000_1200_then_mixed20000_2400steps/final
```

该模型在 100 题 stratified eval 上为 `4/100`，格式已稳定：

| eval | total | solved | format_ok | valid_expr | main failure |
| --- | ---: | ---: | ---: | ---: | --- |
| full100 | 100 | 4 | 100 | 98 | `wrong_value=94` |
| balanced16 | 16 | 0 | 16 | 16 | `wrong_value=16` |

## GRPO v1：target_alignment

Reward：

- correct target: `+1.0`
- complete answer but wrong target / wrong numbers / parse error: `-1.0`
- missing or incomplete answer: `-0.5`

训练：

| item | value |
| --- | --- |
| initial adapter | Countdown curriculum SFT final |
| pool | 51 mixed prompts from train128 G8 audit |
| reward_profile | `target_alignment` |
| steps | 400 |
| lr | `1e-6` |
| beta | `0.001` |
| num_generations | 8 |
| max_completion_length | 512 |

Artifacts：

```text
outputs/experiments/handoff3_countdown_grpo/grpo_stage2_train128_mixed51_targetalign_lr1e6_400/final
outputs/experiments/handoff3_countdown_grpo/eval_targetalign_lr1e6_400_balanced16/countdown_eval_balanced16-eval-report.json
outputs/experiments/handoff3_countdown_grpo/eval_targetalign_lr1e6_400_train128/countdown_grpo_train-eval-report.json
```

结果：

| eval | total | solved | solve_rate | format_ok | valid_expr | failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| balanced16 | 16 | 0 | 0.00% | 16 | 16 | `wrong_value=16` |
| train128 | 128 | 56 | 43.75% | 128 | 128 | `wrong_value=72` |

结论：训练附近略有提升，但 held-out balanced16 没有提升。

## GRPO v2：target_distance

v1 后 sampled audit 显示采样空间里有大量正确答案，但只按二元正确性选 pool
会浪费 all-wrong 但有远近差异的题。因此新增 `target_distance` reward：

- correct target: `+1.0`
- wrong_value: `-min(1, abs(value - target) / max(abs(target), 1))`
- wrong_numbers: `-1.0`
- parse error: `-0.75`
- missing/incomplete answer: `-0.5`

同时更新 pool audit：`mixed_groups` 按 reward 是否有方差计算，而不是只按是否
有正确样本计算。

新 sampled audit：

```text
outputs/experiments/handoff3_countdown_grpo/rollout_targetalign400_train256_g8_distance/summary.json
```

| prompts | outputs | solved outputs | pass@8 | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2048 | 751 | 212/256 | 2048 | 2030 |

Distance pool：

```text
outputs/experiments/handoff3_countdown_grpo/pool_targetalign400_train256_g8_distance/pool-manifest.json
```

| total prompts | selected variance groups | zero_std groups | zero_std rate |
| ---: | ---: | ---: | ---: |
| 256 | 177 | 79 | 30.86% |

训练：

| item | value |
| --- | --- |
| initial adapter | target_alignment v1 final |
| pool | 177 reward-variance prompts |
| reward_profile | `target_distance` |
| steps | 800 |
| lr | `1e-6` |
| beta | `0.001` |
| num_generations | 8 |
| max_completion_length | 512 |

Artifacts：

```text
outputs/experiments/handoff3_countdown_grpo/grpo_targetdistance_from_targetalign400_pool177_lr1e6_800/final
outputs/experiments/handoff3_countdown_grpo/eval_targetdistance_lr1e6_800_balanced16/countdown_eval_balanced16-eval-report.json
outputs/experiments/handoff3_countdown_grpo/eval_targetdistance_lr1e6_800_train128/countdown_grpo_train-eval-report.json
```

结果：

| eval | total | solved | solve_rate | format_ok | valid_expr | failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| balanced16 | 16 | 0 | 0.00% | 16 | 16 | `wrong_value=16` |
| train128 | 128 | 57 | 44.53% | 128 | 128 | `wrong_value=71` |

## 结论

GRPO 确实能在训练附近带来很小的 target 对齐提升：

| model | train128 solved |
| --- | ---: |
| curriculum SFT | 52/128 |
| target_alignment GRPO | 56/128 |
| target_distance GRPO | 57/128 |

但 held-out balanced16 始终是 `0/16`。主要错误始终是 `wrong_value`，且 raw
output 显示模型会明确算出一个不等于 target 的值后仍提交 `<answer>`。

当前结论：在这个起点上，GRPO reward shaping 还不足以完成 Countdown target
迁移。下一条更有希望的路线不是继续盲目加大 GRPO step，而是先用更强的
target-aware SFT / rejection-sampled correct traces 把 greedy 命中率抬起来，再做
GRPO；或者在解码侧使用 sampled strict-verifier rerank 作为单独口径。
