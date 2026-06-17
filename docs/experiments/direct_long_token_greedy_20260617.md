# Direct Long-Token Greedy Evaluation - 2026-06-17

## 结论

对当前最佳 GRPO LoRA adapter
`/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/final`
使用统一 greedy decoding、`max_new_tokens=4096` 复评后，单模型直接推理已经达到
90%+：

- validation：`126/136 = 92.65%`
- test：`129/137 = 94.16%`

这个结果没有使用 verifier-rerank，也没有从多个采样答案里选答案。评估仍使用同一个 strict
AST + Fraction verifier，只是把生成预算从之前的 `1024/2048` 提高到 `4096`。

## 评估配置

- Base model：
  `/root/autodl-tmp/projects/game24-rl/outputs/experiments/baseline_format_v2_full_5000_from800/final`
- LoRA checkpoint：
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/final`
- Split manifest：
  `/root/autodl-tmp/projects/game24-rl/data/processed/splits/standard-game24-v1.json`
- Prompt style：`qwen_chat`
- Decoding：greedy, `do_sample=false`, `temperature=null`, `top_p=null`
- Answer contract：`<answer>...</answer>`
- Verifier：`strict-ast-fraction-v1`

## Direct Greedy 结果

| split | max_new_tokens | solve rate | format rate | valid expr rate | failure mix |
| --- | ---: | ---: | ---: | ---: | --- |
| validation | 1024 | `116/136 = 85.29%` | `116/136 = 85.29%` | `116/136 = 85.29%` | 20 answer-contract |
| validation | 2048 | `123/136 = 90.44%` | `123/136 = 90.44%` | `123/136 = 90.44%` | 13 answer-contract |
| validation | 4096 | `126/136 = 92.65%` | `127/136 = 93.38%` | `127/136 = 93.38%` | 9 answer-contract, 1 wrong-value |
| test | 1024 | `116/137 = 84.67%` | `116/137 = 84.67%` | `116/137 = 84.67%` | 21 answer-contract |
| test | 2048 | `122/137 = 89.05%` | `124/137 = 90.51%` | `122/137 = 89.05%` | 13 answer-contract, 2 wrong-numbers |
| test | 4096 | `129/137 = 94.16%` | `131/137 = 95.62%` | `129/137 = 94.16%` | 6 answer-contract, 2 wrong-numbers |

## Artifacts

- validation 2048：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_validation_greedy_2048/validation-eval-report.json`
- test 2048：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_test_greedy_2048/test-eval-report.json`
- validation 4096：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_validation_greedy_4096/validation-eval-report.json`
- test 4096：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_test_greedy_4096/test-eval-report.json`

## 和 rerank 的关系

Verifier-rerank 仍是更高的 inference-time search-selection 结果：

- validation：`133/136 = 97.79%`
- test：`136/137 = 99.27%`

但 rerank 会从 sampled candidates 里用 strict verifier 选择正确答案，报告时必须与 direct
greedy 分开。当前 `4096` 结果更符合“一个模型直接推理”的展示口径。

## 解释

2048 与 4096 的结果说明，主要瓶颈确实是长搜索和 answer closure，而不是算术 verifier 或
split 泄漏。增加生成预算可以把一批原本没有闭合 `<answer>` 的题救回来；test 从
`122/137` 提升到 `129/137`，validation 从 `123/136` 提升到 `126/136`。

代价也很明确：4096 全量 test 用时约 `40m11s`，validation 用时约 `42m51s`，GPU 利用率
大多在 `39-41%`，显存约 `3.7GB / 49GB`。后续如果继续优化 greedy，应优先做停止/闭合
策略或蒸馏更短的成功轨迹，而不是只继续扩大 token budget。

## 工程修复

本轮发现 `eval_checkpoint.py` 旧实现只在完整 split 结束后一次性写 raw outputs，长 token
评估期间不可观测。已将 `generate_checkpoint_outputs` 改为每个 batch 增量追加 JSONL，并打印
`generated x/y records`，不改变评估口径。
