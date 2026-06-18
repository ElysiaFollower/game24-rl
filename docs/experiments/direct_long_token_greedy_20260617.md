# Direct Long-Token Greedy Evaluation - 2026-06-17

## 结论

同口径补跑 strong SFT final 后，结论需要拆成两层：

1. 直接放大 greedy generation budget 本身就是强基线。Strong SFT final 在
   `max_new_tokens=4096` 下已经达到 validation `123/136 = 90.44%`、test
   `128/137 = 93.43%`。
2. 当前最佳 GRPO LoRA adapter 在同样 `4096` 预算下仍有小幅净增：

- validation：`126/136 = 92.65%`
- test：`129/137 = 94.16%`

这个结果没有使用 verifier-rerank，也没有从多个采样答案里选答案。评估仍使用同一个 strict
AST + Fraction verifier。报告时不能把 `80.88% -> 92.65%` 全部归因于 GRPO；
更严谨的归因是：长 token budget 把 SFT final 推到 90%+，GRPO adapter 在同预算下把
validation 再提高 `+3/136`，test 再提高 `+1/137`。

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

## SFT Long-Token 基线

同样使用 strong SFT final、同样 qwen chat prompt、同样 strict verifier、同样 greedy
`max_new_tokens=4096`，不加载 GRPO adapter：

| split | max_new_tokens | solve rate | format rate | valid expr rate | failure mix |
| --- | ---: | ---: | ---: | ---: | --- |
| validation | 1024 | `110/136 = 80.88%` | `110/136 = 80.88%` | `110/136 = 80.88%` | 26 answer-contract |
| validation | 4096 | `123/136 = 90.44%` | `124/136 = 91.18%` | `124/136 = 91.18%` | 12 answer-contract, 1 wrong-value |
| test | 4096 | `128/137 = 93.43%` | `129/137 = 94.16%` | `128/137 = 93.43%` | 8 answer-contract, 1 wrong-numbers |

因此，当前 90%+ direct greedy 结果的第一性原因是 generation budget 从 `1024` 放到
`4096` 后，很多原本未输出 `<answer>` 的长搜索题得以继续搜索并闭合。GRPO adapter 的
同口径贡献仍存在，但更小：

| split | SFT final 4096 | GRPO adapter 4096 | net change |
| --- | ---: | ---: | ---: |
| validation | `123/136 = 90.44%` | `126/136 = 92.65%` | `+3/136` |
| test | `128/137 = 93.43%` | `129/137 = 94.16%` | `+1/137` |

## Artifacts

- validation 2048：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_validation_greedy_2048/validation-eval-report.json`
- test 2048：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_test_greedy_2048/test-eval-report.json`
- validation 4096：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_validation_greedy_4096/validation-eval-report.json`
- test 4096：
  `/root/autodl-tmp/projects/grpo-direct-long/best116_test_greedy_4096/test-eval-report.json`
- SFT validation 4096：
  `/root/autodl-tmp/projects/sft-direct-long/sft_final_validation_greedy_4096/validation-eval-report.json`
- SFT test 4096：
  `/root/autodl-tmp/projects/sft-direct-long/sft_final_test_greedy_4096/test-eval-report.json`

## 和 rerank 的关系

Verifier-rerank 仍是更高的 inference-time search-selection 结果：

- validation：`133/136 = 97.79%`
- test：`136/137 = 99.27%`

但 rerank 会从 sampled candidates 里用 strict verifier 选择正确答案，报告时必须与 direct
greedy 分开。当前 `4096` 结果更符合“一个模型直接推理”的展示口径。

## 解释

SFT 4096 基线进一步说明，主要瓶颈确实是长搜索和 answer closure，而不是算术 verifier 或
split 泄漏。增加生成预算可以把一批原本没有闭合 `<answer>` 的题救回来；SFT validation
从 `110/136` 提升到 `123/136`。GRPO adapter 在 4096 下继续减少少量失败，但没有改变
主要失败形态。

剩余失败里，answer-contract 失败仍然不是“答案标签写错了”，而是完全没有进入
`<answer>`：SFT validation 4096 的 12 个 answer-contract 失败、SFT test 4096 的 8 个、
GRPO validation 4096 的 9 个、GRPO test 4096 的 6 个，标签计数均为
`<answer>` 0 次、`</answer>` 0 次。输出尾部仍在 rollback/search。因此后续如果继续用
RL 改 greedy，reward 必须在同样长 token 预算下观察这些长搜索失败，或者训练更明确的
停止/闭合行为；短 token 预算下看不到这些样本的完整失败轨迹。

代价也很明确：4096 全量 test 用时约 `40m11s`，validation 用时约 `42m51s`，GPU 利用率
大多在 `39-41%`，显存约 `3.7GB / 49GB`。后续如果继续优化 greedy，应优先做停止/闭合
策略或蒸馏更短的成功轨迹，而不是只继续扩大 token budget。

补跑 SFT final 4096 基线 validation+test 总耗时约 `38m05s`，GPU 生成时约
`55-82%` utilization、显存约 `3.5-3.9GB / 49GB`。

## 工程修复

本轮发现 `eval_checkpoint.py` 旧实现只在完整 split 结束后一次性写 raw outputs，长 token
评估期间不可观测。已将 `generate_checkpoint_outputs` 改为每个 batch 增量追加 JSONL，并打印
`generated x/y records`，不改变评估口径。
