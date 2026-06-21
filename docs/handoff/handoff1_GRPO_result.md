# Handoff1 GRPO Result Summary

## 口径

- 模型：`/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/final`
- Hugging Face：<https://huggingface.co/Prometheus17/game24-rl/tree/main/grpo-lora-final>
- 初始化模型：`outputs/experiments/baseline_format_v2_full_5000_from800/final`
- 数据：repo-local `standard-game24-v1` split 的 validation/test。
- 统计方式：从 raw outputs 重新运行 strict verifier，再用输入数字 multiset 映射到
  `test-time-compute/game-of-24` 的 ToT `tot_index + 1`，最后按 rank 分桶。
- 留档脚本：`scripts/audit_eval_by_tot_rank.py`
- 1024 留档结果：`outputs/audits/handoff1_grpo_best_val_test_1024_by_tot_rank_from_raw.json`
- 4096 留档结果：`outputs/audits/handoff1_grpo_best_val_test_by_tot_rank_from_raw.json`
- Verifier：`strict-ast-fraction-v1`

说明：handoff1 best GRPO adapter 只保留了 validation/test 的 direct greedy eval；
没有该 adapter 的 train split eval。下面的 `val+test all` 是 validation 和 test
合并后的 273 题统计，不是 1362 题全量统计。

## 总表

| split | Easy 1-300 | Medium 301-900 | Hard 901-1100 | Very hard 1101-1362 | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 val | 96.43% | 88.33% | 87.50% | 62.50% | 85.29% |
| 1024 test | 94.12% | 91.94% | 68.42% | 63.64% | 84.67% |
| 1024 val+test all | 95.16% | 90.16% | 79.07% | 63.04% | 84.98% |
| 4096 val | 100.00% | 100.00% | 91.67% | 66.67% | 92.65% |
| 4096 test | 94.12% | 93.55% | 94.74% | 95.45% | 94.16% |
| 4096 val+test all | 96.77% | 96.72% | 93.02% | 80.43% | 93.41% |

## 1024 Val

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 27/28 | 96.43% | 1 no-answer |
| Medium 301-900 | 53/60 | 88.33% | 7 no-answer |
| Hard 901-1100 | 21/24 | 87.50% | 3 no-answer |
| Very hard 1101-1362 | 15/24 | 62.50% | 9 no-answer |
| Overall | 116/136 | 85.29% | 20 no-answer |

## 1024 Test

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 32/34 | 94.12% | 2 no-answer |
| Medium 301-900 | 57/62 | 91.94% | 5 no-answer |
| Hard 901-1100 | 13/19 | 68.42% | 6 no-answer |
| Very hard 1101-1362 | 14/22 | 63.64% | 8 no-answer |
| Overall | 116/137 | 84.67% | 21 no-answer |

## 1024 Val+Test All

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 59/62 | 95.16% | 3 no-answer |
| Medium 301-900 | 110/122 | 90.16% | 12 no-answer |
| Hard 901-1100 | 34/43 | 79.07% | 9 no-answer |
| Very hard 1101-1362 | 29/46 | 63.04% | 17 no-answer |
| Overall | 232/273 | 84.98% | 41 no-answer |

## 4096 Val

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 28/28 | 100.00% | none |
| Medium 301-900 | 60/60 | 100.00% | none |
| Hard 901-1100 | 22/24 | 91.67% | 2 no-answer |
| Very hard 1101-1362 | 16/24 | 66.67% | 7 no-answer, 1 wrong-value |
| Overall | 126/136 | 92.65% | 9 no-answer, 1 wrong-value |

## 4096 Test

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 32/34 | 94.12% | 1 no-answer, 1 wrong-numbers |
| Medium 301-900 | 58/62 | 93.55% | 3 no-answer, 1 wrong-numbers |
| Hard 901-1100 | 18/19 | 94.74% | 1 no-answer |
| Very hard 1101-1362 | 21/22 | 95.45% | 1 no-answer |
| Overall | 129/137 | 94.16% | 6 no-answer, 2 wrong-numbers |

## 4096 Val+Test All

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 60/62 | 96.77% | 1 no-answer, 1 wrong-numbers |
| Medium 301-900 | 118/122 | 96.72% | 3 no-answer, 1 wrong-numbers |
| Hard 901-1100 | 40/43 | 93.02% | 3 no-answer |
| Very hard 1101-1362 | 37/46 | 80.43% | 8 no-answer, 1 wrong-value |
| Overall | 255/273 | 93.41% | 15 no-answer, 2 wrong-numbers, 1 wrong-value |
