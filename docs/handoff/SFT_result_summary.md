# Handoff1 SFT-final Result Summary

## 口径

- 模型：`outputs/experiments/baseline_format_v2_full_5000_from800/final`
- Hugging Face：<https://huggingface.co/Prometheus17/game24-rl/tree/main/sft-final>
- 数据：repo-local `standard-game24-v1` split 的 train/validation/test。
- 统计方式：从 raw outputs 重新运行 strict verifier，再用输入数字 multiset 映射到
  `test-time-compute/game-of-24` 的 ToT `tot_index + 1`，最后按 rank 分桶。
- 留档脚本：`scripts/audit_eval_by_tot_rank.py`
- 留档结果：`outputs/audits/handoff1_sft_final_train_val_test_by_tot_rank_from_raw.json`
- Verifier：`strict-ast-fraction-v1`

## 总表

| split | Easy 1-300 | Medium 301-900 | Hard 901-1100 | Very hard 1101-1362 | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 train | 94.12% | 89.12% | 87.90% | 72.69% | 86.78% |
| 1024 val | 92.86% | 85.00% | 83.33% | 54.17% | 80.88% |
| 1024 test | 91.18% | 91.94% | 73.68% | 59.09% | 83.94% |
| 1024 all | 93.67% | 89.00% | 86.00% | 69.85% | 85.90% |
| 4096 train | 97.06% | 95.61% | 94.90% | 87.50% | 94.21% |
| 4096 val | 100.00% | 100.00% | 87.50% | 58.33% | 90.44% |
| 4096 test | 94.12% | 93.55% | 89.47% | 95.45% | 93.43% |
| 4096 all | 97.00% | 95.83% | 93.50% | 85.50% | 93.76% |

## 1024 Train

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 224/238 | 94.12% | 14 no-answer |
| Medium 301-900 | 426/478 | 89.12% | 50 no-answer, 1 wrong-numbers, 1 wrong-value |
| Hard 901-1100 | 138/157 | 87.90% | 19 no-answer |
| Very hard 1101-1362 | 157/216 | 72.69% | 57 no-answer, 1 wrong-numbers, 1 wrong-value |
| Overall | 945/1089 | 86.78% | 140 no-answer, 2 wrong-numbers, 2 wrong-value |

## 1024 Val

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 26/28 | 92.86% | 2 no-answer |
| Medium 301-900 | 51/60 | 85.00% | 9 no-answer |
| Hard 901-1100 | 20/24 | 83.33% | 4 no-answer |
| Very hard 1101-1362 | 13/24 | 54.17% | 11 no-answer |
| Overall | 110/136 | 80.88% | 26 no-answer |

## 1024 Test

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 31/34 | 91.18% | 3 no-answer |
| Medium 301-900 | 57/62 | 91.94% | 4 no-answer, 1 wrong-numbers |
| Hard 901-1100 | 14/19 | 73.68% | 5 no-answer |
| Very hard 1101-1362 | 13/22 | 59.09% | 9 no-answer |
| Overall | 115/137 | 83.94% | 21 no-answer, 1 wrong-numbers |

## 4096 Train

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 231/238 | 97.06% | 6 no-answer, 1 syntax |
| Medium 301-900 | 457/478 | 95.61% | 17 no-answer, 2 wrong-numbers, 1 wrong-value, 1 syntax |
| Hard 901-1100 | 149/157 | 94.90% | 8 no-answer |
| Very hard 1101-1362 | 189/216 | 87.50% | 24 no-answer, 2 wrong-numbers, 1 wrong-value |
| Overall | 1026/1089 | 94.21% | 55 no-answer, 4 wrong-numbers, 2 wrong-value, 2 syntax |

## 4096 Val

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 28/28 | 100.00% | none |
| Medium 301-900 | 60/60 | 100.00% | none |
| Hard 901-1100 | 21/24 | 87.50% | 3 no-answer |
| Very hard 1101-1362 | 14/24 | 58.33% | 9 no-answer, 1 wrong-value |
| Overall | 123/136 | 90.44% | 12 no-answer, 1 wrong-value |

## 4096 Test

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 32/34 | 94.12% | 2 no-answer |
| Medium 301-900 | 58/62 | 93.55% | 3 no-answer, 1 wrong-numbers |
| Hard 901-1100 | 17/19 | 89.47% | 2 no-answer |
| Very hard 1101-1362 | 21/22 | 95.45% | 1 no-answer |
| Overall | 128/137 | 93.43% | 8 no-answer, 1 wrong-numbers |
