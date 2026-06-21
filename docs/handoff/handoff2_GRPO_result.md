# Handoff2 GRPO Result Summary

## 口径

- 模型：`outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500`
- Hugging Face：<https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500>
- 初始化模型：`outputs/experiments/baseline_format_v2_full_5000_from800/final`
- 训练数据：repo-local `standard-game24-v1` split 的 train。
- 评估数据：repo-local `standard-game24-v1` split 的 validation/test。
- 训练配置摘要：LoRA rank 16，`max_completion_length=4096`，`num_generations=8`，
  `max_steps=2000`，`save_steps=500`，`learning_rate=5e-6`，`beta=0`，
  `scale_rewards=none`，`loss_type=dr_grpo`，reward profile 为
  `closure_control_smooth`。
- 实际交付 checkpoint：因时间原因停止后评估 `checkpoint-500`，不是 2000-step final。
- 评估方式：direct greedy，`max_new_tokens=4096`，不使用 verifier rerank。
- 统计方式：从 raw outputs 重新运行 strict verifier，再用输入数字 multiset 映射到
  `test-time-compute/game-of-24` 的 ToT `tot_index + 1`，最后按 rank 分桶。
- 留档脚本：`scripts/audit_eval_by_tot_rank.py`
- 留档结果：`outputs/audits/handoff2_grpo_checkpoint500_val_test_4096_by_tot_rank_from_raw.json`
- Verifier：`strict-ast-fraction-v1`

说明：本轮没有跑 1024，也没有跑 train split eval。下面的 `val+test all`
是 validation 和 test 合并后的 273 题统计，不是 1362 题全量统计。

## 总表

| split | Easy 1-300 | Medium 301-900 | Hard 901-1100 | Very hard 1101-1362 | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 100.00% | 100.00% | 91.67% | 75.00% | 94.12% |
| test | 88.24% | 96.77% | 94.74% | 95.45% | 94.16% |
| val+test all | 93.55% | 98.36% | 93.02% | 84.78% | 94.14% |

## Val

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 28/28 | 100.00% | none |
| Medium 301-900 | 60/60 | 100.00% | none |
| Hard 901-1100 | 22/24 | 91.67% | 2 no-answer |
| Very hard 1101-1362 | 18/24 | 75.00% | 6 no-answer |
| Overall | 128/136 | 94.12% | 8 no-answer |

## Test

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 30/34 | 88.24% | 2 no-answer, 1 unsupported-expression, 1 wrong-numbers |
| Medium 301-900 | 60/62 | 96.77% | 2 no-answer |
| Hard 901-1100 | 18/19 | 94.74% | 1 no-answer |
| Very hard 1101-1362 | 21/22 | 95.45% | 1 no-answer |
| Overall | 129/137 | 94.16% | 6 no-answer, 1 unsupported-expression, 1 wrong-numbers |

## Val+Test All

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 58/62 | 93.55% | 2 no-answer, 1 unsupported-expression, 1 wrong-numbers |
| Medium 301-900 | 120/122 | 98.36% | 2 no-answer |
| Hard 901-1100 | 40/43 | 93.02% | 3 no-answer |
| Very hard 1101-1362 | 39/46 | 84.78% | 7 no-answer |
| Overall | 257/273 | 94.14% | 14 no-answer, 1 unsupported-expression, 1 wrong-numbers |
