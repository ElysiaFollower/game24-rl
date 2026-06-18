# Official ToT Results 2026-06-18

## 口径

本文件记录本轮 official ToT 实验的最终结果表。执行口径以
`docs/experiments/official_tot_overnight_plan_20260618.md` 为准：

- 评测数据统一为 `test-time-compute/game-of-24` 全量 `1362` 条。
- 每个模型只跑一次全量 greedy eval，再从同一份 per-sample 结果离线切分
  `all_1362`、`easy1262` 和 `hard100`。
- `nlile/24-game` 和 `test-time-compute/game-of-24` 已确认 puzzle 集合完全相同，
  只是排序和元数据不同。
- `hard100` 是 ToT indices `900-999`。只有对
  `SFT-remove-900to1000-5000` 及其 GRPO 后续模型，它才是 held-out subset。
- 默认评测预算为 greedy `max_new_tokens=4096`，answer contract 为
  `<answer>...</answer>`，verifier 为 `strict-ast-fraction-v1`。

## 主结果表

| stage | init model | train set | eval set | max_new_tokens | all_1362 | easy1262 | hard100 | status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Base eval | `Qwen/Qwen2.5-1.5B-Instruct` | none | ToT full, split offline | 4096 | `16/1362 = 1.17%` | `14/1262 = 1.11%` | `2/100 = 2.00%` | done |
| `SFT-full-data-5000` | `Qwen/Qwen2.5-1.5B-Instruct` | `train_full_1362` | ToT full, split offline | 4096 | pending | pending | pending | training/eval pending |
| `SFT-remove-900to1000-5000` | `Qwen/Qwen2.5-1.5B-Instruct` | `train_remove_900to1000_1262` | ToT full, split offline | 4096 | pending | pending | pending | pending |
| `GRPO-full` | `SFT-full-data-5000` | `train_full_1362` | ToT full, split offline | 4096 | pending | pending | pending | pending |
| `GRPO-remove-900to1000` | `SFT-remove-900to1000-5000` | `train_remove_900to1000_1262` | ToT full, split offline | 4096 | pending | pending | pending | pending |

## Base Eval 已完成结果

| group | solved | solve rate | format rate | valid expr rate |
| --- | ---: | ---: | ---: | ---: |
| `all_1362` | `16/1362` | `1.17%` | `17.25%` | `14.98%` |
| `easy1262` | `14/1262` | `1.11%` | `17.67%` | `15.45%` |
| `hard100` | `2/100` | `2.00%` | `12.00%` | `9.00%` |

主要失败分布用于解释，不作为训练/评估脚本硬门槛：

| group | no/incomplete answer | wrong value | wrong numbers | syntax / unsupported | ok |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all_1362` | `1127` | `188` | `23` | `8` | `16` |
| `easy1262` | `1039` | `181` | `21` | `7` | `14` |
| `hard100` | `88` | `7` | `2` | `1` | `2` |

Artifacts：

```text
/root/autodl-tmp/projects/game24-rl/outputs/official_tot_overnight_20260618/eval/base_4096/tot_all_1362-eval-report.json
/root/autodl-tmp/projects/game24-rl/outputs/official_tot_overnight_20260618/eval/base_4096/tot_all_1362-raw-outputs.jsonl
/root/autodl-tmp/projects/game24-rl/outputs/official_tot_overnight_20260618/eval/base_4096/group-summary.json
```

## 待补槽位

后续每完成一个模型，只补同一张主结果表和对应 artifact 路径：

- `SFT-full-data-5000`：完整 `nlile/24-game` 训练，ToT 全量 4096 eval。
- `SFT-remove-900to1000-5000`：排除 ToT indices `900-999` 的 `1262` 条训练，
  ToT 全量 4096 eval。
- `GRPO-full`：从 `SFT-full-data-5000` checkpoint 继续后训练，ToT 全量 4096 eval。
- `GRPO-remove-900to1000`：从 `SFT-remove-900to1000-5000` checkpoint 继续后训练，
  ToT 全量 4096 eval。
