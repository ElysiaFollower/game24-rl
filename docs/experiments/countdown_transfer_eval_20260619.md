# Countdown Transfer Eval 记录

## 目标

基于 handoff2 GRPO checkpoint-500，在
`Jiayi-Pan/Countdown-Tasks-3to4` 上做 3-4 个数凑任意 target 的 zero-shot eval。

注：本文前半部分记录的是早期 `qwen_chat_target` 口径。后续复盘确认该 prompt
不是最小迁移：它把 user 输入改成 `Numbers:` / `Target:`，而 handoff2 的原始
Game24 输入是 user 段裸数字。最小迁移口径应使用 `qwen_chat_minimal_target`：
只把 system prompt 中的 `24` 替换成目标数字，其他行为尽量不动。

## Eval 数据集构建

不直接全量评估 Countdown 数据集。先构建 100 题可复现小评估集。

AutoDL 已下载 `Jiayi-Pan/Countdown-Tasks-3to4` 原始数据留档：

```text
data/raw/hf/Jiayi-Pan__Countdown-Tasks-3to4
```

该 HF repo 当前包含一个 parquet 数据文件：

```text
data/train-00000-of-00001.parquet
```

同时目录中已有展开后的 `default__train.jsonl`，共 `490364` 行。

构建脚本：

```text
scripts/build_countdown_stratified_eval_manifest.py
```

构建逻辑：

- 使用 DFS 暴力搜索每道题的可行表达式数量；
- 只保留 solvable 题目；
- 0 条可行路径题不进入主评估；
- 使用 dataset 原始顺序做流式扫描，不先下载或扫描全量数据；
- 按唯一可行表达式数量分层；
- 当前分层为 `1/2/3/4` 条可行路径四组；
- 每组 25 题，总计 100 题；
- 当四组都凑满 25 题时立即停止扫描；
- 如果某一组不够 25 题，脚本失败并打印缺口。

远端 manifest：

```text
outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json
```

构建结果：

| bucket | selected | observed while scanning |
| --- | ---: | ---: |
| 1 solution | 25 | 607 |
| 2 solutions | 25 | 278 |
| 3 solutions | 25 | 372 |
| 4 solutions | 25 | 25 |
| 5+ solutions | 0 | 1964 |

扫描前 `3246` 行已凑齐 100 题，因此 `1/2/3/4` 四组都足够。

## 旧 Zero-shot Eval 结果

远端输出目录：

```text
outputs/experiments/handoff3_countdown_eval/eval_handoff2_grpo_4096
```

评估口径：

- model base：`outputs/experiments/baseline_format_v2_full_5000_from800/final`
- LoRA checkpoint：`outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500`
- prompt style：`qwen_chat_target`
- max_new_tokens：`4096`
- batch_size：`4`
- decoding：greedy

结果：

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0 | 0.00% | 9 | 5 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 91 |
| `wrong_value` | 5 |
| `wrong_numbers` | 4 |

结论：handoff2 模型 zero-shot 不能直接迁移到 Countdown。主要问题不是
verifier，而是模型仍有 Game24/24 点惯性，并且经常不按 Countdown 的 target
prompt 早闭合答案。

这条结论只适用于旧 `qwen_chat_target` prompt，不用于判断最小迁移。

## Minimal-transfer Eval 结果

后续新增 `qwen_chat_minimal_target` prompt：

```text
<|im_start|>system
Play the 50-point game. Given four numbers, reach 50 using +, -, *, and /, and use each provided number exactly once. Output concise reasoning steps. End with exactly one final expression inside <answer>...</answer>.<|im_end|>
<|im_start|>user
74 5 20 88
<|im_end|>
<|im_start|>assistant
```

远端输出目录：

```text
outputs/experiments/handoff3_countdown_minimal/eval_handoff2_minimal_greedy16
```

评估口径：

- model base：`outputs/experiments/baseline_format_v2_full_5000_from800/final`
- LoRA checkpoint：`outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500`
- prompt style：`qwen_chat_minimal_target`
- max_new_tokens：`4096`
- batch_size：`4`
- decoding：greedy

结果：

| total | solved | solve_rate | format_ok | valid_expr |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0.00% | 1 | 0 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 15 |
| `wrong_numbers` | 1 |

结论：最小迁移 prompt 恢复了 handoff2 的 rollback/search 行为；当前主要失败
不是输出语义错乱，而是模型在 4096 token 内没有稳定闭合 `<answer>`。因此仍需
sampled audit 判断采样空间里是否有正确轨迹。

### Minimal-transfer sampled audit

先尝试 32 题、每题 8 采样、4096 token：

```text
outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_32_g8
```

该审计运行 60 分钟仍无 summary，GPU 持续生成，判断为成本过高后停止。随后改跑
8 题、每题 4 采样、4096 token：

```text
outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_8_g4
```

结果：

| prompts | samples / prompt | total outputs | solved | pass@4 | format_ok | valid_expr | len mean | len p50 | len p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 32 | 0 | 0/8 | 3 | 0 | 4096 | 4096 | 4096 |

失败分布：

| reason | count |
| --- | ---: |
| `answer_contract:expected exactly one <answer>...</answer> block` | 29 |
| `wrong_numbers` | 3 |

典型 answer-contract 失败：prompt 已经是 `Play the 50-point game`，user 只有
`74 5 20 88`，输出恢复为 rollback/search，但完整 4096 token 仍没有闭合
`<answer>`。

典型 wrong_numbers 失败：`26 4 2 -> target 44`，模型中间写出
`(2) * (22) = 44`，但最终回填为
`<answer>(2 * ((26 - 4) - 2))</answer>`，表达式实际值为 40，且重复使用 `2`。

结论：handoff2 的最小迁移 zero-shot / sampled audit 均没有给出可直接 GRPO 的
正样本信号。下一步应先用 `qwen_chat_minimal_target` 重新构建 Countdown SFT
数据，再审计 sampled success。

## SFT Adaptation 记录

### 400 条 / 200 step 版本

训练数据：

```text
outputs/experiments/handoff3_countdown_adapt/sft/countdown-solver-sft-400.jsonl
```

训练输出：

```text
outputs/experiments/handoff3_countdown_adapt/sft_train_handoff2_continue_400_200steps/final
```

训练配置：

| item | value |
| --- | --- |
| start base | `outputs/experiments/baseline_format_v2_full_5000_from800/final` |
| initial adapter | `outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500` |
| max_steps | 200 |
| learning_rate | `1e-5` |
| max_length | 4096 |
| train records | 400 |

训练结果：`train_loss=0.3048`。partial eval 前 8 条只有 `1/8` 正确，
多条仍是长搜索或 answer contract 失败。因此停止该版本完整评估，避免继续浪费评估时间。

### 2000 条 / 600 step 版本

训练数据：

```text
outputs/experiments/handoff3_countdown_adapt/sft/countdown-solver-sft-2000.jsonl
```

数据构建口径：

- 从本地 `default__train.jsonl` 顺序扫描；
- 排除 100 题 stratified eval manifest 中的题；
- DFS 生成 1 条 verified solver trace；
- prompt 使用 `qwen_chat_target`；
- completion 使用 `checked_success`；
- 扫描 2092 行，得到 2000 条训练记录。

训练输出：

```text
outputs/experiments/handoff3_countdown_adapt/sft_train_handoff2_continue_2000_600steps/final
```

训练配置：

| item | value |
| --- | --- |
| start base | `outputs/experiments/baseline_format_v2_full_5000_from800/final` |
| initial adapter | `outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500` |
| max_steps | 600 |
| learning_rate | `2e-5` |
| max_length | 4096 |
| train records | 2000 |

训练结果：`train_loss=0.09375`。

smoke eval：

| eval set | total | solved | solve_rate | format_ok | valid_expr | main failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| first-16 from stratified manifest | 16 | 1 | 6.25% | 16 | 16 | `wrong_value` 15 |
| balanced-16, 4 per bucket | 16 | 1 | 6.25% | 16 | 16 | `wrong_value` 15 |

sampled audit：

| prompts | samples / prompt | total outputs | solved | pass@k | mixed groups | all-wrong groups |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 128 | 1 | 1/16 | 1 | 15 |

sampled audit 失败分布：

| reason | count |
| --- | ---: |
| `wrong_value` | 116 |
| `wrong_numbers` | 5 |
| `syntax_error:'(' was never closed` | 4 |
| `syntax_error:unmatched ')'` | 2 |
| `ok` | 1 |

结论：2000 条 SFT 已经基本修好输出契约和 target prompt 格式，
但没有学会可靠搜索到任意 target。当前 sampled success 太低，直接做 GRPO
缺少足够正样本信号，暂不进入 GRPO 长训。
