# 给队友的简短交接：handoff2 GRPO 补充实验

一句话版：

> 我们没有重新训练 SFT，而是基于 handoff1 的强 SFT-final 模型继续做了一版更贴合
> 4096 token 口径的 GRPO。最终交付的是 `checkpoint-500`：
> validation `128/136 = 94.12%`，test `129/137 = 94.16%`。
> 它比 handoff1 SFT-final 的 4096 结果更强，说明精心设计的 GRPO 后训练确实带来了增益。

Hugging Face 留档：

- handoff2 GRPO checkpoint-500：<https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500>
- handoff1 SFT-final 基座：<https://huggingface.co/Prometheus17/game24-rl/tree/main/sft-final>

## 这次实验接在 handoff1 后面

handoff1 已经跑通了完整路线：

```text
Qwen2.5-1.5B-Instruct -> full SFT final -> GRPO LoRA adapter -> decoding/eval
```

handoff1 最关键的结论是：

- SFT-final 已经是强 baseline；
- 1024 下的失败主要是 no-answer / answer-contract，也就是没有及时闭合答案；
- 打开到 4096 后，SFT 自己就能显著提升；
- 早期 GRPO 有信号，但主要还是在 1024 问题形态下做出来的。

因此 handoff2 的目标不是重做 SFT，也不是重做数据切分，而是：

> 直接从 handoff1 SFT-final 出发，在 4096 口径下做一版更保守、更明确鼓励正确且早闭合的 GRPO。

## 使用的基座模型

基座模型是 handoff1 的 SFT-final：

```text
outputs/experiments/baseline_format_v2_full_5000_from800/final
```

Hugging Face 对应目录：

```text
https://huggingface.co/Prometheus17/game24-rl/tree/main/sft-final
```

这点很重要：handoff2 不是从 base model 训练，也不是从新 SFT checkpoint 训练。
它是对 handoff1 已交付强 SFT 模型做强化学习后训练。

## GRPO 设计口径

这次 GRPO 的设计文档是：

```text
docs/experiments/handoff1_grpo_final_plan_20260619.md
```

实际执行的核心口径：

- 训练 split：只用 `standard-game24-v1` 的 train split。
- validation/test：只用于评估，不进入训练。
- rollout / train / eval token budget：统一 `4096`。
- GRPO group：同一道 puzzle 内采样多个 completion 做组内比较，不跨题比较 reward。
- ToT rank：只用于控制 prompt pool / sampler，不用于跨难度题目直接比较 reward。
- 训练方式：LoRA GRPO adapter。
- reward profile：`closure_control_smooth`。

reward 的主信号仍然是 strict verifier correctness。正确答案拿 `1.0` 基础分，
再根据 `<answer>` 闭合 token 位置给最多 `0.25` 的早闭合 bonus。
错误答案按 no-answer、wrong numbers、wrong value、syntax 等类型给负分。

这版 reward 的意图很明确：

> 不改变 verifier，也不放宽答案格式；只是在正确的前提下，鼓励模型更早收束到 `<answer>`。

## 实际训练状态

原计划是训练 `2000` steps，每 `500` steps 保存一次 checkpoint。

但由于时间原因，我们在 `checkpoint-500` 停下并做评估。因此 handoff2 交付模型是：

```text
outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500
```

Hugging Face 对应目录：

```text
https://huggingface.co/Prometheus17/game24-rl/tree/main/handoff2-grpo-checkpoint-500
```

不是 2000-step final。

训练配置摘要：

```text
num_generations = 8
max_completion_length = 4096
max_steps = 2000
save_steps = 500
learning_rate = 5e-6
beta = 0
scale_rewards = none
loss_type = dr_grpo
mask_truncated_completions = false
training mode = LoRA adapter
```

## 最终评估结果

评估方式：

- direct greedy decoding；
- `max_new_tokens=4096`；
- 不使用 verifier rerank；
- 从 raw outputs 重新运行 strict verifier；
- 再按 ToT rank 分桶统计。

详细结果表在：

```text
docs/handoff/handoff2_GRPO_result.md
```

注意：这里的 `val+test` 是 validation 和 test 合并后的 273 题统计，不是 1362 题全量统计。

核心数字：

| model | validation | test | val+test |
| --- | ---: | ---: | ---: |
| Handoff1 SFT-final 4096 | 123/136 = 90.44% | 128/137 = 93.43% | 251/273 = 91.94% |
| Handoff1 best GRPO 4096 | 126/136 = 92.65% | 129/137 = 94.16% | 255/273 = 93.41% |
| Handoff2 GRPO checkpoint-500 4096 | 128/136 = 94.12% | 129/137 = 94.16% | 257/273 = 94.14% |

所以按同样的 validation/test 4096 greedy 口径：

- 相比 handoff1 SFT-final：validation `+5/136`，test `+1/137`，val+test `+6/273`。
- 相比 handoff1 best GRPO：validation `+2/136`，test 持平，val+test `+2/273`。

这说明 handoff2 的 GRPO 不是只有训练 reward 变好，而是在最终 strict eval 上也有小幅实打实提升。

## 分难度表现

handoff2 在 4096 下的 val+test 合并结果：

| bucket | solved / total | acc | main failures |
| --- | ---: | ---: | --- |
| Easy 1-300 | 58/62 | 93.55% | 2 no-answer, 1 unsupported-expression, 1 wrong-numbers |
| Medium 301-900 | 120/122 | 98.36% | 2 no-answer |
| Hard 901-1100 | 40/43 | 93.02% | 3 no-answer |
| Very hard 1101-1362 | 39/46 | 84.78% | 7 no-answer |
| Overall | 257/273 | 94.14% | 14 no-answer, 1 unsupported-expression, 1 wrong-numbers |

最明显的点是：

- medium 非常稳，val+test 达到 `120/122 = 98.36%`；
- hard 也稳定在 `40/43 = 93.02%`；
- very hard 仍然是主要损失来源，但比 handoff1 SFT-final 的 validation very hard 明显改善；
- 失败类型仍以 no-answer 为主，说明“答案闭合”仍然是后续最值得优化的问题。

## 怎么解释这次结果

这次结果支持一个比较稳的结论：

> handoff1 的强 SFT-final 已经学会了大量 24 点搜索能力；handoff2 的 GRPO 主要是在这个基础上，
> 把一部分本来会长搜索或不闭合答案的题，推向了更可评测、更及时闭合的 greedy 输出。

它不是从零学会 24 点，也不是靠 verifier rerank 后处理作弊。
评估时就是单模型 direct greedy 输出，然后 strict verifier 检查 `<answer>` 里的表达式。

同时也要实事求是：

- 这版只评估了 validation/test，没有跑 train split eval。
- 这版没有跑 1024，因为当前展示口径已经统一到 4096。
- 这版只训练到 `checkpoint-500`，不是完整 2000-step final。
- test 没有超过 handoff1 best GRPO，主要提升体现在 validation 和 val+test 合并。

## 留档位置

建议后续阅读顺序：

1. `docs/handoff/handoff1.md`
   先看 handoff1 为什么认为 SFT 已经很强、GRPO 有信号。

2. `docs/handoff/SFT_result_summary.md`
   看 handoff1 SFT-final 的完整 1024 / 4096 结果。

3. `docs/handoff/handoff1_GRPO_result.md`
   看 handoff1 早期 best GRPO 的 1024 / 4096 结果。

4. `docs/experiments/handoff1_grpo_final_plan_20260619.md`
   看 handoff2 这版 GRPO 的设计口径。

5. `docs/handoff/handoff2_GRPO_result.md`
   看 handoff2 checkpoint-500 的最终 eval 表。

## 当前可交付说法

如果要在报告里压缩成一段话，可以这样写：

> 在 handoff1 的 SFT-final 模型基础上，我们进一步进行了 GRPO 后训练。与早期 GRPO 不同，
> 这次训练统一采用 4096 token 预算，并设计了 `closure_control_smooth` reward：
> strict verifier 正确性是主奖励，同时对更早闭合 `<answer>` 的正确输出给予小幅连续 bonus。
> 训练到 checkpoint-500 后，在 repo-local validation/test 上进行 direct greedy 4096 评估，
> 分别达到 `128/136 = 94.12%` 和 `129/137 = 94.16%`。相比 SFT-final 的
> `123/136 = 90.44%` 和 `128/137 = 93.43%`，GRPO 在相同评估口径下带来了进一步提升。
