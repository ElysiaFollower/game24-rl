# Handoff1 GRPO 最终执行计划

## 状态

本文记录 2026-06-19 确认后的 GRPO 执行口径。该方案已经定版，后续实现和远端训练应按本文执行；除非负责人重新确认，不再自行调整 reward、token budget、steps、split、base checkpoint 或评估口径。

## 1. 基座模型

使用 handoff1 的强 SFT-final：

```text
outputs/experiments/baseline_format_v2_full_5000_from800/final
```

不重新训练 SFT。GRPO 直接从该模型继续后训练。

## 2. 训练数据

只使用 `standard-game24-v1` 的 train split 做 GRPO。

validation/test 只用于评估，不进入训练。

## 3. GRPO 分组

GRPO group 必须按单题构造：

```text
同一道 puzzle -> 采样 G 个 completion -> 组内比较 reward
```

不让简单题和难题的 completion 直接互相比 reward。

## 4. ToT Rank 的作用

ToT rank 只用于控制 prompt pool / sampler：

- hard / very hard 保持足够占比；
- easy / medium 保留一部分做 retention anchor；
- 优先使用 mixed-signal 题目；
- 避免 all-correct / all-wrong 题目占太多训练步数。

ToT rank 不用于跨题 reward 比较。不同难度题目不能在同一个 GRPO group 内竞争。

## 5. Token 设置

全部统一：

```text
max_new_tokens = 4096
```

rollout、GRPO 训练、最终评估都用 4096。

## 6. Reward 设计

使用 reward profile：

```text
closure_control_smooth_v1
```

正确答案：

```text
reward = 1.0 + 0.25 * (1 - answer_close_token / 4096)
```

含义：

- strict verifier 通过是主信号；
- 正确 reward 范围约为 `1.0 ~ 1.25`；
- 越早闭合 `<answer>`，reward 越高；
- 晚闭合不扣分，只是少拿 close bonus；
- 不使用 `max()` 截断；
- 不使用 `after_answer_penalty`。

错误答案：

```text
no complete answer: -0.30
multiple answer blocks: -0.35
syntax / unsupported expression: -0.30
wrong numbers: -0.35
wrong value: -0.35
```

Verifier 继续使用仓库 strict verifier；不放宽 answer contract，不使用 Python `eval`。

## 7. GRPO 配置

初始配置：

```text
num_generations = 8
max_new_tokens = 4096
max_steps = 2000
save_steps = 500
learning_rate = 5e-6
beta = 0
scale_rewards = none
mask_truncated_completions = false
remove_unused_columns = false
training mode = LoRA adapter
```

每 500 step 保存一个 checkpoint。

如果 mixed-signal 不够，再由负责人确认是否把 `num_generations` 提到 16；不要在无人确认时自行改动。

## 8. 评估

GRPO 完成后跑：

- validation 4096；
- test 4096。

从 raw outputs 重新统计：

- overall accuracy；
- ToT rank 分层准确率；
- no-answer / tag error / wrong numbers / wrong value 分布；
- easy/medium retention；
- hard/very hard 提升情况。

评估必须保存 raw outputs，并用 strict verifier 从原始输出重新计算结果。

## 9. 一句话执行口径

用 handoff1 SFT-final 做基座，在 train split 上按单题 group 做 GRPO；ToT rank 只控制采样分布；reward 以 strict correctness 为主，连续鼓励早闭合；统一 4096 token，训 2000 steps，每 500 step 保存 checkpoint，然后做 val/test 4096 评估。
