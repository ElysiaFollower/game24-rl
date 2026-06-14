# 24 点 RL 专用模型：论文、强化学习原理与开发前技术手册

调研日期：2026-06-13  
用途：为后续真正开发、实验设计、训练调参、报告写作做技术准备  
核心任务：Qwen2.5-1.5B-Instruct 在 24 点游戏上的 RLVR/GRPO 训练

## 0. 这份文档怎么用

这不是普通文献综述，而是开发前的工程技术手册。建议后续按下面顺序使用：

1. 开发环境前，先读第 1-3 节，确定为什么这个任务应该建模成 RLVR。
2. 写 `verifier/reward/train_grpo.py` 前，重点读第 4-7 节。
3. 跑实验和调参时，重点读第 8-10 节。
4. 写报告/PPT 时，直接抽第 11-13 节的表述。

结论先说：24 点项目的胜负不在“多蒸馏几万条推理”，而在训练系统是否能稳定地产生有效策略梯度。你们要重点监控：

- 同一 prompt 的 G 个 rollout 是否有 reward 方差。
- correctness 是否真的涨，而不是 reward 被格式或长度 hack。
- pass@k 是否高于 greedy，判断瓶颈是 capability 还是 capture。
- 失败类型是算错、漏数、格式错，还是搜索失败。
- policy entropy、completion length、KL 是否进入异常区间。

## 1. 文献地图

### 1.1 强化学习基础

| 来源 | 主要内容 | 对 24 点项目的作用 |
|---|---|---|
| Sutton et al., 1999, Policy Gradient Methods | 直接优化随机策略的期望回报 | 解释为什么 LM token policy 可以用 reward 更新 |
| Schulman et al., 2015, GAE | 用 advantage 降低 policy gradient 方差 | 解释 baseline/advantage 的必要性 |
| Schulman et al., 2015, TRPO | 用 KL trust region 限制策略更新 | 解释为什么 RL 微调不能让模型分布乱飘 |
| Schulman et al., 2017, PPO | clipped surrogate objective | PPO/GRPO 的直接理论祖先 |
| OpenAI Spinning Up | RL/PPO 教程 | 最适合补理论缺口 |
| Hugging Face Deep RL Course | policy gradient/PPO 实践教程 | 适合组员快速补 RL 基础 |

### 1.2 LLM 对齐与 RLHF

| 来源 | 主要内容 | 对 24 点项目的作用 |
|---|---|---|
| Ziegler et al., 2019 | 语言模型从人类偏好微调 | 解释 LM + RL 的早期范式 |
| Stiennon et al., 2020 | 摘要任务上 RLHF 优于 SFT/ROUGE | 说明“训练目标决定行为” |
| Ouyang et al., 2022, InstructGPT | SFT + RM + PPO | 解释冷启动和 KL 的工程价值 |
| Huang et al., 2024, N+ Implementation Details | RLHF/PPO 复现细节 | 说明实现细节会极大影响结果 |
| RLHF Book | post-training 算法教程 | PPO/RLOO/GRPO 对比入门 |

### 1.3 可验证奖励与推理 RL

| 来源 | 主要内容 | 对 24 点项目的作用 |
|---|---|---|
| DeepSeekMath, 2024 | GRPO + 数学推理 | 课程指定 GRPO 源头 |
| DeepSeek-R1, 2025 | R1-Zero/R1，多阶段 RLVR | 说明 pure RL 可行但冷启动更稳 |
| TinyZero | 小模型 countdown RL 复现 | 最接近 24 点的开源参照 |
| Logic-RL, 2025 | rule-based RL 训练逻辑题 | 支持“合成可验证任务”路线 |
| DAPO, 2025 | Clip-Higher、Dynamic Sampling、token-level loss | 直接指导 GRPO 稳定性 |
| Understanding R1-Zero-Like Training, 2025 | GRPO 长度偏置、Dr.GRPO | 直接指导 loss_type 和 length 监控 |
| REINFORCE++, 2025 | critic-free 稳定 RLHF | 作为 GRPO 失败时的备选 |
| ReMax, 2023 | 简化 trajectory-level reward RL | 说明小项目可不用复杂 critic |

### 1.4 推理、搜索、验证与数据自举

| 来源 | 主要内容 | 对 24 点项目的作用 |
|---|---|---|
| Chain-of-Thought, 2022 | 中间推理提升复杂任务 | 解释为什么需要 `<think>` |
| Zero-shot-CoT, 2022 | 简单提示可诱发推理 | prompt baseline |
| Self-Consistency, 2022 | 多路径采样投票提升推理 | pass@k 和 capture gap 的理论依据 |
| STaR, 2022 | 正确 rationale 自举 SFT | solver/self-generated traces 的先例 |
| Training Verifiers, 2021 | verifier reranking 数学答案 | 说明生成多个候选 + 验证器很强 |
| Let's Verify Step by Step, 2023 | process supervision > outcome supervision | 支持 near-miss/self-check curriculum |
| Tree of Thoughts, 2023 | 24 点需要搜索和回溯 | 说明 24 点不是普通算术 |
| Grokking, 2022 | 小型算法数据集可能晚泛化 | 说明 24 点小数据训练要防止误判欠训练 |

## 2. 强化学习理论背景：从 policy gradient 到 LLM token policy

### 2.1 标准 RL 视角

强化学习问题通常写成 MDP：

```text
state s_t
action a_t
reward r_t
policy pi_theta(a_t | s_t)
objective J(theta) = E_pi[sum_t gamma^t r_t]
```

在 LLM 中：

- state：prompt 加已生成 token 前缀。
- action：下一个 token。
- episode：一次完整 completion。
- reward：verifier 对完整 completion 的评分。
- policy：语言模型 `pi_theta(token | prompt, prefix)`。

24 点任务可以看成极短 episode 的 sparse-reward RL：

```text
prompt: numbers = [a,b,c,d], target = 24
completion: <think>...</think><answer>expr</answer>
reward: verifier(expr, numbers, target)
```

这里没有传统环境中的多步交互状态更新；环境只在结尾给 reward。但 token-by-token 生成本身已经形成序列决策。

### 2.2 Policy gradient 的核心

最朴素的 REINFORCE：

```text
grad J(theta) = E[ R(o) * grad log pi_theta(o | q) ]
```

对一个 completion `o = [y_1, ..., y_T]`：

```text
log pi_theta(o | q) = sum_t log pi_theta(y_t | q, y_<t)
```

如果某个 completion reward 高，就提高整条轨迹里 token 的概率；reward 低，就降低概率。

问题是方差很大，所以引入 baseline：

```text
A(o) = R(o) - b(q)
grad J(theta) = E[ A(o) * grad log pi_theta(o | q) ]
```

`A` 是 advantage。它不是“这个样本绝对好不好”，而是“相对同类样本好不好”。

对 24 点的含义：

- 如果同一个题的 8 个回答全错，所有 reward 都一样，advantage 约为 0。
- 如果 8 个回答里 1 个对、7 个错，这个题提供强学习信号。
- 因此 GRPO 训练集不是越难越好，而是“模型偶尔能做对”的题最好。

这就是 active difficulty sampling 的理论根基。

## 3. PPO、GRPO、RLVR：为什么课程指定 GRPO

### 3.1 PPO 解决什么问题

PPO 的动机是：policy gradient 如果步子太大，模型分布会突然崩掉。PPO 使用 ratio：

```text
r_t(theta) = pi_theta(y_t | prefix) / pi_old(y_t | prefix)
```

再用 clipped objective：

```text
L_clip = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
```

直觉：

- 如果某 token 因为正 advantage 应该被提高，PPO 不允许提高太猛。
- 如果某 token 因为负 advantage 应该被降低，PPO 不允许降低太猛。
- 这是一种便宜的 trust region。

在 LLM 上，还常加 reference KL：

```text
reward_total = reward_task - beta * KL(pi_theta || pi_ref)
```

作用是防止模型为了 reward 偏离原模型太远。

### 3.2 PPO 在 LLM 上为什么重

传统 PPO 通常需要：

- policy model。
- reference model。
- value/critic model。
- reward model 或 rule reward。

对 1.5B 模型和 4090 来说，critic 是额外显存和不稳定源。

24 点是 terminal reward，而且 reward 可精确验证，所以不需要学 reward model，也不一定需要 value model。这就是 GRPO、RLOO、ReMax、REINFORCE++ 这类 critic-free 算法有吸引力的原因。

### 3.3 GRPO 的核心

GRPO 对同一个 prompt 采样 G 个 completions：

```text
o_1, ..., o_G ~ pi_old(. | q)
r_i = verifier(o_i)
```

组内标准化：

```text
A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
```

然后用类似 PPO 的 clipping objective 更新 policy。

好处：

- 不训练 value model。
- advantage 来自同题多样本比较。
- 对可验证任务非常自然。

坏处：

- 需要 `num_generations = G`，rollout 成本高。
- 如果组内 reward 全一样，梯度消失。
- 组内标准差归一化会引入题目难度偏置。
- 原始 GRPO 有长度偏置，需要小心 loss formulation。

### 3.4 RLVR 与 RLHF 的关键差别

RLHF：

```text
human preference -> reward model -> RL optimize learned reward
```

RLVR：

```text
program verifier -> deterministic reward -> RL optimize verifier result
```

24 点属于非常干净的 RLVR：

- 正确性可自动验证。
- 不需要人类偏好。
- 不需要 reward model。
- 可以生成无限训练题。
- reward hacking 主要来自 verifier 漏洞和奖励塑形，而不是 reward model 泛化错误。

但 RLVR 也有边界：

- 它只优化 verifier 能看到的东西。
- 如果 verifier 只看最终值，模型可能漏用数字但凑出 24。
- 如果 verifier 只看 answer，think 是否真实无法保证。
- 如果 reward 太 sparse，训练可能完全没信号。

所以 24 点项目的 verifier 必须比“算出来等于 24”更严格。

## 4. 从论文中提炼出的训练认知

### 4.1 DeepSeek-R1 的经验：pure RL 可行，但 cold-start 更稳

DeepSeek-R1-Zero 显示，在大模型和大规模 RL 下，不经 SFT 也可能出现自检、反思、长推理等行为。但同一报告也指出 pure RL 存在可读性差、语言混杂、格式不稳定等问题，DeepSeek-R1 因此引入 cold-start 和多阶段训练。

对 24 点项目的判断：

- 不建议直接从 Qwen2.5-1.5B-Instruct 纯 GRPO 起步。
- 应该先做短 trace solver-SFT，把模型推到“有正样本”的区域。
- 你可以在报告中做一个 pure-GRPO baseline，但不要把它当主路线。

工程建议：

```text
Base -> short solver SFT -> GRPO
```

比：

```text
Base -> GRPO only
```

更稳、更容易解释、更符合 4090 算力。

### 4.2 DeepSeekMath 的经验：GRPO 是 PPO 的省显存版本，不是魔法

DeepSeekMath 引入 GRPO 的核心动机之一是省掉 critic，降低 PPO 内存占用，同时保持数学推理 RL 的效果。

对 24 点项目的判断：

- GRPO 适合单卡 4090。
- 但 GRPO 能否学到东西取决于 reward 方差和 rollout 质量。
- 不要用“用了 GRPO”替代对 reward、采样、curriculum 的设计。

开发指标：

```text
group_reward_std_mean
zero_std_group_rate
group_acc_0_rate
group_acc_1_rate
```

其中：

- `group_acc_0_rate` 高：题太难或模型太弱。
- `group_acc_1_rate` 高：题太容易，无训练信号。
- `zero_std_group_rate` 高：GRPO 这一步基本浪费。

### 4.3 DAPO 的经验：有效样本比总样本更重要

DAPO 提出四个关键技巧：

1. Clip-Higher：给低概率但有价值的探索 token 更多上升空间，缓解 entropy collapse。
2. Dynamic Sampling：过滤全对/全错等零梯度样本，提高训练效率。
3. Token-level Policy Gradient Loss：避免 sample-level loss 在长 CoT 下的偏差。
4. Overlong Reward Shaping：不要简单惩罚被截断的长样本，否则引入 reward noise。

对 24 点项目最重要的是 Dynamic Sampling：

```text
for each prompt:
    sample G completions
    if all rewards same:
        skip or downweight
    else:
        train on this group
```

这和你们的任务非常契合，因为 24 点全空间小，可以按当前模型成功率主动选题。

项目里的等价实现：

```text
每隔 N step 估计每题 pass@G
保留 0 < correct_count < G 的题作为主训练池
correct_count = 0 的题进入 later/hard buffer
correct_count = G 的题进入 replay/easy buffer，低频采样
```

### 4.4 Dr.GRPO 的经验：长度不是越长越好，原始 GRPO 有长度偏置

“Understanding R1-Zero-Like Training” 指出原始 GRPO 可能产生 response length bias，尤其会让错误回答也变长。TRL 当前文档也明确提供 `loss_type="dr_grpo"` 和 `loss_type="dapo"` 来处理长度归一化问题。

对 24 点项目的判断：

- 24 点不需要 2k token 长推理。
- 长输出会增加截断、格式错、重复、幻觉。
- 你们的 `<think>` 应该短、结构化、可验证。

建议配置：

```text
max_completion_length: 256 或 384
loss_type: "dr_grpo" 或 "dapo"
scale_rewards: 先用 "group"，再消融 "none"
```

按当前 TRL 文档，`loss_type` 默认是 `"dapo"`，可显式设为 `"dr_grpo"`；`scale_rewards` 默认是 `"group"`，Dr.GRPO 相关讨论建议消融 `"none"`，因为组内标准差缩放可能引入题目难度偏置。不要在报告中只写“用了 GRPO”，要写清楚具体 loss formulation。

同时监控：

```text
completion_len_mean
completion_len_p95
truncation_rate
len_correct_mean
len_wrong_mean
```

如果 `len_wrong_mean` 越来越长但 solve rate 不涨，就是长度偏置或 reward hack。

### 4.5 Logic-RL 的经验：格式奖励有用，但也危险

Logic-RL 报告强调严格格式 prompt 和格式 reward 可以避免模型走捷径。但 24 点项目里，格式比最终 correctness 更容易学。如果格式奖励太大，模型会先刷格式。

建议：

```text
format bonus <= 0.05
format bonus only if answer parseable
main correctness reward = 1.0
invalid/wrong-number penalty magnitude > format bonus
```

不要这样：

```text
format = +0.3
correct = +1.0
```

因为在 early training 中，format reward 密度远高于 correct reward，梯度会被格式主导。

### 4.6 Verifier 论文的经验：生成能力和选择能力要分开看

Training Verifiers 和 Self-Consistency 都说明：模型一次生成不对，不代表它完全不会；多采样 + 选择能显著提高表现。

对 24 点项目的关键指标是：

```text
greedy pass@1
sampling pass@1
pass@4
pass@16
pass@64
```

解释：

- 如果 pass@64 高但 greedy 低，模型有 latent capability，瓶颈是 capture。
- 如果 pass@64 也低，模型根本没学会，需要更强 SFT/curriculum。
- 如果 RL 后 greedy 涨但 pass@64 降很多，可能是分布塌缩。
- 如果 RL 后 pass@64 和 greedy 都涨，才是真正的能力提升。

这对报告非常重要：不要只报单发准确率。

### 4.7 Process supervision 的经验：过程监督能缓解 credit assignment

Let's Verify Step by Step 显示，对多步数学推理，逐步过程监督比只看最终答案更可靠。24 点虽然 reward 可以最终验证，但模型常见错误是中间步骤不自检。

不建议你们训练一个 PRM，太重。推荐轻量替代：

- 用 solver 生成 step trace。
- 生成 near-miss 数据：值对但漏数、数字对但值错、非法除法。
- 训练模型在 `<think>` 中显式做两类检查：
  - value check：表达式是否等于 24。
  - multiset check：四个数是否各用一次。

示例：

```text
<think>
Try 13+7+4=24, but it skips 9, reject.
Use 9-7=2; 13-1=12; 12*2=24.
Check value=24 and numbers 1,7,9,13 are each used once.
</think>
<answer>((13-1)*(9-7))</answer>
```

这不是 PRM，但能把 verifier 的两个关键约束注入模型行为。

### 4.8 ToT 的经验：24 点本质是搜索，不是语言推理

Tree of Thoughts 在 Game of 24 上显示普通 CoT 单路推理很弱，而树搜索显著提升。这说明 24 点的核心难点是：

- 第一步合并选择错，后面无法恢复。
- 单条 left-to-right 解码缺少回溯。
- 需要比较多个候选中间状态。

训练时应把 ToT 的搜索结构蒸馏成短 trace，而不是模仿自然语言长思考。

推荐 trace 模板：

```text
<think>
Goal 24. Make 6 and 4: 8-2=6; 7-3=4; 6*4=24.
</think>
<answer>((8-2)*(7-3))</answer>
```

或状态式：

```text
<think>
[8,2,7,3] -> [6,7,3] by 8-2=6 -> [6,4] by 7-3=4 -> [24] by 6*4.
</think>
<answer>((8-2)*(7-3))</answer>
```

状态式更像算法数据，适合你“专用模型可以丧失语言能力”的目标。

### 4.9 Grokking 的经验：小算法数据集上，欠训练和过拟合要分开

24 点全空间只有 1,820 个 multiset，很容易出现：

- 训练集 loss 很低。
- train solve rate 很高。
- heldout 不涨。

也可能出现：

- 小数据训练步数太少，看起来 scaling 差。
- 多训练很久后才泛化。

因此实验必须控制：

- 数据量。
- 总 training steps。
- epoch 数。
- token 数。
- eval split 是否按 multiset 去重。

不要把“3k 数据效果差、30k 数据效果好”直接解释为数据 scaling；可能只是 3k 欠训练。

## 5. 技术路线：从环境到模型

### 5.1 任务建模

把 24 点建成一个可验证任务环境：

```python
Task = {
    "numbers": [a,b,c,d],
    "target": 24,
    "solvable": True/False,
    "difficulty": {...}
}

Policy input:
    "Use numbers [a,b,c,d] exactly once to make 24..."

Policy output:
    "<think>...</think><answer>expr</answer>"

Reward:
    verifier(expr, numbers, target)
```

不要把 solver 暴露给模型推理期。solver 只用于：

- 数据生成。
- reward 校验。
- eval。
- difficulty 分层。

### 5.2 数据生成顺序

阶段 0：全空间枚举

```text
combinations_with_replacement(range(1,14), 4)
total = 1820
solvable = 1362
unsolvable = 458
```

阶段 1：表达式求值

```text
prompt: Compute: ((8-2)*(7-3))
answer: 24
```

作用：提升基础四则运算和括号理解。

阶段 2：一步合并

```text
prompt: From [8,2,7,3], choose a useful pair operation.
answer: 8-2=6, new state [6,7,3]
```

作用：学组合搜索的局部动作。

阶段 3：三数/二数子问题

```text
prompt: Use [6,7,3] to make 24.
answer: (7-3)*6
```

作用：学递归子问题。

阶段 4：完整 24 点

```text
prompt: Use [8,2,7,3] to make 24.
answer: ((8-2)*(7-3))
```

阶段 5：near-miss 自检

```text
prompt: Use [4,7,9,13] to make 24.
think: Try 13+7+4=24 but skips 9. Reject...
answer: ...
```

阶段 6：无解拒答，晚期少量

```text
prompt: Use [1,1,1,1] to make 24.
answer: NO_SOLUTION
```

无解训练比例建议小于 10%，否则模型会在可解题上逃避。

### 5.3 三种输出风格建议

风格 A：自然语言短 CoT

优点：报告好看。缺点：token 多。

```text
<think>First make 6 by 8-2, then make 4 by 7-3. Finally 6*4=24.</think>
<answer>((8-2)*(7-3))</answer>
```

风格 B：状态转移 trace

优点：最适合专用算法模型。

```text
<think>[8,2,7,3] -> [6,7,3] -> [6,4] -> [24]</think>
<answer>((8-2)*(7-3))</answer>
```

风格 C：动作编码

优点：最短；缺点：展示不直观，需解释。

```text
<think>op(8,2,-)=6; op(7,3,-)=4; op(6,4,*)=24</think>
<answer>((8-2)*(7-3))</answer>
```

建议主路线用 B/C 混合。保留一点可读性，但不要追求长推理。

## 6. Reward 设计：主奖励、辅助奖励和陷阱

### 6.1 Verifier 分类

Verifier 不应只返回 bool，而应返回结构化结果：

```python
VerifyResult(
    ok: bool,
    status: Literal[
        "correct",
        "no_answer",
        "parse_error",
        "illegal_token",
        "division_by_zero",
        "wrong_numbers",
        "wrong_value",
        "timeout",
        "abstain"
    ],
    value: Fraction | None,
    used_numbers: list[int],
)
```

这样 reward、eval、失败分析都能复用。

### 6.2 推荐 reward v1

```text
correct: +1.0
right_numbers_wrong_value: -0.15
wrong_numbers_value_24: -0.35
parse_error/no_answer: -0.5
illegal_token/division_by_zero: -0.7
```

format bonus：

```text
if has_think_answer_tags and parseable:
    +0.03
else:
    +0
```

重要：format bonus 必须小，并且不要给无法 parse 的输出。

### 6.3 推荐 reward v2：correctness-gated length

只对 correct 输出给一点长度/trace 奖励：

```text
if correct and 20 <= think_tokens <= 120:
    reward += 0.05
```

不要对错误输出给“推理很长”的奖励。

### 6.4 推荐 reward v3：无解题

对 `solvable=False`：

```text
NO_SOLUTION: +1.0
fabricated expression: -1.0
parse_error/no_answer: -0.3
```

但无解题训练必须晚期、小比例。否则模型可能学会保守拒答。

### 6.5 不推荐的 reward

不推荐：

```text
format +0.3
mentions all numbers +0.3
has plus/minus/mul/div +0.2
length > 200 +0.2
correct +1.0
```

原因：

- 辅助奖励过密，正确性过稀。
- 模型会学会“像在推理”，不是学会解题。
- reward 曲线会上升，solve rate 不一定上升。

## 7. GRPO 训练配置建议

### 7.1 TRL 起步配置

优先用 TRL，因为单卡最省事。

建议起步：

```yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct
max_prompt_length: 256
max_completion_length: 256
num_generations: 8
temperature: 0.8
top_p: 0.95
learning_rate: 5.0e-6  # LoRA GRPO 可从 5e-6 到 2e-5 试
beta: 0.0 或 0.01
loss_type: dr_grpo
scale_rewards: group
gradient_accumulation_steps: 4-16
bf16: true
peft_lora_rank: 32 或 64
```

当前 TRL 文档中，GRPOTrainer 支持自定义 reward function、`loss_type="dapo"`、`loss_type="dr_grpo"`、`scale_rewards` 等选项，并会记录 `frac_reward_zero_std` 这类对 GRPO 特别关键的诊断指标。工程上优先不要魔改 trainer，先把 reward 和 logging 做扎实。

### 7.2 什么时候用 `dapo`，什么时候用 `dr_grpo`

建议：

- 默认试 `dr_grpo`，因为 24 点不需要长 CoT，想避免长度偏置。
- 如果发现训练太保守、entropy 很快塌缩，可以试 `dapo`。
- 原始 `grpo` 不作为主线，只做消融。

### 7.3 `scale_rewards` 的消融

TRL 当前默认 group scaling：

```text
A_i = (r_i - mean_group) / std_group
```

它可以稳定训练，但可能放大简单题/难题之间的不均衡。Dr.GRPO 相关讨论认为按 std 缩放会引入题目难度偏置。

建议做小消融：

| 配置 | 预期 |
---|---|
| `scale_rewards="group"` | 最稳，起步默认 |
| `scale_rewards="none"` | 更忠实 reward，可能更噪 |
| `scale_rewards="batch"` | 如果可用，可做折中 |

报告中建议同时记录 TRL 自带的 `reward_std` 和 `frac_reward_zero_std`。前者表示 reward 的组内/批内波动，后者直接告诉你有多少 prompt 的多次采样几乎没有区分度。

### 7.4 KL/beta 的取舍

对 24 点专用模型，允许偏离语言能力，所以 KL 不应太强。

建议：

```text
beta = 0.0 先跑
如果格式/语言崩掉，试 beta = 0.005 / 0.01
如果 solve rate 被压住，降低 beta
```

注意：KL 不只是保护语言能力，也会抑制模型探索新表达式格式。24 点是专用任务，KL 过强会妨碍策略收敛。

### 7.5 LoRA 还是全参

4090 上建议：

第一阶段：

- SFT 用 LoRA/QLoRA。
- GRPO 用 LoRA。

如果 LoRA 到瓶颈，再尝试 full fine-tune SFT，不建议一开始全参 GRPO。

LoRA 建议：

```yaml
r: 32 或 64
alpha: 64 或 128
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
dropout: 0.0-0.05
```

## 8. Active Difficulty Sampling：本项目最关键的训练技术

### 8.1 为什么需要

GRPO 的学习信号来自组内差异：

```text
same prompt:
    completions = [wrong, wrong, correct, wrong, ...]
```

如果是：

```text
[wrong, wrong, wrong, wrong, ...]
```

或：

```text
[correct, correct, correct, correct, ...]
```

都没有有效 advantage。

### 8.2 难度桶设计

为每个题维护在线估计：

```python
PuzzleStats:
    n_sampled
    n_correct
    pass_rate_ema
    last_seen_step
    difficulty_static
```

采样桶：

```text
hard: pass_rate < 0.02
learnable: 0.02 <= pass_rate <= 0.60
easy: pass_rate > 0.60
```

训练采样比例：

```text
learnable: 70%
hard: 20%
easy/replay: 10%
```

如果 early stage 全部 hard，回退到 SFT 或提高 temperature。

### 8.3 动态采样伪代码

```python
for step in range(num_steps):
    batch_prompts = sampler.sample()
    groups = rollout(model, batch_prompts, G)
    rewards = verifier(groups)

    train_groups = []
    for q, group_rewards in zip(batch_prompts, rewards):
        c = sum(r > 0.9 for r in group_rewards)
        stats[q].update(c / G)
        if 0 < c < G:
            train_groups.append(q)

    if len(train_groups) < min_batch:
        refill_from_learnable_buffer()

    grpo_update(train_groups)
```

### 8.4 和 DAPO Dynamic Sampling 的关系

DAPO 在大规模数学 RL 中过滤零梯度数据；我们在 24 点中做同样思想的轻量实现。区别是：

- DAPO 面对大规模数学题，不知道题目完整空间。
- 24 点空间只有 1,820 个，可完整记录每题状态。
- 因此我们可以更精细地按 puzzle id 做 curriculum。

这可以作为项目创新点写进报告。

## 9. 实验矩阵：真正能指导开发的版本

### 9.1 Baseline 层

| 实验 | 训练 | 评测 | 目的 |
|---|---|---|---|
| B0 | 无训练，prompt-only | greedy/pass@k | 看基座能力 |
| B1 | solver oracle | 100% | 验证环境和上限 |
| B2 | previous repo best | 同 split | 历史对照 |

### 9.2 SFT 层

| 实验 | 数据 | 目的 |
|---|---|---|
| S1 | full 24 solution only | 验证 solver trace 是否足够 |
| S2 | arithmetic forward only | 验证算术预训练 |
| S3 | state transition + subproblem | 验证搜索结构 |
| S4 | arithmetic + state + full | 主 SFT |
| S5 | S4 + near-miss self-check | 减少漏数/算错 |

### 9.3 RL 层

| 实验 | 初始化 | RL 配置 | 目的 |
|---|---|---|---|
| R0 | base | GRPO only | 证明 pure RL 困难 |
| R1 | S1 | GRPO correctness only | RL 是否超过 SFT |
| R2 | S4 | GRPO correctness only | 主线基础 |
| R3 | S4 | GRPO + active sampling | 核心创新 |
| R4 | S5 | GRPO + active sampling | 最佳候选 |
| R5 | S5 | GRPO + unsolvable late mix | hallucination 控制 |

### 9.4 算法消融

| 消融 | 对比 |
|---|---|
| reward dense vs sparse | 是否 reward hack |
| `loss_type=grpo/dapo/dr_grpo` | 长度偏置 |
| `scale_rewards=group/none` | 难度偏置 |
| `num_generations=4/8/16` | 组内方差和成本 |
| `max_completion_length=128/256/384` | 截断和长度 |
| beta=0/0.005/0.01 | KL 约束 |

## 10. 评测指标与训练监控

### 10.1 最终任务指标

```text
solve_rate
valid_expr_rate
format_rate
wrong_number_rate
wrong_value_rate
parse_error_rate
no_answer_rate
```

无解题：

```text
abstain_rate
fabricate_rate
false_no_solution_on_solvable
```

pass@k：

```text
pass@1 greedy
pass@1 sampling
pass@4
pass@16
pass@64
```

### 10.2 RL 训练指标

必须记录：

```text
reward_mean
reward_std
correctness_mean
group_reward_std_mean
zero_std_group_rate
frac_reward_zero_std  # TRL 原生日志名
group_all_wrong_rate
group_all_correct_rate
completion_len_mean
completion_len_p95
truncation_rate
kl
entropy
grad_norm
learning_rate
```

### 10.3 分层评测

按静态难度：

```text
solution_count: 1-2 / 3-10 / 11-50 / >50
requires_division: true/false
requires_fraction_intermediate: true/false
number_pattern: duplicates / no duplicates
```

按数据来源：

```text
nlile heldout
ToT hard 100
ToT non-hard
Countdown OOD
Unsolvable generated
```

## 11. 开发注意事项

### 11.1 Verifier 必须安全

不要用 `eval`。建议使用 Python AST 白名单或自己写 parser。

允许：

```text
int constants
binary + - * /
unary minus, if needed
parentheses
```

禁止：

```text
function call
variable
attribute
list/dict
power operator
float scientific notation
implicit multiplication
```

数字 multiset 校验必须基于 AST 数字节点，不要正则粗糙抓数字，否则 `13` 和 `1,3` 容易出错。

### 11.2 Dataset split 必须按 multiset

不要按行随机切，因为同一个 multiset 可能有多条解。切分 key：

```python
key = tuple(sorted(numbers))
```

训练集、dev、test 之间 key 不能重叠。

ToT hard 100 必须从训练排除。

### 11.3 Prompt 要固定

不要频繁改 prompt，否则实验不可比。

推荐固定系统提示：

```text
You solve arithmetic target puzzles. Use each given number exactly once.
Return <think>short reasoning</think><answer>final expression</answer>.
```

用户提示：

```text
Numbers: 8, 2, 7, 3. Target: 24.
Use + - * / and parentheses. Each number must be used exactly once.
```

### 11.4 Eval 要分 greedy 和 sampling

Greedy：

```text
temperature=0
do_sample=false
```

Sampling：

```text
temperature=0.8 或 1.0
top_p=0.95
k=16/64
```

Sampling 不是作弊；它是为了诊断模型分布里有没有正确策略。但最终课程主指标仍应报告 greedy/pass@1。

## 12. 技术教程阅读路线

### 12.1 最短必读路线

1. OpenAI Spinning Up: Key Concepts in RL  
   <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html>
2. OpenAI Spinning Up: Intro to Policy Optimization  
   <https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html>
3. OpenAI Spinning Up: PPO  
   <https://spinningup.openai.com/en/latest/algorithms/ppo.html>
4. Hugging Face TRL GRPOTrainer  
   <https://huggingface.co/docs/trl/en/grpo_trainer>
5. Hugging Face TRL Reward Functions  
   <https://huggingface.co/docs/trl/en/rewards>

### 12.2 实践框架路线

优先级：

1. TRL：最适合单卡课程项目。
2. Unsloth：如果显存压力大，可尝试，但要确认自定义 reward 和 logging 能否满足。
3. verl：更适合多卡/大规模，技术上强但上手成本高。
4. OpenRLHF：强但系统复杂，课程项目不建议作为第一选择。

对应链接：

- TRL: <https://huggingface.co/docs/trl/en/index>
- TRL GRPO: <https://huggingface.co/docs/trl/en/grpo_trainer>
- PEFT integration: <https://huggingface.co/docs/trl/en/peft_integration>
- verl docs: <https://verl.readthedocs.io/>
- OpenRLHF docs: <https://openrlhf.readthedocs.io/>
- Qwen + LLaMA-Factory SFT: <https://qwen.readthedocs.io/en/latest/training/llama_factory.html>

### 12.3 论文阅读优先级

第一优先级：

1. Tree of Thoughts
2. DeepSeekMath
3. DeepSeek-R1
4. DAPO
5. Understanding R1-Zero-Like Training

第二优先级：

1. PPO
2. InstructGPT
3. N+ Implementation Details of RLHF with PPO
4. Logic-RL
5. TinyZero

第三优先级：

1. STaR
2. Self-Consistency
3. Training Verifiers
4. Let's Verify Step by Step
5. Grokking

## 13. 可直接写进报告的理论表述

### 13.1 任务为何适合 RLVR

24 点游戏具有精确可验证的终止奖励：给定四个整数和目标值，候选表达式是否使用每个数字且仅使用一次、是否只包含允许运算符、是否精确等于目标值，都可以由程序判定。因此该任务不需要人工偏好标注或奖励模型，属于典型的 reinforcement learning with verifiable rewards 场景。

### 13.2 为什么需要 SFT warm start

虽然可验证奖励降低了监督成本，但 24 点的正确奖励极稀疏。若直接从基座模型开始 GRPO，同一题目的多个采样回答可能全部错误，导致组内标准化 advantage 接近零，无法产生有效策略梯度。程序化 solver-SFT 的作用不是替代 RL，而是把策略初始化到能偶尔产生正确轨迹的区域，从而让后续 RL 有可学习信号。

### 13.3 为什么 active difficulty sampling 是关键

GRPO 依赖同一 prompt 下多个 completion 的相对 reward。如果某题对当前模型太难，所有 completion 全错；如果太简单，所有 completion 全对；两者都会产生零或近零 advantage。active difficulty sampling 通过优先选择当前模型成功率处于中间区间的题目，提高非零梯度样本比例，从而提升训练效率和稳定性。

### 13.4 为什么不能只看 reward 曲线

在可验证推理任务中，reward 曲线可能因格式奖励、长度奖励或 verifier 漏洞而上升，但真实解题率不升甚至下降。因此训练监控必须同时记录 final solve rate、failure type、completion length、group reward variance 和 pass@k。只有 correctness 指标与 reward 同步提升，才能说明策略确实学到了任务能力。

### 13.5 为什么 pass@k 重要

单次 greedy 输出衡量的是模型把正确策略放在最高概率路径上的能力，而 pass@k 衡量的是模型分布中是否存在正确策略。如果 pass@k 显著高于 greedy，说明模型已经具备潜在解题能力，但策略分布尚未完成 capture；此时 RL 的目标应是提升正确轨迹概率，而不是继续扩大监督数据规模。

## 14. 推荐最终主方案

最终建议采用：

```text
Qwen2.5-1.5B-Instruct
  -> solver-generated short-trace SFT
  -> near-miss self-verification SFT
  -> GRPO/Dr.GRPO with verifier reward
  -> active difficulty sampling
  -> unsolvable late-stage honesty calibration
```

最小可行版本：

```text
SFT data:
    10k arithmetic/state/full traces
GRPO:
    num_generations=8
    max_completion_length=256
    loss_type=dr_grpo
    reward=correctness-dominant
Evaluation:
    greedy + pass@16 + failure analysis
```

增强版本：

```text
active difficulty
near-miss self-check
pass@64
Countdown OOD
unsolvable hallucination split
reward/loss_type ablations
```

## 15. 参考链接

强化学习基础：

- Policy Gradient Methods for Reinforcement Learning with Function Approximation: <https://papers.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>
- GAE: <https://arxiv.org/abs/1506.02438>
- TRPO: <https://arxiv.org/abs/1502.05477>
- PPO: <https://arxiv.org/abs/1707.06347>
- OpenAI Spinning Up: <https://spinningup.openai.com/en/latest/>
- Hugging Face Deep RL Course: <https://huggingface.co/learn/deep-rl-course/en/unit0/introduction>

LLM RLHF / RLVR：

- Fine-Tuning Language Models from Human Preferences: <https://arxiv.org/abs/1909.08593>
- Learning to Summarize from Human Feedback: <https://arxiv.org/abs/2009.01325>
- InstructGPT: <https://arxiv.org/abs/2203.02155>
- N+ Implementation Details of RLHF with PPO: <https://arxiv.org/abs/2403.17031>
- RLHF Book: <https://rlhfbook.com/>
- DeepSeekMath: <https://arxiv.org/abs/2402.03300>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- Logic-RL: <https://arxiv.org/abs/2502.14768>
- DAPO: <https://arxiv.org/abs/2503.14476>
- Understanding R1-Zero-Like Training: <https://arxiv.org/html/2503.20783v2>
- REINFORCE++: <https://arxiv.org/abs/2501.03262>
- ReMax: <https://arxiv.org/abs/2310.10505>
- TinyZero: <https://github.com/Jiayi-Pan/TinyZero>
- Open-R1: <https://github.com/huggingface/open-r1>

推理、搜索、验证：

- Chain-of-Thought Prompting: <https://arxiv.org/abs/2201.11903>
- Zero-shot-CoT: <https://arxiv.org/abs/2205.11916>
- Self-Consistency: <https://arxiv.org/abs/2203.11171>
- STaR: <https://arxiv.org/abs/2203.14465>
- Training Verifiers: <https://arxiv.org/abs/2110.14168>
- Let's Verify Step by Step: <https://arxiv.org/abs/2305.20050>
- Tree of Thoughts: <https://arxiv.org/abs/2305.10601>
- Grokking: <https://arxiv.org/abs/2201.02177>

工程文档：

- TRL: <https://huggingface.co/docs/trl/en/index>
- TRL GRPOTrainer: <https://huggingface.co/docs/trl/en/grpo_trainer>
- TRL Reward Functions: <https://huggingface.co/docs/trl/en/rewards>
- TRL PEFT Integration: <https://huggingface.co/docs/trl/en/peft_integration>
- Ray TRL GRPO example: <https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html>
- verl docs: <https://verl.readthedocs.io/>
- OpenRLHF docs: <https://openrlhf.readthedocs.io/>
- Qwen LLaMA-Factory guide: <https://qwen.readthedocs.io/en/latest/training/llama_factory.html>
