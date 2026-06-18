# GRPO Pilot 设计

> 2026-06-19 更新：本文记录早期 conservative pilot 设计。当前 handoff1 后续 GRPO
> 训练的冻结执行口径以
> `docs/experiments/handoff1_grpo_final_plan_20260619.md` 为准。

## 摘要

下一阶段应该从 strong SFT checkpoint 启动一轮 conservative RLVR + GRPO
pilot：

`outputs/experiments/baseline_format_v2_full_5000_from800/final`

目标是 validation strict greedy accuracy 达到 `90%+`。当前 validation split
有 `136` 道题，所以 `90%+` 等价于至少 `123/136` 道题通过 strict verifier。
当前 SFT checkpoint 是 `110/136`，因此 GRPO 至少需要净提升 `13` 道
validation 题，同时不能明显损坏原本已经会做的题。

这个目标值得尝试，因为当前剩余失败不是算术能力失败。在
`max_new_tokens=1024` 下，剩余 `26` 个 greedy 失败全是
answer-contract/truncation：模型一直写很长的 rollback/search trace，没能及时
输出完整 `<answer>...</answer>`。采样审计已经证明，多数失败题在 sampling
分布里能采到正确轨迹，所以瓶颈更像是搜索路径选择和及时收束，而不是模型完全不会解。

## 资料来源

- DeepSeekMath 提出 GRPO：对同一 prompt 采样一组 completions，用组内 reward
  估计 advantage，省掉 value/critic model：
  <https://arxiv.org/abs/2402.03300>。
- DeepSeek-R1 说明 RLVR-style rule reward 训练，以及 cold-start 数据对稳定性的价值：
  <https://arxiv.org/abs/2501.12948>。
- Hugging Face TRL `GRPOTrainer` 文档说明自定义 reward function、
  `num_generations`、`loss_type`、`scale_rewards` 和日志指标：
  <https://huggingface.co/docs/trl/en/grpo_trainer>。
- TRL reward function 接口：
  <https://huggingface.co/docs/trl/en/rewards>。
- Ray 的 TRL GRPO 示例可作为后续分布式启动参考：
  <https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html>。
- 仓库内技术综述：
  `docs/research/24game_rl_technical_literature_review.md`。

## 当前证据

Strong SFT：

- Checkpoint：
  `outputs/experiments/baseline_format_v2_full_5000_from800/final`。
- Validation strict greedy：`110/136 = 80.88%`。
- 失败类型：`26` 个 answer-contract failures；在 format-valid 输出里没有
  wrong-value，也没有 wrong-number。
- 记录文档：
  `docs/experiments/sft_full_finetune_search_trace_20260616.md`。

Rollout audit：

- Validation pilot，`G=4`，`temperature=0.8`，`top_p=0.95`，
  `max_new_tokens=1024`：pass@4 是 `30/32`，mixed reward groups 是 `16/32`。
- Targeted greedy failures，`G=8`，同样采样配置：pass@8 是 `22/26`，
  mixed reward groups 是 `19/26`。
- 主要风险：采样 completion 经常打满 token budget；`1024` token audit 中，
  completion length 的 p50/p95 都是 `1024`。
- 记录文档：
  `docs/experiments/grpo_rollout_audit_20260616.md`。

GRPO 条件：

GRPO 需要同一个 prompt 下的一组 completions 有 reward 方差。当前 audit 已经满足
这个条件：很多 prompt 同时存在正确和错误采样输出。这意味着 policy 已经给正确轨迹
非零概率，GRPO 可以把正确且及时收束的轨迹相对过长/截断轨迹提高概率。

但这还不够直接长训。正式进入 GRPO 之前，必须先做两类前置验收：

1. train-split active-difficulty pool 必须先验收，确认 mixed group 数量、zero-std
   比例、correct-vs-truncation 混合比例足够。
2. TRL 兼容性必须在 AutoDL 上用短 probe 证明，确认当前版本的 `GRPOConfig`、
   reward function 额外列、`loss_type`、`scale_rewards`、`beta`、`mask_truncated_completions`
   和 `remove_unused_columns=False` 的组合真的能跑。

## 训练目标

主目标：

- 在不改变 strict verifier 和 `max_new_tokens=1024` 的前提下，把 validation
  greedy solve rate 从 `110/136` 提到至少 `123/136`。

次目标：

- 保持现有 `<answer>...</answer>` 主契约。
- 降低 answer-contract/truncation failure。
- 防止 reward hacking：不能出现 training reward 上升，但 greedy strict solve
  rate、format rate 或 valid-expression rate 下降。
- 控制 completion length，但不能训练成短而错的答案。

## 奖励 v1

第一版 reward 必须以 correctness 为主。Verifier 继续使用仓库的 strict AST +
`Fraction` verifier。

```text
if strict verifier accepts:
    reward = 1.0
elif output is missing/incomplete <answer>...</answer>:
    reward = -0.2
else:
    reward = -0.1
```

理由：

- Correctness 必须占主导。目标行为是输出一个使用输入 multiset、精确等于 `24`
  的合法表达式。
- 当前瓶颈是 missing answer / truncation，所以这类失败给一个小负分。
- Parseable but wrong answer 仍然要有轻微惩罚，避免模型学会“随便闭合一个错误答案”
  比继续搜索更优。这个罚分必须很小，不能压过正确样本。
- 第一版不加独立 format bonus。格式比解题容易学，容易让梯度被格式主导。
- 第一版不加独立 length bonus。长度奖励可能鼓励短但错误的答案。

这不是纯 binary reward，也不是长度奖励。它是一个很轻的 closure-shaping reward。
因此它必须和 answer-close 指标、wrong-answer rate 一起看，不能只看总 reward。

可选的 reward v2：

```text
if strict verifier accepts and completion_tokens <= 768:
    reward += 0.05
```

只有当 v1 已经证明 solve rate 没有退化、但正确样本仍明显过长时，再考虑加这个
correctness-gated length bonus。如果 greedy accuracy 退化，不应该加长度奖励。

## 训练池

第一轮不要均匀采样所有 train prompts，而是用 active-difficulty prompts。

Pool A：audit 中的 mixed validation-style prompts

- 这些 prompt 的 sampled rollout group 有正有负，说明存在直接 GRPO 信号。
- 不能用 validation split 训练后再报告 validation 提升。这个池只用于验证选择规则；
  真实训练前必须在 train split 上重建等价 pool。

Pool B：train-split active-difficulty pool

- 从 strong SFT checkpoint 在 train split 上跑 rollout audit。
- 建议配置：`G=4` 或 `G=8`，`temperature=0.8`，`top_p=0.95`，
  `max_new_tokens=1024`。
- 保留 empirical pass rate 在 `[0.125, 0.875]` 的 prompts。
- 优先保留“正确样本”和 answer-contract/truncation failures 混合出现的 prompts。

Pool C：少量 replay pool

- 每个 batch 可以混入 `10-20%` all-correct 或 high-pass-rate prompts。
- 目的：降低遗忘，保住 easy problems 上已经学会的 answer closure。

Pool D：all-wrong prompts

- 第一轮排除，或者极低比例采样。
- all-wrong groups 组内 advantage 接近 0，大多只浪费 rollout budget。

## 训练池验收

在任何 GRPO 长训之前，train pool 必须先过一个硬门槛：

- pool size 至少 `200` 个 prompt；如果 train split 不足以达到这个数，必须记录原因，
  并改用更高 `G` 或更高 temperature 重建 pool。
- mixed group rate 至少 `25%`。
- zero-std group rate 不超过 `75%`。
- correct-vs-truncation mixed prompts 至少 `50` 个，说明 reward 真的能分出
  “继续搜”和“及时闭合”的差异。
- all-wrong prompts 不能超过 pool 的 `25%`；否则说明 pool 太难。
- 生成 pool 时必须记录 seed、采样配置、source split、checkpoint、reward version。
- 如果 train pool 验收不过，先重建 pool，不进入长训。

实现细节：

- 写一个 pool manifest，字段至少包含 prompt id、numbers、target、audit pass rate、
  reason-count summary 和 source split。
- GRPO 训练 rows 只应包含 prompt；completion 由在线 rollout 生成。
- artifact 必须明确 split identity。最终 validation claim 不能来自训练过
  validation prompts 的模型。

## 建议 GRPO 配置

优先使用 TRL `GRPOTrainer`，因为本项目已有 TRL 依赖，且 ADR 已经选择
Transformers/PEFT/TRL 栈。

Pilot 默认配置：

```yaml
model_name_or_path: outputs/experiments/baseline_format_v2_full_5000_from800/final
output_dir: outputs/experiments/grpo_pilot_v1
prompt_style: qwen_chat
max_prompt_length: 256
max_completion_length: 1024
num_generations: 4
temperature: 0.8
top_p: 0.95
learning_rate: 5.0e-6
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
max_steps: 100
save_steps: 25
eval_steps: 25
bf16: true
loss_type: dr_grpo
scale_rewards: none
beta: 0.0
mask_truncated_completions: false
remove_unused_columns: false
report_to: [tensorboard]
```

说明：

- 第一轮用 `num_generations=4` 控制 rollout 成本。如果 train pool 的 zero-std
  groups 过多，再提升到 `8`。
- 学习率要低。SFT 已经很强，这不是重新学能力，而是调整搜索路径和 answer closure。
- 如果当前 TRL 版本支持，优先显式设置 `loss_type="dr_grpo"`。当前任务的主要风险
  正是过长 completion，长度偏置必须严肃处理。
- 第一轮先 probe `scale_rewards="none"` 和 `scale_rewards="group"`，不要默认锁死其中
  一个。若只看 short probe，二者都应试一次。
- `beta` 先从 `0.0` 或 `0.001` 级别开始 probe，不要直接把 `0.02` 当默认。
- `mask_truncated_completions` 必须显式记录。当前任务需要知道截断样本究竟是进入
  loss 还是被屏蔽。
- `remove_unused_columns=False` 必须显式设置，保证 reward function 能读到
  numbers、target、id 等额外列。
- 如果当前 TRL 版本不支持 `mask_truncated_completions` 或其它配置字段，compatibility
  probe 必须 fail fast，不能悄悄忽略；resolved config 必须写入 artifact。
- 如果 TRL 版本支持 KL/reference control，保留 `beta` 或等价配置；如果 API 名称
  不同，必须把实际 TRL 版本和 resolved config 写入 run metadata。

## 评估与监控

主评估：

- Greedy validation strict solve rate，`max_new_tokens=1024`。
- Pilot 成功阈值：任意 checkpoint 达到 `>=123/136`。
- 更现实的 pilot 早期成功定义：`>110/136`，同时原本 `110` 道已解题的
  retention `>=108/110`，answer-contract failures 低于 `26/136`，wrong-answer
  failures 不超过 `3/136`。
- 退化阈值：warmup 后如果 validation solve rate 低于 `108/136`，停止或回滚。

次评估：

- 只有 validation 提升后，才看 greedy test split strict solve rate。
- Sampled validation pass@4 或 pass@8，用来判断多样性是否还在。
- Verifier failure mix。
- Completion token length mean、p50、p95。
- Answer-contract/truncation rate。
- 原本 `110` 道已解题的 retention：看有多少仍然被 greedy 解出。

训练日志：

- `reward_mean`。
- `reward_std`。
- `frac_reward_zero_std` 或等价 zero-std group rate。
- KL to reference/SFT policy。
- Completion length mean 和 p95。
- `first_<answer>_token`、`first_</answer>_token`、`answer_close_token_index`、
  `tokens_after_answer`、`has_complete_answer`。
- Eval artifact 上的 format rate 和 valid-expression rate。
- 固定小诊断集的 raw sampled examples。

## 停止门禁

- 如果 `reward_mean` 上升，但 greedy validation solve rate 连续两次 eval 下降超过
  `2` 道题，停止。
- 如果 greedy validation 上的 answer-contract failures 超过 SFT baseline
  `26/136`，停止。
- 如果 greedy validation 上 wrong-value 或 wrong-number 合计超过 `3/136`，停止。
- 如果原本 `110` 道已解题的 retention 低于 `108/110`，停止。
- 如果 completion length p95 仍然是 `1024`，并且 `answer_close_token_index` 没有提前，
  且 step `50` 前 solve rate 没有提升，
  停止并重看训练池。
- 如果 zero-std group rate 多个 logging window 都超过 `70%`，停止训练，先重建
  active-difficulty pool。
- 如果 KL 突增，或者输出丢失 `<think>...</think><answer>` 结构，停止。

## 实现计划

1. 在 verifier 附近新增 reward 代码，例如 `src/game24_rl/rewards.py`。
   它应调用 `verify_answer`，返回 reward 和 reason metadata。
2. 增加 focused reward tests：
   correct answer、wrong value、wrong numbers、missing answer、incomplete answer、
   syntax error。
3. 增加 train-pool builder，复用 rollout audit 逻辑，在 train split 上生成
   active-difficulty manifest，写到 `data/processed/experiments/`。
4. 增加 `scripts/train_grpo.py` 或 `src/game24_rl/train_grpo.py`，必须支持 dry-run。
   Dry-run 应该构建 dataset 并跑 reward functions，但不加载模型权重。
5. 增加配置文件，例如 `configs/grpo_pilot_v1.yaml`。
6. 本地验证：
   `python -m compileall src scripts`，focused reward/pool tests，然后 `pytest`。
7. 先做 train-pool 验收，确认 pool size、mixed groups、zero-std、correct-vs-truncation、
   seed 和采样配置都达到预期。
8. AutoDL 上先跑 compatibility probe：
   打印 TRL version，实例化 `GRPOConfig`，验证 reward function 额外列、`beta`、
   `scale_rewards`、`mask_truncated_completions`、`remove_unused_columns=False` 都能跑。
9. 先跑 short probe：`max_steps=25`，`eval_steps=25`，同时比较 `scale_rewards=none`
   与 `group`，`beta=0.0` 与 `0.001`。
10. 如果没有触发停止门禁，再扩到 `100` steps，并评估所有 saved checkpoints。

## 预期结果

最好结果：

- 模型学会在当前失败题上更早收束到 `<answer>`。
- Validation 达到 `123/136` 或更高。

可接受 pilot：

- Validation 高于 `110/136`，answer-contract failures 下降，并且原本已解题没有明显退化。
  后续继续调 `G`、pool selection 或 reward v2。

失败模式：

- Reward 上升但 greedy solve rate 下降：reward hacking 或 loss/config 有问题。
- Zero-std groups 占比高：训练池太容易或太难。
- Length 更差：需要更严格的 active pool filtering、`dr_grpo`，或尝试
  correctness-gated length shaping。
- Solve rate 卡在 `110/136`：低步数 GRPO 没能把 sampled success 转成 greedy；
  下一步考虑增加 `G`，或把 sampled successful short trajectories 做 targeted
  SFT/RL hybrid。
- 如果 train pool 验收不过，先重建 pool，不要直接加长训步数。

## 报告规则

每个 reported GRPO score 必须声明：

- Initial checkpoint。
- GRPO checkpoint path。
- Split manifest 和 split。
- Reward version。
- TRL version 和 GRPO config。
- Decoding config，特别是 `max_new_tokens`。
- Verifier version。
- Raw outputs 和 evaluation report path。
- 是否使用 validation prompts 参与训练。最终主结论不能用训练过 validation prompts 的模型来报 validation gain。
