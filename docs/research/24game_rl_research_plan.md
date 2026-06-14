# 基于强化学习的 24 点专用模型：调研报告与实验方案

调研日期：2026-06-13  
项目背景：NLP 课程大作业，选题三“基于强化学习的 24 点游戏求解”  
建议主线：不用 DeepSeek 蒸馏作为核心路线；用程序生成数据、精确 verifier、课程式 SFT warm start、再用 RLVR/GRPO 做在线强化学习与采样分布压缩。

## 0. 结论先行

我建议把项目定位成：

> 从 Qwen2.5-1.5B-Instruct 出发，通过程序化环境、可验证奖励和主动难度课程，把通用语言模型压缩成一个 24 点专用策略模型；比较 prompt-only、solver-SFT、算术 curriculum-SFT、GRPO/RLVR，以及旧路线中的蒸馏基线。

最佳路线不是“纯 RL 从 0 学会一切”，也不是“继续堆 DeepSeek 蒸馏”。更稳的是：

1. 先搭一个完全可信的 24 点环境：Fraction 精确求值、AST 白名单解析、数字 multiset 校验、无 `eval`。
2. 自己枚举全空间：1..13 取 4 个数的多重集共 1,820 个，其中 1,362 有解、458 无解。
3. 用 solver 生成短而多样的 SFT warm start：表达式求值、二数合并、三数到目标、完整 24 点解、near-miss 自检。
4. 再做 RLVR/GRPO：奖励以最终 correctness 为主，避免稠密格式奖励主导；训练 prompt 需要主动选择“模型偶尔能做对”的题，否则 GRPO 组内奖励方差为 0。
5. 报告重点写“为什么 24 点是可验证 RL 环境、为什么小模型会 reward hack、为什么主动课程比盲目 scaling/蒸馏更合理”。

如果只能保一个创新点，保“主动难度课程 + verifier 诊断 + GRPO 稳定性修复”。这比“我也蒸馏了一个更大的模型”更像强化学习项目。

## 1. 课程要求与现实约束

课程题目明确要求：

- 基座：Qwen2.5-1.5B-Instruct。
- 算法：GRPO / RLVR。
- 输出：`<think>...</think><answer>...</answer>`，最终 answer 是合法算式。
- 数据：`nlile/24-game`、`test-time-compute/game-of-24`、`Jiayi-Pan/Countdown-Tasks-3to4` 可作为训练/测试/扩展。
- 加分点：3-4 个数字凑任意目标数、训练曲线、reward/正确率监控、定性定量分析。

评分标准更看重：

- 挑战描述清晰。
- 算法和技术细节清晰，有公式、伪代码、图示。
- 实验步骤、评价指标、优越性与局限性分析充分。
- 展示中要有针对问题本身的进一步建模。

因此，这个项目不应该只追单个最终分数。应该展示一个“强化学习环境 + 课程 + verifier + 诊断”的完整系统。

## 2. 关键调研证据

### 2.1 Game of 24 是典型搜索任务，不是普通算术题

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) 把 Game of 24 作为需要搜索/规划的 benchmark。它的核心启发是：单条 chain-of-thought 容易早早提交错误路径，而树搜索能显著改善结果。对本项目的含义是：

- 24 点不是“模型会四则运算就行”。
- 模型需要学到一种小规模组合搜索策略：选择两个数合并、保留中间值、递归逼近目标。
- 如果推理期不允许外部工具，那么训练期必须把搜索结构压进模型。

### 2.2 RLVR/GRPO 是合理框架，但稀疏奖励会带来稳定性问题

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 提出 GRPO：对同一 prompt 采样一组 outputs，用组内 reward 均值和标准差归一化优势，避免单独训练 value model。

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 证明了“可验证奖励 + RL”可以诱发推理行为，但也显示纯 RL 会出现可读性差、语言混杂、格式不稳定等问题，所以它后来引入 cold-start 数据和多阶段训练。

本项目的对应判断：

- 24 点是 RLVR 的理想环境，因为 reward 可由程序判定。
- 但 24 点 reward 很稀疏：随机生成一个正确表达式概率很低。
- 如果直接 GRPO，很多 prompt 的 G 个样本全错，优势全 0，等于没有学习信号。
- 所以必须先通过 SFT 或主动采样把正确样本概率推到非零，再 RL。

### 2.3 TinyZero/Logic-RL 支持“规则奖励 + 小模型”的方向，但不能照搬

[TinyZero](https://github.com/Jiayi-Pan/TinyZero) 在 Countdown 任务上复现了小模型通过 RL 学到可验证推理的现象，是本题最接近的开源参照。

[Logic-RL](https://arxiv.org/abs/2502.14768) 强调 rule-based RL 可以在不依赖奖励模型的情况下提升推理能力。

但是 24 点有一个特殊点：状态空间很小且可穷举。这意味着我们可以比 TinyZero 更激进地做：

- 全空间枚举。
- 精确难度分层。
- 训练前知道每题的可解性、解数、是否需要除法、是否需要非整数中间量。
- RL 采样时主动避开“全错/全对”的无梯度样本。

### 2.4 近期 GRPO 改进给出的避坑方向

[DAPO](https://arxiv.org/html/2503.14476v1) 强调 dynamic sampling、token-level policy-gradient loss、clip-higher 等技巧，核心思想是提高有效训练样本比例、减少不稳定更新。

[Understanding R1-Zero-Like Training](https://arxiv.org/html/2503.20783v2) 对 GRPO 类方法的长度偏置、难度偏置提出批评，并提出 Dr.GRPO 等修正思路。

对本项目的直接建议：

- 监控每个 batch 中 reward std 为 0 的比例。
- 不要让格式奖励、长度奖励、拒答奖励压过 correctness。
- 尽量使用 `loss_type="dr_grpo"` 或等价修正；如果框架不支持，至少在报告里讨论 GRPO 的 bias。
- prompt 采样要按当前模型成功率分桶，优先训练成功率约 5%-60% 的题。

### 2.5 数据集坑：不要完全相信课程文档的 solvable=False 描述

课程文档写 `nlile/24-game` 中有 solvable=False 约 100 条，但当前 [nlile/24-game](https://huggingface.co/datasets/nlile/24-game) 页面显示 1.36k 行，`solvable` 只有 1 个类别。结合全空间枚举，合理解释是：

- 1..13 四数多重集总数：C(16,4)=1,820。
- 有解多重集：1,362。
- 无解多重集：458。
- `nlile/24-game` 当前更像“有解全集”，不是可靠的无解集。

建议：无解题不要从 HF 字段拿，直接用自己的 solver 生成 458 个无解组合，抽样作为 hallucination split。

### 2.6 Qwen2.5-1.5B-Instruct 合适但容量有限

[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 是课程指定模型，参数量约 1.5B，单张 4090 可以 LoRA/QLoRA 训练，SFT 和 GRPO 都现实。

但不能期待它在 ToT hard split 上达到大模型水平。旧仓库实测显示，1.5B 的 pass@k 可能远高于 greedy，但 pass@1 capture 很难。这一点要写进报告：模型可能“采样时会”，但“单发稳定输出”是另一个问题。

## 3. 对你原始 idea 的技术评估

你的核心 idea 是：

1. 本地生成大量 24 点题和答案。
2. 先训练算术。
3. 再从表达式求值转到补运算符。
4. 枚举排列，让模型预测一种正确算式。
5. 最后让模型自己学排列。

这个方向总体正确，但需要做几处修正。

### 3.1 “先学算术”有用，但不是主瓶颈

24 点失败常见原因有三类：

- 算错中间值。
- 找到了等于 24 的子表达式，但漏用/重复使用数字。
- 搜索路径错：没找到需要的组合。

“表达式 -> 结果”的算术训练主要解决第一类问题。它不能自动解决第三类搜索问题。因此算术 curriculum 应该作为 warm-up，而不是主训练数据。

建议比例：

- 算术 forward/eval：10%-20%。
- 二数/三数子问题：20%-30%。
- 完整 inverse 24 点：40%-60%。
- near-miss 自检/非法表达式判别：10%-20%。

### 3.2 “预测符号”最好改成“预测结构化动作”

4 个数的表达式可以穷举为：

- 24 种排列。
- 5 种二叉树括号结构。
- 4^3=64 种运算符组合。
- 总计 24*5*64=7,680 个候选结构。

这说明 24 点本质上可以被建成一个小型离散动作搜索问题。

但不要让模型只输出“+ - * /”三个符号，因为括号结构和数字排列同样重要。更好的内部表示是：

```text
<think>
perm = [2,0,3,1]
tree = ((a op b) op (c op d))
ops = [*, -, /]
check = ...
</think>
<answer>((8*3)/(7-6))</answer>
```

最终仍输出表达式，满足课程要求；`think` 内部可以是紧凑动作代码，降低语言负担。

### 3.3 枚举所有排列不要直接复制成 24 倍训练样本

如果每个题生成 24 个排列，且答案高度重复，模型可能学到输入顺序噪声，训练集也会被简单题淹没。

建议：

- prompt 端随机打乱输入顺序，但每个 epoch 动态采样 1-3 个排列。
- 训练数据记录 canonical sorted key，评估时按 multiset 去重。
- reward 端只看 multiset，不看顺序。

### 3.4 “丧失语言能力”可以，但不要丢掉课程格式

课程明确希望 R1 风格 `<think>/<answer>`。可以让 `<think>` 很短、符号化、算法化；不要完全变成裸表达式，否则展示时会吃亏。

推荐目标格式：

```text
<think>Combine 8 and 3 to get 24, and make 1 from 7-6. Then 24/1=24.</think>
<answer>((8*3)/(7-6))</answer>
```

或更专用：

```text
<think>state [8,3,7,6]: 8*3=24; 7-6=1; 24/1=24.</think>
<answer>((8*3)/(7-6))</answer>
```

## 4. 推荐总体路线

### Phase A：环境与数据底座

必须先写：

- `solver.py`：Fraction 精确 DFS，返回所有或若干解。
- `verifier.py`：抽取 `<answer>`，解析表达式，校验数字 multiset，Fraction 求值。
- `dataset.py`：枚举 1,820 个 multiset，分 solvable/unsolvable，生成 train/dev/test manifest。
- `difficulty.py`：计算每题元信息：
  - 解的数量。
  - 是否需要除法。
  - 是否需要非整数中间值。
  - 最短表达式长度。
  - solver 搜索节点数。
  - ToT human solved rate，如果来自 ToT 数据。

Verifier 的硬要求：

- 不用 Python `eval`。
- 只允许 AST 中的数字常量、`+ - * /`、括号。
- 数字 token 必须是 1-13 的整数，`13` 不能被拆成 `1` 和 `3`。
- 对除零、非法字符、科学计数、变量名、函数调用全部拒绝。
- 使用 `Fraction` 做精确计算，最后允许浮点容差仅用于兼容输出小数。

### Phase B：无 teacher 的 solver-SFT warm start

生成 4 类数据：

1. Forward arithmetic：给表达式，输出结果。
2. State transition：给 `[a,b,c,d]`，输出一步合并后的新状态。
3. Full inverse：给四数，输出完整 24 点表达式。
4. Self-verification：先展示 near-miss，再检查数字使用和值，最后修正。

不要一上来生成几十万条。推荐从小规模开始：

- `arith_forward`: 10k。
- `state_transition`: 10k。
- `three_num_to_target`: 5k。
- `full_24`: 每个 solvable 题 3-8 条，总 4k-10k。
- `self_verify`: 2k-5k。

训练目标不是“把 solver 蒸馏到满分”，而是把模型带到 GRPO 能采到正样本的区域。

### Phase C：GRPO/RLVR

基础 GRPO 公式：

```text
A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
```

建议奖励：

```text
if correct expression:
    reward = 1.0
elif parseable and uses exact numbers but value != target:
    reward = -0.2
elif parseable but wrong number multiset:
    reward = -0.4
elif no answer / invalid syntax / timeout:
    reward = -0.6

format_bonus <= 0.05, only if answer is parseable
length_bonus = 0 by default
```

关键原则：

- correctness 是唯一主奖励。
- format 奖励不能独立给，否则模型会刷格式。
- 长度奖励只在 correct 后 gated，否则容易 CoT collapse。
- 对可解题不要奖励拒答。
- 对无解题单独做 late-stage honesty training，比例不要超过 5%-10%。

主动难度课程：

1. 每 N step 对候选训练题采样 K 次，估计当前 pass rate。
2. 只把 pass rate 在 `[0.05, 0.60]` 的题放入 GRPO 主池。
3. 全错题进入 later bucket；全对题降低采样权重。
4. 每轮训练后更新 bucket。

这比均匀训练更像 RL，因为它保证 advantage 有方差。

### Phase D：评测与分析

必须报告以下指标：

- `solve_rate`: 最终表达式正确率。
- `valid_expr_rate`: 语法可解析且数字使用正确。
- `value_correct_but_wrong_numbers`: 值等于 24 但数字不对。
- `wrong_value_with_right_numbers`: 数字对但值不对。
- `format_rate`: tag 格式正确率。
- `abstain_rate` / `fabricate_rate`: 无解题上是否瞎编。
- `pass@k`: k=1/4/16/64，衡量 latent ability。
- `reward_mean`, `reward_std`, `zero_std_group_rate`。
- `avg_completion_len`, `truncation_rate`, `KL`, `entropy`。

分层报告：

- easy / medium / hard by solution count。
- need division vs no division。
- need fractional intermediate vs not。
- ToT hard 100。
- Countdown OOD。
- unsolvable hallucination split。

## 5. 实验矩阵

最小可交付矩阵：

| 编号 | 方法 | 目的 |
|---|---|---|
| E0 | Qwen2.5-1.5B-Instruct prompt-only | 裸模型基线 |
| E1 | Solver full-solution SFT | 程序数据是否足够 |
| E2 | Arithmetic + state curriculum SFT | 验证你的课程 idea |
| E3 | E1 + GRPO | RL 是否超过 SFT |
| E4 | E2 + GRPO | 主路线 |
| E5 | E4 + active difficulty sampling | 核心创新 |
| E6 | E5 + self-verification data | 是否减少漏数/算错 |
| E7 | E5 + unsolvable late mix | 是否减少 hallucination |

建议加的强基线：

- Previous repo distilled model/result，作为历史对照。
- Brute-force solver oracle：100%，说明工具上限。
- pass@64 sampling：说明模型潜力和 capture gap。

关键消融：

- 去掉 arithmetic curriculum。
- 去掉 state transition。
- 去掉 active difficulty。
- dense reward vs correctness-only reward。
- answer-only vs R1 short-think format。
- LoRA rank 16/32/64。

## 6. 4090 训练建议

假设单张 4090 24GB：

SFT：

- Qwen2.5-1.5B-Instruct。
- LoRA/QLoRA 起步，rank 32，alpha 64。
- bf16，如果显存不够用 4-bit QLoRA。
- sequence length 512 或 768，避免长 CoT。
- batch 通过 gradient accumulation 到 global batch 64-128。
- 学习率 `1e-4` 到 `2e-4` for LoRA；full fine-tune 则 `1e-5` 到 `2e-5`。

GRPO：

- TRL `GRPOTrainer` 是最直接路线；参考 [TRL GRPO docs](https://huggingface.co/docs/trl/en/grpo_trainer)。
- `num_generations`: 8 起步，显存允许再到 16。
- `max_completion_length`: 256 或 384，不要 2k 长推理。
- `temperature`: 0.7-1.0，用于 rollout；eval 用 greedy 和 sampling 两套。
- `beta`: 0 到 0.02 起步；如果模型乱飘再加。
- 如框架支持，优先 `loss_type="dr_grpo"`。
- 每 50-100 step 跑小 dev eval，不要只看训练 reward。

如果使用 vLLM：

- 单卡上 vLLM + trainer 同卡可能挤显存；1.5B 可以先不用 vLLM，保证正确性。
- 如果有第二张卡或远端服务，再把 rollout server 拆出去。

## 7. 最大风险与避雷

### 风险 1：数据泄漏

24 点全空间只有 1,820 个 multiset。训练集如果覆盖所有 1,362 有解题，常规 ID test 就失去意义。

处理：

- 明确区分“训练覆盖题”和“heldout multiset”。
- ToT hard 100 必须从训练排除。
- 报告中不要把训练题上的结果当泛化。

### 风险 2：reward hacking

模型可能学会：

- 固定输出格式但答案错。
- 输出很短的假答案。
- 漏用数字但值等于 24。
- 对可解题拒答。

处理：

- verifier 分类失败类型。
- reward 以 final correctness 为主。
- 格式 bonus 必须 gated。
- 曲线中同时画 reward 和 solve rate；reward 涨但 solve rate 不涨就是 hack。

### 风险 3：GRPO 无学习信号

如果每组 G 个样本全错或全对，组内 advantage 近似 0。

处理：

- 监控 zero-std group rate。
- 使用 active difficulty sampling。
- SFT warm start 只要把 pass@k 拉起来即可，不要过拟合。

### 风险 4：算术 curriculum 变成无效预训练

如果 forward arithmetic 太多，模型会更会算表达式，但不会找表达式。

处理：

- 每个 curriculum 都必须有独立 ablation。
- 训练后跑 arithmetic probe 和 24 点 probe，证明 transfer 是否存在。

### 风险 5：输出语言太长

长 CoT 会增加训练成本、截断、格式错误和 reward noise。

处理：

- `<think>` 控制在 1-4 句。
- 用状态转移式短 trace。
- 不追求 DeepSeek 式长推理。

### 风险 6：无解题训练比例过高

无解题太多会让模型学会拒答，损害可解题 solve rate。

处理：

- 无解主要作为 eval。
- 如果训练 honesty，晚期小比例混入。
- 无解答案统一为 `<answer>NO_SOLUTION</answer>`，不要自然语言解释。

## 8. 推荐仓库结构

```text
24game-rl/
  README.md
  pyproject.toml
  configs/
    sft_curriculum.yaml
    grpo_active.yaml
  src/game24rl/
    solver.py
    verifier.py
    difficulty.py
    data_gen.py
    prompts.py
    rewards.py
    train_sft.py
    train_grpo.py
    evaluate.py
    pass_at_k.py
    analyze_failures.py
    plot_metrics.py
  data/
    raw/
    processed/
    manifests/
  outputs/
  reports/
```

必须先写单元测试：

- solver 对 1,820 个 multiset 统计应为 1,362 solvable / 458 unsolvable。
- verifier 接受合法表达式、拒绝漏数、拒绝重复数字、拒绝非法字符、拒绝除零。
- reward 对各类错误给出预期分数。

## 9. 报告故事线

推荐报告标题：

> 基于可验证奖励与主动难度课程的小模型 24 点求解

章节：

1. 任务背景：24 点是可验证组合搜索，不只是四则运算。
2. 方法：精确环境、solver 数据生成、curriculum SFT、GRPO/RLVR。
3. 奖励设计：公式、verifier、失败类型、anti-hack。
4. 实验设置：数据划分、模型、训练参数、指标。
5. 结果：主表 + pass@k + 分层分析。
6. 诊断：reward hacking、zero-std、长度、失败案例。
7. 消融：课程、active sampling、reward 设计。
8. 局限：有限状态空间、1.5B 容量、单发 capture 难。

展示时最有说服力的图：

- 全空间 1,820 个题的 solvable/unsolvable 饼图。
- 按解数划分的难度直方图。
- SFT vs GRPO solve rate 曲线。
- reward mean 与 actual solve rate 同图。
- zero-std group rate 曲线。
- pass@k 曲线，展示 latent ability vs greedy capture。
- failure type 堆叠柱状图。

## 10. 建议时间线

第 1-2 天：

- 环境、solver、verifier、数据枚举、测试。
- 复现 base eval。

第 3-4 天：

- 生成 curriculum 数据。
- 跑 E1/E2 SFT。
- 做 arithmetic probe 和 failure analysis。

第 5-7 天：

- 跑 E3/E4 GRPO。
- 加 active difficulty。
- 调 reward，记录曲线。

第 8-9 天：

- 跑消融和 OOD/unsolvable。
- pass@k、失败案例、图表。

第 10 天：

- 报告和 PPT。

如果时间更充裕，扩展 Countdown 任意目标任务；如果时间紧，砍掉 Countdown 训练，只保留 OOD 评测。

## 11. 最终建议

这条路线最值得做：

> 精确 solver/verifier + 结构化短 trace curriculum SFT + active-difficulty GRPO + verifier-driven diagnostics。

不要把项目做成“我又蒸馏了 DeepSeek”。可以保留蒸馏作为对照，但主张应是：

- 24 点的监督信号来自环境，不来自教师模型。
- 强化学习不是为了生成更长 CoT，而是为了把采样分布中偶尔出现的正确策略压到 greedy/pass@1。
- 专用模型允许牺牲语言能力，但必须保留可验证、可展示、可评分的 `<think>/<answer>` 接口。

## 参考链接

- Tree of Thoughts: <https://arxiv.org/abs/2305.10601>
- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- DeepSeek-R1 / RLVR: <https://arxiv.org/abs/2501.12948>
- TinyZero: <https://github.com/Jiayi-Pan/TinyZero>
- Logic-RL: <https://arxiv.org/abs/2502.14768>
- DAPO: <https://arxiv.org/html/2503.14476v1>
- R1-Zero-like Training Critical Perspective / Dr.GRPO: <https://arxiv.org/html/2503.20783v2>
- TRL GRPOTrainer: <https://huggingface.co/docs/trl/en/grpo_trainer>
- Open-R1: <https://github.com/huggingface/open-r1>
- Qwen2.5-1.5B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- nlile/24-game: <https://huggingface.co/datasets/nlile/24-game>
- test-time-compute/game-of-24: <https://huggingface.co/datasets/test-time-compute/game-of-24>
- Countdown-Tasks-3to4: <https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4>
