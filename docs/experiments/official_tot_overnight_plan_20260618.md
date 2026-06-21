# Official ToT Overnight Plan 2026-06-18

## 目的

把今晚托管实验固定为一个可复现执行合同。核心目标是比较：

- 原始 `Qwen2.5-1.5B-Instruct`。
- 使用完整 `nlile/24-game` 训练的 `SFT-full-data-5000`。
- 排除 ToT hard100 后训练的 `SFT-remove-900to1000-5000`。
- 从上述两个 SFT checkpoint 分别继续 GRPO 后训练得到的 final 模型。

## 已确认数据事实

- `nlile/24-game` 当前下载版本：`1362` 条，全部 `solvable=True`。
- 本项目精确 solver 审计：`1362/1362` 全部可解，HF 字段与 solver 判定不一致数为 `0`。
- `nlile/24-game` 与 `test-time-compute/game-of-24` 已确认 puzzle 集合完全相同：
  - `nlile` unique puzzles：`1362`
  - ToT unique puzzles：`1362`
  - overlap unique puzzles：`1362`
  - `nlile-only`：`0`
  - `ToT-only`：`0`
  - 同下标相同题目：`3/1362`
- 因此 `test-time-compute/game-of-24` 不是非重叠测试集；它提供的是同一批
  `1362` 道题的 ToT 排序和难度字段。
- ToT indices `900-999`，即 Rank `901-1000`，定义为 hard100；这 `100`
  道题也全部包含在 `nlile/24-game` 中。

审计 artifacts：

```text
/root/autodl-tmp/projects/game24-rl/outputs/audits/nlile_24_game_solver_audit.json
/root/autodl-tmp/projects/game24-rl/outputs/audits/hf_dataset_overlap_audit.json
```

## 用户确认版执行口径

1. 在原始模型 `Qwen/Qwen2.5-1.5B-Instruct` 上跑评测，得到 solve rate
   基线。评测使用 `test-time-compute/game-of-24` 全量数据集，
   `max_new_tokens=4096`。
2. 训练两个 SFT-5000 模型：
   - `SFT-full-data-5000`：使用我们之前已经跑通并达到强 SFT baseline 的同类
     SFT 训练配置，在正式的、全量的、完整的 `nlile/24-game` 上进行训练；
     训练数据为 `1362` 条。
   - `SFT-remove-900to1000-5000`：因为 `nlile/24-game` 和
     `test-time-compute/game-of-24` 本质上是同一个 puzzle 集合、只是排序和
     元数据不同，所以从 `test-time-compute/game-of-24` 中选择除 indices
     `900-999` 之外的 `1262` 条数据进行训练。
3. 在两个 SFT-5000 模型上分别进行评估，得到 `4096` token budget 下的评估
   结果。每个模型只在 ToT 全量 `1362` 条上跑一次评测。
4. 在两个 SFT-5000 模型基础上分别进行强化学习后训练，得到两个 final 模型并
   进行评估。GRPO 路线根据之前 repo-local split 上的实验分析结果设计：从
   SFT checkpoint 出发，沿用 strict verifier reward、mixed/active train
   pool、短 probe 中表现相对稳定的配置思路；旧结果只作为设计依据，不能冒充
   本轮 official ToT 口径结果。
5. 评测统一使用 `test-time-compute/game-of-24` 的全量 `1362` 条。先跑全量
   结果，然后直接从同一份 per-sample 结果中切分分组指标：
   - `all_1362`：indices `0-1361`。
   - `easy1262`：indices `0-899` 和 `1000-1361`。
   - `hard100`：indices `900-999`。
6. 分布内 / 分布外口径：
   - 对 `SFT-full-data-5000` 和从它继续训练的 `GRPO-full`，hard100 是训练集内
     hard subset，不是分布外。
   - 对 `SFT-remove-900to1000-5000` 和从它继续训练的
     `GRPO-remove-900to1000`，hard100 是排除数据，因此才是 held-out /
     分布外 subset。

## 关键批注

**评测只跑全量一次。** 每个模型只在 `test-time-compute/game-of-24` 全量
`1362` 条上执行一次 greedy eval，`max_new_tokens=4096`。`all_1362`、
`easy1262` 和 `hard100` 的准确率都从同一份 per-sample 评测结果中离线切分；
不能为了分组重复跑多次 GPU 评测。

**hard100 的含义依模型而变。** 对 `SFT-full-data-5000` 和 `GRPO-full`，
hard100 只是训练集中较难的 subset；对 `SFT-remove-900to1000-5000` 和
`GRPO-remove-900to1000`，hard100 才是真正 held-out subset。

**GRPO 的初始化是 SFT checkpoint。** “在 SFT 4096token 的基础上继续训练”
的准确含义是：先用 `4096` token budget 评估 SFT，然后从对应
`SFT-final` checkpoint 启动 GRPO。4096 是评测预算；若训练 rollout 使用
其他 `max_completion_length`，必须在 run metadata 中单独记录。

## 数据切分

评测数据：

- `eval_tot_all_1362`：`test-time-compute/game-of-24` 全量 indices `0-1361`。

离线分组：

- `all_1362`：indices `0-1361`。
- `easy1262`：indices `0-899` 加 `1000-1361`。
- `hard100`：indices `900-999`。

训练数据：

- `train_full_1362`：完整 `nlile/24-game`，`1362` 条。
- `train_remove_900to1000_1262`：从 `test-time-compute/game-of-24` 排除
  indices `900-999` 后的 `1262` 条。

## 执行矩阵

| 顺序 | 阶段 | 初始化模型 | 训练数据 | 评测 |
| ---: | --- | --- | --- | --- |
| 1 | Base eval | `Qwen/Qwen2.5-1.5B-Instruct` | 无 | ToT 全量一次，4096 |
| 2.1 | `SFT-full-data-5000` | `Qwen/Qwen2.5-1.5B-Instruct` | `train_full_1362` | ToT 全量一次，4096 |
| 2.2 | `SFT-remove-900to1000-5000` | `Qwen/Qwen2.5-1.5B-Instruct` | `train_remove_900to1000_1262` | ToT 全量一次，4096 |
| 4.1 | `GRPO-full` | `SFT-full-data-5000` | `train_full_1362` | ToT 全量一次，4096 |
| 4.2 | `GRPO-remove-900to1000` | `SFT-remove-900to1000-5000` | `train_remove_900to1000_1262` | ToT 全量一次，4096 |

## 结果表格式

每个模型最终输出同一行摘要：

```text
stage | init_model | train_set | eval_set | max_new_tokens | all_1362 | easy1262 | hard100 | failure_mix | raw_outputs | eval_report
```

`failure_mix` 是后处理分析字段，不是训练/评估主链路的硬门槛。主脚本必须优先
保存 raw outputs、per-sample verifier results 和基础 solve rate；如果
failure 分类脚本遇到未覆盖的新错误类型，应记录 warning 并继续，不能因此中断
后续训练或评估。

`failure_mix` 尽量包含：

- no answer / incomplete answer
- tag broken / parse error
- wrong value
- wrong numbers
- other verifier failure

## 执行约束

- 不设置自动 gate；今晚按顺序托管跑完整矩阵。
- `GRPO-remove-900to1000` 的训练数据和 rollout pool 不能包含 ToT indices
  `900-999`，否则 held-out 口径失效。
- 所有阶段必须保存 raw outputs、per-sample verifier results、run config 和基础
  分组 summary；failure 分类可以后期补做，不能作为脚本失败条件。
- 旧 repo-local split 结果只能作为设计依据，不能冒充本轮 official ToT 口径结果。
