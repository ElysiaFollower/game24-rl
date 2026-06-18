<!--
职责：记录跨会话进度、状态变化、阻塞和验证摘要，让新会话能快速恢复。
边界：不要存放聊天记录、原始日志、密钥，或更适合由代码、测试、ADR、任务计划表达的内容。
-->

# 进度日志

## 当前状态

- 当前功能项：`M3-grpo-frontier`
- 当前任务计划：`plans/active/20260616-grpo-pilot-design.md`
- 当前模式：高自治执行；进入 conservative GRPO pilot 设计和后续最小实现。
- 背景：strong full fine-tuning SFT 已达到 validation `110/136 = 80.88%`；剩余 `26` 个 greedy 失败都是 answer-contract/truncation；rollout audit 证明多数 greedy 失败题在采样分布中已有正确轨迹。
- 下一步最佳动作：当前最佳 greedy GRPO short probe 仍是
  `lora_r16_beta001_filtered_g8_lr5e7_5` 的 `116/136 = 85.29%`，
  retention `109/110`，answer-contract failures `20`，wrong-answer `0`。
  second-stage hard-pool continuation 从该 adapter 继续训后退到
  `112/136`，说明不能靠盲目加大 RL 强度突破；同口径补跑 strong SFT
  final 的 direct greedy `max_new_tokens=4096` 后，SFT 本身已达到
  validation `123/136 = 90.44%` 和 test `128/137 = 93.43%`。当前最佳
  GRPO adapter 在同预算下达到 validation `126/136 = 92.65%` 和 test
  `129/137 = 94.16%`，净增分别是 `+3/136` 和 `+1/137`。更高的
  inference-time strict verifier rerank 结果是 validation `133/136 = 97.79%`
  和 test `136/137 = 99.27%`，报告时必须分开。

## 状态约定

- `not_started`：尚未开始。
- `active`：当前唯一在制任务。
- `blocked`：缺少输入、环境、依赖或决策。
- `passing`：验证通过且 evidence 已记录。

## 近期证据

### 2026-06-18 - Official ToT overnight plan fixed

- 执行口径文档：`docs/experiments/official_tot_overnight_plan_20260618.md`。
- 已确认 `nlile/24-game` 当前下载版本为 `1362` 条且全部可解；本项目
  solver 审计为 `1362/1362` 可解，字段不一致数为 `0`。
- 已确认 `nlile/24-game` 与 `test-time-compute/game-of-24` 在 puzzle 集合上
  完全相同：二者各 `1362` 个 unique puzzles，overlap `1362`，
  `nlile-only=0`，`ToT-only=0`，同下标相同题目仅 `3/1362`，说明主要是排序和
  元数据不同。
- 今夜实验矩阵固定为：base eval；`SFT-full-data-5000` 使用完整
  `nlile/24-game` 训练，并复用此前达到强 SFT baseline 的同类 SFT-5000
  训练配置；`SFT-remove-900to1000-5000` 使用 ToT 排除 indices `900-999`
  后的 `1262` 条训练；随后分别从两个 SFT checkpoint 做 GRPO，GRPO 路线根据
  此前 repo-local split 实验分析结果设计，旧结果只作为设计依据。
- 评测关键约束：每个模型只在 `test-time-compute/game-of-24` 全量 `1362`
  条上跑一次 greedy `max_new_tokens=4096` eval；`all_1362`、`easy1262`
  和 `hard100` 指标必须从同一份 per-sample 结果中离线切分，不能重复跑分组
  GPU eval。
- failure 分类只是非阻塞后处理；主链路优先保存 raw outputs、per-sample
  verifier results、run config 和基础分组 solve rate，不能因为 failure
  分类遇到未覆盖类型而中断训练/评估。

### 2026-06-16 - Conservative GRPO pilot design active

- 旧 active plan `plans/active/20260615-sft-audit-and-repair.md` 已归档到
  `plans/archive/20260615-sft-audit-and-repair.md`。
- 新 active plan：`plans/active/20260616-grpo-pilot-design.md`。
- 新 active feature：`M3-grpo-frontier`；`M2-sft-audit-and-repair` 标记为
  `passing`，因为 strong SFT、失败分析和 rollout audit 已满足进入 M3 的证据。
- 设计文档：`docs/experiments/grpo_pilot_design_20260616.md`。
- 目标：validation strict greedy `90%+`，即至少 `123/136`，当前基线为
  `110/136`。
- Reward v1：strict verifier success `+1.0`，missing/incomplete answer
  `-0.2`，parseable wrong `-0.1`；不加独立格式奖励，不先加独立长度奖励。
- 训练池：先在 train split 上重建 active-difficulty pool，优先 mixed-reward
  prompts；硬门槛是 pool size `>=200`、mixed group rate `>=25%`、
  zero-std `<=75%`、correct-vs-truncation mixed prompts `>=50`；不能用
  validation prompts 训练后报告 validation 提升。
- 监控门禁：greedy solve rate、format/valid rate、failure mix、completion
  length、truncation rate、reward std、zero-std group rate 和 KL。
- 本地验证：`./scripts/harness-check.sh` 通过，0 warnings；`python -m compileall src scripts` 通过。
- Review 后修正：长训前必须先验收 train-split active pool；新增
  `answer_close_token_index` 等 answer-close 指标；`beta` 不再默认 `0.02`，先 probe
  `0/0.001`；`scale_rewards` 先做 `none/group` short probe；显式记录
  `mask_truncated_completions` 和 `remove_unused_columns=False`。
- Early success/stop gate 已量化：pilot 早期成功需 `>110/136`、retention
  `>=108/110`、answer-contract failures `<26/136`、wrong-answer failures
  `<=3/136`；若 TRL 不支持声明字段，compatibility probe 必须 fail fast。
- Review 后验证：`./scripts/harness-check.sh` 通过，0 warnings；`python -m compileall src scripts` 通过。
- 实现进展：新增 `src/game24_rl/rewards.py`、`src/game24_rl/grpo.py`、
  `scripts/train_grpo.py`、`scripts/build_grpo_pool.py`、`tests/test_grpo_rewards.py`
  和 `tests/test_grpo_pool.py`。当前支持 reward scoring、answer-close metrics、
  prompt-only dry-run、rollout details pool audit、TRL compatibility metadata 和
  `GRPOConfig` 实例化 probe；
  real GRPOTrainer training 仍故意禁用，等待 AutoDL compatibility probe。
- 实现验证：`python -m compileall src scripts` 通过；`pytest tests/test_grpo_rewards.py tests/test_grpo_pool.py tests/test_training_pipeline.py` 18 tests 通过；`ruff check src scripts tests` 通过；`ruff format --check src scripts tests configs` 通过；`./scripts/harness-check.sh` 通过；`pytest` 47 tests 通过。

### 2026-06-15 - SFT 审计、修复和 v3 复训准备

- 评测代码审计结论：strict verifier/aggregate scoring 未发现明显算分 bug；主要失败是模型输出 correct-format/correct-numbers 但 wrong-value。
- SFT v2 final validation：`outputs/eval/sft_v2_fixed_prompt_final_validation_20260615-010152/validation-eval-report.json`，solve_rate=`42/136=30.88%`，format_rate=`1.0`，valid_expr_rate=`1.0`。
- Case study artifact：`outputs/diagnostics/sft_case_study/summary.json`，checkpoint-1500 为 `29` ok / `107` wrong_value，train/validation overlap count 为 `0`。
- Baseline 对比：临时只读克隆 `https://github.com/LiaoMengqi/LLM4Game24` 到 `/tmp/LLM4Game24`；其强 format-v2 数据约 45k 条、使用 Qwen chat framing、显式最终 target statement 和部分 rollback/search traces。
- 修复：`trace_type` 和 `prompt_style` 从 config/CLI 贯通到 data generation、training dry-run 和 eval report；新增 `configs/sft_v3_checked_chat.yaml`。
- v3 dry-run：`python scripts/train_sft.py --config configs/sft_v3_checked_chat.yaml --dry-run` 通过；生成 `11706` 条 strict-verifier-valid checked/chat SFT records。
- 验证：`./scripts/harness-check.sh`、`python -m compileall src scripts`、focused pytest 38 tests、`pytest` 38 tests、`ruff check src scripts tests`、`ruff format --check src scripts tests configs` 均通过。
- 审计报告：`docs/experiments/sft_audit_report.md`。

### 2026-06-15 - SFT 低分后切换到严格审计任务

- 旧 active plan `0002-sft-training-readiness` 已归档到 `plans/archive/`。
- 新 active plan：`plans/active/20260615-sft-audit-and-repair.md`。
- 新 feature：`M2-sft-audit-and-repair`，唯一 active。
- `M2-first-pass-sft` 标记为 blocked：SFT v2 `checkpoint-500` solve_rate=21/136，`checkpoint-1500` solve_rate=29/136，final solve_rate=42/136。

### 2026-06-14/15 - AutoDL GitHub 网络经验

- AutoDL 直连 `https://github.com` 超时；通过本地 `127.0.0.1:2080` 代理的 SSH 反向转发可通。
- 经验文档：`docs/experiments/remote-git-sync-via-proxy.md`。

## 历史摘要

- M1 solver/verifier/data split foundation 已 passing；验收数字为 1,820 total / 1,362 solvable / 458 unsolvable。
- M2 training readiness 已实现并归档：配置解析、dry-run、checkpoint resume、strict eval artifacts、Miniconda/bootstrap、AutoDL 环境验证。
- M2 审计已收束：strong full fine-tuning SFT 达到 `110/136 = 80.88%`，并通过 rollout audit 证明可进入 M3 GRPO pilot。

### 2026-06-15 - SFT v3 remote training started

- Commit synced to AutoDL: `372e17a`.
- Remote dirty worktree before sync was preserved in stash `autodl-before-sft-v3-sync-20260615-011851`.
- AutoDL validation before launch: `./scripts/harness-check.sh` passed with expected warning for ignored case-study artifact absence; `python -m compileall src scripts` passed; `pytest tests/test_data_gen.py tests/test_training_pipeline.py` passed with 17 tests; v3 dry-run generated `11706/11706` strict-verifier-valid records.
- Training tmux session: `game24-sft-v3-checked-chat`.
- Run dir: `outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat`.
- Log file: `outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat/logs/train-20260615-011937.log`.
- Observed normal progress: tokenization completed, trainer reached about `57/4392` steps, `checkpoint-50` exists, GPU active around `6065 MiB / 49140 MiB` and `30%` utilization.

### 2026-06-15 - Fixed single-path trace experiment started

- Decision recorded: `harness/decisions.md` entry "先训练固定单路径状态转移 trace".
- Audit report updated: `docs/experiments/sft_audit_report.md` section "2026-06-15 Update: Fixed State-Trace Experiment".
- Experimental script: `scripts/experiments/run_fixed_trace_sft_experiment.py`.
- Local verification passed: `./init.sh`; `python -m compileall src scripts`; `ruff check src scripts tests`; `ruff format --check src scripts tests configs`; focused pytest 38 tests; fixed-trace build smoke with 16 strict-verifier-valid records.
- Remote build passed on AutoDL: `data/processed/experiments/fixed_trace_v1_train.jsonl` has `1089` records, `1089` unique train puzzles, one fixed correct path per puzzle, 7 completion lines each.
- Remote tmux session: `game24-fixed-trace-v1`.
- Run dir: `outputs/experiments/fixed_trace_v1_lora`.
- Log file: `outputs/experiments/fixed_trace_v1_lora/logs/run-20260615-fixed-trace-v1.log`.
- Launch command parameters: `max_steps=800`, `save_steps=200`, `learning_rate=1e-4`, `max_length=512`, `max_new_tokens=256`, `eval_checkpoints=all`, prompt style `qwen_chat`, trace type `fixed_single_path_state_trace_v1`.
- Observed normal progress: data generation, model load, EOS addition and tokenization completed; trainer reached early steps (`5/800` observed) with GPU around `6063 MiB / 49140 MiB` and `25%` utilization.

### 2026-06-15 - Parallel higher-LR fixed-trace probe started

- Motivation: GPU/CPU were underutilized by a single LoRA run; run a small parallel probe to test whether stronger LoRA training intensity improves fixed-trace learning speed.
- Experiment: same fixed single-path state-trace data and script as `fixed_trace_v1_lora`, but `learning_rate=3e-4`, `max_steps=400`, `save_steps=200`.
- Remote tmux session: `game24-fixed-trace-v1-lr3e4`.
- Run dir: `outputs/experiments/fixed_trace_v1_lora_lr3e4`.
- Logs: `outputs/experiments/fixed_trace_v1_lora_lr3e4/logs/train-20260615-fixed-trace-v1-lr3e4.log` and `outputs/experiments/fixed_trace_v1_lora_lr3e4/logs/eval-20260615-fixed-trace-v1-lr3e4.log`.
- Observed normal progress: second model loaded, training reached early steps (`18/400` observed). With both runs active, GPU was around `12128 MiB / 49140 MiB` and `84%` utilization.

### 2026-06-15 - Fixed-trace probe completed

- Main run `fixed_trace_v1_lora` finished. Final validation:
  - `checkpoint-200`: `14/136 = 10.29%`
  - `checkpoint-400`: `13/136 = 9.56%`
  - `checkpoint-600`: `20/136 = 14.71%`
  - `checkpoint-800`: `21/136 = 15.44%`
  - `final`: `21/136 = 15.44%`
- Main run failure mix at final: `120` wrong_value, `2` wrong_numbers, `14` ok.
- Higher-LR probe `fixed_trace_v1_lora_lr3e4` finished. Final validation:
  - `checkpoint-200`: `15/136 = 11.03%`
  - `checkpoint-400`: `19/136 = 13.97%`
  - `final`: `19/136 = 13.97%`
- Higher-LR probe failure mix at final: `104` wrong_value, `8` wrong_numbers, `4` syntax_error:unmatched ')', `19` ok.
- Interpretation: fixed single-path trace did not recover the expected baseline order of magnitude, and stronger LR did not improve final solve rate. Current evidence still points to a teacher/design limitation rather than a pure optimization-strength issue.
- Artifact paths:
  - `outputs/experiments/fixed_trace_v1_lora/eval/summary.json`
  - `outputs/experiments/fixed_trace_v1_lora_lr3e4/eval/summary.json`

### 2026-06-15 - Long high-LR fixed-trace run started

- Motivation: validate the user's concern that the previous fixed-trace probes
  were too short and too conservative to test the assumption fairly.
- Experiment: fixed single-path state trace, LoRA, `learning_rate=1e-3`,
  `max_steps=3000`, `save_steps=500`, `max_length=512`, `max_new_tokens=256`.
- TensorBoard enabled with `--report-to tensorboard`.
- Remote tmux session: `game24-fixed-trace-v1-lr1e3-long`.
- Run dir: `outputs/experiments/fixed_trace_v1_lora_lr1e3_long`.
- Train log: `outputs/experiments/fixed_trace_v1_lora_lr1e3_long/logs/train-20260615-fixed-trace-v1-lr1e3-long.log`.
- TensorBoard event file exists under `outputs/experiments/fixed_trace_v1_lora_lr1e3_long/runs/`.
- Observed normal progress: model loaded and training reached about `129/3000`; GPU around `6067 MiB / 49140 MiB` and `24%` utilization.
- Estimated duration from launch: roughly `90-100` minutes including automatic evaluation.

### 2026-06-15 - Baseline gap analysis documented

- Document: `docs/experiments/baseline_gap_analysis.md`.
- Main findings:
  - LLM4Game24 uses full fine-tuning, while this repo has used LoRA.
  - LLM4Game24 format-v1/v2 have about `45k` records and usually `33-36`
    records per puzzle; current fixed trace has `1089` records, one per puzzle.
  - LLM4Game24 format-v2 is a compressed DFS/search-tree trace with rollback,
    not a single success path.
  - LLM4Game24 selects best checkpoint by eval loss; this repo mostly evaluates
    saved checkpoints after training.
  - Larger max length/generation budget matters if moving to search traces.

### 2026-06-15 - Multi-path/search-trace solver idea recorded

- Decision recorded in `harness/decisions.md`.
- Rationale: current solver is fast enough that full-path or search-tree export
  is not the bottleneck; keeping only the first DFS solution is a teacher-data
  design choice, not a compute requirement.
- Intended use: generate controlled SFT ablations that compare first-path,
  multiple-success-path, and compressed-search-trace data while keeping the
  same split, strict verifier, and `<answer>...</answer>` contract.

### 2026-06-15 - Baseline-format v2 full finetune backup result

- Purpose: create a stronger SFT backup by moving closer to the reference
  baseline recipe while preserving this repository's split, strict verifier, and
  `<answer>...</answer>` reported contract.
- Script: `scripts/experiments/run_rollback_sft_experiment.py`.
- Training command used `--mode train` on the already-built dataset, not
  `--mode all`, to avoid rebuilding rollback traces.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Training mode: full finetuning via Transformers `Trainer`, prompt masking,
  `optim=adamw_torch`, `learning_rate=5e-5`, `max_steps=400`,
  `save_steps=200`, `max_length=1024`, `max_new_tokens=1024`.
- Dataset:
  `data/processed/experiments/game24-baseline-format-v2-qwen-answer-train.jsonl`
  with `35324` records, `1011` unique train multisets, `32355` rollback records,
  and validation overlap `0`.
- AutoDL session: `game24-baseline-v2-full`.
- Run dir: `outputs/experiments/baseline_format_v2_full`.
- Eval summary:
  `outputs/experiments/baseline_format_v2_full/eval/summary.json`.
- Result: `checkpoint-400` and `final` both reached `51/136 = 37.50%`
  validation solve rate, with `format_rate=38.97%` and
  `valid_expr_rate=38.24%`.
- Interpretation: this is the best repo-native backup so far and improves over
  SFT v2 final `30.88%`, but it remains far below the reference baseline.
  Remaining work should compare raw failures and training/data differences
  against baseline before another major run.

### 2026-06-16 - Strong full fine-tuning SFT result

- Independent record:
  `docs/experiments/sft_full_finetune_search_trace_20260616.md`.
- Training chain: full fine-tuning continued from the local `400`-step checkpoint
  to `800` steps, then from `checkpoint-800` to `5000` steps.
- Final run dir:
  `outputs/experiments/baseline_format_v2_full_5000_from800`.
- Final training metrics: runtime about `1:28:22`, train loss `0.06988`,
  Trainer eval loss `0.06184`.
- Strict validation artifact:
  `outputs/experiments/baseline_format_v2_full_5000_from800/eval/summary.json`.
- Strict validation result: `checkpoint-4500` and `final` both reached
  `110/136 = 80.88%` solve rate, with `format_rate=80.88%` and
  `valid_expr_rate=80.88%`.
- Interpretation: the `400`-step full fine-tuning result was undertrained; the
  current full SFT checkpoint is now a strong fallback and a credible warm start
  for later optimization.

### 2026-06-16 - TensorBoard enabled for future experiments

- Decision: future SFT experiments should write TensorBoard event files by
  default, because they are easier to inspect and reuse in the course report than
  ad hoc SVG exports.
- Implementation: `scripts/experiments/run_rollback_sft_experiment.py` now
  defaults to `--report-to tensorboard` and supports overriding integrations with
  `--report-to`.
- Config update: `configs/sft_v1.yaml` and `configs/sft_v3_checked_chat.yaml`
  now set `training.report_to: [tensorboard]`.

### 2026-06-16 - Strong SFT curve and failure analysis

- Pulled local analysis artifacts from AutoDL for the strong SFT run:
  `outputs/experiments/baseline_format_v2_full_5000_from800/metrics/` and
  `outputs/experiments/baseline_format_v2_full_5000_from800/eval/`.
- Sparse strict-validation trajectory:
  `checkpoint-400` reached `51/136=37.50%`, `checkpoint-800` reached
  `70/136=51.47%`, and `checkpoint-4500`/`final` both reached
  `110/136=80.88%`.
- Failure mix improved from mostly answer-contract failures at step `400`
  (`83`) and step `800` (`62`) to `26` answer-contract failures and no
  arithmetic/number failures at `4500`/`final`.
- `checkpoint-4500` and `final` produced byte-identical raw greedy outputs for
  all `136` validation puzzles; the equal metrics are therefore a real behavior
  plateau under greedy decoding, not an aggregate reporting accident.
- Case study shows the remaining failures are long rollback/search continuations
  truncated before `</think><answer>...`; none contains an `<answer>` block.
- Training metrics were exported from retained `trainer_state.json`; because
  `save_total_limit=1`, dense checkpoint-wise accuracy cannot be recovered for
  this historical run.
- Observability fix: `scripts/export_training_metrics.py` now only records plots
  that were actually generated, and
  `scripts/experiments/run_rollback_sft_experiment.py` exposes
  `--save-total-limit` for future long SFT runs.

### 2026-06-16 - GRPO rollout audit

- Added script: `scripts/experiments/audit_rollout_distribution.py`.
- Report: `docs/experiments/grpo_rollout_audit_20260616.md`.
- Artifacts:
  `outputs/experiments/baseline_format_v2_full_5000_from800/rollout_audit/`.
- Validation pilot with `temperature=0.8`, `top_p=0.95`, `G=4`,
  `max_new_tokens=1024` on 32 prompts: output solve rate `86/128=67.19%`,
  pass@4 `30/32=93.75%`, mixed groups `16/32`.
- Targeted audit on the `26` greedy validation failures with `G=8`,
  `max_new_tokens=1024`: output solve rate `94/208=45.19%`, pass@8
  `22/26=84.62%`, mixed groups `19/26`.
- Interpretation: GRPO has useful group-level reward variance; most greedy
  failures already have correct sampled trajectories. Main risk is
  length/search-control because sampled completion p50/p95 both hit `1024` and
  answer-contract truncation remains common.

### 2026-06-16 - GRPO short pilot beta0 none failed stop gate

- Local fix commit: `48eede5 Fix GRPO config for TRL 1.6`; removed unsupported
  `max_prompt_length` from `GRPOConfig` kwargs and records the requested prompt
  limit in run metadata instead.
- AutoDL run dir:
  `/root/autodl-tmp/projects/grpo-short-pilot/beta0_none_25`.
- Train command: `max_steps=25`, `num_generations=4`, `temperature=0.8`,
  `top_p=0.95`, `max_completion_length=1024`, `learning_rate=5e-6`,
  `beta=0.0`, `scale_rewards=none`,
  `mask_truncated_completions=false`, `remove_unused_columns=false`,
  `prompt_records=88`.
- Training completed successfully in about `8m14s`; GPU samples during training
  were around `87-98%` utilization and `31133 MiB / 49140 MiB`.
- Strict validation eval completed on the same validation split and prompt
  style as the SFT baseline. Result for both `checkpoint-25` and `final`:
  `89/136 = 65.44%`, `format_rate=66.91%`,
  `valid_expr_rate=65.44%`.
- Baseline comparison: strong SFT final remains `110/136 = 80.88%`; GRPO short
  pilot retained only `79/110` baseline successes, lost `31`, and added `10`
  new successes.
- Failure mix after GRPO: `45` answer-contract failures and `2` wrong-number
  failures. Training logs show high truncation pressure:
  mean `completions/clipped_ratio=0.34`, max `1.0`, final step `0.75`;
  mean completion length `643.19`, final step `928.0`.
- Decision: this exact config fails the early stop gate and must not be
  expanded to long training. Next GRPO attempt should reduce destructive update
  risk before spending more GPU, e.g. shorter/safer probe, lower LR, fewer
  updates, smaller active subset, KL/reference beta probe, or targeted SFT
  baseline first.

### 2026-06-17 - GRPO route search best reaches 116/136

- New implementation support: `scripts/train_grpo.py` now exposes
  `--gradient-accumulation-steps`, enabling `num_generations=8` without TRL
  effective-batch divisibility failures.
- Current best AutoDL run:
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5`.
- Best config: filtered 37-prompt pool selected by `1..2` correct samples plus
  at least one truncation, `num_generations=8`, `gradient_accumulation_steps=8`,
  LoRA r16, `learning_rate=5e-7`, `beta=0.001`, `scale_rewards=none`,
  `max_steps=5`, strict reward.
- Best result: validation `116/136 = 85.29%`, retained `109/110` SFT successes,
  lost `1`, added `7`, failure mix `20` answer-contract failures and `0`
  wrong-answer failures. This passes the pilot early success gate but remains
  below the final `123/136` target.
- Negative probes: same config with `10` steps fell to `114/136`; `beta=0.002`
  fell to `111/136`; original 88-prompt mixed pool with G=8/lr5e-7 fell to
  `108/136`. Do not expand these branches.
- Remaining failures in the best run are still long rollback/search
  continuations that hit the 1024-token budget before any `<answer>` block.
  Next useful route is a bounded closure-aware reward profile or a more targeted
  train pool for remaining long-search failures, not more blind step/beta/pool
  expansion.
- Close-bonus reward profile probe:
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_closebonus_beta001_filtered_g8_lr5e7_5`
  reached `115/136 = 84.56%`, retained `108/110`, added `7`, and had `21`
  answer-contract failures. The small bonus for early correct answer closure did
  not beat strict reward best, so do not expand this exact close-bonus profile.
- Current-checkpoint train hard audit:
  `/root/autodl-tmp/projects/grpo-route-audit/train_hard_le2trunc_best116_g4_t08_len512`
  sampled 43 train hard prompts from the best `116/136` adapter with `G=4` and
  `max_new_tokens=512`; result `pass@4=27/43`, mixed groups `25/43`,
  zero-std groups `18/43`, truncation-like failures `118/172`. The accepted
  pool manifest selected 24 hard mixed prompts, confirming usable train signal
  still exists but the bottleneck remains answer closure.
- Second-stage continuation probe:
  `/root/autodl-tmp/projects/grpo-short-pilot/continue116_trainhard25_g4len512_g8_lr2e7_5`
  loaded the best `116/136` LoRA adapter as trainable initial adapter and ran
  5 GRPO steps on the hard mixed pool with `learning_rate=2e-7`, `beta=0.001`,
  `G=8`. Training was healthy (`reward_std > 0`, KL about `8e-5`) but
  validation fell to `112/136 = 82.35%`: retained `111/116`, lost `5`, gained
  `1`, and all `24` failures were still answer-contract failures with no
  `<answer>` block. Do not expand this route without a new stopping/closure
  objective.
- Verifier-rerank decoding route:
  `eval_checkpoint.py` now supports greedy-then-sampled strict-verifier rerank
  via `--raw-outputs` plus `--sampled-raw-outputs`. Using best `116/136`
  greedy outputs and the existing G8 sampled candidates for its 20 greedy
  failures produced `133/136 = 97.79%` validation with the same strict verifier.
  Artifact:
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/eval_verifier_rerank_greedy_plus_failures_g8/validation-verifier-rerank-eval-report.json`.
  This is a decoding/search-selection result, not a greedy checkpoint gain.
- Test split confirmation:
  best greedy GRPO adapter scored `116/137 = 84.67%` on test; all `21`
  failures were answer-contract/no-answer. Failure-only G8 sampled rollout
  solved `20/21` of those test greedy failures, and greedy-then-sampled strict
  verifier rerank scored `136/137 = 99.27%` on test. Artifacts:
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/eval_test_greedy/test-eval-report.json`
  and
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/eval_test_verifier_rerank_greedy_plus_failures_g8/test-verifier-rerank-eval-report.json`.
- Direct long-token greedy route:
  using the same best GRPO LoRA adapter as a single model with greedy decoding
  and no verifier-rerank, `max_new_tokens=2048` reached validation
  `123/136 = 90.44%` and test `122/137 = 89.05%`; unified
  `max_new_tokens=4096` reached validation `126/136 = 92.65%` and test
  `129/137 = 94.16%`. The matched strong SFT final 4096 baseline reached
  validation `123/136 = 90.44%` and test `128/137 = 93.43%`, so the direct
  90%+ result is primarily a long-token-budget effect, with GRPO adding a
  smaller same-budget gain of `+3/136` validation and `+1/137` test.
  Artifacts:
  `/root/autodl-tmp/projects/grpo-direct-long/best116_validation_greedy_4096/validation-eval-report.json`
  and
  `/root/autodl-tmp/projects/grpo-direct-long/best116_test_greedy_4096/test-eval-report.json`.
  SFT baseline artifacts:
  `/root/autodl-tmp/projects/sft-direct-long/sft_final_validation_greedy_4096/validation-eval-report.json`
  and
  `/root/autodl-tmp/projects/sft-direct-long/sft_final_test_greedy_4096/test-eval-report.json`.
  Remaining answer-contract failures under 4096 still have zero `<answer>` and
  zero `</answer>` tags; outputs are still inside rollback/search, not malformed
  answer blocks.
  Chinese record:
  `docs/experiments/direct_long_token_greedy_20260617.md`.
- Eval observability fix:
  `generate_checkpoint_outputs` now appends raw-output JSONL per batch and prints
  `generated x/y records`, so long `2048/4096` evaluations can be monitored
  before final report writing.
