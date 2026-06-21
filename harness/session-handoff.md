<!--
职责：提供最新的紧凑交接信息，让新 agent 能无歧义恢复当前任务。
边界：只保留当前可恢复状态；历史放 progress.md，稳定事实放 docs 或代码。
-->

# 会话交接

## 仓库状态

- 分支：`dev/grpo-pilot-design`。
- 当前功能项：`M3-grpo-frontier`
- 当前 active plan：`plans/active/20260616-grpo-pilot-design.md`
- 当前本地状态：strong full fine-tuning SFT 的 `80.88%` 结果、rollout audit 和 conservative GRPO pilot 设计已记录。
- 最新已推送代码 commit：本文件可能滞后于当前本地提交；以 `git log -1 --oneline` 为准。

## 当前模式

- Collaboration mode：高自治执行，关键边界升级；当前围绕 M3 GRPO pilot 做可执行设计和后续最小实现。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 当前边界：不改主模型、split、answer contract、verifier 接受标准；conservative GRPO pilot 已授权。
- 当前实验退出条件：设计已落地；当前最佳 greedy GRPO short probe 是
  `lora_r16_beta001_filtered_g8_lr5e7_5` 的 `116/136 = 85.29%`，
  retention `109/110`，answer-contract failures `20`，wrong-answer `0`。
  second-stage hard-pool continuation 已退到 `112/136`，不要继续盲目加大
  RL 强度。同口径补跑 strong SFT final 的 direct greedy
  `max_new_tokens=4096` 后，SFT 本身达到 validation `123/136 = 90.44%`
  和 test `128/137 = 93.43%`；当前最佳 GRPO adapter 在同预算下达到
  validation `126/136 = 92.65%`、test `129/137 = 94.16%`，净增分别是
  `+3/136` 和 `+1/137`。更高的 inference-time strict verifier rerank 是另一个
  口径：validation `133/136 = 97.79%`，test `136/137 = 99.27%`。

## 最新补充：handoff3 Countdown transfer

- handoff3 目标：基于 handoff2 GRPO checkpoint-500，评估
  `Jiayi-Pan/Countdown-Tasks-3to4` 的 3-4 数字凑任意 target 加分项。
- 新增代码：`src/game24_rl/countdown_solver.py`、`scripts/build_countdown_sft_data.py`、
  `scripts/build_countdown_stratified_eval_manifest.py`、
  `scripts/experiments/eval_countdown_stratified.py`、
  `scripts/experiments/run_countdown_sft_adaptation.py`，并扩展 verifier 支持
  3/4 数字任意 target。
- AutoDL 数据：`data/raw/hf/Jiayi-Pan__Countdown-Tasks-3to4/default__train.jsonl`
  共 `490364` 行。
- 100 题 stratified eval manifest：
  `outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json`，
  `1/2/3/4` solution buckets 各 25 题。
- handoff2 zero-shot：`0/100`，`format_ok=9/100`，`valid_expr=5/100`。
- 后续复盘确认上述 handoff2 zero-shot 使用的 `qwen_chat_target` 不是最小迁移
  prompt，因为它把 user 输入改成了 `Numbers:` / `Target:`。当前正确最小迁移
  prompt 是 `qwen_chat_minimal_target`：system 中只把 `24` 替换成目标数字，
  user 段仍只放裸数字。
- handoff2 minimal-transfer greedy16：`0/16`，`format_ok=1/16`，`valid_expr=0/16`，
  15 条是 answer-contract/no-answer，1 条 wrong_numbers。Artifact：
  `outputs/experiments/handoff3_countdown_minimal/eval_handoff2_minimal_greedy16`。
- handoff2 minimal-transfer sampled audit：原计划 `32 x G8 x 4096` 运行 60 分钟
  仍无 summary，停止；改跑 `8 x G4 x 4096`，结果 `0/32`，`pass@4=0/8`，
  `format_ok=3/32`，`valid_expr=0/32`，completion len mean/p50/p95 全是 `4096`。
  失败为 `29` answer-contract/no-answer、`3` wrong_numbers。Artifact：
  `outputs/experiments/handoff3_countdown_minimal/rollout_handoff2_minimal_8_g4`。
  结论：最小迁移能恢复 rollback/search 风格，但没有直接 GRPO 所需的正样本信号。
- Countdown SFT-400-200：partial eval 前 8 条 `1/8`，仍有长输出/格式问题。
- Countdown SFT-2000-600 当前最佳 adapter：
  `outputs/experiments/handoff3_countdown_adapt/sft_train_handoff2_continue_2000_600steps/final`。
  训练 `train_loss=0.09375`。
- SFT-2000-600 smoke eval：first-16 `1/16`，balanced-16 `1/16`；
  两者 `format_ok=16/16`、`valid_expr=16/16`，主失败是 `wrong_value`。
- SFT-2000-600 sampled audit：16 prompts x 8 samples 只有 `1/128` 正确，
  `pass@8=1/16`，15 个 all-wrong group。因此暂不建议直接 GRPO 长训；
  下一步应先做 Countdown curriculum SFT，再复查 sampled success。
- 最新有效 handoff3 路线已切到 balanced low-solution SFT，再接 target-distance
  GRPO：训练 256 个 train prompts，`G=8`，`max_steps=1200`，
  `lr=1e-6`，`beta=0.001`，`scale_rewards=none`，`dr_grpo`。
  最终 adapter：
  `outputs/experiments/handoff3_countdown_balanced_grpo_probe/grpo_targetdistance_from_balanced_sft_train256_g8_lr1e6_beta001_1200/final`。
  full100 held-out greedy `26/100`，train256 greedy `111/256 = 43.36%`；
  full100 sampled audit strict `pass@8=46/100`、solved samples `177/800`。
  结论：相对 balanced SFT 的 full100 greedy `23/100` 只有小幅提升，但
  strict pass@8 只从 `45/100` 到 `46/100`，主要失败仍是 `wrong_value`，
  不是截断或格式问题。注意旧 `summary.pass_at_k=87/100` 是 target-distance
  reward 非零口径，不能当 strict solve pass@8。
- 文档：`docs/handoff/handoff3.md`、
  `docs/experiments/countdown_transfer_eval_20260619.md`、
  `docs/experiments/countdown_adaptation_plan_20260619.md`、
  `docs/experiments/countdown_handoff3_grpo_20260620.md`。

## 当前已验证状态

- SFT v2 final eval：validation `42/136 = 30.88%`，format/valid expression 均为 `100%`。
- 审计报告：`docs/experiments/sft_audit_report.md`。
- Case study artifact：`outputs/diagnostics/sft_case_study/summary.json`。
- 修复：`trace_type` 和 `prompt_style` 现在从 config/CLI 贯通到数据生成、训练 dry-run 和 eval report。
- 新配置：`configs/sft_v3_checked_chat.yaml`，run name `qwen25_15b_lora_sft_v3_checked_chat`。
- 新实验脚本：`scripts/experiments/run_fixed_trace_sft_experiment.py`。
- 新决策记录：`harness/decisions.md` 的 `2026-06-15 - 先训练固定单路径状态转移 trace`。
- 新报告补充：`docs/experiments/sft_audit_report.md` 的 `2026-06-15 Update: Fixed State-Trace Experiment`。
- 本地验证通过：`./scripts/harness-check.sh`；`python -m compileall src scripts`；focused pytest 38 tests；`pytest` 38 tests；`ruff check src scripts tests`；`ruff format --check src scripts tests configs`；v3 dry-run。
- 远端预启动验证：`python -m compileall src scripts`；`pytest tests/test_data_gen.py tests/test_training_pipeline.py` 17 tests；v3 dry-run；`11706/11706` completions strict-verifier valid。
- fixed-trace 本地验证通过：`./init.sh`；`python -m compileall src scripts`；`ruff check src scripts tests`；`ruff format --check src scripts tests configs`；focused pytest 38 tests；16-record build smoke。
- fixed-trace 远端 build 通过：`data/processed/experiments/fixed_trace_v1_train.jsonl` 有 `1089` records / `1089` unique train puzzles。
- fixed-trace 主 run final validation：`21/136 = 15.44%`。
- fixed-trace higher-LR probe final validation：`19/136 = 13.97%`。
- baseline-format v2 full finetune final validation：`51/136 = 37.50%`，
  `format_rate=38.97%`，`valid_expr_rate=38.24%`。
- baseline-format v2 full finetune artifacts：
  `outputs/experiments/baseline_format_v2_full/eval/summary.json`，
  `outputs/experiments/baseline_format_v2_full/eval/final/validation-eval-report.json`。
- strong full fine-tuning SFT result：`110/136 = 80.88%` validation solve rate，
  `format_rate=80.88%`，`valid_expr_rate=80.88%`。
- strong full fine-tuning SFT record：
  `docs/experiments/sft_full_finetune_search_trace_20260616.md`。
- strong full fine-tuning artifacts：
  `outputs/experiments/baseline_format_v2_full_5000_from800/eval/summary.json`，
  `outputs/experiments/baseline_format_v2_full_5000_from800/final/`。
- future SFT experiments should use TensorBoard logging by default; the 80.88%
  historical run only has Trainer state/log-derived metrics, not event files.
- strong SFT analysis artifacts pulled locally:
  `outputs/experiments/baseline_format_v2_full_5000_from800/metrics/`,
  `outputs/experiments/baseline_format_v2_full_5000_from800/eval/`, and
  `outputs/experiments/baseline_format_v2_full_5000_from800/analysis/checkpoint_trajectory_and_failures.json`.
- GRPO rollout audit report:
  `docs/experiments/grpo_rollout_audit_20260616.md`.
- GRPO rollout artifacts:
  `outputs/experiments/baseline_format_v2_full_5000_from800/rollout_audit/`.
- GRPO pilot design:
  `docs/experiments/grpo_pilot_design_20260616.md`.
- GRPO active plan:
  `plans/active/20260616-grpo-pilot-design.md`.
- GRPO scaffold implementation:
  `src/game24_rl/rewards.py`, `src/game24_rl/grpo.py`,
  `scripts/train_grpo.py`, `scripts/build_grpo_pool.py`,
  `tests/test_grpo_rewards.py`, `tests/test_grpo_pool.py`.
- Sparse strict-validation trajectory:
  `checkpoint-400` `51/136=37.50%`, `checkpoint-800` `70/136=51.47%`,
  `checkpoint-4500`/`final` `110/136=80.88%`.
- `checkpoint-4500` and `final` raw greedy outputs are byte-identical for all
  `136` validation puzzles; remaining `26` failures are all
  answer-contract/truncation cases with no `<answer>` block.
- Rollout audit found useful GRPO signal: validation pilot pass@4 `30/32` with
  `16/32` mixed groups; targeted greedy-failure pass@8 `22/26` with `19/26`
  mixed groups. Main risk is length/search-control because sampled completion
  p50/p95 hit the generation budget.

## 当前任务

当前任务是 M3 conservative GRPO pilot。设计目标是把 strong SFT 的 validation
strict greedy `110/136 = 80.88%` 推到 `90%+`，即至少 `123/136`。本轮已完成
方案设计、最小安全 scaffold、AutoDL compatibility probe、train-pool audit 和一轮
真实 short pilot。`beta=0.0`、`scale_rewards=none`、`25` step 配置已经跑通但
严格验证降到 `89/136 = 65.44%`，触发 stop gate；不要扩大这个配置。

## 当前判断

- 低分不是明显 eval 聚合 bug；主要失败是 correct-format/correct-numbers/wrong-value。
- 已确认的工程问题：`trace_type` 配置之前存在但未生效；eval 之前不支持/记录 prompt style。
- 小样本 overfit 未发现核心训练链路 bug；更高概率问题是 teacher/data design。
- baseline-converted format_v2 > format_v1 说明状态/search 信息有用；但长 rollback/search trace 容易诱发 `<answer>` 契约失败。
- fixed trace 是 short-success 与 search-state 的折中：先证明模型能学会稳定的状态更新和 answer closure，再决定是否进入 full finetune、提高学习率或多候选搜索 trace。
- fixed trace final only reached `21/136` and higher LR reached `19/136`，所以“单路径固定状态 trace + LoRA”本身不是当前缺失的关键因素；单纯更高 LR 也没有解决泛化。
- longer full fine-tuning reached `80.88%` strict validation solve rate, so the
  decisive issue in the `400`-step run was undertraining rather than an inherent
  mismatch in the search-trace recipe.
- `4500` 到 `final` 没有变化不是报告聚合 bug；raw outputs 完全相同，结合 cosine
  LR 在 `4500` 已降到约 `1.30e-6`，更像 greedy 解码行为 plateau。
- 当前剩余瓶颈不是格式正确后的算术错误，而是少数题目的 rollback/search trace
  过长并耗尽 `1024` generation budget，没能输出 `<answer>`。
- Sampling audit 支持进入 GRPO pilot：大多数 greedy 失败题在采样分布中已有
  正确轨迹，RL 目标应是把这些正确且及时收束的轨迹提升到 greedy 路径。
- GRPO pilot 必须防止 length reward hack 或搜索失控；不能只看 reward_mean。
- 第一版 reward：strict verifier success `+1.0`，missing/incomplete answer
  `-0.2`，parseable wrong `-0.1`；不加独立格式奖励，不先加独立长度奖励。
- 训练池必须在 train split 上重建 active-difficulty pool；validation audit
  只作为设计证据，不能直接成为训练数据后再报告 validation gain。
- Review 后新增门槛：长训前必须先验收 train pool mixed signal；监控
  `answer_close_token_index`、`tokens_after_answer` 和 wrong-answer rate；AutoDL
  compatibility probe 必须验证 TRL 额外列、`remove_unused_columns=False`、
  `mask_truncated_completions`、`beta=0/0.001` 和 `scale_rewards=none/group`。
- Train-pool audit 硬门槛：pool size `>=200`，mixed group rate `>=25%`，
  zero-std `<=75%`，correct-vs-truncation mixed prompts `>=50`。
- Pilot early success：`>110/136`，retention `>=108/110`，
  answer-contract failures `<26/136`，wrong-answer failures `<=3/136`。
- AutoDL short pilot `grpo-short-pilot/beta0_none_25` 已完成，训练本身能跑通：
  25 steps、约 `8m14s`、GPU 训练采样约 `87-98%` utilization / `31.1GB`。
  但 validation 结果为 `89/136 = 65.44%`，只保留原 SFT `79/110`
  successes，丢 `31`，新增 `10`；failure mix 为 `45` answer-contract 和
  `2` wrong-number。该配置不能长训。
- Probe 结论补充：`lora_r16_beta001_none_10` validation `113/136 = 83.09%`，比 5-step 的 `114/136` 略差；`scale_rewards=group` 的 5-step 只有 `108/136 = 79.41%`，因此当前最优还是 `lora_r16_beta001_none_5`。
- 新增负结果：filtered pool 的 5-step probe 与当前最佳同为 `114/136` 但未突破；filtered + `max_completion_length=512` 退到 `113/136`；targeted SFT full refresh 从 sampled success 训练 `50` step 后退到 `95/136`，retention 只有 `87/110`，不要继续这条 SFT refresh。
- 当前最佳更新：filtered 37-prompt pool + `num_generations=8` +
  `gradient_accumulation_steps=8` + `learning_rate=5e-7` + `beta=0.001` +
  `max_steps=5` reached validation `116/136 = 85.29%` with retention
  `109/110` and `20` answer-contract failures. The corresponding 10-step run
  fell to `114/136`; `beta=0.002` fell to `111/136`; original 88-prompt mixed
  pool with the same G8/lr5e-7 recipe fell to `108/136`. Do not expand those
  negative branches.
- Remaining failures in the best run are still long rollback/search outputs
  truncated before any `<answer>` block. Next useful route is closure-aware
  reward or a more targeted train pool, not more blind step/beta expansion.
- Close-bonus reward profile has been implemented behind `--reward-profile
  close_bonus`, with default strict reward unchanged. Its first AutoDL probe
  reached `115/136 = 84.56%`, retained `108/110`, and had `21` answer-contract
  failures, so do not expand this exact profile.
- Current-checkpoint hard train audit confirmed usable but narrow GRPO signal:
  `/root/autodl-tmp/projects/grpo-route-audit/train_hard_le2trunc_best116_g4_t08_len512`
  sampled 43 train hard prompts from the best adapter with `G=4`,
  `max_new_tokens=512`; `pass@4=27/43`, mixed groups `25/43`, zero-std groups
  `18/43`, truncation-like failures `118/172`. Accepted pool selected 24 hard
  mixed prompts.
- Second-stage continuation from the best `116/136` adapter on that hard pool:
  `/root/autodl-tmp/projects/grpo-short-pilot/continue116_trainhard25_g4len512_g8_lr2e7_5`
  ran 5 GRPO steps with `learning_rate=2e-7`, `beta=0.001`, `G=8`. It fell to
  `112/136 = 82.35%`, retained `111/116`, lost 5, gained 1; all failures still
  had no `<answer>` block. This argues against a simple “more RL intensity hits
  a critical point” strategy.
- Verifier-rerank decoding is now implemented in `eval_checkpoint.py` via
  `--raw-outputs` and `--sampled-raw-outputs`. Using best `116/136` greedy
  outputs plus existing G8 sampled candidates for the 20 greedy failures reached
  `133/136 = 97.79%` validation under the same strict verifier. Artifact:
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/eval_verifier_rerank_greedy_plus_failures_g8/validation-verifier-rerank-eval-report.json`.
  Treat this as decoding/search-selection evidence, not greedy checkpoint
  improvement.
- Test split confirms the rerank route generalizes: best greedy GRPO adapter
  scored `116/137 = 84.67%` on test; all `21` failures were
  answer-contract/no-answer. Failure-only G8 sampled rollout solved `20/21`
  test greedy failures, and strict verifier rerank scored
  `136/137 = 99.27%`. Artifacts:
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/eval_test_greedy/test-eval-report.json`
  and
  `/root/autodl-tmp/projects/grpo-short-pilot/lora_r16_beta001_filtered_g8_lr5e7_5/eval_test_verifier_rerank_greedy_plus_failures_g8/test-verifier-rerank-eval-report.json`.
- Direct long-token greedy evaluation is now the main “single model direct
  inference” result. With the same best GRPO LoRA adapter, greedy
  `max_new_tokens=2048` reached validation `123/136 = 90.44%` and test
  `122/137 = 89.05%`; unified greedy `max_new_tokens=4096` reached validation
  `126/136 = 92.65%` and test `129/137 = 94.16%`. Matched strong SFT final
  4096 baseline reached validation `123/136 = 90.44%` and test
  `128/137 = 93.43%`, so long-token budget is the main direct-greedy gain and
  GRPO adds a smaller same-budget gain. Artifacts:
  `/root/autodl-tmp/projects/grpo-direct-long/best116_validation_greedy_4096/validation-eval-report.json`
  and
  `/root/autodl-tmp/projects/grpo-direct-long/best116_test_greedy_4096/test-eval-report.json`.
  SFT baseline artifacts:
  `/root/autodl-tmp/projects/sft-direct-long/sft_final_validation_greedy_4096/validation-eval-report.json`
  and
  `/root/autodl-tmp/projects/sft-direct-long/sft_final_test_greedy_4096/test-eval-report.json`.
  Remaining 4096 answer-contract failures still have zero `<answer>` and zero
  `</answer>` tags, meaning they are still unclosed rollback/search outputs.
  Chinese record: `docs/experiments/direct_long_token_greedy_20260617.md`.
- Closure-strict reward probe:
  `lora_r16_closurestrict_beta001_filtered_g8_lr5e7_5` completed training and
  validation 1024 eval, but scored only `114/136 = 83.82%`; it did not beat the
  existing strict-reward best `116/136`, so do not expand this exact profile.
- Long eval observability fix: `generate_checkpoint_outputs` now appends raw
  outputs per batch and prints `generated x/y records`, so future long-token
  evaluations can be monitored while running.
- Handoff3 Countdown target-alignment GRPO negative result:
  based on Countdown curriculum SFT, two GRPO attempts were run on AutoDL.
  `target_alignment` reward (`+1/-1`, 400 steps, 51 mixed prompts) reached
  balanced16 `0/16` and train128 `56/128`. `target_distance` reward
  (distance shaping, 800 steps, 177 reward-variance prompts) reached balanced16
  `0/16` and train128 `57/128`. Both kept `format_ok=100%` and failed primarily
  as `wrong_value`; raw outputs show the model computes a value not equal to
  `Target:` and still submits it. Detailed record:
  `docs/experiments/countdown_grpo_alignment_20260620.md`.
- Handoff3 final-chance broad-solvable SFT completed on AutoDL:
  `scripts/experiments/run_countdown_final_chance_fast.sh` built 20000 unique
  solvable Countdown SFT records with eval full100 excluded, then continued
  from balanced low-solution SFT for 6000 steps (`lr=1e-5`, `max_length=1024`).
  The run finished end-to-end and produced checkpoints 1500/3000/4500/6000,
  full100 greedy eval, and G8 sampled audit under
  `outputs/experiments/handoff3_countdown_final_chance_fast`.
  Checkpoint sweep was `15/100`, `19/100`, `16/100`, `19/100`, final `19/100`;
  sampled audit was strict pass@8 `43/100` with 137/800 solved outputs.
  This is worse than balanced SFT `23/100` and target-distance GRPO `26/100`.
  Conclusion: simply adding broad solvable unique puzzles dilutes the low-solution
  eval distribution and is not a useful continuation route. Handoff summary:
  `docs/handoff/handoff3.md`; detailed record:
  `docs/experiments/countdown_handoff3_grpo_20260620.md`.
- Handoff3 minimal-transfer audit supersedes the old target-aware prompt route
  as the current evidence source. Updated records:
  `docs/experiments/countdown_handoff3_sft_baseline_20260620.md`,
  `docs/experiments/countdown_transfer_eval_20260619.md`, and
  `docs/handoff/handoff3.md`.
- Handoff3 target-replacement SFT finished:
  `outputs/experiments/handoff3_countdown_sft/sft_handoff2_target_replacement_20000_2400steps/final`.
  It trained from handoff2 on 20000 solver traces, `max_steps=2400`,
  `lr=1e-5`, `max_length=4096`, prompt = target replacement with bare-number
  user input, completion = solver trace + `<answer>` without `Check:`.
  Full100 greedy eval reached `5/100`, `format_ok=100/100`, `valid_expr=98/100`;
  failures were mostly `wrong_value=93`. Full100 sampled audit `G=8` reached
  `31/800` solved outputs, `pass@8=13/100`, `mixed_groups=13`,
  `all_wrong_groups=87`, mean length `90.84`, p95 length `113`. This is better
  than handoff2 zero-shot but still weak for direct GRPO long training.
- Handoff3 balanced low-solution SFT finished:
  `outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final`.
  It continued from the target-replacement SFT final on 20000 additional solver
  traces balanced across solution-count buckets `1/2/3/4` with `5000` records
  each, still using `qwen_chat_minimal_target`, bare-number user input,
  `max_steps=2400`, `lr=1e-5`, and `max_length=4096`. Full100 greedy eval
  reached `23/100`, `format_ok=100/100`, `valid_expr=98/100`; failures were
  mostly `wrong_value=75`. Full100 sampled audit `G=8` reached `134/800`
  solved outputs, `pass@8=45/100`, `mixed_groups=42`, `all_wrong_groups=55`,
  mean length `88.84`, p95 length `112`. This validates the balanced SFT
  direction and provides enough sampled positive signal for a conservative
  GRPO probe, though it is not yet a final-quality Countdown result.

## 仍损坏或未验证

- direct long-token eval support、verifier-rerank eval support、harness updates
  和 second-stage continuation evidence 已通过本地验证；以最新 git commit 为代码事实源。
- 本地未安装 `trl`，真实 GRPO 训练/评估仍只能在 AutoDL train env 验证。
- 本地 `scripts/audit_sft_dataset.py` 因本地缺少 `transformers` 未运行；已用 repo-local JSONL + strict verifier 做替代数据审计。
- AutoDL 直连 GitHub 不稳定；同步需要代理公式。
- AutoDL 当前 worktree 是 dirty，且包含一次为 rollout audit 直接同步过去的脚本；远端不可作为代码事实源。下次远端执行前，应先由本地提交/推送成为事实源，再在远端按 owner 确认的方式 stash/archive remote-only changes 后 `git pull --ff-only`。
- 远端同步前的旧 dirty worktree 已 stash：`autodl-before-sft-v3-sync-20260615-011851`。
- `/tmp/LLM4Game24` 是临时外部 baseline clone，不属于本仓库。

## 清洁状态

- 构建/测试：GRPO scaffold 后 `python -m compileall src scripts` 通过；focused pytest 18 tests 通过；`ruff check src scripts tests` 通过；`ruff format --check src scripts tests configs` 通过；`./scripts/harness-check.sh` 通过，0 warnings；`pytest` 47 tests 通过。
- 进度状态：`M3-grpo-frontier` 是唯一 active feature；`M2-sft-audit-and-repair` 已 passing。
- 归档状态：旧 `0002-sft-training-readiness` 和 `20260615-sft-audit-and-repair` 计划已移到 `plans/archive/`。
- 临时工件：`data/processed/` 和 `outputs/` 是 ignored runtime artifacts，不应提交。
- 训练状态：AutoDL 当前无 running train/eval/audit command；最近一次 GPU 检查为
  `0%` utilization、`0 MiB / 49140 MiB` after balanced low-solution SFT eval
  completed.

## 下一步最佳动作

不要扩大 `beta0_none_25`、G8 10-step、`beta=0.002`、mixed pool、targeted SFT
refresh、当前 close-bonus/closure-strict profile 或 hard-pool second-stage continuation。
当前可展示主结果应优先报告 direct greedy `4096`，但必须带上 SFT 4096 同口径基线；
verifier-rerank 是更强但不同口径的 inference-time search-selection 结果。若继续研究 greedy 提升，下一步应改变
目标本身：更明确的停止/闭合策略、训练时 EOS/answer closure 建模，或把长 token/采样
成功轨迹蒸馏为更短 direct greedy 输出，而不是简单加 step/LR/beta。

Countdown handoff3 当前已经从 zero-shot `0%` 推进到 balanced SFT greedy
`23/100`、sampled `pass@8=45/100`。旧 `qwen_chat_target` 路线已判定不是
最小迁移；新 `qwen_chat_minimal_target` 路线下，target-replacement SFT 先修复
格式/闭合，balanced low-solution SFT 显著提高了低解数题表现。当前主要失败是
`wrong_value`，不是 answer closure。下一步可以从 balanced SFT final 开始做一轮
保守 GRPO probe；reward 应把 strict success 作为主信号，避免为了 target-distance
或长度 shaping 破坏已稳定的输出格式。

## 命令

查看结果：

```sh
ssh AutoDL4090
cd /root/autodl-tmp/projects/game24-rl
cat outputs/experiments/fixed_trace_v1_lora/eval/summary.json
cat outputs/experiments/fixed_trace_v1_lora_lr3e4/eval/summary.json
```

读取评估汇总：

```sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/experiments/fixed_trace_v1_lora/eval/summary.json")
print(path.read_text() if path.exists() else f"not ready: {path}")
path = Path("outputs/experiments/fixed_trace_v1_lora_lr3e4/eval/summary.json")
print(path.read_text() if path.exists() else f"not ready: {path}")
PY
```
