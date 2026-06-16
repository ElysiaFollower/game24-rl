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
- 当前实验退出条件：设计已落地；下一步应实现 dry-run/compatibility probe，真实长 GRPO 前先 short pilot。

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

## 仍损坏或未验证

- 当前本地有未提交改动：harness/progress.md 和本 handoff 更新，用于记录
  short pilot 结果。
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
- 训练状态：AutoDL 当前无 running train/eval command；最近一次 GPU 检查为
  `0%` utilization、`0 MiB / 49140 MiB`。

## 下一步最佳动作

不要扩大 `beta0_none_25`。下一步应先诊断 destructive update：确认是否需要
更低 learning rate / 更少 steps / `beta=0.001` reference KL / `scale_rewards=group`
A/B / 更小且更稳定的 active subset；并考虑先跑 sampled-success targeted SFT 作为
低成本 baseline。任何新 GRPO 训练都应先用 <=5 step probe，并立即评估 retention。

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
