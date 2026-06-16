<!--
职责：提供最新的紧凑交接信息，让新 agent 能无歧义恢复当前任务。
边界：只保留当前可恢复状态；历史放 progress.md，稳定事实放 docs 或代码。
-->

# 会话交接

## 仓库状态

- 分支：`main`；本次备案提交后将切到
  `dev/baseline-accuracy-improvement` 继续对照 baseline 提分。
- 当前功能项：`M2-sft-audit-and-repair`
- 当前 active plan：`plans/active/20260615-sft-audit-and-repair.md`
- 当前本地状态：baseline-format v2 full finetune 的 `37.50%` 备案结果已记录到实验报告、进度和 feature evidence。
- 最新已推送代码 commit：本文件可能滞后于当前本地提交；以 `git log -1 --oneline` 为准。

## 当前模式

- Collaboration mode：负责人正在对实验路线做强把关；当前只推进已确认的 fixed-trace 短实验。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 当前边界：不改主模型、split、answer contract、verifier 接受标准，不进入 GRPO。
- 当前实验退出条件：fixed-trace 短训练与高 LR probe 都已完成自动评估；结果可直接读取。

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

## 当前任务

Full fine-tuning search-trace SFT 已形成当前强结果 `110/136 = 80.88%`。
当前任务是把配置、结果和必要脚本修复提交并 PR 到 `main`。

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

## 仍损坏或未验证

- 当前本地有未提交改动：`harness/decisions.md`、`harness/progress.md`、`harness/session-handoff.md`、`docs/experiments/sft_audit_report.md`、`scripts/experiments/`。
- 本地 `scripts/audit_sft_dataset.py` 因本地缺少 `transformers` 未运行；已用 repo-local JSONL + strict verifier 做替代数据审计。
- AutoDL 直连 GitHub 不稳定；同步需要代理公式。
- 远端同步前的旧 dirty worktree 已 stash：`autodl-before-sft-v3-sync-20260615-011851`。
- `/tmp/LLM4Game24` 是临时外部 baseline clone，不属于本仓库。

## 清洁状态

- 构建/测试：本地和远端轻量门禁均通过，详见 `harness/progress.md` 和 `harness/feature_list.json` evidence。
- 进度状态：`M2-sft-audit-and-repair` 是唯一 active feature；`M2-first-pass-sft` 仍 blocked，等待 fixed-trace 结果。
- 归档状态：旧 `0002-sft-training-readiness` 计划已移到 `plans/archive/`。
- 临时工件：`data/processed/` 和 `outputs/` 是 ignored runtime artifacts，不应提交。
- 训练状态：AutoDL 当前无 tmux 训练 session；GPU 空闲。主 run log 是 `outputs/experiments/fixed_trace_v1_lora/logs/run-20260615-fixed-trace-v1.log`，higher-LR logs 在 `outputs/experiments/fixed_trace_v1_lora_lr3e4/logs/`。

## 下一步最佳动作

把 `dev/baseline-accuracy-improvement` 的记录和脚本修复 PR 到 `main`。
合并后，下一步可以基于 `80.88%` SFT checkpoint 讨论是否进入 GRPO 或补充
test split 复评。

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
