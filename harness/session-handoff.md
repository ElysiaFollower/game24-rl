<!--
职责：提供最新的紧凑交接信息，让新 agent 能无歧义恢复当前任务。
边界：只保留当前可恢复状态；历史放 progress.md，稳定事实放 docs 或代码。
-->

# 会话交接

## 仓库状态

- 分支：`main`
- 当前功能项：`M2-sft-audit-and-repair`
- 当前 active plan：`plans/active/20260615-sft-audit-and-repair.md`
- 当前本地状态：代码修复已提交并推送；SFT v3 已在 AutoDL tmux 中运行。
- 最新代码 commit：`372e17a`。

## 当前模式

- Collaboration mode：高自治执行。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 当前边界：不改主模型、split、answer contract、verifier 接受标准，不进入 GRPO。
- 本轮退出条件已满足：最终训练已在 AutoDL tmux 中跑起来，并观察到正常 step/checkpoint 日志。

## 当前已验证状态

- SFT v2 final eval：validation `42/136 = 30.88%`，format/valid expression 均为 `100%`。
- 审计报告：`docs/experiments/sft_audit_report.md`。
- Case study artifact：`outputs/diagnostics/sft_case_study/summary.json`。
- 修复：`trace_type` 和 `prompt_style` 现在从 config/CLI 贯通到数据生成、训练 dry-run 和 eval report。
- 新配置：`configs/sft_v3_checked_chat.yaml`，run name `qwen25_15b_lora_sft_v3_checked_chat`。
- 本地验证通过：`./scripts/harness-check.sh`；`python -m compileall src scripts`；focused pytest 38 tests；`pytest` 38 tests；`ruff check src scripts tests`；`ruff format --check src scripts tests configs`；v3 dry-run。
- 远端预启动验证：`python -m compileall src scripts`；`pytest tests/test_data_gen.py tests/test_training_pipeline.py` 17 tests；v3 dry-run；`11706/11706` completions strict-verifier valid。

## 当前任务

SFT v3 正在 AutoDL 上训练。下一次接手应先查看训练是否完成，然后评估 checkpoint。

## 当前判断

- 低分不是明显 eval 聚合 bug；主要失败是 correct-format/correct-numbers/wrong-value。
- 已确认的工程问题：`trace_type` 配置之前存在但未生效；eval 之前不支持/记录 prompt style。
- v3 修复仍是保守实验：同模型、同 split、同 verifier、同 `<answer>` 契约；只改 teacher prompt/trace 风格和 trace 数量。
- 如果 v3 仍低分，下一步应生成 repo-native rollback/search-trace 数据，而不是进入 GRPO。

## 仍损坏或未验证

- v3 准确率尚未知；训练完成后必须用 matching prompt style 评估。
- 本地 `scripts/audit_sft_dataset.py` 因本地缺少 `transformers` 未运行；已用 repo-local JSONL + strict verifier 做替代数据审计。
- AutoDL 直连 GitHub 不稳定；同步需要代理公式。
- 远端同步前的旧 dirty worktree 已 stash：`autodl-before-sft-v3-sync-20260615-011851`。
- `/tmp/LLM4Game24` 是临时外部 baseline clone，不属于本仓库。

## 清洁状态

- 构建/测试：本地和远端轻量门禁均通过，详见 `harness/progress.md` 和 `harness/feature_list.json` evidence。
- 进度状态：`M2-sft-audit-and-repair` 是唯一 active feature；`M2-first-pass-sft` 仍 blocked，等待 v3 结果。
- 归档状态：旧 `0002-sft-training-readiness` 计划已移到 `plans/archive/`。
- 临时工件：`data/processed/` 和 `outputs/` 是 ignored runtime artifacts，不应提交。
- 训练状态：AutoDL tmux session `game24-sft-v3-checked-chat` 正在运行。

## 下一步最佳动作

查看 AutoDL tmux session，等训练完成后评估 `checkpoint-500`、`checkpoint-1500` 和 `final`，并用 `--prompt-style qwen_chat` 保持训练/评测 prompt 一致。

## 命令

查看训练：

```sh
ssh AutoDL4090
cd /root/autodl-tmp/projects/game24-rl
tmux attach -t game24-sft-v3-checked-chat
```

查看日志：

```sh
tail -f outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat/logs/train-20260615-011937.log
```

评估示例：

```sh
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/transformers
export HF_HUB_ENABLE_HF_TRANSFER=0
python scripts/eval_checkpoint.py \
  --manifest data/processed/splits/standard-game24-v1.json \
  --split validation \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat/checkpoint-500 \
  --output-dir outputs/eval/sft_v3_checked_chat_ckpt500_validation \
  --prompt-style qwen_chat
```
