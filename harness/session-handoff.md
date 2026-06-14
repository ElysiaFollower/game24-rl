<!--
职责：提供最新的紧凑交接信息，让新 agent 能无歧义恢复当前任务。
边界：只保留当前可恢复状态；历史放 progress.md，稳定事实放 docs 或代码。
-->

# 会话交接

## 仓库状态

- 分支：`main`
- 当前功能项：`M2-sft-audit-and-repair`
- 当前 active plan：`plans/active/20260615-sft-audit-and-repair.md`
- 当前本地状态：SFT 审计报告、trace/prompt 修复、v3 配置和测试已完成，待提交/同步/启动训练。
- 不提交 ignored runtime artifacts：`data/processed/`、`outputs/`、模型缓存、raw logs、checkpoints。

## 当前模式

- Collaboration mode：高自治执行。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 当前边界：不改主模型、split、answer contract、verifier 接受标准，不进入 GRPO。
- 退出条件：最终训练在 AutoDL tmux 中跑起来并观测到正常日志后，可以结束会话。

## 当前已验证状态

- SFT v2 final eval 已确认：validation `42/136 = 30.88%`，format/valid expression 均为 `100%`。
- 审计报告：`docs/experiments/sft_audit_report.md`。
- Case study artifact：`outputs/diagnostics/sft_case_study/summary.json`。
- 修复：`trace_type` 和 `prompt_style` 现在从 config/CLI 贯通到数据生成、训练 dry-run 和 eval report。
- 新配置：`configs/sft_v3_checked_chat.yaml`，run name `qwen25_15b_lora_sft_v3_checked_chat`。
- 本地验证通过：`./scripts/harness-check.sh`；`python -m compileall src scripts`；focused pytest 38 tests；`pytest` 38 tests；`ruff check src scripts tests`；`ruff format --check src scripts tests configs`；v3 dry-run。

## 当前任务

1. 提交当前修复并推送到 GitHub。
2. 用本地代理反向转发同步 AutoDL。
3. 在 AutoDL tmux 启动 `python scripts/train_sft.py --config configs/sft_v3_checked_chat.yaml`。
4. 观察日志确认训练前几步正常运行后结束会话。

## 当前判断

- 低分不是明显 eval 聚合 bug；主要失败是 correct-format/correct-numbers/wrong-value。
- 已确认的工程问题：`trace_type` 配置之前存在但未生效；eval 之前不支持/记录 prompt style。
- v3 修复仍是保守实验：同模型、同 split、同 verifier、同 `<answer>` 契约；只改 teacher prompt/trace 风格和 trace 数量。
- 如果 v3 仍低分，下一步应生成 repo-native rollback/search-trace 数据，而不是进入 GRPO。

## 仍损坏或未验证

- v3 真实训练尚未启动；准确率尚未知。
- 本地 `scripts/audit_sft_dataset.py` 因本地缺少 `transformers` 未运行；已用 repo-local JSONL + strict verifier 做替代数据审计。
- AutoDL 直连 GitHub 不稳定；同步需要代理公式。
- `/tmp/LLM4Game24` 是临时外部 baseline clone，不属于本仓库。

## 清洁状态

- 构建/测试：本地门禁已通过，详见 `harness/progress.md` 和 `harness/feature_list.json` evidence。
- 进度状态：`M2-sft-audit-and-repair` 是唯一 active feature；`M2-first-pass-sft` 仍 blocked。
- 归档状态：旧 `0002-sft-training-readiness` 计划已移到 `plans/archive/`。
- 临时工件：`data/processed/` 和 `outputs/` 是 ignored runtime artifacts，不应提交。

## 下一步最佳动作

同步到 AutoDL 并启动 v3 训练。训练启动前确认远端没有旧 tmux 训练占用；启动后至少观察到 dataset/model load 完成和前几步 trainer log。

## 命令

本地代理转发：

```sh
ssh -fN -R 127.0.0.1:2080:127.0.0.1:2080 AutoDL4090
```

远端同步：

```sh
cd /root/autodl-tmp/projects/game24-rl
export https_proxy=http://127.0.0.1:2080
export http_proxy=http://127.0.0.1:2080
git pull --ff-only
```

远端训练：

```sh
cd /root/autodl-tmp/projects/game24-rl
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/transformers
export HF_HUB_ENABLE_HF_TRANSFER=0
python scripts/train_sft.py --config configs/sft_v3_checked_chat.yaml
```
