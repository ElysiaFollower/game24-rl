<!--
职责：记录跨会话进度、状态变化、阻塞和验证摘要，让新会话能快速恢复。
边界：不要存放聊天记录、原始日志、密钥，或更适合由代码、测试、ADR、任务计划表达的内容。
-->

# 进度日志

## 当前状态

- 当前功能项：`M2-first-pass-sft`
- 当前任务计划：`plans/active/0002-sft-training-readiness.md`
- 上次验证：本地 `python -m pip install --no-build-isolation -e '.[dev]'`、`./scripts/harness-check.sh`、`python -m compileall src scripts`、`pytest`、`ruff check . && ruff format --check .` 均通过；远程 system-package 路径下 `./init.sh`、compileall、pytest 31 tests、SFT dry-run 和 eval dry-run 通过；Miniconda 已在 RTXpro6000 的 `/home/runner/miniconda3` 安装完成
- 下一步最佳动作：如果继续在 RTXpro6000 上推进，直接用 `./scripts/bootstrap_conda_env.sh train` 补齐训练依赖并从 public clone 启动可续跑 LoRA SFT；如果要减少 RTX 占用，转去 AutoDL 做长训

## 状态约定

- `not_started`：尚未开始。
- `active`：当前唯一在制任务。
- `blocked`：缺少输入、环境、依赖或决策。
- `passing`：验证通过且 evidence 已记录。

## 日志

只保留当前阶段和近期状态。阶段完成后，把旧日志摘要移入 `harness/archive/`，不要让本文件无限增长。

### 2026-06-14 - 协作模式切换为强人类把关

- 旧模式：高自治执行，关键决策升级。
- 新模式：强人类把关，小步授权执行。
- 原因：M2 真实训练准备和后续 GRPO 初始阶段涉及远程/AutoDL 资源、训练口径、评测证据和可报告结论，负责人希望亲自监工关键判断，减少无效训练和范围漂移。
- 保持不变：研究原型 + Python library/CLI 开发原则、主模型、SFT-first 路线、answer contract、split 策略和 strict verifier 红线。
- 下一步：等待负责人提供 AutoDL/远程机器信息；上机后先做环境、依赖、模型加载、最小训练保存和 strict eval 闭环 smoke，再由负责人确认是否放大到完整 SFT。

### 2026-06-14 - AutoDL4090 基础环境准备完成

- 机器：`AutoDL4090`，SSH/Remote Runner 均可达；用户 `root`，工作目录 `/root/autodl-tmp/projects`。
- 硬件：NVIDIA GeForce RTX 4090，约 49GB 显存；`/root/autodl-tmp` 约 750GB 可用。
- 已安装：`tmux 3.2a`、`git`、`curl`、`gh 2.4.0`；`gh auth status` 显示尚未登录，需要负责人执行 `gh auth login`。
- 训练环境：复用 AutoDL base conda `/root/miniconda3`；已有 `torch 2.8.0+cu128` 且 CUDA 可用，已安装 `transformers 5.12.0`、`peft 0.19.1`、`trl 1.6.0`、`datasets 5.0.0`、`accelerate 1.14.0`、`pytest 9.1.0`、`ruff 0.15.17`。
- 临时源码验证：由于本地仓库尚无 git remote，使用 tarball 上传到 `/root/autodl-tmp/projects/game24-rl-work`，只用于环境验证，不作为可报告训练来源。
- 远端验证：base conda 中 `python -m compileall src scripts`、`pytest` 31 tests、`ruff check .`、`ruff format --check .`、`train_sft.py --dry-run`、`eval_checkpoint.py --solver-dry-run --limit 16` 均通过。
- 风险：当前 TRL `SFTConfig` 不包含 `max_seq_length` 参数，真实 SFT 脚本按现状可能与 `trl 1.6.0` 不兼容；启动真实训练前必须做最小模型/API smoke 或调整脚本/固定依赖版本。
- 工具状态：Remote Runner 在 AutoDL 上有早期中断留下的 busy session 记录，但远端 `tmux ls` 无活动 session，未发现训练进程；后续可直接用 SSH 或新建 Remote Runner session。

### 2026-06-14 - M1 基座完成，进入 M2 readiness

- 实现 `solver.py`、`verifier.py`、`datasets.py`、`data_gen.py`：
  - 标准 multiset 枚举为 1,820。
  - Solver 分类为 1,362 solvable / 458 unsolvable。
  - Verifier 使用单一 `<answer>...</answer>` contract、AST allowlist 和 `Fraction`。
  - Split manifest 按 sorted multiset 隔离 train/validation/test，默认 1089/136/137。
- 添加 `scripts/make_splits.py`、`scripts/build_sft_v1.py` 可执行 CLI，并用测试锁定 M1 行为。
- M1 active plan 已归档到 `plans/archive/0001-solver-verifier-foundation.md`，`feature_list.json` 中 M1 标记为 `passing`。
- 创建 M2 active plan：`plans/active/0002-sft-training-readiness.md`。
- 将远程训练纪律写入 `docs/architecture/remote-training-operations.md`。
- 实现 M2 readiness：
  - `src/game24_rl/train_sft.py`：配置解析、输入生成、run metadata、checkpoint auto-resume/explicit resume、dry-run、真实 LoRA SFT lazy import。
  - `scripts/train_sft.py`：SFT CLI，支持 `--dry-run`、`--auto-resume`、`--resume-from-checkpoint`。
  - `src/game24_rl/evaluate.py` / `scripts/eval_checkpoint.py`：raw-output strict verifier 评估、solver dry-run、真实 checkpoint 输出生成 lazy import。
  - `configs/sft_v1.yaml`：补齐 output/run、batch、save/logging、seed、manifest 和 train JSONL 路径。
  - `tests/test_training_pipeline.py`：覆盖 config、resume、dry-run artifact、evaluation report schema。
- 本地 dry-run：
  - `python scripts/train_sft.py --config configs/sft_v1.yaml --dry-run` 通过，写入 ignored artifacts：`data/processed/` 和 `outputs/sft_v1/qwen25_15b_lora_sft_v1/`。
  - `python scripts/eval_checkpoint.py --manifest data/processed/splits/standard-game24-v1.json --solver-dry-run --output-dir outputs/eval/sft_v1_dryrun --split validation --limit 16` 通过，solve/format/valid_expr 均为 1.000。
- `remote-runner` 修复：
  - 重新注册 `rtxpro6000` 为 `runner` 用户、key auth、default cwd `/home/runner/projects`。
  - `remote-runner machine doctor rtxpro6000 --json` 通过。
  - 短 session 确认 `whoami=runner`、`pwd=/home/runner/projects`、目录基本为空；session 已销毁。
- 远程环境探测：
  - 在无 public remote 的情况下，用 tarball 临时上传当前工作树到 `/home/runner/projects/game24-rl-work`，只用于 dry-run/环境验证，不作为正式训练来源。
  - GPU 为 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，显存约 97,887 MiB；根分区剩余约 108G。
  - 远程只有 `python3`，没有 `python`；`python3 -m venv` 因缺少 ensurepip/python3.12-venv 失败。
  - 远程 pip 安装 build dependency 时无法解析 PyPI 域名，真实训练依赖需要预置环境、离线 wheel/cache 或网络修复。
  - 远程 system-package + `REMOTE_SKIP_INSTALL=1` 路径下，`./init.sh`、`python -m compileall src scripts`、`pytest` 31 tests 通过；`ruff` 缺失，readiness 脚本已改为远程没有 ruff 时跳过 lint。
  - 远程 Miniconda 安装完成于 `2026-06-14`，路径 `/home/runner/miniconda3`，`conda 26.3.2`，`python 3.13.13`。
  - 一个早期失败的 remote-runner session 残留为 busy 状态但无运行命令；后续新 session 已正常销毁。
- 补齐本地 packaging 和 Miniconda-first 环境入口：
  - `pyproject.toml`：拆分 base/dev/train extras，添加 `game24-*` console scripts 和 setuptools build metadata。
  - `environment.yml` / `environment-train.yml`：固定 conda env 名称和 Python 3.11 基座。
  - `scripts/bootstrap_conda_env.sh`：一键创建/更新 `game24-rl` conda env，并按 `dev` 或 `train` profile 做 editable install。
  - `README.md`：记录 Miniconda bootstrap 和 installed CLI entrypoints。
  - 本地 `python -m pip install --no-build-isolation -e '.[dev]'` 已通过。

### 2026-06-13 - Harness 初始化与交接准备

- 创建初始 repo scaffold、研究文档目录、ADR、harness scaffold 和 M1 active plan。
- 将前期 research report 放入 `docs/research/` 作为长期参考资料。
- 固化核心方向：高分优先；Qwen2.5-1.5B-Instruct；SFT warm start 后 GRPO；标准 24 点 first；`<answer>...</answer>`；AST + `Fraction` verifier；multiset-isolated split。
- 验证：
  - `./scripts/harness-check.sh`：通过，0 警告。
  - `python -m compileall src scripts`：通过。
  - `./init.sh`：通过，冷启动入口可用。
