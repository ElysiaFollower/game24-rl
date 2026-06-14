<!--
职责：提供最新的紧凑交接信息，让新 agent 能无歧义恢复当前任务。
边界：只保留当前可恢复状态；历史放 progress.md，稳定事实放 docs 或代码。
-->

# 会话交接

## 仓库状态

- 分支：`main`
- 提交：尚无项目提交；当前是新仓库初始 scaffold 加本次 M1/M2 实现。
- Git remote：当前本地 `git remote -v` 为空；远程训练前需要 public repository URL 和已推送代码。
- 脏文件：整个初始仓库尚未提交；新增/修改覆盖 `src/`、`scripts/`、`tests/`、`configs/sft_v1.yaml`、`pyproject.toml`、`environment*.yml`、`docs/architecture/remote-training-operations.md`、`plans/`、`harness/` 等。
- Ignored 运行产物：`data/processed/`、`outputs/` 保留为本次 dry-run evidence；`.pytest_cache`、`.ruff_cache` 和 `__pycache__` 已清理。
- 当前计划：`plans/active/0002-sft-training-readiness.md`
- 当前功能项：`M2-first-pass-sft`

## 当前模式

- Collaboration mode：强人类把关，小步授权执行。
- Development principles：研究原型 + Python library/CLI，Google Python Style 轻量执行。
- 是否适合下一步：适合；M2 真实训练准备需要负责人监工环境、训练、评测和证据口径。
- 是否建议切换：暂不切换。后续每个训练放大、依赖固定、真实 SFT、GRPO 入口都先给负责人确认。

## 当前已验证状态

- `./scripts/harness-check.sh`：通过，0 警告。
- `python -m compileall src scripts`：通过。
- `pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py`：通过，25 tests。
- `pytest tests/test_training_pipeline.py`：通过，6 tests。
- `pytest`：通过，31 tests。
- `ruff check . && ruff format --check .`：通过。
- `python -m pip install --no-build-isolation -e '.[dev]'`：通过；确认 editable install、extras 和 console script metadata 可安装。
- RTXpro6000 上的 Miniconda 已安装完成，路径 `/home/runner/miniconda3`，`conda 26.3.2`，`python 3.13.13`。
- M1 CLI smoke：
  - `python scripts/make_splits.py --output /tmp/game24-standard-game24-v1.json`：输出 total=1820, solvable=1362, unsolvable=458, train=1089, validation=136, test=137；临时文件已删除。
  - `python scripts/build_sft_v1.py --manifest /tmp/game24-standard-game24-v1.json --output /tmp/game24-sft-v1-train.jsonl --traces-per-puzzle 2`：输出 2042 records；临时文件已删除。
- M2 dry-run：
  - `python scripts/train_sft.py --config configs/sft_v1.yaml --dry-run`：通过；写入 ignored artifacts 到 `data/processed/` 和 `outputs/sft_v1/qwen25_15b_lora_sft_v1/`，未加载模型权重。
  - `python scripts/eval_checkpoint.py --manifest data/processed/splits/standard-game24-v1.json --solver-dry-run --output-dir outputs/eval/sft_v1_dryrun --split validation --limit 16`：通过；solve_rate=1.000, format_rate=1.000, valid_expr_rate=1.000。
- Remote runner：
  - `rtxpro6000` 已重新注册为 `runner` 用户、key auth、default cwd `/home/runner/projects`。
  - `remote-runner machine doctor rtxpro6000 --json`：通过。
  - 短 session 确认 `whoami=runner` 和 `pwd=/home/runner/projects`；session 已销毁。
  - 临时 tarball 上传到 `/home/runner/projects/game24-rl-work` 后，system-package 路径下远程 `./init.sh`、compileall、`pytest` 31 tests 通过。
  - 临时 tarball smoke 树清理 macOS `._*` 后，远程 SFT dry-run 和 eval dry-run 通过。
  - 远程 `python3 -m venv` 失败：缺少 ensurepip/python3.12-venv。
  - 远程 pip 拉取 build dependencies 失败：PyPI/DNS name resolution error。
  - 远程 `ruff` 未安装；`scripts/remote_readiness.sh` 已更新为无 ruff 时跳过远程 lint。
  - 远程 Miniconda 安装通过本地 HTTP 镜像完成；直接访问 `repo.anaconda.com` 曾出现 DNS 失败。
  - `2026-06-14 16:30 CST` 只读核对：GPU utilization 0%，未发现 runner 用户下的 training/eval 进程；根分区约 25G 可用。
- AutoDL4090：
  - SSH 和 Remote Runner 均可达；用户 `root`，工作目录 `/root/autodl-tmp/projects`。
  - 已安装 `tmux 3.2a`、`git`、`curl`、`gh 2.4.0`；`gh auth status` 显示尚未登录。
  - 硬件为 NVIDIA GeForce RTX 4090，显存约 49GB；`/root/autodl-tmp` 约 750GB 可用。
  - 复用 `/root/miniconda3` base 环境；`torch 2.8.0+cu128` 可用，CUDA available true。
  - 已安装训练依赖：`transformers 5.12.0`、`peft 0.19.1`、`trl 1.6.0`、`datasets 5.0.0`、`accelerate 1.14.0`、`pytest 9.1.0`、`ruff 0.15.17`。
  - 临时 tarball 源码位于 `/root/autodl-tmp/projects/game24-rl-work`，仅用于环境验证，不作为可报告训练来源。
  - 远端验证通过：`python -m compileall src scripts`、`pytest` 31 tests、`ruff check .`、`ruff format --check .`、`python scripts/train_sft.py --config configs/sft_v1.yaml --dry-run`、`python scripts/eval_checkpoint.py --manifest data/processed/splits/standard-game24-v1.json --solver-dry-run --output-dir outputs/eval/sft_v1_dryrun --split validation --limit 16`。
  - TRL API probe：`SFTConfig` 存在，但当前 `trl 1.6.0` 的 `SFTConfig` 参数中没有 `max_seq_length`；真实 SFT 前需要最小训练/API smoke 或固定兼容依赖。

## 本会话改动

- 完成 M1 solver/verifier/data split/SFT data primitives 和测试。
- 归档 M1 active plan，设置 `M1-solver-verifier-foundation` 为 `passing`。
- 创建 M2 active plan 和 check anchors。
- 写入远程训练操作约束：`docs/architecture/remote-training-operations.md`。
- 实现 SFT training readiness：
  - Config dataclasses and YAML loader。
  - 自动生成 split manifest 和 SFT train JSONL。
  - Checkpoint explicit resume / latest auto-resume。
  - Run metadata、artifact dir、dry-run checkpoint marker。
  - 真训练路径 lazy import Transformers/PEFT/TRL/datasets。
- 实现 evaluation readiness：
  - Raw-output JSONL strict verifier scoring。
  - Report schema includes model/checkpoint/split/decoding/answer contract/verifier version/raw outputs。
  - Solver dry-run evaluation。
  - Optional real checkpoint generation path lazy imports model deps。
- 增加远程运行材料：
  - `scripts/remote_readiness.sh`：支持 `REMOTE_INSTALL_MODE=venv|system-user` 和 `REMOTE_SKIP_INSTALL=1`，适配远程无 `python`、缺 `python3-venv`、缺 ruff 的情况。
  - `docs/architecture/remote-sft-runbook.md`：记录 official public-clone path、临时 tarball path、无公网 package-index 时的处理。
- 补齐本地 packaging / environment：
  - `pyproject.toml`：base deps 保持轻量，新增 `dev` / `train` extras、setuptools build metadata 和 `game24-*` console scripts。
  - `src/game24_rl/cli.py`：集中 console script main functions；`scripts/*.py` 保持 thin wrapper。
  - `environment.yml` / `environment-train.yml`：Miniconda-first Python 3.11 env base。
  - `scripts/bootstrap_conda_env.sh`：支持 `dev` / `train` profile，一键创建或更新 conda env 后执行 editable install。
  - `README.md`：记录 bootstrap 命令和 installed CLI entrypoints。

## 本会话决策

- SFT 数据生成采用“每个 puzzle 最多 N 条唯一 solver trace”；有些 puzzle 少于 N 条，所以默认 8 条不是硬性等量保证。
- 本地 dry-run 不加载 `Qwen/Qwen2.5-1.5B-Instruct`，只验证数据、metadata、resume 和 evaluation artifact schema。
- 真实训练和真实 checkpoint evaluation 的重依赖导入推迟到对应函数内部，避免本地 M1/M2 单元测试依赖 CUDA 或模型下载。
- 当前协作模式改为强人类把关；训练、评测、依赖固定、仓库发布和 GRPO 入口都先给负责人确认。
- AutoDL4090 上因 Remote Runner session 状态偶发 busy 残留，允许使用负责人明确授权的 `ssh AutoDL4090` 做短命令和环境管理；真实长训仍应放在远端 tmux session 中，保留日志和 checkpoint。

## 仍损坏或未验证

- 尚未在 AutoDL4090 上通过 `gh auth login`；`gh` 已安装但未登录。
- 尚未从 public repo clone 正式代码；当前 AutoDL4090 的 `/root/autodl-tmp/projects/game24-rl-work` 是 tarball 临时副本，不能作为可报告训练来源。
- 当前本地仓库没有 git remote，远程机器无法从 public repo 拉到这次代码，直到仓库被推送或提供 public repo URL。
- AutoDL4090 训练依赖已装在 base conda 中，但真实 `SFTTrainer` 训练路径未跑；`trl 1.6.0` 可能与当前脚本中的 `max_seq_length` 参数不兼容。
- 尚未下载 `Qwen/Qwen2.5-1.5B-Instruct` 权重，尚未验证 Hugging Face 模型下载速度和缓存路径。
- 尚未做最小真实训练保存/resume/eval 闭环；不得直接开始长训。
- 尚未在 RTXpro6000 上 clone public repo、安装依赖或启动真实 LoRA SFT。
- RTXpro6000 当前 Python 环境不能直接创建 venv，pip 也不能访问 PyPI/build dependencies；真实训练前需要 prebuilt env、离线 wheels/cache、或修复网络/DNS。
- Miniconda 基座已经补上，但完整训练依赖还没装；train profile 需要再跑一次 bootstrap 或切到预置环境。
- RTXpro6000 根分区只剩约 25G 可用，继续安装完整训练依赖和模型缓存有较高空间风险。
- 远程缺 `ruff`；远程 readiness 现在会跳过 ruff，但本地 ruff 仍是完成门禁。
- 真实 TRL `SFTConfig`/`SFTTrainer` API 未在远程训练环境验证；本地只验证了 lazy-import 外围逻辑和 artifact schema。
- `scripts/bootstrap_conda_env.sh train` 尚未实际创建完整训练环境；本地已验证 `.[dev]` editable install，重依赖仍需在远程或训练机上验证。
- 训练产物、checkpoint、模型生成输出还没有真实分数；M2 仍为 `active`，不是 `passing`。
- `data/processed/` 和 `outputs/` 是 ignored dry-run evidence，不应提交。
- Remote-runner 残留：多个早期 session 仍显示 `busy=true`，但远端 `tmux ls` 未显示 AutoDL 活动 session，未发现训练进程。

## 清洁状态

- 构建/静态检查：通过；命令见“当前已验证状态”。
- 测试/端到端：本地单元、CLI smoke、dry-run eval 通过；真实模型训练未跑。
- 进度状态：`feature_list.json`、`progress.md`、active plan 和本 handoff 已同步；M1 已归档，M2 active。
- 归档状态：`plans/archive/0001-solver-verifier-foundation.*` 存在；当前只保留一个 active plan。
- 当前模式：强人类把关，适合 M2 真实训练准备和初始实验监工。
- 临时工件：Python/pytest/ruff 缓存已清理；ignored dry-run artifacts 保留在 `data/processed/`、`outputs/`；macOS `.DS_Store` 已清理；`.gitignore` 已补 `._*`。
- 启动路径：下一会话先运行 `./init.sh`，再读 `plans/active/0002-sft-training-readiness.md`。

## 下一步最佳动作

1. 负责人在 AutoDL4090 上执行 `gh auth login`，或提供可直接 clone 的 public repo URL。
2. 设置 public git remote 并推送当前代码；用正式 clone 替换 `/root/autodl-tmp/projects/game24-rl-work` 的临时 tarball 副本。
3. 在正式 clone 中复跑 `./init.sh`、compileall、pytest、ruff、SFT dry-run、eval dry-run。
4. 在负责人确认后，做最小真实模型/API smoke：加载 `Qwen/Qwen2.5-1.5B-Instruct` 和 tokenizer，确认 `SFTConfig/SFTTrainer` 参数兼容，跑极小训练步保存 checkpoint，不做长训。
5. smoke 通过后再由负责人确认是否启动完整 LoRA SFT；真实训练放入 tmux，使用 `--auto-resume`，保存日志和 checkpoint。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 本地验证：`python -m compileall src scripts`；`pytest`；`ruff check . && ruff format --check .`
- Miniconda dev env：`./scripts/bootstrap_conda_env.sh dev && conda activate game24-rl`
- Miniconda train env：`./scripts/bootstrap_conda_env.sh train && conda activate game24-rl`
- Split/SFT 数据：`python scripts/make_splits.py --output data/processed/splits/standard-game24-v1.json`；`python scripts/build_sft_v1.py --manifest data/processed/splits/standard-game24-v1.json --output data/processed/sft/game24-sft-v1-train.jsonl`
- SFT dry-run：`python scripts/train_sft.py --config configs/sft_v1.yaml --dry-run`
- Eval dry-run：`python scripts/eval_checkpoint.py --manifest data/processed/splits/standard-game24-v1.json --solver-dry-run --output-dir outputs/eval/sft_v1_dryrun --split validation --limit 16`
- Remote runner health：`conda run -n seedrunner remote-runner machine doctor rtxpro6000 --json`
- AutoDL SSH：`ssh AutoDL4090`
- AutoDL env：`source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`
- AutoDL临时副本：`cd /root/autodl-tmp/projects/game24-rl-work`
- Remote readiness fallback：`REMOTE_INSTALL_MODE=system-user REMOTE_SKIP_INSTALL=1 ./scripts/remote_readiness.sh`
