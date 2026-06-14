<!--
职责：定义本项目被新 agent 无歧义接手的初始化契约。
边界：不要记录业务实现进度；进度放 progress.md，具体任务放 plans/active/。
-->

# 初始化契约

## 自举条件

- 能启动：无常驻服务；从 Python package、scripts/ wrapper 和 pytest/ruff 命令进入开发。
- 能测试：`python -m compileall src scripts`；M1 实现后追加 `pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py`
- 能看进度：`harness/progress.md` 和 `harness/feature_list.json`
- 能接手下一步：`harness/session-handoff.md` 和 `plans/active/`

## 环境

- 技术栈：Python package + thin CLI scripts；Transformers、PEFT、TRL、PyTorch、datasets、pytest、ruff。
- 运行时版本：Python `>=3.10`；GPU 训练环境预计在 4090 级别 Linux 服务器上执行。
- 依赖安装：`python -m pip install -e ".[dev]"`；训练服务器按实际 CUDA/PyTorch 版本安装兼容 wheel。
- 本地服务：无。

## 标准命令

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
./scripts/harness-check.sh
python -m compileall src scripts
pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py
ruff check .
ruff format --check .
pytest
```

## 初始化验收清单

- [ ] 从干净 checkout 可安装依赖。
- [ ] 项目能启动或明确说明为什么不能启动。
- [ ] 至少一个可靠验证命令能运行。
- [ ] `./scripts/harness-check.sh` 通过。
- [ ] 新 agent 只看仓库能回答：是什么、怎么跑、怎么测、当前进度、下一步。

## 已知缺口

- 当前业务模块和测试文件是 scaffold，M1 尚未实现。
- 真实模型训练、数据生成和评估命令需要在 M1 基座完成后再跑。
- 4090 Linux 训练环境的 CUDA/PyTorch 版本尚未固化；不要在本地 scaffold 阶段下载大模型或启动长训练。
