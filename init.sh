#!/usr/bin/env sh
# 职责：初始化本地项目 harness，并运行最便宜且可靠的 sanity checks。
# 边界：不要安装全局工具、写入密钥、启动长运行服务，或意外修改项目源码。

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$repo_root"

echo "项目：Game24 RL - Qwen2.5-1.5B-Instruct 的 24 点 SFT/GRPO 实验"
echo "技术栈：Python 3.10+，Transformers，PEFT，TRL，PyTorch，pytest，ruff"

if [ -x "./scripts/harness-check.sh" ]; then
  ./scripts/harness-check.sh
else
  echo "缺少可执行文件 scripts/harness-check.sh"
fi

cat <<'EOF'

启动命令：
无常驻服务。开发入口是 Python package、scripts/ 下的 CLI wrapper，以及 pytest/ruff 验证命令。

聚焦验证：
python -m compileall src scripts
pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py

完整验证：
ruff check .
ruff format --check .
pytest
EOF
