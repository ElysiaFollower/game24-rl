<!--
职责：总结仓库 harness 的健康状态和下一步维护动作。
边界：不要存放完整审计日志、任务历史或项目架构细节。
-->

# Harness 质量

## 快照

- 上次审查：2026-06-14
- 审查者：Codex
- 总体状态：M1 passing, M2 training readiness active

## 健康信号

- `AGENTS.md` 长度：短入口，保持路由角色
- WIP limit：1
- 功能清单有效性：已建立，M1 passing，当前 active 为 M2
- 当前状态文件紧凑度：可用
- 阶段归档健康度：M1 active plan 已归档到 `plans/archive/`
- 交接新鲜度：当前会话已写入
- 验证命令健康度：harness、compileall、pytest、ruff 均通过
- 冷启动测试：`./init.sh` 可作为入口
- 端到端覆盖：本地 split/SFT/eval dry-run 已覆盖；远程 system-package pytest 已覆盖；真实模型训练尚未覆盖
- 重复失败是否已执行化：verifier/split/resume/eval artifact 已有 regression tests

## 维护队列

- 远程训练前确认 public repo URL、推送状态，并解决 RTXpro6000 缺 python3-venv / pip 无法访问 PyPI / 缺训练栈的问题。
- 远端机器只作为 execution worker：代码唯一事实源是本地仓库/GitHub 分支，远端只通过 `git clone`/`git pull --ff-only` 同步，训练日志、checkpoint 和中间数据只写 ignored artifact 路径。
- 首次远程真实 SFT 前先跑 dry-run 和 small limit eval，避免直接长训暴露低级 pipeline bug。
- M3 开始前补 GRPO reward 日志 schema 和 reward hacking 检查。
- 阶段完成时归档已结束状态，精简 `harness/feature_list.json` 和 `progress.md`。
- 把重复的 review failures 转成测试、lint 规则、schema 或脚本。
