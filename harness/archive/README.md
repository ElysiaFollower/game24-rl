<!--
职责：保存已结束阶段的 harness 状态快照，避免当前状态文件无限膨胀。
边界：不要存放当前 active plan、当前 feature list、当前 handoff、原始日志或密钥。
-->

# Harness Archive

阶段完成、任务归档或长期工作流收束时，把已经结束且不再需要日常读取的状态移到这里。

推荐归档内容：

- 已完成阶段的 `feature_list` 快照。
- 已完成阶段的 `progress` 摘要。
- 已完成或废弃的 session handoff 快照。
- 重要验证 evidence 的路径索引。

命名建议：

- `YYYYMMDD-<phase-slug>/feature_list.json`
- `YYYYMMDD-<phase-slug>/progress.md`
- `YYYYMMDD-<phase-slug>/session-handoff.md`

归档后，`harness/feature_list.json` 和 `harness/progress.md` 只保留当前阶段、近期状态和下一步需要读取的信息。
