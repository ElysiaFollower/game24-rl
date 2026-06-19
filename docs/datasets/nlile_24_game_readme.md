# nlile/24-game 数据说明

## 数据来源

- Hugging Face dataset: `nlile/24-game`
- 下载位置（AutoDL）：
  `/root/autodl-tmp/projects/game24-rl/data/raw/hf/nlile__24-game/default__train.jsonl`
- 下载方式：HF 镜像站 `https://hf-mirror.com`
- 本项目只把原始数据保存在训练机本地，仓库不提交 `data/raw/`。

## 本项目实测结论

2026-06-18 在 AutoDL 上用本项目精确 solver 审计 `nlile/24-game` 当前下载版本：

| 项目 | 数值 |
| --- | ---: |
| 总行数 | 1362 |
| HF `solvable=True` | 1362 |
| 本项目 solver 判定可解 | 1362 |
| HF 字段与 solver 不一致 | 0 |
| 解析失败 | 0 |
| 去重后 puzzle 数 | 1362 |
| 重复 puzzle | 0 |

审计 artifact：

```text
/root/autodl-tmp/projects/game24-rl/outputs/audits/nlile_24_game_solver_audit.json
```

## 口径约定

`nlile/24-game` 当前下载版本应视为 **1362 条全部可解的 24 点题目**。如果外部文档声称该数据集中存在 `solvable=False` 或只有 `1262` 条可解题，应以本项目对实际下载数据的审计结果为准；该说法与当前 HF 数据不一致。

后续训练若使用 `nlile/24-game` 的 solvable 题目，默认就是使用全量 `1362` 条。
