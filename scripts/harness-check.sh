#!/usr/bin/env sh
# 职责：用确定、便宜的检查验证仓库 harness 合同。
# 边界：不要替代项目测试、执行网络 setup、安装依赖，或修改源码文件。

set -eu

failures=0
warnings=0

fail() {
  failures=$((failures + 1))
  printf '%s\n' "失败：$*"
}

warn() {
  warnings=$((warnings + 1))
  printf '%s\n' "警告：$*"
}

require_file() {
  if [ ! -f "$1" ]; then
    fail "缺少必要文件：$1。请运行 harness scaffold 或补齐该 harness 工件。"
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    fail "缺少必要目录：$1。请创建目录并放置对应任务或文档。"
  fi
}

require_file "AGENTS.md"
require_file "init.sh"
require_file "docs/overview.md"
require_file "harness/bootstrap-contract.md"
require_file "harness/current-mode.md"
require_file "harness/feature_list.json"
require_file "harness/progress.md"
require_file "harness/decisions.md"
require_file "harness/session-handoff.md"
require_file "harness/observability.md"
require_file "harness/evaluator-rubric.md"
require_file "harness/quality.md"
require_dir "harness/archive"
require_dir "plans/active"
require_dir "plans/archive"
require_dir "docs/architecture"

if [ -f "AGENTS.md" ]; then
  lines=$(wc -l < "AGENTS.md" | tr -d ' ')
  if [ "$lines" -gt 200 ]; then
    fail "AGENTS.md 有 $lines 行；入口文件应是路由器，请把细节移到 docs、harness、测试或脚本"
  elif [ "$lines" -gt 150 ]; then
    warn "AGENTS.md 有 $lines 行；建议保持 50-150 行"
  elif [ "$lines" -lt 40 ]; then
    warn "AGENTS.md 只有 $lines 行；请确认包含事实来源、启动流程、硬规则、验证阶梯和完成定义"
  fi
  hard_rules=$(awk '
    /^## 硬性规则/ { in_rules = 1; next }
    /^## / && in_rules { in_rules = 0 }
    in_rules && /^[0-9][0-9]*\./ { count++ }
    END { print count + 0 }
  ' "AGENTS.md")
  if [ "$hard_rules" -gt 15 ]; then
    fail "AGENTS.md 中硬性规则有 $hard_rules 条；请压缩到 15 条以内，并把细节路由到专题文档"
  fi
fi

placeholder_files=$(grep -R -l "{{[A-Z0-9_][A-Z0-9_]*}}" AGENTS.md init.sh docs harness 2>/dev/null || true)
if [ -n "$placeholder_files" ]; then
  warn "仍存在未替换占位符；完成初始化前应替换为项目事实，或明确记录为已知缺口。涉及文件：$(printf '%s' "$placeholder_files" | tr '\n' ' ')"
fi

if command -v python3 >/dev/null 2>&1 && [ -f "harness/feature_list.json" ]; then
  python3 - <<'PY' || failures=$((failures + 1))
import json
import pathlib
import sys

path = pathlib.Path("harness/feature_list.json")
allowed = {"not_started", "active", "blocked", "passing"}
required = {"id", "priority", "area", "title", "behavior", "status", "verification", "evidence", "notes"}

try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"失败：{path} 不是合法 JSON：{exc}")
    sys.exit(1)

features = data.get("features")
if not isinstance(features, list):
    print("失败：feature_list.json 必须包含 features 数组")
    sys.exit(1)

active = 0
bad = False
for index, feature in enumerate(features):
    if not isinstance(feature, dict):
        print(f"失败：feature {index} 必须是 object")
        bad = True
        continue
    missing = sorted(required - set(feature))
    if missing:
        print(f"失败：feature {feature.get('id', index)} 缺少字段：{', '.join(missing)}")
        bad = True
    status = feature.get("status")
    if status not in allowed:
        print(f"失败：feature {feature.get('id', index)} 有非法 status：{status!r}，允许值是 {sorted(allowed)}")
        bad = True
    if status == "active":
        active += 1
    verification = feature.get("verification")
    if not verification:
        print(f"失败：feature {feature.get('id', index)} 缺少 verification；每个功能必须有验证命令或手动检查")
        bad = True
    if status == "passing" and not feature.get("evidence"):
        print(f"失败：feature {feature.get('id', index)} 是 passing 但没有 evidence；不能只凭 agent 自信标完成")
        bad = True

if active > 1:
    print(f"失败：feature_list.json 有 {active} 个 active；WIP limit 是 1")
    bad = True

sys.exit(1 if bad else 0)
PY
else
  warn "python3 不可用；跳过 JSON 验证"
fi

if [ -f "harness/session-handoff.md" ]; then
  for heading in "## 仓库状态" "## 当前模式" "## 当前已验证状态" "## 仍损坏或未验证" "## 清洁状态" "## 下一步最佳动作" "## 命令"; do
    if ! grep -q "$heading" "harness/session-handoff.md"; then
      fail "session-handoff.md 缺少标题：$heading。交接必须覆盖状态、证据、风险、清洁状态和下一步。"
    fi
  done
fi

if [ -f "harness/current-mode.md" ]; then
  if ! grep -q "mode-schema: current-harness-mode/v1" "harness/current-mode.md"; then
    fail "current-mode.md 缺少 schema 锚点：mode-schema: current-harness-mode/v1。当前模式文件必须使用统一结构。"
  fi
  if ! grep -q "## Collaboration Mode" "harness/current-mode.md"; then
    fail "current-mode.md 缺少 Collaboration Mode 部分。"
  fi
  if ! grep -q "## Development Style Principles" "harness/current-mode.md"; then
    fail "current-mode.md 缺少 Development Style Principles 部分。"
  fi
fi

if [ -d "plans/active" ]; then
  active_plans=$(find "plans/active" -type f -name "*.md" | wc -l | tr -d ' ')
  if [ "$active_plans" -gt 1 ]; then
    fail "plans/active 中有 $active_plans 个 active plan；默认 WIP=1，请归档过期 active plan"
  fi
fi

if command -v python3 >/dev/null 2>&1 && [ -d "plans/active" ]; then
  python3 - <<'PY' || failures=$((failures + 1))
import json
import pathlib
import re
import sys

root = pathlib.Path(".")
plans = sorted(root.glob("plans/active/*.md"))
if not plans:
    sys.exit(0)

if len(plans) > 1:
    print("失败：存在多个 active plan，跳过动态任务锚点检查。")
    sys.exit(1)

plan = plans[0]
check_path = plan.with_suffix(".check.json")
if not check_path.exists():
    print(f"警告：active plan {plan} 没有配套任务锚点文件 {check_path}。")
    print("建议：任务初始化时写入 3-7 个目标级锚点，用于完成前检查关键产出是否落地。")
    sys.exit(0)

try:
    data = json.loads(check_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"失败：任务锚点文件 {check_path} 不是合法 JSON：{exc}")
    sys.exit(1)

anchors = data.get("anchors")
if not isinstance(anchors, list):
    print(f"失败：任务锚点文件 {check_path} 必须包含 anchors 数组。")
    sys.exit(1)

if len(anchors) > 7:
    print(f"警告：任务锚点有 {len(anchors)} 个；建议保留 3-7 个真正反映目标推进的锚点。")
elif len(anchors) < 3:
    print(f"警告：任务锚点只有 {len(anchors)} 个；确认是否足够覆盖关键产出。")

feature_data = None
feature_path = root / "harness/feature_list.json"
if feature_path.exists():
    try:
        feature_data = json.loads(feature_path.read_text(encoding="utf-8"))
    except Exception:
        feature_data = None

def text_matches(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text, re.MULTILINE) is not None
    except re.error:
        return pattern in text

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def anchor_problem(anchor: dict, message: str) -> bool:
    severity = anchor.get("severity", "warn")
    label = "失败" if severity == "fail" else "警告"
    anchor_id = anchor.get("id", "<missing-id>")
    print(f"{label}：任务锚点 {anchor_id} 未满足：{message}")
    if anchor.get("reason"):
        print(f"原因：{anchor['reason']}")
    suggestion = anchor.get("suggest") or f"回看 {plan} 的关键锚点、验收标准和完成定义；确认是遗漏、实现变更，还是需要更新任务合同。"
    print(f"建议：{suggestion}")
    return severity == "fail"

def required_value(anchor, key):
    value = anchor.get(key)
    if value in (None, ""):
        anchor_problem(anchor, f"锚点定义缺少必要字段：{key}")
        return None
    return str(value)

bad = False
allowed = {"path_exists", "path_contains", "glob_contains", "evidence_contains", "artifact_contains"}
for index, anchor in enumerate(anchors):
    if not isinstance(anchor, dict):
        print(f"失败：任务锚点 {index} 必须是 object。")
        bad = True
        continue
    check = anchor.get("check")
    if check not in allowed:
        bad |= anchor_problem(anchor, f"未知检查类型 {check!r}，允许值是 {sorted(allowed)}。")
        continue

    if check == "path_exists":
        path_value = required_value(anchor, "path")
        if path_value is None:
            bad = True
            continue
        path = root / path_value
        if not path.exists():
            bad |= anchor_problem(anchor, f"期望路径存在，但未找到：{path}")
    elif check in {"path_contains", "artifact_contains"}:
        path_value = required_value(anchor, "path")
        pattern = required_value(anchor, "pattern")
        if path_value is None or pattern is None:
            bad = True
            continue
        path = root / path_value
        if not path.exists():
            bad |= anchor_problem(anchor, f"期望文件存在并包含目标内容，但未找到：{path}")
        elif not text_matches(read_text(path), pattern):
            bad |= anchor_problem(anchor, f"文件 {path} 未匹配期望内容：{pattern!r}")
    elif check == "glob_contains":
        glob_pattern = required_value(anchor, "glob")
        pattern = required_value(anchor, "pattern")
        if glob_pattern is None or pattern is None:
            bad = True
            continue
        matches = [p for p in root.glob(glob_pattern) if p.is_file()]
        if not matches:
            bad |= anchor_problem(anchor, f"glob 没有匹配任何文件：{glob_pattern}")
        elif not any(text_matches(read_text(path), pattern) for path in matches):
            bad |= anchor_problem(anchor, f"glob {glob_pattern} 匹配的文件中没有内容匹配：{pattern!r}")
    elif check == "evidence_contains":
        pattern = required_value(anchor, "pattern")
        if pattern is None:
            bad = True
            continue
        feature_id = anchor.get("feature_id")
        evidence_items = []
        if feature_data and isinstance(feature_data.get("features"), list):
            for feature in feature_data["features"]:
                if feature_id and feature.get("id") != feature_id:
                    continue
                evidence_items.extend(feature.get("evidence") or [])
        evidence_text = json.dumps(evidence_items, ensure_ascii=False)
        if not text_matches(evidence_text, pattern):
            target = f"feature {feature_id}" if feature_id else "feature_list evidence"
            bad |= anchor_problem(anchor, f"{target} 未匹配 evidence 内容：{pattern!r}")

sys.exit(1 if bad else 0)
PY
else
  warn "python3 不可用；跳过动态任务锚点检查"
fi

if [ "$failures" -gt 0 ]; then
  printf '%s\n' "Harness 检查失败，共 $failures 个问题、$warnings 个警告。"
  exit 1
fi

printf '%s\n' "Harness 检查通过，共 $warnings 个警告。"
