#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/bin/python}
RUN_ROOT=${RUN_ROOT:-outputs/official_nlile_strong_sft_20260618}
SHARD_COUNT=${SHARD_COUNT:-32}
DATA=${RUN_ROOT}/data/sft_full_nlile_original_rollback.jsonl
SUMMARY=${RUN_ROOT}/data/sft_full_nlile_original_rollback.summary.json
SHARD_DIR=${RUN_ROOT}/data/shards
LOG_DIR=${RUN_ROOT}/logs
RUN_DIR=${RUN_ROOT}/sft/SFT-full-nlile-single-5000

mkdir -p "$SHARD_DIR" "$LOG_DIR" "${RUN_ROOT}/markers" "${RUN_ROOT}/sft"

log() {
  printf "[%s] %s\n" "$(date "+%F %T")" "$*"
}

log "starting parallel rollback build with ${SHARD_COUNT} shards"
rm -f "$DATA" "$SUMMARY" "$SHARD_DIR"/*.jsonl "$SHARD_DIR"/*.summary.json \
  "$LOG_DIR"/parallel_shard_*.log 2>/dev/null || true

pids=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  out=$(printf "%s/shard_%02d.jsonl" "$SHARD_DIR" "$i")
  log_file=$(printf "%s/parallel_shard_%02d.log" "$LOG_DIR" "$i")
  "$PY" scripts/experiments/build_rollback_sft_shard.py \
    --manifest data/processed/splits/official-tot-overnight-v1.json \
    --split train_full_1362 \
    --output "$out" \
    --shard-index "$i" \
    --shard-count "$SHARD_COUNT" \
    --prompt-style qwen_chat \
    > "$log_file" 2>&1 &
  pids+=("$!")
done
printf "%s\n" "${pids[@]}" > "${RUN_ROOT}/parallel_build.pids"

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  log "one or more shards failed; not starting training"
  exit 1
fi

log "all shards complete; merging"
"$PY" - <<PY
import json
from pathlib import Path

shard_dir = Path("$SHARD_DIR")
out = Path("$DATA")
records = []
for path in sorted(shard_dir.glob("shard_*.jsonl")):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as file:
    for item in records:
        file.write(json.dumps(item, sort_keys=True) + "\\n")

line_counts = [len(str(record["completion"]).splitlines()) for record in records]
summary = {
    "build_mode": "parallel_contiguous_shards_same_rollback_logic",
    "line_count_max": max(line_counts) if line_counts else 0,
    "line_count_median": sorted(line_counts)[len(line_counts) // 2]
    if line_counts
    else 0,
    "line_count_min": min(line_counts) if line_counts else 0,
    "records": len(records),
    "rollback_records": sum(
        "roll back" in str(record["completion"]) for record in records
    ),
    "shard_count": int("$SHARD_COUNT"),
    "unique_puzzles": len({tuple(sorted(record["numbers"])) for record in records}),
}
Path("$SUMMARY").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
if summary["unique_puzzles"] != 1362:
    raise SystemExit(f"bad unique_puzzles={summary['unique_puzzles']}")
if summary["records"] < 35000:
    raise SystemExit(f"records unexpectedly low: {summary['records']}")
if summary["rollback_records"] < 30000:
    raise SystemExit(
        f"rollback_records unexpectedly low: {summary['rollback_records']}"
    )
PY

log "build validation passed; starting single train"
"$PY" scripts/experiments/run_rollback_sft_experiment.py \
  --mode train \
  --dataset "$DATA" \
  --run-dir "$RUN_DIR" \
  --training-mode full \
  --prompt-style qwen_chat \
  --max-steps 5000 \
  --save-steps 500 \
  --max-length 1024 \
  --max-new-tokens 1024 \
  --eval-batch-size 4 \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  > "${LOG_DIR}/train_single_5000.log" 2>&1
touch "${RUN_ROOT}/markers/train_single_5000.done"
log "single train done: ${RUN_DIR}/final"
