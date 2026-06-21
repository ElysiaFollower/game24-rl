"""Build one shard of rollback/search-trace SFT records.

This is a parallel build helper for the historical
``run_rollback_sft_experiment.py`` data recipe. It intentionally keeps the same
record construction logic and only changes execution topology: a contiguous
slice of puzzles is processed by each shard and merged later.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from game24_rl.data_gen import format_prompt  # noqa: E402
from game24_rl.datasets import puzzle_id, read_manifest  # noqa: E402
from game24_rl.verifier import verify_answer  # noqa: E402


def _load_rollback_module() -> Any:
    script = Path(__file__).with_name("run_rollback_sft_experiment.py")
    spec = importlib.util.spec_from_file_location("rollback_sft_experiment", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--samples-per-puzzle", type=int, default=3)
    parser.add_argument("--max-lines", default="6,7,8,9,10,11,12,13,14,15,16,17")
    parser.add_argument("--prompt-style", default="qwen_chat")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")

    rollback = _load_rollback_module()
    manifest = read_manifest(args.manifest)
    split_records = list(manifest["splits"][args.split])
    start = len(split_records) * args.shard_index // args.shard_count
    end = len(split_records) * (args.shard_index + 1) // args.shard_count
    shard_records = split_records[start:end]
    rng = random.Random(args.seed + args.shard_index * 1_000_003)
    max_lines_values = [int(value) for value in args.max_lines.split(",") if value]

    records = []
    seen: set[tuple[str, str]] = set()
    for local_index, record in enumerate(shard_records, start=1):
        global_index = start + local_index
        base_numbers = list(record["numbers"])
        print(
            json.dumps(
                {
                    "build_event": "puzzle_start",
                    "global_index": global_index,
                    "local_index": local_index,
                    "numbers": base_numbers,
                    "records": len(records),
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "total_puzzles": len(split_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for sample_index in range(args.samples_per_puzzle):
            prompt_numbers = list(base_numbers)
            rng.shuffle(prompt_numbers)
            logs = rollback.solve_with_search_logs(prompt_numbers, rng)
            if logs is None:
                raise RuntimeError(f"solver failed for {prompt_numbers}")
            expression = rollback.extract_final_expression(logs)
            for max_lines in max_lines_values:
                compressed = rollback.compress_search_logs(
                    logs,
                    max_lines=max_lines,
                    rng=rng,
                )
                trace_lines = [
                    line for line in compressed if not rollback._is_input_line(line)
                ]
                if not trace_lines or "reach 24! expression:" not in trace_lines[-1]:
                    continue
                completion = (
                    "<think>\n"
                    + "\n".join(trace_lines)
                    + f"\n</think>\n<answer>{expression}</answer>"
                )
                verification = verify_answer(completion, prompt_numbers)
                if not verification.valid:
                    raise RuntimeError(
                        "invalid generated record "
                        f"{prompt_numbers}: {verification.reason}"
                    )
                key = (" ".join(str(number) for number in prompt_numbers), completion)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "id": (
                            f"rollback-{puzzle_id(base_numbers)}-"
                            f"{sample_index:02d}-{max_lines:02d}"
                        ),
                        "numbers": prompt_numbers,
                        "target": 24,
                        "prompt": format_prompt(
                            prompt_numbers,
                            prompt_style=args.prompt_style,
                        ),
                        "completion": completion,
                        "answer": expression,
                        "trace_type": "rollback_search",
                        "prompt_style": args.prompt_style,
                        "source": "temporary_rollback_experiment",
                    }
                )
        print(
            json.dumps(
                {
                    "build_event": "puzzle_done",
                    "global_index": global_index,
                    "local_index": local_index,
                    "records": len(records),
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "total_puzzles": len(split_records),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, sort_keys=True) + "\n")

    summary = rollback.summarize_records(records)
    summary.update(
        {
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "split_start": start,
            "split_end": end,
        }
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
