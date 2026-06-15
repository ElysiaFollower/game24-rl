"""Temporary fixed-state-trace SFT experiment.

This script tests a narrow teacher format:

<think>
<s0> 5 5 5 9
<s1> 5 + 5 = 10 | left: 10 5 9
<s2> 5 + 9 = 14 | left: 10 14
<s3> 10 + 14 = 24 | left: 24
</think>
<answer>((5 + 5) + (5 + 9))</answer>

It intentionally stays outside the main training path while we decide whether
this trace design deserves promotion into repo-native data generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from game24_rl.datasets import puzzle_id, read_manifest  # noqa: E402
from game24_rl.evaluate import (  # noqa: E402
    DecodingConfig,
    evaluate_raw_outputs_file,
    write_jsonl,
)
from game24_rl.solver import Solution, solve_puzzle  # noqa: E402
from game24_rl.verifier import verify_answer  # noqa: E402

FIXED_TRACE_TYPE = "fixed_single_path_state_trace_v1"
FIXED_TRACE_PROMPT_STYLE = "qwen_chat_fixed_trace_v1"
RAW_OUTPUT_SCHEMA_VERSION = "game24-raw-outputs-v1"
FIXED_TRACE_SYSTEM_PROMPT = """Play the 24-point game. Given four numbers, reach 24
using +, -, *, and /. Use each provided number exactly once.

Output exactly this format:
<think>
<s0> original numbers
<s1> a op b = c | left: remaining numbers after step 1
<s2> a op b = c | left: remaining numbers after step 2
<s3> a op b = 24 | left: 24
</think>
<answer>final expression using the original numbers exactly once</answer>

Example:
Input numbers: 5 5 5 9
Output:
<think>
<s0> 5 5 5 9
<s1> 5 + 9 = 14 | left: 5 5 14
<s2> 5 + 5 = 10 | left: 10 14
<s3> 10 + 14 = 24 | left: 24
</think>
<answer>((5 + 5) + (5 + 9))</answer>"""
_STEP_PATTERN = re.compile(
    r"^\((?P<left>[^()]+)\) (?P<op>[+\-*/]) \((?P<right>[^()]+)\) "
    r"= (?P<value>[^,]+), left: (?P<remaining>.+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/standard-game24-v1.json",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated TRL reporting integrations.",
    )
    parser.add_argument(
        "--eval-checkpoints",
        default="all",
        help="'all', 'final', or comma-separated checkpoint names.",
    )
    parser.add_argument(
        "--mode",
        choices=["build", "train", "eval", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.mode in {"build", "all"}:
        build_dataset(args)
    if args.mode in {"train", "all"}:
        train(args)
    if args.mode in {"eval", "all"}:
        evaluate(args)


def build_dataset(args: argparse.Namespace) -> None:
    """Builds one deterministic fixed-state trace per train puzzle."""

    manifest = read_manifest(args.manifest)
    output_path = Path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for split_record in manifest["splits"][args.split]:
        numbers = list(split_record["numbers"])
        solution = solve_puzzle(numbers, target=split_record["target"])[0]
        completion = format_fixed_trace_completion(solution)
        verification = verify_answer(completion, numbers, target=split_record["target"])
        if not verification.valid:
            raise RuntimeError(
                f"invalid generated record {numbers}: {verification.reason}"
            )

        records.append(
            {
                "id": f"fixed-trace-{puzzle_id(numbers)}-00",
                "numbers": numbers,
                "target": split_record["target"],
                "prompt": format_fixed_trace_prompt(numbers),
                "completion": completion,
                "answer": solution.expression,
                "trace": list(solution.trace),
                "trace_type": FIXED_TRACE_TYPE,
                "prompt_style": FIXED_TRACE_PROMPT_STYLE,
                "source": "exact_solver_fixed_single_path",
            }
        )
        if args.max_records and len(records) >= args.max_records:
            break

    write_jsonl(records, output_path)
    summary = summarize_records(records)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset": str(output_path), **summary}, indent=2))


def format_fixed_trace_completion(solution: Solution) -> str:
    """Formats a solver solution as the fixed single-path state trace."""

    lines = [f"<s0> {' '.join(str(number) for number in solution.numbers)}"]
    for step_index, trace_line in enumerate(solution.trace, start=1):
        lines.append(f"<s{step_index}> {normalize_solver_trace_line(trace_line)}")
    return (
        "<think>\n"
        + "\n".join(lines)
        + f"\n</think>\n<answer>{solution.expression}</answer>"
    )


def format_fixed_trace_prompt(numbers: list[int] | tuple[int, ...]) -> str:
    """Formats the explicit few-shot prompt for the fixed-state trace contract."""

    joined = " ".join(str(number) for number in numbers)
    return (
        f"<|im_start|>system\n{FIXED_TRACE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nInput numbers: {joined}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize_solver_trace_line(trace_line: str) -> str:
    """Normalizes repo solver trace text into the fixed state grammar."""

    match = _STEP_PATTERN.match(trace_line)
    if not match:
        raise ValueError(f"unexpected solver trace line: {trace_line!r}")
    remaining = " ".join(part.strip() for part in match.group("remaining").split(","))
    return (
        f"{match.group('left')} {match.group('op')} {match.group('right')} "
        f"= {match.group('value')} | left: {remaining}"
    )


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarizes generated fixed-trace data."""

    line_counts = [len(str(record["completion"]).splitlines()) for record in records]
    unique_puzzles = {tuple(record["numbers"]) for record in records}
    examples = [
        {
            "numbers": record["numbers"],
            "completion": record["completion"],
        }
        for record in records[:3]
    ]
    return {
        "records": len(records),
        "unique_puzzles": len(unique_puzzles),
        "trace_type": FIXED_TRACE_TYPE,
        "prompt_style": FIXED_TRACE_PROMPT_STYLE,
        "line_count_min": min(line_counts) if line_counts else 0,
        "line_count_median": sorted(line_counts)[len(line_counts) // 2]
        if line_counts
        else 0,
        "line_count_max": max(line_counts) if line_counts else 0,
        "examples": examples,
    }


def train(args: argparse.Namespace) -> None:
    """Runs LoRA SFT with TRL on the fixed-trace JSONL."""

    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from peft import LoraConfig  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # pylint: disable=import-outside-toplevel

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        max_length=args.max_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_steps=10,
        logging_dir=str(run_dir / "logs"),
        seed=args.seed,
        bf16=True,
        report_to=parse_report_to(args.report_to),
        eos_token="<|im_end|>",
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))
    write_run_metadata(args, run_dir)


def write_run_metadata(args: argparse.Namespace, run_dir: Path) -> None:
    """Writes lightweight metadata for later experiment reconstruction."""

    metadata = {
        "trace_type": FIXED_TRACE_TYPE,
        "prompt_style": FIXED_TRACE_PROMPT_STYLE,
        "model_name": args.model_name,
        "manifest": args.manifest,
        "dataset": args.dataset,
        "split": args.split,
        "eval_split": args.eval_split,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "report_to": parse_report_to(args.report_to),
        "seed": args.seed,
    }
    (run_dir / "fixed-trace-run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> None:
    """Evaluates selected checkpoints with a batched strict-verifier pipeline."""

    run_dir = Path(args.run_dir)
    checkpoints = select_checkpoints(run_dir, args.eval_checkpoints)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found in {run_dir}")

    decoding = DecodingConfig(do_sample=False, max_new_tokens=args.max_new_tokens)
    eval_root = run_dir / "eval"
    rows = []
    for checkpoint in checkpoints:
        output_dir = eval_root / checkpoint.name
        raw_outputs = output_dir / f"{args.eval_split}-raw-outputs.jsonl"
        report_path = output_dir / f"{args.eval_split}-eval-report.json"
        generate_batched_checkpoint_outputs(
            manifest_path=args.manifest,
            split=args.eval_split,
            output_path=raw_outputs,
            model_name=args.model_name,
            checkpoint=checkpoint,
            decoding=decoding,
            batch_size=args.eval_batch_size,
        )
        report = evaluate_raw_outputs_file(
            raw_outputs_path=raw_outputs,
            report_path=report_path,
            model_name=args.model_name,
            checkpoint=str(checkpoint),
            split_manifest=args.manifest,
            split=args.eval_split,
            decoding=decoding,
            generation_prompt_style=FIXED_TRACE_PROMPT_STYLE,
        )
        row = {
            "checkpoint": checkpoint.name,
            "metrics": report["metrics"],
            "reason_counts": dict(reason_counts(report)),
        }
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    (eval_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_checkpoints(run_dir: Path, selector: str) -> list[Path]:
    """Selects checkpoint paths from a run directory."""

    final = run_dir / "final"
    if selector == "final":
        return [final] if final.exists() else []

    checkpoints = sorted(
        [path for path in run_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    if final.exists():
        checkpoints.append(final)
    if selector == "all":
        return checkpoints

    selected_names = {name.strip() for name in selector.split(",") if name.strip()}
    return [path for path in checkpoints if path.name in selected_names]


def generate_batched_checkpoint_outputs(
    *,
    manifest_path: str | Path,
    split: str,
    output_path: str | Path,
    model_name: str,
    checkpoint: str | Path,
    decoding: DecodingConfig,
    batch_size: int,
) -> None:
    """Generates split outputs in batches and writes raw-output JSONL."""

    import torch  # pylint: disable=import-outside-toplevel
    from peft import PeftModel  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    manifest = read_manifest(manifest_path)
    records = manifest["splits"][split]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model.eval()

    raw_outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [format_fixed_trace_prompt(record["numbers"]) for record in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **generation_kwargs(decoding),
                pad_token_id=tokenizer.eos_token_id,
            )
        for index, record in enumerate(batch):
            output = tokenizer.decode(
                generated[index][input_width:],
                skip_special_tokens=True,
            )
            raw_outputs.append(
                {
                    "schema_version": RAW_OUTPUT_SCHEMA_VERSION,
                    "id": record["id"],
                    "numbers": record["numbers"],
                    "target": record["target"],
                    "prompt": prompts[index],
                    "output": output,
                    "source": "fixed_trace_batched_checkpoint_generation",
                }
            )

    write_jsonl(raw_outputs, output_path)


def generation_kwargs(decoding: DecodingConfig) -> dict[str, Any]:
    """Converts decoding config into non-None generation kwargs."""

    kwargs = asdict(decoding)
    return {key: value for key, value in kwargs.items() if value is not None}


def parse_report_to(value: str) -> list[str]:
    """Parses a comma-separated report_to CLI value."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or ["none"]


def reason_counts(report: dict[str, Any]) -> Counter[str]:
    """Counts strict verifier reasons in an evaluation report."""

    return Counter(detail["reason"] for detail in report["details"])


if __name__ == "__main__":
    main()
