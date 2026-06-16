"""Temporary rollback/search-trace SFT experiment.

This script is intentionally outside the main training path. It tests whether
baseline-style search and rollback traces improve SFT while preserving this
repo's split, strict verifier, and ``<answer>...</answer>`` answer contract.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from game24_rl.data_gen import (  # noqa: E402
    PROMPT_STYLE_QWEN_CHAT,
    PROMPT_STYLE_QWEN_CHAT_SEARCH,
    format_prompt,
)
from game24_rl.datasets import puzzle_id, read_manifest  # noqa: E402
from game24_rl.evaluate import (  # noqa: E402
    DecodingConfig,
    evaluate_raw_outputs_file,
    write_jsonl,
)
from game24_rl.verifier import verify_answer  # noqa: E402


@dataclass
class Card:
    """One intermediate value in a 24-point search state."""

    value: Fraction
    expression: str


@dataclass
class LogNode:
    """DFS log tree node used for random pruning."""

    line: str
    parent: LogNode | None = None
    children: list[LogNode] = field(default_factory=list)
    keep: bool = False
    is_solution: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/processed/splits/standard-game24-v1.json"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--samples-per-puzzle", type=int, default=3)
    parser.add_argument("--max-lines", default="6,7,8,9,10,11,12,13,14,15,16,17")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--save-steps", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--prompt-style",
        choices=[PROMPT_STYLE_QWEN_CHAT, PROMPT_STYLE_QWEN_CHAT_SEARCH],
        default=PROMPT_STYLE_QWEN_CHAT_SEARCH,
    )
    parser.add_argument(
        "--training-mode",
        choices=["lora", "full"],
        default="lora",
    )
    parser.add_argument(
        "--mode", choices=["build", "train", "eval", "all"], default="all"
    )
    parser.add_argument("--eval-split", default="validation")
    args = parser.parse_args()

    if args.mode in {"build", "all"}:
        build_dataset(args)
    if args.mode in {"train", "all"}:
        train(args)
    if args.mode in {"eval", "all"}:
        evaluate(args)


def build_dataset(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.manifest)
    rng = random.Random(args.seed)
    max_lines_values = [int(value) for value in args.max_lines.split(",") if value]
    output_path = Path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    seen: set[tuple[str, str]] = set()
    split_records = list(manifest["splits"][args.split])
    for record in split_records:
        base_numbers = list(record["numbers"])
        for sample_index in range(args.samples_per_puzzle):
            prompt_numbers = list(base_numbers)
            rng.shuffle(prompt_numbers)
            logs = solve_with_search_logs(prompt_numbers, rng)
            if logs is None:
                raise RuntimeError(f"solver failed for {prompt_numbers}")
            expression = extract_final_expression(logs)
            for max_lines in max_lines_values:
                compressed = compress_search_logs(logs, max_lines=max_lines, rng=rng)
                trace_lines = [line for line in compressed if not _is_input_line(line)]
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
                if args.max_records and len(records) >= args.max_records:
                    break
            if args.max_records and len(records) >= args.max_records:
                break
        if args.max_records and len(records) >= args.max_records:
            break

    with output_path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, sort_keys=True) + "\n")
    summary = summarize_records(records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"dataset": str(output_path), **summary}, indent=2, sort_keys=True)
    )


def solve_with_search_logs(numbers: list[int], rng: random.Random) -> list[str] | None:
    cards = [Card(Fraction(number), str(number)) for number in numbers]
    logs = [" ".join(str(number) for number in numbers)]
    return logs if dfs(cards, logs, rng) else None


def dfs(cards: list[Card], logs: list[str], rng: random.Random) -> bool:
    if len(cards) == 1:
        if cards[0].value == 24:
            logs.append(f"reach 24! expression: {cards[0].expression}")
            return True
        return False

    pairs = [(i, j) for i in range(len(cards)) for j in range(i + 1, len(cards))]
    rng.shuffle(pairs)
    for i, j in pairs:
        left = cards[i]
        right = cards[j]
        rest = [cards[k] for k in range(len(cards)) if k not in {i, j}]
        candidates = candidate_operations(left, right)
        rng.shuffle(candidates)
        for first, operator, second, value in candidates:
            expression = f"({first.expression} {operator} {second.expression})"
            new_card = Card(value, expression)
            new_cards = rest + [new_card]
            logs.append(format_step(first, operator, second, value, new_cards))
            if dfs(new_cards, logs, rng):
                return True
            logs.append(f"roll back, left: {format_left(cards)}")
    return False


def candidate_operations(
    left: Card,
    right: Card,
) -> list[tuple[Card, str, Card, Fraction]]:
    candidates = [
        (left, "+", right, left.value + right.value),
        (left, "-", right, left.value - right.value),
        (right, "-", left, right.value - left.value),
        (left, "*", right, left.value * right.value),
    ]
    if right.value:
        candidates.append((left, "/", right, left.value / right.value))
    if left.value:
        candidates.append((right, "/", left, right.value / left.value))
    return candidates


def format_step(
    left: Card,
    operator: str,
    right: Card,
    value: Fraction,
    new_cards: list[Card],
) -> str:
    return (
        f"({format_fraction(left.value)}) {operator} "
        f"({format_fraction(right.value)}) = {format_fraction(value)}, "
        f"left: {format_left(new_cards)}"
    )


def format_left(cards: list[Card]) -> str:
    return ", ".join(format_fraction(card.value) for card in cards)


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def extract_final_expression(logs: list[str]) -> str:
    final = logs[-1]
    marker = "reach 24! expression:"
    if marker not in final:
        raise RuntimeError(f"missing final expression in {final!r}")
    return final.split(marker, 1)[1].strip()


def compress_search_logs(
    logs: list[str],
    *,
    max_lines: int,
    rng: random.Random,
) -> list[str]:
    root = build_tree(logs)
    mark_solution_path(root)
    while True:
        flattened = flatten_tree(root)
        if len(flattened) <= max_lines:
            return flattened
        deletable = gather_deletable_leaves(root)
        if not deletable:
            return flattened
        leaf = rng.choice(deletable)
        assert leaf.parent is not None
        leaf.parent.children.remove(leaf)


def build_tree(logs: list[str]) -> LogNode:
    root = LogNode("<root>")
    stack = [root]
    for line in logs:
        if line.startswith("roll back"):
            if len(stack) > 1:
                stack.pop()
            continue
        node = LogNode(line=line, parent=stack[-1])
        node.is_solution = "reach 24! expression:" in line
        stack[-1].children.append(node)
        stack.append(node)
    return root


def mark_solution_path(root: LogNode) -> None:
    def walk(node: LogNode, path: list[LogNode]) -> bool:
        path.append(node)
        found = node.is_solution
        for child in node.children:
            found = walk(child, path) or found
        if found:
            for item in path:
                item.keep = True
        path.pop()
        return found

    walk(root, [])


def gather_deletable_leaves(root: LogNode) -> list[LogNode]:
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if node is not root and not node.keep and not node.children:
            leaves.append(node)
    return leaves


def flatten_tree(root: LogNode) -> list[str]:
    output = []

    def walk(node: LogNode) -> None:
        if node.line != "<root>":
            output.append(node.line)
        for child in node.children:
            walk(child)
        if node.parent is not None and not node.keep:
            output.append("roll back, left: " + left_state(node.parent.line))

    walk(root)
    return output


def left_state(line: str) -> str:
    if "left: " in line:
        return line.rsplit("left: ", 1)[1]
    return line


def _is_input_line(line: str) -> bool:
    parts = line.split()
    return len(parts) == 4 and all(part.isdigit() for part in parts)


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    line_counts = [len(str(record["completion"]).splitlines()) for record in records]
    rollback_records = sum(
        "roll back" in str(record["completion"]) for record in records
    )
    unique_puzzles = {
        tuple(sorted(record["numbers"]))  # type: ignore[arg-type]
        for record in records
    }
    return {
        "records": len(records),
        "unique_puzzles": len(unique_puzzles),
        "line_count_min": min(line_counts) if line_counts else 0,
        "line_count_median": sorted(line_counts)[len(line_counts) // 2]
        if line_counts
        else 0,
        "line_count_max": max(line_counts) if line_counts else 0,
        "rollback_records": rollback_records,
    }


def train(args: argparse.Namespace) -> None:
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer  # pylint: disable=import-outside-toplevel

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.training_mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenized = dataset.map(
            lambda examples: preprocess_full_sft_batch(
                examples,
                tokenizer=tokenizer,
                max_length=args.max_length,
            ),
            batched=True,
            remove_columns=dataset.column_names,
        )
        splits = tokenized.train_test_split(test_size=0.01, seed=args.seed)
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            run_name=run_dir.name,
            max_steps=args.max_steps,
            learning_rate=5e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            save_steps=args.save_steps,
            save_strategy="steps",
            logging_steps=10,
            logging_dir=str(run_dir / "logs"),
            seed=args.seed,
            bf16=True,
            report_to=["none"],
            optim="adamw_torch",
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            remove_unused_columns=False,
            eval_strategy="steps",
            eval_steps=args.save_steps,
            per_device_eval_batch_size=1,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            save_only_model=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=splits["train"],
            eval_dataset=splits["test"],
            processing_class=tokenizer,
        )
    else:
        from peft import LoraConfig  # pylint: disable=import-outside-toplevel

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        training_args = SFTConfig(
            output_dir=str(run_dir),
            run_name=run_dir.name,
            max_length=args.max_length,
            max_steps=args.max_steps,
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            save_steps=args.save_steps,
            save_strategy="steps",
            logging_steps=10,
            logging_dir=str(run_dir / "logs"),
            seed=args.seed,
            bf16=True,
            report_to=["none"],
            eos_token="<|im_end|>",
            completion_only_loss=True,
        )
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
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


def preprocess_full_sft_batch(
    examples: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenizes prompt/completion records with prompt labels masked."""

    model_inputs: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    pad_token_id = tokenizer.pad_token_id
    for prompt, completion in zip(
        examples["prompt"],
        examples["completion"],
        strict=True,
    ):
        prompt_ids = tokenizer.encode(prompt)
        completion_ids = tokenizer.encode(
            completion,
            add_special_tokens=False,
        ) + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + completion_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids += [pad_token_id] * pad_length
            labels += [-100] * pad_length
            attention_mask += [0] * pad_length
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    return model_inputs


def evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    eval_root = run_dir / "eval"
    decoding = DecodingConfig(do_sample=False, max_new_tokens=args.max_new_tokens)
    checkpoints = sorted(
        [path for path in run_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    final = run_dir / "final"
    if final.exists():
        checkpoints.append(final)

    rows = []
    for checkpoint in checkpoints:
        output_dir = eval_root / checkpoint.name
        raw_outputs = output_dir / f"{args.eval_split}-raw-outputs.jsonl"
        report_path = output_dir / f"{args.eval_split}-eval-report.json"
        generate_checkpoint_outputs(
            manifest_path=args.manifest,
            split=args.eval_split,
            output_path=raw_outputs,
            model_name=args.model_name,
            checkpoint=checkpoint,
            decoding=decoding,
            training_mode=args.training_mode,
            batch_size=args.eval_batch_size,
            prompt_style=args.prompt_style,
        )
        report = evaluate_raw_outputs_file(
            raw_outputs_path=raw_outputs,
            report_path=report_path,
            model_name=args.model_name,
            checkpoint=str(checkpoint),
            split_manifest=args.manifest,
            split=args.eval_split,
            decoding=decoding,
            generation_prompt_style=args.prompt_style,
        )
        rows.append({"checkpoint": checkpoint.name, "metrics": report["metrics"]})
        print(json.dumps(rows[-1], indent=2, sort_keys=True))

    (eval_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_checkpoint_outputs(
    *,
    manifest_path: str | Path,
    split: str,
    output_path: str | Path,
    model_name: str,
    checkpoint: str | Path,
    decoding: DecodingConfig,
    training_mode: str,
    batch_size: int,
    prompt_style: str,
) -> None:
    """Generates strict-eval raw outputs for LoRA or full checkpoints."""

    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    manifest = read_manifest(manifest_path)
    records = manifest["splits"][split]
    tokenizer_source = str(checkpoint) if training_mode == "full" else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if training_mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        from peft import PeftModel  # pylint: disable=import-outside-toplevel

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
        prompts = [
            format_prompt(
                record["numbers"],
                prompt_style=prompt_style,
            )
            for record in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=decoding.do_sample,
                max_new_tokens=decoding.max_new_tokens,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        for index, record in enumerate(batch):
            output = tokenizer.decode(
                generated[index][input_width:],
                skip_special_tokens=True,
            )
            raw_outputs.append(
                {
                    "schema_version": "game24-raw-outputs-v1",
                    "id": record["id"],
                    "numbers": record["numbers"],
                    "target": record["target"],
                    "prompt": prompts[index],
                    "output": output,
                    "source": f"rollback_{training_mode}_checkpoint_generation",
                }
            )

    write_jsonl(raw_outputs, output_path)


if __name__ == "__main__":
    main()
