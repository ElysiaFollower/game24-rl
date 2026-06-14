"""Audit the generated SFT dataset for truncation and format risks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from game24_rl.datasets import read_manifest
from game24_rl.evaluate import read_jsonl
from game24_rl.verifier import verify_answer


def main() -> None:
    """Summarises dataset lengths, format validity, and truncation risk."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/standard-game24-v1.json",
    )
    parser.add_argument(
        "--train-jsonl",
        default="data/processed/sft/game24-sft-v1-train.jsonl",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/sft_dataset_audit.json",
    )
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    records = read_jsonl(args.train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_lengths = []
    completion_lengths = []
    total_lengths = []
    truncated = []
    missing_separator = 0
    verifier_invalid = 0
    for record in records:
        prompt = record["prompt"]
        completion = record["completion"]
        prompt_ids = tokenizer(prompt)["input_ids"]
        completion_ids = tokenizer(completion)["input_ids"]
        total_ids = tokenizer(prompt + completion)["input_ids"]
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(len(completion_ids))
        total_lengths.append(len(total_ids))
        truncated.append(len(total_ids) > args.max_length)
        if not prompt.endswith("\n"):
            missing_separator += 1
        verification = verify_answer(
            completion,
            record["numbers"],
            target=record["target"],
        )
        if not verification.valid:
            verifier_invalid += 1

    summary = {
        "manifest_path": args.manifest,
        "train_jsonl": args.train_jsonl,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "manifest_train_count": len(manifest["splits"]["train"]),
        "record_count": len(records),
        "prompt_len_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_len_max": max(prompt_lengths) if prompt_lengths else None,
        "completion_len_min": min(completion_lengths) if completion_lengths else None,
        "completion_len_max": max(completion_lengths) if completion_lengths else None,
        "total_len_min": min(total_lengths) if total_lengths else None,
        "total_len_max": max(total_lengths) if total_lengths else None,
        "truncated_count": sum(truncated),
        "truncated_rate": sum(truncated) / len(truncated) if truncated else 0.0,
        "missing_separator_count": missing_separator,
        "verifier_invalid_count": verifier_invalid,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
