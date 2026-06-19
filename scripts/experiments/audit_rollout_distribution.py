"""Audit sampled rollout distribution for GRPO readiness."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from game24_rl.data_gen import (  # noqa: E402
    PROMPT_STYLE_QWEN_CHAT,
    PROMPT_STYLE_QWEN_CHAT_SEARCH,
    PROMPT_STYLE_QWEN_CHAT_TARGET,
    format_prompt,
)
from game24_rl.datasets import read_manifest  # noqa: E402
from game24_rl.evaluate import DecodingConfig, write_jsonl  # noqa: E402
from game24_rl.verifier import verify_answer  # noqa: E402


@dataclass(frozen=True)
class RolloutSummary:
    """Aggregate rollout diagnostics for one sampled checkpoint."""

    total_prompts: int
    num_generations: int
    total_outputs: int
    solved: int
    format_ok: int
    valid_expr: int
    pass_at_1_greedy_proxy: int
    pass_at_k: int
    mixed_reward_groups: int
    all_correct_groups: int
    all_wrong_groups: int
    zero_std_groups: int
    truncation_like_failures: int
    completion_len_mean: float
    completion_len_p50: int
    completion_len_p95: int


def main() -> None:
    """Runs sampled generation and writes GRPO-readiness diagnostics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/processed/splits/standard-game24-v1.json"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--training-mode", choices=["full", "lora"], default="full")
    parser.add_argument(
        "--prompt-style",
        choices=[
            PROMPT_STYLE_QWEN_CHAT,
            PROMPT_STYLE_QWEN_CHAT_SEARCH,
            PROMPT_STYLE_QWEN_CHAT_TARGET,
        ],
        default=PROMPT_STYLE_QWEN_CHAT,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--ids-file",
        help="Optional newline-delimited puzzle ids to audit.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.split}-sampled-rollouts.jsonl"
    detail_path = output_dir / f"{args.split}-rollout-details.json"
    summary_path = output_dir / "summary.json"

    records, details, summary = sample_rollouts(args)
    write_jsonl(records, raw_path)
    detail_path.write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_dict = {
        **asdict(summary),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "manifest": args.manifest,
        "prompt_style": args.prompt_style,
        "decoding": asdict(
            DecodingConfig(
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        ),
        "raw_outputs_path": str(raw_path),
        "details_path": str(detail_path),
        "reason_counts": dict(Counter(item["reason"] for item in details)),
    }
    summary_path.write_text(
        json.dumps(summary_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_dict, indent=2, sort_keys=True))


def sample_rollouts(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], RolloutSummary]:
    """Samples multiple completions per puzzle and summarizes reward variance."""

    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    manifest = read_manifest(args.manifest)
    prompts = list(manifest["splits"][args.split])
    if args.ids_file:
        ids = {
            line.strip()
            for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        prompts = [record for record in prompts if record["id"] in ids]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    tokenizer_source = (
        args.checkpoint if args.training_mode == "full" else args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.training_mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        from peft import PeftModel  # pylint: disable=import-outside-toplevel

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    raw_outputs: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    grouped_rewards: list[list[int]] = []
    lengths: list[int] = []

    expanded = []
    for record in prompts:
        prompt = format_prompt(
            record["numbers"],
            target=record.get("target", 24),
            prompt_style=args.prompt_style,
        )
        for sample_index in range(args.num_generations):
            expanded.append((record, sample_index, prompt))

    for start in range(0, len(expanded), args.batch_size):
        batch = expanded[start : start + args.batch_size]
        prompt_texts = [item[2] for item in batch]
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(
            model.device
        )
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        for index, (record, sample_index, prompt) in enumerate(batch):
            generated_ids = generated[index][input_width:]
            output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            lengths.append(int(generated_ids.numel()))
            result = verify_answer(
                output,
                puzzle=record["numbers"],
                target=record.get("target", 24),
            )
            raw_outputs.append(
                {
                    "schema_version": "game24-sampled-rollouts-v1",
                    "id": record["id"],
                    "sample_index": sample_index,
                    "numbers": record["numbers"],
                    "target": record["target"],
                    "prompt": prompt,
                    "output": output,
                }
            )
            details.append(
                {
                    "id": record["id"],
                    "sample_index": sample_index,
                    "numbers": record["numbers"],
                    "target": record["target"],
                    "valid": result.valid,
                    "reward": int(result.valid),
                    "reason": result.reason,
                    "expression": result.expression,
                    "value": str(result.value) if result.value is not None else None,
                    "used_numbers": list(result.numbers),
                    "completion_tokens": int(generated_ids.numel()),
                    "has_answer_open": "<answer>" in output,
                    "has_answer_close": "</answer>" in output,
                }
            )

    by_id: dict[str, list[int]] = {}
    for detail in details:
        by_id.setdefault(str(detail["id"]), []).append(int(detail["reward"]))
    grouped_rewards = list(by_id.values())

    reason_counts = Counter(item["reason"] for item in details)
    solved = sum(int(item["valid"]) for item in details)
    format_ok = sum(
        int(not str(item["reason"]).startswith("answer_contract")) for item in details
    )
    valid_expr = sum(
        int(item["valid"] or item["reason"] == "wrong_value") for item in details
    )
    pass_at_1 = sum(group[0] for group in grouped_rewards)
    pass_at_k = sum(int(any(group)) for group in grouped_rewards)
    mixed = sum(int(0 < sum(group) < len(group)) for group in grouped_rewards)
    all_correct = sum(int(sum(group) == len(group)) for group in grouped_rewards)
    all_wrong = sum(int(sum(group) == 0) for group in grouped_rewards)
    zero_std = all_correct + all_wrong
    sorted_lengths = sorted(lengths)
    summary = RolloutSummary(
        total_prompts=len(prompts),
        num_generations=args.num_generations,
        total_outputs=len(details),
        solved=solved,
        format_ok=format_ok,
        valid_expr=valid_expr,
        pass_at_1_greedy_proxy=pass_at_1,
        pass_at_k=pass_at_k,
        mixed_reward_groups=mixed,
        all_correct_groups=all_correct,
        all_wrong_groups=all_wrong,
        zero_std_groups=zero_std,
        truncation_like_failures=reason_counts[
            "answer_contract:expected exactly one <answer>...</answer> block"
        ],
        completion_len_mean=mean(lengths) if lengths else 0.0,
        completion_len_p50=_percentile(sorted_lengths, 0.50),
        completion_len_p95=_percentile(sorted_lengths, 0.95),
    )
    return raw_outputs, details, summary


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


if __name__ == "__main__":
    main()
