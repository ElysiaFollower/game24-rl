"""Probe checkpoint generation modes for quick SFT failure analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from game24_rl.datasets import read_manifest
from game24_rl.evaluate import (
    DecodingConfig,
    evaluate_raw_outputs_file,
    write_jsonl,
)
from game24_rl.data_gen import format_prompt


RAW_OUTPUT_SCHEMA_VERSION = "game24-raw-outputs-v1"


def main() -> None:
    """Runs small generation-mode probes against one LoRA checkpoint."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--manifest", default="data/processed/splits/standard-game24-v1.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "chat", "raw_prefill", "chat_prefill"],
        choices=["raw", "chat", "raw_prefill", "chat_prefill"],
    )
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    records = manifest["splits"][args.split][: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    decoding = DecodingConfig(max_new_tokens=args.max_new_tokens)

    summaries = []
    for mode in args.modes:
        raw_path = output_dir / mode / f"{args.split}-raw-outputs.jsonl"
        report_path = output_dir / mode / f"{args.split}-eval-report.json"
        raw_outputs = [
            _generate_record(
                model=model,
                tokenizer=tokenizer,
                record=record,
                mode=mode,
                max_new_tokens=args.max_new_tokens,
            )
            for record in records
        ]
        write_jsonl(raw_outputs, raw_path)
        report = evaluate_raw_outputs_file(
            raw_outputs_path=raw_path,
            report_path=report_path,
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            split_manifest=args.manifest,
            split=args.split,
            decoding=decoding,
        )
        summaries.append(
            {
                "mode": mode,
                "metrics": report["metrics"],
                "raw_outputs_path": str(raw_path),
                "report_path": str(report_path),
            }
        )

    summary = {
        "checkpoint": args.checkpoint,
        "limit": args.limit,
        "split": args.split,
        "decoding": asdict(decoding),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _generate_record(
    *,
    model,
    tokenizer,
    record: dict,
    mode: str,
    max_new_tokens: int,
) -> dict:
    prompt = format_prompt(record["numbers"])
    model_input = _format_model_input(tokenizer, prompt, mode)
    inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    if mode.endswith("_prefill"):
        output = "<think>\n" + output
    return {
        "schema_version": RAW_OUTPUT_SCHEMA_VERSION,
        "id": record["id"],
        "numbers": record["numbers"],
        "target": record["target"],
        "prompt": prompt,
        "model_input": model_input,
        "mode": mode,
        "output": output,
        "source": "generation_mode_probe",
    }


def _format_model_input(tokenizer, prompt: str, mode: str) -> str:
    base_mode = mode.removesuffix("_prefill")
    if base_mode == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    if mode.endswith("_prefill"):
        text += "<think>\n"
    return text


if __name__ == "__main__":
    main()
