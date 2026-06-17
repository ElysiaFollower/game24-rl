"""Evaluation utilities for model checkpoints and raw outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from game24_rl.data_gen import PROMPT_STYLE_PLAIN, format_completion, format_prompt
from game24_rl.datasets import read_manifest
from game24_rl.solver import solve_puzzle
from game24_rl.verifier import VERIFIER_VERSION, VerificationResult, verify_answer

ANSWER_CONTRACT = "<answer>...</answer>"
EVALUATION_SCHEMA_VERSION = "game24-eval-report-v1"
RAW_OUTPUT_SCHEMA_VERSION = "game24-raw-outputs-v1"
RERANK_RAW_OUTPUT_SCHEMA_VERSION = "game24-verifier-rerank-raw-outputs-v1"


@dataclass(frozen=True)
class DecodingConfig:
    """Generation settings attached to an evaluation report."""

    do_sample: bool = False
    max_new_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class EvaluationMetricSummary:
    """Aggregate verifier metrics for one raw-output file."""

    total: int
    format_ok: int
    valid_expr: int
    solved: int

    @property
    def format_rate(self) -> float:
        return _safe_rate(self.format_ok, self.total)

    @property
    def valid_expr_rate(self) -> float:
        return _safe_rate(self.valid_expr, self.total)

    @property
    def solve_rate(self) -> float:
        return _safe_rate(self.solved, self.total)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Reads a JSON Lines file into dictionaries."""

    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Writes dictionaries as JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_output_records(
    records: Iterable[dict[str, Any]],
) -> tuple[EvaluationMetricSummary, list[dict[str, Any]]]:
    """Runs the strict verifier over raw output records."""

    details: list[dict[str, Any]] = []
    total = 0
    format_ok = 0
    valid_expr = 0
    solved = 0
    for record in records:
        total += 1
        result = verify_answer(
            record["output"],
            puzzle=record["numbers"],
            target=record.get("target", 24),
        )
        is_format_ok = not result.reason.startswith("answer_contract")
        is_valid_expr = result.valid or result.reason == "wrong_value"
        format_ok += int(is_format_ok)
        valid_expr += int(is_valid_expr)
        solved += int(result.valid)
        details.append(_detail_record(record, result, is_format_ok, is_valid_expr))

    return EvaluationMetricSummary(
        total=total,
        format_ok=format_ok,
        valid_expr=valid_expr,
        solved=solved,
    ), details


def evaluate_raw_outputs_file(
    raw_outputs_path: str | Path,
    report_path: str | Path,
    *,
    model_name: str,
    checkpoint: str | None,
    split_manifest: str | Path,
    split: str,
    decoding: DecodingConfig,
    generation_prompt_style: str | None = None,
) -> dict[str, Any]:
    """Evaluates raw model outputs and writes a machine-readable report."""

    raw_outputs = read_jsonl(raw_outputs_path)
    summary, details = evaluate_output_records(raw_outputs)
    report = build_evaluation_report(
        summary=summary,
        details=details,
        raw_outputs_path=raw_outputs_path,
        model_name=model_name,
        checkpoint=checkpoint,
        split_manifest=split_manifest,
        split=split,
        decoding=decoding,
        generation_prompt_style=generation_prompt_style,
    )
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_verifier_reranked_raw_outputs(
    *,
    greedy_records: Iterable[dict[str, Any]],
    sampled_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Chooses strict-verifier-valid sampled fallbacks for failed greedy outputs.

    Greedy outputs remain the primary answer. Sampled candidates are only used
    when the greedy output fails this repository's strict verifier, and the
    first valid sampled completion for the same puzzle id is selected.
    """

    sampled_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in sampled_records:
        sampled_by_id.setdefault(str(record["id"]), []).append(record)

    reranked = []
    for record in greedy_records:
        greedy_result = verify_answer(
            record["output"],
            puzzle=record["numbers"],
            target=record.get("target", 24),
        )
        selected = dict(record)
        selected["schema_version"] = RERANK_RAW_OUTPUT_SCHEMA_VERSION
        selected["rerank_source"] = "greedy"
        selected["greedy_reason"] = greedy_result.reason
        selected["sample_index"] = None
        if not greedy_result.valid:
            fallback = _first_valid_sampled_record(
                sampled_by_id.get(str(record["id"]), []),
                numbers=record["numbers"],
                target=record.get("target", 24),
            )
            if fallback is not None:
                selected = dict(fallback)
                selected["schema_version"] = RERANK_RAW_OUTPUT_SCHEMA_VERSION
                selected["rerank_source"] = "sampled_verifier_fallback"
                selected["greedy_reason"] = greedy_result.reason
        reranked.append(selected)
    return reranked


def write_verifier_rerank_report(
    *,
    greedy_raw_outputs_path: str | Path,
    sampled_raw_outputs_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    checkpoint: str | None,
    split_manifest: str | Path,
    split: str,
    greedy_decoding: DecodingConfig,
    sampled_decoding: DecodingConfig,
    generation_prompt_style: str | None = None,
) -> dict[str, Any]:
    """Writes reranked raw outputs and a standard strict-verifier report."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reranked_raw_path = output_path / f"{split}-verifier-reranked-raw-outputs.jsonl"
    report_path = output_path / f"{split}-verifier-rerank-eval-report.json"
    reranked_records = build_verifier_reranked_raw_outputs(
        greedy_records=read_jsonl(greedy_raw_outputs_path),
        sampled_records=read_jsonl(sampled_raw_outputs_path),
    )
    write_jsonl(reranked_records, reranked_raw_path)
    report = evaluate_raw_outputs_file(
        raw_outputs_path=reranked_raw_path,
        report_path=report_path,
        model_name=model_name,
        checkpoint=checkpoint,
        split_manifest=split_manifest,
        split=split,
        decoding=sampled_decoding,
        generation_prompt_style=generation_prompt_style,
    )
    report["decoding"] = {
        "policy": "greedy_then_sampled_verifier_rerank",
        "greedy": asdict(greedy_decoding),
        "sampled": asdict(sampled_decoding),
    }
    report["greedy_raw_outputs_path"] = str(greedy_raw_outputs_path)
    report["sampled_raw_outputs_path"] = str(sampled_raw_outputs_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_evaluation_report(
    *,
    summary: EvaluationMetricSummary,
    details: list[dict[str, Any]],
    raw_outputs_path: str | Path,
    model_name: str,
    checkpoint: str | None,
    split_manifest: str | Path,
    split: str,
    decoding: DecodingConfig,
    generation_prompt_style: str | None = None,
) -> dict[str, Any]:
    """Builds the report artifact for one evaluation run."""

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "model_name": model_name,
        "checkpoint": checkpoint,
        "split_manifest": str(split_manifest),
        "split": split,
        "answer_contract": ANSWER_CONTRACT,
        "verifier_version": VERIFIER_VERSION,
        "decoding": asdict(decoding),
        "generation_prompt_style": generation_prompt_style,
        "raw_outputs_path": str(raw_outputs_path),
        "metrics": {
            "total": summary.total,
            "format_rate": summary.format_rate,
            "valid_expr_rate": summary.valid_expr_rate,
            "solve_rate": summary.solve_rate,
            "format_ok": summary.format_ok,
            "valid_expr": summary.valid_expr,
            "solved": summary.solved,
        },
        "details": details,
    }


def _first_valid_sampled_record(
    records: list[dict[str, Any]],
    *,
    numbers: list[int],
    target: int,
) -> dict[str, Any] | None:
    for record in records:
        result = verify_answer(
            record["output"],
            puzzle=numbers,
            target=target,
        )
        if result.valid:
            return record
    return None


def build_solver_raw_outputs(
    manifest: dict[str, Any],
    *,
    split: str,
    limit: int | None = None,
    prompt_style: str = PROMPT_STYLE_PLAIN,
) -> list[dict[str, Any]]:
    """Builds verifier-valid raw outputs from the exact solver for dry runs."""

    records = manifest["splits"][split]
    if limit is not None:
        records = records[:limit]

    outputs = []
    for record in records:
        solutions = solve_puzzle(record["numbers"], target=record["target"])
        if not solutions:
            output = "<answer>1 + 1 + 1 + 1</answer>"
        else:
            output = format_completion(solutions[0])
        outputs.append(
            {
                "schema_version": RAW_OUTPUT_SCHEMA_VERSION,
                "id": record["id"],
                "numbers": record["numbers"],
                "target": record["target"],
                "prompt": format_prompt(record["numbers"], prompt_style=prompt_style),
                "output": output,
                "source": "exact_solver_dry_run",
            }
        )
    return outputs


def evaluate_solver_dry_run(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    limit: int = 16,
    model_name: str = "exact-solver-dry-run",
    prompt_style: str = PROMPT_STYLE_PLAIN,
) -> dict[str, Any]:
    """Writes raw-output and evaluation artifacts without loading a model."""

    manifest = read_manifest(manifest_path)
    output_path = Path(output_dir)
    raw_outputs_path = output_path / f"{split}-raw-outputs.jsonl"
    report_path = output_path / f"{split}-eval-report.json"
    raw_outputs = build_solver_raw_outputs(
        manifest,
        split=split,
        limit=limit,
        prompt_style=prompt_style,
    )
    write_jsonl(raw_outputs, raw_outputs_path)
    return evaluate_raw_outputs_file(
        raw_outputs_path=raw_outputs_path,
        report_path=report_path,
        model_name=model_name,
        checkpoint=None,
        split_manifest=manifest_path,
        split=split,
        decoding=DecodingConfig(),
        generation_prompt_style=prompt_style,
    )


def generate_checkpoint_outputs(
    *,
    manifest_path: str | Path,
    split: str,
    output_path: str | Path,
    model_name: str,
    checkpoint: str | Path,
    decoding: DecodingConfig,
    limit: int | None = None,
    prompt_style: str = PROMPT_STYLE_PLAIN,
    training_mode: str = "lora",
    batch_size: int = 1,
) -> None:
    """Generates model outputs for a split and writes raw-output JSONL.

    Heavy ML dependencies are imported inside this function so local dry-run
    tests do not need Transformers, PEFT, TRL, Torch, or model downloads.
    """

    import torch  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    manifest = read_manifest(manifest_path)
    records = manifest["splits"][split]
    if limit is not None:
        records = records[:limit]

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
    elif training_mode == "lora":
        from peft import PeftModel  # pylint: disable=import-outside-toplevel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
    else:
        raise ValueError(f"unsupported training_mode: {training_mode}")
    model.eval()

    raw_outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            format_prompt(record["numbers"], prompt_style=prompt_style)
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
                    "schema_version": RAW_OUTPUT_SCHEMA_VERSION,
                    "id": record["id"],
                    "numbers": record["numbers"],
                    "target": record["target"],
                    "prompt": prompts[index],
                    "output": output,
                    "source": f"{training_mode}_checkpoint_generation",
                }
            )

    write_jsonl(raw_outputs, output_path)


def _detail_record(
    record: dict[str, Any],
    result: VerificationResult,
    format_ok: bool,
    valid_expr: bool,
) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "numbers": record["numbers"],
        "target": record.get("target", 24),
        "valid": result.valid,
        "reason": result.reason,
        "format_ok": format_ok,
        "valid_expr": valid_expr,
        "expression": result.expression,
        "value": str(result.value) if result.value is not None else None,
        "used_numbers": list(result.numbers),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
