"""Evaluation utilities for model checkpoints and raw outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from game24_rl.data_gen import format_completion, format_prompt
from game24_rl.datasets import read_manifest
from game24_rl.solver import solve_puzzle
from game24_rl.verifier import VERIFIER_VERSION, VerificationResult, verify_answer

ANSWER_CONTRACT = "<answer>...</answer>"
EVALUATION_SCHEMA_VERSION = "game24-eval-report-v1"
RAW_OUTPUT_SCHEMA_VERSION = "game24-raw-outputs-v1"


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
    )
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
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


def build_solver_raw_outputs(
    manifest: dict[str, Any],
    *,
    split: str,
    limit: int | None = None,
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
                "prompt": format_prompt(record["numbers"]),
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
) -> dict[str, Any]:
    """Writes raw-output and evaluation artifacts without loading a model."""

    manifest = read_manifest(manifest_path)
    output_path = Path(output_dir)
    raw_outputs_path = output_path / f"{split}-raw-outputs.jsonl"
    report_path = output_path / f"{split}-eval-report.json"
    raw_outputs = build_solver_raw_outputs(manifest, split=split, limit=limit)
    write_jsonl(raw_outputs, raw_outputs_path)
    return evaluate_raw_outputs_file(
        raw_outputs_path=raw_outputs_path,
        report_path=report_path,
        model_name=model_name,
        checkpoint=None,
        split_manifest=manifest_path,
        split=split,
        decoding=DecodingConfig(),
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
) -> None:
    """Generates model outputs for a split and writes raw-output JSONL.

    Heavy ML dependencies are imported inside this function so local dry-run
    tests do not need Transformers, PEFT, TRL, Torch, or model downloads.
    """

    from peft import PeftModel  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    manifest = read_manifest(manifest_path)
    records = manifest["splits"][split]
    if limit is not None:
        records = records[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model.eval()

    raw_outputs = []
    for record in records:
        prompt = format_prompt(record["numbers"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            do_sample=decoding.do_sample,
            max_new_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        raw_outputs.append(
            {
                "schema_version": RAW_OUTPUT_SCHEMA_VERSION,
                "id": record["id"],
                "numbers": record["numbers"],
                "target": record["target"],
                "prompt": prompt,
                "output": output,
                "source": "checkpoint_generation",
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
