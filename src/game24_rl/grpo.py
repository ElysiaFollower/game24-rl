"""GRPO pilot pool-audit and compatibility helpers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from game24_rl.data_gen import PROMPT_STYLE_QWEN_CHAT, format_prompt
from game24_rl.datasets import read_manifest
from game24_rl.rewards import GRPO_REWARD_VERSION, reward_completions

GRPO_POOL_SCHEMA_VERSION = "game24-grpo-pool-v1"
GRPO_PROBE_SCHEMA_VERSION = "game24-grpo-compat-probe-v1"

TRUNCATION_REASON = "answer_contract:expected exactly one <answer>...</answer> block"


@dataclass(frozen=True)
class GrpoPoolGateConfig:
    """Quantified gates for accepting an active-difficulty GRPO pool."""

    min_pool_size: int = 200
    min_mixed_group_rate: float = 0.25
    max_zero_std_group_rate: float = 0.75
    min_correct_truncation_mixed: int = 50
    max_all_wrong_rate: float = 0.25


@dataclass(frozen=True)
class GrpoPoolAudit:
    """Summary of rollout groups against GRPO pool gates."""

    schema_version: str
    passed: bool
    failures: list[str]
    total_prompts: int
    total_outputs: int
    mixed_groups: int
    all_correct_groups: int
    all_wrong_groups: int
    zero_std_groups: int
    correct_truncation_mixed_groups: int
    mixed_group_rate: float
    zero_std_group_rate: float
    all_wrong_rate: float
    gate: GrpoPoolGateConfig
    selected_prompt_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable audit summary."""

        data = asdict(self)
        data["gate"] = asdict(self.gate)
        return data


@dataclass(frozen=True)
class GrpoCompatibilityConfig:
    """Config choices that the TRL compatibility probe must verify."""

    beta: float = 0.0
    scale_rewards: Literal["none", "group"] = "none"
    mask_truncated_completions: bool = False
    remove_unused_columns: bool = False
    loss_type: str = "dr_grpo"


def audit_rollout_groups(
    details: list[dict[str, Any]],
    *,
    gate: GrpoPoolGateConfig | None = None,
) -> GrpoPoolAudit:
    """Audits sampled rollout details for active-difficulty GRPO signal."""

    gate = gate or GrpoPoolGateConfig()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for detail in details:
        by_id.setdefault(str(detail["id"]), []).append(detail)

    total_prompts = len(by_id)
    mixed = 0
    all_correct = 0
    all_wrong = 0
    correct_truncation_mixed = 0
    selected_prompt_ids: list[str] = []

    for prompt_id, group in by_id.items():
        rewards = [float(item.get("reward", 0.0)) for item in group]
        positive = sum(reward > 0 for reward in rewards)
        has_correct = positive > 0
        has_wrong = positive < len(rewards)
        if has_correct and has_wrong:
            mixed += 1
            selected_prompt_ids.append(prompt_id)
        if positive == len(rewards):
            all_correct += 1
        if positive == 0:
            all_wrong += 1
        reasons = Counter(str(item.get("reason", "")) for item in group)
        if has_correct and reasons[TRUNCATION_REASON] > 0:
            correct_truncation_mixed += 1

    zero_std = all_correct + all_wrong
    mixed_rate = _safe_rate(mixed, total_prompts)
    zero_std_rate = _safe_rate(zero_std, total_prompts)
    all_wrong_rate = _safe_rate(all_wrong, total_prompts)
    failures = _pool_gate_failures(
        gate=gate,
        total_prompts=total_prompts,
        mixed_group_rate=mixed_rate,
        zero_std_group_rate=zero_std_rate,
        correct_truncation_mixed=correct_truncation_mixed,
        all_wrong_rate=all_wrong_rate,
    )

    return GrpoPoolAudit(
        schema_version=GRPO_POOL_SCHEMA_VERSION,
        passed=not failures,
        failures=failures,
        total_prompts=total_prompts,
        total_outputs=len(details),
        mixed_groups=mixed,
        all_correct_groups=all_correct,
        all_wrong_groups=all_wrong,
        zero_std_groups=zero_std,
        correct_truncation_mixed_groups=correct_truncation_mixed,
        mixed_group_rate=mixed_rate,
        zero_std_group_rate=zero_std_rate,
        all_wrong_rate=all_wrong_rate,
        gate=gate,
        selected_prompt_ids=selected_prompt_ids,
    )


def write_pool_manifest(
    audit: GrpoPoolAudit,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Writes a GRPO pool audit manifest."""

    payload = {
        "schema_version": GRPO_POOL_SCHEMA_VERSION,
        "metadata": metadata or {},
        "audit": audit.as_dict(),
        "selected_prompt_ids": audit.selected_prompt_ids,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_rollout_details_file(
    details_path: str | Path,
    output_path: str | Path,
    *,
    gate: GrpoPoolGateConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> GrpoPoolAudit:
    """Audits a rollout-details JSON artifact and writes a pool manifest."""

    details = json.loads(Path(details_path).read_text(encoding="utf-8"))
    if not isinstance(details, list):
        raise ValueError(f"{details_path} must contain a JSON list")
    audit = audit_rollout_groups(details, gate=gate)
    write_pool_manifest(
        audit,
        output_path,
        metadata={"details_path": str(details_path)} | (metadata or {}),
    )
    return audit


def build_grpo_probe_metadata(
    config: GrpoCompatibilityConfig,
    *,
    trl_version: str | None,
    supported_fields: set[str],
) -> dict[str, Any]:
    """Builds fail-fast metadata for a TRL GRPO compatibility probe."""

    required_fields = {
        "beta",
        "scale_rewards",
        "mask_truncated_completions",
        "remove_unused_columns",
        "loss_type",
    }
    unsupported = sorted(required_fields - supported_fields)
    return {
        "schema_version": GRPO_PROBE_SCHEMA_VERSION,
        "trl_version": trl_version,
        "beta": config.beta,
        "scale_rewards": config.scale_rewards,
        "mask_truncated_completions": config.mask_truncated_completions,
        "remove_unused_columns": config.remove_unused_columns,
        "loss_type": config.loss_type,
        "required_fields": sorted(required_fields),
        "unsupported_fields": unsupported,
        "passed": not unsupported,
    }


def build_prompt_dataset(
    *,
    manifest_path: str | Path,
    split: str,
    prompt_style: str = PROMPT_STYLE_QWEN_CHAT,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Builds prompt-only records for online GRPO rollout."""

    manifest = read_manifest(manifest_path)
    records = manifest["splits"][split]
    if limit is not None:
        records = records[:limit]
    return [
        {
            "id": record["id"],
            "numbers": record["numbers"],
            "target": record.get("target", manifest.get("target", 24)),
            "prompt": format_prompt(record["numbers"], prompt_style=prompt_style),
        }
        for record in records
    ]


def run_grpo_dry_run(
    *,
    manifest_path: str | Path,
    split: str = "train",
    output_dir: str | Path,
    prompt_style: str = PROMPT_STYLE_QWEN_CHAT,
    limit: int = 8,
) -> dict[str, Any]:
    """Writes prompt-only GRPO dry-run artifacts without loading model weights."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prompt_records = build_prompt_dataset(
        manifest_path=manifest_path,
        split=split,
        prompt_style=prompt_style,
        limit=limit,
    )
    sample_completions = [
        "<answer>((8 - 2) * (7 - 3))</answer>",
        "<think>still searching",
    ]
    sample_rewards = reward_completions(
        completions=sample_completions,
        numbers=[[8, 2, 7, 3], [8, 2, 7, 3]],
        target=[24, 24],
        id=["dry-run-ok", "dry-run-missing"],
    )
    prompts_path = output_path / f"{split}-prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as file:
        for record in prompt_records:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "schema_version": "game24-grpo-dry-run-v1",
        "manifest_path": str(manifest_path),
        "split": split,
        "prompt_style": prompt_style,
        "prompt_records": len(prompt_records),
        "prompts_path": str(prompts_path),
        "reward_version": GRPO_REWARD_VERSION,
        "sample_rewards": sample_rewards,
        "loads_model_weights": False,
    }
    metadata_path = output_path / "dry-run-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _pool_gate_failures(
    *,
    gate: GrpoPoolGateConfig,
    total_prompts: int,
    mixed_group_rate: float,
    zero_std_group_rate: float,
    correct_truncation_mixed: int,
    all_wrong_rate: float,
) -> list[str]:
    failures = []
    if total_prompts < gate.min_pool_size:
        failures.append("pool_size")
    if mixed_group_rate < gate.min_mixed_group_rate:
        failures.append("mixed_group_rate")
    if zero_std_group_rate > gate.max_zero_std_group_rate:
        failures.append("zero_std_group_rate")
    if correct_truncation_mixed < gate.min_correct_truncation_mixed:
        failures.append("correct_truncation_mixed")
    if all_wrong_rate > gate.max_all_wrong_rate:
        failures.append("all_wrong_rate")
    return failures


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
