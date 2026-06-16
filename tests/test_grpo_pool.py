"""Tests for GRPO active-pool audit gates and probe metadata."""

import json
from pathlib import Path

from game24_rl.cli import _build_grpo_config_kwargs
from game24_rl.datasets import build_split_manifest, write_manifest
from game24_rl.grpo import (
    GRPO_POOL_SCHEMA_VERSION,
    GrpoCompatibilityConfig,
    GrpoPoolGateConfig,
    audit_rollout_details_file,
    audit_rollout_groups,
    build_grpo_probe_metadata,
    run_grpo_dry_run,
    write_pool_manifest,
)


def test_audit_rollout_groups_accepts_quantified_pool_gates(tmp_path: Path) -> None:
    details = _mixed_details(count=8) + _all_correct_details(count=2)
    gate = GrpoPoolGateConfig(
        min_pool_size=8,
        min_mixed_group_rate=0.25,
        max_zero_std_group_rate=0.75,
        min_correct_truncation_mixed=4,
        max_all_wrong_rate=0.25,
    )

    audit = audit_rollout_groups(details, gate=gate)

    assert audit.passed is True
    assert audit.total_prompts == 10
    assert audit.mixed_groups == 8
    assert audit.correct_truncation_mixed_groups == 8
    assert audit.zero_std_group_rate == 0.2

    output = tmp_path / "pool.json"
    write_pool_manifest(
        audit,
        output,
        metadata={"split": "train", "checkpoint": "checkpoint"},
    )
    assert GRPO_POOL_SCHEMA_VERSION in output.read_text(encoding="utf-8")


def test_audit_rollout_details_file_writes_manifest(tmp_path: Path) -> None:
    details_path = tmp_path / "details.json"
    details_path.write_text(
        json.dumps(_mixed_details(count=4)) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "pool.json"

    audit = audit_rollout_details_file(
        details_path,
        output_path,
        gate=GrpoPoolGateConfig(
            min_pool_size=4,
            min_mixed_group_rate=0.25,
            max_zero_std_group_rate=0.75,
            min_correct_truncation_mixed=4,
            max_all_wrong_rate=0.25,
        ),
        metadata={"split": "train"},
    )

    assert audit.passed is True
    assert output_path.exists()
    assert "details_path" in output_path.read_text(encoding="utf-8")


def test_audit_rollout_groups_rejects_weak_pool_signal() -> None:
    details = _all_wrong_details(count=6)
    audit = audit_rollout_groups(
        details,
        gate=GrpoPoolGateConfig(
            min_pool_size=8,
            min_mixed_group_rate=0.25,
            max_zero_std_group_rate=0.75,
            min_correct_truncation_mixed=4,
            max_all_wrong_rate=0.25,
        ),
    )

    assert audit.passed is False
    assert "pool_size" in audit.failures
    assert "mixed_group_rate" in audit.failures
    assert "all_wrong_rate" in audit.failures


def test_build_grpo_probe_metadata_records_fail_fast_contract() -> None:
    metadata = build_grpo_probe_metadata(
        GrpoCompatibilityConfig(
            beta=0.001,
            scale_rewards="none",
            mask_truncated_completions=False,
            remove_unused_columns=False,
        ),
        trl_version="1.6.0",
        supported_fields={
            "beta",
            "loss_type",
            "mask_truncated_completions",
            "remove_unused_columns",
            "scale_rewards",
        },
    )

    assert metadata["beta"] == 0.001
    assert metadata["scale_rewards"] == "none"
    assert metadata["mask_truncated_completions"] is False
    assert metadata["remove_unused_columns"] is False
    assert metadata["unsupported_fields"] == []


def test_build_grpo_config_kwargs_skips_unsupported_optional_fields() -> None:
    config, skipped = _build_grpo_config_kwargs(
        supported_fields={
            "bf16",
            "beta",
            "gradient_accumulation_steps",
            "learning_rate",
            "logging_steps",
            "loss_type",
            "mask_truncated_completions",
            "max_completion_length",
            "max_steps",
            "num_generations",
            "output_dir",
            "per_device_train_batch_size",
            "remove_unused_columns",
            "report_to",
            "save_steps",
            "scale_rewards",
            "top_p",
        },
        output_dir="out",
        beta=0.0,
        scale_rewards="none",
        mask_truncated_completions=False,
        remove_unused_columns=False,
        max_completion_length=1024,
        num_generations=4,
        temperature=0.8,
        top_p=0.95,
        learning_rate=5e-6,
        max_steps=25,
        save_steps=25,
        logging_steps=1,
    )

    assert "max_prompt_length" not in config
    assert config["top_p"] == 0.95
    assert skipped == {"temperature": 0.8}


def test_grpo_dry_run_writes_prompt_and_reward_artifacts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_split_manifest(), manifest_path)

    result = run_grpo_dry_run(
        manifest_path=manifest_path,
        split="train",
        output_dir=tmp_path / "grpo",
        limit=3,
    )

    assert result["loads_model_weights"] is False
    assert result["prompt_records"] == 3
    assert result["sample_rewards"] == [1.0, -0.2]
    assert Path(result["prompts_path"]).exists()
    assert (tmp_path / "grpo" / "dry-run-metadata.json").exists()


def _mixed_details(count: int) -> list[dict[str, object]]:
    records = []
    for index in range(count):
        records.extend(
            [
                {
                    "id": f"mixed-{index}",
                    "reward": 1,
                    "reason": "ok",
                    "numbers": [8, 2, 7, 3],
                    "target": 24,
                },
                {
                    "id": f"mixed-{index}",
                    "reward": 0,
                    "reason": (
                        "answer_contract:expected exactly one "
                        "<answer>...</answer> block"
                    ),
                    "numbers": [8, 2, 7, 3],
                    "target": 24,
                },
            ]
        )
    return records


def _all_correct_details(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"correct-{index}",
            "reward": 1,
            "reason": "ok",
            "numbers": [8, 2, 7, 3],
            "target": 24,
        }
        for index in range(count)
    ]


def _all_wrong_details(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"wrong-{index}",
            "reward": 0,
            "reason": "wrong_value",
            "numbers": [8, 2, 7, 3],
            "target": 24,
        }
        for index in range(count)
    ]
